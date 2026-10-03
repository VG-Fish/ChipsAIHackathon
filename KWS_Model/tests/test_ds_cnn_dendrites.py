"""Native additive dendrite branches on DS-CNN blocks and classifier."""
from pathlib import Path

import pytest
import torch
import yaml

from kws.models.ds_cnn import DSCNN, build_ds_cnn
from kws.models.layers import DSConvBlock
from kws.models.sparknet import DendriticPointwise
from kws.optimize.prune import prune_ds_cnn

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"
MFCC32 = (32, 101)


def _params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _branch_cost(channels: int, dendrites: int, fan_in: int) -> int:
    return channels * dendrites * (fan_in + 2)


def _base_cfg(width: int = 24, blocks: int = 2) -> dict:
    return {"name": "t", "initial_channels": width, "initial_kernel": 5,
            "initial_stride": 2, "block_channels": [width] * blocks, "dropout": 0.2}


def test_branch_cost_formula_matches_class():
    for c, d, f, cin in ((20, 1, 4, 20), (24, 2, 8, 24), (12, 4, 8, 24)):
        assert _params(DendriticPointwise(c, d, f, in_channels=cin)) == _branch_cost(c, d, f)


def test_default_xs_unchanged():
    cfg = yaml.safe_load((CONFIGS / "ds_cnn_xs.yaml").read_text())
    model = build_ds_cnn(cfg, (40, 98), 12)
    assert _params(model) == 4096
    assert model.fc_dendrite_branch is None
    assert all(block.dendrite_branch is None for block in model.blocks)
    assert not any("dendrite" in key for key in model.state_dict())


def test_plain_block_state_dict_keys_unchanged():
    keys = set(DSConvBlock(8, 16).state_dict())
    assert keys == {
        "depthwise.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var",
        "bn1.num_batches_tracked", "pointwise.weight", "bn2.weight", "bn2.bias",
        "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
    }


def test_dendritic_block_forward_adds_branch():
    torch.manual_seed(0)
    block = DSConvBlock(8, 12, dendrites=(2, 4)).eval()
    x = torch.randn(2, 8, 5, 7)
    h = block.relu(block.bn1(block.depthwise(x)))
    assert block.dendrite_branch is not None
    expected = block.relu(block.bn2(block.pointwise(h) + block.dendrite_branch(h)))
    torch.testing.assert_close(block(x), expected)


@pytest.mark.parametrize("width,blocks,d,f", [(20, 2, 1, 4), (24, 2, 2, 8), (20, 3, 2, 4)])
def test_block_dendrite_param_counts(width, blocks, d, f):
    base = _params(build_ds_cnn(_base_cfg(width, blocks), MFCC32, 12))
    cfg = {**_base_cfg(width, blocks), "dendrites": d, "dendrite_fan_in": f}
    model = build_ds_cnn(cfg, MFCC32, 12)
    assert _params(model) == base + blocks * _branch_cost(width, d, f)
    assert model(torch.randn(3, 1, *MFCC32)).shape == (3, 12)


def test_dendrite_blocks_subset():
    cfg = {**_base_cfg(24, 3), "dendrites": 2, "dendrite_fan_in": 4, "dendrite_blocks": [1]}
    model = build_ds_cnn(cfg, MFCC32, 12)
    assert [b.dendrite_branch is not None for b in model.blocks] == [False, True, False]
    base = _params(build_ds_cnn(_base_cfg(24, 3), MFCC32, 12))
    assert _params(model) == base + _branch_cost(24, 2, 4)
    with pytest.raises(ValueError):
        build_ds_cnn({**cfg, "dendrite_blocks": [3]}, MFCC32, 12)


def test_missing_fan_in_rejected():
    with pytest.raises(ValueError):
        build_ds_cnn({**_base_cfg(), "dendrites": 2}, MFCC32, 12)


def test_fc_dendrites():
    base = _params(build_ds_cnn(_base_cfg(24), MFCC32, 12))
    cfg = {**_base_cfg(24), "fc_dendrites": 4, "fc_dendrite_fan_in": 8}
    model = build_ds_cnn(cfg, MFCC32, 12).eval()
    assert _params(model) == base + _branch_cost(12, 4, 8)
    x = torch.randn(3, 1, *MFCC32)
    assert model(x).shape == (3, 12)
    z = model.forward_features(x)
    assert model.fc_dendrite_branch is not None
    expected = model.fc(z) + model.fc_dendrite_branch(z[:, :, None, None]).flatten(1)
    torch.testing.assert_close(model.classify_features(z), expected)


def test_dendritic_backward():
    cfg = {**_base_cfg(20), "dendrites": 2, "dendrite_fan_in": 4,
           "fc_dendrites": 2, "fc_dendrite_fan_in": 8}
    model = build_ds_cnn(cfg, (40, 98), 12)
    model(torch.randn(4, 1, 40, 98)).sum().backward()
    for block in model.blocks:
        assert block.dendrite_branch is not None
        assert block.dendrite_branch.dendrite.weight.grad is not None


def test_prune_refuses_dendritic_model():
    model = build_ds_cnn({**_base_cfg(20), "dendrites": 1, "dendrite_fan_in": 4}, MFCC32, 12)
    with pytest.raises(ValueError, match="dendrite"):
        prune_ds_cnn(model, 0.5)
    plain = build_ds_cnn(_base_cfg(20), MFCC32, 12)
    assert isinstance(prune_ds_cnn(plain, 0.5), DSCNN)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("ds_cnn_w*_mfcc.yaml")), ids=lambda p: p.name)
def test_mfcc_config_param_comment(path):
    first = path.read_text().splitlines()[0]
    claimed = int(first.split("Params:")[1].split("for")[0].replace(",", ""))
    model = build_ds_cnn(yaml.safe_load(path.read_text()), MFCC32, 12)
    assert _params(model) == claimed
