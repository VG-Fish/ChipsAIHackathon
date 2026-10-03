import math
from pathlib import Path

import pytest
import torch
import yaml

from kws.models.dnn import DNN
from kws.models.ei_conv1d import EIConv1d, local_window_index
from kws.models.registry import build_model, build_model_from_checkpoint

SHAPE = (32, 101)
CLASSES = 12
FRAMES = 26  # ceil(ceil(101 / 2) / 2)
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ei_base_params(c1: int, c2: int, n_mfcc: int = 32) -> int:
    return (n_mfcc * c1 * 3 + c1) + (c1 * c2 * 3 + c2) + (c2 * FRAMES * CLASSES + CLASSES)


def ei(**cfg) -> EIConv1d:
    return build_model({"family": "ei_conv1d", **cfg}, SHAPE, CLASSES)


@pytest.mark.parametrize("c1,c2", [(8, 16), (4, 8), (6, 12), (16, 32)])
def test_ei_base_param_count(c1, c2):
    assert n_params(ei(c1=c1, c2=c2)) == ei_base_params(c1, c2)


def test_ei_default_is_edge_impulse_6180():
    assert n_params(ei()) == 6180


@pytest.mark.parametrize("c1,c2,d,f", [(4, 8, 1, 32), (4, 8, 4, 32), (8, 16, 2, 64)])
def test_ei_time_local_head_param_count(c1, c2, d, f):
    model = ei(c1=c1, c2=c2, head_dendrites=d, head_dendrite_fan_in=f)
    assert n_params(model) == ei_base_params(c1, c2) + CLASSES * d * (f + 2)


def test_ei_full_head_param_count():
    model = ei(c1=8, c2=16, head_dendrites=1, head_dendrite_layout="full")
    assert model.head_dendrites.fan_in == 16 * FRAMES
    assert n_params(model) == ei_base_params(8, 16) + CLASSES * (16 * FRAMES + 2)


@pytest.mark.parametrize("cfg", [
    {}, {"c1": 4, "c2": 8, "head_dendrites": 2, "head_dendrite_fan_in": 32},
    {"head_dendrites": 1, "head_dendrite_layout": "full"}, {"batchnorm": True},
])
def test_ei_forward_shape(cfg):
    model = ei(**cfg).eval()
    assert model(torch.randn(3, 1, *SHAPE)).shape == (3, CLASSES)
    assert model(torch.randn(3, *SHAPE)).shape == (3, CLASSES)
    assert model.frames == FRAMES and model.input_shape == SHAPE


def test_ei_time_local_windows_are_consecutive_frames_over_all_channels():
    c2, d, f = 16, 4, 64
    model = ei(c1=8, c2=c2, head_dendrites=d, head_dendrite_fan_in=f)
    index = model.head_dendrites.index.view(CLASSES, d, f)
    starts = []
    for n in range(CLASSES):
        for j in range(d):
            window = index[n, j].tolist()
            frames, channels = zip(*(divmod(i, c2) for i in window))
            assert window == list(range(window[0], window[0] + f))
            assert window[0] % c2 == 0  # frame aligned
            assert sorted(set(frames)) == list(range(frames[0], frames[0] + f // c2))
            assert max(frames) < FRAMES  # no wrap past the last pooled frame
            for t in set(frames):  # every channel present in every covered frame
                assert sorted(c for fr, c in zip(frames, channels) if fr == t) == list(range(c2))
        starts.append(index[n, 0, 0].item())
    assert starts[0] == 0 and starts[-1] == (FRAMES - d * f // c2) * c2
    assert starts == sorted(starts)


def test_ei_time_local_flatten_is_time_major():
    model = ei(c1=4, c2=8, head_dendrites=1, head_dendrite_fan_in=32)
    x = torch.arange(2 * 8 * FRAMES, dtype=torch.float).view(2, 8, FRAMES)
    z = model.flatten(x)
    t, c = 5, 3
    assert z[1, t * 8 + c] == x[1, c, t]


def test_ei_time_local_dendrites_ignore_frames_outside_window():
    """Class 0's dendrite reads only the first 4 pooled frames."""
    model = ei(c1=4, c2=8, head_dendrites=1, head_dendrite_fan_in=32).eval()
    z = torch.randn(1, 8 * FRAMES, requires_grad=True)
    model.head_dendrites.dendrite.bias.data.fill_(1.0)  # keep the ReLU open
    model.head_dendrites(z[:, :, None, None]).flatten(1)[0, 0].backward()
    touched = z.grad[0].nonzero().flatten()
    assert touched.max().item() < 32


def test_ei_rejects_bad_fan_in():
    with pytest.raises(ValueError):
        ei(c2=16, head_dendrites=1, head_dendrite_fan_in=40)
    with pytest.raises(ValueError):
        ei(head_dendrites=1, head_dendrite_layout="full", head_dendrite_fan_in=10)


def test_local_window_index_falls_back_to_cyclic_when_span_too_long():
    index = local_window_index(12, 2, 16, 16)
    assert len(index) == 12 * 32 and max(index) < 16


def dnn_params(hidden, features=32 * FRAMES):
    total, width = 0, features
    for h in hidden + [CLASSES]:
        total += width * h + h
        width = h
    return total


@pytest.mark.parametrize("hidden", [[16], [24], [32, 16]])
def test_dnn_param_count_and_shape(hidden):
    model = build_model({"family": "dnn", "hidden": hidden}, SHAPE, CLASSES)
    assert isinstance(model, DNN)
    assert n_params(model) == dnn_params(hidden)
    assert model(torch.randn(2, 1, *SHAPE)).shape == (2, CLASSES)


@pytest.mark.parametrize("d", [2, 4])
def test_dnn_dendrite_param_count(d):
    model = build_model({"family": "dnn", "hidden": [16], "dendrites": d, "dendrite_fan_in": 32},
                        SHAPE, CLASSES)
    expected = dnn_params([16]) + 16 * d * (32 + 2) + CLASSES * d * (16 + 2)
    assert n_params(model) == expected
    assert model(torch.randn(2, 1, *SHAPE)).shape == (2, CLASSES)
    index = model.hidden[0].dendrites.index.view(16, d, 32)
    assert index.max().item() < 32 * FRAMES
    assert all(index[n, 0, 0].item() % 32 == 0 for n in range(16))  # whole frames


def test_dnn_time_pool_is_time_major_average():
    model = build_model({"family": "dnn", "hidden": [4], "input_norm": "none"}, SHAPE, CLASSES)
    captured = {}
    model.hidden[0].register_forward_hook(lambda m, i, o: captured.update(x=i[0]))
    x = torch.randn(1, 1, *SHAPE)
    model(x)
    assert captured["x"].shape == (1, 32 * FRAMES)
    assert torch.allclose(captured["x"][0, 1 * 32 + 7], x[0, 0, 7, 4:8].mean())
    assert torch.allclose(captured["x"][0, 25 * 32 + 3], x[0, 0, 3, 100])  # ceil_mode tail


@pytest.mark.parametrize("cfg", [
    {"family": "ei_conv1d", "name": "x", "c1": 4, "c2": 8, "head_dendrites": 2,
     "head_dendrite_fan_in": 32},
    {"family": "dnn", "name": "y", "hidden": [16], "dendrites": 2, "dendrite_fan_in": 32},
])
def test_checkpoint_rebuild_roundtrip(cfg):
    model = build_model(cfg, SHAPE, CLASSES).eval()
    checkpoint = {"model_state_dict": model.state_dict(), "model_family": cfg["family"],
                  "model_cfg": cfg, "input_shape": model.input_shape,
                  "num_classes": model.fc.out_features}
    rebuilt = build_model_from_checkpoint(checkpoint).eval()
    rebuilt.load_state_dict(checkpoint["model_state_dict"])
    x = torch.randn(2, 1, *SHAPE)
    assert torch.equal(model(x), rebuilt(x))


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("ei_c*.yaml")) + sorted(CONFIGS.glob("dnn_h*.yaml")),
                         ids=lambda p: p.stem)
def test_config_header_param_count(path):
    text = path.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["name"] == path.stem
    stated = int(text.split(" params", 1)[0].lstrip("# "))
    assert n_params(build_model(cfg, SHAPE, CLASSES)) == stated
    assert math.isfinite(stated)


@pytest.mark.parametrize("mode", ["none", "cmvn", "batchnorm"])
def test_input_norm_is_parameter_free(mode):
    model = ei(input_norm=mode)
    assert n_params(model) == 6180
    assert model(torch.randn(4, 1, *SHAPE) * 20 - 30).shape == (4, CLASSES)


def test_cmvn_normalizes_each_coefficient_over_time():
    from kws.models.ei_conv1d import InputNorm
    x = torch.randn(2, 32, 101) * 24 - 34
    y = InputNorm("cmvn", 32)(x)
    assert torch.allclose(y.mean(-1), torch.zeros(2, 32), atol=1e-4)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(2, 32), atol=1e-3)
