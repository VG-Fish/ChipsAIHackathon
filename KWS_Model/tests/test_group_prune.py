import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.models.ds_cnn import DSCNN
from kws.models.sparknet import SparkNet
from kws.optimize.group_prune import (
    count_trainable_params,
    dependency_groups,
    group_prune,
    planned_params,
    resolve_widths,
    verify_pruned,
)
from kws.optimize.prune import prune_ds_cnn, prune_sparknet

SPARK_IN = (32, 101)
DS_IN = (40, 98)


def _randomize_bn(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every BN non-trivial affine params and running stats, then eval()."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.2)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
    return model.eval()


def _sparknet(channels: int = 16, **kw) -> SparkNet:
    torch.manual_seed(0)
    return _randomize_bn(SparkNet(n_feat=SPARK_IN[0], num_classes=12, channels=channels,
                                  gate_channels=32, input_shape=SPARK_IN, **kw))


def _ds_cnn(block_channels=(64, 64, 64, 64), initial_channels: int = 48, **kw) -> DSCNN:
    torch.manual_seed(0)
    return _randomize_bn(DSCNN(input_shape=DS_IN, num_classes=6,
                               initial_channels=initial_channels, initial_kernel=10,
                               initial_stride=2, block_channels=list(block_channels), **kw))


def _batches(input_shape, classes, n=3, bs=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    data = [(torch.randn(bs, 1, *input_shape, generator=g),
             torch.randint(0, classes, (bs,), generator=g)) for _ in range(n)]
    return lambda: iter(data)


# ----------------------------------------------------------------- structure


def test_sparknet_groups_couple_residual_and_consumers():
    groups = {g.name: g.modules() for g in dependency_groups(_sparknet())}
    assert groups["blocks.0"] == {"blocks.0.pointwise", "blocks.0.bn", "blocks.1.depthwise",
                                  "blocks.1.pointwise", "blocks.1.res_conv"}
    assert groups["blocks.2"] == {"blocks.2.pointwise", "blocks.2.bn", "blocks.2.res_conv",
                                  "blocks.2.res_bn", "blocks.3.depthwise",
                                  "blocks.3.pointwise", "blocks.3.res_conv"}
    assert groups["blocks.3"] == {"blocks.3.pointwise", "blocks.3.bn", "blocks.3.res_conv",
                                  "blocks.3.res_bn", "gate_conv"}


def test_ds_cnn_groups_include_stem_and_fc():
    groups = {g.name: g.modules() for g in dependency_groups(_ds_cnn())}
    assert groups["stem"] == {"stem.0", "stem.1", "blocks.0.depthwise", "blocks.0.bn1",
                              "blocks.0.pointwise"}
    assert groups["blocks.3"] == {"blocks.3.pointwise", "blocks.3.bn2", "fc"}
    assert "stem" not in {g.name for g in dependency_groups(_ds_cnn(), prune_stem=False)}


# ---------------------------------------------------------- identity / shape


@pytest.mark.parametrize("criterion", ["l2_group", "l1_producer"])
def test_sparknet_identity_prune_is_exact(criterion):
    model = _sparknet()
    result = group_prune(model, width=16, criterion=criterion)
    x = torch.randn(3, 1, *SPARK_IN)
    with torch.no_grad():
        assert torch.equal(result.model(x), model(x))
    assert result.pruned_params == count_trainable_params(model)
    assert result.achieved_rate == 0


@pytest.mark.parametrize("prune_stem", [True, False])
def test_ds_cnn_identity_prune_is_exact(prune_stem):
    model = _ds_cnn()
    result = group_prune(model, width=64, prune_stem=prune_stem)
    x = torch.randn(3, 1, *DS_IN)
    with torch.no_grad():
        assert torch.equal(result.model(x), model(x))


def test_input_model_is_not_modified():
    model = _sparknet()
    before = copy.deepcopy(model.state_dict())
    group_prune(model, rate=0.5)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


@pytest.mark.parametrize("criterion", ["l2_group", "l1_producer", "taylor"])
def test_sparknet_rate_prune_forward_and_counts(criterion):
    model = _sparknet()
    result = group_prune(model, rate=0.5, criterion=criterion,
                         batches=_batches(SPARK_IN, 12), loss_fn=F.cross_entropy)
    width = result.widths["blocks.0"]
    assert set(result.widths.values()) == {width} and width < 16
    x = torch.randn(3, 1, *SPARK_IN)
    with torch.no_grad():
        assert result.model(x).shape == (3, 12)
    fresh = SparkNet(n_feat=SPARK_IN[0], num_classes=12, channels=width, gate_channels=32,
                     input_shape=SPARK_IN)
    assert result.pruned_params == count_trainable_params(fresh) == verify_pruned(result.model)
    assert abs(result.achieved_rate - 0.5) < 0.05


def test_ds_cnn_rate_prune_forward_and_counts():
    model = _ds_cnn()
    result = group_prune(model, rate=0.6)
    x = torch.randn(3, 1, *DS_IN)
    with torch.no_grad():
        assert result.model(x).shape == (3, 6)
    fresh = DSCNN(input_shape=DS_IN, num_classes=6, initial_channels=result.widths["stem"],
                  initial_kernel=10, initial_stride=2,
                  block_channels=[result.widths[f"blocks.{i}"] for i in range(4)])
    assert result.pruned_params == count_trainable_params(fresh)
    assert abs(result.achieved_rate - 0.6) < 0.05


def test_planned_params_match_constructed_models():
    spark = _sparknet()
    for c in range(1, 17):
        fresh = SparkNet(n_feat=SPARK_IN[0], num_classes=12, channels=c, gate_channels=32,
                         input_shape=SPARK_IN)
        assert planned_params(spark, {f"blocks.{i}": c for i in range(4)}) == \
            count_trainable_params(fresh)
    ds = _ds_cnn()
    widths = {"stem": 7, "blocks.0": 5, "blocks.1": 9, "blocks.2": 3, "blocks.3": 11}
    fresh = DSCNN(input_shape=DS_IN, num_classes=6, initial_channels=7, initial_kernel=10,
                  initial_stride=2, block_channels=[5, 9, 3, 11])
    assert planned_params(ds, widths) == count_trainable_params(fresh)


def test_rate_solver_picks_closest_uniform_width():
    model = _sparknet()
    total = count_trainable_params(model)
    for rate in (0.1, 0.3, 0.5, 0.7, 0.9):
        widths = resolve_widths(model, rate=rate)
        target = (1 - rate) * total
        errors = {c: abs(planned_params(model, {f"blocks.{i}": c for i in range(4)}) - target)
                  for c in range(1, 17)}
        assert abs(planned_params(model, widths) - target) == min(errors.values())


# -------------------------------------------------------- legacy equivalence


def test_l1_producer_reproduces_prune_sparknet():
    model = _sparknet()
    legacy = prune_sparknet(model, 9)
    result = group_prune(model, width=9, criterion="l1_producer")
    x = torch.randn(3, 1, *SPARK_IN)
    with torch.no_grad():
        assert torch.equal(result.model(x), legacy(x))


def test_l1_producer_without_stem_reproduces_prune_ds_cnn():
    model = _ds_cnn()
    legacy = prune_ds_cnn(model, keep_ratio=0.5).eval()
    result = group_prune(model, width=32, criterion="l1_producer", prune_stem=False)
    x = torch.randn(3, 1, *DS_IN)
    with torch.no_grad():
        assert torch.equal(result.model(x), legacy(x))


# ------------------------------------------------------------ importances


def _toy_sparknet_with_dead_but_heavy_channel(k: int = 3) -> SparkNet:
    """Block-0 channel ``k`` has the largest producer row but a ~0 BN gamma.

    (Exactly 0 would be rejected by ``verify_pruned`` wherever it survives.)
    """
    model = SparkNet(n_feat=SPARK_IN[0], num_classes=12, channels=8, gate_channels=32,
                     input_shape=SPARK_IN).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.fill_(0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1.0)
                m.bias.zero_()
        model.blocks[0].pointwise.weight[k] = 0.15
        model.blocks[0].bn.weight[k] = 1e-3
    return model


@pytest.mark.parametrize("tp_compat", [True, False])
def test_l2_group_prunes_zero_gamma_channel_that_l1_producer_keeps(tp_compat):
    model = _toy_sparknet_with_dead_but_heavy_channel(k=3)
    l2 = group_prune(model, width=7, criterion="l2_group", tp_compat=tp_compat)
    l1 = group_prune(model, width=7, criterion="l1_producer")
    assert 3 not in l2.keep["blocks.0"]
    assert 3 in l1.keep["blocks.0"]
    assert l2.scores["blocks.0"][3] == min(l2.scores["blocks.0"])


@pytest.mark.parametrize("taylor_mode", ["molchanov", "tp"])
def test_taylor_scores_dead_channel_zero_and_prunes_it(taylor_mode):
    model = _sparknet(channels=8)
    k = 5
    with torch.no_grad():
        model.blocks[0].bn.weight[k] = 0.0
        model.blocks[0].bn.bias[k] = 0.0
    result = group_prune(model, width=7, criterion="taylor", taylor_mode=taylor_mode,
                         batches=_batches(SPARK_IN, 12), loss_fn=F.cross_entropy,
                         taylor_batches=2)
    scores = result.scores["blocks.0"]
    assert scores[k] == 0.0
    assert all(s > 0 for i, s in enumerate(scores) if i != k)
    assert k not in result.keep["blocks.0"]
    for grp in ("blocks.1", "blocks.2", "blocks.3"):
        # Channels whose ReLU never fires on this data legitimately score 0.
        assert sum(s > 0 for s in result.scores[grp]) >= 4, "taylor scores look degenerate"


def test_taylor_requires_calibration_data():
    with pytest.raises(ValueError, match="batches and loss_fn"):
        group_prune(_sparknet(), rate=0.5, criterion="taylor")


def test_taylor_leaves_grads_and_mode_untouched():
    model = _sparknet()
    model.train()
    group_prune(model, rate=0.5, criterion="taylor", batches=_batches(SPARK_IN, 12),
                loss_fn=F.cross_entropy)
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_ds_cnn_taylor_runs():
    model = _ds_cnn()
    result = group_prune(model, rate=0.5, criterion="taylor",
                         batches=_batches(DS_IN, 6), loss_fn=F.cross_entropy)
    assert verify_pruned(result.model) == result.pruned_params


# ------------------------------------------------------------ verification


def test_verify_rejects_zeroed_conv_row_and_masked_bn():
    model = _sparknet()
    assert verify_pruned(model) == count_trainable_params(model)
    zeroed = copy.deepcopy(model)
    with torch.no_grad():
        zeroed.blocks[1].pointwise.weight[2].zero_()
    with pytest.raises(ValueError, match="all-zero output channels"):
        verify_pruned(zeroed)
    masked = copy.deepcopy(model)
    with torch.no_grad():
        masked.blocks[2].bn.weight[4] = 0.0
    with pytest.raises(ValueError, match="zero gamma"):
        verify_pruned(masked)


def test_verify_rejects_torch_prune_masks():
    model = _ds_cnn()
    import torch.nn.utils.prune as torch_prune

    torch_prune.l1_unstructured(model.fc, "weight", amount=0.1)
    with pytest.raises(ValueError, match="mask still attached"):
        verify_pruned(model)


# ------------------------------------------------------------ refusals


@pytest.mark.parametrize("kw", [
    dict(dendrites=2, dendrite_fan_in=4),
    dict(dendrites=2, dendrite_fan_in=4, dendrite_mode="add"),
])
def test_sparknet_dendrites_are_refused(kw):
    with pytest.raises(ValueError, match="dendrites"):
        group_prune(_sparknet(**kw), rate=0.5)


@pytest.mark.parametrize("kw", [dict(dendrites=(2, 4)), dict(fc_dendrites=(2, 4))])
def test_ds_cnn_dendrites_are_refused(kw):
    with pytest.raises(ValueError, match="dendrite"):
        group_prune(_ds_cnn(**kw), rate=0.5)


def test_width_and_rate_are_exclusive_and_validated():
    model = _sparknet()
    with pytest.raises(ValueError):
        group_prune(model)
    with pytest.raises(ValueError):
        group_prune(model, rate=0.5, width=8)
    with pytest.raises(ValueError):
        group_prune(model, width=17)


# ------------------------------------------------ Torch-Pruning cross-check


def _tp_prune(model, example, ignored, ratio, importance, batches=None):
    tp = pytest.importorskip("torch_pruning")
    model.eval()
    if batches is not None:
        for x, y in batches():
            F.cross_entropy(model(x), y).backward()
    pruner = tp.pruner.MetaPruner(model, example, importance=importance, pruning_ratio=ratio,
                                  global_pruning=False, ignored_layers=ignored)
    pruner.step()
    return model.eval()


def test_torch_pruning_sparknet_l2_matches_exactly():
    tp = pytest.importorskip("torch_pruning")
    model = _sparknet()
    x = torch.randn(3, 1, *SPARK_IN)
    ref = copy.deepcopy(model)
    ref = _tp_prune(ref, x, [ref.blocks[0].depthwise, ref.gate_conv, ref.fc], 0.5,
                    tp.importance.GroupMagnitudeImportance(p=2))
    result = group_prune(model, width=8)
    assert result.pruned_params == count_trainable_params(ref)
    with torch.no_grad():
        assert torch.equal(result.model(x), ref(x))


def test_torch_pruning_sparknet_taylor_matches_exactly():
    tp = pytest.importorskip("torch_pruning")
    model = _sparknet()
    x = torch.randn(3, 1, *SPARK_IN)
    batches = _batches(SPARK_IN, 12)
    ref = copy.deepcopy(model)
    ref = _tp_prune(ref, x, [ref.blocks[0].depthwise, ref.gate_conv, ref.fc], 0.5,
                    tp.importance.GroupTaylorImportance(), batches=batches)
    result = group_prune(model, width=8, criterion="taylor", taylor_mode="tp",
                         batches=batches, loss_fn=F.cross_entropy, taylor_batches=None)
    with torch.no_grad():
        assert torch.equal(result.model(x), ref(x))


def test_torch_pruning_ds_cnn_l2_matches_exactly():
    tp = pytest.importorskip("torch_pruning")
    model = _ds_cnn()
    x = torch.randn(3, 1, *DS_IN)
    ref = copy.deepcopy(model)
    ref = _tp_prune(ref, x, [ref.fc], 0.5, tp.importance.GroupMagnitudeImportance(p=2))
    result = group_prune(model, width=32)
    assert result.pruned_params == count_trainable_params(ref)
    with torch.no_grad():
        assert torch.equal(result.model(x), ref(x))
