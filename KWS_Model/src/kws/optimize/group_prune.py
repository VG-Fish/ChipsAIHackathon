"""Dependency-group structured channel pruning for SparkNet and DS-CNN.

Follows the protocol of "Pruning Then Perforating": Torch-Pruning DepGraph
groups, ``GroupMagnitudeImportance(p=2)`` (Study B: first-order Taylor,
Molchanov et al. 2019), ``global_pruning=False`` (every group keeps the same
fraction of its channels), classifier untouched, one-shot at a target
*parameter* prune rate. Channels are physically removed into a freshly built,
narrower model, and :func:`verify_pruned` rejects any masked or zeroed
channel so the reported parameter count is that of the executed network.

Torch-Pruning itself is not a dependency: the groups of these two fixed
topologies are written out explicitly (:func:`dependency_groups`) and
``tests/test_group_prune.py`` cross-checks them against Torch-Pruning when it
is installed.

A prunable channel ``j`` of a group is every parameter slice that must be
removed together with it:

SparkNet, block ``i`` output channel ``j``
    ``blocks.i.pointwise`` out row ``j``, ``blocks.i.bn[j]``,
    ``blocks.i.res_conv`` out row ``j`` and ``blocks.i.res_bn[j]`` (blocks
    1-3; the residual sum couples the two BNs), then the consumers
    ``blocks.{i+1}.depthwise`` filter ``j``, ``blocks.{i+1}.pointwise`` and
    ``blocks.{i+1}.res_conv`` input column ``j`` -- or ``gate_conv`` input
    column ``j`` after the last block.

DS-CNN, stem / block ``i`` output channel ``j``
    ``stem.0`` out row ``j`` + ``stem.1[j]`` (stem group), or
    ``blocks.i.pointwise`` out row ``j`` + ``blocks.i.bn2[j]``, then the
    consumers ``blocks.{i+1}.depthwise`` filter ``j``, ``blocks.{i+1}.bn1[j]``,
    ``blocks.{i+1}.pointwise`` input column ``j`` -- or ``fc`` input column
    ``j`` after the last block.

The model input (MFCC bins), the SparkNet gate width and the classifier
outputs are never pruned.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Literal, cast

import torch
import torch.nn as nn

from kws.models.ds_cnn import DSCNN
from kws.models.sparknet import SparkNet, TCSBlock

Criterion = Literal["l2_group", "taylor", "l1_producer"]
CRITERIA: tuple[str, ...] = ("l2_group", "taylor", "l1_producer")

Batches = Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------
# Dependency groups
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSlice:
    """Slice ``[..., j, ...]`` along ``dim`` of ``<module>.<param>``.

    ``other_group`` names the group that owns this tensor's *other* channel
    dim (a producer's input columns, or a consumer's output rows). Groups are
    pruned one after another, so once that group has been decided the slice
    only counts its surviving channels -- scoring the network that will
    actually execute, as Torch-Pruning's sequential ``step()`` does.
    """

    module: str
    param: str
    dim: int
    producer: bool = False
    primary: bool = False  # the slice ``l1_producer`` ranks by
    other_group: str | None = None
    depthwise: bool = False


@dataclass(frozen=True)
class ChannelGroup:
    name: str
    width: int
    slices: tuple[ParamSlice, ...]

    def modules(self) -> set[str]:
        return {s.module for s in self.slices}


def _bn_slices(name: str, *, producer: bool) -> list[ParamSlice]:
    return [ParamSlice(name, "weight", 0, producer=producer),
            ParamSlice(name, "bias", 0, producer=producer)]


def _check_sparknet(model: SparkNet) -> list[TCSBlock]:
    blocks = [cast(TCSBlock, b) for b in model.blocks]
    for i, block in enumerate(blocks):
        if not isinstance(block.pointwise, nn.Conv2d) or (
            getattr(block, "dendrite_branch", None) is not None
        ):
            raise ValueError(
                f"group pruning does not support SparkNet dendrites (blocks.{i}); "
                "the restricted-receptive-field index tables cannot be sliced cleanly"
            )
    widths = {b.pointwise.out_channels for b in blocks}
    if len(widths) != 1:
        raise ValueError("SparkNet group pruning needs a uniform-width backbone")
    return blocks


def _check_ds_cnn(model: DSCNN) -> None:
    if getattr(model, "fc_dendrite_branch", None) is not None or any(
        getattr(block, "dendrite_branch", None) is not None for block in model.blocks
    ):
        raise ValueError("group pruning does not support DS-CNNs with native dendrite branches")


def sparknet_groups(model: SparkNet) -> list[ChannelGroup]:
    blocks = _check_sparknet(model)
    groups: list[ChannelGroup] = []
    for i, block in enumerate(blocks):
        prev = f"blocks.{i - 1}" if i > 0 else None
        nxt_name = f"blocks.{i + 1}"
        s: list[ParamSlice] = [
            ParamSlice(f"blocks.{i}.pointwise", "weight", 0, producer=True, primary=True,
                       other_group=prev),
            *_bn_slices(f"blocks.{i}.bn", producer=True),
        ]
        if block.res_conv is not None:
            s += [ParamSlice(f"blocks.{i}.res_conv", "weight", 0, producer=True,
                             other_group=prev),
                  *_bn_slices(f"blocks.{i}.res_bn", producer=True)]
        if i + 1 < len(blocks):
            nxt = blocks[i + 1]
            s += [ParamSlice(f"{nxt_name}.depthwise", "weight", 0, depthwise=True),
                  ParamSlice(f"{nxt_name}.pointwise", "weight", 1, other_group=nxt_name)]
            if nxt.res_conv is not None:
                s.append(ParamSlice(f"{nxt_name}.res_conv", "weight", 1,
                                    other_group=nxt_name))
        else:
            s.append(ParamSlice("gate_conv", "weight", 1))
        groups.append(ChannelGroup(f"blocks.{i}", block.pointwise.out_channels, tuple(s)))
    return groups


def ds_cnn_groups(model: DSCNN, *, prune_stem: bool = True) -> list[ChannelGroup]:
    _check_ds_cnn(model)
    blocks = list(model.blocks)
    groups: list[ChannelGroup] = []

    def consumers(i: int) -> list[ParamSlice]:
        if i < len(blocks):
            return [ParamSlice(f"blocks.{i}.depthwise", "weight", 0, depthwise=True),
                    *_bn_slices(f"blocks.{i}.bn1", producer=False),
                    ParamSlice(f"blocks.{i}.pointwise", "weight", 1,
                               other_group=f"blocks.{i}")]
        return [ParamSlice("fc", "weight", 1)]

    if prune_stem:
        stem = cast(nn.Conv2d, model.stem[0])
        groups.append(ChannelGroup("stem", stem.out_channels, (
            ParamSlice("stem.0", "weight", 0, producer=True, primary=True),
            *_bn_slices("stem.1", producer=True),
            *consumers(0),
        )))
    prev: str | None = "stem" if prune_stem else None
    for i, block in enumerate(blocks):
        groups.append(ChannelGroup(f"blocks.{i}", block.pointwise.out_channels, (
            ParamSlice(f"blocks.{i}.pointwise", "weight", 0, producer=True, primary=True,
                       other_group=prev),
            *_bn_slices(f"blocks.{i}.bn2", producer=True),
            *consumers(i + 1),
        )))
        prev = f"blocks.{i}"
    return groups


def dependency_groups(model: nn.Module, *, prune_stem: bool = True) -> list[ChannelGroup]:
    """The prunable channel groups of ``model``, in forward (pruning) order."""
    if isinstance(model, SparkNet):
        return sparknet_groups(model)
    if isinstance(model, DSCNN):
        return ds_cnn_groups(model, prune_stem=prune_stem)
    raise TypeError(f"group pruning supports SparkNet and DSCNN, not {type(model).__name__}")


# --------------------------------------------------------------------------
# Importance
# --------------------------------------------------------------------------


def _slice_tensor(t: torch.Tensor, s: ParamSlice,
                  keep: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``t`` with the channel dim first, other channel dim restricted, flattened to (C, -1)."""
    if s.other_group is not None and s.other_group in keep and t.dim() > 1:
        t = t.index_select(1 - s.dim, keep[s.other_group].to(t.device))
    return t.movedim(s.dim, 0).reshape(t.shape[s.dim], -1)


def _param(model: nn.Module, s: ParamSlice) -> torch.Tensor | None:
    return getattr(model.get_submodule(s.module), s.param, None)


def _is_bn(model: nn.Module, s: ParamSlice) -> bool:
    return isinstance(model.get_submodule(s.module), nn.modules.batchnorm._BatchNorm)


def _group_importance(
    model: nn.Module,
    group: ChannelGroup,
    criterion: str,
    keep: Mapping[str, torch.Tensor],
    *,
    p: float,
    include_bn_bias: bool,
    tp_compat: bool,
    taylor_mode: str,
    taylor_grads: Sequence[Mapping[str, torch.Tensor]] | None,
) -> torch.Tensor:
    if criterion == "l1_producer":
        (s,) = [s for s in group.slices if s.primary]
        w = cast(torch.Tensor, _param(model, s)).detach()
        return _slice_tensor(w, s, keep).abs().sum(1)
    if criterion == "taylor" and not taylor_grads:
        raise ValueError("taylor importance needs calibration gradients")

    local: list[torch.Tensor] = []
    for s in group.slices:
        if s.param == "bias" and not include_bn_bias:
            continue
        # Torch-Pruning 1.6 never scores depthwise filters: their handler is
        # DepthwiseConvPruner, which fails the importance's handler check.
        if s.depthwise and tp_compat:
            continue
        w = _param(model, s)
        if w is None:
            continue
        w = w.detach()
        if criterion == "l2_group":
            local.append(_slice_tensor(w, s, keep).abs().pow(p).sum(1))
        elif criterion == "taylor":
            key = f"{s.module}.{s.param}"
            grads = cast(Sequence[Mapping[str, torch.Tensor]], taylor_grads)
            if taylor_mode == "tp":
                # tp.importance.GroupTaylorImportance(): .grad accumulated over
                # the batches, elementwise |w * g| summed, and BN skipped (its
                # BN branch tests prune_groupnorm_out_channels).
                if _is_bn(model, s):
                    continue
                g_sum = torch.stack([g[key] for g in grads]).sum(0)
                local.append(_slice_tensor(w * g_sum, s, keep).abs().sum(1))
            elif taylor_mode == "molchanov":
                # |sum over the slice of w * dL/dw| per batch, averaged over batches.
                per_batch = [_slice_tensor(w * g[key], s, keep).sum(1).abs() for g in grads]
                local.append(torch.stack(per_batch).mean(0))
            else:
                raise ValueError(f"unknown taylor_mode {taylor_mode!r}")
        else:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
    # Torch-Pruning group_reduction="mean": average the per-slice importances.
    # (Its normalizer="mean" rescales a whole group and cannot change the ranking.)
    return torch.stack(local).mean(0)


def taylor_gradients(
    model: nn.Module,
    batches: Batches,
    loss_fn: LossFn,
    *,
    num_batches: int | None = 4,
    train_mode: bool = False,
    param_names: Iterable[str] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Per-batch gradients of ``loss_fn(model(x), y)`` for first-order Taylor scores.

    Uses ``torch.autograd.grad`` so ``.grad`` fields are left untouched. The
    model runs in eval mode by default (BN running statistics, no SparkNet gate
    noise) so the scores describe the network that will be deployed.
    """
    params = dict(model.named_parameters())
    names = list(param_names) if param_names is not None else list(params)
    tensors = [params[n] for n in names]
    device = tensors[0].device
    was_training = model.training
    model.train(train_mode)
    out: list[dict[str, torch.Tensor]] = []
    try:
        for x, y in islice(batches(), num_batches):
            loss = loss_fn(model(x.to(device)), y.to(device))
            grads = torch.autograd.grad(loss, tensors, allow_unused=True)
            out.append({
                n: (g.detach() if g is not None else torch.zeros_like(t))
                for n, t, g in zip(names, tensors, grads)
            })
            aux = getattr(model, "auxiliary_losses", None)
            if callable(aux):
                aux()  # drop SparkNet's pending training-mode gate graph
    finally:
        model.train(was_training)
    if not out:
        raise ValueError("taylor importance received no calibration batches")
    return out


# --------------------------------------------------------------------------
# Widths and parameter counts
# --------------------------------------------------------------------------


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sparknet_params(model: SparkNet, c: int) -> int:
    blocks = [cast(TCSBlock, b) for b in model.blocks]
    total, cin = 0, blocks[0].depthwise.in_channels
    for block in blocks:
        total += cin * block.depthwise.kernel_size[-1] + cin * c + 2 * c
        if block.res_conv is not None:
            total += cin * c + 2 * c
        cin = c
    g = model.gate_conv.out_channels
    total += c * g + (g if model.gate_conv.bias is not None else 0) + 2 * g
    return total + model.fc.in_features * model.fc.out_features + model.fc.out_features


def _ds_cnn_params(model: DSCNN, widths: Mapping[str, int]) -> int:
    stem = cast(nn.Conv2d, model.stem[0])
    s = widths.get("stem", stem.out_channels)
    total = stem.in_channels * s * stem.kernel_size[0] * stem.kernel_size[1] + 2 * s
    cin = s
    for i, block in enumerate(model.blocks):
        c = widths[f"blocks.{i}"]
        kh, kw = block.depthwise.kernel_size
        total += cin * kh * kw + 2 * cin + cin * c + 2 * c
        cin = c
    return total + cin * model.fc.out_features + model.fc.out_features


def planned_params(model: nn.Module, widths: Mapping[str, int]) -> int:
    """Trainable parameters of ``model`` rebuilt at ``widths`` (analytic)."""
    if isinstance(model, SparkNet):
        (c,) = set(widths.values())
        return _sparknet_params(model, c)
    if isinstance(model, DSCNN):
        return _ds_cnn_params(model, widths)
    raise TypeError(type(model).__name__)


def _uniform_widths(groups: Sequence[ChannelGroup], keep_ratio: float) -> dict[str, int]:
    return {g.name: max(1, min(g.width, int(round(g.width * keep_ratio)))) for g in groups}


def resolve_widths(
    model: nn.Module,
    *,
    rate: float | None = None,
    width: int | Mapping[str, int] | None = None,
    prune_stem: bool = True,
) -> dict[str, int]:
    """Per-group channel counts for a parameter prune ``rate`` or an explicit ``width``.

    ``rate``: the uniform (``global_pruning=False``) width whose trainable
    parameter count is closest to ``(1 - rate) * original``; ties go to the
    wider model. ``width``: an int is the backbone width (SparkNet), or for
    DS-CNN the width of the widest group with every other group scaled by the
    same keep ratio; a mapping gives each group explicitly.
    """
    groups = dependency_groups(model, prune_stem=prune_stem)
    if (rate is None) == (width is None):
        raise ValueError("give exactly one of rate or width")
    ref = max(g.width for g in groups)
    if width is not None:
        if isinstance(width, Mapping):
            widths = {g.name: int(width[g.name]) for g in groups}
        else:
            if isinstance(width, bool) or not isinstance(width, int):
                raise TypeError("width must be an int or a mapping of group widths")
            widths = (
                {g.name: width for g in groups} if isinstance(model, SparkNet)
                else _uniform_widths(groups, width / ref)
            )
        for g in groups:
            if not 0 < widths[g.name] <= g.width:
                raise ValueError(f"width {widths[g.name]} for {g.name} not in 1..{g.width}")
        if isinstance(model, SparkNet) and len(set(widths.values())) != 1:
            raise ValueError("SparkNet needs one uniform backbone width")
        return widths
    rate = float(cast(float, rate))
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    target = (1 - rate) * count_trainable_params(model)
    best: tuple[float, int, dict[str, int]] | None = None
    for n in range(ref, 0, -1):  # widest first, so ties keep the wider model
        widths = _uniform_widths(groups, n / ref)
        err = abs(planned_params(model, widths) - target)
        if best is None or err < best[0]:
            best = (err, n, widths)
    return cast(tuple, best)[2]


# --------------------------------------------------------------------------
# Channel selection and physical rebuild
# --------------------------------------------------------------------------


def select_channels(
    model: nn.Module,
    widths: Mapping[str, int],
    criterion: str = "l2_group",
    *,
    p: float = 2.0,
    include_bn_bias: bool = False,
    prune_stem: bool = True,
    tp_compat: bool = True,
    taylor_mode: str = "molchanov",
    taylor_grads: Sequence[Mapping[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Rank every group and keep its top ``widths[name]`` channels (sorted indices).

    ``tp_compat`` (magnitude/Taylor criteria) reproduces Torch-Pruning 1.6's
    ``MetaPruner.step()`` on these graphs: every group is scored on the
    *unpruned* model before any channel is removed, and depthwise filters are
    left out of the score (TP's importance never matches their handler).
    Without it, groups are decided in forward order, a producer slice only
    counts the input columns that survived the group before it, and depthwise
    filters are scored. ``l1_producer`` always runs the sequential forward
    way, reproducing ``prune_sparknet`` / ``prune_ds_cnn``. Returns
    ``(keep, scores)``.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
    keep: dict[str, torch.Tensor] = {}
    scores: dict[str, torch.Tensor] = {}
    sequential = criterion == "l1_producer" or not tp_compat
    for group in dependency_groups(model, prune_stem=prune_stem):
        score = _group_importance(model, group, criterion, keep if sequential else {}, p=p,
                                  include_bn_bias=include_bn_bias, tp_compat=tp_compat,
                                  taylor_mode=taylor_mode, taylor_grads=taylor_grads)
        scores[group.name] = score
        k = int(widths[group.name])
        # Stable ranking: equal scores keep the lower channel index.
        order = torch.sort(-score, stable=True).indices
        keep[group.name] = order[:k].sort().values
    return keep, scores


def _copy_bn(old: nn.BatchNorm2d, new: nn.BatchNorm2d, idx: torch.Tensor | None) -> None:
    sel = (lambda t: t) if idx is None else (lambda t: t[idx])
    with torch.no_grad():
        for name in ("weight", "bias", "running_mean", "running_var"):
            src, dst = getattr(old, name), getattr(new, name)
            if src is None or dst is None:
                raise ValueError("group pruning needs affine BatchNorms with running stats")
            dst.copy_(sel(src))
        if old.num_batches_tracked is not None and new.num_batches_tracked is not None:
            new.num_batches_tracked.copy_(old.num_batches_tracked)
    new.eps, new.momentum = old.eps, old.momentum


def _rebuild_sparknet(model: SparkNet, keep: Mapping[str, torch.Tensor]) -> SparkNet:
    old_blocks = [cast(TCSBlock, b) for b in model.blocks]
    width = len(keep["blocks.0"])
    ref = next(model.parameters())
    new = SparkNet(
        n_feat=old_blocks[0].depthwise.in_channels,
        num_classes=model.fc.out_features,
        channels=width,
        gate_channels=model.gate_conv.out_channels,
        kernels=tuple(b.depthwise.kernel_size[-1] for b in old_blocks),
        block_bn_eps=old_blocks[0].bn.eps,
        sparsity_weight=model.sparsity_weight,
        input_shape=model.input_shape,
    ).to(device=ref.device, dtype=ref.dtype)
    new_blocks = [cast(TCSBlock, b) for b in new.blocks]
    keep_in = torch.arange(old_blocks[0].depthwise.in_channels, device=ref.device)
    with torch.no_grad():
        for i, (old, nb) in enumerate(zip(old_blocks, new_blocks)):
            out = keep[f"blocks.{i}"].to(ref.device)
            nb.depthwise.weight.copy_(old.depthwise.weight[keep_in])
            pw = cast(nn.Conv2d, old.pointwise)
            cast(nn.Conv2d, nb.pointwise).weight.copy_(pw.weight[out][:, keep_in])
            _copy_bn(old.bn, nb.bn, out)
            if old.res_conv is not None:
                assert nb.res_conv is not None and old.res_bn is not None and nb.res_bn is not None
                nb.res_conv.weight.copy_(old.res_conv.weight[out][:, keep_in])
                _copy_bn(old.res_bn, nb.res_bn, out)
            keep_in = out
        new.gate_conv.weight.copy_(model.gate_conv.weight[:, keep_in])
        if model.gate_conv.bias is not None:
            cast(torch.Tensor, new.gate_conv.bias).copy_(model.gate_conv.bias)
    new.gate_bn.load_state_dict(model.gate_bn.state_dict())
    new.gate_bn.eps, new.gate_bn.momentum = model.gate_bn.eps, model.gate_bn.momentum
    new.fc.load_state_dict(model.fc.state_dict())
    new.train(model.training)
    return new


def _rebuild_ds_cnn(model: DSCNN, keep: Mapping[str, torch.Tensor]) -> DSCNN:
    stem = cast(nn.Conv2d, model.stem[0])
    ref = next(model.parameters())
    stem_keep = keep.get("stem")
    blocks = list(model.blocks)
    new = DSCNN(
        input_shape=model.input_shape,
        num_classes=model.fc.out_features,
        initial_channels=len(stem_keep) if stem_keep is not None else stem.out_channels,
        initial_kernel=stem.kernel_size[0],
        initial_stride=stem.stride[0],
        block_channels=[len(keep[f"blocks.{i}"]) for i in range(len(blocks))],
        dropout=model.dropout.p,
    ).to(device=ref.device, dtype=ref.dtype)
    new_stem = cast(nn.Conv2d, new.stem[0])
    if (new_stem.kernel_size, new_stem.stride, new_stem.padding) != (
        stem.kernel_size, stem.stride, stem.padding
    ):
        raise ValueError("DS-CNN stem geometry is not reproducible by the constructor")
    with torch.no_grad():
        if stem_keep is None:
            new_stem.weight.copy_(stem.weight)
            keep_in: torch.Tensor | None = None
        else:
            stem_keep = stem_keep.to(ref.device)
            new_stem.weight.copy_(stem.weight[stem_keep])
            keep_in = stem_keep
        _copy_bn(cast(nn.BatchNorm2d, model.stem[1]), cast(nn.BatchNorm2d, new.stem[1]), keep_in)
        for i, (old, nb) in enumerate(zip(blocks, new.blocks)):
            out = keep[f"blocks.{i}"].to(ref.device)
            dw = old.depthwise.weight if keep_in is None else old.depthwise.weight[keep_in]
            nb.depthwise.weight.copy_(dw)
            _copy_bn(old.bn1, nb.bn1, keep_in)
            pw = old.pointwise.weight[out]
            nb.pointwise.weight.copy_(pw if keep_in is None else pw[:, keep_in])
            _copy_bn(old.bn2, nb.bn2, out)
            keep_in = out
        new.fc.weight.copy_(model.fc.weight[:, cast(torch.Tensor, keep_in)])
        if model.fc.bias is not None:
            cast(torch.Tensor, new.fc.bias).copy_(model.fc.bias)
    new.train(model.training)
    return new


def rebuild(model: nn.Module, keep: Mapping[str, torch.Tensor]) -> nn.Module:
    """A fresh, physically narrower copy of ``model`` holding the ``keep`` channels."""
    if isinstance(model, SparkNet):
        widths = {len(v) for v in keep.values()}
        if len(widths) != 1:
            raise ValueError("SparkNet needs one uniform backbone width")
        return _rebuild_sparknet(model, keep)
    if isinstance(model, DSCNN):
        return _rebuild_ds_cnn(model, keep)
    raise TypeError(type(model).__name__)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_pruned(model: nn.Module) -> int:
    """Reject masked/zeroed structure and return the executed trainable param count.

    Raises if any Conv/Linear output channel is all-zero, any BatchNorm
    channel has gamma exactly 0 (a masked channel whose output is a constant)
    or non-finite / negative running statistics, any parameter is non-finite,
    or a ``torch.nn.utils.prune`` reparametrisation (``*_orig`` / ``*_mask``)
    is still attached.
    """
    problems: list[str] = []
    for name, _ in model.named_buffers():
        if name.endswith("_mask"):
            problems.append(f"{name}: pruning mask still attached")
    for name, param in model.named_parameters():
        if name.endswith("_orig"):
            problems.append(f"{name}: pruning reparametrisation still attached")
        if not torch.isfinite(param).all():
            problems.append(f"{name}: non-finite values")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            rows = module.weight.detach().reshape(module.weight.shape[0], -1)
            dead = (rows == 0).all(dim=1).nonzero().flatten().tolist()
            if dead:
                problems.append(f"{name}: all-zero output channels {dead}")
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                dead = (module.weight.detach() == 0).nonzero().flatten().tolist()
                if dead:
                    problems.append(f"{name}: zero gamma (masked) channels {dead}")
            var, mean = module.running_var, module.running_mean
            if var is not None and (not torch.isfinite(var).all() or (var < 0).any()):
                problems.append(f"{name}: degenerate running_var")
            if mean is not None and not torch.isfinite(mean).all():
                problems.append(f"{name}: non-finite running_mean")
    if problems:
        raise ValueError("pruned model failed verification:\n  " + "\n  ".join(problems))
    return count_trainable_params(model)


# --------------------------------------------------------------------------
# One-shot entry point
# --------------------------------------------------------------------------


@dataclass
class GroupPruneResult:
    model: nn.Module
    criterion: str
    widths: dict[str, int]
    keep: dict[str, list[int]]
    original_params: int
    pruned_params: int
    target_rate: float | None
    achieved_rate: float
    scores: dict[str, list[float]] = field(repr=False, default_factory=dict)

    def summary(self) -> dict:
        return {
            "criterion": self.criterion,
            "widths": self.widths,
            "original_params": self.original_params,
            "pruned_params": self.pruned_params,
            "target_rate": self.target_rate,
            "achieved_rate": self.achieved_rate,
            "keep": self.keep,
        }


def group_prune(
    model: nn.Module,
    *,
    rate: float | None = None,
    width: int | Mapping[str, int] | None = None,
    criterion: str = "l2_group",
    batches: Batches | None = None,
    loss_fn: LossFn | None = None,
    taylor_batches: int | None = 4,
    p: float = 2.0,
    include_bn_bias: bool = False,
    prune_stem: bool = True,
    tp_compat: bool = True,
    taylor_mode: str = "molchanov",
) -> GroupPruneResult:
    """One-shot dependency-group channel pruning to a parameter ``rate`` or a ``width``.

    ``criterion``: ``"l2_group"`` (Torch-Pruning ``GroupMagnitudeImportance(p)``:
    per-slice sum of ``|w|^p``, mean over the group's slices, BN gamma included,
    BN beta only with ``include_bn_bias`` -- TP's ``bias=False`` default),
    ``"taylor"`` (first-order Taylor over ``taylor_batches`` calibration
    batches from ``batches()``; needs ``batches`` and ``loss_fn(logits, y)``;
    ``taylor_mode="molchanov"`` scores each slice by ``|sum(w * dL/dw)|`` per
    batch, averaged, BN included; ``"tp"`` reproduces TP's
    ``GroupTaylorImportance()``) or ``"l1_producer"`` (the legacy ranking: L1
    of the producing pointwise/stem conv only). ``tp_compat``: see
    :func:`select_channels`. ``prune_stem`` (DS-CNN only) also prunes the stem
    conv's output channels, as Torch-Pruning would; ``False`` keeps the
    legacy ``prune_ds_cnn`` scope. The input model is not modified.
    """
    original = count_trainable_params(model)
    widths = resolve_widths(model, rate=rate, width=width, prune_stem=prune_stem)
    grads = None
    if criterion == "taylor":
        if batches is None or loss_fn is None:
            raise ValueError("taylor importance needs batches and loss_fn")
        names = {f"{s.module}.{s.param}"
                 for g in dependency_groups(model, prune_stem=prune_stem) for s in g.slices}
        grads = taylor_gradients(model, batches, loss_fn, num_batches=taylor_batches,
                                 param_names=sorted(names))
    keep, scores = select_channels(model, widths, criterion, p=p,
                                   include_bn_bias=include_bn_bias, prune_stem=prune_stem,
                                   tp_compat=tp_compat, taylor_mode=taylor_mode,
                                   taylor_grads=grads)
    pruned = rebuild(model, keep)
    executed = verify_pruned(pruned)
    planned = planned_params(model, widths)
    if executed != planned:
        raise AssertionError(f"rebuilt model has {executed} params, planned {planned}")
    return GroupPruneResult(
        model=pruned,
        criterion=criterion,
        widths=dict(widths),
        keep={k: v.tolist() for k, v in keep.items()},
        original_params=original,
        pruned_params=executed,
        target_rate=rate,
        achieved_rate=1 - executed / original,
        scores={k: v.tolist() for k, v in scores.items()},
    )
