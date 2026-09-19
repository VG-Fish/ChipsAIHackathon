"""Integer-only SparkNet inference for the Raspberry Pi RP2040.

The RP2040's Cortex-M0+ cores have no FPU and no SIMD, so this module lowers
a trained SparkNet -- a scratch checkpoint or one grown with PerforatedAI
dendrites -- to a graph of integer multiply-accumulates, shifts and table
lookups, and emits it as C for ``rp2040_c/kws_engine.c``.

Lowering, in order:

1. :func:`fold_sparknet` folds every BatchNorm into its 1x1 conv.  The float
   forward of the folded graph must reproduce the torch model
   (:meth:`FoldedSparkNet.forward`).
2. :func:`calibrate` records per-channel activation ranges on a calibration
   set (never the test split).
3. :func:`quantize` produces a :class:`QuantizedSparkNet`: int8 weights,
   symmetric per output channel; activations per-channel symmetric integers of
   ``act_bits`` (8 or 16) bits, with the post-ReLU block outputs on the
   non-negative half.  Every per-channel input scale is folded into the
   consuming weights, so each kernel needs one output scale per channel.
   Requantization is ``(sum_i acc_i * M_i + B) >> S`` in int64 with the
   rounding term inside ``B``.  The dendrite tanh and the gate's
   ``clamp(tanh(a) + 0.5, 0, 1)`` are 255-entry lookup tables.  Dendrites
   may sit on any block's pointwise conv and on the classifier (fc).
4. :meth:`QuantizedSparkNet.forward_int` is the numpy integer reference.  It
   is the specification: the C engine must match its logits bit for bit.
5. :func:`write_c_model` emits the constants as a C header.

Tensor layout everywhere is ``[channel][time]`` for one clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from kws.models.sparknet import SparkNet
from kws.optimize.grow_clean_rebuild import PlainSequential, RebuiltDendriteModule

GATE_SATURATION = math.atanh(0.5)  # |a| beyond this: clamp(tanh(a) + 0.5, 0, 1) is 0 or 1
LUT_HALF = 127  # LUT index range [-127, 127]
TANH_ONE = 32767  # dendrite tanh LUT output scale
GATE_ONE = 255  # gate LUT output scale
LOGIT_FRACTION_BITS = 16  # logits are Q16 int32
DENDRITE_LUT_MAX_RANGE = 4.0  # tanh(4) = 0.9993; wider ranges waste LUT resolution
MULTIPLIER_BITS = 30
MAX_SHIFT = 62


# ---------------------------------------------------------------------------
# 1. folding


@dataclass
class FoldedBlock:
    """One TCS block with every BatchNorm folded; float64."""

    dw: np.ndarray  # (Cin, K) depthwise taps
    pw: np.ndarray  # (C, Cin) on the depthwise output
    pw_b: np.ndarray  # (C,)
    res: np.ndarray | None = None  # (C, Cin) on the block input
    res_b: np.ndarray | None = None
    den: np.ndarray | None = None  # (C, Cin) dendrite on the depthwise output
    den_b: np.ndarray | None = None
    den_skip: np.ndarray | None = None  # (C,)

    @property
    def kernel(self) -> int:
        return int(self.dw.shape[1])


@dataclass
class FoldedSparkNet:
    blocks: list[FoldedBlock]
    gate_w: np.ndarray  # (G, C)
    gate_b: np.ndarray
    fc_w: np.ndarray  # (classes, G)
    fc_b: np.ndarray
    fc_den: np.ndarray | None = None  # (classes, G) dendrite on the pooled gate
    fc_den_b: np.ndarray | None = None
    fc_den_skip: np.ndarray | None = None  # (classes,)

    def forward(self, x: np.ndarray, observe: Callable[[str, np.ndarray], None] | None = None) -> np.ndarray:
        """Float logits of ``x`` (B, F, T).  ``observe(name, tensor)`` sees every activation."""
        seen = observe or (lambda name, value: None)
        h = np.asarray(x, dtype=np.float64)
        seen("input", h)
        for index, block in enumerate(self.blocks):
            d = depthwise(h, block.dw)
            seen(f"b{index}.dw", d)
            y = np.matmul(block.pw, d) + block.pw_b[:, None]
            if block.res is not None:
                y = y + np.matmul(block.res, h) + block.res_b[:, None]
            if block.den is not None:
                pre = np.matmul(block.den, d) + block.den_b[:, None]
                seen(f"b{index}.den", pre)
                y = y + block.den_skip[:, None] * np.tanh(pre)
            h = np.maximum(y, 0.0)
            seen(f"b{index}.out", h)
        gate = np.matmul(self.gate_w, h) + self.gate_b[:, None]
        z = np.clip(np.tanh(gate) + 0.5, 0.0, 1.0).mean(axis=2)
        logits = z @ self.fc_w.T + self.fc_b
        if self.fc_den is not None:
            pre = z @ self.fc_den.T + self.fc_den_b
            seen("fc.den", pre[:, :, None])
            logits = logits + self.fc_den_skip * np.tanh(pre)
        return logits

    @property
    def parameter_count(self) -> int:
        count = self.gate_w.size + self.gate_b.size + self.fc_w.size + self.fc_b.size
        for extra in (self.fc_den, self.fc_den_b, self.fc_den_skip):
            count += 0 if extra is None else extra.size
        for block in self.blocks:
            count += block.dw.size + block.pw.size + block.pw_b.size
            for extra in (block.res, block.res_b, block.den, block.den_b, block.den_skip):
                count += 0 if extra is None else extra.size
        return int(count)


def depthwise(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """'Same' zero-padded depthwise temporal conv of (B, C, T) by (C, K); keeps x's dtype."""
    batch, channels, frames = x.shape
    kernel = taps.shape[1]
    pad = kernel // 2
    padded = np.zeros((batch, channels, frames + 2 * pad), dtype=x.dtype)
    padded[:, :, pad:pad + frames] = x
    out = np.zeros_like(x)
    for k in range(kernel):
        out += taps[None, :, k, None].astype(x.dtype) * padded[:, :, k:k + frames]
    return out


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().double().numpy()


def _batchnorm_affine(bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    scale = _numpy(bn.weight) / np.sqrt(_numpy(bn.running_var) + bn.eps)
    return scale, _numpy(bn.bias) - _numpy(bn.running_mean) * scale


def _fold_pointwise(module: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """``(W, b)`` of a 1x1 Conv2d, optionally followed by BatchNorms (PlainSequential)."""
    layers = list(module.model) if isinstance(module, PlainSequential) else [module]
    conv, rest = layers[0], layers[1:]
    if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1) or conv.groups != 1:
        raise ValueError(f"expected a 1x1 Conv2d, got {conv!r}")
    weight = _numpy(conv.weight)[:, :, 0, 0]
    bias = _numpy(conv.bias) if conv.bias is not None else np.zeros(weight.shape[0])
    for layer in rest:
        weight, bias = _apply_bn(layer, weight, bias)
    return weight, bias


def _apply_bn(bn: nn.Module, weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bn, nn.Identity):
        return weight, bias
    if not isinstance(bn, nn.BatchNorm2d):
        raise ValueError(f"cannot fold {type(bn).__name__}")
    scale, shift = _batchnorm_affine(bn)
    return weight * scale[:, None], bias * scale + shift


def _single_tanh_dendrite(module: RebuiltDendriteModule, where: str) -> tuple[nn.Module, nn.Module, np.ndarray]:
    """``(dendrite, main, skip)`` of a rebuilt module with one tanh dendrite."""
    if module.n_dendrites != 1:
        raise ValueError(f"{where}: {module.n_dendrites} dendrites; one is supported")
    if module.forward_function is not torch.tanh:
        raise ValueError(f"{where}: the dendrite forward function must be torch.tanh")
    return module.layer_array[0], module.layer_array[-1], _numpy(module.skip_weights[0])[0]


def _linear(module: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(module, nn.Linear):
        raise ValueError(f"expected an nn.Linear, got {module!r}")
    bias = _numpy(module.bias) if module.bias is not None else np.zeros(module.out_features)
    return _numpy(module.weight), bias


def fold_sparknet(model: nn.Module) -> FoldedSparkNet:
    """Fold a SparkNet: plain, or rebuilt with one tanh dendrite on any pointwise conv or fc."""
    if not isinstance(model, SparkNet):
        raise TypeError(f"expected a SparkNet, got {type(model).__name__}")
    if model.training:
        raise ValueError("fold an eval-mode model (BatchNorm running statistics)")
    blocks = []
    for index, block in enumerate(model.blocks):
        if not isinstance(block.depthwise, nn.Conv2d):
            raise ValueError(f"block {index}: only pointwise dendrites are supported")
        dw = _numpy(block.depthwise.weight)[:, 0, 0, :]
        folded = FoldedBlock(dw=dw, pw=np.empty(0), pw_b=np.empty(0))
        pointwise = block.pointwise
        if isinstance(pointwise, RebuiltDendriteModule):
            dendrite, main, skip = _single_tanh_dendrite(pointwise, f"block {index}")
            main_w, main_b = _fold_pointwise(main)
            den_w, den_b = _fold_pointwise(dendrite)
            # A BatchNorm after the dendrite module is affine per channel: it
            # scales the main path and the skip weight alike.
            if isinstance(block.bn, nn.BatchNorm2d):
                scale, shift = _batchnorm_affine(block.bn)
                main_w, main_b, skip = main_w * scale[:, None], main_b * scale + shift, skip * scale
            elif not isinstance(block.bn, nn.Identity):
                raise ValueError(f"block {index}: unexpected {type(block.bn).__name__} after the dendrite")
            folded.den, folded.den_b, folded.den_skip = den_w, den_b, skip
        else:
            main_w, main_b = _fold_pointwise(pointwise)
            main_w, main_b = _apply_bn(block.bn, main_w, main_b)
        folded.pw, folded.pw_b = main_w, main_b
        if block.res_conv is not None:
            folded.res, folded.res_b = _apply_bn(block.res_bn, *_fold_pointwise(block.res_conv))
        blocks.append(folded)
    gate_w, gate_b = _apply_bn(model.gate_bn, *_fold_pointwise(model.gate_conv))
    if isinstance(model.fc, RebuiltDendriteModule):
        dendrite, main, skip = _single_tanh_dendrite(model.fc, "fc")
        return FoldedSparkNet(blocks, gate_w, gate_b, *_linear(main), *_linear(dendrite), skip)
    return FoldedSparkNet(blocks, gate_w, gate_b, *_linear(model.fc))


# ---------------------------------------------------------------------------
# 2. calibration


@dataclass
class Calibration:
    """Per-channel range of every activation, from a calibration set."""

    ranges: dict[str, np.ndarray]
    method: str
    examples: int


def calibrate(folded: FoldedSparkNet, batches: Iterable[np.ndarray], method: str = "max") -> Calibration:
    """Per-channel |activation| range: ``max`` or ``pNN.NN`` (a percentile, e.g. ``p99.99``).

    Dendrite pre-activations get one range per block (their LUT is shared by
    the block's channels).
    """
    samples: dict[str, list[np.ndarray]] = {}
    examples = 0

    def observe(name: str, value: np.ndarray) -> None:
        values = np.abs(value).transpose(1, 0, 2).reshape(value.shape[1], -1)
        if name.endswith(".den"):
            values = values.reshape(1, -1)
        samples.setdefault(name, []).append(values)

    for batch in batches:
        folded.forward(batch, observe)
        examples += len(batch)
    ranges = {}
    for name, parts in samples.items():
        values = np.concatenate(parts, axis=1)
        if method == "max":
            ranges[name] = values.max(axis=1)
        elif method.startswith("p"):
            ranges[name] = np.percentile(values, float(method[1:]), axis=1)
        else:
            raise ValueError(f"unknown calibration method {method!r}")
    return Calibration(ranges, method, examples)


# ---------------------------------------------------------------------------
# 3. quantization


@dataclass
class Requant:
    """``(sum_i acc_i * multipliers[i] + bias) >> shift`` per output channel."""

    multipliers: list[np.ndarray]  # each int64 (C,), values < 2^31
    bias: np.ndarray  # int64 (C,), rounding term included
    shift: np.ndarray  # int64 (C,)


def make_requant(real_multipliers: list[np.ndarray], real_bias: np.ndarray | None = None) -> Requant:
    """Integer multipliers and shift for ``round(sum_i acc_i * m_i + b)``."""
    channels = len(real_multipliers[0])
    bias = np.zeros(channels) if real_bias is None else np.asarray(real_bias, dtype=np.float64)
    largest = np.max(np.abs(np.stack(real_multipliers)), axis=0)
    shift = np.empty(channels, dtype=np.int64)
    for c in range(channels):
        exponent = math.frexp(largest[c])[1] if largest[c] > 0 else 0  # largest in [2^(e-1), 2^e)
        shift[c] = min(max(MULTIPLIER_BITS - exponent, 1), MAX_SHIFT)
    scale = np.ldexp(1.0, shift)
    multipliers = [np.round(np.asarray(m) * scale).astype(np.int64) for m in real_multipliers]
    rounding = np.left_shift(np.int64(1), shift - 1)
    total_bias = np.round(bias * scale)
    if np.any(np.abs(total_bias) >= 2.0 ** 62):
        raise OverflowError("requantization bias does not fit int64")
    return Requant(multipliers, total_bias.astype(np.int64) + rounding, shift)


def quantize_weights(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per output row: ``weight ~= q * scale[:, None]``."""
    scale = np.max(np.abs(weight), axis=1) / 127.0
    scale = np.where(scale > 0, scale, 1.0)
    return np.clip(np.round(weight / scale[:, None]), -127, 127).astype(np.int64), scale


@dataclass
class QuantizedBlock:
    cin: int
    cout: int
    kernel: int
    dw_w: np.ndarray  # (Cin, K) int8
    dw_rq: Requant  # depthwise accumulator -> signed activation
    pw_w: np.ndarray  # (C, Cin) int8
    res_w: np.ndarray | None  # (C, Cin) int8
    den_w: np.ndarray | None  # (C, Cin) int8
    den_rq: Requant | None  # dendrite accumulator -> LUT index
    den_lut: np.ndarray | None  # (255,) int16, tanh * TANH_ONE
    out_rq: Requant  # [acc_pw, acc_res, tanh_q] -> block output
    # float scales kept for reports and debugging
    scales: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class QuantizedSparkNet:
    act_bits: int
    input_scale: np.ndarray  # (F,) float: x_q = round(x / input_scale)
    blocks: list[QuantizedBlock]
    gate_w: np.ndarray  # (G, C) int8
    gate_rq: Requant  # gate accumulator -> LUT index
    gate_lut: np.ndarray  # (255,) uint8
    fc_w: np.ndarray  # (classes, G) int8
    fc_rq: Requant  # [z-sum accumulator, fc dendrite tanh] -> Q16 logit
    n_frames: int
    calibration: str = ""
    fc_den_w: np.ndarray | None = None  # (classes, G) int8, on the z-sum
    fc_den_rq: Requant | None = None  # fc dendrite accumulator -> LUT index
    fc_den_lut: np.ndarray | None = None  # (255,) int16, tanh * TANH_ONE

    @property
    def act_max(self) -> int:
        return (1 << (self.act_bits - 1)) - 1

    @property
    def relu_max(self) -> int:
        return 255 if self.act_bits == 8 else self.act_max

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        """Float features (B, F, T) -> the integer input (round half up, saturating)."""
        scaled = np.asarray(x, dtype=np.float64) / self.input_scale[None, :, None]
        return np.clip(np.floor(scaled + 0.5), -self.act_max, self.act_max).astype(np.int64)

    def forward_int(self, xq: np.ndarray) -> np.ndarray:
        """Q16 int64 logits of integer inputs (B, F, T): the bit-exact specification."""
        h = np.asarray(xq, dtype=np.int64)
        for block in self.blocks:
            acc = depthwise(h, block.dw_w)
            d = _clip(_requant([acc], block.dw_rq), -self.act_max, self.act_max)
            terms = [np.matmul(block.pw_w, d)]
            if block.res_w is not None:
                terms.append(np.matmul(block.res_w, h))
            else:
                terms.append(np.zeros_like(terms[0]))
            if block.den_w is not None:
                index = _clip(_requant([np.matmul(block.den_w, d)], block.den_rq), -LUT_HALF, LUT_HALF)
                terms.append(block.den_lut[index + LUT_HALF].astype(np.int64))
            else:
                terms.append(np.zeros_like(terms[0]))
            h = _clip(_requant(terms, block.out_rq), 0, self.relu_max)
        index = _clip(_requant([np.matmul(self.gate_w, h)], self.gate_rq), -LUT_HALF, LUT_HALF)
        z_sum = self.gate_lut[index + LUT_HALF].astype(np.int64).sum(axis=2)  # (B, G)
        terms = [z_sum @ self.fc_w.T]  # (B, classes)
        if self.fc_den_w is not None:
            index = _clip(_requant_rows([z_sum @ self.fc_den_w.T], self.fc_den_rq), -LUT_HALF, LUT_HALF)
            terms.append(self.fc_den_lut[index + LUT_HALF].astype(np.int64))
        return _requant_rows(terms, self.fc_rq)

    def logits_float(self, q16: np.ndarray) -> np.ndarray:
        return np.asarray(q16, dtype=np.float64) / (1 << LOGIT_FRACTION_BITS)

    @property
    def weight_bytes(self) -> int:
        """int8 weight bytes (LUTs and requant constants excluded)."""
        total = self.gate_w.size + self.fc_w.size
        total += 0 if self.fc_den_w is None else self.fc_den_w.size
        for block in self.blocks:
            total += block.dw_w.size + block.pw_w.size
            total += 0 if block.res_w is None else block.res_w.size
            total += 0 if block.den_w is None else block.den_w.size
        return int(total)


def _requant(terms: list[np.ndarray], rq: Requant) -> np.ndarray:
    """(B, C, T) accumulators -> (sum_i acc_i * M_i + B) >> S, channel on axis 1."""
    total = rq.bias[None, :, None]
    for acc, multiplier in zip(terms, rq.multipliers):
        total = total + acc.astype(np.int64) * multiplier[None, :, None]
    return np.right_shift(total, rq.shift[None, :, None])


def _requant_rows(terms: list[np.ndarray], rq: Requant) -> np.ndarray:
    """(B, C) accumulators -> (sum_i acc_i * M_i + B) >> S, channel on axis 1."""
    total = rq.bias[None, :]
    for acc, multiplier in zip(terms, rq.multipliers):
        total = total + acc.astype(np.int64) * multiplier[None, :]
    return np.right_shift(total, rq.shift[None, :])


def _clip(x: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(x, low, high)


def _scale(range_: np.ndarray, top: int) -> np.ndarray:
    range_ = np.asarray(range_, dtype=np.float64)
    positive = range_[range_ > 0]
    floor = positive.min() if positive.size else 1.0
    return np.where(range_ > 0, range_, floor) / top


def quantize(folded: FoldedSparkNet, calibration: Calibration, *, act_bits: int = 8,
             n_frames: int = 101) -> QuantizedSparkNet:
    """Lower a folded SparkNet to the integer graph ``forward_int`` runs."""
    if act_bits not in (8, 16):
        raise ValueError("act_bits must be 8 or 16")
    act_max = (1 << (act_bits - 1)) - 1
    relu_max = 255 if act_bits == 8 else act_max
    ranges = calibration.ranges
    input_scale = _scale(ranges["input"], act_max)
    in_scale = input_scale
    blocks = []
    for index, block in enumerate(folded.blocks):
        dw_q, dw_s = quantize_weights(block.dw)
        d_scale = _scale(ranges[f"b{index}.dw"], act_max)
        dw_rq = make_requant([in_scale * dw_s / d_scale])
        pw_q, pw_s = quantize_weights(block.pw * d_scale[None, :])
        out_scale = _scale(ranges[f"b{index}.out"], relu_max)
        multipliers = [pw_s / out_scale]
        bias = block.pw_b.copy()
        res_q = None
        if block.res is not None:
            res_q, res_s = quantize_weights(block.res * in_scale[None, :])
            multipliers.append(res_s / out_scale)
            bias = bias + block.res_b
        else:
            multipliers.append(np.zeros_like(pw_s))
        den_q = den_rq = den_lut = None
        scales = {"input": in_scale, "dw": d_scale, "out": out_scale}
        if block.den is not None:
            den_q, den_s = quantize_weights(block.den * d_scale[None, :])
            lut_range = min(float(ranges[f"b{index}.den"][0]), DENDRITE_LUT_MAX_RANGE)
            lut_step = lut_range / LUT_HALF
            den_rq = make_requant([den_s / lut_step], block.den_b / lut_step)
            steps = np.arange(-LUT_HALF, LUT_HALF + 1) * lut_step
            den_lut = np.round(np.tanh(steps) * TANH_ONE).astype(np.int64)
            multipliers.append(block.den_skip / (TANH_ONE * out_scale))
            scales["den_lut_step"] = np.array([lut_step])
        else:
            multipliers.append(np.zeros_like(pw_s))
        out_rq = make_requant(multipliers, bias / out_scale)
        blocks.append(QuantizedBlock(
            cin=block.dw.shape[0], cout=block.pw.shape[0], kernel=block.kernel,
            dw_w=dw_q, dw_rq=dw_rq, pw_w=pw_q, res_w=res_q, den_w=den_q, den_rq=den_rq,
            den_lut=den_lut, out_rq=out_rq, scales=scales,
        ))
        in_scale = out_scale
    gate_q, gate_s = quantize_weights(folded.gate_w * in_scale[None, :])
    gate_step = GATE_SATURATION / LUT_HALF
    gate_rq = make_requant([gate_s / gate_step], folded.gate_b / gate_step)
    steps = np.arange(-LUT_HALF, LUT_HALF + 1) * gate_step
    gate_lut = np.round(np.clip(np.tanh(steps) + 0.5, 0.0, 1.0) * GATE_ONE).astype(np.int64)
    fc_q, fc_s = quantize_weights(folded.fc_w)
    z_step = 1.0 / (GATE_ONE * n_frames)  # z = z_sum * z_step
    # logit = fc_s * acc * z_step + b [+ skip * tanh(pre)], emitted in Q16
    fc_multipliers = [fc_s * z_step * (1 << LOGIT_FRACTION_BITS)]
    fc_den_q = fc_den_rq = fc_den_lut = None
    if folded.fc_den is not None:
        fc_den_q, fc_den_s = quantize_weights(folded.fc_den)
        lut_step = min(float(ranges["fc.den"][0]), DENDRITE_LUT_MAX_RANGE) / LUT_HALF
        fc_den_rq = make_requant([fc_den_s * z_step / lut_step], folded.fc_den_b / lut_step)
        steps = np.arange(-LUT_HALF, LUT_HALF + 1) * lut_step
        fc_den_lut = np.round(np.tanh(steps) * TANH_ONE).astype(np.int64)
        fc_multipliers.append(folded.fc_den_skip * (1 << LOGIT_FRACTION_BITS) / TANH_ONE)
    fc_rq = make_requant(fc_multipliers, folded.fc_b * (1 << LOGIT_FRACTION_BITS))
    model = QuantizedSparkNet(
        act_bits=act_bits, input_scale=input_scale, blocks=blocks, gate_w=gate_q, gate_rq=gate_rq,
        gate_lut=gate_lut, fc_w=fc_q, fc_rq=fc_rq, n_frames=n_frames,
        calibration=f"{calibration.method} over {calibration.examples} clips",
        fc_den_w=fc_den_q, fc_den_rq=fc_den_rq, fc_den_lut=fc_den_lut,
    )
    check_integer_bounds(model)
    return model


def check_integer_bounds(model: QuantizedSparkNet) -> dict[str, float]:
    """Worst-case |value| (log2) of every int32 accumulator and int64 requant sum.

    The C engine accumulates in int32 and requantizes in int64; this proves
    neither can overflow for any input, not just the calibration set.
    """
    worst: dict[str, float] = {}

    def bound(weights: np.ndarray, activation: int) -> np.ndarray:
        return np.abs(weights).sum(axis=1).astype(np.float64) * activation

    def check(name: str, accumulators: list[np.ndarray], rq: Requant) -> np.ndarray:
        for i, acc in enumerate(accumulators):
            if np.any(acc >= 2.0 ** 31):
                raise OverflowError(f"{name}: int32 accumulator {i} can overflow")
        total = np.abs(rq.bias).astype(np.float64)
        for acc, multiplier in zip(accumulators, rq.multipliers):
            if np.any(np.abs(multiplier) >= 2 ** 31):
                raise OverflowError(f"{name}: multiplier does not fit int32")
            total = total + acc * np.abs(multiplier).astype(np.float64)
        if np.any(total >= 2.0 ** 63):
            raise OverflowError(f"{name}: int64 requantization can overflow")
        worst[name] = float(np.log2(max(total.max(), 1.0)))
        return total

    act_in = model.act_max
    for index, block in enumerate(model.blocks):
        check(f"b{index}.dw", [bound(block.dw_w, act_in)], block.dw_rq)
        terms = [bound(block.pw_w, model.act_max)]
        terms.append(bound(block.res_w, act_in) if block.res_w is not None else np.zeros(block.cout))
        if block.den_w is not None:
            check(f"b{index}.den", [bound(block.den_w, model.act_max)], block.den_rq)
            terms.append(np.full(block.cout, float(TANH_ONE)))
        else:
            terms.append(np.zeros(block.cout))
        check(f"b{index}.out", terms, block.out_rq)
        act_in = model.relu_max
    check("gate", [bound(model.gate_w, model.relu_max)], model.gate_rq)
    z_max = GATE_ONE * model.n_frames
    fc_terms = [bound(model.fc_w, z_max)]
    if model.fc_den_w is not None:
        check("fc.den", [bound(model.fc_den_w, z_max)], model.fc_den_rq)
        fc_terms.append(np.full(model.fc_w.shape[0], float(TANH_ONE)))
    logits = check("fc", fc_terms, model.fc_rq)
    # the C engine stores the shifted fc result, the Q16 logit, in an int32
    if np.any(logits / np.ldexp(1.0, model.fc_rq.shift) >= 2.0 ** 31):
        raise OverflowError("fc: a Q16 logit can overflow int32")
    return worst


# ---------------------------------------------------------------------------
# 5. C emission


def _c_array(ctype: str, name: str, values: np.ndarray, per_line: int = 16) -> str:
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    suffix = "LL" if ctype == "int64_t" else ""
    lines = []
    for start in range(0, len(flat), per_line):
        lines.append("    " + ", ".join(f"{v}{suffix}" for v in flat[start:start + per_line]) + ",")
    body = "\n".join(lines) if lines else "    0,"
    return f"static const {ctype} {name}[{max(len(flat), 1)}] = {{\n{body}\n}};\n"


def write_c_model(model: QuantizedSparkNet, path: str | Path, symbol: str, comment: str = "") -> Path:
    """Emit ``static const`` constants and a ``kws_model_t`` named ``symbol``."""
    path = Path(path)
    out = [
        f"/* Generated by kws.export.rp2040 -- do not edit.\n * {comment}\n"
        f" * act_bits={model.act_bits}, calibration: {model.calibration}\n */\n",
        "#pragma once\n#include \"kws_engine.h\"\n\n",
    ]
    names = []
    for index, block in enumerate(model.blocks):
        p = f"{symbol}_b{index}"
        out.append(_c_array("int8_t", f"{p}_dw_w", block.dw_w))
        out.append(_c_array("int32_t", f"{p}_dw_m", block.dw_rq.multipliers[0]))
        out.append(_c_array("int64_t", f"{p}_dw_b", block.dw_rq.bias))
        out.append(_c_array("uint8_t", f"{p}_dw_s", block.dw_rq.shift))
        out.append(_c_array("int8_t", f"{p}_pw_w", block.pw_w))
        if block.res_w is not None:
            out.append(_c_array("int8_t", f"{p}_res_w", block.res_w))
        if block.den_w is not None:
            out.append(_c_array("int8_t", f"{p}_den_w", block.den_w))
            out.append(_c_array("int32_t", f"{p}_den_m", block.den_rq.multipliers[0]))
            out.append(_c_array("int64_t", f"{p}_den_b", block.den_rq.bias))
            out.append(_c_array("uint8_t", f"{p}_den_s", block.den_rq.shift))
            out.append(_c_array("int16_t", f"{p}_den_lut", block.den_lut))
        for i, tag in enumerate(("m1", "m2", "m3")):
            out.append(_c_array("int32_t", f"{p}_out_{tag}", block.out_rq.multipliers[i]))
        out.append(_c_array("int64_t", f"{p}_out_b", block.out_rq.bias))
        out.append(_c_array("uint8_t", f"{p}_out_s", block.out_rq.shift))
        has_res, has_den = block.res_w is not None, block.den_w is not None
        names.append(
            "    {"
            f" {block.cin}, {block.cout}, {block.kernel},"
            f" {p}_dw_w, {p}_dw_m, {p}_dw_b, {p}_dw_s, {p}_pw_w,"
            f" {p + '_res_w' if has_res else 'NULL'},"
            f" {p + '_den_w' if has_den else 'NULL'},"
            f" {p + '_den_m' if has_den else 'NULL'}, {p + '_den_b' if has_den else 'NULL'},"
            f" {p + '_den_s' if has_den else 'NULL'}, {p + '_den_lut' if has_den else 'NULL'},"
            f" {p}_out_m1, {p}_out_m2, {p}_out_m3, {p}_out_b, {p}_out_s }},"
        )
    out.append(_c_array("int8_t", f"{symbol}_gate_w", model.gate_w))
    out.append(_c_array("int32_t", f"{symbol}_gate_m", model.gate_rq.multipliers[0]))
    out.append(_c_array("int64_t", f"{symbol}_gate_b", model.gate_rq.bias))
    out.append(_c_array("uint8_t", f"{symbol}_gate_s", model.gate_rq.shift))
    out.append(_c_array("uint8_t", f"{symbol}_gate_lut", model.gate_lut))
    out.append(_c_array("int8_t", f"{symbol}_fc_w", model.fc_w))
    out.append(_c_array("int32_t", f"{symbol}_fc_m", model.fc_rq.multipliers[0]))
    out.append(_c_array("int64_t", f"{symbol}_fc_b", model.fc_rq.bias))
    out.append(_c_array("uint8_t", f"{symbol}_fc_s", model.fc_rq.shift))
    has_fc_den = model.fc_den_w is not None
    if has_fc_den:
        out.append(_c_array("int32_t", f"{symbol}_fc_m2", model.fc_rq.multipliers[1]))
        out.append(_c_array("int8_t", f"{symbol}_fc_den_w", model.fc_den_w))
        out.append(_c_array("int32_t", f"{symbol}_fc_den_m", model.fc_den_rq.multipliers[0]))
        out.append(_c_array("int64_t", f"{symbol}_fc_den_b", model.fc_den_rq.bias))
        out.append(_c_array("uint8_t", f"{symbol}_fc_den_s", model.fc_den_rq.shift))
        out.append(_c_array("int16_t", f"{symbol}_fc_den_lut", model.fc_den_lut))
    fc_den = ", ".join(
        f"{symbol}_{name}" if has_fc_den else "NULL"
        for name in ("fc_m2", "fc_den_w", "fc_den_m", "fc_den_b", "fc_den_s", "fc_den_lut")
    )
    # input_scale is float: the frontend divides MFCCs by it before rounding.
    scales = ", ".join(f"{v:.9g}f" for v in model.input_scale)
    out.append(f"static const float {symbol}_input_scale[{len(model.input_scale)}] = {{ {scales} }};\n")
    out.append(f"static const kws_block_t {symbol}_blocks[{len(model.blocks)}] = {{\n" + "\n".join(names) + "\n};\n")
    first = model.blocks[0]
    out.append(
        f"static const kws_model_t {symbol} = {{\n"
        f"    {first.cin}, {model.n_frames}, {model.blocks[-1].cout}, {model.gate_w.shape[0]},"
        f" {model.fc_w.shape[0]}, {len(model.blocks)}, {model.act_max}, {model.relu_max},\n"
        f"    {symbol}_blocks, {symbol}_gate_w, {symbol}_gate_m, {symbol}_gate_b, {symbol}_gate_s,"
        f" {symbol}_gate_lut,\n"
        f"    {symbol}_fc_w, {symbol}_fc_m, {symbol}_fc_b, {symbol}_fc_s,\n"
        f"    {fc_den},\n"
        f"    {symbol}_input_scale,\n}};\n"
    )
    path.write_text("".join(out))
    return path
