"""Edge Impulse's default keyword-spotting CNN, with optional dendritic head.

The Edge Impulse "1D Convolutional (Default)" KWS model (the model used in
arXiv 2605.15647, Perforated Backprop for KWS) treats MFCC coefficients as
channels and time as the conv axis::

    Conv1D(8, kernel_size=3, activation='relu', padding='same')
    MaxPooling1D(pool_size=2, strides=2, padding='same')
    Dropout(0.25)
    Conv1D(16, kernel_size=3, activation='relu', padding='same')
    MaxPooling1D(pool_size=2, strides=2, padding='same')
    Dropout(0.25)
    Flatten()
    Dense(classes, activation='softmax')

Keras ``padding='same'`` pooling with stride 2 yields ``ceil(T / 2)`` frames,
which is ``ceil_mode=True`` here (T=101 -> 51 -> 26). Unlike the SparkNet and
DS-CNN families, almost all parameters sit in the Flatten -> Dense head, whose
fan-in (``c2 * T'``) is large and whose fan-out (the classes) is fixed: the
setting where dendrites are expected to beat plain widening.

Head dendrites (``head_dendrites`` = d > 0) add a :class:`DendriticPointwise`
branch to the logits: ``logits = fc(z) + dendrites(z)``, costing
``num_classes * d * (f + 2)`` params. ``head_dendrite_layout``:

* ``"time_local"``: ``z`` is flattened time-major (index ``t * c2 + c``), so a
  window of ``f`` (a multiple of ``c2``) inputs is every channel over
  ``f / c2`` consecutive pooled frames. Each class's span of ``d * f`` inputs
  is frame aligned and never wraps past the last frame (see
  :func:`local_window_index`); classes' spans are spread evenly over time.
* ``"full"``: ``f = len(z)``; each dendrite sees every input (Brenner/PAI style).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from kws.models.sparknet import DendriticPointwise

HEAD_LAYOUTS = ("time_local", "full")
INPUT_NORMS = ("none", "cmvn", "batchnorm")


class InputNorm(nn.Module):
    """Parameter-free normalization of ``(B, n_mfcc, T)`` MFCC input.

    ``cmvn``: per-utterance, per-coefficient mean and variance normalization
    over time, what Edge Impulse's MFCC block does (its normalization window
    defaults to the whole 1 s clip). ``batchnorm``: a non-affine BatchNorm1d
    over coefficients (running stats are buffers, not params). ``none``: raw.
    Raw MFCC c0 here has mean ~-34 and std ~24; without normalization and
    without BN, the paper SGD recipe (effective lr 1.0) kills every ReLU.
    """

    def __init__(self, mode: str, n_mfcc: int, eps: float = 1e-5):
        super().__init__()
        if mode not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {mode!r}")
        self.mode, self.eps = mode, eps
        self.bn = nn.BatchNorm1d(n_mfcc, affine=False) if mode == "batchnorm" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "cmvn":
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True, unbiased=False)
            return (x - mean) / (std + self.eps)
        if self.bn is not None:
            return self.bn(x)
        return x


def local_window_index(channels: int, dendrites: int, fan_in: int, in_channels: int,
                       align: int = 1) -> list[int]:
    """Input index for a :class:`DendriticPointwise`, windows kept in range.

    Neuron ``n``'s ``dendrites`` contiguous windows of ``fan_in`` form one span
    of ``dendrites * fan_in`` inputs, as in ``DendriticPointwise``; here the
    span starts at ``n * (in - span) / (channels - 1)`` rounded down to a
    multiple of ``align``, so no span wraps from the last input to the first.
    When the span is longer than the input it cannot fit, and the module's own
    cyclic layout is returned unchanged.
    """
    span = dendrites * fan_in
    if span > in_channels:
        return [(n * in_channels // channels + j * fan_in + i) % in_channels
                for n in range(channels) for j in range(dendrites) for i in range(fan_in)]
    room = in_channels - span
    index = []
    for n in range(channels):
        start = 0 if channels == 1 else (n * room) // (channels - 1)
        start -= start % align
        index.extend(start + k for k in range(span))
    return index


def dendritic_branch(channels: int, dendrites: int, fan_in: int, in_channels: int,
                     align: int = 1) -> DendriticPointwise:
    """A ``DendriticPointwise`` reading in-range, ``align``-snapped windows."""
    branch = DendriticPointwise(channels, dendrites, fan_in, in_channels=in_channels)
    index = local_window_index(channels, dendrites, fan_in, in_channels, align)
    branch.index = torch.tensor(index, dtype=torch.long)  # non-persistent buffer
    return branch


def pooled_length(length: int, times: int = 2) -> int:
    for _ in range(times):
        length = math.ceil(length / 2)
    return length


class EIConv1d(nn.Module):
    def __init__(self, n_mfcc: int, time_steps: int, num_classes: int,
                 c1: int = 8, c2: int = 16, kernel_size: int = 3, dropout: float = 0.25,
                 head_dropout: float = 0.0, batchnorm: bool = False,
                 head_dendrites: int = 0, head_dendrite_fan_in: int = 0,
                 head_dendrite_layout: str = "time_local", input_norm: str = "cmvn"):
        super().__init__()
        if head_dendrite_layout not in HEAD_LAYOUTS:
            raise ValueError(f"head_dendrite_layout must be one of {HEAD_LAYOUTS}, "
                             f"got {head_dendrite_layout!r}")
        self.input_shape = (n_mfcc, time_steps)
        self.time_major = head_dendrites > 0 and head_dendrite_layout == "time_local"
        pad = kernel_size // 2

        def stage(cin: int, cout: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv1d(cin, cout, kernel_size, padding=pad,
                                                 bias=not batchnorm)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(cout))
            layers += [nn.ReLU(inplace=True), nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(dropout)]
            return layers

        self.input_norm = InputNorm(input_norm, n_mfcc)
        self.features = nn.Sequential(*stage(n_mfcc, c1), *stage(c1, c2))
        self.frames = pooled_length(time_steps)
        in_features = c2 * self.frames
        self.head_dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.head_dendrites: DendriticPointwise | None = None
        if head_dendrites > 0:
            if head_dendrite_layout == "full":
                fan_in = head_dendrite_fan_in or in_features
                if fan_in != in_features:
                    raise ValueError(f"'full' layout needs fan_in == {in_features}, got {fan_in}")
                self.head_dendrites = DendriticPointwise(
                    num_classes, head_dendrites, fan_in, in_channels=in_features)
            else:
                fan_in = head_dendrite_fan_in
                if fan_in <= 0 or fan_in % c2:
                    raise ValueError(f"time_local fan_in must be a positive multiple of c2={c2}, "
                                     f"got {fan_in}")
                self.head_dendrites = dendritic_branch(
                    num_classes, head_dendrites, fan_in, in_features, align=c2)

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten ``(B, c2, T')``; time-major (``t * c2 + c``) for time_local."""
        if self.time_major:
            x = x.transpose(1, 2)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        z = self.head_dropout(self.flatten(self.features(self.input_norm(x))))
        logits = self.fc(z)
        if self.head_dendrites is not None:
            logits = logits + self.head_dendrites(z[:, :, None, None]).flatten(1)
        return logits


def build_ei_conv1d(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> EIConv1d:
    return EIConv1d(
        n_mfcc=input_shape[0],
        time_steps=input_shape[1],
        num_classes=num_classes,
        c1=model_cfg.get("c1", 8),
        c2=model_cfg.get("c2", 16),
        kernel_size=model_cfg.get("kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.25),
        head_dropout=model_cfg.get("head_dropout", 0.0),
        batchnorm=model_cfg.get("batchnorm", False),
        head_dendrites=model_cfg.get("head_dendrites", 0),
        head_dendrite_fan_in=model_cfg.get("head_dendrite_fan_in", 0),
        head_dendrite_layout=model_cfg.get("head_dendrite_layout", "time_local"),
        input_norm=model_cfg.get("input_norm", "cmvn"),
    )
