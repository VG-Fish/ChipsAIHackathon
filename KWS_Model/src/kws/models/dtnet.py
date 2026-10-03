"""DTNet (Dendritic Timescale Network): dendrites that integrate over time.

Motivation. Over many experiment batches, dendrites added to SparkNet, DS-CNN
and MLP keyword spotters never beat plain widening or deepening at equal
parameter count. The one dendritic mechanism reported to win on Google Speech
Commands at equal params is temporal dendritic heterogeneity (Zheng et al.
2024, Nat Commun 15:277, DH-SNN): each branch low-pass filters its own input
with its own learnable time constant, so a single neuron nonlinearly combines
several timescales. DTNet is a non-spiking, streaming-friendly version: every
temporal operation is a causal first-order IIR, so on a microcontroller it
costs one multiply-add per branch per frame and one state word per branch.

Input ``(B, 1, n_mfcc, T)`` is squeezed to ``(B, n_mfcc, T)``. With
``input_transform: idct`` it is multiplied by the (orthonormal, parameter-free)
inverse of torchaudio's MFCC DCT, recovering the log-mel energies, so channels
are tonotopic and a "local" window is a local frequency band. Then
:class:`~kws.models.ei_conv1d.InputNorm` (``cmvn`` by default, which is
per-utterance and therefore the only non-causal step; use ``batchnorm`` or
``none`` for a strictly causal stream).

Each :class:`DTLayer` has ``N`` neurons of ``K`` branches. Branch ``(n, j)``:

1. reads ``f`` input channels (``layout: local``: a contiguous, non-wrapping
   window; the ``N*K`` window starts are spread evenly over ``[0, C_in - f]``
   in neuron-major order, ``start(n, j) = round((n*K + j) / (N*K - 1) *
   (C_in - f))``, so a neuron's branches sit on adjacent, overlapping bands
   and neurons tile the input from low to high; ``layout: random``:
   ``randperm(C_in)[:f]`` from a generator seeded 0, so the layout is identical
   at load time); ``f >= C_in`` (or ``fan_in: full``) reads every input,
   ``u = w . x + b``;
2. leaky-integrates it, ``v[t] = a v[t-1] + (1 - a) u[t]`` with ``v[-1] = 0``
   and ``a = sigmoid(rho)`` (``tau: hetero``: one ``rho`` per branch;
   ``shared``: one per neuron, shared by its branches; ``none``: skipped);
3. optionally ``BatchNorm1d`` (``branch_norm``) and ``branch_activation``
   (``relu`` or ``identity``).

The soma sums its branches with learned weights (no bias), leaky-integrates
the sum with its own per-neuron ``rho`` (``soma_tau``), then ``BatchNorm1d``
and ReLU. The readout is the time mean of the last layer, optional dropout,
and ``fc = Linear(N_last, num_classes)``.

Time constants are initialised deterministically, log-spaced between
``tau_min`` and ``tau_max`` frames with ``a = exp(-1 / tau)``: across the ``K``
branches of each neuron for ``hetero`` (across neurons when ``K == 1``),
across neurons for ``shared`` and for the soma. With ``branches: 1``,
``fan_in: full``, ``branch_activation: identity``, ``tau: none`` and
``branch_norm: false`` a layer is a plain leaky dense layer (point-neuron
control); its redundant soma weight is initialised to 1.

Parameter cost of a layer with ``C_in`` inputs (``f`` clamped to ``C_in``)::

    N*K*(f + 2)                       branch weights + biases, soma weights
    + {hetero: N*K, shared: N, none: 0}   branch rho
    + 2*N*K   if branch_norm
    + N       if soma_tau
    + 2*N                             soma BatchNorm

plus ``N_last * num_classes + num_classes`` for ``fc``. E.g. ``hetero`` with
``branch_norm`` and ``soma_tau`` costs ``N * (K * (f + 5) + 3)``.

Training computes each leaky integration exactly as a causal Toeplitz matmul
(``T x T`` kernel ``(1 - a) a^(t - s)``), the fastest correct option on MPS for
``T = 101`` (vs. a doubling scan, FFT or a length-``T`` depthwise conv); it is
equivalent to the per-frame recurrence used when streaming. The branch and soma
weights live in grouped ``Conv1d`` modules (``index_select`` + grouped conv is
the reference semantics) but are applied as dense matmuls with the weights
scattered into ``(N*K, C_in)`` and block-diagonal ``(N, N*K)`` matrices, which
is several times faster on MPS and numerically the same map.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from kws.models.ei_conv1d import InputNorm

TAU_MODES = ("hetero", "shared", "none")
LAYOUTS = ("local", "random")
ACTIVATIONS = ("relu", "identity")
INPUT_TRANSFORMS = ("none", "idct")


def leaky_integrate(u: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Per-channel causal leaky integration of ``u`` ``(B, C, T)``.

    ``v[t] = a v[t-1] + (1 - a) u[t]``, ``v[-1] = 0``, ``a = sigmoid(rho)``
    with ``rho`` of shape ``(C,)``; differentiable in ``u`` and ``rho``.
    Computed as ``v[b, c, t] = sum_s L[c, t, s] u[b, c, s]`` with the lower
    triangular ``L[c, t, s] = (1 - a_c) a_c^(t - s)``. ``log a`` comes from
    ``logsigmoid`` and the lag is clamped before masking, so neither extreme
    ``rho`` nor the masked upper triangle can produce inf or NaN gradients.
    """
    steps = u.shape[-1]
    t = torch.arange(steps, device=u.device)
    lag = t[:, None] - t[None, :]
    causal = (lag >= 0).to(u.dtype)
    log_a = F.logsigmoid(rho).to(u.dtype)
    gain = torch.sigmoid(-rho).to(u.dtype)  # 1 - a
    kernel = torch.exp(log_a[:, None, None] * lag.clamp(min=0).to(u.dtype))
    kernel = kernel * causal * gain[:, None, None]
    return torch.einsum("bcs,cts->bct", u, kernel)


def log_spaced_taus(count: int, tau_min: float, tau_max: float) -> torch.Tensor:
    """``count`` time constants log-spaced over ``[tau_min, tau_max]`` (float64).

    A single time constant is the geometric mean of the range.
    """
    if count == 1:
        return torch.tensor([math.sqrt(tau_min * tau_max)], dtype=torch.float64)
    return torch.logspace(math.log10(tau_min), math.log10(tau_max), count, dtype=torch.float64)


def tau_to_rho(tau: torch.Tensor) -> torch.Tensor:
    """``rho`` with ``sigmoid(rho) = exp(-1 / tau)``."""
    return torch.logit(torch.exp(-1.0 / tau)).float()


def rho_to_tau(rho: torch.Tensor) -> torch.Tensor:
    """Time constant in frames, ``-1 / log(sigmoid(rho))``."""
    return -1.0 / F.logsigmoid(rho)


class IDCT(nn.Module):
    """Parameter-free inverse of torchaudio's orthonormal MFCC DCT.

    torchaudio computes ``mfcc = (mel^T @ D)^T`` with
    ``D = create_dct(n_mfcc, n_mels, "ortho")``; ``D`` is orthogonal when
    ``n_mfcc == n_mels``, so ``mel^T = mfcc^T @ D^T``. torchaudio builds ``D``
    in float32, where it is orthogonal only to ~1e-6, so the buffer holds the
    float64 inverse of that exact matrix (equal to ``D^T`` within 2e-6), which
    cuts the float32 reconstruction error from ~6e-5 to ~4e-6.
    """

    inverse: torch.Tensor

    def __init__(self, n_mfcc: int):
        super().__init__()
        dct = torchaudio.functional.create_dct(n_mfcc, n_mfcc, "ortho")  # (n_mels, n_mfcc)
        inverse = torch.linalg.inv(dct.double()).float()  # ~ dct.T
        self.register_buffer("inverse", inverse, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x.transpose(-1, -2), self.inverse.to(x.dtype)).transpose(-1, -2)


def branch_index(in_channels: int, neurons: int, branches: int, fan_in: int,
                 layout: str) -> torch.Tensor:
    """Flat input index, ``fan_in`` channels for each of ``neurons * branches``."""
    if layout == "random":
        g = torch.Generator().manual_seed(0)
        return torch.cat([torch.randperm(in_channels, generator=g)[:fan_in]
                          for _ in range(neurons * branches)])
    units, room = neurons * branches, in_channels - fan_in
    # start(u) = round(u / (units - 1) * room), rounding half up, in exact integers.
    starts = [0 if units == 1 else (2 * u * room + units - 1) // (2 * (units - 1))
              for u in range(units)]
    return torch.tensor([start + i for start in starts for i in range(fan_in)],
                        dtype=torch.long)


class DTLayer(nn.Module):
    """``N`` neurons with ``K`` leaky-integrating branches each, on ``(B, C_in, T)``."""

    index: torch.Tensor | None

    def __init__(self, in_channels: int, neurons: int, branches: int = 4,
                 fan_in: int | str = 8, tau: str = "hetero", layout: str = "local",
                 branch_activation: str = "relu", branch_norm: bool = True,
                 soma_tau: bool = True, tau_min: float = 1.0, tau_max: float = 50.0):
        super().__init__()
        if tau not in TAU_MODES:
            raise ValueError(f"tau must be one of {TAU_MODES}, got {tau!r}")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if branch_activation not in ACTIVATIONS:
            raise ValueError(f"branch_activation must be one of {ACTIVATIONS}, "
                             f"got {branch_activation!r}")
        if neurons < 1 or branches < 1:
            raise ValueError("neurons and branches must be positive")
        if not 0 < tau_min <= tau_max:
            raise ValueError("need 0 < tau_min <= tau_max")
        if fan_in == "full":
            fan_in = in_channels
        if not isinstance(fan_in, int) or fan_in < 1:
            raise ValueError(f"fan_in must be a positive int or 'full', got {fan_in!r}")
        fan_in = min(fan_in, in_channels)
        units = neurons * branches
        self.in_channels, self.neurons, self.branches, self.fan_in = (
            in_channels, neurons, branches, fan_in)
        self.tau, self.branch_activation = tau, branch_activation

        if fan_in == in_channels:  # every branch reads every input: a dense 1x1 conv
            self.register_buffer("index", None, persistent=False)
            self.branch = nn.Conv1d(in_channels, units, 1, bias=True)
        else:
            self.register_buffer("index", branch_index(in_channels, neurons, branches,
                                                       fan_in, layout), persistent=False)
            self.branch = nn.Conv1d(units * fan_in, units, 1, groups=units, bias=True)

        self.branch_rho: nn.Parameter | None = None
        if tau == "hetero":
            per_neuron = branches > 1
            taus = log_spaced_taus(branches if per_neuron else neurons, tau_min, tau_max)
            taus = taus.repeat(neurons) if per_neuron else taus
            self.branch_rho = nn.Parameter(tau_to_rho(taus))
        elif tau == "shared":
            self.branch_rho = nn.Parameter(tau_to_rho(log_spaced_taus(neurons, tau_min, tau_max)))
        self.branch_bn = nn.BatchNorm1d(units) if branch_norm else None

        self.soma = nn.Conv1d(units, neurons, 1, groups=neurons, bias=False)
        if branches == 1:  # a lone branch's soma weight only rescales ahead of the BN
            nn.init.ones_(self.soma.weight)
        self.soma_rho = (nn.Parameter(tau_to_rho(log_spaced_taus(neurons, tau_min, tau_max)))
                         if soma_tau else None)
        self.bn = nn.BatchNorm1d(neurons)

    def branch_taus(self) -> torch.Tensor | None:
        """Per-branch time constants in frames, ``(N * K,)``, or None."""
        rho = self.expanded_branch_rho()
        return None if rho is None else rho_to_tau(rho)

    def expanded_branch_rho(self) -> torch.Tensor | None:
        if self.branch_rho is None:
            return None
        if self.tau == "shared":
            return self.branch_rho.repeat_interleave(self.branches)
        return self.branch_rho

    def dense_branch_weight(self) -> torch.Tensor:
        """Branch weights scattered into a dense ``(N * K, C_in)`` matrix."""
        units = self.neurons * self.branches
        weight = self.branch.weight.view(units, -1)
        if self.index is None:
            return weight
        dense = weight.new_zeros(units, self.in_channels)
        return dense.scatter_add(1, self.index.view(units, self.fan_in), weight)

    def dense_soma_weight(self) -> torch.Tensor:
        """Soma weights as a block-diagonal ``(N, N * K)`` matrix."""
        n, k = self.neurons, self.branches
        cols = torch.arange(n * k, device=self.soma.weight.device).view(n, k)
        return self.soma.weight.new_zeros(n, n * k).scatter(1, cols, self.soma.weight.view(n, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equal to ``self.branch(x.index_select(1, self.index))`` and ``self.soma(u)``
        # (the grouped convs hold the weights), but dense matmuls are ~5x faster on MPS.
        u = torch.matmul(self.dense_branch_weight(), x) + self.branch.bias[:, None]
        rho = self.expanded_branch_rho()
        if rho is not None:
            u = leaky_integrate(u, rho)
        if self.branch_bn is not None:
            u = self.branch_bn(u)
        if self.branch_activation == "relu":
            u = torch.relu(u)
        s = torch.matmul(self.dense_soma_weight(), u)
        if self.soma_rho is not None:
            s = leaky_integrate(s, self.soma_rho)
        return torch.relu(self.bn(s))


def per_layer(value, layers: int, name: str) -> list:
    """Broadcast a scalar config value to every layer, or check a list's length."""
    if isinstance(value, (list, tuple)):
        if len(value) != layers:
            raise ValueError(f"{name} has {len(value)} entries for {layers} layers")
        return list(value)
    return [value] * layers


class DTNet(nn.Module):
    def __init__(self, n_mfcc: int, time_steps: int, num_classes: int,
                 neurons: list[int] | tuple[int, ...] = (34, 34),
                 branches: int | list[int] = 4, fan_in: int | str | list = 8,
                 tau: str = "hetero", tau_min: float = 1.0, tau_max: float = 50.0,
                 layout: str = "local", branch_activation: str = "relu",
                 branch_norm: bool = True, soma_tau: bool = True,
                 input_transform: str = "none", input_norm: str = "cmvn",
                 dropout: float = 0.0):
        super().__init__()
        if input_transform not in INPUT_TRANSFORMS:
            raise ValueError(f"input_transform must be one of {INPUT_TRANSFORMS}, "
                             f"got {input_transform!r}")
        if not neurons:
            raise ValueError("neurons must list at least one layer")
        self.input_shape = (n_mfcc, time_steps)
        self.input_transform = IDCT(n_mfcc) if input_transform == "idct" else nn.Identity()
        self.input_norm = InputNorm(input_norm, n_mfcc)
        layers = []
        width = n_mfcc
        for n, k, f in zip(neurons, per_layer(branches, len(neurons), "branches"),
                           per_layer(fan_in, len(neurons), "fan_in")):
            layers.append(DTLayer(width, n, k, f, tau=tau, layout=layout,
                                  branch_activation=branch_activation, branch_norm=branch_norm,
                                  soma_tau=soma_tau, tau_min=tau_min, tau_max=tau_max))
            width = n
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width, num_classes)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Last layer's causal activations, ``(B, N_last, T)``."""
        if x.dim() == 4:
            x = x.squeeze(1)
        return self.layers(self.input_norm(self.input_transform(x)))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x).mean(-1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def build_dtnet(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> DTNet:
    return DTNet(
        n_mfcc=input_shape[0],
        time_steps=input_shape[1],
        num_classes=num_classes,
        neurons=list(model_cfg.get("neurons", [34, 34])),
        branches=model_cfg.get("branches", 4),
        fan_in=model_cfg.get("fan_in", 8),
        tau=model_cfg.get("tau", "hetero"),
        tau_min=float(model_cfg.get("tau_min", 1.0)),
        tau_max=float(model_cfg.get("tau_max", 50.0)),
        layout=model_cfg.get("layout", "local"),
        branch_activation=model_cfg.get("branch_activation", "relu"),
        branch_norm=model_cfg.get("branch_norm", True),
        soma_tau=model_cfg.get("soma_tau", True),
        input_transform=model_cfg.get("input_transform", "none"),
        input_norm=model_cfg.get("input_norm", "cmvn"),
        dropout=model_cfg.get("dropout", 0.0),
    )
