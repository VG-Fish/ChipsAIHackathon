"""SparkNet: sparse-binarization keyword spotting (Svirsky, Shaham, and
Lindenbaum, "Sparse Binarization for Fast Keyword Spotting," Interspeech 2024).

The 1D time-channel separable convolutions are implemented as ``nn.Conv2d``
with ``(1, K)`` kernels on a ``(B, F, 1, T)`` view of the ``(B, 1, F, T)``
feature input, with frequency bins as channels. This keeps the model visible
to the project's Conv2d-only MAC/parameter profiling, clustering, and QAT
Conv+BN fusion (see PLAN.md Finding 7).
"""
from typing import cast

import torch
import torch.nn as nn

GATE_NOISE_STD = 0.5


class DendriticPointwise(nn.Module):
    """A 1x1 conv replaced by neurons with restricted-receptive-field dendrites.

    Output neuron ``n`` of ``channels`` has ``dendrites`` ReLU dendrites;
    dendrite ``j`` reads the ``fan_in`` contiguous input channels starting at
    ``n * in_channels // channels + j * fan_in`` (mod ``in_channels``), so each
    neuron sees one local window of ``dendrites * fan_in`` inputs, and the
    windows tile the inputs evenly (Chavlis & Poirazi, Nat. Commun. 2025).
    ``in_channels`` defaults to ``channels``. The soma sums its
    dendrites with learned weights and no bias, as the conv it replaces has
    none. Trained jointly by gradient descent. Costs ``C * dendrites * (fan_in + 2)``
    params against the conv's ``in_channels * channels``.
    """

    index: torch.Tensor

    def __init__(self, channels: int, dendrites: int, fan_in: int, in_channels: int | None = None):
        super().__init__()
        if dendrites < 1 or fan_in < 1:
            raise ValueError("dendrites and fan_in must be positive")
        self.channels, self.dendrites, self.fan_in = channels, dendrites, fan_in
        in_channels = channels if in_channels is None else in_channels
        units = channels * dendrites
        index = [(n * in_channels // channels + j * fan_in + i) % in_channels
                 for n in range(channels) for j in range(dendrites) for i in range(fan_in)]
        self.register_buffer("index", torch.tensor(index, dtype=torch.long), persistent=False)
        self.dendrite = nn.Conv2d(units * fan_in, units, kernel_size=1, groups=units, bias=True)
        self.soma = nn.Conv2d(units, channels, kernel_size=1, groups=channels, bias=False)
        # "identity" is an ablation: the branch is then linear in x, a
        # reparameterisation of the layer it is added to.
        self.activation = "relu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dendrite(x.index_select(1, self.index))
        return self.soma(h if self.activation == "identity" else torch.relu(h))


class MultiScaleDendriticPointwise(nn.Module):
    """Dendritic neurons whose branches read different temporal scales.

    The input is the scale-major concatenation of ``scales`` depthwise outputs
    of ``in_channels`` each, so channel ``c`` of scale ``s`` is input
    ``s * in_channels + c``. Output neuron ``n`` of ``channels`` has
    ``branches`` dendrites; branch ``j`` reads scale ``j % scales``, and within
    it the ``fan_in`` contiguous channels starting at
    ``n * in_channels // channels + (j // scales) * fan_in`` (mod
    ``in_channels``). A neuron therefore sees each of its ``branches // scales``
    local windows once at every scale, and the windows tile the channels as
    in ``DendriticPointwise``. Each branch is a biased linear unit followed by
    ``activation`` (``"relu"``; ``"identity"`` makes the layer a linear sparse
    multi-scale mix, the MixConv-style control); the soma sums its branches
    with learned weights and no bias. Costs ``channels * branches * (fan_in + 2)``
    params.
    """

    index: torch.Tensor

    def __init__(self, in_channels: int, channels: int, scales: int, branches: int,
                 fan_in: int, activation: str = "relu"):
        super().__init__()
        if scales < 1 or branches < 1 or fan_in < 1:
            raise ValueError("scales, branches and fan_in must be positive")
        if branches % scales:
            raise ValueError(f"branches ({branches}) must be a multiple of the number of "
                             f"scales ({scales})")
        if activation not in ("relu", "identity"):
            raise ValueError(f"activation must be 'relu' or 'identity', got {activation!r}")
        self.in_channels, self.channels = in_channels, channels
        self.scales, self.branches, self.fan_in = scales, branches, fan_in
        self.activation = activation
        units = channels * branches
        index = [(j % scales) * in_channels
                 + (n * in_channels // channels + (j // scales) * fan_in + i) % in_channels
                 for n in range(channels) for j in range(branches) for i in range(fan_in)]
        self.register_buffer("index", torch.tensor(index, dtype=torch.long), persistent=False)
        self.dendrite = nn.Conv2d(units * fan_in, units, kernel_size=1, groups=units, bias=True)
        self.soma = nn.Conv2d(units, channels, kernel_size=1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dendrite(x.index_select(1, self.index))
        return self.soma(h if self.activation == "identity" else torch.relu(h))


class MSDBlock(nn.Module):
    """Multi-scale dendritic TCS block.

    ``len(dilations)`` depthwise convs share one kernel size but differ in
    dilation, so their receptive fields are ``d * (k - 1) + 1`` frames; their
    outputs are concatenated scale-major and mixed by a pointwise stage, then
    BN -> (+ residual) -> ReLU exactly as in ``TCSBlock``. With
    ``pointwise="dendritic"`` the mix is a ``MultiScaleDendriticPointwise``
    whose branch ``j`` reads scale ``j % len(dilations)``; with ``"dense"`` it is
    a plain ``Conv2d(in_channels * len(dilations), out_channels, 1)``.

    Registration order keeps QAT fusion (``_fuse_conv_bn_for_qat``) correct:
    the depthwise convs sit in a ``ModuleList`` with no BatchNorm, a dense
    pointwise conv is registered right before ``bn`` (the pair fuses), a
    dendritic pointwise is not a ``Conv2d`` (no pair), and ``res_conv`` is
    registered right before ``res_bn``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, residual: bool,
                 dilations: tuple[int, ...] = (1, 2, 4), pointwise: str = "dendritic",
                 branches: int = 0, fan_in: int = 0, activation: str = "relu",
                 bn_eps: float = 1e-3):
        super().__init__()
        dilations = tuple(int(d) for d in dilations)
        if not dilations or min(dilations) < 1 or len(set(dilations)) != len(dilations):
            raise ValueError(f"dilations must be distinct positive integers, got {dilations!r}")
        if kernel_size % 2 == 0:
            raise ValueError(f"MSDBlock needs an odd kernel for 'same' padding, got {kernel_size}")
        if pointwise not in ("dendritic", "dense"):
            raise ValueError(f"pointwise must be 'dendritic' or 'dense', got {pointwise!r}")
        self.dilations = dilations
        scales = len(dilations)
        self.depthwise = nn.ModuleList(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),
                      padding=(0, d * (kernel_size // 2)), dilation=(1, d),
                      groups=in_channels, bias=False)
            for d in dilations
        )
        self.pointwise: nn.Module = (
            MultiScaleDendriticPointwise(in_channels, out_channels, scales, branches,
                                         fan_in, activation)
            if pointwise == "dendritic"
            else nn.Conv2d(in_channels * scales, out_channels, kernel_size=1, bias=False)
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels, eps=bn_eps)
        self.res_conv: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if residual else None
        )
        self.res_bn: nn.BatchNorm2d | None = (
            nn.BatchNorm2d(out_channels, eps=bn_eps) if residual else None
        )
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.depthwise) == 1:
            d = self.depthwise[0](x)
        else:
            d = torch.cat([conv(x) for conv in self.depthwise], dim=1)
        y = self.bn(self.pointwise(d))
        if self.res_conv is not None:
            y = y + cast(nn.BatchNorm2d, self.res_bn)(self.res_conv(x))
        return self.relu(y)


class TCSBlock(nn.Module):
    """Time-channel separable conv block: depthwise -> pointwise -> BN -> (+ residual) -> ReLU.

    Each ``Conv2d`` is registered immediately before its own ``BatchNorm2d``,
    the order QAT fusion (``src/kws/optimize/quantize_qat.py``) discovers
    adjacent conv/BN pairs in.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 residual: bool, bn_eps: float = 1e-3,
                 dendrites: tuple[int, int] | None = None, dendrite_mode: str = "replace"):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise: nn.Conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, kernel_size),
            padding=(0, padding), groups=in_channels, bias=False,
        )
        if dendrite_mode not in ("replace", "add"):
            raise ValueError(f"dendrite_mode must be 'replace' or 'add', got {dendrite_mode!r}")
        if dendrites is not None and dendrite_mode == "replace" and in_channels != out_channels:
            raise ValueError("a replacing dendritic pointwise needs in_channels == out_channels")
        # "replace" swaps the pointwise conv for dendrites; "add" keeps the conv
        # and sums a dendritic branch into its output, ahead of the BN.
        self.pointwise: nn.Module
        if dendrites is not None and dendrite_mode == "replace":
            self.pointwise = DendriticPointwise(in_channels, *dendrites)
        else:
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dendrite_branch: DendriticPointwise | None = (
            DendriticPointwise(out_channels, *dendrites, in_channels=in_channels)
            if dendrites is not None and dendrite_mode == "add" else None
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels, eps=bn_eps)
        self.res_conv: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if residual else None
        )
        self.res_bn: nn.BatchNorm2d | None = (
            nn.BatchNorm2d(out_channels, eps=bn_eps) if residual else None
        )
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.depthwise(x)
        y = self.pointwise(d)
        if self.dendrite_branch is not None:
            y = y + self.dendrite_branch(d)
        y = self.bn(y)
        if self.res_conv is not None:
            y = y + cast(nn.BatchNorm2d, self.res_bn)(self.res_conv(x))
        return self.relu(y)


class SparkNet(nn.Module):
    """Reference architecture, verified from the released checkpoints (PLAN.md Finding 3).

    Four ``TCSBlock``s (residual on the last three) feed a 1x1 gate conv with
    bias, then a BatchNorm at the framework default eps, then tanh. In
    training only, the gate is perturbed by ``N(0, 0.5**2)`` noise before
    being clamped to ``[0, 1]`` and averaged over time; the classifier reads
    that pooled gate occupancy.

    Training also returns the reference sparsity term through
    ``auxiliary_losses``: the mean probability that a noisy gate is open,
    ``mean(Phi((tanh_out + 0.5) / 0.5))``, weighted by ``sparsity_weight``.
    The reference trains on ``100 * CE + 1 * term``; AdamW is invariant to the
    overall loss scale, so the default 0.01 keeps that ratio on a CE of weight 1
    (PLAN.md Finding 4).

    ``block_type="msd"`` swaps the blocks listed in ``msd_blocks`` for
    ``MSDBlock``s (multi-scale depthwise + multi-scale pointwise); the default
    ``"tcs"`` builds the reference architecture unchanged.
    """

    input_shape: tuple[int, int] | None
    blocks: nn.ModuleList
    gate_conv: nn.Conv2d
    gate_bn: nn.BatchNorm2d
    fc: nn.Linear

    def __init__(
        self,
        n_feat: int,
        num_classes: int,
        channels: int = 16,
        gate_channels: int = 32,
        kernels: tuple[int, ...] = (11, 15, 19, 29),
        block_bn_eps: float = 1e-3,
        sparsity_weight: float = 0.01,
        input_shape: tuple[int, int] | None = None,
        dendrites: int = 0,
        dendrite_fan_in: int = 0,
        dendrite_mode: str = "replace",
        dendrite_blocks: tuple[int, ...] = (1, 2, 3),
        block_type: str = "tcs",
        msd_dilations: tuple[int, ...] = (1, 2, 4),
        msd_branches: int = 0,
        msd_fan_in: int = 0,
        msd_activation: str = "relu",
        msd_pointwise: str = "dendritic",
        msd_blocks: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        if len(kernels) != 4:
            raise ValueError(f"SparkNet expects 4 kernel sizes, got {kernels!r}")
        if block_type not in ("tcs", "msd"):
            raise ValueError(f"block_type must be 'tcs' or 'msd', got {block_type!r}")
        msd_block_set = set(msd_blocks) if block_type == "msd" else set()
        if not msd_block_set <= set(range(len(kernels))):
            raise ValueError(f"msd_blocks must index blocks 0-3, got {msd_blocks!r}")
        if input_shape is not None and input_shape[0] != n_feat:
            raise ValueError(f"input_shape {input_shape!r} does not have {n_feat} feature bins")
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be non-negative")
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.sparsity_weight = float(sparsity_weight)
        # Pre-noise tanh output of the latest training forward, consumed by
        # ``auxiliary_losses``. A plain attribute, so it never enters state_dict.
        self._pending_gate: torch.Tensor | None = None

        blocks: list[nn.Module] = []
        in_channels = n_feat
        for i, kernel_size in enumerate(kernels):
            # blocks.0 maps the feature bins to C, so only an additive branch fits there.
            if dendrites and 0 in dendrite_blocks and dendrite_mode != "add":
                raise ValueError("dendrites on blocks.0 need dendrite_mode 'add'")
            block_dendrites = ((dendrites, dendrite_fan_in)
                               if dendrites and i in dendrite_blocks else None)
            if i in msd_block_set:
                if block_dendrites is not None:
                    raise ValueError(f"blocks.{i} cannot be both an MSD block and carry "
                                     "DendriticPointwise dendrites")
                blocks.append(MSDBlock(in_channels, channels, kernel_size, residual=(i > 0),
                                       dilations=tuple(msd_dilations), pointwise=msd_pointwise,
                                       branches=msd_branches, fan_in=msd_fan_in,
                                       activation=msd_activation, bn_eps=block_bn_eps))
            else:
                blocks.append(TCSBlock(in_channels, channels, kernel_size, residual=(i > 0),
                                        bn_eps=block_bn_eps, dendrites=block_dendrites,
                                        dendrite_mode=dendrite_mode))
            in_channels = channels
        self.blocks = nn.ModuleList(blocks)

        self.gate_conv = nn.Conv2d(channels, gate_channels, kernel_size=1, bias=True)
        self.gate_bn = nn.BatchNorm2d(gate_channels)  # default eps 1e-5, matches the reference
        self.fc = nn.Linear(gate_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean-pooled stochastic gate occupancy, shape (B, gate_channels)."""
        x = x.permute(0, 2, 1, 3)  # (B, 1, F, T) -> (B, F, 1, T): frequency bins as channels
        for block in self.blocks:
            x = block(x)
        gate = torch.tanh(self.gate_bn(self.gate_conv(x)))
        if self.training:
            self._pending_gate = gate
            gate = gate + torch.randn_like(gate) * GATE_NOISE_STD
        else:
            self._pending_gate = None
        z = torch.clamp(gate + 0.5, 0.0, 1.0)
        return z.mean(dim=(2, 3))

    def auxiliary_losses(self) -> dict[str, tuple[torch.Tensor, float]]:
        """Pop the sparsity term of the latest training forward as ``(value, weight)``.

        The value is the mean probability that a gate is open under the
        training noise. Returns an empty mapping after an eval forward or when
        nothing is pending, so a caller that polls every step never reuses a
        stale graph.
        """
        pending_gate, self._pending_gate = self._pending_gate, None
        if pending_gate is None:
            return {}
        open_probability = torch.special.ndtr((pending_gate + 0.5) / GATE_NOISE_STD).mean()
        return {"gate_sparsity": (open_probability, self.sparsity_weight)}

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a pooled gate-occupancy representation."""
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


MSD_CONFIG_KEYS = ("msd_dilations", "msd_branches", "msd_fan_in", "msd_activation",
                   "msd_pointwise", "msd_blocks")


def build_sparknet(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> SparkNet:
    block_type = model_cfg.get("block_type", "tcs")
    stray = [key for key in MSD_CONFIG_KEYS if key in model_cfg]
    if block_type != "msd" and stray:
        # Catch a config that sets MSD options but would silently train plain TCS blocks.
        raise ValueError(f"{stray} need block_type: msd (got {block_type!r})")
    return SparkNet(
        n_feat=input_shape[0],
        num_classes=num_classes,
        channels=model_cfg["channels"],
        gate_channels=model_cfg["gate_channels"],
        sparsity_weight=model_cfg.get("sparsity_weight", 0.01),
        input_shape=input_shape,
        dendrites=model_cfg.get("dendrites", 0),
        dendrite_fan_in=model_cfg.get("dendrite_fan_in", 0),
        dendrite_mode=model_cfg.get("dendrite_mode", "replace"),
        dendrite_blocks=tuple(model_cfg.get("dendrite_blocks", (1, 2, 3))),
        block_type=block_type,
        msd_dilations=tuple(model_cfg.get("msd_dilations", (1, 2, 4))),
        msd_branches=model_cfg.get("msd_branches", 0),
        msd_fan_in=model_cfg.get("msd_fan_in", 0),
        msd_activation=model_cfg.get("msd_activation", "relu"),
        msd_pointwise=model_cfg.get("msd_pointwise", "dendritic"),
        msd_blocks=tuple(model_cfg.get("msd_blocks", (0, 1, 2, 3))),
    )
