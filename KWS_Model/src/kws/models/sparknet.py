"""SparkNet: sparse-binarization keyword spotting (Svirsky, Shaham, and
Lindenbaum, "Sparse Binarization for Fast Keyword Spotting," Interspeech 2024).

The 1D time-channel separable convolutions are implemented as ``nn.Conv2d``
with ``(1, K)`` kernels on a ``(B, F, 1, T)`` view of the ``(B, 1, F, T)``
feature input, with frequency bins as channels. This keeps the model visible
to the project's Conv2d-only MAC/parameter profiling, clustering, and QAT
Conv+BN fusion (see PLAN.md Finding 7).
"""
import torch
import torch.nn as nn


class TCSBlock(nn.Module):
    """Time-channel separable conv block: depthwise -> pointwise -> BN -> (+ residual) -> ReLU.

    Each ``Conv2d`` is registered immediately before its own ``BatchNorm2d``,
    the order QAT fusion (``src/kws/optimize/quantize_qat.py``) discovers
    adjacent conv/BN pairs in.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 residual: bool, bn_eps: float = 1e-3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise: nn.Conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, kernel_size),
            padding=(0, padding), groups=in_channels, bias=False,
        )
        self.pointwise: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels, eps=bn_eps)
        self.res_conv: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if residual else None
        )
        self.res_bn: nn.BatchNorm2d | None = (
            nn.BatchNorm2d(out_channels, eps=bn_eps) if residual else None
        )
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(self.pointwise(self.depthwise(x)))
        if self.res_conv is not None:
            y = y + self.res_bn(self.res_conv(x))
        return self.relu(y)


class SparkNet(nn.Module):
    """Reference architecture, verified from the released checkpoints (PLAN.md Finding 3).

    Four ``TCSBlock``s (residual on the last three) feed a 1x1 gate conv with
    bias, then a BatchNorm at the framework default eps, then tanh. In
    training only, the gate is perturbed by ``N(0, 0.5**2)`` noise before
    being clamped to ``[0, 1]`` and averaged over time; the classifier reads
    that pooled gate occupancy.
    """

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
    ):
        super().__init__()
        if len(kernels) != 4:
            raise ValueError(f"SparkNet expects 4 kernel sizes, got {kernels!r}")

        blocks: list[TCSBlock] = []
        in_channels = n_feat
        for i, kernel_size in enumerate(kernels):
            blocks.append(TCSBlock(in_channels, channels, kernel_size, residual=(i > 0),
                                    bn_eps=block_bn_eps))
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
            gate = gate + torch.randn_like(gate) * 0.5
        z = torch.clamp(gate + 0.5, 0.0, 1.0)
        return z.mean(dim=(2, 3))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a pooled gate-occupancy representation."""
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def build_sparknet(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> SparkNet:
    return SparkNet(
        n_feat=input_shape[0],
        num_classes=num_classes,
        channels=model_cfg["channels"],
        gate_channels=model_cfg["gate_channels"],
    )
