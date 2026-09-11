import torch
import torch.nn as nn
from typing import Protocol

from kws.models.layers import DSConvBlock, DSConvBlockSequence


class FeatureModel(Protocol):
    """Interface shared by DS-CNN and wrappers that expose pooled features."""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...

    def classify_features(self, features: torch.Tensor) -> torch.Tensor: ...


class DSCNN(nn.Module):
    """Depthwise-separable CNN for keyword spotting (Zhang et al., "Hello Edge").

    Topology: initial standard conv -> N depthwise-separable conv blocks ->
    fixed-size average pool -> FC -> softmax (softmax applied by the loss, not here).

    Uses a fixed-size (not adaptive) average pool, computed once at construction
    time from `input_shape`, so the graph has no dynamic shapes -- this keeps it
    friendly to both ONNX export and Perforated AI's dendrite-hook integration.
    """

    input_shape: tuple[int, int]
    stem: nn.Sequential
    blocks: DSConvBlockSequence
    pool: nn.AvgPool2d
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        initial_channels: int,
        initial_kernel: int,
        initial_stride: int,
        block_channels: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_shape = input_shape

        pad = initial_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, initial_channels, kernel_size=initial_kernel, stride=initial_stride,
                      padding=pad, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
        )

        blocks: list[DSConvBlock] = []
        in_ch = initial_channels
        for out_ch in block_channels:
            blocks.append(DSConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = DSConvBlockSequence(*blocks)

        pool_h, pool_w = self._compute_feature_map_size(input_shape)
        self.pool = nn.AvgPool2d(kernel_size=(pool_h, pool_w))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def _compute_feature_map_size(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.blocks(self.stem(dummy))
        return out.shape[2], out.shape[3]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled encoder representation before dropout/classification."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a pooled encoder representation."""
        return self.fc(self.dropout(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def build_ds_cnn(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> DSCNN:
    return DSCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        initial_channels=model_cfg["initial_channels"],
        initial_kernel=model_cfg["initial_kernel"],
        initial_stride=model_cfg["initial_stride"],
        block_channels=model_cfg["block_channels"],
        dropout=model_cfg.get("dropout", 0.2),
    )
