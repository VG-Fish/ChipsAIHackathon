from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn


class DSConvBlockSequence(nn.Module):
    """A sequential container whose items are known DS-CNN blocks.

    ``nn.Sequential`` exposes its children as the base ``nn.Module`` type in
    PyTorch's stubs.  The DS-CNN topology is fixed, so preserving the more
    specific type here makes channel surgery and callers of ``model.blocks``
    type-safe without changing the runtime container behavior.
    """

    def __init__(self, *blocks: "DSConvBlock") -> None:
        super().__init__()
        for index, block in enumerate(blocks):
            self.add_module(str(index), block)

    def __getitem__(self, index: int) -> "DSConvBlock":
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("DSConvBlockSequence index out of range")
        return cast(DSConvBlock, self._modules[str(index)])

    def __iter__(self) -> Iterator["DSConvBlock"]:
        return (cast(DSConvBlock, block) for block in self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self:
            x = block(x)
        return x


class DSConvBlock(nn.Module):
    """Depthwise-separable conv block: depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise: nn.Conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(in_channels)
        self.pointwise: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x
