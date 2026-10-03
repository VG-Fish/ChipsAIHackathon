from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn

from kws.models.sparknet import DendriticPointwise


class DSConvBlockSequence(nn.Sequential):
    """A sequential container whose items are known DS-CNN blocks.

    ``nn.Sequential`` exposes its children as the base ``nn.Module`` type in
    PyTorch's stubs.  The DS-CNN topology is fixed, so preserving the more
    specific type here makes channel surgery and callers of ``model.blocks``
    type-safe without changing the runtime container behavior.
    """

    def __init__(self, *blocks: "DSConvBlock") -> None:
        super().__init__(*blocks)

    def __iter__(self) -> Iterator["DSConvBlock"]:
        return (cast(DSConvBlock, block) for block in self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

class DSConvBlock(nn.Module):
    """Depthwise-separable conv block: depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 dendrites: tuple[int, int] | None = None):
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
        # Optional additive dendritic branch (``dendrites`` = (per-neuron
        # dendrites, fan_in)), summed into the pointwise output ahead of bn2,
        # as SparkNet's TCSBlock does with dendrite_mode="add". Registered
        # after bn2 so a None branch leaves the module order and state_dict
        # keys of the plain block untouched.
        self.dendrite_branch: DendriticPointwise | None = (
            DendriticPointwise(out_channels, *dendrites, in_channels=in_channels)
            if dendrites is not None else None
        )
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        y = self.pointwise(x)
        branch = getattr(self, "dendrite_branch", None)  # absent on pre-dendrite pickles
        if branch is not None:
            y = y + branch(x)
        return self.relu(self.bn2(y))
