"""Dendritic CNN architecture for sign language classification.

The model uses a dendritic branching structure where each convolutional
layer is split into independent branches (inspired by biological dendritic
trees). This structure enables efficient structured pruning - entire
branches can be evaluated for importance and removed to compress the model.

Design targets:
- Input: 128x128 RGB images
- Output: 29 classes (ASL alphabet + space/delete/nothing)
- Uncompressed size: ~3-4 MB (float32)
- Compressed target: < 4 MB (after dendritic pruning + quantization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class DendriticConvBlock(nn.Module):
    """Convolutional block with dendritic branching structure.
    
    Splits the convolution into multiple independent branches (dendrites),
    each responsible for a subset of output channels. Each branch has a
    learnable importance gate that controls its contribution to the output.
    
    During compression, branches with low importance (small gate values
    and/or small weight magnitudes) can be pruned entirely, removing all
    associated parameters.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (must be divisible by num_branches).
        num_branches: Number of dendritic branches.
        kernel_size: Convolution kernel size.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_branches: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        if out_channels % num_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"num_branches ({num_branches})"
            )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = num_branches
        self.branch_channels = out_channels // num_branches
        
        # Each branch is an independent convolution
        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                self.branch_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False  # BN handles bias
            )
            for _ in range(num_branches)
        ])
        
        # Learnable importance gates for each branch (initialized to 1.0)
        # During training, gates learn to reflect branch importance
        self.branch_gates = nn.Parameter(torch.ones(num_branches))
        
        # Batch normalization over concatenated branch outputs
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all dendritic branches.
        
        Each branch processes the input independently, its output is
        scaled by the branch gate, and all branch outputs are concatenated.
        """
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            # Apply branch convolution and scale by gate
            out = branch(x) * self.branch_gates[i].view(1, 1, 1, 1)
            branch_outputs.append(out)
        
        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        return out
    
    def get_branch_importance(self) -> List[float]:
        """Compute importance score for each dendritic branch.
        
        Importance is based on:
        1. The absolute value of the branch gate
        2. The L1 norm of the branch's convolution weights
        
        Returns:
            List of importance scores, one per branch.
        """
        importances = []
        for i, branch in enumerate(self.branches):
            weight_norm = branch.weight.data.abs().mean().item()
            gate_value = self.branch_gates.data[i].abs().item()
            importances.append(weight_norm * gate_value)
        return importances
    
    def prune_branch(self, branch_idx: int):
        """Zero out a branch's gate (soft pruning).
        
        The branch's parameters remain but produce zero output.
        Use physically_prune_branches() on the full model for hard pruning.
        """
        with torch.no_grad():
            self.branch_gates.data[branch_idx] = 0.0
    
    def get_active_branches(self) -> List[int]:
        """Return indices of branches with non-zero gates."""
        return [
            i for i in range(self.num_branches)
            if self.branch_gates.data[i].abs().item() > 1e-8
        ]


class DendriticCNN(nn.Module):
    """CNN with dendritic branching for sign language classification.
    
    Architecture:
        Stage 1: 128x128 →  64x64  (3  → 32 channels,  4 branches × 8)
        Stage 2:  64x64  →  32x32  (32 → 64 channels,  4 branches × 16)
        Stage 3:  32x32  →  16x16  (64 → 128 channels, 4 branches × 32)
        Stage 4:  16x16  →   8x8   (128 → 256 channels, 4 branches × 64)
        Stage 5:   8x8   →   GAP   (256 → 256 channels, 4 branches × 64)
        Classifier: 256 → 128 → num_classes
    
    Total parameters: ~1M (≈ 3.9 MB float32)
    After dendritic pruning + quantization: < 4 MB target
    
    Args:
        num_classes: Number of output classes (default 29 for ASL).
        num_branches: Number of dendritic branches per conv block.
        input_size: Expected input spatial size (default 128).
    """
    
    def __init__(
        self,
        num_classes: int = 29,
        num_branches: int = 4,
        input_size: int = 128,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_branches = num_branches
        self.input_size = input_size
        
        # Stage 1: 128x128 → 64x64
        self.stage1 = DendriticConvBlock(3, 32, num_branches)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Stage 2: 64x64 → 32x32
        self.stage2 = DendriticConvBlock(32, 64, num_branches)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3: 32x32 → 16x16
        self.stage3 = DendriticConvBlock(64, 128, num_branches)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Stage 4: 16x16 → 8x8
        self.stage4 = DendriticConvBlock(128, 256, num_branches)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Stage 5: 8x8 → global average pooling
        self.stage5 = DendriticConvBlock(256, 256, num_branches)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 128, 128).
            
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.global_pool(self.stage5(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def get_dendritic_blocks(self) -> List[Tuple[str, DendriticConvBlock]]:
        """Return all DendriticConvBlock modules with their names."""
        blocks = []
        for name, module in self.named_modules():
            if isinstance(module, DendriticConvBlock):
                blocks.append((name, module))
        return blocks
    
    def get_all_branch_importances(self) -> Dict[str, List[float]]:
        """Get importance scores for all branches across all dendritic blocks.
        
        Returns:
            Dictionary mapping block name to list of branch importance scores.
        """
        importances = {}
        for name, block in self.get_dendritic_blocks():
            importances[name] = block.get_branch_importance()
        return importances
    
    def get_active_branch_count(self) -> Dict[str, int]:
        """Count active (non-pruned) branches in each dendritic block."""
        counts = {}
        for name, block in self.get_dendritic_blocks():
            counts[name] = len(block.get_active_branches())
        return counts
