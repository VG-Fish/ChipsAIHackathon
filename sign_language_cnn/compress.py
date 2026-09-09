"""Dendritic compression for the Sign Language CNN.

Implements structured pruning inspired by biological dendritic trees.
Each convolutional block has multiple 'dendritic branches' (groups of
filters). Compression works by:

1. Scoring branch importance (gate values × weight magnitudes)
2. Ranking branches globally across all layers
3. Pruning least important branches (soft pruning via gate zeroing)
4. Physically removing pruned branches (hard pruning via architecture rebuild)
5. Optional INT8 quantization for further size reduction
6. Fine-tuning to recover accuracy

Target: Compress model to fit within 4 MB of RAM.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


class DendriticPruner:
    """Handles dendritic branch pruning for model compression.
    
    Implements a global pruning strategy where branches across all
    layers are ranked by importance and the least important ones
    are removed.
    
    Args:
        model: The DendriticCNN model to prune.
        target_size_mb: Target model size in megabytes.
    """
    
    def __init__(self, model: nn.Module, target_size_mb: float = 4.0):
        self.model = model
        self.target_size_mb = target_size_mb
    
    def compute_model_size_mb(self) -> float:
        """Compute current model size in MB."""
        total_bytes = 0
        for p in self.model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in self.model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 * 1024)
    
    def get_global_branch_ranking(self) -> List[Tuple[str, int, float]]:
        """Rank all branches across all layers by importance.
        
        Returns:
            List of (block_name, branch_index, importance_score),
            sorted by importance ascending (least important first).
        """
        all_branches = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_branch_importance'):
                importances = module.get_branch_importance()
                active = module.get_active_branches()
                for i, importance in enumerate(importances):
                    if i in active:  # Only rank active branches
                        all_branches.append((name, i, importance))
        
        # Sort by importance (ascending = least important first)
        all_branches.sort(key=lambda x: x[2])
        return all_branches
    
    def soft_prune(
        self,
        prune_ratio: float = 0.25,
        min_branches_per_block: int = 1,
    ) -> Dict[str, List[int]]:
        """Soft prune branches by zeroing their gates.
        
        Does not remove parameters, but effectively disables branches.
        
        Args:
            prune_ratio: Fraction of total branches to prune.
            min_branches_per_block: Minimum branches to keep per block.
        
        Returns:
            Dictionary mapping block name to list of pruned branch indices.
        """
        ranking = self.get_global_branch_ranking()
        total_active = len(ranking)
        num_to_prune = int(total_active * prune_ratio)
        
        pruned = {}
        prune_count = 0
        
        for block_name, branch_idx, importance in ranking:
            if prune_count >= num_to_prune:
                break
            
            # Check minimum branches constraint
            block = dict(self.model.named_modules())[block_name]
            active = block.get_active_branches()
            
            if len(active) <= min_branches_per_block:
                continue  # Can't prune this block further
            
            # Prune this branch
            block.prune_branch(branch_idx)
            
            if block_name not in pruned:
                pruned[block_name] = []
            pruned[block_name].append(branch_idx)
            prune_count += 1
        
        return pruned
    
    def hard_prune(self) -> nn.Module:
        """Physically remove pruned branches and rebuild the model.
        
        Creates a new model with pruned branches removed entirely,
        reducing the actual parameter count and model size.
        
        Returns:
            New model with pruned branches physically removed.
        """
        from .model import DendriticConvBlock, DendriticCNN
        
        model = copy.deepcopy(self.model)
        
        for name, module in model.named_modules():
            if isinstance(module, DendriticConvBlock):
                active_indices = module.get_active_branches()
                
                if len(active_indices) == module.num_branches:
                    continue  # No branches pruned
                
                if len(active_indices) == 0:
                    print(f"WARNING: All branches pruned in {name}, keeping one")
                    active_indices = [0]
                
                # Rebuild with only active branches
                new_branches = nn.ModuleList([
                    module.branches[i] for i in active_indices
                ])
                new_gates = nn.Parameter(
                    module.branch_gates.data[active_indices].clone()
                )
                
                # Update batch norm to match new channel count
                new_out_channels = len(active_indices) * module.branch_channels
                new_bn = nn.BatchNorm2d(new_out_channels)
                
                # Copy matching BN weights for active channels
                channel_indices = []
                for idx in active_indices:
                    start = idx * module.branch_channels
                    end = start + module.branch_channels
                    channel_indices.extend(range(start, end))
                
                with torch.no_grad():
                    new_bn.weight.data = module.bn.weight.data[channel_indices].clone()
                    new_bn.bias.data = module.bn.bias.data[channel_indices].clone()
                    new_bn.running_mean.data = module.bn.running_mean.data[channel_indices].clone()
                    new_bn.running_var.data = module.bn.running_var.data[channel_indices].clone()
                
                # Replace module components
                module.branches = new_branches
                module.branch_gates = new_gates
                module.bn = new_bn
                module.num_branches = len(active_indices)
                module.out_channels = new_out_channels
        
        # Fix inter-layer channel mismatches after pruning
        self._fix_channel_mismatches(model)
        
        return model
    
    def _fix_channel_mismatches(self, model: nn.Module):
        """Fix channel dimension mismatches between consecutive layers after pruning.
        
        When a layer's output channels are reduced by pruning, the next layer's
        input channels must be adjusted accordingly.
        """
        from .model import DendriticConvBlock
        
        # Get ordered list of dendritic blocks
        blocks = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticConvBlock):
                blocks.append((name, module))
        
        # Fix each consecutive pair
        for i in range(len(blocks) - 1):
            current_name, current_block = blocks[i]
            next_name, next_block = blocks[i + 1]
            
            current_out = current_block.out_channels
            
            # Check if next block's input channels match
            if next_block.branches[0].in_channels != current_out:
                # Rebuild next block's branches with correct input channels
                for branch in next_block.branches:
                    old_weight = branch.weight.data
                    new_in_channels = current_out
                    
                    if new_in_channels != old_weight.shape[1]:
                        # Adjust weights by truncating or zero-padding input channels
                        new_weight = torch.zeros(
                            old_weight.shape[0],
                            new_in_channels,
                            old_weight.shape[2],
                            old_weight.shape[3],
                            device=old_weight.device,
                            dtype=old_weight.dtype,
                        )
                        min_channels = min(new_in_channels, old_weight.shape[1])
                        new_weight[:, :min_channels] = old_weight[:, :min_channels]
                        branch.weight = nn.Parameter(new_weight)
                        branch.in_channels = new_in_channels
        
        # Fix classifier input if last dendritic block was pruned
        last_block_name, last_block = blocks[-1]
        last_out = last_block.out_channels
        
        if hasattr(model, 'classifier'):
            first_linear = None
            for module in model.classifier:
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            
            if first_linear is not None and first_linear.in_features != last_out:
                old_weight = first_linear.weight.data
                old_bias = first_linear.bias.data
                
                new_linear = nn.Linear(last_out, first_linear.out_features)
                with torch.no_grad():
                    min_features = min(last_out, old_weight.shape[1])
                    new_linear.weight.data[:, :min_features] = old_weight[:, :min_features]
                    new_linear.bias.data = old_bias.clone()
                
                # Replace in classifier
                for i, module in enumerate(model.classifier):
                    if module is first_linear:
                        model.classifier[i] = new_linear
                        break
    
    def iterative_prune_to_target(
        self,
        prune_step: float = 0.1,
        min_branches_per_block: int = 1,
    ) -> nn.Module:
        """Iteratively prune until target size is reached.
        
        Args:
            prune_step: Fraction of remaining branches to prune each iteration.
            min_branches_per_block: Minimum branches to keep per block.
        
        Returns:
            Hard-pruned model that fits within target size.
        """
        print(f"\nStarting iterative dendritic pruning...")
        print(f"Target size: {self.target_size_mb:.2f} MB")
        print(f"Current size: {self.compute_model_size_mb():.2f} MB")
        
        iteration = 0
        while self.compute_model_size_mb() > self.target_size_mb:
            iteration += 1
            
            # Soft prune a fraction of branches
            pruned = self.soft_prune(
                prune_ratio=prune_step,
                min_branches_per_block=min_branches_per_block,
            )
            
            if not pruned:
                print(f"No more branches can be pruned (min constraint).")
                break
            
            # Report
            total_pruned = sum(len(v) for v in pruned.values())
            print(
                f"  Iteration {iteration}: Pruned {total_pruned} branches | "
                f"Size: {self.compute_model_size_mb():.2f} MB"
            )
        
        # Physically remove pruned branches
        pruned_model = self.hard_prune()
        
        # Report final size
        final_size = sum(
            p.numel() * p.element_size()
            for p in pruned_model.parameters()
        ) + sum(
            b.numel() * b.element_size()
            for b in pruned_model.buffers()
        )
        print(f"\nAfter hard pruning: {final_size / (1024*1024):.2f} MB")
        
        return pruned_model


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to the model for saving.
    
    Dynamic quantization quantizes weights to INT8 and computes
    activations in INT8 at runtime, reducing model size by ~4x.
    
    NOTE: PyTorch dynamic quantization requires CPU. This is only
    called at final export time AFTER all GPU training is complete.
    The model is moved to CPU solely for the quantization export step.
    
    Args:
        model: The model to quantize (will be copied to CPU).
    
    Returns:
        Quantized model (on CPU, for saving/inference).
    """
    # Move a copy to CPU for quantization export (PyTorch requirement)
    model_cpu = copy.deepcopy(model).cpu().eval()
    
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize linear layers
        dtype=torch.qint8,
    )
    
    return quantized


def save_compressed_model(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict] = None,
):
    """Save a compressed model with metadata.
    
    Args:
        model: The (potentially pruned/quantized) model.
        save_path: Path to save the model.
        metadata: Optional metadata to include.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
    }
    
    torch.save(save_dict, save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Model saved to {save_path} ({file_size_mb:.2f} MB)")
    return file_size_mb


def compress_pipeline(
    model: nn.Module,
    target_size_mb: float = 4.0,
    save_dir: str = 'checkpoints',
    quantize: bool = True,
) -> Tuple[nn.Module, float]:
    """Full compression pipeline: dendritic pruning + optional quantization.
    
    Args:
        model: Trained DendriticCNN model.
        target_size_mb: Target size in MB.
        save_dir: Directory to save compressed model.
        quantize: Whether to apply INT8 quantization after pruning.
    
    Returns:
        Tuple of (compressed_model, final_size_mb).
    """
    print("\n" + "="*60)
    print(" DENDRITIC COMPRESSION PIPELINE")
    print("="*60)
    
    # Step 1: Dendritic pruning
    pruner = DendriticPruner(model, target_size_mb=target_size_mb)
    pruned_model = pruner.iterative_prune_to_target()
    
    # Step 2: Optional quantization
    final_model = pruned_model
    if quantize:
        print("\nApplying INT8 dynamic quantization...")
        final_model = quantize_model_dynamic(pruned_model)
    
    # Step 3: Save
    save_path = os.path.join(save_dir, 'model_compressed.pth')
    final_size = save_compressed_model(
        final_model,
        save_path,
        metadata={
            'target_size_mb': target_size_mb,
            'quantized': quantize,
            'compression': 'dendritic_pruning',
        },
    )
    
    # Report compression ratio
    original_size = sum(
        p.numel() * 4 for p in model.parameters()  # Assuming float32
    ) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f" Compression Results")
    print(f"{'='*60}")
    print(f" Original size (float32): {original_size:.2f} MB")
    print(f" Compressed size (saved): {final_size:.2f} MB")
    print(f" Compression ratio:       {original_size / max(final_size, 0.01):.1f}x")
    print(f" Target met:              {'YES' if final_size <= target_size_mb else 'NO'}")
    print(f"{'='*60}")
    
    return final_model, final_size
