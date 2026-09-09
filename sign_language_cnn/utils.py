"""Utility functions for model inspection, logging, and visualization."""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes (assuming float32 weights)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 * 1024)


def saved_model_size_mb(path: str) -> float:
    """Get the size of a saved model file in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def print_model_summary(model: nn.Module, title: str = "Model Summary"):
    """Print a summary of the model architecture and size."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    total_params = count_parameters(model)
    size_mb = model_size_mb(model)
    
    print(f" Total trainable parameters: {total_params:,}")
    print(f" Model size (float32):       {size_mb:.2f} MB")
    print(f" Model size (int8 est.):     {size_mb / 4:.2f} MB")
    print(f"{'='*60}")
    
    # Per-layer breakdown
    print(f"\n {'Layer':<40} {'Params':>12} {'Size (MB)':>10}")
    print(f" {'-'*40} {'-'*12} {'-'*10}")
    
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            layer_size = sum(
                p.numel() * p.element_size()
                for p in module.parameters(recurse=False)
            ) / (1024 * 1024)
            print(f" {name:<40} {params:>12,} {layer_size:>10.4f}")
    
    print()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy for the given predictions and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results
