"""Training script with progressive resolution curriculum.

Implements the multi-round training strategy:
1. Start from a high effective resolution (default 1024)
2. Each round, halve the intermediate resolution
3. Stop when next halving would reach model input size (128)
4. Apply dendritic compression after training

Training Schedule Example (start=1024, target=128):
    Round 1: Images resized 1024 → 128 (sharpest quality)
    Round 2: Images resized 512 → 128 (slight quality loss)
    Round 3: Images resized 256 → 128 (more quality loss)
    Stop:    256/2 = 128 = model input size

This curriculum teaches the model to handle varying image qualities,
making it robust to low-resolution inputs.
"""

import os
import sys
import time
import argparse
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from .config import Config
from .model import DendriticCNN
from .dataset import download_dataset, create_dataloaders
from .compress import compress_pipeline
from .utils import (
    count_parameters,
    model_size_mb,
    print_model_summary,
    AverageMeter,
    accuracy,
)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
) -> dict:
    """Train the model for one epoch using mixed-precision on GPU.
    
    Returns:
        Dictionary with 'loss' and 'acc' metrics.
    """
    model.train()
    losses = AverageMeter('Loss')
    accs = AverageMeter('Acc')
    
    total_batches = len(train_loader)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed-precision forward pass (FP16 on GPU)
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Scaled backward pass for FP16 stability
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first for correct norm)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        acc = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))
        
        # Log progress
        if (batch_idx + 1) % max(1, total_batches // 5) == 0:
            print(
                f"    Batch [{batch_idx+1}/{total_batches}] "
                f"Loss: {losses.avg:.4f} | Acc: {accs.avg:.2f}%"
            )
    
    return {'loss': losses.avg, 'acc': accs.avg}


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model using mixed-precision on GPU.
    
    Returns:
        Dictionary with 'loss' and 'acc' metrics.
    """
    model.eval()
    losses = AverageMeter('Loss')
    accs = AverageMeter('Acc')
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
    
    return {'loss': losses.avg, 'acc': accs.avg}


def train_progressive(
    config: Config,
    resume_from: str = None,
):
    """Run the full progressive resolution training pipeline.
    
    Args:
        config: Training configuration.
        resume_from: Path to checkpoint to resume from.
    """
    device = config.get_device()
    config.ensure_dirs()
    
    # Enable GPU optimizations
    cudnn.benchmark = True  # Autotune convolution algorithms for fixed input sizes
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(" SIGN LANGUAGE CNN - PROGRESSIVE RESOLUTION TRAINING (GPU)")
    print("="*60)
    print(f" Device:         {device} ({torch.cuda.get_device_name(0)})")
    print(f" Mixed Precision: FP16 (torch.amp)")
    print(f" cuDNN Bench:    Enabled")
    print(f" Model input:    {config.model_input_size}x{config.model_input_size}")
    print(f" Num classes:    {config.num_classes}")
    print(f" Branches/block: {config.num_branches}")
    print(f" Batch size:     {config.batch_size}")
    print(f" Epochs/round:   {config.epochs_per_round}")
    
    # Build resolution schedule
    schedule = config.build_resolution_schedule()
    print(f"\n Resolution Schedule:")
    for i, res in enumerate(schedule):
        print(f"   Round {i+1}: {res}x{res} → {config.model_input_size}x{config.model_input_size}")
    print(f"   Stop: {schedule[-1]}//2 = {schedule[-1]//2} = model input size")
    
    # Step 1: Download dataset
    print("\n" + "-"*60)
    print(" Step 1: Dataset")
    print("-"*60)
    download_dataset(config.data_dir, config.dataset_name)
    
    # Step 2: Create model
    print("\n" + "-"*60)
    print(" Step 2: Model Architecture")
    print("-"*60)
    model = DendriticCNN(
        num_classes=config.num_classes,
        num_branches=config.num_branches,
        input_size=config.model_input_size,
    ).to(device)
    
    print_model_summary(model)
    
    # Resume from checkpoint if provided
    start_round = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_round = checkpoint.get('round', 0)
    
    # Step 3: Progressive resolution training
    print("\n" + "-"*60)
    print(" Step 3: Progressive Resolution Training")
    print("-"*60)
    
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_acc = 0.0
    training_start = time.time()
    
    # Mixed-precision gradient scaler for GPU FP16 training
    scaler = GradScaler('cuda')
    
    for round_idx, resolution in enumerate(schedule):
        if round_idx < start_round:
            continue
        
        round_start = time.time()
        
        print(f"\n{'*'*60}")
        print(f" ROUND {round_idx + 1}/{len(schedule)}: Resolution {resolution}x{resolution}")
        print(f"{'*'*60}")
        
        # Create dataloaders for current resolution
        print(f"\n  Creating dataloaders (intermediate res: {resolution})...")
        train_loader, val_loader = create_dataloaders(
            data_dir=config.data_dir,
            intermediate_resolution=resolution,
            model_input_size=config.model_input_size,
            batch_size=config.batch_size,
            val_split=0.15,
            num_workers=4,
        )
        
        # Optimizer and scheduler for this round
        # Use lower learning rate for later rounds (fine-tuning)
        round_lr = config.learning_rate * (0.5 ** round_idx)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=round_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs_per_round,
            eta_min=round_lr * 0.01,
        )
        
        print(f"  Learning rate: {round_lr:.6f}")
        
        # Train for this round
        for epoch in range(config.epochs_per_round):
            print(f"\n  Epoch [{epoch+1}/{config.epochs_per_round}]")
            
            # Train (mixed-precision FP16 on GPU)
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler,
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Step scheduler
            scheduler.step()
            
            print(
                f"  => Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.2f}%"
            )
            
            # Save best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                save_path = os.path.join(config.model_dir, 'model_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'round': round_idx,
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'resolution': resolution,
                    'config': {
                        'num_classes': config.num_classes,
                        'num_branches': config.num_branches,
                        'input_size': config.model_input_size,
                    },
                }, save_path)
                print(f"  => New best model saved (val_acc: {best_val_acc:.2f}%)")
        
        # Save round checkpoint
        round_save = os.path.join(
            config.model_dir, f'model_round{round_idx+1}_res{resolution}.pth'
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'round': round_idx + 1,
            'resolution': resolution,
            'val_acc': val_metrics['acc'],
            'config': {
                'num_classes': config.num_classes,
                'num_branches': config.num_branches,
                'input_size': config.model_input_size,
            },
        }, round_save)
        
        round_time = time.time() - round_start
        print(f"\n  Round {round_idx+1} completed in {timedelta(seconds=int(round_time))}")
    
    # Training complete
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f" Total time:     {timedelta(seconds=int(total_time))}")
    print(f" Best val acc:   {best_val_acc:.2f}%")
    print(f" Model size:     {model_size_mb(model):.2f} MB")
    
    # Step 4: Dendritic compression
    print("\n" + "-"*60)
    print(" Step 4: Dendritic Compression")
    print("-"*60)
    
    compressed_model, final_size = compress_pipeline(
        model=model,
        target_size_mb=config.target_size_mb,
        save_dir=config.model_dir,
        quantize=True,
    )
    
    return model, compressed_model


def main():
    """Entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train Sign Language CNN with Progressive Resolution'
    )
    parser.add_argument(
        '--start-resolution', type=int, default=1024,
        help='Starting resolution for curriculum (default: 1024)'
    )
    parser.add_argument(
        '--epochs-per-round', type=int, default=10,
        help='Training epochs per resolution round (default: 10)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Training batch size (default: 64)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Initial learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/asl_alphabet',
        help='Dataset directory (default: data/asl_alphabet)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='checkpoints',
        help='Model checkpoint directory (default: checkpoints)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--target-size', type=float, default=4.0,
        help='Target compressed model size in MB (default: 4.0)'
    )
    
    args = parser.parse_args()
    
    config = Config(
        start_resolution=args.start_resolution,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        target_size_mb=args.target_size,
    )
    
    train_progressive(config, resume_from=args.resume)


if __name__ == '__main__':
    main()
