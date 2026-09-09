"""Dataset loading and preprocessing for ASL Alphabet classification.

Handles downloading the Kaggle ASL Alphabet dataset, loading images with
progressive resolution transforms for curriculum training, and creating
PyTorch DataLoaders.

Dataset: grassknoted/asl-alphabet
- 87,000 training images (200x200 pixels)
- 29 classes: A-Z + space, delete, nothing
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


def download_dataset(data_dir: str, dataset_name: str = 'grassknoted/asl-alphabet'):
    """Download the ASL Alphabet dataset from Kaggle.
    
    Requires the Kaggle API to be installed and configured:
        pip install kaggle
        Place kaggle.json in ~/.kaggle/
    
    Args:
        data_dir: Directory to download the dataset to.
        dataset_name: Kaggle dataset identifier.
    """
    train_dir = os.path.join(data_dir, 'asl_alphabet_train', 'asl_alphabet_train')
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print(f"Dataset already exists at {data_dir}")
        return
    
    print(f"Downloading dataset '{dataset_name}' from Kaggle...")
    print("Note: Requires 'kaggle' package and API credentials.")
    print("  pip install kaggle")
    print("  Place kaggle.json in ~/.kaggle/")
    print()
    
    try:
        import kaggle
        os.makedirs(data_dir, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name,
            path=data_dir,
            unzip=True,
        )
        print(f"Dataset downloaded and extracted to {data_dir}")
    except ImportError:
        print("ERROR: 'kaggle' package not installed.")
        print("Install with: pip install kaggle")
        print("Then place your kaggle.json API key in ~/.kaggle/")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{dataset_name}")
        print(f"  Extract to: {data_dir}")
        sys.exit(1)


class ProgressiveResolutionTransform:
    """Transform that simulates progressive resolution degradation.
    
    Resizes images through an intermediate resolution to simulate
    quality loss, then resizes to the model's input size.
    
    For curriculum training:
    - Higher intermediate_resolution → sharper final images (more detail preserved)
    - Lower intermediate_resolution → blurrier final images (detail lost in downscale + upscale)
    
    Args:
        intermediate_resolution: The intermediate resolution to pass through.
        model_input_size: The final resolution for model input (default 128).
        augment: Whether to apply data augmentation.
    """
    
    def __init__(
        self,
        intermediate_resolution: int,
        model_input_size: int = 128,
        augment: bool = True,
    ):
        self.intermediate_resolution = intermediate_resolution
        self.model_input_size = model_input_size
        self.augment = augment
        
        # Build the transform pipeline
        transform_list = []
        
        # Step 1: Resize to intermediate resolution
        # This simulates capturing at that resolution
        transform_list.append(
            transforms.Resize(
                (intermediate_resolution, intermediate_resolution),
                interpolation=transforms.InterpolationMode.LANCZOS,
            )
        )
        
        # Step 2: Resize to model input size
        # If intermediate < model_input, this upscales (adding blur)
        # If intermediate > model_input, this downscales (losing detail)
        if intermediate_resolution != model_input_size:
            transform_list.append(
                transforms.Resize(
                    (model_input_size, model_input_size),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                )
            )
        
        # Step 3: Data augmentation (training only)
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ])
        
        # Step 4: Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)
    
    def __repr__(self) -> str:
        return (
            f"ProgressiveResolutionTransform("
            f"intermediate={self.intermediate_resolution}, "
            f"output={self.model_input_size}, "
            f"augment={self.augment})"
        )


class ASLAlphabetDataset(Dataset):
    """PyTorch Dataset for the ASL Alphabet image classification task.
    
    Loads images from the directory structure:
        data_dir/asl_alphabet_train/asl_alphabet_train/<class_name>/<image>.jpg
    
    Args:
        data_dir: Root directory of the dataset.
        transform: Image transform to apply.
        max_per_class: Maximum images per class (for faster iteration). None for all.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        max_per_class: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        
        # Find training images directory
        self.image_dir = os.path.join(
            data_dir, 'asl_alphabet_train', 'asl_alphabet_train'
        )
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"Dataset not found at {self.image_dir}. "
                f"Run download_dataset() first or download manually."
            )
        
        # Discover classes and build file list
        self.classes = sorted(os.listdir(self.image_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples: List[Tuple[str, int]] = []
        for class_name in self.classes:
            class_dir = os.path.join(self.image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            files = sorted(os.listdir(class_dir))
            if max_per_class is not None:
                files = files[:max_per_class]
            
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images across {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        
        # Load image
        image = Image.open(filepath).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: just resize and convert to tensor
            image = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])(image)
        
        return image, label
    
    def set_transform(self, transform: Callable):
        """Update the transform (used to change resolution between rounds)."""
        self.transform = transform


def create_dataloaders(
    data_dir: str,
    intermediate_resolution: int,
    model_input_size: int = 128,
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 4,
    max_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.
    
    Args:
        data_dir: Root dataset directory.
        intermediate_resolution: Current curriculum resolution.
        model_input_size: Model input size (128).
        batch_size: Batch size.
        val_split: Fraction of data for validation.
        num_workers: DataLoader workers.
        max_per_class: Max images per class (None for all).
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Training transform with augmentation
    train_transform = ProgressiveResolutionTransform(
        intermediate_resolution=intermediate_resolution,
        model_input_size=model_input_size,
        augment=True,
    )
    
    # Validation transform without augmentation
    val_transform = ProgressiveResolutionTransform(
        intermediate_resolution=intermediate_resolution,
        model_input_size=model_input_size,
        augment=False,
    )
    
    # Load full dataset with training transform
    full_dataset = ASLAlphabetDataset(
        data_dir=data_dir,
        transform=train_transform,
        max_per_class=max_per_class,
    )
    
    # Split into train and validation
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Override validation transform (no augmentation)
    # Note: Since random_split creates Subset objects, we need a workaround
    # We create a separate dataset for validation
    val_full_dataset = ASLAlphabetDataset(
        data_dir=data_dir,
        transform=val_transform,
        max_per_class=max_per_class,
    )
    
    # Use the same indices from the split
    val_dataset_proper = torch.utils.data.Subset(val_full_dataset, val_dataset.indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset_proper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
