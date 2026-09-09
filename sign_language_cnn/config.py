import os
import math
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration for Sign Language CNN training and compression."""
    
    # Model architecture
    model_input_size: int = 128
    num_classes: int = 29
    num_branches: int = 4
    
    # Training hyperparameters
    batch_size: int = 64
    epochs_per_round: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Progressive resolution curriculum
    start_resolution: int = 1024
    
    # Compression target
    target_size_mb: float = 4.0
    
    # Dataset
    dataset_name: str = 'grassknoted/asl-alphabet'
    data_dir: str = 'data/asl_alphabet'
    model_dir: str = 'checkpoints'
    
    # ASL Alphabet class names (29 classes)
    class_names: List[str] = field(default_factory=lambda: [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
    ])
    
    def build_resolution_schedule(self) -> List[int]:
        """Build the progressive resolution training schedule.
        
        Starting from start_resolution, halve the resolution each round.
        Stop when the next halving would produce model_input_size.
        
        Example with start_resolution=1024, model_input_size=128:
            Schedule: [1024, 512, 256]
            - Round 1: Train on images processed at 1024 effective resolution
            - Round 2: Train on images processed at 512 effective resolution  
            - Round 3: Train on images processed at 256 effective resolution
            - Stop: 256 / 2 = 128 = model_input_size
        
        Each round, images are first resized to the round's resolution,
        then resized to model_input_size (128x128) for model input.
        Higher intermediate resolutions preserve more detail; lower ones
        introduce progressive quality degradation that teaches robustness.
        """
        schedule = []
        res = self.start_resolution
        
        while res >= self.model_input_size * 2:
            schedule.append(res)
            res //= 2
        
        # Include the final resolution where res/2 == model_input_size
        if res == self.model_input_size * 2:
            schedule.append(res)
        
        if not schedule:
            # If start_resolution is already at 2x model_input_size
            schedule = [self.model_input_size * 2]
        
        return schedule
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the GPU compute device. Raises if no CUDA GPU is available."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. This model requires GPU training. "
                "Ensure CUDA drivers and a compatible GPU are available."
            )
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        return device
    
    def ensure_dirs(self):
        """Create necessary directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
