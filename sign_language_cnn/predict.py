"""Inference script for the Sign Language CNN.

Supports:
- Single image prediction
- Batch prediction on a directory
- Loading both compressed and uncompressed models
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from .config import Config
from .model import DendriticCNN


def load_model(
    checkpoint_path: str,
    device: torch.device = None,
) -> DendriticCNN:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model onto.
    
    Returns:
        Loaded DendriticCNN model in eval mode.
    """
    if device is None:
        device = Config.get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    num_classes = model_config.get('num_classes', 29)
    num_branches = model_config.get('num_branches', 4)
    input_size = model_config.get('input_size', 128)
    
    model = DendriticCNN(
        num_classes=num_classes,
        num_branches=num_branches,
        input_size=input_size,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def get_inference_transform(input_size: int = 128) -> transforms.Compose:
    """Get the image transform for inference."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(
    model: DendriticCNN,
    image_path: str,
    config: Config = None,
    device: torch.device = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Predict the sign language letter in an image.
    
    Args:
        model: Trained DendriticCNN model.
        image_path: Path to the image file.
        config: Configuration (for class names).
        device: Compute device.
        top_k: Number of top predictions to return.
    
    Returns:
        List of (class_name, confidence) tuples, sorted by confidence.
    """
    if config is None:
        config = Config()
    if device is None:
        device = Config.get_device()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_inference_transform(config.model_input_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference on GPU with FP16
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    
    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        class_name = config.class_names[idx] if idx < len(config.class_names) else f"class_{idx}"
        results.append((class_name, prob))
    
    return results


def predict_directory(
    model: DendriticCNN,
    directory: str,
    config: Config = None,
    device: torch.device = None,
) -> List[dict]:
    """Run prediction on all images in a directory.
    
    Args:
        model: Trained model.
        directory: Directory containing images.
        config: Configuration.
        device: Compute device.
    
    Returns:
        List of prediction result dictionaries.
    """
    if config is None:
        config = Config()
    if device is None:
        device = Config.get_device()
    
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue
        
        filepath = os.path.join(directory, filename)
        predictions = predict_image(model, filepath, config, device)
        
        result = {
            'filename': filename,
            'predicted_class': predictions[0][0],
            'confidence': predictions[0][1],
            'top_predictions': predictions,
        }
        results.append(result)
        
        print(
            f"  {filename}: {predictions[0][0]} "
            f"({predictions[0][1]*100:.1f}%)"
        )
    
    return results


def main():
    """Entry point for prediction."""
    parser = argparse.ArgumentParser(
        description='Run sign language prediction on images'
    )
    parser.add_argument(
        'input', type=str,
        help='Path to image file or directory of images'
    )
    parser.add_argument(
        '--model', type=str, default='checkpoints/model_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--top-k', type=int, default=5,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    config = Config()
    device = config.get_device()
    
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    print(f"Model loaded on {device}")
    
    if os.path.isdir(args.input):
        print(f"\nPredicting on directory: {args.input}")
        results = predict_directory(model, args.input, config, device)
        print(f"\nProcessed {len(results)} images.")
    else:
        print(f"\nPredicting: {args.input}")
        predictions = predict_image(
            model, args.input, config, device, top_k=args.top_k
        )
        
        print("\nResults:")
        for rank, (class_name, confidence) in enumerate(predictions, 1):
            bar = '█' * int(confidence * 30)
            print(f"  {rank}. {class_name:>10}: {confidence*100:6.2f}% {bar}")


if __name__ == '__main__':
    main()
