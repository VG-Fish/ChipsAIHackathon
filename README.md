# Sign Language CNN Classifier with Dendritic Compression

A CNN-based American Sign Language (ASL) alphabet classifier that uses **dendritic branching** for structured model compression. The model recognizes 29 ASL hand signs (A-Z + space, delete, nothing) from 128×128 RGB images.

## Architecture

### Dendritic CNN

The model uses a biologically-inspired **dendritic branching** structure where each convolutional layer is split into independent branches (dendrites). Each branch:

- Processes a subset of output channels independently
- Has a **learnable importance gate** that controls its contribution
- Can be evaluated and **pruned entirely** for compression

```
Input: 128×128×3 (RGB)
  │
  ├─ Stage 1: DendriticConvBlock(3 → 32, 4 branches × 8ch) + MaxPool2d
  ├─ Stage 2: DendriticConvBlock(32 → 64, 4 branches × 16ch) + MaxPool2d
  ├─ Stage 3: DendriticConvBlock(64 → 128, 4 branches × 32ch) + MaxPool2d
  ├─ Stage 4: DendriticConvBlock(128 → 256, 4 branches × 64ch) + MaxPool2d
  ├─ Stage 5: DendriticConvBlock(256 → 256, 4 branches × 64ch) + GlobalAvgPool
  │
  └─ Classifier: Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→29)

Total parameters: ~1,015,889 (~3.88 MB float32)
After dendritic pruning: ~1.00 MB
After INT8 quantization: ~0.25 MB
```

### Dendritic Compression

Compression follows a structured pruning pipeline inspired by biological dendritic trees:

1. **Branch Importance Scoring** — Each branch's importance = gate value × weight L1 norm
2. **Global Ranking** — All branches across all layers are ranked by importance
3. **Soft Pruning** — Least important branches have their gates zeroed
4. **Hard Pruning** — Zeroed branches are physically removed from the model
5. **INT8 Quantization** — Dynamic quantization for further 4× size reduction

Target: **< 4 MB RAM** after compression.

## Progressive Resolution Training

Training uses a **curriculum learning** strategy with decreasing image resolution:

```
Round 1: Images at 1024×1024 → resize to 128×128  (sharpest quality)
Round 2: Images at  512×512  → resize to 128×128  (slight quality loss)
Round 3: Images at  256×256  → resize to 128×128  (more quality loss)
Stop:    256 / 2 = 128 = model input size
```

Each round, images are first resized to the round's **intermediate resolution**, then resized to 128×128 for model input. Higher intermediate resolutions preserve more detail (better anti-aliasing), while lower ones introduce progressive quality degradation. This teaches the model to be **robust to varying image qualities**.

## Dataset

Uses the [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) Kaggle dataset:
- **87,000** training images at 200×200 pixels
- **29 classes**: A-Z + space, delete, nothing

### Setup Kaggle API

```bash
pip install kaggle
# Place your API key at ~/.kaggle/kaggle.json
# Download from: https://www.kaggle.com/settings → "Create New Token"
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
# Full training with default settings
python run_training.py

# Custom settings
python run_training.py \
  --start-resolution 1024 \
  --epochs-per-round 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --data-dir data/asl_alphabet \
  --target-size 4.0

# Resume from checkpoint
python run_training.py --resume checkpoints/model_best.pth
```

### Predict

```bash
# Single image
python run_predict.py path/to/hand_sign.jpg

# Directory of images
python run_predict.py path/to/image_dir/ --model checkpoints/model_best.pth

# Show top-3 predictions
python run_predict.py image.jpg --top-k 3
```

### Programmatic Usage

```python
from sign_language_cnn.model import DendriticCNN
from sign_language_cnn.predict import load_model, predict_image
from sign_language_cnn.compress import compress_pipeline
from sign_language_cnn.config import Config

# Load and predict
model = load_model('checkpoints/model_best.pth')
predictions = predict_image(model, 'hand_sign.jpg')
for letter, confidence in predictions:
    print(f"{letter}: {confidence*100:.1f}%")

# Compress a trained model
compressed, size_mb = compress_pipeline(
    model, target_size_mb=4.0, quantize=True
)
```

## Project Structure

```
ChipsAIHackathon/
├── sign_language_cnn/
│   ├── __init__.py       # Package exports
│   ├── config.py         # Configuration (hyperparams, paths, resolution schedule)
│   ├── model.py          # DendriticCNN + DendriticConvBlock architecture
│   ├── dataset.py        # Dataset download, loading, progressive transforms
│   ├── train.py          # Progressive resolution training loop
│   ├── compress.py       # Dendritic pruning + INT8 quantization
│   ├── predict.py        # Single/batch inference
│   └── utils.py          # Metrics, logging, model inspection
├── run_training.py       # Training entry point
├── run_predict.py        # Prediction entry point
├── requirements.txt      # Dependencies
└── README.md
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dendritic branches (4 per block)** | Enables structured pruning at branch granularity — more flexible than channel-wise, more efficient than weight-wise |
| **Learnable gate parameters** | Gates naturally learn to reflect branch importance during training, making pruning decisions data-driven |
| **128×128 input size** | Balances detail preservation with computational efficiency; matches the compression target |
| **Progressive resolution curriculum** | Trains robustness to varying image quality — critical for real-world deployment |
| **No bias in conv layers** | BatchNorm absorbs the bias term, reducing parameters without accuracy loss |
| **Kaiming initialization** | Proper initialization for ReLU networks prevents gradient issues |
