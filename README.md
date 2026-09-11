# KWS on MRAM/ESP32 — Purdue Chips & AI Hardware Hackathon

An optimized, accurate keyword-spotting (KWS) model, compressed with
[Perforated AI](https://github.com/PerforatedAI/PerforatedAI) dendrites, and
deployed to an ESP32 with attached MRAM for on-device wake-word detection.

## Project phases

1. **Accurate model** — train a DS-CNN keyword spotter on Google Speech Commands v2.
2. **Optimize** — structured pruning, knowledge distillation, quantization spikes.
3. **Compress with dendrites** — grow Perforated AI dendrites on a tiny base
   model to close the accuracy gap without the parameter cost of a large network.
4. **Deploy** — run the compressed model on an ESP32 + MRAM against live
   microphone audio.

This repo currently covers phases 1-3.

**Reference benchmark**: Edge Impulse's published small-keyword-spotting
numbers (~3,800 parameters, ~92% accuracy) are the target this project is
measured against for the final, dendrite-compressed model.

The first size-focused dendrite cycle starts from the warm-distilled XS model,
prunes it to a 1,716-parameter XXS base, and initially runs PerforatedAI's
capacity check on its complete depthwise-separable blocks and classifier:

```bash
uv run python -m kws.optimize.dendritic
```

The capacity check uses validation accuracy only. After it reports that three
dendrites were added successfully, set `testing_dendrite_capacity: false` in
`configs/train/dendritic_cycle1.yaml` and use a new `--save-name` for the full
one-dendrite experiment.

## Status

- **M0 (data pipeline, models, tests)** — done. Full pipeline validated
  end-to-end against the real downloaded dataset.
- **M1 (baseline sanity check, no augmentation)** — done. 15 epochs,
  `configs/train/baseline.yaml`:

  | Model | Params | Test accuracy | FAR | FRR |
  |---|---|---|---|---|
  | DS-CNN-M | 146,902 | 96.52% | 2.52% | 4.19% |
  | DS-CNN-L | 467,942 | 97.27% | 1.45% | 3.76% |

  Both already approach/exceed the ~92% reference without augmentation —
  expected, since this 6-way task is easier than the 35-word benchmark DS-CNN
  was originally designed for. Confirms the data/model/training loop are
  correct.
- **Augmentation (`src/kws/data/augment.py`)** — implemented and verified
  (unit tests + a real-data end-to-end check). Fixed a bug found during
  verification: `speed_perturb` was resampling down then immediately back up,
  which round-trips to ~the original signal (no real speed change) at 2x the
  compute; now does a single resample reinterpreted at the original rate.
- **M2 (full augmented training across XS/S/M/L)** — not yet run.

## Task setup

Target classes: a handful of keywords (`configs/data/speech_commands_v2.yaml
-> target_keywords`, default `["yes", "no", "on", "off"]`) plus two synthesized
classes:
- `_unknown_` — pooled from the ~29 remaining Speech Commands v2 words (real,
  in-domain negative examples, not an external corpus)
- `_silence_` — fresh random crops of background noise, resampled every epoch

## Repo layout

```
configs/
  data/speech_commands_v2.yaml   # dataset, target keywords, features, splits
  model/ds_cnn_{xs,s,m,l}.yaml   # DS-CNN size variants
  train/{baseline,full,qat}.yaml # training regimens
src/kws/
  data/         # download, splits, dataset, silence, unknown, features, augment
  models/       # DSConvBlock + parameterized DSCNN
  optimize/     # structured pruning, distillation, PTQ/QAT spikes
  export/       # ONNX export + parity check
  utils/        # device selection, metrics (FAR/FRR), seeding, logging
  train.py      # training entrypoint
  evaluate.py   # accuracy / per-class F1 / confusion matrix / FAR-FRR
tests/          # split integrity, feature/augmentation correctness, model shapes,
                # pruning, ONNX parity
```

## Setup

```bash
uv sync
```

Install `uv` first if needed by following the instructions at
<https://docs.astral.sh/uv/getting-started/installation/>. `uv sync` creates a
project-local `.venv`, installs the package in editable mode, and installs the
locked runtime and development dependencies from `uv.lock`.

Works on Mac, Windows, and Linux. Two portability notes baked into the code:
- Audio I/O uses `soundfile`, not `torchaudio.load`/`save` — recent
  `torchaudio` routes those through TorchCodec, which needs a
  system-installed FFmpeg. `soundfile` ships self-contained wheels on all
  three platforms, so no extra system install is needed.
- Training device auto-selects CUDA (Windows/Linux+NVIDIA) → MPS (Apple
  Silicon) → CPU (`kws.utils.device.get_device`).

### Windows setup

1. **Install `uv`** using the
   [official installer](https://docs.astral.sh/uv/getting-started/installation/).
   `uv` will provision a compatible Python version when needed.
2. **Clone and enter the repo**:
   ```powershell
   git clone <this-repo-url>
   cd ChipsAIHackathon
   ```
3. **Create the environment and install dependencies**:
   ```powershell
   uv sync
   ```
   Run project commands through `uv run`; shell activation is not required.
   The resolved PyTorch build uses CUDA when a compatible build and NVIDIA GPU
   are available, otherwise the code falls back to CPU at runtime.
4. **Optional CUDA-specific PyTorch build**: if the default resolved build is
   not appropriate for your CUDA version, configure a PyTorch package index as
   documented by PyTorch and regenerate the lockfile. PyTorch on Windows
   requires the
   [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   (most Windows machines already have it).
   `torch` on Windows can use CUDA if
   you have a compatible NVIDIA GPU + driver, otherwise it falls back to CPU
   at runtime — either way `kws.utils.device.get_device()` picks the right
   device automatically. To target a specific CUDA version instead of the
   default, use the selector at
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
   for the correct `--index-url`.
5. Everything else — dataset download, training, evaluation, export — uses
   the exact same `uv run python -m kws....` commands shown in Usage below (no
   `source`/path-separator changes needed; Python's path handling normalizes
   the forward slashes used in this repo's configs and CLI args on Windows
   too). Make sure you have a few GB of free disk space before step 1 of
   Usage — the dataset download + extraction is ~2.3GB.

## Usage

```bash
# 1. Download + MD5-verify Google Speech Commands v2 (~2.3GB)
uv run python -m kws.data.download

# 2. Train (pick a model size and regimen)
uv run python -m kws.train \
  --model-config configs/model/ds_cnn_xs.yaml \
  --train-config configs/train/full.yaml \
  --checkpoint models/checkpoints/ds_cnn_xs.pt

# 3. Evaluate (accuracy, per-class F1, confusion matrix, FAR/FRR)
uv run python -m kws.evaluate --checkpoint models/checkpoints/ds_cnn_xs.pt \
  --report reports/ds_cnn_xs.json

# 4. Export to ONNX (with a PyTorch-vs-ONNXRuntime parity check)
uv run python -m kws.export.to_onnx --checkpoint models/checkpoints/ds_cnn_xs.pt \
  --onnx-path models/exported/ds_cnn_xs.onnx

# 5. IMC-oriented feature + response distillation (Large teacher -> fresh XS student)
uv run python -m kws.optimize.distill \
  --teacher-checkpoint models/checkpoints/ds_cnn_l.pt \
  --student-model-config configs/model/ds_cnn_xs.yaml \
  --out-checkpoint models/checkpoints/ds_cnn_xs_distilled.pt

# 6. Structured pruning + fine-tune (DS-CNN-L, secondary comparison arm)
uv run python -m kws.optimize.prune --checkpoint models/checkpoints/ds_cnn_l.pt \
  --keep-ratio 0.5 --out-checkpoint models/checkpoints/ds_cnn_l_pruned.pt
```

Run tests with `uv run pytest tests/`.

Training configs expose `num_workers`, `persistent_workers`, and
`prefetch_factor` for overlapping audio preparation with accelerator work. The
default configs use four persistent workers with two batches prefetched per
worker. Speed/pitch augmentation snaps random factors to a fine rational rate
grid and reuses prebuilt resampling kernels, avoiding the very large one-off
sinc kernels produced by arbitrary integer sample-rate pairs.

## Model sizes (6-way task: 4 keywords + unknown + silence)

| Variant   | Params  | Role                                              |
|-----------|---------|----------------------------------------------------|
| DS-CNN-XS | ~3,850  | Primary Phase 3 dendrite-growth starting point     |
| DS-CNN-S  | ~24K    | Secondary MCU-friendly reference point             |
| DS-CNN-M  | ~147K   | Mid-size comparison                                |
| DS-CNN-L  | ~468K   | Accuracy ceiling / pruning & KD teacher             |
