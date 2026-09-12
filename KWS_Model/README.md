# KWS on MRAM/ESP32 — Purdue Chips & AI Hardware Hackathon

An optimized, accurate keyword-spotting (KWS) model, compressed with
[Perforated AI](https://github.com/PerforatedAI/PerforatedAI) dendrites, and
deployed to an ESP32 with attached MRAM for on-device wake-word detection.

## The compression framework

The whole project is one pipeline, and every stage distills from the same
fixed teacher:

```
1. Train a strong full-precision teacher.

2. Choose a deployment-shaped student architecture.
   Distill teacher -> student to a strong baseline.

3. For each sparsity target:
   a. Apply structured or N:M pruning to the student.
   b. Fine-tune with task loss + KD loss from the fixed teacher.
   c. Run a perforation/dendrite phase:
      - freeze the relevant base weights,
      - add/train dendritic residual nodes,
      - select/freeze useful dendrites.
   d. Resume KD fine-tuning of the active student parameters.
   e. Record validation accuracy, latency, memory, and compute.
   f. Stop when the Pareto frontier stops improving,
      not merely when accuracy stops increasing.

4. Apply layer-wise weight clustering / codebook learning.

5. Run quantization-aware distillation fine-tuning:
   task loss + KD loss + fake-quantized forward pass.

6. Export and benchmark the exact inference graph
   on the intended target.
```

Run all of it, or any subset:

```bash
uv run --env-file .env python -m kws.pipeline
uv run --env-file .env python -m kws.pipeline --stages sparsity
uv run --env-file .env python -m kws.pipeline --stages cluster,quantize,benchmark
```

Stages reuse whatever earlier stages already produced (`--force` overrides),
because step 3 alone takes hours and re-running it to reach step 5 would make
the pipeline unusable. Everything is configured from
`configs/train/pipeline.yaml`, and the run's decision trail lands in
`reports/pipeline/pipeline.yaml`.

### Where each step lives

| Step | Module | Notes |
|---|---|---|
| 1 | `kws.train` | DS-CNN-L teacher, 97.5% test accuracy |
| 2 | `kws.optimize.distill` | Song et al. feature + response + task losses |
| 3a | `kws.optimize.prune` | structured channel pruning **or** N:M masks |
| 3b | `kws.optimize.prune` | KD fine-tune against the fixed teacher |
| 3c | `kws.optimize.dendritic` | PerforatedAI phases, with an audited freeze trail |
| 3d | `kws.optimize.dendritic` | KD resume on the clean deployment graph |
| 3e | `kws.utils.profile` | params, MACs, weight bytes, activation peak, latency |
| 3f | `kws.optimize.pareto` | frontier tracking and the patience stop rule |
| 4 | `kws.optimize.cluster` | per-layer k-means + learned codebooks |
| 5 | `kws.optimize.quantize_qat` | task + KD through a fake-quantized forward |
| 6 | `kws.export.benchmark` | exact ONNX/TorchScript graph accuracy and latency |

`kws.optimize.kd` holds the one KD implementation all of these share, so "KD
loss from the fixed teacher" means the same teacher and the same objective in
steps 2, 3b, 3d, 4, and 5.

Step 6 benchmarks ONNX for floating-point graphs. The current PyTorch/qnnpack
build cannot export eager int8 convolution weights to ONNX, so step 5 saves a
traced TorchScript int8 graph and step 6 benchmarks that exact artifact.
The configured automated target is currently the host CPU/qnnpack runtime; it
is a deployment proxy, not an ESP32 measurement. Final ESP32 + MRAM latency and
memory still require running the exported model through that device's runtime
and are not claimed by the checked-in reports.

When clustering is enabled, step 5 also writes a packed codebook sidecar next to
the TorchScript graph (`*.pt.codebook.pt`). It contains per-layer centroid tables
and packed 4-bit indices for a target-native runtime; the TorchScript file remains
the dense host-compatibility artifact. Unsupported target names are rejected
instead of being mislabeled as ESP32 measurements.

### Notes on three of the steps

**3f, the stopping rule.** The earlier search stopped at the first candidate
below the then-configured 90% accuracy floor. That ends the sweep at a point
that may still have had cheaper deployable models behind it, so the floor is
now an *admission* constraint (currently 85%) and the Pareto frontier is the stopping rule: the sweep continues
while candidates keep extending the accuracy-versus-cost frontier and stops
after `pareto_patience` consecutive candidates that do not. Configured accuracy
and relative-cost tolerances make "do not" mean no material frontier progress,
and each completed report records whether the streak came from inadmissible
accuracy or Pareto stagnation. Replaying the
An audit-only replay of the historical runs keeps all four of
w18/w17/w16/w15 on the frontier and would have continued past w14 rather than
stopping there. Those runs do not contain the other new step-3 substages, so
the executable framework deliberately starts a fresh sweep under the
`dendritic_framework_w*` prefix instead of reusing them.

**3c, the freeze.** PerforatedAI freezes the base weights during its dendrite
phase itself. The pipeline re-asserts it each epoch and records a phase trail of
what was actually frozen, so the framework step is checkable in the run record
instead of assumed. An unreadable PAI mode changes nothing, so a library change
degrades to PAI's own behaviour rather than corrupting a run.

**N:M on a model this small.** N:M groups along the flattened input dimension,
so a layer only admits a 2:4 pattern if that dimension is a multiple of 4. On
DS-CNN-XS the depthwise convs reduce over `1x3x3 = 9` weights, the stem over
`1x5x5 = 25`, and the first pointwise layer over 18 inputs; those layers are
left dense rather than forced into a pattern, and the skipped layers are
logged. Structured channel pruning remains the default,
because it is the only form of sparsity a TFLite Micro / ESP-DL runtime turns
into real savings today.

## Project phases

1. **Accurate model** - train a DS-CNN keyword spotter on Google Speech Commands v2.
2. **Optimize** - structured/N:M pruning, knowledge distillation, quantization.
3. **Compress with dendrites** - grow Perforated AI dendrites on a tiny base
   model to close the accuracy gap without the parameter cost of a large network.
4. **Deploy** - run the compressed model on an ESP32 + MRAM against live
   microphone audio.

This repo currently covers phases 1-3.

**Reference benchmark**: Edge Impulse's published small-keyword-spotting
numbers (~3,800 parameters, ~92% accuracy) are the target this project is
measured against for the final, dendrite-compressed model.

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
  expected, since this historical six-way task is easier than the 35-word benchmark DS-CNN
  was originally designed for. Confirms the data/model/training loop are
  correct.
- **Augmentation (`src/kws/data/augment.py`)** — implemented and verified
  (unit tests + a real-data end-to-end check). Fixed a bug found during
  verification: `speed_perturb` was resampling down then immediately back up,
  which round-trips to ~the original signal (no real speed change) at 2x the
  compute; now does a single resample reinterpreted at the original rate.
- **M2 (full augmented training across XS/S/M/L)** — not yet run.
- **Framework restructure** — done, code-complete and unit-tested; the stages
  that need long GPU runs have not been re-run yet. What that means concretely:

  | Framework step | Implemented | Run |
  |---|---|---|
  | 1 teacher | yes (existing checkpoint) | yes |
  | 2 student | yes | historical checkpoint exists; re-distill to stamp teacher-content provenance |
  | 3a structured pruning | yes | yes (widths 18-14) |
  | 3a N:M pruning | yes | not run |
  | 3b KD fine-tune after pruning | yes | not run — the recorded w14-w18 runs pre-date it |
  | 3c dendrites + audited freeze trail | yes | phases yes, trail not recorded for existing runs |
  | 3d KD resume | yes | not run |
  | 3e latency/memory/compute | yes | not recorded for existing runs |
  | 3f Pareto stopping rule | yes | audit-only replay; not re-swept |
  | 4 clustering + codebooks | yes | not run |
  | 5 quantization-aware distillation | yes | not run |
  | 6 export + benchmark | yes | not rerun; the checked-in report has no graph/cost artifact |

  Two things worth knowing before the next sweep. First, the existing
  pre-restructure `dendritic_prune_w1*` runs carry no KD pre-fine-tune, corrected
  freeze trail, KD resume, or cost record. They are retained as historical
  evidence but fail the framework-version guard and are not reused. Second, the
  XS evaluation reports contain accuracy and confusion metrics, but no exported
  graph, latency, or parity artifact; step 6 must therefore be run for the
  selected final candidate.

## Task setup

Target classes: ten keywords (`configs/data/speech_commands_v2.yaml
-> target_keywords`: yes, no, up, down, left, right, on, off, stop, go) plus two
classes:
- `_unknown_` — pooled from the remaining Speech Commands v2 words (real,
  in-domain negative examples, not an external corpus)
- `_silence_` — fresh random crops of background noise, resampled every epoch

For each split, `_unknown_` and `_silence_` are each sampled at twice the mean
number of examples in the ten keyword classes.

## Repo layout

```
configs/
  data/speech_commands_v2.yaml   # dataset, target keywords, features, splits
  model/ds_cnn_{xs,s,m,l}.yaml   # DS-CNN size variants
  train/{baseline,full,qat}.yaml # training regimens
src/kws/
  data/         # download, splits, dataset, silence, unknown, features, augment
  models/       # DSConvBlock + parameterized DSCNN
  optimize/
    kd.py                   # the one KD implementation every stage shares
    distill.py              # step 2
    prune.py                # steps 3a-3b: structured + N:M, KD fine-tune
    dendritic.py            # steps 3c-3e: perforation, KD resume, cost
    dendritic_prune_loop.py # step 3: the sweep
    pareto.py               # step 3f: frontier + stopping rule
    cluster.py              # step 4: clustering + codebook learning
    quantize_qat.py         # step 5: quantization-aware distillation
    quantize_ptq.py         # PTQ characterization spike
  export/
    to_onnx.py    # ONNX export + parity check
    benchmark.py  # step 6: benchmark the exported graph itself
  utils/        # device, metrics (FAR/FRR), seeding, logging, cost profiling
  pipeline.py   # the framework, stage by stage
  train.py      # training entrypoint + the shared fine-tuning loop
  evaluate.py   # accuracy / per-class F1 / confusion matrix / FAR-FRR
tests/          # split integrity, feature/augmentation correctness, model shapes,
                # pruning + N:M, clustering, cost profiling, KD, Pareto rule,
                # dendrite phase freezing, ONNX parity
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
  --checkpoint models/checkpoints/ds_cnn_xs_12class.pt

# 3. Evaluate (accuracy, per-class F1, confusion matrix, FAR/FRR)
uv run python -m kws.evaluate --checkpoint models/checkpoints/ds_cnn_xs_12class.pt \
  --report reports/ds_cnn_xs.json

# 4. Export to ONNX (with a PyTorch-vs-ONNXRuntime parity check)
uv run python -m kws.export.to_onnx --checkpoint models/checkpoints/ds_cnn_xs_12class.pt \
  --onnx-path models/exported/ds_cnn_xs.onnx

# 5. Step 2: distill the fixed teacher into the deployment-shaped student
uv run python -m kws.optimize.distill \
  --teacher-checkpoint models/checkpoints/ds_cnn_l_12class.pt \
  --student-model-config configs/model/ds_cnn_xs.yaml \
  --student-checkpoint models/checkpoints/ds_cnn_xs_12class.pt \
  --out-checkpoint models/checkpoints/ds_cnn_xs_distilled_warm_12class.pt

# 6. Steps 3a-3b: sparsify + KD fine-tune (structured, or --kind nm --n 2 --m 4)
uv run python -m kws.optimize.prune --checkpoint models/checkpoints/ds_cnn_l_12class.pt \
  --kind structured --keep-ratio 0.5 \
  --teacher-checkpoint models/checkpoints/ds_cnn_l_12class.pt \
  --out-checkpoint models/checkpoints/ds_cnn_l_pruned.pt

# 7. Step 3: the full sparsity sweep with the Pareto stopping rule
uv run --env-file .env python -m kws.optimize.dendritic_prune_loop

# 8. Steps 4-6: cluster the selected frontier point, run QAD, benchmark it
uv run --env-file .env python -m kws.pipeline \
  --stages cluster,quantize,benchmark
```

Run tests with `uv run pytest tests/`.

Training, optimization, and ONNX-export entry points accept an optional
`--seed` flag to override the seed in their YAML config or parity check. The
pipeline-level form,
`uv run python -m kws.pipeline --seed 0`, applies one seed consistently to every
stochastic stage it runs; evaluation and benchmarking also accept `--seed` for
deterministic dataset construction.

Training configs expose `num_workers`, `persistent_workers`, and
`prefetch_factor` for overlapping audio preparation with accelerator work. The
default configs use four persistent workers with two batches prefetched per
worker. Speed/pitch augmentation snaps random factors to a fine rational rate
grid and reuses prebuilt resampling kernels, avoiding the very large one-off
sinc kernels produced by arbitrary integer sample-rate pairs.

## Model sizes (12-way task: 10 keywords + unknown + silence)

| Variant   | Params  | Role                                              |
|-----------|---------|----------------------------------------------------|
| DS-CNN-XS | 4,096   | Primary Phase 3 dendrite-growth starting point     |
| DS-CNN-S  | 24,188  | Secondary MCU-friendly reference point             |
| DS-CNN-M  | 147,940 | Mid-size comparison                                |
| DS-CNN-L  | 469,604 | Accuracy ceiling / pruning & KD teacher             |
