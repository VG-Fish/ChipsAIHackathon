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

For a focused, budget-matched compression experiment (without running the
later clustering/export stages), use `configs/train/compression_experiment.yaml`.
Always inspect the candidate widths, selective PAI placements, and projected
parameter/MAC admission decisions first:

```bash
uv run python -m kws.pipeline --extreme-prune --dry-run \
  --output-dir outputs/compression-plan
uv run --env-file .env python -m kws.pipeline --extreme-prune \
  --output-dir outputs/compression-run

# Optional multi-dendrite run. PAI can add up to three dendrites per selected
# module, but its history/retry policy may stop earlier when validation stops
# improving.
uv run --env-file .env python -m kws.pipeline --extreme-prune \
  --max-dendrites 3 \
  --output-dir outputs/compression-run-multi-dendrite

# Remove the hard dendrite cap. PAI stops after validation no longer improves
# and its configured candidate retries are exhausted.
uv run --env-file .env python -m kws.pipeline --extreme-prune \
  --max-dendrites -1 \
  --output-dir outputs/compression-run-unlimited-dendrites
```

`--extreme-prune` selects `configs/train/compression_experiment.yaml` by
default. The shipped recipe compares classifier-only dendrites, late-block plus
classifier dendrites, and dendrites on every `DSConvBlock` plus the classifier.
Projected budget checks may skip the broader placement at larger widths. Pass
`--config PATH` to use another compression-experiment config.
Use `--max-dendrites N` for a separate multi-dendrite run. This is an upper
bound, not a request to force exactly `N`: PerforatedAI retries a non-improving
candidate according to `max_dendrite_tries`, then stops when its validation
history no longer supports another dendrite. Use `--max-dendrites -1` to remove
the reachable hard cap and rely on that no-improvement stop. The pipeline maps
`-1` to an effectively unreachable positive PAI limit rather than relying on
undocumented negative-value behavior in the third-party API. Since an
unlimited search has no finite worst-case size, its pre-training budget check
uses and labels a one-dendrite lower bound; the measured clean export is still
rejected if its actual parameter count or MACs exceed the experiment budget.

The executable run trains each conventional structured backbone once, reuses
that exact checkpoint for each eligible dendritic placement, and rejects a PAI
candidate both before training (projected cost) and after clean export
(measured cost) if it exceeds the shared budget. The report at
`reports/compression_experiment.yaml` records validation accuracy, parameters,
MACs, matched-budget conventional comparisons, and the joint Pareto frontier.
PAI integrations also receive immutable `cycle_checkpoints/` snapshots in
addition to the existing resumable `latest.pt` pair. Set
`perforatedai.on_unavailable: skip` to keep conventional results when PAI is
not installed/licensed; dry-run mode never imports it.

PerforatedAI/PerforatedBP validates its license while its modules are imported.
Run every PAI-dependent command from this `KWS_Model` directory with the
`uv run --env-file .env ...` form above. A direct `.venv/bin/python` invocation
from a temporary working directory bypasses the configured run environment and
can make the package fall back to an interactive `email:` prompt, which cannot
work in a headless pipeline or test process.

Every output-producing command also accepts `--output-dir PATH`. `PATH` is the
exact run root (it is not wrapped in a timestamp), so invoking the command
again with the same path is how a compatible phase resumes. The pipeline may
set the same value with the top-level `output_dir` config key; the CLI flag wins.
For example:

```bash
uv run --env-file .env python -m kws.pipeline \
  --output-dir outputs/my-kws-run
```

Resumable commands also accept `--resume-dir PATH` as a shorthand for using
that existing run root with `--resume`. It validates that the directory exists
and selects its canonical `latest.pt`; `--resume-from PATH` can still override
the checkpoint when needed. For example:

```bash
uv run python -m kws.train \
  --model-config configs/model/sparknet_c12.yaml \
  --train-config configs/train/light.yaml \
  --stage student \
  --resume-dir outputs/phase_b/step2_sparknet_c12_light_20260913T215424Z
```

The pipeline accepts the same flag when reusing or continuing stages in an
existing run root.

### Controlled SparkNet frontend/gate comparison

For a reproducible Phase-B comparison, keep `configs/train/light.yaml`, the
seed, batch size, workers, and output layout fixed while changing only the
data frontend and gate width. Run the four supervised controls:

```bash
for frontend in speech_commands_v2.yaml speech_commands_v2_mfcc32.yaml; do
  for gate in 32 40; do
    name="sparknet_c12_${frontend%.yaml}_g${gate}"
    model=configs/model/sparknet_c12.yaml
    [ "$gate" = 40 ] && model=configs/model/sparknet_c12_g40.yaml
    uv run python -m kws.train \
      --data-config "configs/data/$frontend" \
      --model-config "$model" \
      --train-config configs/train/light.yaml \
      --stage student --seed 0 --graphs \
      --output-dir "outputs/phase_b/$name"
  done
done
```

Select the highest validation accuracy (and retain the parameter/MAC cost)
from those four runs. Then use that exact data/model pair for the annealed KD
experiment. `configs/train/light_kd_annealed.yaml` starts with pure
classification loss and linearly ramps response KD from 0 to 0.5 over epochs
0–40, logging the effective weights each epoch:

```bash
uv run python -m kws.optimize.distill \
  --teacher-checkpoint models/checkpoints/ds_cnn_l_12class.pt \
  --data-config configs/data/speech_commands_v2.yaml \
  --student-model-config configs/model/sparknet_c12.yaml \
  --train-config configs/train/light_kd_annealed.yaml \
  --output-dir outputs/phase_b/sparknet_c12_logmel_g32_kd_annealed \
  --graphs --seed 0
```

The current DS-CNN-L teacher expects 40-bin log-mel inputs. If MFCC-32 wins
the supervised comparison, first provide a teacher checkpoint trained on the
same MFCC-32 frontend; distillation deliberately rejects a teacher/student
feature-shape mismatch instead of comparing incompatible inputs.

### Paper-faithful SparkNet C16 reproduction

The paper's C16 result is a distinct benchmark, not a compression baseline:
it uses MFCC-32, 1× unknown/silence balancing, a fixed silence set drawn from
`_background_noise_`, SGD on `100 * CE + gate_sparsity`, and a
warmup–hold–polynomial schedule. Run it without KD, pruning, or PAI:

```bash
uv run python -m kws.train \
  --data-config configs/data/speech_commands_v2_mfcc32_paper.yaml \
  --model-config configs/model/sparknet_c16_paper.yaml \
  --train-config configs/train/sparknet_c16_paper.yaml \
  --stage paper_replication \
  --seed 0 \
  --output-dir outputs/sparknet-paper-replication/c16-seed0
```

The expected paper reference is 95.7% SC2 test accuracy, 4,636 parameters,
and 454.5K MACs measured with `thop`.

The original balanced JSON manifests were not released. They came from NeMo's
`process_speech_commands_data.py --class_split sub --rebalance`, so this
config reproduces that script's rules against the official split lists rather
than copying its outputs: `rounding: ceil` sizes `_unknown_` and `_silence_`
as `ceil` of the keyword-class mean (3077 / 371 / 408 on v0.02), and
`materialize: true` draws each split's silence once — gain-scaled one-second
`_background_noise_` crops, from a seed-derived per-split stream — so
evaluation is reproducible. The individual clips still differ from the
authors', so record the measured result separately rather than claiming it is
an exact evaluation of their hidden manifests.

The paper's 95.7 ± 0.17 is a mean and standard deviation over several runs.
Sweep `--seed` and compare the resulting mean and spread; a single run's
number is not directly comparable.

The run root contains `manifest.yaml`, append-only invocation logs under
`logs/`, effective config snapshots under `metadata/configs/`, canonical
epoch JSONL under `metrics/`, separate resumable `latest.pt` and deployable
`best.pt` checkpoints under `models/checkpoints/`, PAI-native files under
`pai/candidates/`, exported graphs under `models/exported/`, and stage reports
under `reports/`. The default in-checkout root `KWS_Model/outputs/` is ignored;
an arbitrary directory inside the checkout cannot be ignored automatically.

Adding `--graphs` to any of those commands turns each phase's JSONL into a
watchable chart after every epoch: `graphs/<stage>/<phase>.csv` holds the same
records one row per epoch, and `graphs/<stage>/<phase>.html` is an interactive
Plotly chart with hover values, zoom/pan, autoscale, trace toggles, and image
export. The page loads a pinned Plotly.js build from the official CDN and
reloads itself every few seconds while `live` is checked. Open
`graphs/index.html` to see every phase in the run. Charts are derived data, so a
failure to write one is logged and never ends a training run, and a run that was
started without the flag can be charted after the fact:

```bash
uv run python -m kws.utils.graphs outputs/my-kws-run --watch
```

`manifest.yaml` is the run-identity authority. Every project-owned checkpoint
records the manifest's `run_id`, including teacher/student training
checkpoints, each sparsity candidate's prune/KD and resume-KD checkpoints, the
PAI sidecar and its paired native PAI checkpoint, cluster and quantize
checkpoints, and the exported/packed-codebook deployment artifacts. Resume and
stage reuse reject a checkpoint whose `run_id` does not match the active
manifest. External input checkpoints (an explicit `--checkpoint`,
`--teacher-checkpoint`, or a `--resume-from` path outside the active root)
remain inputs and are recorded with their own path and digest instead.

Stage reuse means a completed artifact is validated and consumed again. Epoch
resume is different: `latest.pt` contains the optimizer, scheduler, KD/QAT
state, metric commit, and process RNG state, and continues from the next
completed epoch with `--resume` or `--resume-from PATH`. Recipe, upstream
digest, seed, and schedule changes are rejected instead of silently starting
over. A legacy best-only checkpoint remains valid for inference or an explicit
warm start, but is not a resumable training state. Multi-worker augmentation
has functional continuation guarantees; exact batch-for-batch replay is
supported by tests with `num_workers: 0` and is not promised for persistent
multi-worker loaders.

Stages reuse whatever earlier stages already produced (`--force` overrides),
because step 3 alone takes hours and re-running it to reach step 5 would make
the pipeline unusable. Everything is configured from
`configs/train/pipeline.yaml`, and the run's decision trail lands in
`reports/pipeline.yaml`.

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
accuracy or Pareto stagnation. An audit-only replay of the historical runs
keeps all four of w18/w17/w16/w15 on the frontier and would have continued past
w14 rather than stopping there. Those runs do not contain the other new step-3
substages, so the executable framework deliberately starts a fresh sweep under
the `dendritic_framework_w*` prefix instead of reusing them.

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

  This table predates the later C16-derived `.fc` audit. That audit has
  validation-only C12/C10/C8 artifacts under
  `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/`, including resume
  and clean-graph cost records, but it is not a paper-test evaluation and its
  directory-labeled five runs reused downstream seed 0. The runner/configs now
  record and propagate each experiment seed; treat the old d3 outputs as
  historical evidence, not as independently seeded replications.

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

Run tests with `uv run python -m pytest tests/`.

Training, optimization, and ONNX-export entry points accept an optional
`--seed` flag to override the seed in their YAML config or parity check. The
pipeline-level form,
`uv run python -m kws.pipeline --seed 0`, applies one seed consistently to every
stochastic stage it runs; evaluation and benchmarking also accept `--seed` for
deterministic dataset construction.

Training configs expose `num_workers`, `persistent_workers`, and
`prefetch_factor` for overlapping audio preparation with accelerator work. The
default configs use four persistent workers with two batches prefetched per
worker. Worker startup is attempted on macOS as well; if the host cannot start
PyTorch's shared-memory manager, the loader logs the reason and retries the
same epoch with `num_workers=0` while restoring the shuffle generator state.
Deterministic features, including non-augmented training features, are
precomputed once per process and cached in memory. When an augmented training
config sets `cache_train_features: true`, it samples one waveform + SpecAugment
view per entry at startup (including synthesized silence) and reuses those
features throughout the run. This trades augmentation diversity between
epochs for much lower input and feature-extraction overhead. The Phase B
`light.yaml` and `light_kd.yaml` recipes instead set
`cache_train_features: false`, so they resample augmentation every epoch. The
default training batch is 256
(override it in a train YAML if the available accelerator memory is smaller).
The default recipe uses ±150 ms shifts, 0.85–1.15 speed changes, 75% noise
mixing down to −5 dB SNR, and two larger SpecAugment masks. Speed/pitch
augmentation snaps random factors to a fine rational rate grid and reuses
prebuilt resampling kernels, avoiding the very large one-off sinc kernels
produced by arbitrary integer sample-rate pairs.

For SparkNet's Phase B response-only KD experiment, `light_kd.yaml` uses
temperature 2, response/classification weights 0.5/0.5, label smoothing 0.1,
and no feature matching. The original T=1, 1/7 response / 6/7 classification
recipe produced a weak, mostly CE-aligned logit gradient in paired training-view
diagnostics; the teacher remained accurate on the student's light augmentation.
The stronger response setting is an evidence-motivated experiment, not a proven
accuracy improvement. See `PLAN.md` for measurements and remaining uncertainty,
including the unverified historical teacher training recipe. Other KD configs
retain their existing settings. Start a fresh run for the changed recipe;
`--resume-dir` intentionally rejects a checkpoint with different KD weights.

Shared KD stages log `train_teacher_accuracy`, `train_teacher_confidence`,
`train_teacher_true_class_probability`, `train_teacher_entropy_nats`,
`train_weighted_response_loss`, `train_weighted_classification_loss`,
`train_kd_logit_grad_norm_ratio`, and `train_kd_logit_grad_cosine` to JSONL and,
with `--graphs`, the live HTML/CSV. Teacher probabilities are measured at T=1
on the actual augmented inputs. Gradient metrics compare weighted response/CE
gradients with respect to logits (not model parameters), using the configured
temperature. Epoch values are sample-weighted batch means, so the cosine is
not a global epoch-gradient cosine. These diagnostics are detached, do not
change the objective, and require no extra backward pass or per-batch CPU sync.

## Model sizes (12-way task: 10 keywords + unknown + silence)

Plan the no-KD SparkNet C12 dendritic-pruning comparison without training:

```bash
uv run python -m kws.optimize.sparknet_dendritic_prune_experiment \
  --config configs/experiment/sparknet_c12_dendritic_prune_no_kd.yaml \
  --no-KD --dry-run
```

Then launch the full validation-only C10/C8/C6 sweep:

```bash
uv run python -m kws.optimize.sparknet_dendritic_prune_experiment \
  --config configs/experiment/sparknet_c12_dendritic_prune_no_kd.yaml \
  --no-KD
```

`--no-KD` (also accepted as `--no-kd`) forces a null teacher even though the
standalone dendritic command retains its historical DS-CNN teacher default.
For each width the runner saves one supervised prune-fine-tune baseline, starts
PAI from that exact best checkpoint, and performs a supervised post-PAI resume.
PAI uses the cap, threshold schedule, candidate retries, and eight-epoch
history lookback written in the selected YAML; these settings are part of the
experiment identity. In particular, the current checked-in C12 config uses a
one-dendrite cap, while the historical C16 `.fc` artifacts above used three.
Do not compare those output families as one sweep.
The report records raw validation gain, gain above the measured pruning-only
curve when its parameter range brackets the dendritic model, and parameter/MAC
growth. It records costs but applies no budget admission or rejection.

| Variant   | Params  | Role                                              |
|-----------|---------|----------------------------------------------------|
| DS-CNN-XS | 4,096   | Primary Phase 3 dendrite-growth starting point     |
| DS-CNN-S  | 24,188  | Secondary MCU-friendly reference point             |
| DS-CNN-M  | 147,940 | Mid-size comparison                                |
| DS-CNN-L  | 469,604 | Accuracy ceiling / pruning & KD teacher             |
