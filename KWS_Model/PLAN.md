# Unified KWS Run Outputs, Metrics, and Resume Plan

## Goal

Give every KWS command one optional, user-selected run directory and keep every
artifact produced by that run inside it: console logs, epoch metrics, best and
latest training checkpoints, PerforatedAI (PAI) files, deployable models, and
reports. A run interrupted after a completed epoch should be able to continue
from that epoch without losing optimizer or scheduler progress.

This plan is based on the repository's tracked source, configs, tests, project
documentation, and PAI Skills package. The ignored virtual environments,
caches, raw data, historical reports, model binaries, and `dendritic_*` run
directories are generated state, not source. A historical dendritic directory
was inspected only to confirm PAI's native filenames and output conventions;
`.env` was not read.

## Original baseline and gaps

The bullets in this section describe the repository when this plan was first
written. The dated post-implementation audit below is the current defect list.

- Output paths are split among CLI flags, two YAML files, implicit paths rooted
  at the current working directory, and stdout. The pipeline alone currently
  has separate paths for its report, teacher, student, sparsity summary and
  prefix, clustered graph, quantized graph, ONNX graph, and benchmark report
  (`configs/train/pipeline.yaml` and
  `configs/train/dendritic_prune_loop.yaml`).
- `kws.utils.logging.get_logger` creates stdout-only handlers, so no durable log
  captures project logging, PAI `print` output, warnings, or tracebacks.
- `run_finetune` computes epoch `train_loss`, `val_loss`, `val_acc`, and KD loss
  components, and returns them in memory. `train_model` and distillation reduce
  that result to a single best validation accuracy, so their histories are
  lost. The PAI loop computes train accuracy and additional phase information,
  but KWS does not persist a uniform epoch history.
- Ordinary training saves only a best-model checkpoint. It does not save the
  completed epoch, global step, optimizer, scheduler, KD adapter, history, or
  RNG state. Cluster and QAT best state exists only in memory until the phase
  completes; the QAT `.pt` is a converted deployment graph, not resumable
  training state.
- Pipeline reuse is stage-level reuse of completed artifacts. It is not
  epoch-level crash recovery. A nonempty partial PAI candidate is currently
  rejected with advice to resume manually.
- PAI owns a large family of CSV, PNG, config, and checkpoint files under its
  `save_name`. The installed PAI version also creates a native `latest.pt` and
  exposes `save_system`/`load_system` for network plus tracker state, but KWS
  still needs a sidecar for its optimizer, scheduler, KD adapter, RNG, and
  uniform metrics.
- The installed PAI implementation expects a leaf `save_name` and warns that
  slash-containing relative paths are unsupported. Supplying
  `<output-dir>/pai/<candidate>` directly is therefore not a valid routing
  strategy.
- Raw Speech Commands data and preprocessing caches are large, reusable inputs.
  They should remain controlled by the data config rather than being copied
  into every run. The download command's console output can still be logged to
  a selected output directory.

## Post-implementation audit findings (2026-09-12)

The implementation is not yet plan-complete. The existing Python and PAI
Skills test suites pass, but the audit found the following unresolved defects.
All items below must be fixed and covered by regression tests before using the
definition of done.

Verification after the current fix pass: `184` Python tests passed,
`PAI Skills/test.sh` passed all `26` checks, compilation passed, and
`git diff --check` passed. The remaining unchecked items are not masked by
those passing suites.

### Blocking training failures

- [x] `DistillationCriterion` has no `state_dict` or `load_state_dict`, while
  the shared training loop calls both when saving and restoring checkpoints.
  Every KD-backed path can therefore fail at its first completed checkpoint or
  on resume. Define an explicit KD state contract that includes trainable
  adapters and use it consistently in student distillation, pruning/KD,
  clustering/KD, QAT/KD, and post-PAI KD resume.
- [x] `kws.optimize.distill` calls `atomic_torch_save` with three arguments even
  though the helper accepts two. The first improved validation epoch crashes
  before the best checkpoint and latest state can be committed.

### Resume and checkpoint correctness

- [x] `kws.optimize.prune --resume` does not resolve the phase's standard
  `latest.pt`; resume only occurs when `--resume-from` is also supplied. Make
  the flag behave like train, distill, and QAT, and prevent a fresh epoch-1 run
  from appending to old metrics.
- [x] The PAI sidecar stores adapter optimizer/scheduler state but not the KD
  adapter weights/state. A resumed PAI cycle therefore uses a newly initialized
  adapter. Save and restore the adapter before loading its optimizer.
- [x] Standalone `kws.optimize.dendritic` has no `--resume` or `--resume-from`
  option and does not safely handle a nonempty native PAI candidate when its KWS
  sidecar is absent. It must reject incompatible partial state, resume a valid
  pair, or reuse explicitly completed metadata; it must never silently
  overwrite the candidate.
- [x] PAI pair validation accepts missing native checkpoint/digest metadata in
  some paths, and loads the sidecar model with `strict=False` without validating
  missing or unexpected keys. Require both members, both digests, compatible
  metadata, and an exact expected state schema.
- [x] The first post-restructure PAI sidecar/native save occurs before the
  adapter optimizer is reset. An interruption before the second save can pair a
  post-restructure model with a pre-restructure optimizer. Make the paired
  commit represent one internally consistent epoch boundary.
- [x] QAT writes `latest.pt` but never writes the standard
  `models/checkpoints/quantize/best.pt`; its in-memory best state is only used
  for the exported TorchScript graph. Persist distinct best and latest training
  checkpoints.
- [x] Post-PAI KD resume can return `no_improvement` without ever creating
  `best.pt`. The first completed validation epoch must establish a best
  checkpoint even when its metric is zero or fails to improve over imported
  metadata.
- [x] Cluster `best.pt` is assembled by copying the full latest training
  checkpoint and replacing only `model_state_dict`, leaving it labeled as a
  latest-style `kws_training_state`. Give best checkpoints an unambiguous,
  loadable schema consistent with every other phase.

### Metrics, manifests, and provenance

- [x] PAI epoch metrics discard the named KD classification, response, and
  feature losses and record only total loss. They also omit the adapter
  optimizer learning rate. Persist every component required by this plan.
- [x] `MetricsRecorder` raises on a malformed trailing JSONL fragment instead
  of trimming the incomplete final record. Recover only the trailing fragment,
  preserve every prior valid record, then reconcile metrics against the latest
  checkpoint.
- [x] A forced/fresh rerun can restart at epoch 1 while retaining an existing
  recorder history, producing duplicate epoch records. Fresh and forced phase
  execution must archive, replace, or explicitly reset its canonical history.
- [x] Registering an artifact path a second time returns its old manifest
  record without refreshing digest and size. Repeated invocations that
  overwrite reports or checkpoints therefore leave stale integrity metadata;
  the final containment/integrity failure is then swallowed by logging cleanup.
  Refresh the record atomically and do not suppress manifest finalization
  errors.
- [x] The manifest schema does not store the required effective-config digest,
  and the pipeline snapshots only a subset of configs. Record the pipeline,
  data, model, phase train, sparsity, cluster, quantization, benchmark, and
  effective merged configs, plus warm-start inputs and digests.
- [x] Tie checkpoint run IDs to the manifest run ID. `ArtifactLayout` now owns
  one stable manifest ID per output root; every project-owned training,
  clustering, pruning, QAT, PAI, native-PAI, packed-codebook, and deployment
  checkpoint records that ID, while resume and pipeline reuse reject mismatched
  checkpoints. External input checkpoints remain allowed as recorded inputs.
- [x] Resume fingerprints are incomplete for QAT, clustering, post-PAI KD, and
  pruning. Include data/model recipes, source and teacher content digests,
  upstream graph digests, and all schedule-critical optimization settings so an
  incompatible run cannot be accepted accidentally.

### Output containment, portability, and documentation

- [x] Several active-root entrypoints reduce user-provided output paths to
  `Path(...).name`. Absolute paths and `..` escapes are therefore silently
  rewritten instead of rejected. Evaluation, ONNX export, benchmarking, PTQ,
  QAT, pipeline output mapping, and artifact stage/phase helpers must validate
  the original value before any normalization.
- [x] Pipeline and sparsity reports store absolute run-owned output paths, and
  candidate reuse consumes those literal paths. Store run-owned outputs
  relative to the run root so moving a complete run does not break reuse.
- [x] The README documents the unified pipeline report as
  `reports/pipeline/pipeline.yaml`, while the plan and implementation use
  `reports/pipeline.yaml`. Make the documented tree and actual layout agree.
- [ ] The current test suite does not exercise the fatal KD checkpoint path or
  the required real/dependency-backed PAI interruption/resume contract. Passing
  tests are not completion evidence until the missing cases below are added.

## Required behavior

1. Add `--output-dir PATH` to every output-producing `python -m kws...`
   entrypoint. Add an optional top-level `output_dir` to the pipeline config.
2. Treat `PATH` as the exact run root; do not silently add a timestamped parent
   or child. Reusing the same path is how a user resumes the same run.
3. With an active output root, every project-owned runtime artifact must resolve
   below it. Input configs, input/warm-start checkpoints, and the shared dataset
   remain external inputs and are recorded by path and digest rather than
   copied.
4. Without an output root, retain today's paths, required flags, and best
   checkpoint formats so existing commands and old model artifacts keep
   working.
5. Store the best deployable weights separately from `latest.pt`, which is a
   full training-state checkpoint written after every completed epoch.
6. Persist every metric the code computes during training, including accuracy,
   losses, KD components, learning rate, and PAI phase/cost fields where they
   exist. JSONL is the canonical record; TensorBoard may mirror it but must not
   be the only copy.
7. Resume only when the checkpoint's recipe and upstream artifact digests match
   the requested run. Never silently turn an incompatible resume into a fresh
   run.
8. Use atomic replacement for checkpoints, manifests, YAML/JSON summaries, and
   converted model metadata. A crash must leave either the previous valid file
   or the new valid file, not a partially written file.
9. Preserve all invocations and failures in append-only/per-invocation logs.
   Do not record `.env` contents, credentials, or tokens in logs or manifests.

## Output directory contract

Use one shared `ArtifactLayout` to produce paths; callers must not concatenate
their own run-root strings.

```text
<output-dir>/
  manifest.yaml
  .run.lock
  logs/
    2026-09-11T120000_kws.pipeline.log
    2026-09-11T153000_kws.pipeline.log
  metadata/
    configs/                         # effective non-secret YAML snapshots
  metrics/
    teacher/train.jsonl
    student/distill.jsonl
    sparsity/w18/prune_kd.jsonl
    sparsity/w18/pai.jsonl
    sparsity/w18/resume_kd.jsonl
    cluster/codebook.jsonl
    quantize/qat.jsonl
    summaries.yaml
  models/
    checkpoints/
      teacher/{best.pt,latest.pt}
      student/{best.pt,latest.pt}
      sparsity/w18/prune_kd/{best.pt,latest.pt}
      sparsity/w18/pai/latest.pt     # KWS sidecar, not PAI's native file
      sparsity/w18/resume_kd/{best.pt,latest.pt}
      cluster/{best.pt,latest.pt}
      quantize/{best.pt,latest.pt}
    exported/
      clustered.pt
      kws_int8.pt
      kws_int8.pt.yaml
      kws_int8.pt.codebook.pt
      kws.onnx
  pai/
    candidates/w18/                  # native PAI CSV/PNG/config/checkpoints
      latest.pt                      # PAI network/tracker format
      ...
    reload/                          # reconstruction-only PAI state if retained
  reports/
    pipeline.yaml
    sparsity.yaml
    evaluation.json
    benchmark.json
    export.json
    ptq.json
```

The exact model basenames may remain configurable, but their category and parent
directory may not. Stage and phase names must be stable because resume and
report links depend on them. Store paths in the manifest and reports relative
to the run root when possible so the entire directory can be moved.

Every checkpoint in the tree above carries an explicit `run_id` field equal to
this run's `manifest.yaml` `run_id`: teacher/student training checkpoints,
every `sparsity/<candidate>/{prune_kd,resume_kd}` checkpoint, the PAI sidecar
at `sparsity/<candidate>/pai/latest.pt` and its paired native
`pai/candidates/<candidate>/latest.pt`, cluster and quantize checkpoints, and
the exported/packed-codebook deployment artifacts under `models/exported/`.
`ArtifactLayout.manifest_run_id` is the single source of that ID per output
root; `validate_checkpoint_run_id` and pipeline stage reuse reject any
checkpoint whose `run_id` does not match it. External input checkpoints
(`--checkpoint`, `--teacher-checkpoint`, `--resume-from` pointing outside the
active root) are exempt and are instead recorded as manifest inputs by path
and digest.

### Path and precedence rules

- `--output-dir` overrides `pipeline.yaml:output_dir`.
- If neither is present, use all existing CLI/YAML output paths unchanged.
- Once a root is active, legacy output flags/fields may provide relative names
  below the appropriate category. Reject an absolute path or `..` traversal
  that escapes the root. Do not silently write the conflicting path.
- Input-role options (`--checkpoint` on evaluation/optimization,
  `--teacher-checkpoint`, `--student-checkpoint`, data/model/train configs) are
  never rebased. Resolve them to absolute paths before entering PAI's working
  directory.
- Output-role options (`kws.train --checkpoint`, every `--out-checkpoint`,
  `--report`, `--onnx-*-path`, pipeline report/checkpoint fields,
  `save_prefix`) are mapped by `ArtifactLayout` when a root is active.
- With `--output-dir`, formerly required output-only flags become optional and
  use the standard paths above. With no root they remain required for backward
  compatibility.
- Acquire an advisory `.run.lock` while mutating the run. Reject a second writer
  instead of allowing two PAI jobs or metric writers to corrupt one directory.
  Read-only evaluation can use its own invocation log but must serialize report
  and manifest updates.

## Shared implementation components

### 1. Artifact layout and run manifest

Add `src/kws/utils/artifacts.py` with:

- `ArtifactLayout(root)` and typed helpers for logs, phase metrics, best/latest
  checkpoints, PAI candidate directories, exported models, and reports.
- Descendant checks based on resolved paths, directory creation, a run lock,
  atomic text/JSON/YAML writers, and atomic `torch.save` through a temporary file
  in the destination directory followed by `Path.replace`.
- A manifest schema containing a format version, run ID, invocation IDs,
  command/argv, start/end timestamps, running/completed/failed/interrupted
  status, seed, device and package versions, git commit plus dirty boolean,
  effective config digests, upstream input paths/digests, and produced artifact
  path/digest/size. Never serialize environment variables or `.env` contents.
- A final containment audit over every artifact registered in the manifest.

Add `KWS_Model/outputs/` to the root `.gitignore` as the documented local
default. Keep existing legacy ignore patterns. An arbitrary directory name
inside the repository cannot be dynamically ignored, so recommend either the
ignored default or a path outside the checkout.

### 2. Durable console logging

Refactor `src/kws/utils/logging.py` so logger creation is separate from one-time
run configuration.

- Configure the root/project logger once per CLI invocation with terminal and
  UTF-8 file output; remove/close only handlers installed by KWS.
- Tee Python stdout and stderr to the invocation log so PAI prints, warnings,
  and tracebacks are captured as well as `logging` records. Install the tee
  before creating stream handlers so handlers do not retain the old stream.
- Append a clear invocation start/end record and flush on normal exit,
  `KeyboardInterrupt`, and error. Preserve normal terminal behavior.
- If distributed execution is later enabled, use rank-qualified logs and allow
  only rank zero to update shared manifests/metrics.

Wrap every public `main()` with the same run-session helper. The data downloader
uses this only for logs/manifest; its raw dataset path remains a shared cache.

### 3. Canonical metrics recorder

Add a recorder (in `artifacts.py` or a focused
`src/kws/utils/checkpointing.py`) and pass it into `run_finetune` and the PAI
loop. Each completed epoch writes one JSON object containing:

- schema version, stage, phase, epoch, global step, and elapsed seconds;
- train total loss, all named KD/component losses, and train accuracy;
- validation loss and validation accuracy;
- learning rate for every optimizer parameter group;
- effective seed and parameter count; and
- phase-specific fields such as PAI mode, restructure status, frozen/trainable
  base and dendrite counts, or QAT backend.

Enhance `run_finetune` to compute train accuracy and expose an `on_epoch_end`
hook. Keep its in-memory `FinetuneResult.history`, but return the full result
from `train_model` and distillation instead of throwing it away. During a
transition, a compatibility wrapper/property can still expose
`best_val_acc` to callers expecting a float.

JSONL commit protocol:

1. Append and `flush`/`fsync` the completed epoch record.
2. Atomically replace `latest.pt`, including the committed metric count/digest.
3. If this epoch is best, materialize/update `best.pt` atomically.

On resume, treat `latest.pt` as the commit point: validate the JSONL prefix and
truncate a trailing record that was appended before an interrupted checkpoint
replacement. If `best.pt` is missing or stale, regenerate it from best state
kept in `latest.pt`. Never duplicate an epoch record.

Write `metrics/summaries.yaml` atomically at phase/stage boundaries with best
and final metrics plus relative artifact references. Evaluation, export,
benchmark, pruning, PTQ, and standalone QAT must always write their returned
metrics/report under the active root even when the legacy `--report` option was
omitted.

TensorBoard support can be enabled by a config flag and mirror the canonical
records into `metrics/tensorboard/<stage>/<phase>/`; resume correctness must not
depend on TensorBoard event files.

### 4. Latest training checkpoint and resume contract

`latest.pt` is training state, not the deployment model. Give it a versioned
schema with at least:

```text
format_version, kind="kws_training_state", run_id, stage, phase
completed_epoch, next_epoch, global_step, target_epochs
model_state_dict, optimizer_state_dict, scheduler_state_dict
best_metric_name, best_metric_value, best_epoch, best_model_state_dict
history_length, last_metric_digest
python_rng_state, numpy_rng_state, torch_cpu_rng_state, torch_cuda_rng_states
effective data/model/train/KD config and recipe fingerprint
upstream checkpoint paths and SHA-256 digests
stage_specific_state
```

Stage-specific state includes the KD feature adapter and its optimizer,
N:M masks, clustered assignments/centroids, QAT backend and prepared fake-quant
observer buffers, and any secondary scheduler. Move optimizer tensors to the
selected device after loading.

Resume semantics:

- `epochs` is the total target, not an additional count. A checkpoint completed
  at epoch N continues with N+1 through the configured total.
- Add `--resume` (use this run's phase `latest.pt`) and
  `--resume-from PATH` (explicit training-state input). The pipeline should
  automatically resume an incomplete compatible stage in the same output root;
  `--resume` on standalone commands remains explicit.
- Load/rebuild the model and stage wrapper first, then optimizer/scheduler and
  stage-specific objects, then restore RNG immediately before creating/iterating
  the next training epoch.
- Reject changes to architecture, labels, data recipe, teacher contents, KD
  weights, optimizer/schedule, seed, pruning masks, cluster recipe, QAT backend,
  or PAI recipe. Offer a separately named warm-start/reset-optimizer path for
  intentional fine-tuning; do not call it resume.
- Legacy best-only checkpoints remain valid for inference and warm starts, but
  produce a clear "not a resumable training-state checkpoint" error when passed
  to `--resume-from`.
- Write a usable best checkpoint even when the first validation accuracy is
  exactly zero; initialize the best value to negative infinity or always accept
  the first epoch.
- Resume is initially supported at completed epoch boundaries. Mid-batch
  replay is out of scope.

The current module-global Python/Torch randomness and persistent DataLoader
workers prevent a bit-for-bit guarantee after restart. In v1, save all process
RNG and explicit DataLoader generator states and guarantee correct functional
continuation. Tests for numerical equivalence should use `num_workers=0`.
Document that multi-worker augmentation batches may differ after a restart.
A later exact-replay enhancement can make augmentation/silence randomness a
function of `(seed, epoch, sample index)` or restart workers from deterministic
per-epoch seeds without persistent worker state.

## PAI-specific containment and recovery

PAI cannot be treated like an ordinary fixed-graph `state_dict` phase.

1. Resolve the data root, configs, teacher/source checkpoints, and every KWS
   output path absolutely before starting a PAI candidate.
2. Create `<output-dir>/pai/candidates`, temporarily change the working
   directory to it, and pass only a validated leaf such as `w18` to
   `UPA.perforate_model`. Restore the original working directory in `finally`.
   Keep the absolute candidate directory in KWS reports/objects; do not confuse
   it with PAI's leaf filename prefix.
3. Keep PAI's native filenames intact inside `pai/candidates/w18/`, including
   its CSVs, figures, config, `best_model*`, `switch_*`, and native `latest.pt`.
   Do not rename vendor files that PAI's loader expects.
4. After every validation call, and again after a restructure plus optimizer
   reset, save a mutually consistent pair:
   - PAI network/tracker state using its supported `save_system` API; and
   - KWS state at
     `models/checkpoints/sparsity/w18/pai/latest.pt` with epoch, optimizers,
     schedulers, KD adapter, phase trail, metrics commit, RNG, and the native
     PAI checkpoint digest.
5. To recover a partial candidate, recreate the base wrapper, use PAI's
   `load_system(..., load_from_restart=True)` for the matching native `latest`,
   rebuild optimizers against the restored graph, load the KWS sidecar states,
   validate the paired digests/epoch, and continue. Saving must occur after any
   restructure-induced optimizer reset so parameter groups match on load.
6. Replace the current blanket rejection of a nonempty partial candidate with:
   compatible paired latest state -> resume; incomplete/mismatched state -> fail
   without overwriting; completed metadata -> existing completed-run reuse.
7. Add a contract test against the pinned PAI version. If network/tracker plus
   optimizer recovery cannot be demonstrated, explicitly limit PAI v1 recovery
   to the last consistent PAI phase/candidate boundary. Do not claim exact
   latest-epoch PAI resume until that test passes.

`load_candidate_model` currently creates an adjacent `<save_name>_reload`
directory. Route this under `pai/reload/` or use a temporary directory inside
the output root and register/remove it deliberately.

## File-by-file implementation plan

### Phase 1: Foundation

- Add `src/kws/utils/artifacts.py` and
  `src/kws/utils/checkpointing.py` for the layout, locking, manifest, atomic
  writes, metric commits, checkpoint schema, fingerprints, and RNG capture.
- Refactor `src/kws/utils/logging.py` for invocation configuration and
  stdout/stderr teeing.
- Extend `src/kws/utils/seed.py` with capture/restore helpers, feature-detecting
  CUDA/MPS support without assuming either backend exists.
- Extend `src/kws/data/loader.py` to accept explicit generators and expose the
  resume reproducibility limitation described above.
- Add focused foundation tests before integrating model stages.

### Phase 2: Shared training loop

- Update `src/kws/train.py` so `run_finetune` accepts prior state, starts at the
  next epoch, tracks train accuracy/LR/global step/duration, invokes the epoch
  recorder, and returns full results.
- Make `train_model` use standard best/latest paths and a reusable checkpoint
  metadata builder while preserving the legacy best checkpoint shape.
- Add output/resume CLI options to `kws.train` and persist a phase summary.
- Add synthetic CPU tests for metrics, best/latest behavior, atomicity,
  interruption, resume, and recipe mismatch.

### Phase 3: Fixed-graph optimization phases

- `src/kws/optimize/distill.py`: use the shared recorder; store/restore the KD
  adapter and its optimizer; return/persist the complete result.
- `src/kws/optimize/prune.py`: route best/latest/report paths and restore the
  exact structured graph or N:M masks before optimizer loading.
- `src/kws/optimize/cluster.py`: replace RAM-only best state with durable
  codebook training state; restore assignments, parametrizations, centroids,
  optimizer, and scheduler before resuming.
- `src/kws/optimize/quantize_qat.py`: save a resumable prepared fake-quant
  training graph separately from the final TorchScript artifact; restore
  observers/fake-quant buffers, backend, projector, optimizer, and scheduler.
- Add the common output/resume CLI options to all standalone modules while
  retaining legacy output options when no root is active.

### Phase 4: PAI cycle and sparsity search

- Update `src/kws/optimize/dendritic.py` to separate `pai_run_name` from the
  absolute `pai_run_dir`, contain PAI through the scoped working directory, emit
  canonical epoch metrics, and create/recover the paired PAI/KWS latest state.
- Apply the shared recorder to both pre-PAI prune/KD and post-PAI KD resume.
  Persist no-improvement histories too.
- Update `src/kws/optimize/dendritic_prune_loop.py` to derive candidate paths
  from `ArtifactLayout`, resume partial compatible candidates, atomically update
  the search summary, and retain completed candidate reuse/Pareto state.
- Increment the framework/checkpoint format versions so old completed evidence
  is not misrepresented as a fully resumable run.

### Phase 5: Pipeline, reports, evaluation, and exports

- Refactor `src/kws/pipeline.py` to construct one run context, derive every
  output from it, pass recorders into stages, and keep all recorded paths
  portable. Checkpoint the pipeline manifest/report before, during, and after
  each stage; stage invalidation must not delete unrelated run history.
- Update `src/kws/evaluate.py` to always persist full accuracy/F1/confusion/FAR/
  FRR metrics under an active root.
- Update `src/kws/export/to_onnx.py` to route the graph and persist parity plus
  source/digest metadata.
- Update `src/kws/export/benchmark.py` to route ONNX/TorchScript outputs and the
  full runtime accuracy, FAR/FRR, latency percentiles, parity, and graph-size
  report.
- Update `src/kws/optimize/quantize_ptq.py` to route both graphs and persist the
  returned accuracy/size metrics, which its CLI currently discards.
- Update `src/kws/data/download.py` to accept the common log/manifest option
  without moving the configured shared data root.

### Phase 6: Config, docs, ignore rules, and PAI Skills

- Add optional `output_dir: null` to `configs/train/pipeline.yaml`. Preserve the
  existing path fields as legacy mode and document which are output-role fields
  overridden/rebased in unified mode.
- Remove output ownership from
  `configs/train/dendritic_prune_loop.yaml` when a run context is supplied;
  `save_prefix` becomes a logical candidate prefix and `summary_path` a legacy
  fallback.
- Update `KWS_Model/README.md` with one `--output-dir` pipeline example, the
  directory tree, best-vs-latest semantics, stage reuse vs epoch resume,
  conflict/fingerprint errors, multi-worker reproducibility limits, and legacy
  behavior. Keep `BUG.md`, `FINDINGS.md`, and checked-in reports as historical
  evidence; do not rewrite their old paths.
- Add `KWS_Model/outputs/` to the repository `.gitignore` while preserving its
  environment/cache/data/model/report/dendritic ignore rules.
- The PAI Skills dashboard currently fixes runtime state under
  `.perforated_tools/`. Treat `.mcp.json`, installed skills, and the launcher as
  installation state outside the run-output promise. Add an artifact-root
  option/environment value for runtime `dashboard.log` and visualizer exports,
  mount that directory read-write, and update `dashboard-run.sh`, `install.sh`,
  `uninstall.sh`, the package README, `train-my-model`, `visualize-model`,
  `perforatedai-analyze`, and their shell tests to discover
  `<output-dir>/pai/...` without changing PAI-native filenames.

## Test plan

Add `tests/test_artifacts.py` and `tests/test_training_resume.py`, then extend
the existing pipeline, pruning, dendritic, cluster, QAT, export, and seed tests.

Required cases:

- Layout creates the documented tree, resolves paths containing spaces, rejects
  traversal/symlink escapes, and enforces the run lock.
- CLI/config precedence is identical across entrypoints; every legacy command
  still behaves as before with no output root.
- Logging captures KWS logs, plain stdout/stderr, warnings, a simulated PAI
  print, and an exception without duplicating handlers on repeated test calls.
- Every completed epoch produces one complete JSONL record. Values match the
  returned history; resume does not duplicate records and reconciles the
  append-before-checkpoint crash window. A malformed trailing JSONL fragment is
  removed without losing prior valid records, and a forced/fresh rerun cannot
  retain conflicting epoch history.
- `latest.pt` advances every epoch, including epochs with no validation
  improvement. `best.pt` changes only on improvement and exists after an
  all-zero first validation epoch.
- Every KD-backed phase completes at least one epoch, saves latest/best state,
  and resumes with identical criterion/adapter parameters and optimizer state.
  Include a regression test for the distillation best-checkpoint save call.
- A tiny deterministic CPU run for N epochs matches a run interrupted after K
  epochs and resumed to N (`num_workers=0`): model, optimizer, scheduler LR,
  global step, metric history, and next Python/NumPy/Torch random values.
- Resume rejects changes to model, labels/data, total schedule-critical config,
  seed, teacher digest, KD weights, pruning/cluster/QAT recipe, or PAI pairing.
- KD adapter state, N:M masks, codebook assignments/centroids, and prepared QAT
  observer/fake-quant state survive a split run.
- Old best-only checkpoints still load for inference/warm start and fail clearly
  as `--resume-from` inputs.
- A mocked lightweight pipeline creates no registered artifact outside
  `tmp_path`, resumes a partial stage, reuses completed stages from the same
  root, and preserves downstream invalidation/provenance behavior. Repeat an
  invocation that overwrites the same registered path and assert that manifest
  digest/size metadata is refreshed and finalization errors are visible.
- Every active-root CLI rejects absolute output-role paths and traversal before
  basename or slug normalization. Move a completed run directory and verify
  that pipeline/candidate reuse resolves all run-owned report paths relative to
  the new root.
- A mocked PAI test proves the library receives a basename while its working
  directory is `<root>/pai/candidates`, paired latest state is saved after
  restructure, partial runs call `load_system`, and completed runs still reuse.
  Interrupt immediately after restructure and verify that model, native PAI
  state, KD adapter, both optimizers, scheduler, metrics, and digests recover as
  one compatible pair. Also cover missing-sidecar and digest-mismatch rejection
  through the standalone PAI CLI.
- PAI Skills shell tests cover custom runtime log/export roots, Docker mount
  paths, uninstall behavior, and preservation of user artifacts.

Verification commands after implementation:

```bash
uv run python -m compileall src tests
uv run pytest tests/
sh "PAI Skills/test.sh"
```

Also run a tiny real or dependency-backed PAI interruption/resume contract test
before marking PAI latest-epoch recovery complete. Long Speech Commands training
is not required for the unit suite.

## Migration and safety

- Do not move, rename, or rewrite existing ignored checkpoints, reports, or
  `dendritic_*` directories automatically. Their reports contain literal paths
  and provenance fingerprints that can be invalidated by a move.
- A legacy artifact can be supplied as an external input/warm start. The new
  manifest records its absolute path and SHA-256; new outputs still remain under
  the selected root.
- Starting in a nonempty root without compatible resumable state must fail with
  an actionable message. `--force` may recompute a requested stage and update
  dependent report entries, but it must not recursively delete the run root.
- Temporary files must stay beside their destination and be cleaned only when
  their exact, validated path is known. Never follow a user path with a broad
  recursive deletion.
- Fingerprints should use effective config content, artifact content digests,
  and logical artifact roles, not the absolute output-root string, so a complete
  run directory remains portable.

## Definition of done

- A user can run the full pipeline with one output argument, inspect the
  documented tree, and find every KWS/PAI runtime artifact and console log under
  that root.
- Every training phase has durable epoch metrics and a distinct atomic
  `best.pt` and `latest.pt`.
- Killing a supported phase after epoch N and rerunning with the same compatible
  root continues at N+1 with restored model/optimizer/scheduler/stage/RNG state.
- Partial PAI recovery is demonstrated against the pinned library or is clearly
  and narrowly reported as phase-boundary-only; it is never silently restarted
  or overwritten.
- Evaluation, export, benchmark, PTQ, stage reports, and manifests persist their
  complete returned results automatically in unified mode.
- All new and existing Python tests plus the PAI Skills shell tests pass, legacy
  no-root commands remain compatible, and no test/run creates a project-owned
  artifact outside its selected temporary output root.

## Future ideas (not scheduled)

- **Turn step 3 into a true compounding prune+dendrite loop.** Today's step 3
  (`dendritic_prune_loop.py`) sweeps a precomputed list of descending base
  widths; each width is an *independent restart* pruned from the same original
  KD-distilled checkpoint, grows its own dendrites, and the sweep stops via a
  multi-axis Pareto frontier with patience (see the module docstring: "each
  sparsity target gets its own base network... PerforatedAI adds capacity; it
  does not structurally prune a learned dendritic network"). A requested
  alternative is a genuinely iterative loop that keeps pruning *and* growing
  dendrites on the same evolving model round over round -- prune the current
  best (already-dendritic) model further, grow new dendrites on top, resume
  KD, and repeat until validation performance stalls or degrades for N
  consecutive rounds -- rather than independently restarting from the original
  checkpoint at each width.
  - This directly conflicts with the documented constraint that PAI cannot
    structurally re-prune a network it has already grown dendrites onto, so
    doing this properly needs a real answer first: either (a) collapse/merge
    trained dendrites back into the base weights before the next prune pass,
    or (b) only ever prune the base graph and discard-and-regrow dendrites
    each round (losing prior dendrite training), or (c) some other scheme PAI
    supports. Whichever is chosen touches `dendritic.py`'s PAI integration
    assumptions, not just the sweep driver.
  - A smaller, lower-risk version of this idea keeps today's per-width restart
    architecture (still valid, still sidesteps the re-pruning problem) but
    replaces the multi-axis Pareto-frontier/patience stop with a plain "stop
    when best validation accuracy stalls or drops for N consecutive
    candidates" rule, dropping the cost-axis frontier logic in
    `pareto.py`/`ParetoSearch` in favor of a single-metric criterion.
  - Raised 2026-09-12; deferred pending a decision on which of the above
    (compounding-model rewrite vs. simplified single-metric stop vs. leave as
    is and just document today's sweep as "iterative until stalling") is
    wanted, since each has different blast radius across `dendritic.py`,
    `dendritic_prune_loop.py`, `pareto.py`, configs, README, and tests.
