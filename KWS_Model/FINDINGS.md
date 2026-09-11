# Compression Framework Implementation Review

**Date:** 2026-09-11  
**Branch:** `KWS_Model`  
**Base commit:** `4670f0e`

## Scope and method

This is the current, consolidated review of all changed and added files for the
six-step compression framework. It supersedes the older findings that were
previously appended to this file; those findings mixed pre-fix defects with later
fix confirmations and were no longer an accurate statement of the repository.

The review covered the pipeline, pruning/dendritic search, Pareto logic, knowledge
distillation, clustering, quantization-aware training, profiling, export,
benchmarking, configuration, documentation, and tests.

Validation performed for this updated review:

- Static trace through every framework stage and its artifact hand-offs.
- Focused synthetic tests covering sparsity, orchestration, provenance, codebook
  packing, and quantization/export.
- Full suite: `121 passed` with 10 PyTorch quantization deprecation warnings.
- `git diff --check`: passed.

No full training sweep, QAT run on the real dataset, or ESP32/MRAM hardware benchmark
was performed. Findings about those paths are based on code and artifact inspection.

## Implementation status after this review

The actionable host-side findings were implemented and regression-tested:

- F1/F2: Pareto progress now uses configurable accuracy/cost tolerances, treats
  equivalent points as stalls, and records whether patience was spent on
  inadmissible accuracy or true frontier stagnation.
- F5/F8/F9: downstream sparsity fingerprints, source/teacher/content digests,
  complete student recipe fingerprints, and transitive stage invalidation are
  enforced. Upstream reruns clear dependent report stages.
- F6: the pruned base is moved to the selected device before pre-dendrite KD.
- F7: deployment reports label projected weight precision and use a conservative
  branch-liveness estimate for residual graphs. Exact target allocator accounting
  is still not available on the host.
- F3: packed codebook/index artifacts are now retained beside the host graph and
  validated on reuse. The compatibility TorchScript graph still contains dense
  weights; a target-native codebook operator remains future work.
- F4/F10 remain bounded: no physical ESP32/MRAM runner exists in this repository,
  and the tests are comprehensive component/invalidation tests rather than a
  real-data six-stage training sweep.

## Overall verdict

The repository now implements the requested framework's **overall stage structure**,
and several earlier critical defects have been corrected. It is not yet a complete
end-to-end implementation of the intended deployment method, however. The largest
remaining gaps are:

1. A target-native codebook operator/export is still absent; the host graph remains
   dense for compatibility while a packed sidecar is retained.
2. The exact exported graph is benchmarked on a host runtime, not on the intended
   ESP32 + MRAM deployment runtime.
3. Memory accounting remains a host-side estimate rather than target allocator data.

| Framework step | Status | Assessment |
|---|---:|---|
| 1. Train full-capacity teacher | Implemented | Teacher training and reuse exist. |
| 2. Distill deployment-shaped student | Implemented | Fixed-teacher KD and complete recipe provenance are enforced. |
| 3a. Structured pruning | Implemented | Surviving-input channel ranking and pruned checkpoint handling are present. |
| 3b. KD fine-tune after pruning | Implemented | The pruned graph is placed on the selected device before training. |
| 3c. Add/train/select/freeze dendrites | Implemented | PAI branch interpretation and frozen-dendrite handling are now consistent. |
| 3d. Resume KD on active parameters | Implemented | The fixed teacher and active-parameter filtering are wired through. |
| 3e. Record accuracy/latency/memory/compute | Partial | Metrics are labeled, but target allocator accounting is unavailable on the host. |
| 3f. Stop on Pareto stagnation | Implemented | Material tolerances, duplicate handling, and stop-cause reporting are present. |
| 4. Layer-wise clustering/codebooks | Implemented with host boundary | Packed codebook/index artifacts are retained; host QAT uses a dense compatibility graph. |
| 5. QAT distillation | Implemented | Task loss, KD loss, and fake-quantized forward training are combined. |
| 6. Export and benchmark exact graph | Host implementation only | The exact converted TorchScript graph is measured on CPU; target-device execution is absent. |

## Current findings

### F1 — Resolved in host implementation: Pareto stopping collapses to admissibility stopping in the normal width sweep

The frontier implementation is internally coherent, but it does not provide the
requested stopping behavior for a decreasing-width search. A narrower accepted
candidate normally improves at least the `deployed_params` objective. An earlier,
wider point therefore cannot dominate it, so each accuracy-admissible candidate is
treated as a frontier extension. Patience is generally consumed only after a
candidate fails the accuracy floor or maximum-drop check.

Consequently, the run may report `pareto_frontier_stalled` even though the actual
stopping signal was consecutive accuracy-inadmissible candidates. That is materially
different from detecting that the multi-objective frontier stopped improving.

Evidence:

- [`dendritic_prune_loop.py`](src/kws/optimize/dendritic_prune_loop.py#L446) sweeps
  widths sequentially and adds only accepted candidates to the frontier.
- [`pareto.py`](src/kws/optimize/pareto.py#L138) calls any non-dominated candidate an
  extension.
- [`dendritic_prune_loop.py`](src/kws/optimize/dendritic_prune_loop.py#L510) separately
  consumes patience for accuracy-inadmissible candidates.

Implemented correction: `ParetoFrontier` supports configurable material-progress
tolerances, the normal search config uses them, and the search summary records the
actual stop cause separately from accuracy admissibility.

### F2 — Resolved: exact duplicate points reset Pareto patience

Dominance requires at least one strict improvement. An exact duplicate is therefore
not dominated, and `add()` marks it as extending the frontier. Reused or rounded
measurements can reset the stagnation counter without improving any objective.

Evidence: [`pareto.py`](src/kws/optimize/pareto.py#L51) and
[`pareto.py`](src/kws/optimize/pareto.py#L149).

Implemented correction: equality within configured accuracy/cost tolerances is
treated as a non-extending candidate and consumes patience.

### F3 — Partially resolved: codebook compression is not preserved in the host graph

Stage 4 learns per-layer assignments and centroids, and the QAT loop reprojects weights
onto those assignments. At the stage boundary, `bake_codebooks()` still removes the
parametrizations and materializes ordinary dense tensors for host QAT. Stage 5 now
also saves packed centroid tables plus 4-bit indices as a `.codebook.pt` payload beside
the dense TorchScript compatibility graph. No target runtime consumes that payload yet.

The host graph remains dense, but the deployment-oriented compressed weight payload
now realizes the configured 4-bit codebook/index representation.

Evidence:

- [`cluster.py`](src/kws/optimize/cluster.py) retains packed codebooks while still
  folding them into dense weights for host QAT.
- [`quantize_qat.py`](src/kws/optimize/quantize_qat.py#L286) converts and serializes
  the resulting dense int8 graph.
- [`pipeline.yaml`](configs/train/pipeline.yaml#L42) describes 4-bit weight indices;
  the stage-5 sidecar now stores them for a target adapter.

Remaining correction: add a codebook-aware operator/export consumed by the target
runtime, including centroid tables and packed indices. The TorchScript file is
explicitly a host compatibility graph; the sidecar is the deployment-oriented
compressed payload.

### F4 — Open: step 6 does not benchmark the intended ESP32 + MRAM runtime

The pipeline correctly reloads and benchmarks the exact converted TorchScript artifact,
which fixes the earlier wrong-graph problem. The benchmark still runs on host CPU using
TorchScript/qnnpack (or ONNX Runtime for float-compatible paths). It does not execute
the ESP32 kernel implementation, external-MRAM access pattern, allocator, or firmware
graph that determines real latency and peak memory.

Evidence:

- [`pipeline.yaml`](configs/train/pipeline.yaml#L50) explicitly calls the current
  benchmark a host proxy.
- [`benchmark.py`](src/kws/export/benchmark.py#L266) forces the scripted model to CPU
  and reports `torchscript-cpu`.

Recommended correction: export the actual target-runtime graph and add a device-side
benchmark harness. Keep host measurements as CI smoke tests, labeled as proxies.

### F5 — Resolved in host implementation: later-stage-only runs can trust a stale step-3 report

Before the fix, `stage_sparsity()` validated a search fingerprint only when it was
invoked. A later invocation that requested only clustering, quantization, or
benchmarking could load `3_sparsity` from the existing pipeline report without checking
the current data, teacher, student, train, and search configuration. Candidate
reconstruction also lacked full artifact content-digest checks.

Evidence: [`pipeline.py`](src/kws/pipeline.py#L575),
[`pipeline.py`](src/kws/pipeline.py#L599), and
[`pipeline.py`](src/kws/pipeline.py#L326).

Implemented correction: downstream-only invocations recompute the step-3 fingerprint;
candidate metadata records and validates source, teacher, and clean-artifact content
digests before reconstruction.

### F6 — Resolved: pruned KD fine-tuning has a non-CPU device mismatch

Before the fix, `run_cycle()` chose CUDA/MPS when available while
`build_cycle_base()` returned a CPU model. The pre-dendrite KD phase called
`train_model(base, ..., device)` before moving `base` to that device, so the first
forward pass failed when the selected device was not CPU.

Evidence: [`dendritic.py`](src/kws/optimize/dendritic.py#L692),
[`dendritic.py`](src/kws/optimize/dendritic.py#L733), and
[`train.py`](src/kws/train.py#L152).

Implemented correction: `run_cycle()` moves the pruned base to `device` before the
pre-KD call. The device-neutral construction boundary remains explicit.

### F7 — Partially resolved: memory and compute metrics are partly projected and undercount residual liveness

Step 3 profiles a float graph while its configuration requests 8 bits per weight. This
produces a projected weight-memory number alongside measured float latency and compute.
The current profiler now uses a conservative branch-liveness sum for dendritic residual
wrappers, but it remains an estimate rather than a target allocator trace.

Evidence:

- [`dendritic_cycle1.yaml`](configs/train/dendritic_cycle1.yaml#L47) requests projected
  8-bit deployment cost for the float cycle graph.
- [`dendritic.py`](src/kws/optimize/dendritic.py#L919) records that profile in step 3e.
- [`profile.py`](src/kws/utils/profile.py#L102) uses the largest adjacent leaf pair.
- [`quantize_qat.py`](src/kws/optimize/quantize_qat.py#L110) retains residual branch
  outputs and performs accumulation after dequantization.

Implemented correction: reports label projected weight precision and distinguish
sequential from conservative branch-liveness estimates. Exact runtime-arena
measurement remains target-specific work.

### F8 — Resolved: upstream reruns can leave a stale stage-6 report

Before the fix, clustering or quantization reruns could leave a stale stage-6 result in
the report. The report could therefore show a benchmark for an artifact that was no
longer the current stage-5 output.

Evidence: [`pipeline.py`](src/kws/pipeline.py#L645) and
[`pipeline.py`](src/kws/pipeline.py#L701).

Implemented correction: cluster changes clear stages 5 and 6 transitively, and a
changed quantization artifact clears stage 6. Teacher, student, and sparsity changes
also invalidate all dependent stages.

### F9 — Resolved: student checkpoint reuse does not fingerprint the complete recipe

Before the fix, student reuse checked the model configuration, teacher path, and teacher
SHA-256 but not the data configuration, distillation/training configuration, random
seed, or warm-start checkpoint. Changing those inputs could silently reuse a student
produced by a different recipe.

Evidence: [`pipeline.py`](src/kws/pipeline.py#L174).

Implemented correction: stage 2 persists and validates a canonical recipe fingerprint
covering model/data/train/KD settings plus teacher and warm-start content hashes.

### F10 — Partially resolved: orchestration coverage stops short of an end-to-end stage hand-off test

The test suite has good unit and component coverage, including synthetic QAT conversion
and artifact checks. It does not execute a lightweight six-stage pipeline that proves
report invalidation, later-stage resume, exact artifact selection, and recipe changes
work together. This is why the stale-report and device-placement paths can survive while
all tests pass.

Implemented correction: explicit regression tests now cover stale fingerprints,
duplicate/tolerant Pareto points, packed codebook payloads, recipe changes, and
component hand-offs. A tiny six-stage stubbed run and a real-data sweep remain useful
future coverage.

## Operational migration note

The checked-in distilled student checkpoint predates the new teacher-digest metadata.
Its distillation metadata names the teacher but has no `teacher_sha256`. With the current
reuse validation, the default pipeline will reject it and request re-distillation with
`--force`. Either regenerate that checkpoint once or provide an explicit artifact
migration path; document this requirement so a clean checkout does not appear broken.

## Verified improvements from the earlier audit

The following previously reported defects are corrected in the current code and should
not remain listed as open issues:

- The PAI clean-graph branch convention is handled correctly: selected dendrites are
  `layer_array[:-1]`, and the base branch is last.
- Selected dendrites and skip coefficients are frozen while active base parameters are
  available for resumed KD fine-tuning.
- The post-pruning KD phase imports and calls the shared training path.
- Structured pruning ranks output channels from the surviving input-channel slice.
- Learning-rate schedules are sized for the actual phase length.
- QAT combines task loss, fixed-teacher KD, fake quantization, and codebook projection.
- Conv/BatchNorm preparation and the quantization-safe residual wrapper convert in the
  supported host backend.
- Stage 6 loads the exact saved TorchScript int8 artifact instead of benchmarking a
  float or fake-quantized surrogate.
- Clustered and quantized artifacts carry candidate/recipe provenance checks for the
  paths that load them.

## Recommended implementation order

1. Add a target-native codebook operator/export that consumes the packed sidecar.
2. Add the real target-runtime export/benchmark and exact memory accounting.
3. Add a lightweight six-stage orchestration smoke test and regenerate legacy
   checkpoints with the current provenance metadata.

## Final assessment

The repository is now a credible **host-side framework prototype** with explicit
provenance, material Pareto stopping, packed codebook payload retention, and complete
host-side regression coverage. It is not yet a completed target-deployment
implementation: F3's target operator, F4's ESP32/MRAM benchmark, exact target memory
accounting, and a full six-stage orchestration smoke test remain open.
