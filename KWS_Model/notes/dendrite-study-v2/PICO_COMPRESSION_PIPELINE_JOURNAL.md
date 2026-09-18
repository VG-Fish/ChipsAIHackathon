# RP2040 KWS compression pipeline: research and decision journal

> **DS-CNN extension (2026-09-18):** See
> [`DSCNN_DENDRITIC_PIPELINES.md`](DSCNN_DENDRITIC_PIPELINES.md) for five
> DS-CNN-specific dendritic pipelines, exact budget calculations, matched
> controls, and the recommended run order.

**Last updated:** 2026-09-18  
**Status:** design and evidence audit complete; implementation and Pico measurements not yet started  
**Target:** 12-class Google Speech Commands v2 KWS on an RP2040 / Raspberry Pi
Pico-class device, deployed through an integer TensorFlow Lite Micro graph

This is the continuation document for the compression project. It records the
repository evidence, external research, five candidate pipelines, experiment gates,
and the exact next work. Three focused supporting audits contain the lower-level
evidence:

- [`agent-repo-results-audit.md`](agent-repo-results-audit.md): all local runs,
  PerforatedAI result CSVs, KD runs, failures, and current study state.
- [`agent-sparknet-pico-deployment-research.md`](agent-sparknet-pico-deployment-research.md):
  SparkNet, RP2040, TFLM/CMSIS-NN, quantization, and ExecuTorch assessment.
- [`agent-perforatedai-primary-research.md`](agent-perforatedai-primary-research.md):
  official PerforatedAI site, documentation, and GitHub behavior.

## 1. Decision summary

The best route to a demonstrably better Pico KWS artifact is **not** to run the
proposed full pruning -> KD -> dendrite -> KD loop immediately. The current evidence
supports this sequence:

1. Deploy and measure the existing SparkNet C12, C10, C8, and C16 checkpoints as
   fully integer TFLite Micro models. This establishes the first real Pico frontier.
2. Run same-family structured pruning and KD from a paper-faithful SparkNet C16
   teacher into C12/C10/C8 students. Compare to the already strong scratch-width
   controls.
3. Search non-uniform SparkNet channel/gate configurations using the exported int8
   graph and Pico cost, rather than parameter count alone.
4. Only then re-test one corrected PerforatedAI configuration: one `fc` dendrite,
   no destructive identity fine-tune, matched controls, and comparisons at equal
   **total deployed** parameter cost.
5. Admit dendrites into the iterative compression loop only if step 4 first extends
   a validation Pareto frontier across multiple seeds.

The existing dendritic study is valuable negative evidence. It does not show a
dendritic improvement, but it reveals a repairable training handoff problem and a
clear preferred PAI placement (`fc`). The right project claim today is:

> We are testing whether corrected, budget-matched dendritic recovery can extend the
> accuracy-versus-device-cost frontier after conventional compression.

It is not yet defensible to claim that dendrites compress SparkNet or improve Pico
KWS.

## 2. Target and scope corrections

### Actual RP2040 target

The RP2040 has two Cortex-M0+ cores at 133 MHz officially and 264 kB SRAM split
across six banks. It has no internal flash; Pico boards normally provide external
QSPI flash. The baseline measurements should use 133 MHz and a declared single- or
dual-core policy. Any 200 MHz result is an explicitly labeled overclock result, not
the baseline.

The proposal's phrase “Cortex-M0-class, 8–32 kB SRAM” describes a broader extreme
TinyML tier, **not the RP2040**. Keep these as separate targets:

- **Pico tier:** total firmware data must fit 264 kB, with a recommended initial
  engineering guard band of at least 64 kB. A useful first limit is at most 200 kB
  for arena + frontend/audio buffers + application mutable state + stack.
- **Extreme-MCU research tier:** a separately declared 32 kB total or tensor-arena
  limit. Achieving it will likely require streaming audio/features rather than a
  one-second 32 kB int16 waveform buffer.

SparkNet's weights are not likely to be the Pico SRAM bottleneck. C12 has only 3,400
parameters and C16 has 4,636; int8 weights are approximately 3.4 and 4.6 kB before
metadata. The tensor arena, activations, frontend scratch, audio buffering, stack,
and runtime dominate the deployment question. Therefore parameter count is a useful
diagnostic but cannot be the primary definition of compression success on RP2040.

### Required deliverable

The project succeeds only when the **exact quantized artifact** runs on a Pico and
extends a measured frontier. Host PyTorch/ONNX latency, projected 8-bit weight bytes,
and MAC estimates remain preflight data.

## 3. What the repository already proves

### 3.1 Current SparkNet scratch frontier

The fresh read-only aggregation of
`outputs/sparknet-dendritic-study-v2` found all 30 scratch cells and all 150 arm
cells complete. The committed `aggregate_study.json` is a stale intermediate
snapshot; use the underlying output artifacts or regenerate an aggregate to `/tmp`.

Paper-faithful MFCC-32 validation results, five seeds:

| Width | Parameters | MACs | Best validation | Held-out test |
|---:|---:|---:|---:|---:|
| C12 | 3,400 | 277,124 | 93.88% ± 0.46 | 93.73% ± 0.17 |
| C10 | 2,854 | 224,806 | 92.80% ± 0.33 | 92.51% ± 0.40 |
| C8 | 2,356 | 177,336 | 91.23% ± 0.32 | 90.80% ± 0.36 |
| C6 | 1,906 | 134,714 | 87.87% ± 0.68 | 86.88% ± 0.54 |
| C4 | 1,504 | 96,940 | 82.87% ± 0.86 | 82.25% ± 0.93 |
| C2 | 1,150 | 64,014 | 63.34% ± 1.49 | 61.53% ± 1.31 |

SparkNet C12 is the strongest fully replicated local small point. C10 and C8 are
credible lower-cost points. C6 and below lose accuracy sharply and are useful mainly
as stress tests.

The paper reports C16 at 4,636 parameters, 454.5 kMACs, and 95.7% ± 0.17 SC2 test
accuracy. The local C16 paper replication is about 94.81% ± 0.22 test. Protocol and
checkpoint provenance explain why the paper number must remain a reference rather
than being silently substituted for a local result.

### 3.2 Completed dendritic study outcome

All four one-dendrite placements ended below their paired scratch checkpoints at
every width. Mean final validation delta versus scratch, C12 -> C2:

| Placement | C12 | C10 | C8 | C6 | C4 | C2 |
|---|---:|---:|---:|---:|---:|---:|
| pointwise | -0.40 pp | -0.54 | -0.50 | -0.50 | -0.81 | -2.42 |
| `fc` | -0.36 pp | -0.52 | -0.50 | -0.49 | -0.76 | -2.41 |
| gate convolution | -0.41 pp | -0.54 | -0.50 | -0.51 | -0.83 | -2.38 |
| depthwise | -0.38 pp | -0.53 | -0.48 | -0.50 | -0.85 | -2.40 |
| no-dendrite control | -0.19 pp | -0.17 | -0.34 | -0.26 | -0.51 | -2.02 |

The validation selector chose the no-dendrite control at all six widths. Only
scratch and selected controls were taken to held-out test; **no dendritic model has
a held-out test result**. The controls also lose to scratch, confirming that the
post-checkpoint training handoff itself is harmful.

At C12, the deployed costs were:

| Candidate | Parameters | MACs | Mean validation |
|---|---:|---:|---:|
| scratch | 3,400 | 277,124 | 93.88% |
| control | 3,400 | 277,124 | 93.69% |
| pointwise dendrite | 3,712 | 308,636 | 93.48% |
| `fc` dendrite | 3,808 | 277,520 | 93.51% |
| gate-conv dendrite | 3,848 | 319,140 | 93.47% |
| depthwise dendrite | 4,000 | 337,724 | 93.49% |

Every dendritic point is dominated by the C12 scratch point on validation,
parameters, and MACs. The same conclusion holds when compared with an interpolated
scratch-width curve at equal parameter budget. A dendrite adds a copy of a selected
module and combination weights; it is architecture growth, not pruning.

### 3.3 Why the current PAI run failed to establish a benefit

The starting scratch model was trained with SGD, momentum 0.9, learning rate 0.01,
weight decay 1e-3, task loss scaled by 100, and the paper schedule. The identity
fine-tune then changed to AdamW, learning rate 1e-3, weight decay 1e-4, label
smoothing 0.1, and a different schedule. Across all 30 pointwise cells this identity
handoff averaged -0.619 pp and was negative in 28/30 cells. The later dendrite phase
averaged another -0.243 pp and was positive in only 4/30 cells.

The integration itself was largely sound: requested modules were wrapped, p-mode
froze base optimizer parameters, BN statistics were frozen, optimizers were rebuilt
after restructuring, device placement was restored, and fixed clean graphs were
exported. Remaining integration risks include nonzero AdamW decay despite vendor
cautions, reliance on output-dimension defaults, insufficient permanent coverage of
the unwrapped-module escape hatch, and block grouping/BN adjacency differences.

PerforatedAI result analysis found:

- 117/120 dendritic cells integrated one dendrite; three integrated none.
- 96 cells ended the resume stage with `no_improvement`.
- `fc` is the only placement with a consistently strong PB correlation signal:
  final best-ever PB score about 0.132, versus roughly 0.006–0.010 for convolutional
  placements.
- A positive internal “best architecture vs minimum row” value is not a causal
  control. It compares rows inside one PAI search and the differences are usually
  fractions of a percentage point.
- There are no dedicated PAI train-score CSVs, so the expected train/validation
  overfitting check cannot be completed from that artifact family.

The correct conclusion is that this **recipe** failed. The result does not prove
that dendrites intrinsically hurt, but it strongly rejects running the same recipe
again.

### 3.4 KD evidence

The existing strong 40-bin log-mel DS-CNN-L teacher reaches about 97.69%, but the
MFCC-32 DS-CNN-L teacher reaches only 91.76% validation and is weaker than the C12
student. Three existing SparkNet C12 KD comparisons produced 0.00, -0.10, and
-1.37 pp changes. Diagnostics show that the strong-teacher KD path generated
reasonable teacher confidences, so simple implementation failure is not the main
explanation.

KD is therefore unproven locally. The next KD test must use a teacher that shares
the exact student frontend and split. A local paper-faithful SparkNet C16 is the
lowest-confound first teacher; a newly trained stronger MFCC teacher can follow.
C8 offers more plausible recovery headroom than the already strong C12.

### 3.5 Deployment evidence and gaps

The repository currently supports FP32 ONNX export and host benchmarking. Its QAT
path produces host PyTorch int8/TorchScript artifacts through qnnpack/fbgemm. It has
no TFLite/TFLM exporter, Pico firmware, target tensor-arena measurement, device
latency, or power result.

Other apparent compression mechanisms are not yet deployment-real:

- N:M sparsity is a logical mask. Standard dense TFLM kernels on Cortex-M0+ will
  not skip those zeros.
- Weight clustering writes a compact sidecar, but the current runtime does not
  decode or consume it; the compatibility graph stays dense.
- SparkNet's learned sparse activation/gate representation does not automatically
  reduce dense convolution work on TFLM.

Use structured channel changes that alter the exported tensor shapes. Defer N:M and
codebooks until a target kernel or decoder actually consumes them.

## 4. External research conclusions

### SparkNet

SparkNet applies four time-channel-separable blocks with temporal kernels
11/15/19/29, residual connections in the last three blocks, a learned gate
convolution with BN and tanh, temporal averaging, and a linear 12-class head. The
paper's C16 point is compelling because it reports roughly BC-ResNet-0.625 accuracy
at about one quarter of its MAC count. It reports host model metrics, not MCU
latency, quantized accuracy, or memory.

The paper recipe uses official GSC v2 splits, 32 MFCCs, time shift and noise
augmentation, SGD with momentum, weight decay 1e-3, task-loss scale 100, and 200
epochs. The local paper-faithful replication should remain the primary experimental
base because it already has five seeds and held-out test data.

### PerforatedAI

Official open-source PAI wraps Conv/Linear modules, trains candidate dendrites, and
integrates accepted fixed branches. It requires validation feedback, model/device
rebinding, and optimizer reconstruction after a structural change. It documents a
Python/PyTorch checkpoint and cleanup workflow, but no official TFLite, TFLM, ONNX,
or quantization contract.

The public GitHub package is not the commercial Perforated Backpropagation system
behind the vendor's headline compression claims. The local study used commercial
`perforatedbp` through `perforatedai 3.2.8`; open-source code should not be assumed
to reproduce proprietary PB scoring or the vendor's 800-run KWS claims.

The cleaned fixed graph may be exportable, but this must be verified operator by
operator. Keep the native PAI checkpoint and the clean fixed graph, then prove
logit/accuracy parity before quantization.

### TFLM, CMSIS-NN, Edge Impulse, and ExecuTorch

TFLite Micro is the recommended first runtime. It has a static arena, a constrained
operator set, upstream Micro Speech examples, and a Raspberry Pi Pico port. Pin
exact TFLM, Pico SDK, and compiler revisions because the Raspberry Pi generated port
is read-only/best effort.

CMSIS-NN can provide TFLM-compatible kernels, but the RP2040's Cortex-M0+ lacks the
DSP/MVE instructions responsible for many advertised Cortex-M speedups. Reference
and pure-C kernel performance must be measured; do not assume Cortex-M4/M55 results.

The supplied Edge Impulse ExecuTorch article is not evidence for RP2040 deployment.
It states that its current Python harness is not a microcontroller runtime, `.pte`
excludes MFCC/DSP, and a C++ executor is future work. Current official ExecuTorch
Cortex-M examples favor newer MVE-capable cores. Do not use ExecuTorch as the Pico
runtime now.

Edge Impulse's mature EON route is useful as an **optional second compiler** after a
valid int8 `.tflite` artifact exists. It may reduce interpreter/code overhead. Compare
it with stock TFLM; do not make it the only reproducible deployment path.

A direct PyTorch-to-LiteRT/TFLite spike using Google's `litert-torch` is preferable
to an assumed PyTorch -> ONNX -> TensorFlow chain. If direct conversion cannot lower
SparkNet's fixed graph to TFLM-supported integer operators, reimplement the small
fixed inference graph in Keras/TensorFlow and transfer weights. ONNX-to-TensorFlow is
a fallback, not the default architecture.

## 5. Pareto contract

### 5.1 Artifact metrics

Every candidate row must contain:

| Category | Required measurements |
|---|---|
| Quality | quantized held-out test accuracy, FAR, FRR, per-class F1/confusion matrix, noisy-SNR evaluation |
| Exact storage | `.tflite`/C-array bytes, deployed parameters, linked firmware flash delta, biases/scales/metadata |
| Mutable memory | TFLM arena high-water mark, frontend/audio buffers, static RAM, stack high-water mark, total peak SRAM |
| Compute | exported-graph MACs plus operator breakdown; no sparsity discount without a sparse kernel |
| Runtime | warm model-only and end-to-end p50/p90/p99 cycles or microseconds at 133 MHz, declared core mode |
| Quantization | FP32-to-int8 accuracy delta, PTQ/QAT method, representative set, tensor dtypes/scales, float fallbacks |
| Energy | joules or microjoules per decision when measurement hardware is available |
| Reproduction | dataset/split hash, checkpoint hash, firmware and runtime revisions, toolchain, build command, binary hash |

The first model-only latency target should be p99 below 10 ms if inference is issued
on every 10 ms feature hop. If the product uses a 20 ms or longer inference cadence,
declare that cadence and set the deadline before running comparisons. Measure both
model-only and full microphone -> frontend -> decision latency.

### 5.2 Dominance and acceptance

Candidate A dominates B when A is no worse on every required objective and strictly
better on at least one. The main frontier axes should be:

1. quantized held-out accuracy/error,
2. total peak SRAM,
3. firmware flash,
4. p99 latency,
5. energy per decision when available.

Parameters and MACs are secondary diagnostics. FAR/FRR are feasibility constraints,
not numbers that may silently regress in exchange for headline accuracy.

For a quick screen, use a predeclared accuracy tolerance of 0.2 percentage points.
For a final claim, retain seed-wise rows and use at least five independent seeds,
mean, sample standard deviation, and a paired confidence interval or paired test.
Promote a training method only if it beats its matched control at the same **total
deployed** cost; comparing a narrow base before counting dendrites is invalid.

## 6. Five plausible pipelines

### Pipeline 1 — Native SparkNet width frontier + full-int8 deployment

**Priority:** 1; recommended shipping baseline  
**Risk:** low  
**Research value:** establishes the first honest Pico frontier

```text
paper-faithful C16/C12/C10/C8
  -> freeze checkpoint
  -> fold BN / fixed inference graph
  -> full-int8 PTQ
  -> QAT only if PTQ loses >0.2 pp
  -> .tflite operator/parity audit
  -> TFLM Pico benchmark
  -> optional EON comparison
```

Rationale: the scratch-width frontier is the only strong, replicated local result.
It is small enough that direct width selection may beat a complicated compression
stack. Start with existing checkpoints, so the first export/deployment experiment
requires no new training.

Implementation details:

- Calibrate on 100–300 representative MFCC tensors from training only.
- Require int8 inputs, weights, activations, and outputs unless a fallback is
  explicitly labeled. Audit `tanh`, residual adds, `(1,K)` depthwise Conv2d, temporal
  mean, and Linear lowering.
- Fold BatchNorm and compare FP32 PyTorch, clean fixed graph, host `.tflite`, and Pico
  outputs on a fixed parity bundle.
- Benchmark C12 and C16 first. Add C10/C8 if C12 misses a target or to populate the
  frontier. C4 is a feasibility floor, not a useful accuracy target.
- Quantize the feature frontend consistently. Model-only inference on precomputed
  float MFCCs is a smoke test, not the final system.

Screen success:

- host int8 test loss no more than 0.2 pp versus the same FP32 checkpoint;
- integer-only TFLM graph compiles and agrees on fixed inputs;
- exact Pico measurements fit the declared SRAM and latency budgets.

Expected contribution: a reliable baseline and potentially a deployable prototype,
but not a novel dendritic compression claim.

### Pipeline 2 — Same-family structured pruning + fixed-teacher KD + QAT

**Priority:** 2; best conventional compression experiment  
**Risk:** medium  
**Research value:** tests whether iterative recovery beats training the same width
from scratch

```text
fixed paper-faithful SparkNet C16 teacher
  -> structured channel reduction C14
  -> short matched recovery, with/without KD
  -> C12 -> C10 -> C8 as separate nested candidates
  -> full-int8 QAT for promoted candidates
  -> TFLM Pico benchmark
```

Use a same-frontend C16 teacher first. This removes the current DS-CNN/MFCC mismatch.
Rank removable channels using a predeclared structured rule such as BN scale or
filter norm, but physically rebuild smaller tensors so exported dimensions and dense
kernel work shrink. At every width compare:

1. scratch-width baseline,
2. pruned + ordinary recovery,
3. pruned + KD recovery.

Preserve the source model's optimizer, loss scale, augmentation, and scheduler
semantics during continuation. A safe first KD objective is
`CE + alpha * T^2 * KL(student/T, teacher/T)` with a small grid over `alpha` and
`T`, selected on validation only. Do not use the weak MFCC DS-CNN teacher.

Time-box: 15–25 recovery epochs per candidate, stopping early on a fixed validation
patience. Current scratch runs take about 10–11 minutes on the available local
system, so one candidate fits the 20–30 minute ceiling. Run one seed for screening;
promote only winners to five seeds.

Kill the KD branch if it fails to improve the paired no-KD candidate by at least
0.2 pp across three screening seeds. A pruned C10 must also beat or dominate the C10
scratch checkpoint; beating only its immediately post-prune state is insufficient.

### Pipeline 3 — Deployment-aware non-uniform SparkNet search

**Priority:** 3; likely stronger than uniform width scaling  
**Risk:** medium  
**Research value:** hardware-aware architecture compression without assuming pruning
or dendrites help

```text
search block widths + gate width + selected kernel choices
  -> short successive-halving training
  -> int8 export feasibility and cost model
  -> Pareto select
  -> full paper schedule for finalists
  -> TFLM Pico benchmark and latency-LUT correction
```

The current C2/C4/.../C16 family uses one width everywhere. That is convenient but
unlikely to be optimal. Search a bounded space such as:

- individual widths for four separable blocks, constrained to supported alignment;
- gate channels (for example 16/24/32/40);
- whether expensive late temporal kernels can be shortened without losing accuracy;
- frontend choice only as a separate experiment because it changes comparability;
- PTQ versus QAT and int8 versus an int16-activation fallback.

The objective must use deployed graph cost: int8 FlatBuffer bytes, predicted arena,
actual exported MACs, and a measured Pico operator/latency lookup table. Do not
reward activation sparsity unless the runtime skips the work.

Use successive halving: short-train many candidates, retain the nondominated set,
then train only finalists for the full schedule and five seeds. If the GB10 is used,
create a pinned shell script that launches independent candidates/seeds and writes a
manifest; do not make the workstation wait on an interactive remote run.

This pipeline succeeds if a non-uniform model dominates C12/C10/C8 after exact int8
deployment. It is a credible paper/control contribution even if PAI remains negative.

### Pipeline 4 — Repaired, budget-matched shrink-first dendritic recovery

**Priority:** 4; required dendrite ablation, not the shipping default  
**Risk:** high  
**Research value:** cleanly tests the central dendrite hypothesis

```text
C8 or C10 scratch checkpoint
  -> no identity optimizer/objective reset
  -> one fc-only PAI/PB dendrite
  -> matched no-dendrite continuation
  -> clean fixed graph + parity
  -> compare at equal total params to scratch-width frontier
  -> int8 QAT and TFLM only if it first wins in FP32
```

The `fc` placement is selected because it is the only one with a strong PB
correlation signal and adds almost no convolutional MACs. Test C8 first for recovery
headroom and stability; include C10 if the first gate passes. Use one dendrite, exact
output-dimension configuration, zero dendrite-training weight decay unless current
vendor guidance says otherwise, frozen BN statistics in p mode, optimizer rebuild
after restructuring, and a fixed phase cap.

Most importantly:

- start directly from the best scratch checkpoint;
- preserve the paper optimizer/objective when ordinary weights are trained;
- give the matched control exactly the same examples, epochs, optimizer, scheduler,
  and checkpoint selection;
- omit KD from this first causal ablation;
- count the clean graph's total branch parameters/MACs;
- compare C8+`fc` not only with C8 but also with the interpolated/native scratch
  candidate nearest its total parameter count.

Screen with three seeds. Stop the branch if the mean paired gain versus the matched
control is below +0.2 pp, any clean-export parity issue remains, or it is dominated by
a native-width model. Only after a three-seed FP32 win should it receive QAT, five
seeds, held-out test, and Pico benchmarking.

This strict gate prevents another 120-cell sweep around a broken handoff. A negative
result would still be publishable ablation evidence when reported honestly.

### Pipeline 5 — Conditional iterative Pareto compressor

**Priority:** 5; final research framework  
**Risk:** high  
**Research value:** closest to the proposed paper contribution

```text
fixed same-frontend teacher
  -> structured channel shrink
  -> matched KD recovery
  -> benchmark candidate
  -> conditional dendritic recovery only if Pipeline 4 passed
  -> optional second KD only if a paired ablation passed
  -> QAT
  -> exact .tflite/TFLM benchmark
  -> retain nondominated candidates
  -> repeat from retained parents
```

This is not a rigid `prune -> KD -> dendrite -> KD` ritual. Every stage emits a
candidate and its matched control; a stage stays in the loop only after it extends
the current validation frontier. The fixed teacher remains unchanged. Student
weights may warm-start from a retained parent, but test data remain sealed until
selection freezes.

Recommended state for each candidate:

- parent/model lineage and random seed;
- exact architecture and clean graph hash;
- structured width change;
- recovery recipe and teacher hash;
- PAI placement/dendrite count/PB diagnostics when applicable;
- quantization recipe and representative-set hash;
- host and Pico artifact metrics;
- dominance reason and accept/reject decision.

Stop a lineage after two successive smaller candidates fail to extend the frontier,
or immediately when SRAM/latency improves only trivially while accuracy crosses a
predeclared floor. Keep N:M sparsity and clustering outside this loop until the Pico
runtime has a real sparse kernel or codebook decoder; otherwise they change a file
format, not inference cost.

The full framework becomes scientifically credible only after Pipelines 1–4 provide
stage-level controls. It is the long-term paper pipeline, not the first hackathon run.

## 7. Ranking and execution order

| Rank | Pipeline | First question answered | Go criterion |
|---:|---|---|---|
| 1 | Native SparkNet + int8 | What is the real Pico frontier now? | integer graph, <=0.2 pp loss, device fit |
| 2 | Structured prune + KD | Does recovery beat scratch at the same width? | paired gain and device Pareto extension |
| 3 | Non-uniform hardware search | Is uniform width leaving accuracy/cost on the table? | dominates a native-width point |
| 4 | Corrected `fc` dendrite | Can PAI add useful capacity after a fair handoff? | >=+0.2 pp screen and equal-budget nondominance |
| 5 | Conditional iterative loop | Do validated stages compose repeatedly? | at least one new frontier point per iteration |

For a prototype deadline, complete Pipeline 1 and one Pipeline 2 comparison. For a
paper, add Pipeline 3, the corrected Pipeline 4 ablation, and only then Pipeline 5.

## 8. Under-30-minute experiment discipline

No single local training invocation should exceed 20–30 minutes. Use two promotion
levels:

### Screen

- one seed for export/integration failures;
- at most three seeds for a training hypothesis;
- 15–25 continuation epochs or an existing checkpoint;
- validation only;
- automatic wall-clock limit and an explicit incomplete status;
- never interpret partial PAI snapshots or `noImprove` files as trained models.

### Confirm

- only candidates that passed the screen;
- five seeds, full declared schedule;
- freeze selection, then evaluate held-out test once;
- export and benchmark every promoted seed or a predeclared representative seed for
  hardware metrics, while reporting exactly which policy was used.

The current scratch run times (~10–11 minutes each) already satisfy the per-run
ceiling. Parallelism may reduce wall time but does not change the experimental unit.
If the GB10 becomes necessary, create a user-executable script that validates the
environment, pins config/checkpoint hashes, runs independent jobs with per-job logs,
and supports safe resume. No GB10 script was needed for this design/audit stage.

## 9. First implementation slice

The next agent should implement **only Pipeline 1's artifact path** before changing
training:

1. Select one C12 and one C16 checkpoint plus 100–300 training-only representative
   MFCC tensors and a fixed parity/test bundle.
2. Produce a clean inference graph and verify PyTorch parity.
3. Try direct `litert-torch` full-int8 conversion. Record every operator and dtype.
4. If conversion fails, determine whether a small Keras inference twin with weight
   transfer is less risky than ONNX-to-TensorFlow conversion.
5. Create a minimal Pico SDK + pinned TFLM application with a fixed feature tensor;
   log exact arena usage, linked flash/static RAM, output parity, and 100+ warm
   invocations at 133 MHz.
6. Add the streaming MFCC/microphone frontend only after model execution is correct.
7. Record the first frontier rows. Then decide whether PTQ is enough or QAT is
   required.

Do not start by integrating pruning, PAI, clustering, or a GPU sweep. Without a
measured deployment path, those stages optimize proxies whose relevance is unknown.

## 10. Claims policy

Supported now:

- the five-seed SparkNet scratch validation/test frontier above;
- the current PAI recipe is dominated and the selector chose controls everywhere;
- `fc` has the best PB correlation signal but no causal/test-set gain;
- current KD trials are neutral/negative;
- no Pico measurement exists yet;
- TFLM is the lower-risk RP2040 runtime.

Not supported now:

- dendrites improve KWS accuracy or quantization tolerance;
- 90–99% compression at >=90% retained useful performance;
- teacher-level accuracy at equal or fewer total parameters;
- the model fits a 32 kB MCU simply because weights are small;
- host latency, projected memory, MACs, N:M masks, or codebook sidecars imply Pico
  speed, SRAM, flash, or energy savings;
- MRAM/ReRAM area/energy/latency improvement without a pinned simulator mapping and
  comparison against the exact deployed network.

The MRAM/ReRAM work should consume the final fixed int8 graphs from the Pico frontier.
That keeps the algorithm comparison identical across MCU and CIM evaluation and
avoids simulating a model that cannot be reproduced in the actual prototype.

## 11. Work log and handoff

Actions performed in this research/design pass:

- Read the repository notes under `notes/dendrite-study-v2`, model/config/export
  code, output manifests/reports, selection files, KD diagnostics, and PAI artifacts.
- Applied the PerforatedAI analysis checklist to the completed study artifacts.
- Regenerated the study aggregation read-only into `/tmp`; no output/checkpoint was
  rewritten. The current study is 30/30 scratch and 150/150 arms complete.
- Read the SparkNet paper and official SparkNet repository material.
- Reviewed official PerforatedAI site, documentation, source repository, and paper.
- Reviewed official RP2040, TFLM, CMSIS-NN, ExecuTorch, LiteRT/Torch conversion, and
  Raspberry Pi Pico TFLM material; assessed the supplied Edge Impulse article.
- Ran no new training and made no source/config/checkpoint changes. Existing results
  already answered the first causal question, and the next useful work is the
  missing exact deployment path.
- Did not create a branch because only new research notes were added. The preexisting
  untracked `DATA_INVENTORY.md` was preserved.

Read-only aggregation command used from `KWS_Model/`:

```bash
env UV_CACHE_DIR=/tmp/kws_uv_cache uv run python \
  notes/dendrite-study-v2/aggregate_study.py \
  --study-root outputs/sparknet-dendritic-study-v2 \
  --json /tmp/kws-aggregate-current.json
```

Before beginning implementation, inspect `git status`, preserve user work, and create
a feature branch if source or build files will be changed. Do not overwrite the
existing completed study.

## 12. Primary source inventory

- [SparkNet paper](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf)
- [SparkNet reference repository](https://github.com/jsvir/sparknet)
- [PerforatedAI official site](https://perforatedai.com/)
- [PerforatedAI official repository](https://github.com/PerforatedAI/PerforatedAI)
- [PerforatedAI generated API documentation](https://docs.perforatedai.com/perforatedai/)
- [RP2040 specifications](https://www.raspberrypi.com/products/rp2040/specifications/)
- [RP2040 datasheet](https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Raspberry Pi Pico TFLM port](https://github.com/raspberrypi/pico-tflmicro)
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
- [LiteRT Torch converter](https://github.com/google-ai-edge/litert-torch)
- [ExecuTorch runtime overview](https://github.com/pytorch/executorch/blob/main/docs/source/runtime-overview.md)
- [Edge Impulse ExecuTorch article supplied for review](https://www.edgeimpulse.com/blog/from-pytorch-to-the-edge-getting-started-with-executorch-and-edge-impulse/)
