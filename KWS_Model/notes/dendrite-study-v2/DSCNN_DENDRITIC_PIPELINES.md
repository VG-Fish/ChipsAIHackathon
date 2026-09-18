# DS-CNN dendritic compression pipelines for RP2040

**Decision and handoff journal — 2026-09-18**

This journal extends `PICO_COMPRESSION_PIPELINE_JOURNAL.md` with DS-CNN-specific
designs. It records the local evidence, cost calculations, experiment controls,
deployment gates, and recommended execution order. No accuracy or Pico result
below is claimed unless it already exists as an artifact. Proposed architectures
and thresholds are hypotheses to test.

## 1. Decision summary

Five plausible DS-CNN pipelines are defined below. The two to run first are:

1. **Pipeline 1 — existing XS-to-width frontier plus one classifier dendrite.**
   This is the lowest-code-risk experiment and the cleanest way to determine
   whether the strong classifier PBScore signal is causal. Sweep widths 24, 22,
   20, and 18 from the strong 4,096-parameter XS checkpoint; compare every
   dendritic arm to an identical no-dendrite continuation and a conventional
   model with the same total deployed parameter budget.
2. **Pipeline 2 — deep-strided DS-CNN Micro plus one classifier dendrite.**
   Add an early stride-2 depthwise-separable block so the model can have four
   feature blocks while remaining below the current XS in parameters and MACs.
   The proposed base has 3,668 parameters and about 0.983M MACs; one classifier
   dendrite produces a 4,076-parameter, approximately 0.984M-MAC candidate.
   This has the best estimated Pareto upside but requires a small architecture
   change and fresh training.

Pipeline 3 tests a last-pointwise feature dendrite on a still smaller
deep-strided backbone. Pipelines 4 and 5 are later accuracy-first and iterative
research stages. They should not consume the first experiment budget.

The current repository does **not** yet demonstrate that dendrites improve a
DS-CNN Pareto frontier. Existing DS-CNN PAI runs are one-seed, validation-only,
schedule-confounded, and not TFLite Micro deployments. The next run must answer
that causal question before adding clustering, repeated pruning cycles, or
ReRAM simulation.

Training was not launched during this design pass. The selected PerforatedAI
integration skill explicitly prohibits the agent from running user training
scripts. The run recipes below are therefore prepared for a human-triggered
workstation or GB10 execution after the missing controls and time caps exist.

## 2. Hardware and scientific contract

### RP2040 target

- two Arm Cortex-M0+ cores, officially up to 133 MHz;
- 264 kB on-chip SRAM across six banks;
- execute-in-place external QSPI flash, with boards commonly providing far
  more flash than SRAM;
- scalar Cortex-M0+ execution: CMSIS-NN compatibility does not imply the SIMD
  acceleration available on Cortex-M4/M7/M33-class devices;
- TensorFlow Lite Micro is the primary runtime unless an alternative produces
  a smaller/faster measured artifact without weakening reproducibility.

The practical constraint is not just weight bytes. The tensor arena, frontend
state, input/audio buffers, stack, runtime metadata, and application code share
the SRAM budget. A model only qualifies as a Pico result after the actual int8
FlatBuffer allocates and invokes on the target.

### Scientific claim required

A dendritic model must be compared with both:

1. the exact same base checkpoint continued without a dendrite for the same
   optimizer steps and wall-clock/schedule treatment; and
2. an ordinary DS-CNN allocated the same total deployed parameters and, where
   materially different, another ordinary DS-CNN with similar MACs.

A gain over the smaller pre-dendrite base alone only shows that adding capacity
helps. It is not evidence of more efficient capacity.

## 3. What the repository currently establishes

### 3.1 Input shape and artifact provenance

The deployment log-mel configuration uses 16 kHz, a 30 ms window, a 10 ms hop,
and 40 mel bins. Direct inspection of the strong teacher and warm XS checkpoint
metadata gives **`input_shape=(40, 101)`**. Some older unit tests use `(40, 98)`;
those synthetic profiles must not be mixed with the trained artifacts.

Two similarly named XS checkpoints have very different provenance:

| Checkpoint | Provenance | Relevant value |
| --- | --- | ---: |
| `outputs/full-run-20260912T063022Z/models/checkpoints/student/ds_cnn_xs_distilled_warm_12class.pt` | strong full-run source | 85.11% validation, 83.64% held-out test |
| `models/checkpoints/ds_cnn_xs_distilled_warm_12class.pt` | stale/weak source used by the legacy compression run | metadata `source_val_acc=73.02%` |

Do not combine candidates derived from these sources on one causal curve.

### 3.2 Best relevant baselines

| Artifact | Accuracy | Parameters | Status |
| --- | ---: | ---: | --- |
| DS-CNN-L log-mel teacher | 97.6858% test | 469,604 | strong fixed teacher/reference |
| DS-CNN-XS warm student | 85.1109% validation | 4,096 | strong available DS-CNN student source |
| DS-CNN-XS warm student | 83.6431% test | 4,096 | held-out baseline; FAR 18.53%, FRR 15.29% |
| DS-CNN-L MFCC-32 retrain | 91.76% best validation | 469,604 | overfit and too weak to supervise the stronger MFCC SparkNet |

The 40-bin teacher and 40-bin student are frontend-compatible. The MFCC
teacher is a separate failed recipe and should not be used to infer that KD is
generally ineffective.

### 3.3 Existing direct DS-CNN dendrite evidence

The completed historical candidates all perforated only `fc` and accepted one
dendrite:

| Source/run | Width | Conventional/pruned validation | PAI final validation | Clean deployed params | Recorded MACs | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| weak legacy source | 18 | 65.21% | 70.16% | 2,070 | 1,450,668 | within-run recovery, not a matched causal control |
| weak legacy source | 14 | 64.69% | 66.44% | 1,714 | 1,209,900 | same limitation |
| strong full-run source | 18 | 79.38% prune-KD best | 82.68% PAI best | 2,070 | 1,450,668 | pipeline incomplete; internal PAI zero-row was already 82.58% |

For the strong-source w18 run, the PAI internal architecture rows changed only
from 82.584% with zero dendrites to 82.681% with one dendrite, approximately
**+0.10 percentage point**. The later KD resume peaked at 82.53%, below the PAI
selection score. Nearly all of the apparent gain over the 79.38% pruned
checkpoint occurred during a much longer neuron-training phase, so it cannot be
attributed to the dendrite without a schedule-matched continuation.

The classifier PBScore was nevertheless consistently large: roughly 0.133 in
the strong-source run and 0.136–0.147 in the direct legacy runs. Under the PAI
analysis rubric that is a strong placement correlation signal. It just is not
an accuracy effect by itself.

### 3.4 Runtime and failure evidence

- Legacy w18 used about 0.52 h of prune/KD, 2.91 h of PAI, and 0.10 h of resume
  KD. The PAI lifecycle reached 249 epochs.
- Legacy w14 used about 0.47 h of prune/KD, 1.62 h of PAI, and 0.10 h of resume
  KD. The PAI lifecycle reached 140 epochs.
- The strong full-run contains missing-checkpoint, RNG-state, recipe mismatch,
  debugger, index, interruption, and cleanup failures. It is not end-to-end
  evidence.
- No DS-CNN dendritic candidate has a held-out test result, an int8 TFLite
  artifact, a TFLM tensor-arena measurement, or RP2040 latency/power data.

The history-driven PAI recipe therefore violates the requested 20–30 minute
screen. A fixed, hard-capped lifecycle and a matched continuation arm are
prerequisites to launching the new screen.

## 4. Cost model and architecture observations

### 4.1 Why structured channel pruning is primary

`prune_ds_cnn` performs real channel surgery: it removes pointwise output
channels and threads the surviving indices through the following depthwise
filters, BatchNorm tensors, pointwise inputs, and final classifier. This changes
the dense graph and can reduce MCU weights, MACs, and activations.

The current N:M path masks weights but retains dense tensor shapes. It should
not be credited with M0+/TFLM latency or memory savings without a measured
sparse kernel. N:M remains relevant to a future CIM mapping, not the first Pico
claim.

### 4.2 Existing two-block width formula

For the checked-in two-block DS-CNN family with an 18-channel, 5x5, stride-2
stem, equal block width `w`, 12 classes, and `(40,101)` input:

```text
base parameters = w^2 + 45w + 696
base MACs       = 1020w^2 + 27552w + 624240
one fc dendrite parameter overhead = 12w + 24
one fc dendrite MAC overhead       = 12w + 12
```

The parameter overhead includes a copied `Linear(w,12)` including bias and 12
learned residual scales. The MAC overhead includes the copied matrix multiply
and scale. Historical clean reports counted only +12 MACs, apparently omitting
the copied classifier. New reports must use the analytical/projected cost or a
profiler that observes both branches.

| Width | Base params | Base MACs | + one `fc` dendrite params | + one `fc` dendrite MACs |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 2,352 | 1,873,008 | 2,664 | 1,873,308 |
| 22 | 2,170 | 1,724,064 | 2,458 | 1,724,340 |
| 20 | 1,996 | 1,583,280 | 2,260 | 1,583,532 |
| 18 | 1,830 | 1,450,656 | 2,070 | 1,450,884 |

### 4.3 Proposed deep-strided backbones

The current XS obtains width by using only two DS blocks, both at the full
20x51 post-stem feature map. `DSConvBlock` already implements a stride
parameter, but `DSCNN` does not expose per-block strides. Adding a stride-2
first DS block permits more depth and receptive-field stages at substantially
lower compute.

All values below are derived for `(40,101)`, 12 classes, and current module
semantics. They must be re-profiled after implementation.

| Candidate | Stem | Blocks / strides | Base params | Base MACs | Dendrite | Final params | Approx. final MACs |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Deep-strided Micro | 12 ch, k5, s2 | `[16,24,24,32]` / `[2,1,1,1]` | 3,668 | 983,424 | `fc`, one | 4,076 | 983,820 |
| Deep-strided Feature | 12 ch, k5, s2 | `[16,20,20,24]` / `[2,1,1,1]` | 2,924 | 827,328 | `.blocks.3.pointwise`, one | 3,428 | 958,368 |

The Micro stem output is 20x51x12 = 12,240 int8 elements and later feature
maps are 10x26. That is smaller than the checked-in XS stem output despite
having four blocks. The Feature pointwise branch adds 480 weights, 24 scales,
124,800 copied pointwise MACs, and 6,240 scale operations.

These are attractive graph designs, not accuracy estimates. Early downsampling
can destroy useful temporal/frequency detail; that is the main empirical risk.

## 5. Five plausible DS-CNN pipelines

### Pipeline 1 — strong XS, structured width sweep, classifier dendrite

**Purpose:** obtain the lowest-risk causal answer with existing model code.

```text
fixed log-mel DS-CNN-L teacher
  -> strong warm DS-CNN-XS checkpoint
  -> structured surgery to w={24,22,20,18}
  -> bounded KD recovery
  -> fork the exact checkpoint and optimizer state
       A: matched ordinary continuation
       B: PAI on .fc, one dendrite
  -> clean graph -> full-int8 QAT/PTQ comparison -> TFLM gate
```

Why it is plausible:

- the classifier is the only DS-CNN placement with completed local runs;
- its PBScore correlation is consistently strong;
- its deployed compute overhead is negligible relative to the convolutions;
- Linear/Tanh/Mul/Add is less risky than grouped/depthwise PAI wrapping.

Required controls:

- exact no-dendrite continuation with identical total optimizer steps;
- ordinary same-budget widths. For example, w20 + `fc` is exactly 2,260
  parameters, the same as an ordinary w23 model, while using fewer MACs;
- scratch and prune/KD provenance kept separate;
- no held-out test evaluation until the validation recipe and width are frozen.

Success gate: the dendritic arm must improve the matched continuation and be
non-dominated by the ordinary same-total-parameter width after full int8
conversion. For confirmation, require at least three seeds, a mean advantage of
at least 0.3 percentage point or a confidence interval excluding a tie, and no
greater Pico arena/latency at an equivalent accuracy point.

Risk: the two-block backbone may be representationally too weak. The best
strong-source w18 result was only 82.68% validation, below the 85% screen floor.

### Pipeline 2 — deep-strided Micro, classifier dendrite (recommended upside)

**Purpose:** reallocate the current XS budget from wide, high-resolution blocks
to a deeper hierarchy, then test the lowest-risk dendrite placement.

```text
fixed log-mel DS-CNN-L teacher
  -> train deep-strided Micro [16,24,24,32], strides [2,1,1,1]
  -> optional response KD only if a same-schedule no-KD arm benefits
  -> fork the exact converged base
       A: matched ordinary continuation
       B: one .fc dendrite
  -> compare with a conventional architecture near 4,076 total params
  -> clean -> int8 -> TFLM -> Pico
```

Budget hypothesis: 4,076 final parameters and about 0.984M analytical MACs,
slightly fewer parameters than the existing XS and less than one-third of its
3.358M recorded MACs.

Required controls:

- base trained normally to avoid conflating architecture benefit with KD;
- same-checkpoint continuation for the PAI lifecycle;
- conventional width/channel allocation tuned to approximately 4,076 params;
- optional two-block XS control at the same training schedule;
- report architecture gain and dendrite gain separately.

Success gate: first require at least 85% validation in the short screen, then
require the dendritic arm to beat its matched continuation by at least 0.2
percentage point and remain on the validation-accuracy/params/MAC frontier.
Final evidence needs three seeds, full-int8 degradation no worse than 0.5
percentage point, TFLM parity, and measured Pico admission.

Risk: the new early stride may lower accuracy. If the base is weak, test a
less-aggressive placement of the stride or asymmetric `[1,2]` downsampling
before concluding that additional DS-CNN depth is ineffective.

### Pipeline 3 — deep-strided Feature, late pointwise dendrite

**Purpose:** test whether dendrites need feature-map capacity rather than a
second classifier projection.

```text
train deep-strided Feature [16,20,20,24], strides [2,1,1,1]
  -> matched checkpoint fork
       A: ordinary continuation
       B: PAI only on .blocks.3.pointwise, one dendrite
       C: same base + .fc dendrite placement control
  -> conventional 3,428-param and ~0.96M-MAC controls
  -> clean/int8/TFLM gate
```

The last pointwise copy operates on 10x26 features, so it can correct
high-level time/frequency features at bounded cost. It avoids the undocumented
grouped-depthwise PAI case.

Success gate: it must beat both the matched no-dendrite continuation and the
same-base classifier placement enough to justify its extra 131,040 MACs. It
must also be non-dominated by a conventional same-budget DS-CNN.

Risks:

- no local PBScore evidence exists yet for this exact pointwise module;
- the residual branch retains spatial activations and may enlarge the TFLM
  arena;
- BN fusion and residual quantization are more complex than for `fc`;
- Tanh/Mul/Add must remain fully integer with no float fallback.

### Pipeline 4 — DS-CNN-S to narrow accuracy-first recovery

**Purpose:** determine whether a better pre-pruning representation gives PAI a
more recoverable student than the underperforming XS source.

```text
fixed teacher
  -> train DS-CNN-S student
  -> structured keep-ratio sweep, e.g. 0.75/0.50/0.35
  -> bounded KD recovery
  -> one late pointwise OR classifier dendrite (separate arms)
  -> matched continuation and same-total-cost ordinary controls
  -> clean/int8/TFLM/Pico gate
```

This is an accuracy-first research arm, not the first prototype. The unpruned S
has 24,188 parameters and about 23.9M `(40,101)` MACs; it must be structurally
narrowed before it is a competitive always-on RP2040 point.

Success gate: its final candidate must dominate Pipelines 1–3 on at least one
accuracy/cost frontier. If it only achieves more accuracy by spending more
weights, MACs, arena, and latency, retain it as a reference rather than the Pico
winner.

Risk: training and pruning cost exceeds the screen budget, and the final graph
may remain activation-heavy despite few weights.

### Pipeline 5 — conditional iterative Pareto compressor

**Purpose:** test the original `prune -> KD -> dendrite -> KD` research idea
only after one single-cycle dendritic gain is causal and deployable.

```text
best validated DS-CNN student
  -> structured prune one step
  -> matched KD recovery
  -> dendrite only if the single-cycle placement passed
  -> matched post-integration KD
  -> clean/int8/deploy/profile
  -> add candidate to frontier
  -> repeat only while hypervolume/frontier improves
  -> optional clustering after architecture selection
```

Every cycle must restart from the selected clean graph, not a live PAI wrapper.
Each stage records source hash, seed, data split, training steps, parameters,
MACs, activation estimate, FlatBuffer bytes, TFLM arena, and target latency.

Stop after two consecutive size targets fail to add a non-dominated point, or
immediately if the clean/int8/TFLM gate fails. Do not use validation accuracy
alone as the stopping rule.

Risk: without the preceding ablation, iteration compounds selection bias and
training-budget confounds. This pipeline is a paper-stage experiment, not the
fastest route to a demonstrable Pico model.

## 6. Twenty-to-thirty-minute screening protocol

This is a feasibility/ranking screen, not the final paper run.

### Static admission, 1–2 minutes

- instantiate base, dendritic projection, and conventional matched controls;
- validate exact PAI module IDs;
- count both branch copies and residual scale/add operations;
- reject over-budget graphs before dataset or PAI initialization;
- check that clean export and reload are possible on an untrained toy model.

### Base/control phase, 8–10 minutes

- cached deterministic log-mel features;
- one declared seed, batch size 256 if the device supports it;
- identical short horizon, augmentation, optimizer, and scheduler for all arms;
- save model, optimizer, scheduler, RNG, split, and recipe fingerprints;
- use validation only.

For Pipeline 1, begin with w20 because its final 2,260-parameter dendritic graph
has an exact-parameter ordinary w23 control. If time remains, screen w22 and
w18. For Pipeline 2, first establish that the new base forwards, trains, and
exports before invoking PAI.

### Dendrite/matched-continuation phase, 8–12 minutes

- fork one exact checkpoint;
- use a single, predetermined PAI switch/candidate attempt and
  `max_dendrites=1`;
- cap the total p-mode, integration, and post-integration epochs explicitly;
- give the no-dendrite arm the same total optimizer steps and scheduler
  treatment;
- retain PAI Scores, switch epochs, PBScore, LR, best-architecture, accepted
  dendrite count, and elapsed-time files.

The existing dynamic history mode is not admissible for this screen because it
has taken 104–211 minutes for the PAI portion alone. If a hard lifecycle cap is
not supported by the current wrapper, implement and test that control before
starting training; do not simply interrupt a dynamic run and report its best
validation row.

### Deployment smoke gate, 3–5 minutes

- clean/finalize PAI into the fixed deployment graph;
- verify live-PAI to clean-PyTorch output parity;
- run the repository's quantizable dendritic replacement path;
- require a full-int8 model with int8 input/output and no float fallback;
- enumerate FlatBuffer operators and versions;
- compare int8 host accuracy/logits with clean FP32;
- compile a minimal TFLM resolver and attempt tensor allocation/invocation.

Only a passing screen advances to three seeds and held-out test evaluation.

## 7. Pareto and acceptance rules

Maintain separate validation and final-test frontiers. A candidate `a` dominates
`b` if `a` is no worse in every declared objective and strictly better in at
least one. Initial objectives are:

- maximize classification accuracy;
- minimize deployed parameter count;
- minimize analytical/exported MACs;
- minimize int8 FlatBuffer bytes;
- minimize measured TFLM tensor-arena peak;
- minimize Pico p50/p95/p99 inference latency;
- later minimize measured energy per inference.

FAR and FRR are reported diagnostics, not folded into accuracy. If always-on
false accepts matter more than aggregate accuracy, declare that constraint
before selection.

Hard gates:

- full-int8 conversion with no unsupported/floating fallback;
- TFLM `AllocateTensors()` and `Invoke()` success;
- application-level SRAM below 264 kB with explicit headroom, not merely an
  activation estimate;
- deterministic class mapping and frontend parity;
- no candidate selected on held-out test;
- no dendritic claim from a one-seed or unmatched comparison.

Suggested screen thresholds are at least 85% validation and no more than 0.5
percentage point lost to int8. These are triage thresholds, not paper claims.
The longer-term 90%+ target remains aspirational until a qualifying artifact
exists.

## 8. TFLite Micro deployment plan

### Expected base operators

The ordinary DS-CNN should reduce to Conv2D, DepthwiseConv2D, fused ReLU,
AveragePool2D, FullyConnected, and graph-shape operations such as Reshape or
Squeeze. BatchNorm must be folded and Dropout removed.

Official TFLite int8 supports per-axis weights for Conv2D and
DepthwiseConv2D. CMSIS-NN lists int8 implementations for convolution,
depthwise convolution, fully connected, Add, Mul, average pooling, and
softmax. This is compatibility evidence, not an RP2040 speed result.

### Dendritic graph risks

The cleaned PAI branch is conceptually:

```text
main(x) + scale * tanh(copied_module(x))
```

Therefore the exact FlatBuffer may additionally require Tanh, Mul, and Add.
The current repository has a `QuantizableDendriticResidual` because eager int8
cannot safely multiply an int8 branch tensor by floating scales directly. The
final graph must prove that the entire residual is integer-compatible. A
converter that silently inserts Dequantize/Quantize around the branch fails the
Pico compression claim.

The first resolver should be generated from the actual FlatBuffer rather than
hard-coded from the PyTorch architecture. Measure:

- FlatBuffer bytes and weight-storage section;
- exact operator list and versions;
- per-channel weight scales and input/output quantization parameters;
- arena used after `AllocateTensors()`;
- PyTorch-clean, host-TFLite, and device-TFLM logits/predictions;
- model-only and end-to-end frontend+model latency;
- p50, p95, p99, clock rate, build flags, and runtime commit;
- power/energy only when measurement setup is repeatable.

Edge Impulse or ExecuTorch may be used as a comparison/export aid, but the
primary result should remain the smallest working runtime/artifact measured on
the Pico. Edge Impulse cannot substitute for reporting the exact generated
model and arena.

## 9. Implementation work required before a run

No source branch was created in this design pass because no source was changed.
When Pipeline 2 is implemented, create a feature branch and keep the first patch
small:

1. add optional `block_strides` to `DSCNN` and `build_ds_cnn`;
2. validate that its length equals `block_channels`, defaulting to all ones so
   old checkpoints/configs remain compatible;
3. add shape, parameter, MAC, prune, checkpoint, and ONNX parity tests;
4. add the two proposed model configs and exact cost assertions;
5. add a genuinely hard PAI lifecycle cap plus a schedule-matched no-dendrite
   dispatcher path;
6. fix profiling so the cleaned copied branch MACs are counted;
7. add clean-graph and int8/TFLM smoke commands before long training.

Pipeline 1 can reuse current model code, but it still needs items 5–7 and must
point to the strong full-run XS checkpoint explicitly. The current default
compression config points to the weak root checkpoint and should not be reused
unchanged.

For GB10 execution, the eventual shell script should:

- fail fast on the wrong checkpoint hash, dirty recipe mismatch, or missing PAI
  runtime/license;
- run a static dry-run first;
- launch one arm at a time with a wall-clock timeout below 30 minutes;
- write a manifest before training and atomically mark each arm complete;
- never delete or overwrite prior outputs;
- stop after the deployment smoke gate fails;
- print the exact resume command rather than guessing at partial state.

## 10. Recommended execution order

1. Fix/verify clean branch cost accounting and the hard run cap.
2. Run Pipeline 1 w20: matched continuation, one `fc` dendrite, and ordinary
   w23. This is the fastest causal placement test.
3. If the dendrite beats the matched arm, screen w22/w18 and confirm the best
   width with three seeds.
4. In parallel with step 2 implementation, add `block_strides` and run Pipeline
   2 base-only versus ordinary XS. Do not invoke PAI if the base itself is not
   competitive.
5. If the deep-strided base passes, run its classifier dendrite and matched
   continuation.
6. Run Pipeline 3 only if classifier PAI is causal or if its PBScore/output
   analysis strongly suggests the representation, rather than the head, is the
   bottleneck.
7. Freeze the validation choice, run held-out test, then perform full Pico
   benchmarking.
8. Only then evaluate Pipeline 4, iterative Pipeline 5, clustering, CIM, and
   ReRAM/MRAM simulation.

If no dendritic arm survives, deploy the best ordinary DS-CNN/SparkNet Pareto
point and report dendritic placement as a controlled negative result. That is
more credible than treating PAI's in-search architecture rows as a compression
win.

## 11. Claims policy

Safe now:

- a strong 97.69% DS-CNN-L teacher and an 83.64% test DS-CNN-XS student exist
  on the log-mel task;
- structured DS-CNN channel surgery exists and changes the dense graph;
- classifier PAI runs completed and show strong PBScore alignment;
- no controlled DS-CNN dendritic Pareto or Pico deployment result exists yet;
- the two recommended pipelines have plausible analytical Pico-scale costs.

Not safe until experiments pass:

- dendrites improve DS-CNN accuracy per parameter or per MAC;
- the proposed deep-strided architecture is more accurate than XS;
- any dendritic graph is fully int8/TFLM compatible;
- 90–99% compression preserves 90%+ of useful performance;
- RP2040 latency, SRAM, energy, or keyword-count improvements;
- better quantization tolerance for dendritic models;
- MRAM/ReRAM energy or area improvement.

## 12. Work log and handoff inventory

### Actions performed in this pass

- ran Memorable status/recall; no matching procedure was available;
- read the PerforatedAI setup and results-analysis skill instructions;
- delegated three independent audits: local DS-CNN results, primary-source
  deployment research, and pipeline/control design;
- inspected DS-CNN architecture, pruning, placement, PAI config, checkpoints,
  reports, manifests, metrics, and PAI CSV findings;
- directly verified `(40,101)` metadata for the strong teacher and warm XS
  checkpoint;
- reconciled parameter/MAC formulas and identified historical copied-FC MAC
  undercounting;
- designed and ranked five pipelines;
- did not change source, configs, models, checkpoints, or outputs;
- did not launch training.

### Supporting reports

- `agent-dscnn-results-audit.md` — local artifacts, provenance, failures, PAI
  checklist, and causal limitations. Its opening correction notes that the
  trained log-mel checkpoints are `(40,101)`; its `(40,98)` tables are only
  synthetic profiles.
- `agent-dscnn-primary-research.md` — Hello Edge, structured pruning, official
  int8/TFLM/CMSIS-NN evidence, and PAI unknowns.
- `agent-dscnn-pipeline-designs.md` — existing XXS/XS placement designs,
  same-budget controls, and 20–30-minute screen.
- `PICO_COMPRESSION_PIPELINE_JOURNAL.md` — prior complete SparkNet/Pico study.
- `agent-perforatedai-primary-research.md` — PAI primary-source behavior and
  limitations.
- `agent-sparknet-pico-deployment-research.md` and
  `agent-repo-results-audit.md` — earlier architecture/deployment and repository
  evidence.

### Primary external sources used

- Arm, *Hello Edge: Keyword Spotting on Microcontrollers*:
  <https://arxiv.org/abs/1711.07128>
- Arm ML-KWS-for-MCU reference implementation:
  <https://github.com/ARM-software/ML-KWS-for-MCU>
- TensorFlow Lite int8 quantization specification:
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_spec.md>
- TensorFlow post-training integer quantization guidance:
  <https://www.tensorflow.org/lite/performance/post_training_quantization>
- TensorFlow Lite Micro `micro_speech` example:
  <https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech>
- CMSIS-NN supported operator documentation:
  <https://github.com/ARM-software/CMSIS-NN>
- PerforatedAI official repository:
  <https://github.com/PerforatedAI/PerforatedAI>

### Immediate next owner action

Implement the hard-capped matched-control harness and correct clean-branch MAC
profiling, then run Pipeline 1 w20. In a separate small feature branch, add
backward-compatible `block_strides` and statically validate Pipeline 2. Do not
start the multi-cycle compressor until one of those single-cycle experiments
produces a replicated, fully deployed Pareto improvement.
