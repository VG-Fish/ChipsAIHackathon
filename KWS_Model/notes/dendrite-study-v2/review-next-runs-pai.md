# Review: next PAI/KWS Pareto runs

Audited 2026-09-18 against `PAI_KNOWLEDGE.md`,
`agent-perforatedai-primary-research.md`, `DSCNN_DENDRITIC_PIPELINES.md`, and
the checked-in KWS implementation. This is a review of the proposed next runs;
it does not claim a new training result.

## Executive decision

Run one tightly controlled SparkNet classifier experiment first: repaired C8,
one `.fc` dendrite, one exact schedule-matched no-dendrite continuation, and
one conventional dense control at the same final parameter count. Keep the PAI
lifecycle to one candidate and one integration attempt, with a hard wall-clock
limit below 30 minutes per arm. This is the cheapest direct test because the
head is already exercised and C8 keeps its relative cost manageable.

Do not start with gate+classifier, deep-strided architecture, a late pointwise
arm, repeated prune/KD/PAI cycles, or a broad width sweep. The C8 gate+fc arm is
the second, exploratory same-budget test; DS-CNN Micro is a secondary
architecture test. These are worthwhile follow-ups only after the first
experiment proves that the dendrite itself improves the matched continuation.
The current evidence supports classifier placement as a cheap hypothesis, not
as a demonstrated Pareto win.

## What is established, and what is not

| Question | Evidence status | Consequence |
| --- | --- | --- |
| Is a classifier a plausible PAI placement? | Plausible. Existing runs report PBScore around 0.133–0.147. | Use `.fc` for the first causal screen. |
| Did the classifier dendrite improve accuracy? | Not established. The strong-source internal PAI row moved about 82.584% to 82.681% (+0.10 pp), and the PAI run was much longer than its comparison. | Do not call PBScore or the 82.68% row a gain. |
| Is it a compression/Pareto result? | No. A dendrite copies the selected module and adds combination weights. | Compare against matched continuation and ordinary same-budget dense models. |
| Is the old run reproducible under the requested budget? | No. Historical PAI portions took about 104–211 minutes; legacy full lifecycles took 1.62–2.91 hours. | A hard lifecycle cap is a prerequisite. |
| Is a late pointwise dendrite promising? | Mechanistically plausible, but no local PBScore or accuracy evidence exists for that placement. | Test only after the classifier causal screen, or as a separate placement ablation. |
| Is the deep-strided model a next PAI run? | It is a new architecture hypothesis with estimated costs, not an accuracy result. | First train/screen the ordinary base; do not spend PAI budget on a weak base. |

The strongest evidence is therefore a placement-ranking hypothesis, not a
ranking of finished models.

## Evidence-to-proposal gaps

### 1. PBScore is being asked to do too much

The notes correctly describe PBScore as a correlation/selection signal. It is
not a validation metric and cannot establish that the accepted dendrite adds
useful capacity. In particular, a high classifier PBScore does not show that
the classifier branch beats an ordinary extra-width model, or even that its
zero-initialized top weight escaped zero after integration.

For every arm, report all of the following together:

1. zero-dendrite row and accepted-dendrite row from the same PAI search;
2. exact matched no-dendrite continuation;
3. accepted dendrite count, switch epoch, PBScore, and top-weight norm;
4. validation accuracy at the same optimizer-step count;
5. final clean parameter/MAC counts, and then int8 artifact costs.

If the dendrite never integrates, classify the result as “integration did not
occur,” not “dendrite underperformed.” If it integrates but the top-weight norm
and accuracy remain unchanged, classify it as a functional zero/control.

### 2. The classifier arm is not parameter-efficient by itself

For DS-CNN, `.fc` is `Linear(in_channels, 12)`. A copied classifier is nearly
the entire head again, plus PAI combination parameters. The pipeline note's
width-20 calculation is useful: base 1,996 parameters plus one classifier
dendrite gives 2,260. That is an ordinary width-23 two-block model by the same
formula, so an ordinary width-23 control is the minimum credible dense control.

The comparison must not be “width-20 base versus width-20 + dendrite.” That
only demonstrates that more parameters can help. The primary screen should be:

| Arm | Purpose |
| --- | --- |
| width-20 checkpoint, no dendrite, same continuation | isolates training/recovery effect |
| same checkpoint, one `.fc` dendrite | isolates PAI effect at +264 parameters |
| ordinary width-23 model, same final ~2,260 parameters and same short schedule | tests whether PAI is better than dense capacity |

The dense control should be trained from the same declared initialization/data
recipe; if it cannot start from the width-20 checkpoint because topology
differs, say so and treat it as an architecture control rather than a paired
continuation. Report MACs too: equal parameters alone is insufficient for an
edge Pareto claim.

### 3. Late pointwise and classifier branches are not directly rankable

A one-dendrite pointwise copy can be very cheap at narrow width, while a
classifier copy has a much larger relative parameter cost. Conversely, a
pointwise branch evaluates over the feature map and can add substantial MACs
and live activations. Therefore “classifier versus pointwise” should first be
an within-base placement ablation, not a claim that one is the better Pareto
arm. Each placement needs its own same-final-parameter dense control and its
own clean/int8/TFLM measurements.

The pointwise branch is also a higher deployment-risk hypothesis: it introduces
a residual branch over spatial features and must preserve integer-compatible
Mul/Add behavior after cleaning. The existing source wraps leaf convolutions,
while BN/ReLU live outside them; this is compatible enough to probe but is not
evidence that the grouped block is the vendor-preferred unit.

### 4. KD during PB/candidate training is a confound, not a free improvement

The proposed pipelines leave response KD as an optional detail. It must be an
explicit factor. In PAI `p` mode the base weights are frozen, while the
dendrite candidate is trained against the backward/error signal. Adding a KD
term changes that target and can make the candidate correlate with teacher
residuals rather than the ordinary classification error. A later KD resume also
changes the number of optimizer steps and can make most of an apparent gain
occur after integration rather than in the dendrite phase.

For the first screen, use this order:

1. start from one fixed, already-trained checkpoint;
2. use the same loss for PAI and the no-dendrite continuation (CE, or the
   declared KD objective, but not a hidden change);
3. measure the candidate/integration effect before any post-integration KD;
4. if the dendrite passes, run a separately labelled matched post-integration
   KD arm with identical steps for the control.

If the project specifically wants KD as the objective, run a small 2x2 followup
(CE/KD × no-dendrite/dendrite), not a single KD-enhanced dendritic run.

### 5. The 20–30-minute requirement is per arm, not per proposal

The static admission, base/control, PAI continuation, and deployment smoke
budget in the pipeline journal is reasonable only if each arm has an enforced
deadline. A dynamic history loop cannot satisfy that requirement: its default
patience waits for validation plateaus, recreates optimizers after structure
changes, and may try multiple candidates/cycles.

Before any real run, add or verify:

- a monotonic wall-clock deadline that aborts the arm and writes a manifest;
- one predetermined switch and `max_dendrites=1`;
- explicit caps on n-mode, p-mode, integration, and post-integration epochs;
- cached deterministic features and identical batches for every arm;
- checkpoint/optimizer/scheduler/RNG fingerprints at the fork;
- a dry-run proving clean export/reload and MAC counting on a toy batch.

The repository already contains the relevant branch-call profiling repair and
tests; do not spend the next run reimplementing that fix. The remaining timing
problem is experimental control: the nominal short schedule (for example,
3×30+8) is still shorter than observed PAI lifecycles (128+ epochs in the
current probe/history family), and structure changes can reset optimizers and
schedulers. Record actual per-phase elapsed time and optimizer steps, and use a
fixed, replayable phase schedule for the paired continuation.

An interrupted dynamic run is not a valid short-screen result. Its best
validation row is subject to informative stopping and must be labelled
incomplete.

### 6. Proposed deep-strided costs are not yet deployable evidence

The deep-strided Micro/Feature tables are analytical estimates. The current
`DSCNN` constructor creates every block with the default stride (`ds_cnn.py:55–60`;
`DSConvBlock` accepts a stride, but `DSCNN` does not expose per-block strides).
Thus Pipeline 2 requires source changes and fresh shape, checkpoint, export, and
cost tests before it can be compared with the existing XS. Its early stride can
also lose temporal/frequency information. The correct gate is ordinary-base
validation plus clean/int8 export, before invoking PAI.

### 7. Test-set and deployment claims are premature

All current direct DS-CNN dendrite evidence is validation-only and one-seed.
There is no held-out test result, int8 FlatBuffer, TFLM arena allocation, or
RP2040 latency result in the reviewed notes. Keep validation selection separate
from final test reporting. A model is not an RP2040 Pareto point until the
cleaned graph converts, allocates, and invokes with explicit SRAM headroom.

## Recommended priority

### Priority 0 — instrumentation and admission (must happen first)

1. Correct/verify clean-branch MAC accounting; the copied classifier branch
   must count, not only its scale operation.
2. Add the hard per-arm deadline and matched-continuation dispatcher.
3. Verify the exact strong checkpoint hash and teacher/frontend provenance.
4. Add a toy clean/export/reload test and assert the PAI module IDs.

### Priority 1 — smallest worthwhile causal diagnostic

Run repaired SparkNet C8, one seed, one `.fc` dendrite, one candidate, and the
three arms above. Keep the horizon short but identical after the fork. The pass criterion
for continuing is not “PBScore is high”; it is a positive dendrite-minus-
continuation delta, plus no worse validation accuracy than the same-budget
dense control at its measured parameter/MAC point. This is a screening result,
not a paper result.

Record: base/continuation/dendrite accuracies, optimizer steps, elapsed time,
switch/integration state, PBScore, top-weight norm, parameter/MAC counts, and
whether the clean branch matches live PAI outputs.

### Priority 2 — confirm the effect

Only if Priority 1 passes, run three seeds at C8 and then the C8 gate+fc arm
with its ordinary same-budget control. Use the same arm structure. Then run
full-int8 conversion and TFLM
allocation before claiming a Pareto point. A suggested confirmation threshold
of roughly 0.2–0.3 percentage point is a triage rule, not a universal
significance claim; report paired deltas and uncertainty.

### Priority 3 — placement and architecture followups

1. Compare late pointwise versus classifier on the same SparkNet base, each
   with a same-budget dense control.
2. Implement the DS-CNN Micro base and test it without PAI first.
3. Run the deep-strided classifier dendrite only if its ordinary base is
   competitive and the first classifier screen is causal.
4. Defer iterative compression, clustering, CIM/ReRAM, and broad sweeps until
   a clean/int8/TFLM artifact survives.

## Corrections and provenance notes

### PB availability conflict

`agent-perforatedai-primary-research.md` says the public OSS release has PB
disabled/unavailable. That statement is appropriate when discussing the public
repository alone, but it is not the runtime state of this workspace:
`KWS_Model/.venv/bin/python` finds both `perforatedai` and `perforatedbp`, as
also documented in `PAI_KNOWLEDGE.md` (installed `perforatedai 3.2.8` and
`perforatedbp 3.2.7`). Reports must state the exact environment and log the PB
startup banner. Never generalize a local licensed/installed PB result to an OSS
reproduction claim.

### PAI mechanism and cost

The local knowledge note's source reconstruction supports the key accounting:
one accepted dendrite on a module with (P) module parameters and (C) output
channels adds approximately (P+C); with (D) accepted dendrites the derived
total is (D P + D^2 C), excluding non-counted parent/candidate residency.
This is why classifier and pointwise placements must not be compared by PBScore
alone. Parent/candidate copies also affect training memory and wall time even
when excluded from the clean parameter count.

### Source citations used in this review

- `KWS_Model/notes/dendrite-study-v2/PAI_KNOWLEDGE.md`: reconstructed forward,
  zero-initialized top weights, cost formula, phase defaults, narrow-width
  confounds, and artifact diagnostics (especially §§1.2–1.5, 5, 8).
- `KWS_Model/notes/dendrite-study-v2/agent-perforatedai-primary-research.md`:
  official API/repository claims, mode switching, optimizer recreation, and
  export uncertainty (especially §§2–3 and 8).
- `KWS_Model/notes/dendrite-study-v2/DSCNN_DENDRITIC_PIPELINES.md`: local run
  outcomes, cost tables, controls, 20–30-minute protocol, deployment gates, and
  proposed execution order (especially §§3–7 and 10).
- `KWS_Model/src/kws/models/ds_cnn.py:34–71`: current DS-CNN exposes one
  `initial_stride` and constructs all blocks without per-block stride.
- `KWS_Model/src/kws/models/layers.py:26–45`: depthwise/pointwise block,
  grouped depthwise convolution, BN, and in-place ReLU semantics.
- `KWS_Model/notes/dendrite-study-v2/probe_pai_loop.py:90–145`: current probe
  demonstrates explicit mode transitions and optimizer recreation, but its
  fixed epoch loop is not a production wall-clock cap.
- `KWS_Model/src/kws/pipeline.py:565–716`: clean PAI reconstruction requires
  the original metadata, matching checkpoint provenance, and a clean artifact;
  this supports treating export/reload as a gate rather than an assumption.

## Bottom line

The next best run is not “all promising arms.” It is one cheap, paired causal
test that can falsify the dendrite hypothesis inside the requested budget. If
the classifier dendrite does not beat its schedule-matched continuation and
same-budget dense control after clean/int8 checks, report a controlled negative
result and stop expanding the PAI search. If it does, then spend the next
budget on width/seed confirmation and only afterward test late pointwise or
deep-strided designs.
