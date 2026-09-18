# DS-CNN + dendrite screening pipelines for RP2040

Design note, 2026-09-18. This is a read-only design exercise: no training was
run and no source/config/output was changed. The proposals below are
experiments to screen, not claims that a dendrite improves KWS.

## Evidence and accounting conventions

The implementation is `src/kws/models/ds_cnn.py` plus `src/kws/models/layers.py`:

```
stem Conv2d -> BN -> ReLU
  -> [depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU] * N
  -> fixed AvgPool -> Dropout -> Linear(12)
```

The repository's trained DS-CNN checkpoints and feature tests use `(40, 101)`
for the deployment log-mel frontend. Some DS-CNN YAML comments and synthetic
unit tests say `(40, 98)`; that changes MACs but not parameter counts. All cost
numbers in this note use `(40, 101)`, 12 classes, and the repository's
`kws.utils.profile.count_macs` convention. Re-profile the actual checkpoint
before admitting a candidate.

PAI makes a full trainable copy of the selected module. For one accepted
dendrite on a module with `P` trainable parameters and `C` output channels:

```
added parameters = P + C
added MACs       = copied-module MACs + output elements (the top scale)
```

This is exact for the one-dendrite designs below. A later multi-dendrite
experiment must count the triangular dendrite-to-top and dendrite-to-dendrite
weights from the clean exported graph; do not multiply the one-dendrite cost
without re-profiling. PAI's parent/candidate copies are training-time RAM, not
RP2040 deployment cost, but they make on-device training out of scope.

The current SparkNet study is a warning, not a reason to abandon dendrites:
its matched controls beat all tested placements, and the identity-fine-tune
handoff cost about 0.62 percentage points in the complete snapshot. The next
DS-CNN study must therefore preserve the source checkpoint/optimizer state (or
run a clearly matched no-dendrite control) and must never compare only against
PAI's in-search minimum-parameter row.

## Candidate costs

The static costs below were calculated by instantiating the checked-in DS-CNN
class and probing it with batch-1 `(1, 1, 40, 101)` input. `int8 weight bytes`
is the optimistic one-byte weight payload only; FlatBuffer metadata, biases,
scales, and tensor arena are additional.

| Candidate | PAI placement / cap | Params | MACs | Delta vs same base | int8 weight payload |
|---|---|---:|---:|---:|---:|
| XXS control | no PAI, `[18,18]` | 1,830 | 1,450,656 | — | 1.83 KiB |
| P1 | XXS + `.fc`, one dendrite | 2,070 | 1,450,884 | +240 / +228 | 2.07 KiB |
| P2 | XXS + `.blocks.1.pointwise`, one dendrite | 2,172 | 1,799,496 | +342 / +348,840 | 2.17 KiB |
| XS control | no PAI, `[40,40]` | 4,096 | 3,358,320 | — | 4.00 KiB |
| P3 | XS + `.fc`, one dendrite | 4,600 | 3,358,812 | +504 / +492 | 4.49 KiB |

The source modules behind those deltas are:

| Wrapped module | `P` | `C` | copied MACs | top-scale MACs |
|---|---:|---:|---:|---:|
| XXS/XS `.fc` | 228 / 492 | 12 | 216 / 480 | 12 |
| XXS `.blocks.1.pointwise` | 324 | 18 | 330,480 | 18,360 |

For a parameter-matched conventional control, the closest ordinary DS-CNN
widths (same stem and log-mel input) are `[21,21]` at 2,082 parameters and
1,652,652 MACs for P1, `[22,22]` at 2,170 parameters and 1,724,064 MACs for
P2, and `[40,49]` at 4,582 parameters and 3,725,628 MACs for P3. For a
compute-matched comparison, `[23,23]` is 2,260 parameters and 1,797,516 MACs
for P2, while `[40,50]` is 4,636 parameters and 3,766,440 MACs for P3. These
custom widths are controls, not proposed production configs. They prevent a
claim that merely adding parameters to XXS/XS is compression.

## Pipeline P1 — XXS classifier dendrite (lowest deployment risk)

**Recipe.** Train the `ds_cnn_xxs` base (`[18,18]`) on the deployment
`speech_commands_v2.yaml` log-mel-40 frontend, then perforate only `.fc` with
`max_dendrites: 1`, `forward_function: tanh`, and the same validation metric.
The classifier branch adds only 216 MACs plus 12 top-scale operations, so the
RP2040 cost is effectively unchanged. If P1 passes, a separate P1-D2 follow-up
may try two classifier dendrites, but it must use the clean graph's measured
cost and a fresh parameter-matched control.

**Why it might work.** The XS/XXS bottleneck is the 18-dimensional pooled
representation. A classifier dendrite supplies a second learned projection of
that representation and can correct class-specific residuals without copying a
large feature map. This is the safest PAI integration target: `Linear` has a
single tensor output, no BatchNorm state, and the existing quantization path
already handles linear modules.

**Controls.** Keep three comparisons: (a) XXS scratch; (b) a no-dendrite arm
through the identical handoff/schedule, to expose optimizer-reset damage; and
(c) conventional `[21,21]` at nearly the same parameter budget. The exact
source checkpoint, data split, seed, augmentation, optimizer, and scheduler
must be recorded for every arm. Use validation for selection and reserve the
held-out test set for selected arms only.

**Kill/success.** Kill if PAI cannot cleanly export/reload the one-dendrite
graph, if the accepted branch is not actually present in the state dict, or if
int8/TFLM conversion fails. Kill for compression if P1 is more than 0.5 pp
below `[21,21]` in FP32 or int8, or if it fails the RP2040 arena/latency limit.
Call it a Pareto success only if it is at least as accurate as `[21,21]` within
0.2 pp while staying no more expensive in the measured deployed graph; a
three-seed confirmation should show the advantage is at least 0.3 pp or the
confidence interval must exclude a tie. A win over XXS alone is not a
compression result because P1 has more parameters.

## Pipeline P2 — late pointwise feature dendrite

**Recipe.** Start from the same XXS checkpoint and perforate only
`.blocks.1.pointwise`, one dendrite. The wrapped pointwise lies after the last
depthwise/ReLU input and before that block's BN/ReLU, so PAI adds a nonlinear
feature correction at the last spatial feature map. Keep `.fc` unwrapped.

**Why it might work.** Unlike a classifier-only branch, this candidate can
repair frequency/time patterns before global pooling and classification. It is
still small in parameter count: +324 copied pointwise weights and +18 channel
scales. The cost is mostly the second 1x1 feature-map pass, making this a useful
test of whether dendrites need spatially distributed capacity rather than a
larger head. It is deliberately a late block: wrapping early layers would
retain larger, less semantic activations and make the Pico arena harder to
bound.

**Controls and a useful placement sub-arm.** Compare P2 with the same `[22,22]`
parameter control, the `[23,23]` compute control, and the XXS
scratch/no-dendrite-handoff controls. A cheap
diagnostic sub-arm may wrap `.blocks.1.depthwise` instead: its one-dendrite
projection is approximately 2,010 parameters and 1,634,256 MACs (the copied
depthwise pass is 165,240 MACs plus 18,360 scales). That branch cannot mix
channels, so it is a mechanistic negative control for the pointwise hypothesis,
not a second headline result.

**Kill/success.** Kill if the measured clean graph exceeds the team's chosen
4M-MAC or 160 KiB arena ceiling, if the pointwise arm is more than 0.5 pp below
the `[22,22]` control after int8 conversion, or if the depthwise sub-arm is
mistakenly treated as equivalent capacity. A Pareto success requires matching
or beating `[23,23]` within 0.2 pp at lower/equal measured MACs while also
matching `[22,22]` within 0.2 pp at nearly equal parameters; it is
strong evidence if P2 beats both P1 and the conventional control at the same
budget. Expect a larger activation arena than P1: the XXS last feature map is
18x20x51 = 18,360 int8 elements; a conservative branch-retention estimate is
about 53 KiB before allocator reuse and other tensors. Measure the final graph,
not this estimate.

## Pipeline P3 — XS head dendrite (accuracy-recovery arm)

**Recipe.** Use the checked-in `ds_cnn_xs` (`[40,40]`) and perforate only `.fc`,
one dendrite. This is not the first Pico candidate; it asks whether a tiny
classifier branch can recover accuracy when the wider XS encoder is already
close to a useful operating point. Its parameter-matched conventional control
is `[40,49]` (4,582 parameters), not DS-CNN-L.

**Why it might work.** PAI's dendrite method is most plausible when the base is
under-parameterized but not already at its noise floor. P3 retains a 40-channel
feature representation and pays only +492 MACs for the extra head branch. It
also tests whether P1's result is caused by an extremely narrow encoder rather
than by useful classifier residual learning.

**Kill/success.** Kill early if 3.36M MACs or the measured roughly 80 KiB int8 base
activation plus runtime overhead cannot fit the RP2040 application budget. Kill
the dendritic claim if P3 does not beat `[40,49]` within the same 0.2 pp tie
margin, or if its int8 drop exceeds 0.5 pp relative to its FP32 result. The
arm is a success only if it moves the accuracy/parameter frontier versus the
4,582-parameter ordinary control (and preferably the 4,636-parameter
compute-near control); a gain over the 4,096-parameter XS control alone is
expected capacity gain, not evidence of parameter efficiency.

## Optional P4 — MFCC-32 transferability check

If the application has a validated MFCC frontend, repeat P1 with
`speech_commands_v2_mfcc32.yaml` or the paper-balanced MFCC-32 config, keeping
the teacher/control frontend identical. The XXS cost is 1,830 parameters and
1,160,568 MACs at `(32,101)`; one `.fc` dendrite is 2,070 parameters and
1,160,796 MACs. The int8 base activation estimate is about 29 KiB.

This is a frontend portability check, not a free improvement: MFCC computation
may cost more on an RP2040 than log-mel, and the repository's MFCC-32 DS-CNN-L
run was only 91.76% validation and is not a trustworthy teacher. Do not mix
P4's numbers with the log-mel arms or reuse a log-mel checkpoint. Keep P4 only
if feature-extraction cost, FP32 accuracy, and TFLM operator support are all
measured together.

## Time-boxed screening plan (20–30 minutes)

This is intentionally an integration screen, not a final five-seed study.
Use cached deterministic features, a fixed seed, batch 256, and no held-out
test evaluation until selection:

1. **Static admission (1–2 min).** Instantiate each candidate and its custom
   control, run `profile_model`, verify module IDs, and reject any graph over
   the parameter/MAC/arena ceiling before training.
2. **Baseline/control warm-up (8–10 min).** Train XXS, `[21,21]`, and `[22,22]`
   for the same short screen horizon (for example, 8–12 epochs). Train XS and
   `[40,49]` only if the P3 budget is acceptable. Save the best checkpoint and
   optimizer/scheduler state; do not perform the SparkNet-style identity
   fine-tune without its no-dendrite control.
3. **PAI cycles (8–12 min).** Warm-start P1/P2/P3 from their paired base
   checkpoints. Use one fixed switch with roughly 4 p-mode and 4 n-mode epochs,
   `max_dendrites: 1`, and a short post-integration adaptation. Record switch
   epochs, accepted dendrite count, PB correlation, and whether the optimizer
   was recreated after restructuring. If the host cannot complete this in the
   budget, screen P1 first and run P2/P3 in separate sessions; do not shorten
   only the dendritic arm.
4. **Deployment gate (3–5 min).** Clean the PAI graph, run PyTorch/ONNX parity,
   run the repository's QAT path (including
   `replace_clean_pai_modules`), convert to int8, re-profile parameters/MACs/
   activations, and enumerate operators. A candidate that cannot produce a
   clean graph is a failed integration regardless of validation accuracy.

The short horizon is suitable for ranking integration behavior only. Any arm
that survives should be rerun with at least three seeds, the exact fixed
validation/test protocol, and enough base training to establish convergence.

## Quantization and TFLM implications

PAI's live training wrapper is not an RP2040 artifact. Deployment must use the
clean PAI graph, verify its state dict and output parity, then quantize the
clean graph. The repository's QAT code has an explicit
`QuantizableDendriticResidual` because ordinary eager int8 conversion cannot
multiply an int8 branch tensor by a floating per-channel PAI coefficient. For
P1/P3, the linear branch is easy to inspect; for P2, Conv+BN fusion must be
performed independently in every copied branch and the residual wrapper must
remain explicit.

The expected deployed graph contains the original branch, a copied Conv/Linear
branch, a tanh (or configured PAI activation), per-channel scale, and add. TFLM
support for the resulting exact operator sequence—not PyTorch or ONNX success
alone—must be checked in a pinned TensorFlow Lite Micro build. If tanh or the
residual arithmetic is not available as a fully integer kernel, either reject
the arm or replace it with a documented integer-compatible graph and retrain;
silently dequantizing the branch invalidates the RP2040 memory/latency claim.

Use the actual FlatBuffer size and arena measurement for admission. As rough
screening bounds, the XXS base's int8 feature-map peak is about 36 KiB for
log-mel-40 and 29 KiB for MFCC-32; P2's retained parallel feature map can be
about 53 KiB before allocator reuse. Add stack, audio buffers, frontend state,
TFLM scratch, bias/scales, and application code before comparing with the
RP2040's 264 KiB SRAM. Quantized one-byte weight payloads are useful
bookkeeping, not a flash or RAM guarantee.

## Decision rule

Report each arm as `scratch`, `matched no-dendrite`, `dendritic FP32`, and
`dendritic int8`, with validation and held-out test results kept separate. A
credible dendrite compression result needs all of the following:

* a matched conventional model at the same or lower deployed parameters;
* no larger measured MAC/arena cost than that control (unless the accuracy
  gain is explicitly traded and shown on the Pareto plot);
* no unexplained optimizer/handoff penalty relative to the matched no-dendrite
  arm;
* clean export, fully integer TFLM execution, and a real RP2040 benchmark; and
* a replicated accuracy advantage rather than a single validation seed.

If none passes, ship the ordinary DS-CNN/SparkNet Pareto point and retain the
negative result. That outcome is scientifically useful and avoids repeating
the current SparkNet study's unsupported “dendrites compress” conclusion.
