# Repository/results audit: SparkNet, PerforatedAI, and Pico compression

Audited 2026-09-18, read-only. Paths below are relative to `KWS_Model/` unless
otherwise stated. Numerical claims are copied from run artifacts, not inferred
from checkpoint filenames.

## Executive conclusion

The strongest reproducible result is the paper-faithful, no-dendrite SparkNet
scratch frontier on MFCC-32 balanced Speech Commands v2: C12 = **3,400
parameters, 277,124 MACs, 93.88% validation accuracy (sd 0.46 pp, five
seeds)**. The current dendritic arm recipe is not an accuracy-preserving
compression method: across completed study cells it first applies a harmful
40-epoch identity fine-tune, then adds one dendrite while usually ending below
the paired scratch checkpoint. The dendrite-only within-search gain is tiny and
is not a causal control comparison. No selected dendritic arm has a held-out
test result; the study's test report evaluates only scratch and no-dendrite
control targets.

For a Pico target, start with an ordinary quantized/pruned SparkNet checkpoint
and treat dendrites as a follow-up capacity experiment. Do not claim that the
current artifacts demonstrate accuracy-per-parameter improvement from PAI.

## 1. Data, splits, and architectures

### Common KWS data

All runs use Google Speech Commands v2, ten target words plus pooled
`_unknown_` and `_silence_` (12 classes). The two materially different front
ends are:

* `configs/data/speech_commands_v2.yaml`: deployment-oriented 40-bin log-mel,
  30 ms window/10 ms hop, unknown and silence ratios 2.0.
* `configs/data/speech_commands_v2_mfcc32.yaml`: MFCC-32, 25 ms/10 ms, log
  energies, ratios 2.0.
* `configs/data/speech_commands_v2_mfcc32_paper.yaml`: paper-faithful MFCC-32,
  balanced manifests, ratios 1.0, materialized silence. This is the current
  dendrite study and paper-replication split.

The normal run objective is validation accuracy; `testing_list.txt` is the
held-out test split. Study arm reports explicitly say
`selection_split: validation` and `test_split_used: false`, for example
`outputs/sparknet-dendritic-study-v2/arms/pointwise/c12-seed0/reports/sparknet_dendritic_prune_experiment.yaml`.

### Model families

| Family/run | Architecture and setting | Parameter/MAC evidence |
|---|---|---:|
| Paper SparkNet | C16, C12, C10, C8, C6, C4, C2; gate channels 32; MFCC-32 paper recipe | C16 4,636 params/396,304 MACs; study scratch C12/C10/C8/C6/C4/C2 = 3,400/277,124; 2,854/224,806; 2,356/177,336; 1,906/134,714; 1,504/96,940; 1,150/64,014 (current generated aggregation `/tmp/current-aggregate-study.json`, configs under `configs/model/`). |
| SparkNet arm | Same widths, identity prune fine-tune, then PAI placement: `pointwise`, `fc`, `gate_conv`, `depthwise`, or `control` | C12 one-dendrite deployed costs: pointwise 3,712/308,636; fc 3,808/277,520; gate_conv 3,848/319,140; depthwise 4,000/337,724; control 3,400/277,124 (`.../reports/...yaml`). |
| DS-CNN | DS-CNN-L teacher and DS-CNN-XS student, default 40-bin log-mel in the legacy pipeline; one MFCC-32 DS-CNN-L retrain | DS-CNN-L 469,604 params (`ENHANCEMENTS.md` §1.1); DS-CNN-XS config says 4,096 params, though the pipeline's deployed pruned candidates are much smaller. |

## 2. Current `sparknet-dendritic-study-v2` matrix

The study contains **30 scratch cells** (six widths × five seeds) and **150 arm
cells** (five placements × six widths × five seeds). The fresh read-only
aggregation at `2026-09-18T19:11:28+00:00` reports **30/30 scratch and 150/150
arm cells complete**. The committed `notes/dendrite-study-v2/aggregate_study.json`
is an earlier stale snapshot (72 complete, 2 running, 76 not started), so all
current study numbers below come from `/tmp/current-aggregate-study.json` and
the underlying per-cell artifacts. This file was produced read-only with
`KWS_Model/.venv/bin/python notes/dendrite-study-v2/aggregate_study.py` and an
output path under `/tmp`; no output was rewritten. The scratch models train 200 epochs with
the paper recipe. Dendritic arms use the paired scratch best checkpoint,
40-epoch identity fine-tune, PAI, and usually an 8-epoch resume.

### Scratch frontier (best validation accuracy, mean ± sample sd)

| Width | Params | MACs | Best validation |
|---:|---:|---:|---:|
| C12 | 3,400 | 277,124 | **93.88% ± 0.46** |
| C10 | 2,854 | 224,806 | **92.80% ± 0.33** |
| C8 | 2,356 | 177,336 | **91.23% ± 0.32** |
| C6 | 1,906 | 134,714 | **87.87% ± 0.68** |
| C4 | 1,504 | 96,940 | **82.87% ± 0.86** |
| C2 | 1,150 | 64,014 | **63.34% ± 1.49** |

Source: `/tmp/current-aggregate-study.json.summary.scratch`; individual source records are
`outputs/sparknet-dendritic-study-v2/scratch/c{width}-seed{seed}/metrics/summaries.yaml`.

### Completed dendritic arm summary

Values below are mean final validation accuracy over completed seeds; `Δ
scratch` is paired final arm minus that seed's scratch best. They are not test
accuracies.

| Arm | Width(s) complete | Final val | Δ scratch | Params / MACs at C12 |
|---|---|---:|---:|---:|
| pointwise | C12,C10,C8,C6,C4,C2 (5 each) | 93.48, 92.26, 90.73, 87.37, 82.05, 60.92% | −0.40, −0.54, −0.50, −0.50, −0.81, −2.42 pp | 3,712 / 308,636 |
| fc | C12,C10,C8,C6,C4,C2 (5 each) | 93.51, 92.28, 90.72, 87.38, 82.11, 60.93% | −0.36, −0.52, −0.50, −0.49, −0.76, −2.41 pp | 3,808 / 277,520 |
| gate_conv | C12,C10,C8,C6,C4,C2 (5 each) | 93.47, 92.26, 90.73, 87.37, 82.04, 60.96% | −0.41, −0.54, −0.50, −0.51, −0.83, −2.38 pp | 3,848 / 319,140 |
| depthwise | C12,C10,C8,C6,C4,C2 (5 each) | 93.49, 92.27, 90.75, 87.37, 82.02, 60.94% | −0.38, −0.53, −0.48, −0.50, −0.85, −2.40 pp | 4,000 / 337,724 |
| control | C12,C10,C8,C6,C4,C2 (5 each) | 93.69, 92.63, 90.89, 87.61, 82.35, 61.32% | −0.19, −0.17, −0.34, −0.26, −0.51, −2.02 pp | 3,400 / 277,124 |

Source: `/tmp/current-aggregate-study.json.summary.arms` and the per-cell
reports under `outputs/sparknet-dendritic-study-v2/arms/`. Every row now has
five seeds. The control is the best validation arm at every width; all four
dendritic placements trail it by about 0.2–0.5 pp at C6–C12 and by about 2.4
pp at C2.

### Selector and held-out test outcome

`selection/selected_arms.json` now contains 60 targets: the selector chose
`control` at **every width** by mean validation accuracy across five seeds,
and includes 30 scratch references. It explicitly records
`selection_split: validation` and `test_split_used: false`. Only after this
selection did `selection/test_report.json` evaluate the scratch/control
targets on test (60 evaluations, five seeds each); no dendritic arm was sent
to test. The resulting held-out means are:

| Width | Scratch test | Control test | Params |
|---:|---:|---:|---:|
| C12 | 93.73% ± 0.17 | **93.34% ± 0.38** | 3,400 |
| C10 | 92.51% ± 0.40 | 92.37% ± 0.52 | 2,854 |
| C8 | 90.80% ± 0.36 | 90.58% ± 0.30 | 2,356 |
| C6 | 86.88% ± 0.54 | 86.41% ± 0.49 | 1,906 |
| C4 | 82.25% ± 0.93 | 81.72% ± 0.70 | 1,504 |
| C2 | 61.53% ± 1.31 | 60.01% ± 1.78 | 1,150 |

The test report's external paper reference is 95.70% ± 0.17%; it is a
reference row, not a study arm. Thus the current repository does support
scratch/control test baselines, but still does not support a dendrite test-set
claim.

### Identity-fine-tune confound

The paired pointwise analysis in `ENHANCEMENTS.md` §3.1 is the clearest causal
diagnostic. Its earlier 28-cell snapshot reported identity fine-tuning at
−0.546 pp and negative in 26/28 cells. Recomputing all 30 current pointwise
cells gives identity fine-tuning **−0.619 pp**, negative in 28/30 cells; its
damage ranges from about −0.16 pp at C12 to −2.40 pp at C2. The subsequent
dendrite phase averages another **−0.243 pp** (positive in only 4/30 cells).
The handoff changes SGD to AdamW, restarts at 1e−3, changes weight decay and
label smoothing, and changes scheduler behavior (`ENHANCEMENTS.md` §3.1).
Therefore a negative arm result is not evidence that dendrites intrinsically
hurt; it is evidence that this complete recipe does not preserve the source
checkpoint.

## 3. PerforatedAI checklist and phase evidence

The study configuration is consistent across arm configs, e.g.
`arms/pointwise/c12-seed0/study_experiment.yaml`:

* `max_dendrites: 1`, history mode, `history_lookback: 8`,
  `n_epochs_to_switch: 10`;
* improvement thresholds `[0.005, 0.002, 0.001]`, candidate initialization
  multiplier `0.01`, 40 initial-correlation batches, three tries;
* `forward_function: tanh`, `conversion: module_ids`,
  `post_integration_lr_multiplier: 0.25`, BN statistics frozen in p mode,
  no KD, and `enforce_base_weight_freeze: false`.

For the representative C12 pointwise run, the PAI JSONL shows 128 epochs and
`n → p → n → n`; switch epochs are 38 and 60, the accepted dendrite adds 288
copied plus 24 residual/scale parameters, and p-mode has zero base parameters
in the optimizer (`.../metrics/sparsity/sparknet_c12_multilayer/pai.jsonl`).
Across the 120 completed dendritic cells, 117 integrated one dendrite and
three integrated none; mode sequences are 114 `npnn`, 3 `npnpnn`, and 3
`npnpnpnn`. The 30 controls have no PAI lifecycle. Resume status is
`complete` for 24 dendritic cells and `no_improvement` for 96.

### Required CSV inventory

PAI's filenames concatenate the save name and suffix (`...Scores.csv`,
`...switch_epochs.csv`), so an exact lowercase glob such as `*_scores.csv`
would miss most files.

Within study-v2, canonical (non-`beforeSwitch`, non-`before_final`,
non-`noImprove`) artifacts are:

| Artifact | Count | Finding |
|---|---:|---|
| `*Scores.csv` | 150 | 120 dendritic lifecycle score traces plus 30 control traces |
| `*_best_arch_scores.csv` | 121 | 120 dendritic architecture comparisons plus one stray control snapshot |
| `*switch_epochs.csv` | 121 | Same stray-control caveat; controls bypass PAI |
| `*learning_rate.csv` | 121 | Same stray-control caveat |
| `*Best PBScores.csv` | 120 | One per dendritic cell; none for controls |
| `*noImprove_lr*` | 96 files / 6 cells | failed dendrite-addition retries; cells are pointwise C8-S0, fc C8-S0/C2-S1/C4-S2, gate_conv C8-S0, depthwise C8-S0 |
| `*_train_scores.csv` | 0 | absent |

`*_best_arch_scores.csv` uses a minimum-parameter row as a within-search
reference, not a separately trained no-dendrite control. The current PAI
internals show positive best-architecture-vs-minimum-row gains in some cells,
but these are generally fractions of a percentage point and are not causal.
The valid paired comparisons are the scratch checkpoint and the completed
control arm; every placement/width row is negative against both at the current
recipe's relevant no-dendrite reference.

Switches generally occur around epochs 38–46 and 54–75. In the representative
run, validation is 0.94106 immediately before the first switch and the final
best architecture is 0.94286 versus 0.94263 for the minimum-parameter row;
the `Scores.csv`, switch CSV, and `best_arch_scores.csv` are
`.../pai/candidates/sparknet_c12_multilayer/`. Switches therefore coincide
with at most small, noisy post-switch changes, not a demonstrated step-change
in accuracy. The `noImprove_lr` files are critical: per the PAI results
checklist they mean no dendrite was added for that retry, not “a low-quality
dendrite.”

### PB correlation scores

Canonical `Best PBScores.csv` traces show a strong placement distinction:

* `fc`: final best-ever scores mean about **0.132** (all 30 cells above 0.02).
* pointwise: both late pointwise modules remain below 0.02 (roughly 0.006–0.008
  means across trace rows).
* gate_conv: roughly 0.0095 mean, below 0.02.
* depthwise: roughly 0.006–0.007 mean; most rows are below 0.01.

Under the skill's practical rubric (>0.02 is good placement, <0.01 is weak),
`fc` is the only clearly aligned placement. This is a placement/correlation
signal, not an accuracy win: fc's completed study cells still end below paired
scratch. The score-column evidence is in each arm's canonical `*Best PBScores.csv`.

There are no dedicated PAI train-score CSVs, so the requested train-vs-validation
overfitting check cannot be performed from the expected file. JSONL and
`best_arch_scores.csv` do contain train values; for example C12 pointwise's
best architecture has train 0.93974 versus validation 0.94286, but its train
loss is on the SparkNet task-loss scale and should not be compared directly to
validation cross-entropy.

### Cost evidence

Study reports provide projected logical 8-bit weight bytes, conservative
activation peak bytes, and CPU latency. Representative C12 pointwise is
3,712 weight bytes, 6,464 activation bytes, 1.764 ms mean / 1.731 ms p50 /
1.946 ms p90 (`.../reports/...yaml`). These are estimates/measurements under
the report's stated methods, not a Pico-board benchmark. The legacy compression
report similarly labels weight bytes as projected logical precision and reports
CPU timings; it does not measure RAM, flash footprint, energy, or latency on a
Pico board. No output artifact contains a board-side RAM/flash measurement, so
those claims remain absent rather than zero.

## 4. Earlier experiments and failures

### Phase-B supervised/KD comparisons

All are 200-epoch validation runs, generally one seed, and should not be mixed
with the paper-balanced study:

| Run | Front end / model | Best val | Final val | Notes |
|---|---|---:|---:|---|
| `phase_b/sparknet_c12_speech_commands_v2_g32` | log-mel-40, C12 | 91.78% | 91.73% | 3,400-param SparkNet |
| `phase_b/sparknet_c12_speech_commands_v2_g40` | log-mel-40, C12/gate 40 | 92.11% | 92.07% | wider gate |
| `phase_b/sparknet_c12_speech_commands_v2_mfcc32_g32` | MFCC-32, C12 | **92.61%** | 92.42% | source for older C12 dendrite runs |
| `phase_b/sparknet_c12_speech_commands_v2_mfcc32_g40` | MFCC-32, C12/gate 40 | 91.40% | 91.26% | worse than g32 |
| `phase_b/r1_c12_light` | log-mel-40, C12 | 91.98% | 91.90% | light recipe |
| `phase_b/teacher_ds_cnn_l_mfcc32_nw0` | MFCC-32 DS-CNN-L | 91.76% | 91.01% | 469,604-param teacher is weaker than C12 student |
| `phase_b/sparknet_c12_mfcc32_g32_kd_annealed` | MFCC-32 C12 + KD | 91.24% | 91.11% | −1.37 pp versus matched 92.61% control |
| `phase_b/step2_sparknet_c12_light_kd_t2_20260914T013331Z` | log-mel-40 C12 + strong teacher, T=2 | 92.15% | 91.92% | exact tie with no-KD control at 92.15% |

Source values: each run's `metrics/summaries.yaml`; KD diagnostics are in the
JSONL under `metrics/student/distill.jsonl` and summarized in
`notes/dendrite-study-v2/ENHANCEMENTS.md` §1. The KD result supports “current
KD is not useful here,” not “KD can never help.”

`outputs/diagnostics/kd_20260914/` and its `replicate_27182/` child are
diagnostic replications, not new trained models: the strong 40-bin teacher is
about 98.5–98.7% accurate on sampled training views, with roughly 0.90 mean
confidence and 0.49–0.50 nats entropy. They support the claim that the
strong-teacher KD path was functioning; they do not provide a new test result.

### Older C12/C16 dendrite runs

* `sparknet-c12-dendritic-prune-no-kd-fc-only-d3`: one partially completed
  C12 run from the 92.61% MFCC source, fc-only, `max_dendrites: 3`, 25-epoch
  switch interval, thresholds `[0.001,0.0001,0]`; width-10 candidate reached
  92.32% validation at 4,114 params, but the manifest remained interrupted/
  running and other widths were incomplete.
* `sparknet-c12-dendritic-prune-no-kd-unlimited`: same source, modules
  `.blocks.3`, `.gate_conv`, `.fc`, unlimited dendrites; no completed candidate
  result, manifest still running. Its many snapshot CSVs must not be treated as
  independent experiments.
* `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed0..4`: five completed
  no-KD, MFCC-32 paper C16 source runs (source 4,636 params/396,304 MACs),
  fc-only, up to three dendrites. Reports contain pruned C12/C10/C8 candidate
  validation results: roughly C12 92.31–93.12%, C10 90.64–92.80%, C8
  87.11–88.86%, with deployed parameters varying about 2,764–4,228. All are
  validation-only; they are not matched five-seed arm comparisons at each
  width.
* `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed0..4`: gate-conv C16
  runs were still marked running with no candidate result in reports at the
  snapshot.

### Legacy `compression-run`

`outputs/compression-run/` is a separate, interrupted `kws.pipeline
--extreme-prune` using default log-mel-40 Speech Commands, a 469,604-param
DS-CNN-L teacher, a distilled DS-CNN-XS source, and a 4,096-param/1.5M-MAC
budget. The report contains:

* conventional w18: **65.21% val**, 1,830 params, 1,450,656 MACs;
* dendritic w18 classifier: **70.16% val**, 2,070 params, 1,450,668 MACs,
  0.290 ms mean CPU latency;
* conventional w14: 64.69%, 1,522 params, 1,209,888 MACs;
* dendritic w14 classifier: 66.44%, 1,714 params, 1,209,900 MACs;
* wider feature+dendrite placements were skipped by the MAC budget; w10/w6
  dendritic candidates were planned but not completed.

These are poor KWS accuracies for a Pico product target and use a different
frontend/recipe. They are evidence of the budget accounting, not evidence that
the legacy pipeline is production-ready.

### Failed `full-run-20260912T063022Z`

This older end-to-end DS-CNN pipeline reused a 97.69% val teacher, an 85.07%
val distilled student, and a w18 PAI candidate at 82.68% val, 2,070 params,
1,450,668 MACs. It never became a valid end-to-end result: manifest invocations
record missing PAI native checkpoint, RNG-state type errors, debugger exits,
resume recipe mismatch, external stops, an index error, and finally
`PAI cleanup module 'fc' has no two-branch layer_array` and recipe-fingerprint
mismatches (`outputs/full-run-20260912T063022Z/manifest.yaml`). No downstream
cluster/quantize/benchmark claim is supported.

### Paper replication/test evidence

`outputs/sparknet-paper-replication/c16-seed0..4` are five completed 200-epoch
MFCC-32 paper C16 runs: validation mean about **94.97%** (individual best
94.69–95.32%) and local `test_report_fixedseed0.json` mean **94.81% ± 0.22 pp**.
Those reports are useful baselines but their test evaluation is separate from
study-v2 selection. The released checkpoint evaluations are **93.351–93.354%**
test (`released-c16-eval` and `released-c16-exact-count-eval`), demonstrating
that protocol/checkpoint provenance matters.

## 5. Claims: supported versus unsupported

Supported:

* Paper-balanced SparkNet scratch accuracy/parameter/MAC frontier above.
* PAI actually wraps the requested module IDs, enters p mode, freezes base
  optimizer parameters, and exports clean checkpoints (see
  `notes/dendrite-study-v2/INTEGRATION_AUDIT.md` and PAI JSONL).
* `fc` has much stronger PB correlation scores than the other tested placements.
* Current KD comparisons are neutral or negative, with the caveat of one seed
  and frontend differences.
* The identity-fine-tune handoff is a large, measured confound.

Unsupported:

* “Dendrites improve SparkNet accuracy” as a causal or test-set claim: the
  completed matched no-dendrite control beats every dendritic placement at
  every width on validation, and test evaluation has only scratch/control
  targets.
* “The best architecture row proves dendrite gain”: it is an in-search row and
  not an independent no-dendrite training control.
* Any RAM/latency number as a Pico hardware guarantee: current numbers are
  projected/logical bytes and host-CPU measurements.
* Any end-to-end pruning + clustering + quantization claim from the failed
  full-run.
* Any result from a manifest marked running/interrupted/failed as a completed
  experiment.

## 6. Five plausible Pico compression pipelines

1. **Recommended baseline: C12 paper SparkNet, no PAI.** Keep 3,400 params /
   277k MACs, quantize after validation, and benchmark the actual Pico build.
   It has the best stable study accuracy and a held-out study test mean around
   93.34% for the C12 control targets. This is the only low-risk starting point.
2. **Balanced reduction: C10 no PAI.** 2,854 params / 224,806 MACs and 92.80%
   validation mean. Use if the measured memory/latency budget requires about
   16% fewer parameters than C12; the validation drop is about 1.08 pp.
3. **Aggressive conventional: C8 no PAI.** 2,356 params / 177,336 MACs and
   91.23% validation mean. This is a plausible Pico point only if roughly 91%
   is acceptable; it is better supported than any current dendritic claim.
4. **Post-handoff PAI ablation: C8/C10 fc-only, one dendrite.** `fc` has the
   strongest PBScore signal and costs roughly +408 params at C12, but current
   study fc cells still lose about 0.5 pp versus scratch. Re-run with repaired
   identity handoff, matched scratch controls, five seeds, and test only after
   selection; treat this as an experiment, not a deployable result.
5. **Legacy DS-CNN classifier-dendrite route.** The w14/w18 classifier-only
   runs fit the 1.5M-MAC budget and measured ~0.28–0.29 ms host-CPU latency,
   but only reach 66–70% validation. Keep it only as a very-low-accuracy
   fallback or a code path for future teacher/recipe repair; it is not a
   credible replacement for SparkNet for normal KWS.

## Git/worktree note

The read-only status check showed pre-existing untracked note files, including
the other `dendrite-study-v2` notes. No source, config, output, checkpoint, or
user change was modified by this audit; this report is the sole new artifact.
