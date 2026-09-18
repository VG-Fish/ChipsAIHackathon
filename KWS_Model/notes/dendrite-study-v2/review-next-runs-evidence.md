# Evidence review: what to run next

**Reviewed:** 2026-09-18. This is a read-only reconciliation of
`DATA_INVENTORY.md`, `ENHANCEMENTS.md`, `INTEGRATION_AUDIT.md`,
`agent-repo-results-audit.md`, and `agent-dscnn-results-audit.md`. The raw
artifacts below were checked without training or source/config changes.

## Decision-level conclusion

The current frontier is paper-recipe SparkNet, not DS-CNN+dendrites. The fresh
aggregation (`/tmp/current-aggregate-study.json`, generated from the per-cell
reports) shows a five-seed C12 scratch mean of **93.88% validation**, at
**3,400 parameters / 277,124 MACs**. After the study's handoff and PAI phases,
the matched C12 control is **93.69% validation** and every dendritic placement
is lower: pointwise 93.48%, fc 93.51%, gate_conv 93.47%, depthwise 93.49%.
The selected scratch/control test report gives **93.73% / 93.34%** respectively
on held-out test. No dendritic arm has a held-out test result.

This materially outranks the DS-CNN plan's plausible deployment candidates:
the 4,096-parameter DS-CNN-XS is **83.64% test** (`reports/phase_a/xs_test.json`)
and the older DS-CNN dendritic rows are validation-only, recipe-confounded,
and below the SparkNet frontier. DS-CNN should therefore be a controlled
cross-family baseline, not the primary compression path.

## Reconciled evidence

### SparkNet frontier and causal status

The current study has 30 scratch cells and 150 arm cells, all complete. The
frontier (mean best validation over five seeds) is:

| Model | Params | MACs | Best validation | Held-out test |
|---|---:|---:|---:|---:|
| SparkNet C12 scratch | 3,400 | 277,124 | **93.88% ± 0.46** | **93.73% ± 0.17** |
| SparkNet C10 scratch | 2,854 | 224,806 | 92.80% ± 0.33 | 92.51% ± 0.40 |
| SparkNet C8 scratch | 2,356 | 177,336 | 91.23% ± 0.32 | 90.80% ± 0.36 |
| SparkNet C6 scratch | 1,906 | 134,714 | 87.87% ± 0.68 | 86.88% ± 0.54 |
| SparkNet C4 scratch | 1,504 | 96,940 | 82.87% ± 0.86 | 82.25% ± 0.93 |
| SparkNet C2 scratch | 1,150 | 64,014 | 63.34% ± 1.49 | 61.53% ± 1.31 |

Sources: `outputs/sparknet-dendritic-study-v2/scratch/c{12,10,8,6,4,2}-seed*/metrics/summaries.yaml`,
`outputs/sparknet-dendritic-study-v2/selection/test_report.json`, and the
fresh aggregation. The external paper reference (95.70% ± 0.17%) is not a
study arm.

The arm ranking is not a dendrite win hidden by a bad summary. The selector
chose **control at every width** on validation; the arm report at
`arms/fc/c12-seed0/reports/sparknet_dendritic_prune_experiment.yaml` records
the same scratch source, identity fine-tune baseline, zero-dendrite row, and
deployed cost. Its canonical PAI CSV shows a within-search improvement from
3,400/0.942632 to 3,796/0.943532, but that is not an independently trained
control. The raw canonical file is
`arms/fc/c12-seed0/pai/candidates/sparknet_c12_multilayer/sparknet_c12_multilayer_best_arch_scores.csv`.
The final arm still loses to the paired scratch/control means.

### The dominant confound

The identity fine-tune is not identity in optimization terms. Across current
pointwise cells it averages about **−0.619 pp** from scratch (negative in
28/30); the subsequent dendrite phase averages another **−0.243 pp**. The
earlier 28-cell calculation in `ENHANCEMENTS.md` was −0.546 pp and −0.214 pp;
the difference is completion of the remaining cells, not conflicting results.
The handoff changes SGD to AdamW, learning rate, weight decay, label smoothing,
scheduler, and warmup. Thus “dendrites hurt” is not identified until the
source checkpoint can survive that handoff.

The integration audit finds no verified PAI lifecycle failure. It does identify
future hazards: AdamW `weight_decay: 1e-4`, implicit output dimensions, and the
vendor-guard escape hatch. These are secondary to the measured handoff loss.
The raw PAI timing file
`arms/fc/c12-seed0/pai/candidates/sparknet_c12_multilayer/sparknet_c12_multilayerTimes.csv`
confirms that a full arm is materially more expensive than the diagnostic
identity phase; the study's measured medians are 10.2 min for scratch and
11.0 min for a full arm, with the identity phase about 2.3 min.

### DS-CNN evidence and what it does not establish

The strong 40-bin DS-CNN-L is **97.6858% test / 469,604 parameters**
(`reports/ds_cnn_l_teacher_current.json`). The warm-started XS is only
**85.1109% validation / 83.6431% test / 4,096 parameters**
(`reports/phase_a/xs_val.json`, `xs_test.json`). The MFCC-32 DS-CNN-L retrain
is **91.7647% validation** after about 7.02 h, weaker than SparkNet C12, so its
−1.37 pp KD comparison is a weak-teacher result, not evidence against all KD.
The strong-teacher log-mel KD comparison is an exact one-seed tie at 92.15%
with unmatched initialization. Keeping `--no-KD` for the current SparkNet
study is justified, but “KD cannot help” is not.

The historical DS-CNN PAI rows (legacy log-mel recipe) are validation-only:
w18 conventional 65.21% versus fc-dendritic 70.16% at 1,830 versus 2,070
parameters; w14 64.69% versus 66.44% at 1,522 versus 1,714 parameters. They
are useful lifecycle evidence, not a fair same-cost causal result or a board
measurement. The later full-run w18 82.68% row is likewise not end-to-end
valid because of failed/restarted invocations and no matched no-PAI control.

## What the earlier DS-CNN plan missed

1. **It ranked architecture plausibility before the measured cross-family
   frontier.** At roughly the same budget, SparkNet C12 has 3,400 params,
   277k MACs, and 93.73% test; DS-CNN-XS has 4,096 params and 83.64% test.
   A DS-CNN experiment is still useful, but it cannot be the default winner.
2. **It treated `fc` placement as a plausible DS-CNN next step without first
   measuring an ordinary same-frontend DS-CNN baseline.** The audit itself
   says the direct candidates have no held-out test and no independently
   trained same-cost conventional control.
3. **It did not elevate the SparkNet handoff ablation to a prerequisite.** The
   −0.6 pp identity loss is larger than the observed dendrite effect and is
   width-dependent, especially catastrophic at C2.
4. **It risks spending on KD before fixing teacher quality and frontend
   comparability.** The 91.76% MFCC teacher is below SparkNet C12; the 97.69%
   teacher is log-mel and cannot be paired directly with MFCC-32.
5. **It did not reserve a held-out test evaluation for a selected dendritic
   candidate.** Current test evidence can rank scratch/control only.

## Three next experiments (in order)

### 1. Identity-handoff ablation (cheap prerequisite)

Run 40-epoch identity fine-tunes from existing scratch checkpoints at C12, C6,
and C2, seeds 0–4: current AdamW recipe, repaired scratch-like SGD recipe,
and zero-epoch pass-through. This is the documented 45-run design in
`ENHANCEMENTS.md`; expected cost is about **1.7 h**. Use validation only and
preserve the five seeds. Adopt the recipe that matches pass-through, then
freshly rerun any dendrite comparison from fresh scratch baselines.

Required change: add a separate v3 config/driver path; do not mutate the live
v2 matrix or reuse v2 arm rankings as causal evidence.

### 2. Matched PAI ablation after handoff repair

At C12, C6, and C2, compare **control vs pointwise** under the winning handoff
recipe, five seeds, with equal epochs, data, selection rule, and deployed-cost
accounting. Include a paired `weight_decay=0` condition (the integration audit's
P1) and make PAI output dimensions explicit as `[-1, 0, -1, -1]` before runs.
This is the minimum experiment that can say whether pointwise dendrites help
after the largest confound is removed. Do not add feature placements until this
passes.

Required change: add a conversion/parameter-accounting regression test and
record canonical PAI CSVs only; treat `*_beforeSwitch*` and `noImprove_lr`
snapshots as diagnostics, not extra experiments.

### 3. One frozen selected-arm test and deployment comparison

After experiment 2, select only on validation, then evaluate the winning
SparkNet arm and its scratch/control comparator once on held-out test. Report
params, MACs, weight bytes, host timing, and (separately) an actual RP2040
benchmark. In parallel, train one fair DS-CNN baseline on the exact intended
MFCC frontend with the same seed count and held-out protocol; it is a sanity
check against the SparkNet frontier, not a replacement for it.

Required change: enforce `test_split_used: false` during selection and consume
the test budget only after the arm is frozen. Do not call host CPU milliseconds
an RP2040 result.

## Bottom line for planning

Prioritize SparkNet v3 handoff repair and matched control/pointwise evidence.
Keep KD disabled for now. Keep DS-CNN as a controlled baseline until it has a
strong same-frontend teacher and a fair same-cost no-PAI comparator. No source,
config, or training artifacts were changed by this review.
