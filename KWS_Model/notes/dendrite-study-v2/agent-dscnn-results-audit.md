# DS-CNN results audit: baselines, KD, pruning, and dendrites

**Audit date:** 2026-09-18  
**Scope:** read-only audit of DS-CNN code, configs, notes, checkpoints, logs, and
completed/partial outputs under `KWS_Model`. No training was launched and no
source/config/checkpoint/output artifact was changed.  
**Evidence labels:** **Artifact** means copied from a recorded report/metric;
**Derived** means recomputed from the current checked-out code without training;
**Inference** is an engineering interpretation and is not a measured device
result.

> **2026-09-18 input-shape correction.** A later direct metadata check of
> `models/checkpoints/ds_cnn_l_12class.pt` and the warm XS checkpoint found
> `input_shape=(40, 101)`. The `(40, 98)` log-mel figures below are useful only
> as synthetic current-code profiles and do not describe those trained
> artifacts. For deployment-relevant `(40, 101)` costs, use
> `agent-dscnn-pipeline-designs.md` and the consolidated DS-CNN journal. The
> accuracy values copied from artifacts are unaffected.

## Executive result

The repository has a complete DS-CNN-L teacher / DS-CNN-XS student pipeline on
the deployment-oriented 40-bin log-mel frontend, but it does not yet establish a
useful DS-CNN dendritic compression result. The strongest teacher is **469,604
parameters and 97.69% test accuracy**; the warm-started 4,096-parameter XS
student is **85.11% validation / 83.64% test**. The MFCC-32 DS-CNN-L retrain is
only **91.76% validation**, below the small SparkNet student it was intended to
teach, so KD comparisons using that teacher are confounded by teacher quality.

The direct DS-CNN PAI runs are historical and validation-only. Under the legacy
log-mel recipe, classifier-only PAI improved the pruned w18 row from 65.21% to
70.16% validation (2,070 vs 1,830 parameters), and w14 from 64.69% to 66.44%
(1,714 vs 1,522 parameters). These are internally completed validation results,
not held-out or device results; they are far below a credible KWS target. The
later full pipeline's w18 classifier run reached 82.68% validation at the same
2,070-parameter / 1,450,668-MAC cost, but its surrounding pipeline is marked
incomplete and contains multiple failed/restarted invocations. The discrepancy
between 70.16% and 82.68% is a recipe/provenance difference, not evidence of a
PAI gain.

The current code supports three DS-CNN dendrite placement modes: all
`DSConvBlock`s plus `fc`, `fc` only, or exact module IDs. Only `fc`-only direct
DS-CNN runs completed under the legacy compression budget. Wider feature
placements were rejected by the 1.5M-MAC admission budget. The `fc` placement is
the most plausible future placement because it adds only the classifier copy and
small residual scales, but the current artifacts do not prove a causal accuracy
benefit.

## 1. Architecture, frontends, and measured model sizes

### 1.1 Implementation

`src/kws/models/ds_cnn.py` defines `DSCNN` as:

1. a standard 2-D stem convolution + BatchNorm + ReLU;
2. a sequence of depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU
   blocks (`src/kws/models/layers.py`);
3. a fixed-size average pool, dropout, and a 12-way linear classifier.

The pool size is computed at construction from `input_shape`, so exported graphs
have fixed dimensions. This is helpful for ONNX/MCU export and for PAI hooks, but
the graph is not an MCU deployment artifact yet.

The two relevant frontends are materially different:

| Frontend | Config | Input shape used by DS-CNN | Notes |
|---|---|---:|---|
| 40-bin log-mel | `configs/data/speech_commands_v2.yaml` | `(40, 98)` | 30 ms window, 10 ms hop |
| MFCC-32 | `configs/data/speech_commands_v2_mfcc32.yaml` | `(32, 101)` | 25 ms window, 10 ms hop, log energies |
| MFCC-32 paper split | `configs/data/speech_commands_v2_mfcc32_paper.yaml` | `(32, 101)` | balanced/materialized silence; used by the later SparkNet study |

Do not compare accuracy across these frontends as an architecture effect.

### 1.2 Configured variants and current-code profiling

The model YAMLs define the following 12-class variants:

| Variant | Stem channels | Block channels | Configured params | Derived MACs, log-mel `(40,98)` | Derived MACs, MFCC `(32,101)` |
|---|---:|---|---:|---:|---:|
| DS-CNN-XXS | 18 | `[18,18]` | 1,830 | 1,393,776 | 1,160,568 |
| DS-CNN-XS | 18 | `[40,40]` | 4,096 | 3,226,640 | 2,686,752 |
| DS-CNN-S | 48 | `[64,64,64,64]` | 24,188 | 23,436,768 | 19,352,208 |
| DS-CNN-M | 64 | `[172,172,172,172,172]` | 147,940 | 149,639,664 | 123,559,968 |
| DS-CNN-L | 172 | `[276,276,276,276,276,276]` | 469,604 | 482,499,312 | 398,407,152 |

The MACs and parameter counts above are **Derived** by instantiating each current
YAML with 12 classes and calling `kws.utils.profile.count_macs`; they are not
board measurements. Stored historical reports use a different profiler/revision
for some values: for example, `reports/sparknet_phase_a.yaml` records the XS at
3,358,320 log-mel MACs. Preserve the stored report value when reproducing that
historical experiment; use the current profiler only for like-for-like new
comparisons.

The current profiler's conservative one-byte activation peaks for MFCC `(32,101)`
are approximately **29.4 kB (XXS), 65.3 kB (XS), 111.0 kB (S), 298.2 kB (M),
and 478.6 kB (L)**. These are forward-hook liveness estimates, not RP2040 arena
measurements. They show why XS/XXS are plausible starting points for an RP2040
experiment, while M/L are teacher/reference models rather than deployment
targets. The RP2040 has 264 kB SRAM, and frontend/audio buffers, stack, runtime
metadata, and other state still need to fit; activation peak alone is not an
admission test.

## 2. Baselines and teacher/student evidence

### 2.1 Early no-augmentation sanity check

`README.md` records the M1 15-epoch, no-augmentation test:

| Model / recorded historical count | Test accuracy | FAR | FRR |
|---|---:|---:|---:|
| DS-CNN-M / 146,902 params | 96.52% | 2.52% | 4.19% |
| DS-CNN-L / 467,942 params | 97.27% | 1.45% | 3.76% |

The current 12-class YAMLs count M/L as 147,940/469,604 because the classifier
has 12 outputs. These M1 values are a sanity check and use an older recorded
count/recipe; they should not be mixed with the later augmented 12-class records.

### 2.2 Strong 40-bin teacher

**Artifact:** `reports/ds_cnn_l_teacher_current.json` records:

- accuracy **0.9768583450** (97.6858%);
- **469,604 parameters**;
- FAR 0.0190 and FRR 0.0236;
- full per-class F1/confusion matrix.

The corresponding full-run metric file
`outputs/full-run-20260912T063022Z/metrics/teacher/ds_cnn_l.jsonl` has 200
epochs, best validation **0.9768563163**, final validation **0.9743490839**,
and about **181.7 s/epoch (10.09 h summed epoch time)**. This is a strong
teacher for the 40-bin log-mel student only. It cannot be paired directly with
the MFCC-32 students because `distill.py` correctly rejects frontend shape
mismatches.

### 2.3 MFCC-32 teacher retrain

**Artifact:** `outputs/phase_b/teacher_ds_cnn_l_mfcc32_nw0/metrics/summaries.yaml`
records best validation **0.9176470588** (91.76%) and final validation
**0.9101253616** (91.01%) for the same 469,604-parameter architecture. Its
JSONL averages **126.4 s/epoch**, about **7.02 h** over 200 epochs. The final
JSONL rows show training accuracy **0.99998** while validation is about 0.91,
so this run is strongly overfit and is not a credible teacher for a small
student without a recipe repair. The mismatch with the strong 40-bin teacher
is a frontend/split/recipe confound, not an architectural conclusion.

### 2.4 XS student

**Artifact:** `reports/phase_a/xs_val.json` and `xs_test.json` evaluate the
4,096-parameter independently trained/warm-started XS on the 40-bin log-mel
split:

- validation **85.1109%**, FAR 17.48%, FRR 13.85%;
- test **83.6431%**, FAR 18.53%, FRR 15.29%.

`reports/sparknet_phase_a.yaml` confirms the test report and records 3,358,320
historical MACs for this XS artifact. The full pipeline metric file records the
warm-distilled student at best validation **85.0723%** over 200 epochs; summed
epoch time is about **4.29 h** (77.3 s/epoch). The project is therefore starting
from a weak, but deployment-sized, student rather than an already competitive
XS classifier.

## 3. KD evidence and confounds

The KD implementation is shared in `src/kws/optimize/kd.py` and supports feature,
response, and classification losses. The relevant recipes are:

- `configs/train/distill_imc.yaml`: T=1, feature/response/classification
  weights 0.3/0.1/0.6;
- `configs/train/light_kd.yaml`: T=2, response/classification 0.5/0.5;
- `configs/train/light_kd_annealed.yaml`: linearly ramps response KD from 0 to
  0.5 over epochs 0--40.

### 3.1 Recorded comparisons

| Run | Frontend / student | Teacher | Best val | Final val | Interpretation |
|---|---|---|---:|---:|---|
| `full-run-20260912T063022Z` | log-mel XS, 4,096 params | 40-bin DS-CNN-L, 97.69% | **85.072%** | 84.571% | Warm-distilled student; not matched to a no-KD run in this artifact |
| `phase_b/sparknet_c12_mfcc32_g32_kd_annealed` | MFCC C12 SparkNet | MFCC DS-CNN-L, 91.76% | **91.244%** | 91.109% | −1.37 pp vs matched 92.613% no-KD control, but teacher is weaker than student |
| `phase_b/step2_sparknet_c12_light_kd_t2...` | log-mel C12 SparkNet | strong 40-bin teacher | **92.150%** | 91.919% | Exact one-seed tie with no-KD control at 92.150%; init not matched |

The MFCC comparison is causally clean only for “this weak MFCC teacher hurts.”
It does **not** establish that KD itself hurts because the teacher is worse than
the student. The strong-teacher comparison has functioning diagnostics
(teacher accuracy about 98.6%, confidence about 0.896, KD logit-gradient ratio
about 1.15) but is one seed with different initialization; it supports “no
detectable gain under this recipe,” not “KD cannot help.”

The repository's analysis correctly recommends a same-frontend, stronger teacher
before spending a KD budget. For an RP2040-oriented student, an MFCC-32 teacher
must clear a high validation bar first; otherwise KD should remain disabled.

## 4. Structured pruning and direct DS-CNN PAI placements

### 4.1 Pruning implementation

`src/kws/optimize/prune.py::prune_ds_cnn` structurally narrows only the
pointwise block output channels. It threads surviving channels through later
depthwise filters, BatchNorm statistics, pointwise inputs, and finally the
classifier inputs. The stem width and classifier output count remain fixed.
The channel ranking is L1 magnitude after accounting for already-removed input
channels. N:M pruning is a separate mask-only mechanism; it does not alter dense
tensor shapes and therefore does not imply MCU MAC savings.

For the XS source `[40,40]`, a 0.45 keep ratio rounds to `[18,18]`, the XXS
backbone with 1,830 params. The direct pipeline also explored `[14,14]`,
`[10,10]`, and `[6,6]` widths. Pruning is one-shot channel surgery followed by
fine-tuning; it is not a PAI dendrite effect.

### 4.2 Placement code and projected cost

`src/kws/optimize/dendritic_config.py` resolves:

- `fc_only` -> `fc`;
- `blocks_and_linear` -> every `DSConvBlock` plus `fc`;
- `module_ids` -> exact paths such as `.blocks.1` and `.fc`.

`configs/train/compression_experiment.yaml` used a 4,096-param / 1.5M-MAC
budget, `max_dendrites: 1`, and attempted classifier-only, late-block+classifier,
and all-block+classifier placements. The actual clean graph is the final cost
authority; projected costs are only admission estimates.

### 4.3 Legacy `outputs/compression-run`

This is a separate interrupted `kws.pipeline --extreme-prune` on the default
40-bin log-mel frontend. Its manifest is `status: interrupted` after several
restarts, and `reports/compression_experiment.yaml` remains `status: running`.
The individual completed candidate artifacts are still useful as validation
evidence:

| Candidate | Best val | Deployed params | MACs | Host CPU latency |
|---|---:|---:|---:|---:|
| conventional w18 `[18,18]` | 65.21% | 1,830 | 1,450,656 | 0.383 ms p50 (stored report) |
| dendritic w18, `fc` | **70.16%** | 2,070 | 1,450,668 | 0.282 ms p50 |
| conventional w14 `[14,14]` | 64.69% | 1,522 | 1,209,888 | 0.318 ms p50 |
| dendritic w14, `fc` | **66.44%** | 1,714 | 1,209,900 | 0.255 ms p50 |

The w18 `best_arch_scores.csv` compares 1,830 -> 2,058 PAI parameters and
0.68698 -> 0.70164 validation; the clean deployed report is 2,070 because it
includes the residual scale parameters. w14 similarly compares 1,522 -> 1,702
in PAI's architecture CSV and deploys at 1,714. The PAI JSONL contains accepted
restructures at w18 epochs 98/144/249 and w14 epochs 57/81/140. The w18 and
w14 canonical PBScore files reach roughly 0.147 and 0.136 respectively, which
is a placement-alignment signal, not a causal accuracy comparison.

The wider `late_feature_and_classifier` and `all_feature_blocks_and_classifier`
placements were skipped by the 1.5M-MAC projected budget at w18/w14. w10 and w6
classifier candidates were planned but not completed. The completed direct run
therefore says only that an `fc` dendrite can recover some accuracy in this
particular weak pruned recipe while adding parameters; it does not show that
the model beats an independently trained same-budget DS-CNN.

Training-time evidence from the JSONL is substantial: w18 used 40 prune-KD
epochs (~0.52 h), 249 PAI epochs (~2.91 h), and 8 resume-KD epochs (~0.10 h);
w14 used ~0.47 h + ~1.62 h + ~0.10 h. Host latency is not RP2040 latency.

### 4.4 Later `full-run-20260912T063022Z`

This run used the 40-bin teacher, warm-distilled XS, structured pruning, KD,
classifier-only PAI, and resume KD. Its meaningful w18 row is:

- pruned `w18` (1,830 params): prune-KD best validation **79.38%**;
- PAI `best_arch_scores.csv`: 1,830 params **82.584%**, 2,058 PAI params
  **82.681%**;
- clean deployed cost: **2,070 params, 1,450,668 MACs, 2,070 weight bytes**;
- resume KD best **82.53%**, so resume did not improve the pre-resume 82.68%
  baseline;
- stored host CPU p50 **0.349 ms**.

The canonical w18 PAI run has switch epochs 79 and 111, one accepted dendrite,
and a `Best PBScores.csv` peak around 0.1333. A `noImprove_lr` retry also exists;
its two-row architecture CSV ends at 82.584% for 1,830 and 82.276% for 2,058.
Per the completed-results checklist, this confirms that the PAI lifecycle ran
and that a candidate was added, but the internal architecture-row delta is not
a causal control. There is no independent same-recipe no-dendrite w18 test
comparison in this run.

The w17 prune-KD phase reached **78.94%** best validation, but its PAI invocation
was interrupted at epoch 3. Its PAI CSV has only the 1,750-parameter row at
79.286%; it is not a completed dendritic result. The logs also show macOS
DataLoader workers being disabled and multiple resume/restart attempts.

The full-run manifest contains multiple interrupted/failed invocations,
including missing native PAI checkpoint, RNG-state errors, an `IndexError`,
recipe-fingerprint mismatch, and finally `PAI cleanup module 'fc' has no
two-branch layer_array`. The downstream cluster/quantize/benchmark stages are
therefore not valid completed end-to-end evidence.

## 5. PerforatedAI completed-results checklist

Applied to the direct DS-CNN artifacts where the required files exist:

| Checklist item | Finding |
|---|---|
| Locate canonical result files | Present under `compression-run/pai/candidates/{w18,w14}_classifier` and `full-run/.../candidate_w18`; snapshots and `noImprove_lr` retries are separate and must not be globbed as canonical |
| Validation progression | Present in `*Scores.csv`; direct w18/w14 and full-run w18 show progression through n/p/n lifecycle |
| Switch epochs | Present: direct w18 72/118, w14 31/55; full-run w18 79/111 |
| Dendrite count | Direct w18/w14 each end with one accepted classifier dendrite; full-run w18 likewise; w17 did not complete |
| `noImprove_lr` | Present in full-run w18; indicates a retry that did not produce a better candidate, not “no dendrite ever” for the canonical run |
| Architecture comparison | Present, but within-search rows only; not a no-dendrite causal control |
| Train-vs-validation overfit check | PAI CSVs have train columns, but older direct runs use a different loss scale and no dedicated train-score artifact; do not infer generalization from row gaps alone |
| PBScore module placement | Classifier-only files show ~0.136--0.147 peaks in direct runs and ~0.133 in full-run; useful alignment evidence, not proof of accuracy benefit |
| LR schedule / stability | Present in `*learning_rate.csv`; direct PAI phases are multi-hour and resume phases did not improve full-run w18 |
| Clean deployment cost | Present for direct candidates and full-run w18; all are host/profile estimates, not board measurements |
| Held-out test | **Absent for all direct DS-CNN dendritic candidates**; do not call any of them test accuracy |

## 6. Causal vs confounded conclusions

### Evidence that is reasonably causal within its stated scope

- `prune_ds_cnn` performs real shape-changing structured channel surgery, and
  the derived parameter/MAC reductions follow directly from the resulting model
  graph.
- In the legacy recipe, the completed `fc` dendritic candidate has higher
  validation accuracy than its own paired pruned starting row (w18 +4.02 pp,
  w14 +1.31 pp). This is a within-run before/after observation, not proof of
  superiority over a separately trained conventional model.
- PAI did wrap the requested classifier, switched through n/p/n modes, wrote
  canonical score/switch/PBScore artifacts, and produced clean graph files for
  the completed direct candidates.
- Wider feature placements consume much more MAC budget. This is a mechanical
  graph-cost conclusion and remains true independent of accuracy.

### Comparisons that are confounded or unsupported

- MFCC DS-CNN-L KD vs no-KD: the teacher is weaker than the student and uses a
  different frontend/split context than the strong log-mel teacher.
- Full-run w18 PAI vs the warm XS: pruning, KD, changed optimizer/schedule,
  frontend/recipe history, and PAI all vary; no matched no-PAI w18 control is
  present.
- Direct compression-run w18/w14 dendrite gains: the baseline is a low-scoring
  pruned/KD recipe and the report is validation-only; there is no independent
  same-cost conventional control trained under the same complete lifecycle.
- PBScore magnitude or a positive `best_arch_scores.csv` delta is not a causal
  gain. It selects the best row within the same PAI search and is vulnerable to
  correlated validation selection.
- Any host CPU latency, projected int8 bytes, or activation estimate is not an
  RP2040 latency/SRAM/power result. No TFLite Micro export, Pico firmware, arena
  measurement, or board benchmark exists in this tree.

## 7. RP2040 plausibility and recommended DS-CNN experiments

### Plausible width/placement ranking (inference from local evidence)

1. **XXS, `fc` only:** 1,830 params; ~1.16M current-code MFCC MACs; smallest
   direct base. Accuracy is not recorded as a standalone DS-CNN XXS result, so
   treat it as a search point, not a deployment recommendation.
2. **XS, `fc` only:** 4,096 params; ~2.69M MFCC MACs; the most credible DS-CNN
   starting point for a repaired teacher/KD or PAI experiment. The stored XS
   accuracy is only 83.64% test, so it needs a quality improvement before PAI
   conclusions are meaningful.
3. **S, `fc` only:** 24,188 params; ~19.35M MFCC MACs; likely feasible in flash
   but its ~111 kB activation estimate needs a real TFLM arena measurement. It
   is a useful middle reference, not yet trained/evaluated in this output tree.
4. **M/L:** teacher/reference only. Their activation estimates exceed or consume
   most of RP2040 SRAM before audio/frontend/runtime state; they are not plausible
   direct targets without a measured memory plan.

`fc` is preferred over block placements for a first PAI trial because the copied
   classifier has a small, predictable cost. For XS, one classifier dendrite
   adds the 228 classifier weights plus 12 output-scale parameters (about 240
   deployed parameters by the current cost model), while copying a feature block
   adds both parameters and substantial MACs. This is a cost argument; PBScore
   evidence is positive for `fc` but has not yielded a causal DS-CNN test result.

### What to run next (not run by this audit)

1. Establish a no-PAI DS-CNN XS/XXS/S baseline with the exact intended MFCC
   frontend, multiple seeds, held-out test, and current profile costs.
2. Repair or replace the MFCC teacher first; reject KD if it does not clearly
   outperform the student.
3. Compare structured-pruned `fc`-only PAI against a same-checkpoint, same
   optimizer/schedule, no-PAI control at equal **total deployed** params/MACs.
4. Keep one dendrite initially (`max_dendrites: 1`), preserve canonical
   `Scores`, `switch_epochs`, `Best PBScores`, `best_arch_scores`, LR, and timing
   files, and evaluate only the selected configuration on held-out test.
5. Require exact clean-graph parity, int8/TFLM operator compatibility, and a
   board-side arena/latency measurement before making an RP2040 claim.

## Primary evidence paths

- `src/kws/models/ds_cnn.py`, `src/kws/models/layers.py`
- `configs/model/ds_cnn_{xxs,xs,s,m,l}.yaml`
- `configs/data/speech_commands_v2*.yaml`
- `src/kws/optimize/prune.py`, `src/kws/optimize/dendritic_config.py`
- `configs/train/{pipeline,distill_imc,dendritic_cycle1,dendritic_prune_loop,compression_experiment}.yaml`
- `reports/ds_cnn_l_teacher_current.json`, `reports/phase_a/xs_{val,test}.json`
- `outputs/full-run-20260912T063022Z/`
- `outputs/compression-run/`
- `outputs/phase_b/teacher_ds_cnn_l_mfcc32_nw0/`
- `notes/dendrite-study-v2/ENHANCEMENTS.md`, `agent-repo-results-audit.md`,
  `PICO_COMPRESSION_PIPELINE_JOURNAL.md`
