# Where the experiment data lives

**Written:** 2026-09-18. **Base:** `/Users/vishy/Desktop/ChipsAIHackathon/KWS_Model` (BASE below).
All paths are relative to BASE unless stated otherwise. Every count in this file
was verified against disk on 2026-09-18.

An index of every directory holding experiment information or generated run
output, what each one contains, and which parts are current versus stale.

---

## 1. The latest study — `outputs/sparknet-dendritic-study-v2/`

277 MB, last written **2026-09-18 10:02**. The completed 150-run dendritic
placement matrix plus its 30 scratch baselines. This is the study that matters.

### 1.1 Top-level layout

```
outputs/sparknet-dendritic-study-v2/
├── study.log                      12 MB, 156,076 lines — see 1.5
├── calib_arm.log                  arm-side throughput calibration
├── calib_scratch.log              scratch-side throughput calibration
├── scratch/                       30 baseline cells (6 widths x 5 seeds)
│   ├── c{12,10,8,6,4,2}-seed{0..4}/
│   └── c{12,10,8,6,4,2}-seed{0..4}.log    per-cell stdout
├── arms/                          150 cells = 5 arms x 6 widths x 5 seeds
│   ├── pointwise/   c{12,10,8,6,4,2}-seed{0..4}/
│   ├── fc/          (same 30)
│   ├── gate_conv/   (same 30)
│   ├── depthwise/   (same 30)
│   └── control/     (same 30)
└── selection/
    ├── selected_arms.json         the 60 frozen test targets
    └── test_report.json           held-out test results
```

Verified counts: **150** arm cells, **30** scratch cells, **150**
`sparknet_dendritic_prune_experiment.yaml` reports, **150**
`study_experiment.yaml` configs.

### 1.2 Inside every arm cell

Example: `arms/pointwise/c12-seed0/`. `{W}` is the width (12, 10, 8, 6, 4, 2).

| Path | Contents |
|---|---|
| `reports/sparknet_dendritic_prune_experiment.yaml` | **The main result.** Validation accuracies, deployed params, MACs, the full resume epoch history, and the comparison deltas. One per cell, 150 total. |
| `study_experiment.yaml` | The exact config the driver serialized for this cell. The driver raises `ValueError` if a re-run would change it. |
| `manifest.yaml` | **Run provenance, and the only per-cell record of the launch command.** See 1.4. |
| `pai/candidates/sparknet_c{W}_multilayer/` | **All PerforatedAI artifacts** — see 1.3. |
| `metrics/sparsity/sparknet_c{W}/prune_supervised.jsonl` | The 40-epoch identity-prune fine-tune, one JSON record per epoch. |
| `metrics/sparsity/sparknet_c{W}_multilayer/pai.jsonl` | The PAI lifecycle, one record per epoch. |
| `metrics/sparsity/sparknet_c{W}_multilayer/resume_supervised.jsonl` | The 8-epoch resume phase. Dendritic arms only. |
| `metrics/sparsity/sparknet_c{W}_multilayer/standard_control.jsonl` | The 98-epoch plain fine-tune. **Control arms only**, 30 files. |
| `metrics/summaries.yaml` | Per-stage rollup. |
| `models/checkpoints/sparsity/sparknet_c{W}/prune_supervised/` | `best.pt`, `latest.pt` for the prune fine-tune. |
| `models/checkpoints/sparsity/sparknet_c{W}_multilayer/` | `resume_supervised/` (dendritic) or `standard_control/` (control). |
| `logs/*.log` | Per-invocation run log. Control cells have **two** — the 2026-09-17 crash and the 2026-09-18 rerun. |
| `metadata/configs/` | Present but empty in these runs. |
| `.run.lock` | Concurrency guard. |
| `models/checkpoints/{teacher,student,cluster,quantize}/`, `models/exported/` | Created by the pipeline scaffold, unused here. |

### 1.3 PAI artifacts per cell

Under `pai/candidates/sparknet_c{W}_multilayer/`, with save-name prefix
`sparknet_c{W}_multilayer`. **Each artifact exists in up to three variants** —
this matters when globbing:

| Variant | Example | Meaning |
|---|---|---|
| canonical | `sparknet_c12_multilayer_best_arch_scores.csv` | Final state. **This is the one to read.** |
| `before_final_` | `sparknet_c12_multilayerbefore_final_best_arch_scores.csv` | Snapshot before the final cleanup. Note: no underscore before `before`. |
| `_beforeSwitch_N` | `sparknet_c12_multilayer_beforeSwitch_0_best_arch_scores.csv` | Snapshot at each mode switch. |

| Canonical file | Contents |
|---|---|
| `...Best PBScores.csv` | Per-module correlation between dendrite activations and network gradients. Rubric: > 0.02 good placement, < 0.01 wasting parameters. **30 per dendritic arm, 120 total. Control cells have none.** |
| `..._best_arch_scores.csv` | Best score at each dendrite count. With `max_dendrites: 1` every one has exactly **2 rows**. 30 per dendritic arm. |
| `...Scores.csv` | Validation score per epoch across the whole lifecycle. |
| `...switch_epochs.csv` | Epochs at which PAI switched neuron/candidate mode. |
| `...learning_rate.csv` | LR schedule. **This is where the restart at the first neuron phase is visible.** |
| `...param_counts.csv` | Parameter count trajectory. |
| `...Times.csv` | Wall-clock per phase. |
| `....png` | PAI's auto-generated summary plot. |
| `beforeSwitch_N.pt`, `beforeSwitch_N_pai.pt` | Weights at each switch. |
| `best_model.pt`, `best_model_pai.pt`, `best_model_beforeSwitch_N.pt` | Best architecture found. |
| `final_clean_pai.pt` | The deployed dendritic checkpoint. |
| `latest.pt`, `latest_pai.pt`, `kws_native_latest.pt` | Latest state, PAI-wrapped and native. |
| `cycle_checkpoints/`, `cycle_metadata.yaml` | Per-cycle bookkeeping. |

**`*noImprove_lr*` markers** (a failed dendrite-addition retry): **96 files
across 6 distinct cells**, not spread evenly — `pointwise/c8-seed0`,
`fc/c8-seed0`, `fc/c2-seed1`, `fc/c4-seed2`, `gate_conv/c8-seed0`,
`depthwise/c8-seed0`. Four of the six are the C8 seed0 cell in four different
arms.

**Control cells** bypass PAI, so this directory holds only `Scores.csv`,
`Times.csv`, the `.png`, one `beforeSwitch_0` pair, and the checkpoints. The one
exception is `arms/control/c12-seed0/`, which still carries a stray
`_best_arch_scores.csv` left behind by the crashed 2026-09-17 PAI attempt.

### 1.4 `manifest.yaml` — provenance

Worth calling out separately because it is the only place the launch command is
recorded per cell:

```yaml
format_version: 1
run_id: a3aa34ac092d4256aeeebd96f4281553
status: completed
command: kws.optimize.sparknet_dendritic_prune_experiment
argv:
  - .../sparknet_dendritic_prune_experiment.py
  - --config
  - .../arms/pointwise/c12-seed0/study_experiment.yaml
  - --no-KD                 # <- the per-cell KD evidence lives here
started_at: '2026-09-17T07:33:42+00:00'
seed: 0
platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
python: 3.13.11
packages:
  torch: 2.14.0
  numpy: 2.5.3
  pyyaml: 6.0.3
  perforatedai: 3.2.8     # <- the PAI version the whole study ran against
  kws: 0.1.0
git:
  commit: 9c1b1798d04ff990924debeed2f6e9fffab099b4
  dirty: true
invocations:
  - id: c18b1bc81930
```

### 1.5 `study.log` — what is actually in it

12 MB, 156,076 lines, four different things interleaved:

1. **Scratch-sweep driver lines**, anchored at column 0: `run` (33), `skip`
   (33), `FAIL` (1, C10 seed4), `=== RESTART` (3 — widths expanded 12/10/8/6/4,
   then a resilient baseline stage, then 12/10/8/6/4/2).
2. **Raw PAI stdout**, the bulk of the file: `Score`, `Returning`, `Checking`,
   `Adding`, `With`, `Nodes` lines, plus timestamped training records.
3. **The held-out test evaluation**, a `[N/60]` block near the end listing each
   frozen target's test accuracy and params.
4. **The final aggregate table** at roughly line **156,059**, giving per-arm,
   per-width TEST and VALIDATION accuracy with sample sd, and the paper
   reference row (SparkNet C16 TEST, 95.70 +/- 0.17).

The arm stage is **not** logged here cell-by-cell; per-cell arm provenance is in
each cell's `manifest.yaml` and `logs/`.

### 1.6 Inside every scratch cell

Example: `scratch/c12-seed0/`.

| Path | Contents |
|---|---|
| `models/checkpoints/paper_replication/best.pt` | The scratch checkpoint every arm starts from. |
| `models/checkpoints/paper_replication/latest.pt` | Final epoch. |
| `metrics/paper_replication/sparknet_c{W}_paper.jsonl` | Per-epoch training record. |
| `metrics/summaries.yaml`, `manifest.yaml`, `logs/` | Same roles as the arm cells. |
| `reports/` | Present, empty. |

---

## 2. Earlier and one-off dendrite runs

Newest first, by last write.

| Directory | Size | Last write | Shape |
|---|---|---|---|
| `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/` | 92 MB | 09-17 01:04 | `seed0..4/` — fc-only placement, `max_dendrites: 3`. **The closest existing precedent to the `.fc` + d3 re-run worth doing.** |
| `outputs/sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/` | 18 MB | 09-16 23:55 | `seed0..4/` — gate_conv placement, `max_dendrites: 3`. |
| `outputs/sparknet-c12-dendritic-prune-no-kd-unlimited/` | 30 MB | 09-16 09:53 | Single run, no dendrite cap. |
| `outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3/` | 6.9 MB | 09-16 09:53 | Single run. |
| `outputs/compression-run/` | 5.5 MB | 09-16 09:53 | Older single pipeline run. |
| `outputs/full-run-20260912T063022Z/` | 15 MB | 09-16 09:53 | Older full pipeline run; also has `graphs/`. |
| `outputs/phase_b/` | 23 MB | 09-16 09:53 | Pre-dendrite KD and teacher comparisons. |
| `outputs/sparknet-paper-replication/` | 2.3 MB | 09-16 09:53 | The C16 reference, 95.70 +/- 0.17. |
| `outputs/diagnostics/kd_20260914/` | 1.0 MB | 09-13 20:47 | KD diagnostics. |
| `outputs/_eval_scratch/` | 0 B | — | Empty. |

Single-run directories (`compression-run`, `full-run-*`, the two c12 runs) use
the same internal layout as one study cell: `logs/ manifest.yaml metadata/
metrics/ models/ pai/ reports/`.

### 2.1 `outputs/phase_b/` contents

```
comparison_launcher_20260914.log
r1_c12_light/
sparknet_c12_mfcc32_g32_kd_annealed/
sparknet_c12_speech_commands_v2_g32/
sparknet_c12_speech_commands_v2_g40/
sparknet_c12_speech_commands_v2_mfcc32_g32/
sparknet_c12_speech_commands_v2_mfcc32_g40/
step2_sparknet_c12_light_20260913T215424Z/
step2_sparknet_c12_light_kd_t2_20260914T013331Z/
teacher_ds_cnn_l_mfcc32/
teacher_ds_cnn_l_mfcc32_nw0/
```

### 2.2 `outputs/sparknet-paper-replication/` contents

```
c16-seed{0,1,2,3,4}/
c16-canonical-seed1948/
c16-canonical-exact-counts-seed1948/
released-c16-eval/
released-c16-exact-count-eval/
```

---

## 3. Written analysis and provenance

Not generated run output — these are hand-maintained.

| Path | Contents |
|---|---|
| `DENDRITE_JOURNAL.md` | Running session log. State at session start, the arm to train-config map, live progress refreshes, the control blocker, and a "what the runs show" section. |
| `SPARKNET_DENDRITE_FIXES.md` | Audit trail: scope checked, initial evidence, confirmed problems and fixes, paper comparison, verification, type-check follow-up. |
| `notes/dendrite-study-v2/RESULTS_ANALYSIS.md` | Detailed results snapshot taken 2026-09-17T18:18:03Z. |
| `notes/dendrite-study-v2/ENHANCEMENTS.md` | Performance enhancements and the KD verdict. |
| `notes/dendrite-study-v2/INTEGRATION_AUDIT.md` | SparkNet / PerforatedAI integration audit. |
| `notes/dendrite-study-v2/PAI_KNOWLEDGE.md` | Mechanism digest taken from the package source rather than marketing. |
| `notes/dendrite-study-v2/PAI_PAPERS.md` | Paper reading record. |
| `notes/dendrite-study-v2/aggregate_study.py` / `.json` | Aggregation script and its output. |
| `notes/dendrite-study-v2/placements.json`, `verify_placements.py` | Placement definitions and their verification. |
| `notes/dendrite-study-v2/probe_pai_loop.py` | PAI loop probe. |
| `notes/dendrite-study-v2/DATA_INVENTORY.md` | This file. |
| `reports/phase_a/`, `reports/pipeline/`, `reports/sparknet_phase_a.yaml`, `reports/ds_cnn_l_teacher_current.json` | Older phase-A and pipeline reports. |

There is **no** `ERRORS.md` in this repo; the crash log referred to by that name
elsewhere is a section of `DENDRITE_JOURNAL.md`.

### 3.1 Staleness warning

- `notes/dendrite-study-v2/RESULTS_ANALYSIS.md` is a **mid-study snapshot from
  before the control arm finished**. Its conclusions predate the control result
  that reframed the whole study.
- `DENDRITE_JOURNAL.md` section 3 was last refreshed **2026-09-18 02:13 UTC**,
  also before the control landed.

Neither reflects the final decomposition, in which the PAI lifecycle costs
-0.318 pp against a budget-matched plain fine-tune (30/30 cells) while the
dendrite itself adds +0.057 pp on top of that handicapped base.

---

## 4. Configuration that defines the runs

| Path | Role |
|---|---|
| `scripts/run_sparknet_dendritic_study.py` | **The driver.** Defines `WIDTHS`, `SEEDS`, `ARM_ORDER`, `REFERENCE_ARMS`, and the arm to train-config map. Appends `--no-KD` to every launch. |
| `scripts/select_sparknet_arms.py` | Picks which arms reach the held-out test. |
| `scripts/run_sparknet_scratch_sweep.sh` | The scratch-baseline sweep that produced most of `study.log`. |
| `configs/train/sparknet_c16_dendritic_prune_no_kd*.yaml` | Per-arm train configs: `_backbone_pointwise`, plain (fc), `_gate_conv`, `_backbone_depthwise`, `_control`. |
| `configs/train/sparknet_narrow_paper_fast_io.yaml` | The scratch training config. |
| `configs/experiment/sparknet_c16_dendritic_prune_no_kd_*_seed{0..4}.yaml` | 25 per-arm, per-seed experiment configs, plus `sparknet_c12_dendritic_prune_no_kd.yaml`. |
| `src/kws/optimize/dendritic.py` | `run_cycle`, `run_standard_control`, `standard_control_epoch_budget`. |
| `src/kws/optimize/sparknet_dendritic_prune_experiment.py` | Dispatches empty `module_ids` to the standard control instead of PAI. |
| `src/kws/optimize/dendritic_config.py` | `normalize_module_ids`, `placement_module_names`, `pai_module_ids`. |

---

## 5. Quick recipes

All verified to return the stated counts.

```bash
BASE=/Users/vishy/Desktop/ChipsAIHackathon/KWS_Model
STUDY=$BASE/outputs/sparknet-dendritic-study-v2

# every report                                    -> 150
find "$STUDY" -name sparknet_dendritic_prune_experiment.yaml

# one arm's reports                               -> 30
find "$STUDY/arms/fc" -name sparknet_dendritic_prune_experiment.yaml

# canonical PBScores, excluding both snapshot variants   -> 120
find "$STUDY/arms" -name '*multilayerBest PBScores.csv'

# canonical dendrite-count curves                 -> 121 (120 + control stray)
find "$STUDY/arms" -name '*multilayer_best_arch_scores.csv'

# failed dendrite-addition retries                -> 96 files, 6 cells
find "$STUDY/arms" -name '*noImprove_lr*'
find "$STUDY/arms" -name '*noImprove_lr*' | sed "s|$STUDY/arms/||" | cut -d/ -f1,2 | sort -u

# control-only plain fine-tune histories          -> 30
find "$STUDY/arms/control" -name standard_control.jsonl

# scratch-sweep driver decisions (anchored; the log is full of PAI stdout)
grep -nE '^(run|skip|FAIL|=== RESTART)' "$STUDY/study.log"

# the final aggregate results table
tail -25 "$STUDY/study.log"

# per-cell launch command and PAI version
grep -A3 '^argv:' "$STUDY/arms/pointwise/c12-seed0/manifest.yaml"
grep -rh 'perforatedai:' "$STUDY/arms"/*/*/manifest.yaml | sort -u
```
