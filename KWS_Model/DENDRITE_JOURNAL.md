# SparkNet x PerforatedAI Dendrite Journal

**Base directory:** `KWS_Model/` (repo root is `/Users/vishy/Desktop/ChipsAIHackathon`)
**Branch:** `KWS_Model`
**Journal started:** 2026-09-17 (session: coordinator + subagents)
**Purpose:** a running, hand-off-ready record of (a) what the latest five-seed
SparkNet dendrite runs actually show, (b) what PerforatedAI's own papers/docs
say about adding dendrites correctly, (c) what in our integration is wrong or
suboptimal, and (d) concrete performance enhancements (KD removal, etc.).

A future agent should be able to read *this file alone* and continue.

---

## 0. How this work is organised

| Path | What it holds |
| --- | --- |
| `KWS_Model/DENDRITE_JOURNAL.md` | This journal (authoritative running record) |
| `KWS_Model/notes/dendrite-study-v2/` | Per-workstream deliverables written by subagents |
| `KWS_Model/SPARKNET_DENDRITE_FIXES.md` | Prior session's audit of the `.fc` d3 five-seed run |
| `KWS_Model/outputs/sparknet-dendritic-study-v2/` | The **currently running** scratch-start study |
| `KWS_Model/outputs/sparknet-c16-dendritic-prune-no-kd-*/` | Historical (pruned-from-C16) runs |
| `KWS_Model/PAI Skills/` | Vendored PerforatedAI skill docs (gitignored) |

Workstream deliverables (written by fresh subagents, synthesised here):

- `notes/dendrite-study-v2/RESULTS_ANALYSIS.md` — what the runs show
- `notes/dendrite-study-v2/PAI_KNOWLEDGE.md` — papers + docs digest
- `notes/dendrite-study-v2/INTEGRATION_AUDIT.md` — our code vs PAI best practice
- `notes/dendrite-study-v2/ENHANCEMENTS.md` — performance levers incl. KD removal

---

## 1. State of the world at session start (2026-09-17 ~14:00 ET)

### 1.1 Git
- HEAD = `83ed601` "added sparknet c2, c4, and c6 and fixed some tests" (2026-09-17 13:48 ET).
- Working tree clean. `KWS_Model/outputs/` and `PAI Skills/` are gitignored, so
  run artifacts are **not** in version control.

### 1.2 A study is RUNNING right now — do not kill it casually
```
uv run python scripts/run_sparknet_dendritic_study.py --study-root outputs/sparknet-dendritic-study-v2
```
- PID ~12113/12115, started 2026-09-17 05:45 ET.
- Matrix: arms `(pointwise, fc, gate_conv, depthwise, control)` x widths
  `(12, 10, 8, 6, 4, 2)` x seeds `(0..4)` = **150 arm runs**, on top of
  **30 from-scratch baselines** (6 widths x 5 seeds).
- Design (from the script docstring): every arm starts from *its own seed's
  from-scratch* SparkNet checkpoint, not from a pruned C16. `control` is the
  empty-placement (zero-dendrite) budget-matched arm and rides to test at every
  width. Pruning method is `identity` (no pruning) — width is set by the model
  config, not by pruning.
- **Progress at session start:** 30/30 scratch baselines complete; pointwise at
  28/30 (running `c2-seed2`). Arms `fc`, `gate_conv`, `depthwise`, `control`
  have not started. So roughly 122/150 arm runs remain.
- Live log: `outputs/sparknet-dendritic-study-v2/study.log` (~4 MB and growing).

### 1.3 What "the latest five-seed c16→c2 runs" refers to
Two different things exist and must not be mixed:

1. **Historical**: `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/` —
   five directories, `.fc` placement, `max_dendrites: 3`, no KD, pruned down
   from a paper-replication **C16** checkpoint to C12/C10/C8. Audited in
   `SPARKNET_DENDRITE_FIXES.md`. Key caveat already established there: the five
   directories were **not** five independently seeded downstream runs (all PAI
   sidecars recorded `seed: 0`); only the source checkpoints differed.
2. **Current**: `outputs/sparknet-dendritic-study-v2/` — the scratch-start,
   genuinely five-seed, five-arm study described in 1.2. This one *is* properly
   seeded per run, and it spans C12 down to **C2**.

---

## 2. Session log

### 2026-09-17 — coordinator session
- Surveyed repo, confirmed no prior journal existed; created this one.
- Confirmed study-v2 is mid-flight (see 1.2) and left it running.
- Dispatched four fresh subagents (results, papers/docs, integration audit,
  enhancements). Findings land in `notes/dendrite-study-v2/` and are
  synthesised into sections 3-6 below.

- Dispatched four fresh subagents (coordinator does not duplicate their work):
  1. **Results analyst** -> `notes/dendrite-study-v2/RESULTS_ANALYSIS.md`
     (inventory/completeness, scratch baselines, pointwise arm, paired per-seed
     deltas, parameter/MAC accounting, dendrite-vs-wider-network comparison,
     PAI-internals health check, historical runs, findings, red flags).
  2. **PAI researcher** -> `notes/dendrite-study-v2/PAI_KNOWLEDGE.md`
     (perforatedai.com + GitHub docs + the actual research papers + the
     installed `perforatedai` package source: mechanism, canonical recipe,
     hyperparameters, placement guidance, pitfalls, small-model applicability).
  3. **Integration auditor** -> `notes/dendrite-study-v2/INTEGRATION_AUDIT.md`
     (our lifecycle vs PAI requirements, conformance table, confirmed bugs,
     small-model hazards, empirical per-arm module-wrapping table, fix list).
  4. **Enhancement investigator** -> `notes/dendrite-study-v2/ENHANCEMENTS.md`
     (KD keep/drop verdict + KD recipe defects, ranked performance levers,
     which levers must stay frozen across arms, next-experiment proposal).

#### Arm -> train-config map (from `scripts/run_sparknet_dendritic_study.py`)

| Arm | Train config |
| --- | --- |
| pointwise | `configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_pointwise.yaml` |
| fc | `configs/train/sparknet_c16_dendritic_prune_no_kd.yaml` |
| gate_conv | `configs/train/sparknet_c16_dendritic_prune_no_kd_gate_conv.yaml` |
| depthwise | `configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_depthwise.yaml` |
| control | `configs/train/sparknet_c16_dendritic_prune_no_kd_control.yaml` |
| (scratch baselines) | `configs/train/sparknet_narrow_paper_fast_io.yaml` |

Arm order is deliberate: pointwise first because it targets cross-channel
mixing (what narrowing actually removes) and is the cheapest, so the most
informative result lands first if the matrix has to be cut short.

### 2026-09-17 22:25 UTC — continuation / handoff refresh
- **Decision: keep the existing study running; do not kill or restart it.**
  The active process is `arms/fc/c4-seed2`. The pointwise and partial FC
  results are negative, but the scheduled no-dendrite control and three arms
  remain unmeasured. Restarting now would discard the only clean way to
  separate a PAI effect from the identity-fine-tune/optimizer handoff effect.
  This satisfies the operational requirement that something remains training.
- Refreshed the read-only partial aggregate at
  `notes/dendrite-study-v2/aggregate_study.json`
  (snapshot `2026-09-17T22:25:04+00:00`): scratch **30/30** complete; arms
  **52 complete, 1 running, 97 not started**. Pointwise is **30/30 complete**;
  FC is complete through C6 and has C4 seeds 0--1 complete.
- Added `notes/dendrite-study-v2/PAI_PAPERS.md`, a full reading record for the
  three direct PB/PerforatedAI papers located, distinct from background
  dendrite literature.
- Added `notes/dendrite-study-v2/INTEGRATION_AUDIT.md` and ran its read-only
  conversion probe. All requested modules wrapped exactly; no lifecycle defect
  warrants changing the in-flight study. The post-v2 follow-ups are explicit
  output dimensions, a zero-weight-decay ablation, and a permanent
  wrapper-coverage test.

### 2026-09-18 00:18 UTC — live progress refresh
- Refreshed `aggregate_study.json` from committed artifacts: scratch remains
  **30/30**; arm cells are **62 complete, 1 running, 87 not started**.
- The FC arm is now **30/30 complete**. Its paired result matches the
  pointwise direction at every width: C12 -0.36 pp, C10 -0.52 pp, C8 -0.50 pp,
  C6 -0.49 pp, C4 -0.76 pp, C2 -2.41 pp versus the same-seed scratch model.
- The launcher has advanced to `gate_conv/c12-seed2`, which was active at the
  snapshot. Gate-conv C12 seeds 0--1 had completed and were both below their
  paired scratch references (aggregate mean -0.39 pp, n=2).
- Keep the original decision: **do not restart**. The gate/depthwise/control
  results are still needed, and training remains active.

### 2026-09-18 01:21 UTC — user-authorised parallel workers
- The user asked to finish the study faster. The built-in launcher is serial
  and has no worker-count option. Before adding concurrency, checked that the
  M3 Pro host still had ~66% CPU idle, but only ~4.9 GB uncompressed memory
  available with compression already in use. Chose a conservative ceiling of
  **three concurrent training workers**, not an unbounded fan-out.
- Kept the serial launcher on its current `gate_conv/c10-seed3` cell. Started
  two independent cells the serial launcher cannot reach for many more runs:
  `depthwise/c12-seed0` and `control/c12-seed0`. They have separate output
  directories and source checkpoints; both manifests record `status: running`
  and PB startup. Do not start a second process for either path.
- Used the launcher module to render the expected YAML, then checked the two
  on-disk configs byte-for-byte against `build_arm_runs` with absolute paths
  before launch. This matters: when the serial launcher reaches them, its
  `_write_config` byte-equality guard will accept the existing config and its
  report-complete check will skip a successful parallel result.
- Managed session identifiers at launch were 70102 (depthwise) and 98509
  (control). They are only useful while this interactive session exists;
  durable evidence is each run's `manifest.yaml`, internal `logs/`, PAI files,
  and final report under its output directory.
- The earlier `nohup` attempt did **not** start duplicate workers: the execution
  environment reaped the wrappers before the trainer process appeared; their
  output directories contained only the YAML. The managed sessions were then
  used and verified by process list and manifests.

### 2026-09-18 02:11--02:13 UTC — parallel-worker outcome and control blocker
- Read-only aggregate snapshot at 02:11 UTC: **72 arm cells complete**, 76
  unstarted. It reports two `running` cells, but one is stale (see below), so
  process-list evidence is the source of truth for liveness.
- Parallel `depthwise/c12-seed0` completed cleanly. Its final validation
  accuracy was 94.29%, **-0.27 pp** versus its paired seed-0 scratch baseline
  (94.56%). It exercised PB normally (`n→p→n→n`, one integration, no NaNs).
- Parallel `control/c12-seed0` failed at the first history-triggered PAI
  switch after its identity fine-tune and 49 PAI n-mode epochs. PAI printed:
  `load_net ... must be called with a net after perforate_model` and flagged a
  model with no `pai_modules`; the headless debugger conversion raised the
  recorded RuntimeError. This is a **real control-arm implementation blocker**,
  not a parallel-process failure. Its manifest is `failed`, while its partial
  report still says `running`, which explains the aggregate misclassification.
  Do not blindly resume/relaunch this empty-placement cell: vendor PAI cannot
  switch a model that has no wrapper modules. A replacement control design is
  needed before the serial launcher reaches this arm.
- To keep acceleration active without duplicating the failed control, launched
  `depthwise/c12-seed1` from a byte-verified frozen config. At 02:13 UTC it
  had reached PB startup alongside the serial `gate_conv/c8-seed1` worker.
  The active process count is therefore **two**. Managed-session ID for the
  replacement is 39621; durable state is under its output directory.

<!-- APPEND NEW ENTRIES ABOVE THIS LINE -->

---

## 3. What the runs show
_Live snapshot: 2026-09-17T22:25:04Z. Every result below is **validation**
accuracy. The held-out test split remains untouched. Complete per-cell data is
in `notes/dendrite-study-v2/aggregate_study.json`; detailed analysis is in
`notes/dendrite-study-v2/RESULTS_ANALYSIS.md`._

Reproduce the read-only partial snapshot from the base directory:

```bash
cd KWS_Model
uv run python notes/dendrite-study-v2/aggregate_study.py \
  --study-root outputs/sparknet-dendritic-study-v2 \
  --json notes/dendrite-study-v2/aggregate_study.json
```

### 3.1 Current completion state

| Cell type | Complete | Running | Not started |
| --- | ---:| ---:| ---:|
| Scratch baselines | 30 / 30 | 0 | 0 |
| Pointwise arm | **30 / 30** | 0 | 0 |
| FC arm | 22 / 30 | 1 (`C4`, seed 2) | 7 |
| Gate-conv arm | 0 / 30 | 0 | 30 |
| Depthwise arm | 0 / 30 | 0 | 30 |
| Scheduled no-dendrite control | 0 / 30 | 0 | 30 |

The study must continue unchanged. `control`, `gate_conv`, and `depthwise`
are required to distinguish a PAI effect from the identical scheduler and
identity-fine-tune trajectory, so no arm can be selected yet.

### 3.2 Five-seed scratch reference

| Width | Best validation accuracy (mean ± SD) | Parameters |
| --- | ---:| ---:|
| C12 | 93.88% ± 0.46 | 3,400 |
| C10 | 92.80% ± 0.33 | 2,854 |
| C8 | 91.23% ± 0.32 | 2,356 |
| C6 | 87.87% ± 0.68 | 1,906 |
| C4 | 82.87% ± 0.86 | 1,504 |
| C2 | 63.34% ± 1.49 | 1,150 |

### 3.3 Completed pointwise result

Every arm cell starts from its same-seed scratch checkpoint; delta is therefore
paired against the correct baseline.

| Width | Final validation (mean ± SD) | Paired delta vs own scratch | Improved seeds | Params |
| --- | ---:| ---:| ---:| ---:|
| C12 | 93.48% ± 0.59 | -0.40 pp ± 0.14 | 0 / 5 | 3,712 |
| C10 | 92.26% ± 0.37 | -0.54 pp ± 0.10 | 0 / 5 | 3,074 |
| C8 | 90.73% ± 0.35 | -0.49 pp ± 0.10 | 0 / 5 | 2,471 |
| C6 | 87.37% ± 0.88 | -0.50 pp ± 0.40 | 0 / 5 | 1,990 |
| C4 | 82.05% ± 1.16 | -0.81 pp ± 0.62 | 0 / 5 | 1,544 |
| C2 | 60.92% ± 1.69 | -2.42 pp ± 1.60 | 1 / 5 | 1,162 |

This is a clearly negative outcome for the frozen pointwise recipe, not yet a
general verdict on PB. The 40-epoch `identity` (no-pruning) fine-tune changes
the optimizer, label smoothing, LR schedule, and LR before PAI starts, and it
already loses accuracy. Section 6.3 documents that stage-separated confound.
A future recipe change needs fresh scratch baselines; it must not be spliced
into study-v2.

### 3.4 Partial FC result

The complete FC widths point in the same direction: C12 -0.36 pp, C10 -0.52
pp, C8 -0.50 pp, and C6 -0.49 pp against their paired scratch checkpoints
(five seeds each). C4 has only two completed seeds (mean -0.64 pp), so is not
interpretable yet. This is evidence to finish the controls, not evidence to
terminate the study.

## 4. PerforatedAI: how dendrites are supposed to be added
_Source: `notes/dendrite-study-v2/PAI_KNOWLEDGE.md` (1,103 lines; every claim
labelled [SRC] / [VENDOR] / [PREPRINT] / [PEER] / [INFER]). Subagent returned
2026-09-17._

### 4.1 Mechanism (from package source, not marketing)

A PAI dendrite is a **full deep copy of the wrapped module**
(`create_dendrite` -> `UPA.deep_copy_pai(parent_module)`,
`modules_perforatedai.py:1209`). For dendrite `d`:

```
z_d = layers[d](x) + sum_{e<d} W_dd[d][e,:] * a_e
a_d = f(z_d)                      # f = pai_forward_function, default sigmoid
y_out = y + sum_d a_d * W_top[D-1][d,:]
```

So the combination is **additive with one learned scalar gain per output
channel** -- *not* multiplicative and *not* gated. A newly accepted dendrite's
`dendrites_to_top` row is **zero-initialised**, so acceptance is exactly
output-preserving at the moment of the switch.

In `p` mode the base weights leave the optimizer and dendrites train by a
Cascade-Correlation rule via a second hook-fired `backward()`; gradient never
flows through the dendrites (paper Eq. 2 zeroes that term).

### 4.2 VERIFIED: perforated backprop is genuinely active in our runs

The researcher flagged a potential silent-failure mode: the actual dendrite
learning rule lives in the **closed-source `perforatedbp`** wheel (`.so` only,
unreadable), and if it is missing the flag `perforated_backpropagation` stays
False and **every candidate path is skipped, silently degrading dendrites to
zero-gain gradient-descent copies**.

**I checked this directly. It is NOT happening here:**
- `perforatedbp` 3.2.7 is installed:
  `.venv/lib/python3.13/site-packages/perforatedbp/__init__.cpython-313-darwin.so`
- `perforatedai` 3.2.8 is installed.
- Importing the globals module prints **"Building dendrites with Perforated
  Backpropagation"**, and that same line appears throughout
  `outputs/sparknet-dendritic-study-v2/study.log`.

=> Study-v2's dendrites are real PB dendrites. This failure mode is ruled out.

### 4.3 Parameter cost of a dendrite (derived from source)

Total for D dendrites: `D * P_M + D^2 * C`. One dendrite: `P_M + C`
(`P_M` = parameters of the wrapped module, `C` = its output channels).

For SparkNet (40 mel bins, 12 classes):

| Placement | Width | Added params for 1 dendrite |
| --- | --- | --- |
| `.fc` (Linear(32,12)) | any | **+408** |
| `.gate_conv` | C2 | **+128** |
| `.blocks.3.pointwise` | C12 | **+156** |
| `.blocks.3.pointwise` | C2 | **+6** |
| depthwise | C2 | **+60** |

A depthwise dendrite **inherits `groups`**, so it structurally cannot mix
channels -- worth remembering when interpreting the depthwise arm.

### 4.4 The papers -- what is actually claimed, and how strongly

The complete direct-PB reading record is
`notes/dendrite-study-v2/PAI_PAPERS.md`. It identifies **three** PB papers as
of this refresh. All are arXiv preprints, not peer reviewed, and have
PerforatedAI-affiliated authors.

1. **Brenner & Itti, arXiv:2501.18018v2** -- the method paper.
   - TrimNet/Tox21: +13.6% error reduction over **50 seeds**; but 2/50 runs got
     zero dendrites, and the *published* TrimNet (0.860) beats PAI's average
     (0.822).
   - HIST/CSI300: +8.6%, normalised against chance 0.5.
   - EMNIST: 4.3%.
   - **Only the mTAN/PhysioNet experiment is parameter-matched**: Net 0.125 +
     3 dendrites beats Net 1 at under 1/10 the size. This is the single result
     that actually supports "dendrites beat width".
   - The paper states plainly that Net 1 and Net 2 "regularly overfit with the
     addition of the first dendrite". Training cost is ~7x.
2. **Brenner et al., arXiv:2506.00356** -- a hackathon write-up. **No error
   bars; every number is a single run.** 88.7% parameter reduction on IMDB
   BERT-tiny DSN; ProteinBERT at 21% params; MobileNetV3 81.99 -> 83.05%; and a
   x0.5-width MobileNetV3 + dendrites reaching 82.25% at 35% fewer parameters.
3. **Gopal et al., arXiv:2605.15647** -- a KWS/Edge Impulse sweep of 800
   hyperparameter trials, the closest published domain match. It reports a
   1,556-parameter PB model at 93.3% test accuracy versus a 3,859-parameter
   92.1% baseline. It is still a preprint and a sweep-selected endpoint without
   reported independent-seed uncertainty; it motivates this matrix but does
   not validate the SparkNet recipe.

The perforatedai.com headline claim of "70% accuracy improvement" **matches
nothing in either paper.**

Background citations (Cascade-Correlation 1989; Sci Rep 2023; Nat Commun 2025)
were read only as repo summaries/abstracts; four are paywalled and were
explicitly flagged unread.

### 4.5 Defaults that matter (PAI globals)

| Global | Default | Note |
| --- | --- | --- |
| `switch_mode` | `DOING_HISTORY` | |
| `n_epochs_to_switch` | 10 | |
| `history_lookback` | 1 | it is an **EMA**, not a plain mean |
| `improvement_threshold` | `[0.001, 0.0001, 0.0]` | a **schedule indexed by dendrite count**, not a list of alternatives |
| `improvement_threshold_raw` | 1e-5 | **absolute**, therefore scale-sensitive |
| `max_dendrites` | 100 | |
| `max_dendrite_tries` | 2 | |
| `candidate_weight_initialization_multiplier` | 0.01 | |
| `testing_dendrite_capacity` | **True by default** | a debug-ish mode that must be understood before trusting a run |
| `global_candidates` | 1 | values > 1 hit an unimplemented `pdb` trap |
| `p_epochs_to_switch` (PB) | 2 | |

### 4.6 Placement guidance

- Defaults wrap `Conv1d/2d/3d`, `Linear`, and `PAISequential`.
- **Group norms with their preceding layer.**
- **Later layers are the most efficient** place for dendrites.
- Prune any placement whose PB correlation is < 0.001 (noise) or < 0.02
  (wasteful).
- **Never wrap**: Softmax-feeding-NLL, tuple-returning modules, modules reused
  more than once in a forward pass, or frozen/unused modules.

### 4.7 Pitfalls most likely to be biting a 1K-5K parameter model

1. **`weight_decay = 1e-4` is enabled while PAI warns against it.** This is the
   worst confound: decay pulls on a **zero-initialised** dendrite gain, so a
   dendrite can be regularised into irrelevance before it ever contributes.
2. **`.fc` dendrites cost 32% of an entire C2 model**, so the arms are *not*
   budget-comparable to each other -- only against the `control` arm.
3. `out_channels = 2` makes the PB correlation a **2-sample statistic**.
4. With `max_dendrites: 1` only the **first** threshold entry is ever consulted
   (~+0.47 pp), which is plausibly inside seed noise.
5. Zero-init gain + decayed cosine LR + `post_integration_lr_multiplier: 0.25`
   can leave an accepted dendrite **functionally inert**.
6. `testing_dendrite_capacity` defaulting True.
7. Stale model references after `restructured=True`.
8. **5 seeds against the paper's 50.**

A degenerate run already present in the repo was documented as a worked
example: `outputs/compression-run/.../w10_classifier` -- one switch,
`param_counts.csv` flat at 1246, a single row in `best_arch_scores.csv`. That
is a baseline, not a dendrite result.


## 5. Integration audit: our code vs PAI best practice
_Full evidence and reproduction instructions:
`notes/dendrite-study-v2/INTEGRATION_AUDIT.md`; raw conversion-probe output:
`notes/dendrite-study-v2/placements.json`._

### 5.1 Verdict: no live-study blocker

The audit exercised the actual conversion path at every study width and arm.
It observed the exact requested targets at C12:

| Arm | Verified target(s) | One-dendrite parameter projection |
| --- | --- | ---:|
| pointwise | `blocks.2.pointwise`, `blocks.3.pointwise` | 3,712 |
| fc | `fc` | 3,808 |
| gate_conv | `gate_conv` | 3,848 |
| depthwise | `blocks.2.depthwise`, `blocks.3.depthwise` | 4,000 |
| control | none | 3,400 |

PB is active; configuration precedes wrapping; PAI receives validation scores;
the returned model goes back to its device; optimizer/scheduler are rebuilt on
restructure; base optimizer parameters and BatchNorm state are guarded in p
mode; and paired restart state plus clean final export are saved. **Do not
change these paths during v2.**

### 5.2 Forward-looking hazards, not current failures

1. All arms pass `weight_decay: 1e-4` to AdamW although vendor documentation
   cautions against dendrite weight decay. Run a paired zero-decay ablation
   after v2, not a mid-study configuration edit.
2. The desired Conv2d output dimension (`[-1, 0, -1, -1]`) comes from the
   installed PAI default rather than an explicit call. Make it explicit and
   add a regression test before v3.
3. `unwrapped_modules_confirmed=True` suppresses a vendor guard. Today's
   converter probe proves coverage; promote equivalent coverage checks to CI
   before changing the SparkNet module tree.
4. Pointwise-only wrapping deliberately leaves adjacent BatchNorm outside the
   copied module. That is correct isolation for v2, but is not the vendor's
   whole-layer grouping recommendation and should be a separate v3 arm.

## 6. Performance enhancements (incl. removing KD)
_Source: `notes/dendrite-study-v2/ENHANCEMENTS.md` (721 lines). Subagent
returned 2026-09-17. Numbers below are the subagent's; the load-bearing ones
are cross-checked against the results analyst in section 3._

### 6.1 KD verdict: KEEP `--no-KD`. Confidence: HIGH.

Three KD-vs-no-KD comparisons exist in `outputs/`; none is positive.

| # | Control (no-KD) | KD arm | Delta |
| --- | --- | --- | --- |
| A | `phase_b/sparknet_c12_speech_commands_v2_mfcc32_g32` 92.613% | `..._g32_kd_annealed` 91.244% | **-1.37 pp** |
| B | `phase_b/step2_sparknet_c12_light_20260913T215424Z` 92.150% | `..._light_kd_t2_...` 92.150% | **0.00 pp** (exact tie, 4777/5184) |
| C | same as B, 92.15% | superseded T=1 run | -0.10 pp |

Crucial confound: comparison **A**'s teacher is the MFCC-32 DS-CNN-L at **91.76%
val** (`outputs/phase_b/teacher_ds_cnn_l_mfcc32_nw0/metrics/summaries.yaml`) --
*worse than the 92.61% student it teaches*. A therefore measures peer
distillation, not KD. Comparison **B** uses the strong teacher (97.69% test,
`reports/ds_cnn_l_teacher_current.json`) with a well-scaled gradient
(`train_kd_logit_grad_norm_ratio` ~= 1.15) and yields **exactly zero**.

The historical arc: the old `KD_DIAGNOSIS.md` (recoverable via
`git show e0ecadd:KWS_Model/KD_DIAGNOSIS.md`) concluded KD was *mis-weighted,
not broken*; that was fixed (T=1/alpha 0.1 -> T=2/0.5/0.5). So the arc is
mis-weighted -> correctly weighted -> **still unhelpful**. Residual cause is
redundancy: `kd_logit_grad_cosine` 0.72-0.96, i.e. the KD gradient mostly
duplicates the task gradient.

The experiment that would settle it (retrain the MFCC-32 teacher properly ~7 h,
then 5-seed KD vs no-KD at C8 ~1.7 h) is **not recommended** ahead of the
levers in 6.3.

### 6.2 Defects in the current KD recipe (if KD is ever re-enabled)

1. **Blocker -- KD is currently unreachable in SparkNet runs.** Three guards:
   `src/kws/train.py:319-320` and `src/kws/optimize/dendritic.py:2349-2350`
   raise when `task_loss_scale != 1.0` (every arm config sets `100.0`), and
   `src/kws/optimize/sparknet_dendritic_prune_experiment.py:78-79` requires a
   null teacher.
2. **Latent gate collapse.** The guard checks `task_loss_scale` but not
   `sparsity_weight`. `train.py:486-491` adds the gate-sparsity aux term
   *outside* the convex KD mix, so setting `task_loss_scale: 1` with a
   `*_paper` model config (`sparsity_weight: 1.0`) silently gives a 1:1 instead
   of 100:1 ratio -- exactly the "crush the learned gates" failure the config
   comment warns about.
3. **Annealing is inert under PAI.** `train.py:460-464` calls `kd.set_epoch`;
   `dendritic.py` never does.
4. **KD would change dendrite *selection*, not just training.** It is applied
   ungated during `p` phases (`dendritic.py:2474-2489`), so PAI correlates
   candidate dendrites against KD-mixed error. That is not a clean two-factor
   design.
5. Feature KD silently degenerates to two losses under PAI wrapping
   (`src/kws/optimize/kd.py:390-399`).
6. Clean parts: teacher forced to eval (`kd.py:353-356`), same augmented tensor
   for teacher and student (`train.py:487` -> `kd.py:501`), KL and T^2 scaling
   correct.

### 6.3 BURIED LEAD -- larger than both KD and dendrites

**The 40-epoch identity fine-tune loses accuracy.** Pruning method is
`identity` -- nothing is actually removed -- yet the fine-tune phase *reduces*
validation accuracy in **26 of 28** completed pointwise runs, mean **-0.55 pp**,
and **-2.40 pp at C2**. Dendrites then lose a further **-0.21 pp** (positive in
only 4/28). Net effect of the whole arm pipeline: **-0.76 pp versus the source
scratch checkpoint**, while *adding* parameters.

Attributed cause: four unforced recipe changes at the scratch -> fine-tune
handoff -- `label_smoothing` 0.0 -> 0.1, SGD -> AdamW, polynomial_hold ->
cosine, and an LR restart to 1e-3.

If this holds, **study-v2 is currently measuring dendrites through a damaged
fine-tune**, and the damage is width-dependent (-0.16 pp at C12 -> -2.40 pp at
C2), i.e. it is worst exactly where dendrites are supposed to help most.

Note on `task_loss_scale: 100`: it does what the recipe intends **under AdamW**
(Adam is scale-invariant, so only the 100:1 aux ratio survives), but under
**SGD** it acts as a 100x effective-LR multiplier. That is why the scratch and
fine-tune phases sit in different optimisation regimes.

### 6.4 Ranked performance levers

| # | Lever | Expected gain | Compute |
| --- | --- | --- | --- |
| L1 | Repair the scratch -> fine-tune handoff (L/S/schedule/LR) | **+0.5 pp; +2.4 pp @ C2** | **0 h** |
| L2 | Sweep the gate-sparsity ratio (never swept) | +/-0.3-0.8 pp | 1.5 h |
| L3 | Restore augmentation (bg-noise off, SpecAugment off, white noise at -90 dB = a no-op) | +0.5-1.5 pp | 1.5 h |
| L4 | Oversample `_unknown_` (F1 0.878 vs >=0.934 for other classes) | +0.3-0.8 pp | 0.5 h |
| L5 | EMA + smoothed checkpoint selection (none exists today; best-of-200 selection bias ~= +0.5-0.9 pp, i.e. **larger than every effect being measured**) | +0.2-0.5 pp | 0 h |
| L6 | Width-scale `gate_channels` (the 396-param head is 34% of all C2 params) and LR | several pp @ C2 | 3.4 h |
| L7 | Longer post-dendrite resume | +0.1-0.3 pp | 5 h |
| L8 | Reinterpret `improvement_threshold` (a 0.5 pp acceptance bar against a -0.21 pp measured effect) | reporting only | 0 h |

### 6.5 Levers that must be FROZEN across arms

**All eight.** L1 especially: its damage is width-dependent, so it
differentially depresses the narrow widths where dendrites should shine. L6
re-ranks arms mechanically. L5 changes the very selection rule by which arms
are chosen. L3/L4 flip the under/over-fitting regime.

**Practical consequence: study-v2 must finish on its current recipe.** These
levers define a *v3* recipe and will need fresh scratch baselines.

### 6.6 Proposed next experiment (cheapest decisive step)

**Identity-fine-tune ablation, no PAI at all.** 3 recipes (A = current,
B = repaired handoff, C = `epochs: 0`) x 3 widths (C12, C6, C2) x 5 seeds
= 45 runs. Reuses the existing scratch checkpoints; only the 2.3-min fine-tune
phase runs. **~1.7 h.**

Follow-on if B or C wins: `{control, pointwise}` x 3 widths x 5 seeds
= 30 runs x 11.0 min ~= **5.5 h**. Total ~7.2 h, against **22.4 h** to finish
the current matrix.

Measured timings (parsed from `study.log`): scratch 10.2 min/run, arm
11.0 min/run; per-phase `prune_supervised` 2.3 min / `pai` 9.4 min /
`resume` 0.7 min.

### 6.7 Side finding -- missing audit files

`KD_DIAGNOSIS.md`, `PLAN.md`, `ERRORS.md`, and `SPARK_NET_FIXES.md` are **absent
from the working tree**, even though `SPARKNET_DENDRITE_FIXES.md` claims they
were restored and `README.md` still cites `PLAN.md`. Recoverable at `e0ecadd`
or `9c1b179^`.


## 7. Open questions / next actions for a future agent
- [ ] Study-v2 must finish (or be deliberately truncated) before arm selection.
