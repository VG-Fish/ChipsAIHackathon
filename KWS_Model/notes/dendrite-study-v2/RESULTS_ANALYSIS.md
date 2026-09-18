# SparkNet + PerforatedAI dendrite study — results analysis

**Historical detailed snapshot: 2026-09-17T18:18:03Z. The study was still
running.**

**Current status supersedes its progress counts:**
[`aggregate_study.json`](aggregate_study.json) was refreshed at
2026-09-17T22:25:04Z and the authoritative summary is journal §3 in
[`../../DENDRITE_JOURNAL.md`](../../DENDRITE_JOURNAL.md). At that refresh,
pointwise was 30/30 complete and FC C4 seed 2 was running. The tables below
remain useful as a documented pointwise analysis, but their inventory and the
in-flight-cell sentence are historical.

The launcher `scripts/run_sparknet_dendritic_study.py` (PID 12115, started
2026-09-17 05:45 local / 09:45 UTC) was executing
`arms/pointwise/c2-seed4` at the moment this snapshot was taken. Every number
below is read from artifacts already on disk; nothing in `outputs/` was
modified and no process was interrupted.

**Every accuracy in this document is validation accuracy.** No run in this
study has evaluated the test split — every study report carries
`selection_split: validation` and `test_split_used: false`, and
`outputs/sparknet-dendritic-study-v2/selection/` does not exist, so the
one-time held-out-test stage has not run. Do not quote any figure here as a
test number, and do not compare any of it to the SparkNet paper's
95.7 ± 0.17, which is a test figure.

Reproduce every table with:

```bash
uv run python notes/dendrite-study-v2/aggregate_study.py \
  --study-root outputs/sparknet-dendritic-study-v2 \
  --json notes/dendrite-study-v2/aggregate_study.json
```

The script is `notes/dendrite-study-v2/aggregate_study.py`; its JSON output
(`notes/dendrite-study-v2/aggregate_study.json`) carries every per-cell value
quoted below, including per-seed rows.

Why a new script rather than the study's own aggregator: `scripts/select_sparknet_arms.py`
raises on the first width whose arm set is incomplete (`build_selection`,
"missing C{width} {arm} seeds"), so it cannot report a matrix with four of five
arms untrained. `scripts/report_test_accuracy.py` was deliberately **not** run:
it loads the test split, and running it before the validation selection is
frozen would spend the study's one-shot test budget.

---

## 1. Inventory and completeness

### Scratch baselines — complete

`outputs/sparknet-dendritic-study-v2/scratch/c{W}-seed{S}/`

| | |
|---|---|
| Planned | 6 widths (C12, C10, C8, C6, C4, C2) × 5 seeds = **30** |
| Complete | **30 / 30** |
| Completeness test | `metrics/summaries.yaml` present with one phase, `models/checkpoints/paper_replication/best.pt` present, `manifest.yaml` `status: completed` |
| Epochs | 200 in every cell (`completed_epoch: 200`) |
| Recipe | `configs/train/sparknet_narrow_paper_fast_io.yaml` (SGD, lr 0.01, momentum 0.9, wd 1e-3, batch 128, `task_loss_scale: 100`, polynomial-hold schedule) |

### Dendrite arms — 1 of 5 arms run

`outputs/sparknet-dendritic-study-v2/arms/{arm}/c{W}-seed{S}/`

| Arm | Complete | Running | Not started |
|---|---:|---:|---:|
| `pointwise` (`.blocks.2.pointwise`, `.blocks.3.pointwise`) | **29 / 30** | 1 (`c2-seed4`) | 0 |
| `fc` (`.fc`) | 0 | 0 | 30 |
| `gate_conv` (`.gate_conv`) | 0 | 0 | 30 |
| `depthwise` (`.blocks.2/3.depthwise`) | 0 | 0 | 30 |
| `control` (empty placement, budget-matched) | 0 | 0 | 30 |
| **Total** | **29** | **1** | **120** |

Complete `pointwise` cells: C12 seeds 0–4, C10 seeds 0–4, C8 seeds 0–4,
C6 seeds 0–4, C4 seeds 0–4, C2 seeds 0–3. The only incomplete cell in the arm
is C2 seed 4.

Arm recipe (`configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_pointwise.yaml`),
identical in all five arm configs except `perforatedai.module_ids`:
identity "prune" (`pruning.method: identity`, `prune_fraction: 0.0`) →
40-epoch fine-tune at lr 1e-3, batch 256, label smoothing 0.1 →
PAI search (`max_dendrites: 1`, `switch_mode: history`, `history_lookback: 8`,
`n_epochs_to_switch: 10`, `improvement_threshold: [0.005, 0.002, 0.001]`,
`max_dendrite_tries: 3`, `dendritic_schedule_epochs: 30`) →
8-epoch supervised resume. **This is a different recipe from the historical
`max_dendrites: 3` runs in §6; the two families must not be pooled.**

### Wall-clock cost and projection

Derived from `manifest.yaml` `started_at` / `ended_at` of each run directory.

| Stage | n | Mean per run | Total |
|---|---:|---:|---:|
| Scratch baseline | 30 | 624 s (C12 668 s → C2 609 s) | 5.20 h of compute |
| `pointwise` arm run | 29 completed | 676 s (median 656 s) | 5.45 h of compute |

Runs are strictly serial with ~1 s of launcher overhead between them. The
observed steady-state throughput of the main arm block — 26 runs from
13:11:28Z to 18:04:11Z — is **675 s/run including overhead** (4.88 h / 26).

Outlier: `pointwise/c8-seed0` took **1299 s** (286 PAI epochs against a
105–139 range everywhere else); see §5.

**Projection.** 121 runs remain (1 in flight + 120 not started).
At 675 s/run that is **81,675 s ≈ 22.7 h**, finishing around
**2026-09-18T17:00Z**. Caveats on that number: it is extrapolated from the
`pointwise` arm only; the `control` arm has an empty placement and may be
cheaper, while `fc`, `gate_conv` and `depthwise` copy differently sized
modules, and PAI's stopping behaviour (not a fixed epoch cap) is what actually
sets each run's length. A `c8-seed0`-style exhausted search (one in 29 so
far) adds ~10 minutes each time it happens.

---

## 2. Scratch baselines — the honest no-dendrite reference

Validation accuracy over 5 independently seeded 200-epoch runs; SD is the
sample SD (n = 5, ddof = 1). `best` is the accuracy of the saved `best.pt`,
which is the checkpoint every arm at that (width, seed) starts from, so it is
the correct paired reference. Parameters and MACs are the values the arm
reports record for the exact loaded checkpoint
(`source.deployed_params`, `source.macs`) — identical across all cells at a
given width.

| Width | n | Best val (mean ± SD) | Final-epoch val (mean) | Deployed params | MACs | Mean wall clock |
|---|---:|---|---|---:|---:|---:|
| C12 | 5 | **93.88 % ± 0.46** | 93.74 % | 3,400 | 277,124 | 668 s |
| C10 | 5 | **92.80 % ± 0.33** | 92.53 % | 2,854 | 224,806 | 628 s |
| C8 | 5 | **91.23 % ± 0.32** | 90.92 % | 2,356 | 177,336 | 616 s |
| C6 | 5 | **87.87 % ± 0.68** | 87.45 % | 1,906 | 134,714 | 614 s |
| C4 | 5 | **82.87 % ± 0.86** | 82.20 % | 1,504 | 96,940 | 610 s |
| C2 | 5 | **63.34 % ± 1.49** | 61.70 % | 1,150 | 64,014 | 609 s |

Context only, not part of this study:
`outputs/sparknet-paper-replication/c16-seed0/metrics/summaries.yaml` records
C16 at 95.32 % best validation, and
`outputs/sparknet-paper-replication/c16-seed0/test_report_fixedseed0.json`
records 95.07 % **test** accuracy at 4,636 params.

---

## 3. Pointwise dendrite arm

`final` = `candidates[0].dendritic.validation_accuracy` in each run's
`reports/sparknet_dendritic_prune_experiment.yaml` — the field
`scripts/select_sparknet_arms.py` selects on.

The paired delta is computed per seed against that seed's own scratch
`best_val_acc`, i.e. against the exact checkpoint that arm run loaded
(verified: every arm report's `source.checkpoint` points at
`scratch/c{W}-seed{S}/models/checkpoints/paper_replication/best.pt`, and its
`source.validation_accuracy` equals that run's `best_val_acc`). The t-test is
a one-sample t on the paired differences (H0: mean delta = 0), two-sided,
computed with `scipy.stats.t`.

| Width | n | Final val (mean ± SD) | Paired Δ vs own scratch (mean ± SD) | Seeds improved | t | p (2-sided) | Mean params |
|---|---:|---|---|---:|---:|---:|---:|
| C12 | 5 | 93.48 % ± 0.59 | **−0.40 pp ± 0.14** | **0 / 5** | −6.20 | 0.0034 | 3,712 |
| C10 | 5 | 92.26 % ± 0.37 | **−0.54 pp ± 0.10** | **0 / 5** | −12.48 | 0.0002 | 3,074 |
| C8 | 5 | 90.73 % ± 0.35 | **−0.49 pp ± 0.10** | **0 / 5** | −11.44 | 0.0003 | 2,471 |
| C6 | 5 | 87.37 % ± 0.88 | **−0.50 pp ± 0.40** | **0 / 5** | −2.82 | 0.0480 | 1,990 |
| C4 | 5 | 82.05 % ± 1.16 | **−0.81 pp ± 0.62** | **0 / 5** | −2.92 | 0.0431 | 1,544 |
| C2 | 4 | 60.47 % ± 1.56 | **−2.35 pp ± 1.85** | 1 / 4 | −2.55 | 0.0841 | 1,162 |

**0 of 25 seeds at C12–C4 improved on their own scratch baseline.** At C2
(4 of 5 seeds so far) one seed improved by +0.07 pp and three lost 1.9–3.9 pp.

### Decomposition: which stage loses the accuracy?

The arm pipeline has two stages before the number above is produced. Both
deltas are paired against the same scratch baseline.

| Width | scratch best | after 40-epoch identity fine-tune | Δ (fine-tune) | after PAI + resume | Δ (total) | Δ attributable to PAI stage |
|---|---|---|---|---|---|---|
| C12 | 93.88 % | 93.71 % | −0.16 pp | 93.48 % | −0.40 pp | −0.24 pp |
| C10 | 92.80 % | 92.62 % | −0.18 pp | 92.26 % | −0.54 pp | −0.36 pp |
| C8 | 91.23 % | 90.96 % | −0.27 pp | 90.73 % | −0.49 pp | −0.23 pp |
| C6 | 87.87 % | 87.56 % | −0.31 pp | 87.37 % | −0.50 pp | −0.19 pp |
| C4 | 82.87 % | 82.17 % | −0.69 pp | 82.05 % | −0.81 pp | −0.12 pp |
| C2 | 62.82 %¹ | 60.78 % | −2.05 pp | 60.47 % | −2.35 pp | −0.30 pp |

¹ mean over the 4 seeds that have an arm result, not the 5-seed 63.34 %.

Roughly **40–85 % of the loss happens before PAI is involved**: the 40-epoch
identity fine-tune at the arm's optimizer/schedule (lr 1e-3, batch 256, label
smoothing 0.1) moves the network off its 200-epoch SGD optimum and does not
get back. This is the confound the `control` arm exists to measure, and the
`control` arm has not run.

### Within-PAI dendrite delta (labelled; not a causal control)

`dendritic.validation_accuracy` minus `dendritic.zero_dendrite_validation_accuracy`,
i.e. the best-architecture row minus the minimum-parameter row of PAI's own
`*_best_arch_scores.csv`. Per `SPARKNET_DENDRITE_FIXES.md` this is a
*within-search* comparison, **not** a matched no-dendrite counterfactual.

| Width | Mean ± SD | Seeds > 0 | t | p |
|---|---|---:|---:|---:|
| C12 | +0.085 pp ± 0.070 | 5 / 5 | 2.73 | 0.053 |
| C10 | +0.031 pp ± 0.070 | 1 / 5 | 1.00 | 0.374 |
| C8 | +0.013 pp ± 0.030 | 1 / 5 | 1.00 | 0.374 |
| C6 | +0.004 pp ± 0.010 | 1 / 5 | 1.00 | 0.374 |
| C4 | +0.067 pp ± 0.067 | 3 / 5 | 2.24 | 0.089 |
| C2 | +0.022 pp ± 0.045 | 1 / 4 | 1.00 | 0.391 |

Even taken at face value, the largest within-search dendrite effect
(+0.085 pp at C12) is **one fifth** of the pipeline loss at that width
(−0.40 pp) and one eighth of the scratch seed SD (0.46 pp).

---

## 4. Parameter / MAC accounting and the width trade

### What one dendrite costs

From `candidates[0].dendritic.one_dendrite_cost_projection` (C12 base
`.../c12-seed0/reports/...yaml`, same structure at every width). PAI copies
both pointwise modules (`blocks.2.pointwise`, `blocks.3.pointwise`), each
C×C, plus one residual scale term per module per dendrite.

| Width | Base params | Copied | Residual | Total params | Δ params | Base MACs | Δ MACs |
|---|---:|---:|---:|---:|---|---:|---|
| C12 | 3,400 | 288 | 24 | 3,712 | **+312 (+9.18 %)** | 277,124 | +31,512 (+11.37 %) |
| C10 | 2,854 | 200 | 20 | 3,074 | **+220 (+7.71 %)** | 224,806 | +22,220 (+9.88 %) |
| C8 | 2,356 | 128 | 16 | 2,500 | **+144 (+6.11 %)** | 177,336 | +14,544 (+8.20 %) |
| C6 | 1,906 | 72 | 12 | 1,990 | **+84 (+4.41 %)** | 134,714 | +8,484 (+6.30 %) |
| C4 | 1,504 | 32 | 8 | 1,544 | **+40 (+2.66 %)** | 96,940 | +4,040 (+4.17 %) |
| C2 | 1,150 | 8 | 4 | 1,162 | **+12 (+1.04 %)** | 64,014 | +1,212 (+1.89 %) |

Realised per-width means differ from the projection only at C8, where
`c8-seed0` ended with no dendrite at all (2,356 params) and pulls the C8 mean
to 2,471.2 params / 188,971 MACs.

### What width costs on the same curve

Marginal cost/benefit along the scratch baselines of §2:

| Step | Δ params | Δ val | Accuracy per parameter |
|---|---:|---|---|
| C2 → C4 | +354 | +19.53 pp | +5.52 pp per 100 params |
| C4 → C6 | +402 | +5.01 pp | +1.25 pp per 100 params |
| C6 → C8 | +450 | +3.35 pp | +0.75 pp per 100 params |
| C8 → C10 | +498 | +1.57 pp | +0.32 pp per 100 params |
| C10 → C12 | +546 | +1.08 pp | +0.20 pp per 100 params |

### The decision-relevant comparison: dendrite vs. a wider network

**(a) Parameter-matched.** A dendrite buys a *fraction* of a width step, so
the fair alternative is the point on the scratch width curve that costs the
same as `C{W} + dendrite`. Linear interpolation of the measured scratch means
between the two bracketing widths; every arm's cost falls inside its bracket,
so this is interpolation and not extrapolation.

| Arm | Params | Arm val | Scratch curve at the same params | **Δ** | Bracket |
|---|---:|---|---|---|---|
| C10 + dendrite | 3,074 | 92.26 % | 93.23 % | **−0.97 pp** | C10 (2,854 / 92.80 %) – C12 (3,400 / 93.88 %) |
| C8 + dendrite | 2,471 | 90.73 % | 91.59 % | **−0.86 pp** | C8 (2,356 / 91.23 %) – C10 (2,854 / 92.80 %) |
| C6 + dendrite | 1,990 | 87.37 % | 88.50 % | **−1.13 pp** | C6 (1,906 / 87.87 %) – C8 (2,356 / 91.23 %) |
| C4 + dendrite | 1,544 | 82.05 % | 83.36 % | **−1.31 pp** | C4 (1,504 / 82.87 %) – C6 (1,906 / 87.87 %) |
| C2 + dendrite | 1,162 | 60.47 % | 64.00 % | **−3.53 pp** | C2 (1,150 / 63.34 %) – C4 (1,504 / 82.87 %) |

C12 + dendrite (3,712 params) has no bracket: no C14 scratch baseline exists,
so this study cannot answer the question at its widest point.

**(b) Full width step.** The cruder version of the same comparison — the
dendrite arm against the *next width up*, which costs more:

| Arm | Params | Val | vs. scratch | Params | Val | Δ val | Δ params |
|---|---:|---|---|---:|---|---|---|
| C10 + dendrite | 3,074 | 92.26 % | C12 | 3,400 | 93.88 % | −1.62 pp | −326 |
| C8 + dendrite | 2,471 | 90.73 % | C10 | 2,854 | 92.80 % | −2.07 pp | −383 |
| C6 + dendrite | 1,990 | 87.37 % | C8 | 2,356 | 91.23 % | −3.86 pp | −366 |
| C4 + dendrite | 1,544 | 82.05 % | C6 | 1,906 | 87.87 % | −5.82 pp | −362 |
| C2 + dendrite | 1,162 | 60.47 % | C4 | 1,504 | 82.87 % | −22.39 pp | −342 |

**Verdict: on this arm, with this recipe, the dendrite is strictly worse than
spending the same parameters on width — by 0.86 to 3.53 pp at matched
parameter count.** It is also worse than *not spending them at all* (§3): the
dendrite arm is below the same-width scratch baseline everywhere. The MAC
picture is worse than the parameter picture, because a copied pointwise module
runs over the full (C, T) tensor: MAC growth exceeds parameter growth at every
width (+11.4 % vs +9.2 % at C12, +9.9 % vs +7.7 % at C10).

---

## 5. PAI internals health check

Read from every complete arm run's
`pai/candidates/sparknet_c{W}_multilayer/`:
`*switch_epochs.csv`, `*param_counts.csv`, `*_best_arch_scores.csv`,
`*Scores.csv`, and `cycle_metadata.yaml` (`result.phase_trail`).

### Aggregate (all 29 complete runs; every one has a closed PAI cycle)

| Property | Value |
|---|---|
| `cycle_metadata.yaml` `status` | `complete` in all 29 |
| Mode trajectory | `n → p → n → n` in 28 runs; `n p n p n p n n` in 1 (`c8-seed0`) |
| Switches recorded in `switch_epochs.csv` | 2 in 28 runs; **0 in `c8-seed0`** |
| First switch epoch | min 36, max 46, mean 39.4 |
| Second switch epoch | min 41, max 69, mean 54.8 |
| Total PAI epochs | mean 131.1, min 105, max **286** (`c8-seed0`) |
| Dendrites integrated (`dendrites_integrated_after`) | 1 in 28 runs; **0 in `c8-seed0`** |
| Architectures scored in `*_best_arch_scores.csv` | 2 rows in 28 runs; 1 row in `c8-seed0` |
| **Dendrite architecture held PAI's best score** | **9 / 29** |
| **Dendrite architecture rejected** (min-param row held the best score) | **20 / 29** |
| Post-PAI resume outcome | `no_improvement` in 25, `complete` (resume improved) in 4 |
| NaN validation scores | **0** across 3,001 recorded validation scores |

Nobody hit `max_dendrites`-as-a-wall in a way that truncated the search:
`max_dendrites: 1` and 28 of 29 runs integrated exactly one dendrite, so the
cap bound every successful run by construction. `history_lookback: 8` and
`n_epochs_to_switch: 10` are both far shorter than the observed 36–46 epoch
first switch, so no run switched prematurely.

### Where the dendrite did win the search

`dendrite_architecture_won_search = True` in exactly 9 cells, and the margin
(best row minus min-param row in `*_best_arch_scores.csv`) is tiny:

| Cell | Arch-score gain |
|---|---|
| `c12-seed0` / `seed1` / `seed2` / `seed3` / `seed4` | +0.022 / +0.045 / +0.090 / +0.090 / +0.067 pp |
| `c10-seed0` | +0.157 pp |
| `c8-seed4` | +0.067 pp |
| `c6-seed4` | +0.022 pp |
| `c2-seed0` | +0.090 pp |

The largest dendrite benefit PAI ever measured anywhere in this arm is
**+0.157 pp** — about a third of the C10 scratch seed SD (0.33 pp).

### Degenerate / notable runs

1. **`pointwise/c8-seed0` — dendrite search fully exhausted and abandoned.**
   `.../c8-seed0/pai/candidates/sparknet_c8_multilayer/sparknet_c8_multilayerswitch_epochs.csv`
   contains only a header. `..._best_arch_scores.csv` has one row
   (2,356 / 0.90956). The phase trail is `n p n p n p n n` — three dendrite
   candidate phases, matching `max_dendrite_tries: 3`, all rejected — and PAI
   reverted to the base architecture. 286 epochs, 1,299 s, twice every other
   run. This is honest behaviour, not corruption, but it is a different
   experiment from the other 28 cells and its `deployed_params` (2,356) is the
   only one in the arm that equals the base.
2. **No validation collapse after a switch.** Worst dip within 5 epochs of any
   recorded switch, over all 29 runs: `c2-seed3` −1.35 pp, `c2-seed2` −0.88 pp,
   `c2-seed1` −0.85 pp, `c4-seed2` −0.38 pp; everything else ≤ 0.34 pp. All of
   the meaningful dips are at C2, the width with the widest seed noise. No
   NaNs anywhere.
3. **The resume is almost always wasted.** 25 of 29 runs end
   `resume.status: no_improvement`; the 8 extra epochs at
   `post_integration_lr_multiplier: 0.25` never beat PAI's own best.
4. **PAI trained on `mps`, cost was profiled on `cpu`.** Every
   `pai/candidates/*/sparknet_c{W}_multilayer_config.json` records
   `"device": "mps"`, while `candidates[0].dendritic.full_cost.device` is
   `cpu`. Not a bug, but the latency figures in `full_cost` are CPU-only.

---

## 6. Historical runs (pre-study) — NOT comparable

These families prune down from an already-trained checkpoint with a different
dendrite cap, different thresholds and a different epoch budget. They share no
experimental identity with the scratch-start study above and must never be
pooled with it.

### Caveats carried forward verbatim from `SPARKNET_DENDRITE_FIXES.md`

> The five historical directories are not five independently seeded downstream
> runs: their prune/PAI sidecars all report seed 0, and their manifests report
> null. Only the source checkpoints differ by directory seed.

> The prior report headline changed meaning whenever a PAI minimum-parameter
> row was present: final post-resume accuracy was subtracted from that row and
> called a dendrite gain. The headline now remains the final end-to-end delta
> versus the prune/fine-tune baseline. PAI-row deltas are separate, explicitly
> labeled within-search comparisons; they are not causal no-dendrite controls.

> The corresponding final-minus-prune deltas are +2.50, +5.19, and +8.12
> percentage points, but those include the longer PAI and post-PAI resume
> schedules. Final-minus-PAI-minimum-row deltas average only +0.23, +0.45, and
> +0.88 points; these are search-row comparisons, not causal dendrite effects.
> No test accuracy is present.

This analysis independently re-derived those numbers from the reports and they
match to the digit. Confirmed again here: all five report files carry
`seed: null`, so the ±SD column below is a spread **over directories**, not
over seeds.

### `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/` — 5 dirs, all `complete`

Source: `outputs/sparknet-paper-replication/c16-seed0/.../best.pt`, C16,
95.32 % val, 4,636 params. Placement `.fc`, `max_dendrites: 3`,
thresholds `[0.001, 0.0001, 0.0]`, `n_epochs_to_switch: 25`.

| Width | n | Final val (± over dirs) | Prune-fine-tune baseline | Δ vs prune-ft | Δ vs PAI zero-dendrite row | Mean params | Scratch C{W} (§2) |
|---|---:|---|---|---|---|---:|---|
| C12 | 5 | 92.65 % ± 0.32 | 90.15 % | +2.50 pp | +0.229 pp | 4,060 | 93.88 % at 3,400 params |
| C10 | 5 | 90.98 % ± 0.26 | 85.79 % | +5.19 pp | +0.445 pp | 3,519 | 92.80 % at 2,854 params |
| C8 | 5 | 88.04 % ± 0.62 | 79.93 % | +8.12 pp | +0.882 pp | 3,189 | 91.23 % at 2,356 params |

The report's own `validation_accuracy_above_pruning_curve` field averages
−0.08 pp (C12), +0.67 pp (C10), +0.29 pp (C8) — i.e. against its own measured
pruning curve the `.fc` dendrite result is roughly a wash.

### `outputs/sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/` — **no results**

All five `seed*/reports/sparknet_dendritic_prune_experiment.yaml` files carry
`status: running` with every candidate at `status: planned`, and no
`pai/candidates/*/cycle_metadata.yaml` exists in any of them. The run was
started 2026-09-16 22:26 and abandoned by 23:54. **There is no gate_conv d3
result to report.** Any prior claim about gate_conv performance is unsupported
by this directory.

### Other historical directories

| Directory | State |
|---|---|
| `outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3/` | `status: running`; C10 complete (92.32 % from a 92.61 % Phase-B **student** source, 4,114 params); C8 and C6 `planned`. Different source family entirely. |
| `outputs/sparknet-c12-dendritic-prune-no-kd-unlimited/` | `status: running`; all three widths `planned`. No results. |

---

## 7. Findings

### Supported by data

1. **The pointwise dendrite arm loses to its own from-scratch baseline at
   every width measured.** Paired mean deltas: −0.40 pp (C12), −0.54 (C10),
   −0.49 (C8), −0.50 (C6), −0.81 (C4), −2.35 (C2). **0 of 25 seeds at C12–C4
   improved.** Paired two-sided p ≤ 0.048 at all five of those widths.
   Source: `notes/dendrite-study-v2/aggregate_study.json` →
   `summary.arms[*].paired_delta_final_vs_scratch_*`, derived from each
   `arms/pointwise/c{W}-seed{S}/reports/sparknet_dendritic_prune_experiment.yaml`
   and `scratch/c{W}-seed{S}/metrics/summaries.yaml`.
2. **At matched parameters, width beats the dendrite at every testable
   width — by 0.86 to 3.53 pp.** Comparing each arm against the interpolated
   scratch width curve at its own parameter count: −0.97 (C10), −0.86 (C8),
   −1.13 (C6), −1.31 (C4), −3.53 pp (C2). Source: §4(a), built from the
   §2 scratch table. This is the single decision-relevant number and it points
   one way.
3. **Most of the loss is not the dendrite — it is the arm's 40-epoch
   fine-tune.** The identity fine-tune alone costs −0.16 to −0.69 pp (−2.05 pp
   at C2) before PAI runs; the PAI stage adds a further −0.12 to −0.36 pp.
   At C12–C4 the fine-tune accounts for 40–85 % of the total loss.
   Source: `candidates[0].baseline.validation_accuracy` vs
   `source.validation_accuracy` in each arm report; table in §3.
4. **PAI's own search rejected the dendrite in 20 of 29 completed cells**, and
   where it accepted, the measured margin was +0.022 to +0.157 pp. Source:
   `pai/candidates/*/[name]_best_arch_scores.csv` in each run — the
   minimum-parameter row holds the best score in 20 of 29 files.
5. **A dendrite on the SparkNet pointwise convs costs proportionally more
   compute than parameters.** +9.18 % params / +11.37 % MACs at C12,
   +7.71 % / +9.88 % at C10, +6.11 % / +8.20 % at C8. Source:
   `candidates[0].dendritic.one_dendrite_cost_projection` and
   `candidates[0].comparison.{parameter,mac}_growth_fraction`.
6. **The study is mechanically healthy.** 30/30 scratch baselines complete at
   200 epochs; 29/30 pointwise cells complete; zero NaNs across 3,001 PAI
   validation-score records; no post-switch collapse worse than −1.35 pp; every
   arm report's `source.checkpoint` verified to point at its own seed's scratch
   baseline, so the paired design is intact (this was the exact failure mode
   `SPARKNET_DENDRITE_FIXES.md` flagged in the historical runs, and it has been
   fixed here — every report carries an explicit integer `seed`).
7. **The `gate_conv` d3 historical family produced no results at all** — five
   `planned` reports, zero cycle metadata. §6.
8. **No test accuracy exists for any run in this study.** Every report:
   `selection_split: validation`, `test_split_used: false`; no
   `selection/selected_arms.json` and no `selection/test_report.json`.

### Hypothesis (not established by these data)

9. *The arm's fine-tune recipe, not the dendrite placement, is the binding
   constraint.* Finding 3 shows the identity fine-tune is where most of the
   loss occurs; if that is causal, then all five arms will land below their
   scratch baselines and the arm comparison will be measuring recipe damage
   rather than placement quality. **The `control` arm (empty placement, same
   epoch budget) is the test of this and it has not run.** Until it does, no
   claim of the form "dendrites hurt" can be separated from "this fine-tune
   hurts".
10. *A single dendrite is simply too small an intervention at this scale.*
    The largest within-search dendrite effect anywhere in the arm (+0.157 pp)
    is below the seed noise floor at every width (scratch SD 0.32–1.49 pp).
    If true, the `max_dendrites: 1` cap adopted for this study cannot resolve a
    dendrite effect from seed noise at n = 5, regardless of placement. Testing
    this would need more dendrites or more seeds, not a different placement.
11. *Pointwise is the arm most likely to fail.* The config's own rationale
    argues pointwise maps onto what narrowing removes. That reasoning is
    untested: with four arms unrun, nothing here says whether `fc`,
    `gate_conv` or `depthwise` behaves differently.

---

## 8. Red flags and data-quality problems

1. **Accuracy and cost come from different architectures in 19 of 29 cells.**
   When PAI's best score sits on the minimum-parameter row,
   `dendritic.validation_accuracy` is a score achieved by the *zero-dendrite*
   architecture (`pai_deployed_params` == `zero_dendrite_params`), while
   `dendritic.deployed_params` / `macs` are profiled on the exported clean
   graph, which carries the dendrite. `src/kws/optimize/dendritic.py:2791-2792`
   makes this explicit: `pai_deployed_params = deployed_params` (PAI's best-row
   count) then `deployed_params = int(cost["params"])` (the profiled export).
   Example: `arms/pointwise/c10-seed1` reports
   `validation_accuracy: 0.91699`, `pai_deployed_params: 2854`,
   `deployed_params: 3074`, `macs: 247026`, and its
   `*_best_arch_scores.csv` shows 2,854 → 0.91699 beating 3,074 → 0.91384.
   The affected cells are C10 seeds 1–4, C8 seeds 1–3, C6 seeds 0–3, C4 seeds
   0–4, C2 seeds 1–3 (19 cells).
   **Consequence:** the parameter/MAC columns in §3–§4 are the cost of a model
   that has not been shown to reach the reported accuracy, and
   `scripts/select_sparknet_arms.py` would freeze exactly this
   accuracy/checkpoint pair for test evaluation. *Resolve before any test
   number is produced:* evaluate `final_clean_pai.pt` directly and confirm it
   reproduces `dendritic.validation_accuracy`. If it does not, §4 understates
   the case against the dendrite rather than overstating it — but the numbers
   would still need correcting.
2. **No budget-matched no-dendrite control exists yet.** The `control` arm is
   0/30. Without it, every delta in §3 confounds "dendrite" with "40 fine-tune
   epochs + ~131 PAI epochs + 8 resume epochs at a different optimizer,
   batch size and label smoothing". §3's decomposition is the best available
   substitute and it already attributes most of the loss to the non-dendrite
   stage.
3. **C2 is incomplete and unstable.** 4 of 5 seeds; SD of the paired delta is
   1.85 pp, wider than the effect at any other width; p = 0.084. It also holds
   the three worst post-switch validation dips in the arm. C2 also has
   the widest scratch SD (1.49 pp) and the lowest accuracy (63 %), so it is
   near the width at which the architecture stops working. Treat C2 as
   directional only.
4. **`pointwise/c8-seed0` is not the same experiment as its four siblings.**
   Zero dendrites integrated, 286 epochs, base parameter count. It is included
   in the C8 mean above (its paired delta, −0.52 pp, happens to be
   unremarkable), but it makes the C8 mean parameter count (2,471) a number no
   single model has.
5. **C12 cannot be answered at matched parameters.** No C14 scratch baseline
   exists, so the widest and most decision-relevant width has no width-curve
   bracket. Training C14 (or C13) seeds would close this.
6. **One scratch cell was retrained after an interrupted attempt.**
   `study.log` records `FAIL C10 seed4` with
   `ValueError: metrics already contain 77 records for
   paper_replication/sparknet_c10_paper`. The current
   `scratch/c10-seed4/manifest.yaml` shows a **single** invocation
   (`f944a594a7d1`, 09:21:25Z → 09:32:30Z, `status: completed`) and the metrics
   file holds exactly 200 records, so the cell was cleanly retrained rather
   than resumed. No correction is needed, but the directory was manually
   cleared between attempts, which is not recorded in any artifact other than
   the log.
7. **The study's width set changed mid-run.** `study.log` carries
   `=== RESTART 2026-09-17 05:17:39 : widths expanded to 12/10/8/6/4 ===` and
   `=== RESTART 2026-09-17 05:45:05 : widths 12/10/8/6/4/2 ===`. C12 cells
   (`scratch/c12-seed0`, `arms/pointwise/c12-seed0`) were run as calibration at
   03:21 and 03:33 local, hours before the launcher that produced everything
   else. Same configs, same machine, but they did not share the launcher's
   process environment.
8. **The `.run.lock` files are still present in every run directory**,
   including completed ones. Harmless here, but it means "lock exists" cannot
   be used as an in-flight signal.
9. **Snapshot risk.** `pointwise/c2-seed4` was running when this was written
   and the `fc` arm will start immediately after. Every count in §1 and the C2
   row in §3 will change. Re-run `aggregate_study.py` before quoting any of it.

---

## Appendix: field glossary

| Field in `reports/sparknet_dendritic_prune_experiment.yaml` | Meaning used here |
|---|---|
| `source.validation_accuracy` | the scratch baseline's best val acc (paired reference) |
| `candidates[0].baseline.validation_accuracy` | after the 40-epoch identity fine-tune, before PAI |
| `candidates[0].dendritic.validation_accuracy` | the arm's headline; what `select_sparknet_arms.py` selects on |
| `candidates[0].dendritic.pai_search_validation_accuracy` | best inside the PAI search, before the resume |
| `candidates[0].dendritic.zero_dendrite_validation_accuracy` | PAI's minimum-parameter arch row — **within-search only** |
| `candidates[0].dendritic.pai_deployed_params` | param count of PAI's best-scoring architecture |
| `candidates[0].dendritic.deployed_params` | param count profiled on the exported clean graph |
| `candidates[0].comparison.validation_accuracy_gain` | gain vs the prune/fine-tune baseline (basis field states it) |
