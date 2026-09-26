# KWS_Model: Dendritic SparkNet — Project Findings

*Compiled 2026-09-23 from `README.md`, `notes/`, `outputs/`, `reports/` and the
experiment logs. All accuracies are Google Speech Commands v2, 12-class
(10 keywords + `_unknown_` + `_silence_`), MFCC-32 front end, test n = 4,890
unless marked "val". Model selection is always on validation; test is read once.*

Sections are self-contained so they can be forwarded separately.

---

## 1. Summary

- We ported **SparkNet** (Svirsky et al., 2024) faithfully and trained a
  5-seed Pareto frontier at widths C2–C18. Our C16 replication averages
  **94.81% test (best seed 95.07%)** against the paper's **95.7 ± 0.17%**. We
  could not close that ~0.9 pp gap.
- Using PerforatedAI dendrites on top of that frontier, we produced three
  deployable models that **beat our faithful-replication frontier at equal
  parameter count**. Each was exported to int8 for the RP2040, and the int8
  output matches the float model bit for bit:

  | Tier | Model | Params | Test (float / int8) | Margin over our frontier at equal params |
  |---|---|---:|---|---|
  | 85 | C6, gate 16, dendrite on blocks.2 (seed 1) | 1,624 | 87.18 / 87.04 | +2.77 pp (best-seed envelope) |
  | 90 | C10, gate 16, dendrite on blocks.2 (seed 2) | 2,584 | 92.33 / 92.21 | +0.28 pp |
  | 95 | C18, gate 16, dendrite on fc (seed 0) | 5,014 | 95.54 / 95.54 | +0.16 pp |

  For comparison with the *published* numbers: our C6g16 model (1,624 params)
  is +2.9 pp over published C4 (83.5%, 1,416 params), and our C10g16 model
  (2,584 params) is +0.07 pp over published C8 (92.1%, 2,292 params). Neither
  comparison is at equal parameters. Our best C18 model (5,176 params, 95.56%
  int8) is still 0.14 pp below published C16.
- **We cannot honestly credit most of that margin to the dendrites.** Paired
  against same-seed, same-architecture scratch runs, the dendrite added
  **+0.1 to +0.2 pp on average**. That is real but **no cheaper than simply
  widening the network**. Most of the frontier margin came from shrinking the
  classifier head (gate channels 32 → 16 or 8). The dendrites then roughly
  broke even with width. See §4 for the exact breakdown.
- The one placement that looked **better than width** was a single dendrite
  on `blocks.2.pointwise` (grouped with its BatchNorm) at the narrowest width
  we tested, **C4**. This is one seed only and has not been replicated.
- **We never managed to reproduce PerforatedAI's documented recipe
  (whole-network perforation, history switching, up to 3 dendrites) as a win
  here.** On our side it either lost accuracy or declined to add dendrites
  (§5.6).

---

## 2. Baseline and replication

**Published SparkNet (paper, SC-v2 test, THOP MACs)**

| Model | Params | MACs | Test |
|---|---:|---:|---:|
| C4 | 1,416 | 105 K | 83.5 ± 0.60 |
| C8 | 2,292 | 190 K | 92.1 ± 0.33 |
| C16 | 4,636 | 454.5 K | 95.7 ± 0.17 |
| C32 | 11,500 | 1.2 M | 97.0 ± 0.18 |

**Our faithful replication (gate channels G = 32, 5 seeds each, test, val-selected checkpoint)**

| Width | Params | Test mean | Test best seed |
|---|---:|---:|---:|
| C4 | 1,504 | 82.25 | 83.03 |
| C6 | 1,906 | 86.88 | 87.67 |
| C8 | 2,356 | 90.80 | 91.31 |
| C10 | 2,854 | 92.51 | 92.92 |
| C12 | 3,400 | 93.73 | 93.93 |
| C16 | 4,636 | 94.69 | 95.01 |
| C18 | 5,326 | 95.31 | 95.69 |

Parameters follow 6C² + 141C + 844 at G = 32. The dedicated C16 replication
directory (`outputs/sparknet-paper-replication`, fixed-seed test report)
gives 95.07 / 94.52 / 94.79 / 94.68 / 94.97, mean **94.81**.

**Replication gaps and failures**
- **C16 is about 0.9 pp short of the paper.** The likely causes are
  augmentation RNG streams and dataloader differences. We did not confirm
  either.
- **Parameter counts differ at low width.** At C16 our graph matches the
  paper exactly under the pinned THOP counter (4,636 params, 454,480 MACs).
  At C8 we count 2,356 params against the paper's 2,292, and at C4 1,504
  against 1,416. The cause is unexplained. It is not the MFCC bin count.
- **Our MAC counter differs from the paper's.** Our hook counter does not
  see PAI's wrapper skip weights (16 ParameterList weights and 1,616
  elementwise MACs on one C8 model). MAC plots therefore mix counting
  conventions unless labelled.
- **The released C16 checkpoint scored 93.35% on our test pipeline.** In an
  earlier phase-A evaluation it scored 84.54% overall because of a
  synthesized-silence mismatch (silence false-accept rate 0.571; 91.49%
  with silence excluded).

---

## 3. What worked: the deployed models

| Tier | Model | Params | Float test | Int8 test | Paired dendrite gain vs same-seed scratch | Where the margin actually came from |
|---|---|---:|---:|---:|---|---|
| 85 | c6g16 + b2 dendrite, s1 | 1,624 | 87.18 | 87.04 | +0.10 pp | gate cut and a lucky seed |
| 90 | c10g16 + b2 dendrite, s2 | 2,584 | 92.33 | 92.21 | +0.10 pp | gate cut |
| 95 | c18g16 + fc dendrite, s0 | 5,014 | 95.54 | 95.54 | +0.20 pp (seeds 1–2: −0.43) | seed 0; sham got 95.40 |

"b2" means one dendrite on `blocks.2.pointwise`, grouped with its BatchNorm
into a `PAISequential`. Seeds tried: c6g16+b2 used 5 seeds
(85.69 / 87.18 / 86.40 / 86.89 / 86.32, all above 85). c10g16+b2 and
c18g16 used 3 seeds each.

**Breakdown of the 85-tier margin** (+3.55 pp over the G32 mean chord at
1,624 params; chord slope 11.52 pp per 1,000 params):
- Gate G32 → G16: −1.03 pp accuracy, but 336 fewer params, which widening
  would have charged 3.87 pp for. Net **+2.84 pp**.
- Seed 1 is +1.23 pp above the c6g16 scratch mean (84.66 / 87.08 / 85.81).
- The b2 dendrite: +0.10 pp for 54 params, while those 54 params spent on
  width buy about 0.62 pp. Net **−0.52 pp**.

**RP2040 deployment**
- The pipeline folds BatchNorm, uses int8 weights with 8- or 16-bit
  activations, and produces a C host reference and UF2 firmware.
- Every export is bit-exact against the C host (4,890/4,890 clips), and
  int8 accuracy is at or above float.
- Each firmware image is about 154–159 kB of flash and uses about 25 kB of
  static RAM.
- **Not yet run on real hardware.** We have no on-device latency or power
  measurement.

---

## 4. The central dendrite result: roughly the same value as extra width

**Replicated single-block placement** (g16 bases, C8 / C10 / C12 × seeds 0–2):

| Placement | Params per dendrite | Test gain vs no-dendrite control | Net of equal-param widening | Val gain | Val net of widening |
|---|---|---|---|---|---|
| blocks.1 pointwise | C² + 3C | **+0.12 ± 0.09** | −0.20 ± 0.09 | +0.16 ± 0.08 | −0.14 ± 0.08 |
| blocks.0 (stem pointwise, 32 → C) | 35C | +0.14 ± 0.07 | −0.76 ± 0.15 | +0.25 ± 0.08 | −0.60 ± 0.07 |

**Single-seed sweep across all four blocks** (C2–C12 means, seed 0):

| Placement | Raw test gain | Net of widening |
|---|---|---|
| b0 | +0.69 | −1.13 |
| b1 | +0.23 | −0.19 (only block with val gain ≥ 0 at all six widths) |
| b2 | −0.05 | −0.47 |
| b3 | −0.01 | −0.43 |

Pairs such as b23 at C6 / C8 / C10 came out −0.28 to −1.30 pp against
widening.

**Sham controls.** In a sham run the dendrite is added and then zeroed
right after the switch. Shams lost −0.39 pp (c4g16) and −0.25 pp (c10g16).
So **the switch itself costs about 0.3 pp**, and every raw gain above
understates the dendrite by roughly that amount.

**Per-seed scatter** is ±0.33 pp. Any single-seed placement claim under
about 0.5 pp is noise.

**Where dendrites came closest to paying for themselves** (seed 0 only,
Δ on last-40-epoch mean ÷ the gain that equal-param widening would give):

| Width | blocks.2 only | blocks.1–3 |
|---|---|---|
| C12 | ~40% | — |
| C8 | 0% | ~55% |
| C4 | **~120%** (Δlast40 +0.42, Δlast5 +0.63; beat its sham every one of the last 40 epochs, mean +0.52) | ~30% |

Gains were **not additive** across blocks. At C4 one block already gave
the whole ~0.4–0.6 pp. At C8 one block gave 0 and three blocks gave +0.5.

**Trend across the pruning runs.** In the C16 → C12 / C10 / C8 prune +
fc-dendrite runs (§5.5), the dendrite's increment over its own
zero-dendrite continuation grew as the base got more starved:
C12 ≈ +0.23 pp, C10 ≈ +0.44 pp, C8 ≈ +0.88 pp (val, 5-seed means).

---

## 5. Failures and negative results

### 5.1 First placement study (v2), 150 arms: all null, and confounded
- **Design.** Arms were control, pointwise (blocks.2/3), fc, gate_conv and
  depthwise (blocks.2/3), at C2–C12 × 5 seeds. Each arm was a
  scratch → 40-epoch identity fine-tune → PAI run with 1 dendrite.
- **Result.** Control was the best arm at every width. All four dendrite
  arms trailed it by 0.2–0.5 pp (about 2.4 pp at C2), and at each width
  the four placements landed within 0.01–0.04 pp of each other.
- **Root cause: a broken handoff.** The fine-tune stage silently changed
  the training recipe: SGD → AdamW, a learning-rate restart, weight decay
  1e-3 → 1e-4, cosine instead of polynomial-hold, and **label smoothing
  0 → 0.1**. That alone cost −0.62 pp on average (negative in 28/30
  cells, −2.40 pp at C2). The loss was 2.6× larger than any dendrite effect
  and hid it.
- **PAI behaviour in this study.** 117 of 120 dendritic cells integrated
  exactly one dendrite and 3 integrated none. There were 96
  `noImprove_lr` retry markers, and 96 of 120 cells ended resume as
  `no_improvement`.
- **PB scores.** Only fc cleared the >0.02 rubric (about 0.13 in all 30
  cells). Pointwise scored 0.006–0.008, gate_conv about 0.0095 and
  depthwise 0.006–0.007. **Even so, fc still lost to control.** A high PB
  score did not translate into accuracy.
- **Missing controls.** No matched zero-dendrite arm with an equal epoch
  budget existed in v2. Dendrite arms were never test-evaluated.

### 5.2 Grow-dendrites pilots at C8: all inside the sham noise band
All six arms (fc, fc switch at 170, fc with dendrite wd 1e-3, pointwise,
pointwise with input ×75, fc-sham) landed within |Δ| ≤ 0.18 pp of paired
scratch.

- **fc (switch 120).** The dendrite is used (switching it off drops
  accuracy about 2 pp), but it *substitutes* for the base fc rather than
  adding to it. It is mostly linear, with R² 0.86–0.89 against its
  pre-activation.
- **fc switched at epoch 170.** The learning rate at the switch was
  0.00074 (versus 0.0053 at epoch 120). Skip weights stayed around 0.07
  and the dendrite was inert.
- **Pointwise before we fixed it.** The dendrite was attached to the
  *pre-BN* pointwise output. That output's input has std about 54
  (blocks.2) and about 104 (blocks.3), versus 0.16 at fc. The tanh
  saturated: 13% / 46% of |z| > 2 at the 0.01 init, and 90% / 97.5% after
  candidate training.
- **Pointwise with input rescaled (fixed ×75, or auto-calibrated at
  C12).** Rescaling removed the saturation, but the dendrite went
  **inert**: output std 0.00024× the base's, off-drop 0.00 pp, PB score
  flat at 0.046. The cause: PAI adds the dendrite *before* the block's BN,
  and BN divides by σ ≈ 100. That gives the skip weights an effective
  learning rate of about lr/σ², roughly 1e-4 of normal.
- **Fix.** We grouped `pointwise + bn` into `PAISequential`, as PAI's
  docstring advises. That cured the inertness: off-drop 3.9–4.5 pp, skip
  |w| 1.60 versus 0.039. The dendrite still mostly substituted at C12 and
  C8 (Δlast40 +0.06 / 0.00). The grouped placement only paid off at C4.

### 5.3 Deployed-model dendrite diagnostics (c6g16+b2 s1)
Zeroing the dendrite after training dropped val from 87.87 to 75.43. **This
12 pp is co-adaptation, not gain.** The base learned to rely on the
dendrite, while the paired scratch control reached nearly the same
accuracy without one.

Other diagnostics: skip |w| 2.53, PB score 0.093 (frozen from candidate
epoch 123 on), linear R² 0.63 against the pre-activation, tanh saturated
fraction 0.43. The dendrite is mostly a linear rescale plus clipping.

### 5.4 Knowledge distillation: never helped
- **Teacher weaker than student.** With a DS-CNN-L MFCC-32 teacher at
  91.76% val, KD scored −1.37 pp against no-KD (C12).
- **Stronger teacher, still nothing.** With a DS-CNN-L log-mel teacher at
  97.69% test, KD exactly tied no-KD at 92.15% (one seed). Another
  comparison gave −0.10 pp.
- **Ensemble KD hurt.** A C16×5 ensemble teacher at α = 0.9, T = 4 gave
  C4 79.04% (underfit) and C14 94.17%.
- **Integration bugs.** A `task_loss_scale: 100` guard blocked KD entirely
  in the dendrite path. The sparsity term sat outside the KD mix. KD
  annealing never called `set_epoch()` during PAI phases, so it was
  silently inert.

### 5.5 Prune-then-perforate (C16 → C12 / C10 / C8, fc dendrites, up to 3, 5 seeds)
Dendrites recovered a lot of the pruning damage: val went from 90.2 to
92.6 at C12 and from 79.9 to 88.9 at C8 (seed 0). Most of that recovery
also appears in the **zero-dendrite continuation**, though, and **every
final model is below a scratch model of the same width with fewer
params**:
- C12: 92.3–93.1 val at 3,808–4,228 params, versus scratch C12 93.88 at
  3,400.
- C10: 90.6–91.2 at 2,854–4,114 params, versus scratch 92.80 at 2,854.
- C8: 87.1–88.9 at 2,764–3,616 params, versus scratch 91.23 at 2,356.

The companion gate_conv prune arm (5 seeds) and the C12 → C10 / C8 / C6 fc
run never finished. An "unlimited dendrites" run failed on a manifest
conflict after an interrupt.

### 5.6 PerforatedAI as documented: no gain, and the cost is prohibitive
- **Setup.** We used PAI's native loop: history switching with PAI 3.2.8
  defaults (lookback 1, 10 epochs to switch, threshold 1e-5), up to 3
  dendrites, perforating the whole supported surface (blocks.0–3,
  gate_conv, fc). The control was a matched zero-dendrite run with an
  equal epoch budget.
- **Cost problem.** One dendrite copies each wrapped module, so each
  dendrite costs about a whole extra network (C8: 2,356 → 4,724 params per
  dendrite).

  | Width | PAI val | PAI params | Matched control val | Control params |
  |---|---:|---:|---:|---:|
  | C4 | 82.16 | 3,004 (1 dendrite) | 82.32 | 1,504 |
  | C8 | 90.98 | 7,168 (3 dendrites), 551 K MACs | 91.25 | 2,356 |
  | C12 | 94.24 | 3,400 (PAI added no dendrite) | 94.53 | 3,400 |

- **Also:** for C4 and C8 the val-selected PAI accuracy equals its own
  zero-dendrite accuracy, i.e. the added dendrites did not raise val.
- **Unfinished.** The C6, C10 and C16 PAI runs never finished. Stage 2 was
  planned but not run: keep only modules with PB > 0.02, perforate nearest
  the output, shrink the base first.
- **A bug we found along the way.** Our per-phase cosine learning-rate
  schedule annealed to exactly 0 at phase-epoch 30. That guaranteed a
  plateau, so "history" switching was in effect a fixed, forced switch.
  It was fixed before this batch (LR horizon decoupled to 120 epochs), but
  earlier PAI runs are affected.

### 5.7 DS-CNN track (earlier, superseded)
Note: the DS-CNN rows use the **40-bin log-mel** front end unless marked
MFCC-32, so they are not directly comparable with the SparkNet numbers.

**Baselines (test unless marked val)**

| Model | Params | Front end | Accuracy | FAR / FRR |
|---|---:|---|---:|---|
| DS-CNN-L teacher (200 ep) | 469,604 | log-mel | **97.69** | 1.90 / 2.36 |
| DS-CNN-L teacher retrain | 469,604 | MFCC-32 | 91.76 val (train ≈ 100%, overfit) | — |
| DS-CNN-M, early 15-ep sanity run | 146,902 | log-mel, no aug | 96.52 | 2.52 / 4.19 |
| DS-CNN-L, early 15-ep sanity run | 467,942 | log-mel, no aug | 97.27 | 1.45 / 3.76 |
| DS-CNN-XS student (warm-distilled) | 4,096 | log-mel | **83.64** (85.11 val) | 18.53 / 15.29 |

**Pruning + fc dendrite (val only, no held-out test, no same-cost control)**

| Run | Pruned base | + 1 fc dendrite | Params | MACs |
|---|---:|---:|---|---:|
| Legacy compression run, w18 `[18,18]` | 65.21 | 70.16 | 1,830 → 2,070 | 1.45 M |
| Legacy compression run, w14 `[14,14]` | 64.69 | 66.44 | 1,522 → 1,714 | 1.21 M |
| Full pipeline 2026-09-12, w18 | 82.58 | 82.68 (resume KD 82.53) | 1,830 → 2,070 | 1.45 M |

- The legacy +4.95 / +1.75 pp are measured against the separately trained
  pruned row. Within PAI's own architecture search the dendrite added only
  +1.46 (w18, 68.70 → 70.16) and +0.31 (w14, 66.13 → 66.44), so most of
  the headline delta came from continued training on a weak recipe. The later, better recipe shrank the dendrite effect to
  +0.10 pp. fc PB scores peaked around 0.13–0.15, the same pattern as
  SparkNet: fc correlates well but adds little.
- Wider placements (late blocks + fc, all blocks + fc) were rejected by the
  1.5 M-MAC budget and never ran. w10 / w6 candidates never completed.
- **Full pipeline run on 2026-09-12 failed.** Causes: missing PAI native
  checkpoint, an RNG-state `TypeError`, an `IndexError`, a resume-recipe
  mismatch, and a cleanup error (`'fc' has no two-branch layer_array`). The
  w17 PAI run was interrupted at epoch 3. Downstream cluster / quantize /
  benchmark stages are therefore not valid.
- **Structured N:M sparsity** could not be applied to the DS-CNN-XS
  depthwise, stem and first-pointwise layers (too few weights), so those
  stayed dense. N:M is mask-only and gives no MCU MAC savings anyway.
- **Why it was dropped.** SparkNet C12 (3,400 params, 93.73% test) dominates
  DS-CNN-XS (4,096 params, 83.64% test) at fewer params and far fewer MACs
  (XS ≈ 2.7 M MFCC MACs), so SparkNet became the main path.

### 5.8 Other incomplete items
- **Gate-width ladder.** The no-dendrite c9g8 is the cheapest ≥ 90 model:
  2,023 params, 90.16%, +1.55 pp over the envelope. It is one seed and has
  not been replicated or re-exported.
- **Dendrite placements never tested alone:** gate_conv and depthwise
  outside the confounded v2 study, and b0 / b1 above C12.
- **ReRAM / NeuroSim V2.1 wrapper.** The code and unit tests exist, but
  the simulator source is not installed, so there are no hardware
  numbers.
- **Test suite.** 17 pre-existing test failures remain in one PAI test
  file.

---

## 6. Why did dendrites work better on some layers than others? Our current reading

These are hypotheses from the diagnostics. We would value correction.

1. **Where the dendrite sits relative to BatchNorm matters most.** On a bare
   pointwise conv, PAI adds the dendrite before BN. The pre-BN signal has
   σ ≈ 50–100, so the dendrite either saturates its tanh or gets its
   learning rate divided by about σ² and goes inert. Grouping conv + BN is
   what made pointwise dendrites train at all. fc has no following BN and
   its input σ is about 0.16, which is likely why fc dendrites always
   trained and always had the highest PB scores.
2. **A high PB score did not mean accuracy gain.** fc correlated best
   (about 0.13), but its dendrite mostly substituted for the base fc
   weights: linear R² about 0.86–0.89, with co-adaptation rather than
   added capacity.
3. **Early blocks gave the largest raw gain but cost the most.** b0 (the
   stem, 32 → C) cost 35C params per dendrite. Its val gain exceeded its
   test gain at nearly every width, which suggests it overfits. b1
   (C² + 3C) was the only block with val gain ≥ 0 at every width.
4. **Width dependence.** Dendrites paid for themselves only where the base
   was capacity-starved: C4, and the heavily pruned C8. At C8–C18 the
   network appears to have enough capacity that one dendrite duplicates
   what width would give. At C2 everything hurt, though C2 is also where
   our fine-tune handoff did the most damage.
5. **Depthwise dendrites** can only form per-channel temporal filters, with
   no cross-channel mixing. That fits their lowest PB scores
   (0.006–0.007).

---

## 7. Questions for PerforatedAI

1. **Why was one layer so much more effective than the others?** In our
   hands, pointwise conv + BN at blocks.1 or blocks.2 behaved very
   differently from fc, gate_conv or depthwise, and b1 was the only block
   whose dendrite never hurt val. Is there a principled way to predict the
   best module from PB scores, activation statistics, or position?
2. **The BatchNorm question.** Is grouping conv + BN into `PAISequential`
   the intended fix for pre-BN dendrites? Would you also rescale the
   dendrite output per channel, or change `candidate_weight_initialization_multiplier`
   for high-variance inputs?
3. **Substitution versus addition.** Our dendrites often co-adapt: 12 pp
   off-drop yet only +0.1 pp net gain. Is that expected, and is there a
   training setting (base freeze, LR multiplier, weight decay on skip
   weights) that pushes them toward adding capacity instead?
4. **Very small models.** At 1.5 k–5 k params, the documented
   whole-network placement costs about one extra network per dendrite.
   What placement and dendrite budget would you recommend for sub-5 k
   param keyword-spotting models? Is "shrink then perforate" meant for
   this regime?
5. **Activation choice.** We used tanh. PAI's default is sigmoid. Would
   either change the saturation or substitution behaviour above?
6. **Switching schedule.** With history switching, how should the learning
   rate schedule interact with plateau detection? We found that an
   annealed LR can force the switch.
7. **Replicating our paper gap.** Do you have experience with SparkNet or
   similar TCS-block models, or a sense of whether dendrites might close
   the 0.9 pp gap between our C16 replication and the paper?

---

## Sources
- Plots and aggregates: `outputs/plots/sparknet-dendritic-comparison/`
  (`model_stats.md`, `*_summary.csv`)
- v2 placement study: `outputs/sparknet-dendritic-study-v2/`, and in
  `notes/dendrite-study-v2/`: `agent-repo-results-audit.md`,
  `ENHANCEMENTS.md`, `INTEGRATION_AUDIT.md`
- Grow-dendrites pilots and sweeps: `outputs/sparknet-grow-dendrites-v3/`
  (`report_c8_seed0_pilots.md`, `**/reports/grow_summary.yaml`)
- PAI as documented: `outputs/sparknet-pai-documented/{pai,control}/`
- Prune + fc dendrites: `outputs/sparknet-c16-dendritic-prune-no-kd-*`,
  `outputs/sparknet-c12-dendritic-prune-no-kd-*`
- DS-CNN: `reports/ds_cnn_l_teacher_current.json`, `reports/phase_a/xs_{val,test}.json`,
  `outputs/compression-run/`, `outputs/full-run-20260912T063022Z/`,
  `notes/dendrite-study-v2/agent-dscnn-results-audit.md`
- Replication: `outputs/sparknet-paper-replication/`,
  `notes/dendrite-study-v2/sparknet-paper-macs-audit.md`
- RP2040: `outputs/rp2040/*/reports/rp2040.yaml`,
  `notes/dendrite-study-v2/PICO_COMPRESSION_PIPELINE_JOURNAL.md`
- PAI mechanics: `notes/dendrite-study-v2/PAI_KNOWLEDGE.md`
