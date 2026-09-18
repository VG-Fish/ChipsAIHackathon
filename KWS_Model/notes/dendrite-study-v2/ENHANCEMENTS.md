# SparkNet + dendrite: performance enhancements and the KD verdict

**Written:** 2026-09-17. **Base:** `/Users/vishy/Desktop/ChipsAIHackathon/KWS_Model` (BASE below).
**Scope:** how to make SparkNet+dendrite runs perform better, KD first.
**Constraint honoured:** nothing was launched, killed, or modified. The study-v2 run
was read only. Every config change below is a *proposed diff*, not an applied one.

All paths are relative to BASE. All `file:line` citations are against the tree at
HEAD `83ed601`.

---

## 0. Executive summary

1. **KD is not broken any more, and it is not helping.** Three independent
   comparisons put KD at **0.00 pp, −0.10 pp and −1.37 pp** against a matched
   supervised control. The one large loss is explained by a teacher that is
   *worse than its own student*. Recommendation: **keep `--no-KD` for the
   dendrite study.** Confidence: high for "do not use KD as currently
   configured"; medium for "KD cannot help SparkNet at all".
2. **KD is currently unreachable in the SparkNet dendrite path** — three
   separate guards forbid it (§2.1). Any plan that assumes it can be switched on
   with a flag is wrong.
3. **The biggest accuracy lever in this repo is not KD and not dendrites.** The
   40-epoch "identity fine-tune" that every arm runs before PAI *destroys*
   accuracy in **26 of 28** completed runs, averaging **−0.55 pp** and reaching
   **−2.40 pp at C2** (§3.1). The dendrite phase then loses a further −0.21 pp.
   Net, the current pipeline delivers **−0.76 pp versus the checkpoint it started
   from**, while adding parameters.
4. Fixing that handoff is a **config-level change with zero extra compute**, and
   it is worth roughly **5× the entire measured dendrite effect**.

---

## 1. Question 1 — does KD help or hurt here?

### 1.1 The teachers

| Teacher | Checkpoint / run | Accuracy | Params |
| --- | --- | ---: | ---: |
| DS-CNN-L, 40-bin log-mel | `models/checkpoints/ds_cnn_l_12class.pt` | **97.69 %** test | 469,604 |
| DS-CNN-L, MFCC-32 | `outputs/phase_b/teacher_ds_cnn_l_mfcc32_nw0` | **91.76 %** val | 469,604 |

The 40-mel teacher's numbers are in `reports/ds_cnn_l_teacher_current.json`
(`accuracy: 0.9768583450210379`, `num_params: 469604`).

The MFCC-32 teacher's number is in
`outputs/phase_b/teacher_ds_cnn_l_mfcc32_nw0/metrics/summaries.yaml`
(`best_val_acc: 0.9176470588235294`), from 200 epochs costing 421 min
(126.4 s/epoch, measured from that run's `metrics/teacher/ds_cnn_l.jsonl`).

**This is the single most important fact in the KD story.** A 469 K-parameter
DS-CNN-L that scores 91.76 % is broken — it is *below* the 3,400-parameter
SparkNet C12 student it was built to teach (92.61 %, §1.2). Distilling from it
is distilling from a worse peer. That it exists at all is a consequence of
`distill.py:53-69` (`validate_data_feature_shape`), which correctly refuses a
teacher/student frontend mismatch and therefore *forced* a retrain when the
project moved to the MFCC-32 frontend. The retrain was never checked against
the 40-mel teacher's quality.

### 1.2 Every valid KD vs no-KD comparison in `outputs/`

There are exactly three. All are SparkNet C12, 200 epochs, `light`-family
AdamW recipe.

| # | Control (no KD) | KD arm | Control | KD | Δ | Init matched? |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | `phase_b/sparknet_c12_speech_commands_v2_mfcc32_g32` | `phase_b/sparknet_c12_mfcc32_g32_kd_annealed` | 92.613 % | 91.244 % | **−1.37 pp** | **Yes** |
| B | `phase_b/step2_sparknet_c12_light_20260913T215424Z` | `phase_b/step2_sparknet_c12_light_kd_t2_20260914T013331Z` | 92.150 % | 92.150 % | **0.00 pp** | No |
| C | same control as B | superseded T=1 run (deleted) | 92.15 % | 92.05 % | −0.10 pp | No |

Sources: each run's `metrics/summaries.yaml` (`best_val_acc`). Comparison C is
quoted from the recovered `KD_DIAGNOSIS.md` (`git show e0ecadd:KWS_Model/KD_DIAGNOSIS.md`),
"Limits and provenance": *"The completed no-KD baseline peaked at 92.15%. The old
KD run was stopped after 169 epochs, peaking at 92.05% at epoch 160."*

**Comparison A is the cleanest experiment in the repo and it is also the most
confounded.** Clean, because the annealed recipe starts at `response_weight: 0.0`
(`configs/train/light_kd_annealed.yaml:37`) so epoch 1 is *pure* supervised loss —
and indeed both runs record `val_acc = 0.0897` at epoch 1 and track to within
0.003 through epoch 5. The initialisation and data order are demonstrably
identical. Confounded, because the teacher is the 91.76 % MFCC-32 one. Its
in-training diagnostics confirm it:
`sparknet_c12_mfcc32_g32_kd_annealed/metrics/student/distill.jsonl` records
`train_teacher_accuracy ≈ 0.914`, `train_teacher_confidence ≈ 0.806`,
`train_teacher_entropy_nats ≈ 0.787` at every epoch.

**So A measures "distilling from a peer hurts by 1.37 pp". It does not measure KD.**

**Comparison B is the one that actually bears on KD.** Its teacher is the strong
one: the same JSONL fields read `train_teacher_accuracy ≈ 0.9862`,
`train_teacher_confidence ≈ 0.8957`, `train_teacher_entropy_nats ≈ 0.514` — a
genuinely strong, genuinely soft teacher on the student's own augmented views.
The KD gradient is well scaled: `train_kd_logit_grad_norm_ratio ≈ 1.15`
throughout, i.e. the weighted KD logit-gradient is the same size as the CE one.
And the outcome is an **exact tie**: both runs peak at
`0.9215043394406943` (= 4777/5184 correct), the control at epoch 127 and the
KD run at epoch 117.

The tie is not a bug. It is two runs landing on the same integer correct-count.
Their initialisations are *not* matched (epoch 1: 0.0779 vs 0.1431), which is the
RNG-divergence caveat `KD_DIAGNOSIS.md` already flagged, so B is a one-seed
comparison with ±0.4 pp of noise. It cannot prove KD is exactly neutral. It does
show KD is not worth a detectable amount.

### 1.3 What the prior diagnosis concluded

`KD_DIAGNOSIS.md` (recovered from `e0ecadd`) is unambiguous, and its conclusion
holds up against the newer runs:

- **Not broken.** *"The code's KL direction, batch reduction, and temperature-squared
  scaling are correct; the weakness is in the recipe, not a reversed KL or
  detached student gradient."* *"The teacher stays frozen in evaluation mode,
  receives exactly the same input tensor as the student, and has the same feature
  shape and class ordering."*
- **Was mis-weighted, and was then fixed.** The original T=1 / response-weight-0.1
  recipe gave a KD gradient only ~15 % the size of CE with cosine ≈ 0.90 — small
  and redundant. It was replaced with T=2 / 0.5 / 0.5.
- **The fix did not buy accuracy.** That is comparison B: ratio 1.15, cosine
  0.72–0.96, result 0.00 pp.

So the arc is: *mis-weighted → correctly weighted → still unhelpful.* The
remaining diagnosis is **redundancy**: even at T=2, the epoch-mean
`train_kd_logit_grad_cosine` stays at **0.72–0.96** (run B). The teacher is
telling a 3,400-parameter student almost exactly what the labels already tell it.
There is little dark knowledge left to transfer once the student is this far
below the teacher's capacity.

### 1.4 Recommendation and confidence

**Recommendation: keep `--no-KD`.** Do not spend study budget on KD.

- Confidence **high** (three comparisons, none positive, best case an exact tie)
  that *KD as currently configured* does not help SparkNet at C12.
- Confidence **medium** that KD cannot help at all. Two real gaps remain: only
  one seed per comparison, and no KD run has ever used a strong teacher *on the
  MFCC-32 frontend the study actually uses*.

### 1.5 The single experiment that would settle it

Not a KD run — a **teacher repair**, because the study's frontend has no good
teacher. In order:

1. Train a DS-CNN-L on `configs/data/speech_commands_v2_mfcc32_paper.yaml` with
   the *teacher* recipe (`configs/train/full_mfcc_teacher.yaml`), not the light
   AdamW one, and require ≥ 96 % val before using it. Cost ≈ **7 h**
   (measured: 126.4 s/epoch × 200).
2. Only then: SparkNet C8 (the width with the most headroom that is still stable),
   **5 seeds**, KD vs no-KD, matched init, 200 epochs, no PAI. Cost ≈
   **10 × 10.2 min ≈ 1.7 h** (§5 timing basis).

If step 1 cannot clear 96 %, KD is moot for this project and the question is
closed. **I would not fund step 1 before the levers in §3.** A 7 h teacher retrain
buys, at best, the ~0 pp that comparison B already measured; the §3.1 fix buys
+0.55 pp for free.

---

## 2. Question 2 — what is wrong with the current KD recipe

Answering each sub-question, then the defects.

| Sub-question | Finding | Evidence |
| --- | --- | --- |
| Temperature | T=2. Correct, and already the *fixed* value. | `configs/train/light_kd.yaml:34` |
| Alpha / loss weighting | 0.5 response / 0.5 classification / 0.0 feature. Convex mix enforced. Gradient ratio ≈ 1.15 — well balanced. | `kd.py:52-63`, run B JSONL |
| Teacher in eval mode? | **Yes, unconditionally.** `train()` is overridden to force `False`. | `kd.py:353-356` |
| Same augmented view? | **Yes.** The student's exact `features` tensor is passed to the teacher. | `train.py:487` → `kd.py:501` |
| Interacts with `task_loss_scale: 100`? | **Fatally — it raises.** See §2.1. | `train.py:319-320`, `dendritic.py:2349-2350` |
| Applied in `p` phases as well as `n`? | **Yes, ungated.** No phase check anywhere in the PAI step. | `dendritic.py:2474-2489` |
| Teacher good enough? | 40-mel: yes (97.69 %). MFCC-32: **no** (91.76 %). | `reports/ds_cnn_l_teacher_current.json`; §1.1 |

### 2.1 Defect 1 (blocker) — KD is unreachable in the SparkNet dendrite path

Three independent guards:

```
src/kws/train.py:319-320
    if kd is not None and task_loss_scale != 1.0:
        raise ValueError("task_loss_scale is not supported with knowledge distillation")

src/kws/optimize/dendritic.py:2349-2350
    (identical guard)

src/kws/optimize/sparknet_dendritic_prune_experiment.py:78-79
    if cfg.get("teacher_checkpoint") is not None:
        raise ValueError("SparkNet experiment is no-KD; teacher_checkpoint must be null")
```

Every SparkNet arm config sets `task_loss_scale: 100.0`
(`configs/train/sparknet_c16_dendritic_prune_no_kd.yaml:55` and the four sibling
arm configs). So enabling KD today raises before the first batch. `--no-KD` is
not a choice the study made; it is the only reachable state.

### 2.2 Defect 2 (latent, severe) — the guard checks the wrong knob

The guard checks `task_loss_scale` but **not `sparsity_weight`**, and the
auxiliary gate-sparsity term is added *after* the KD mix, outside it:

```
src/kws/train.py:486-491
    losses = kd(features, logits, labels, student_features)
for name, (value, weight) in collect_auxiliary_losses(model).items():
    ...
    losses["total"] = losses["total"] + weight * value
```

With KD on, `losses["total"]` is a **convex** mix (weights sum to 1,
`kd.py:62-63`), so it is on the scale of one CE, ≈ 1–2.5. The SparkNet paper
model configs carry `sparsity_weight: 1.0`
(`configs/model/sparknet_c12_paper.yaml:23`, and every other `*_paper.yaml`).
Result: a **1:1** task-to-sparsity ratio instead of the intended **100:1** —
precisely the failure the arm config warns about in prose:

> `configs/train/sparknet_c16_dendritic_prune_no_kd.yaml:13`
> *"Omitting it here would fine-tune at 1:1 and crush the learned gates."*

So the obvious "fix" for defect 1 — set `task_loss_scale: 1` so KD is allowed —
silently walks into exactly the gate collapse the comment predicts, and nothing
errors. **Any KD-enabled SparkNet recipe must pair `task_loss_scale: 1` with a
model config carrying `sparsity_weight: 0.01`** (the ratio
`configs/model/sparknet_c12.yaml:11` already uses). The guard should be widened
to check the effective ratio, not one of its two factors.

### 2.3 Defect 3 — KD annealing is silently inert during PAI

`train.py:460-464` calls `kd.set_epoch(epoch)` every epoch, which is what makes
`KDAnnealing` (`kd.py:103-199`) do anything. **`dendritic.py` never calls it.**
`grep -n set_epoch src/kws/optimize/dendritic.py` returns nothing.

Consequence: `distillation.anneal` is honoured in stage-2 distillation and
ignored in every PAI phase, where the criterion keeps `base_weights` forever
(`kd.py:406-417`). A recipe that reads as "ramp KD in over 40 epochs" would apply
full-strength KD from the first dendrite batch. Nothing warns.

### 2.4 Defect 4 — KD changes *which* dendrites get selected, not just how they train

`dendritic.py:2474-2489` applies the same criterion in `n` and `p` phases with no
gating. In `p` phase PAI scores candidate dendrites by correlating them against
the backpropagated **neuron error**. Under KD that error is the KD-mixed error,
not cross-entropy. So KD does not sit on top of the dendrite search as a neutral
accuracy overlay — it changes the search objective itself.

*SPECULATION* on magnitude: unquantified, no run exists. But it means "add KD to
the dendrite study" is **not** a design where KD and dendrites are separable
factors, and a KD × placement matrix would not be a clean 2-factor design.

### 2.5 Defect 5 — feature KD silently degenerates under PAI

`kd.py:390-399`: when `student_feature_dim` is `None`, `without_features()`
renormalises the two surviving weights. `supports_pooled_features`
(`kd.py:553-575`) returns `False` for a PAI-wrapped model by design. So the
paper's three-loss recipe (Song et al.) becomes a two-loss recipe in exactly the
stage this study cares about. This is deliberate and well documented — logged at
`kd.py:392-399` — but it means "we use the Song et al. recipe" is not true of
the dendrite phases.

### 2.6 Not defects (checked and clean)

- KL direction, `batchmean` reduction, and `T²` rescaling are correct
  (`kd.py:240-245`).
- The teacher cannot leave eval mode (`kd.py:353-356`) — BN statistics cannot drift.
- Teacher construction is RNG-isolated via `torch.random.fork_rng`
  (`kd.py:328-333`) so a KD run and its control start from identical student
  weights. This is what makes comparison A trustworthy.
- Diagnostics are detached and add no backward pass (`kd.py:260-309`).

---

## 3. Question 3 — performance levers, ranked

Ranked by **expected accuracy gain per unit of engineering + compute**.
Baseline for "expected gain" is the completed study-v2 evidence (§3.1) and the
phase_b sweeps. Cost basis in §5.

Measured scratch frontier (paper recipe, 200 epochs, 5 seeds, val n = 4445):

| Width | Params | Val acc | sd |
| --- | ---: | ---: | ---: |
| C12 | 3,400 | 93.88 % | 0.46 |
| C10 | 2,854 | 92.80 % | 0.33 |
| C8 | 2,356 | 91.23 % | 0.32 |
| C6 | 1,906 | 87.87 % | 0.68 |
| C4 | 1,504 | 82.87 % | 0.86 |
| C2 | 1,150 | 63.34 % | 1.49 |

(from each `outputs/sparknet-dendritic-study-v2/scratch/c*-seed*/metrics/summaries.yaml`)

---

### L1 — Repair the scratch → arm fine-tune handoff ★★★★★

**The single highest-value change in this document.**

The arm pipeline's first step is a 40-epoch fine-tune with
`pruning.method: identity` — nothing is pruned, so it should be a no-op or a
small gain. Measured across all 28 completed pointwise runs:

| Width | scratch | after identity-FT | Δ FT | after dendrite | Δ dendrite | Δ vs scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C12 | 93.88 | 93.71 | **−0.16** | 93.48 | −0.24 | −0.40 |
| C10 | 92.80 | 92.62 | **−0.18** | 92.26 | −0.36 | −0.54 |
| C8 | 91.23 | 90.96 | **−0.27** | 90.73 | −0.23 | −0.49 |
| C6 | 87.87 | 87.56 | **−0.31** | 87.37 | −0.19 | −0.50 |
| C4 | 82.87 | 82.17 | **−0.69** | 82.05 | −0.12 | −0.81 |
| C2 | 63.04 | 60.64 | **−2.40** | 60.53 | −0.10 | −2.50 |

Aggregate: identity fine-tune **−0.546 pp mean, negative in 26/28 runs**
(sign test p < 1e-5). Dendrite phase **−0.214 pp mean, positive in 4/28**.
End-to-end **−0.760 pp vs the source checkpoint, positive in 1/28**.

Computed from the `source.validation_accuracy`,
`candidates[0].baseline.validation_accuracy` and
`candidates[0].dendritic.validation_accuracy` fields of each
`outputs/sparknet-dendritic-study-v2/arms/pointwise/c*-seed*/reports/sparknet_dendritic_prune_experiment.yaml`.

**Root cause: four unforced changes at the handoff.** The scratch checkpoint is
trained by `configs/train/sparknet_narrow_paper_fast_io.yaml`; the arm fine-tune
by `configs/train/sparknet_c16_dendritic_prune_no_kd.yaml`:

| Setting | Scratch (200 ep) | Arm FT (40 ep) | Cite |
| --- | --- | --- | --- |
| optimizer | `sgd`, momentum 0.9 | *unset* → **AdamW** | `..._fast_io.yaml:45-46`; `train.py:167` |
| lr | 0.01 (×100 scale → eff. 1.0) | 0.001 | `:47` vs `..._no_kd.yaml:51` |
| weight_decay | 0.001 | 0.0001 | `:48` vs `:52` |
| **label_smoothing** | **0.0** | **0.1** | `:49` vs `:53` |
| scheduler | `polynomial_hold` | *unset* → **cosine** | `:51` vs (absent) |
| warmup | 5 % of 200 ep | 5 % of 40 ep, **restarting at 1e-3** | `:52` vs `:54` |

The label-smoothing flip alone changes the objective the checkpoint converged
under. Combined with an LR restart to 1e-3 on a converged solution, the model is
knocked off its minimum and 40 AdamW epochs do not recover it. The damage scales
with how fragile the model is — hence −0.16 pp at C12 and −2.40 pp at C2.

**Proposed diff** (apply *identically* to all five arm configs, `control` included):

```diff
--- a/configs/train/sparknet_c16_dendritic_prune_no_kd.yaml
+++ b/configs/train/sparknet_c16_dendritic_prune_no_kd.yaml
-lr: 0.001
-weight_decay: 0.0001
-label_smoothing: 0.1
-warmup_fraction: 0.05
+# Continue the objective and optimizer family the source checkpoint converged
+# under. A converged SGD solution must not be restarted into AdamW at 1e-3.
+optimizer: sgd
+momentum: 0.9
+lr: 0.0005          # x100 task scale -> effective 0.05, ~1/20 of the scratch peak
+weight_decay: 0.001
+label_smoothing: 0.0
+warmup_fraction: 0.0
+scheduler: polynomial_hold
+hold_fraction: 0.0
+polynomial_power: 2.0
+min_lr: 0.000001
 task_loss_scale: 100.0     # UNCHANGED
 epochs: 40                 # UNCHANGED
 dendritic_schedule_epochs: 30   # UNCHANGED
 resume_epochs: 8           # UNCHANGED
 perforatedai: {...}        # UNCHANGED, every key
```

- **Expected gain:** +0.5 pp mean, +2.4 pp at C2. Upper bound is the measured loss.
- **Cost:** config-only, **zero extra compute** (same 40 epochs).
- **Risk:** low on accuracy. **High on study validity** — see §4; it must be
  applied to every arm or not at all.
- **Validate:** the cheap ablation in §5, 1.5 h. Success = `Δ FT ≥ 0` at every width.

**Caveat that limits L1 to the pre-PAI phase.** `dendritic.py:1947-1951` hardcodes
`torch.optim.AdamW` for the PAI phases and ignores `train_cfg["optimizer"]`;
`dendritic.py:1952-1957` calls `build_lr_scheduler` without a `name`, so it is
always cosine (`train.py:86`). So the diff above repairs the 40-epoch fine-tune —
which is where the −0.55 pp is measured — but the PAI phases will still run
AdamW + cosine at lr 1e-3. Fully closing the loop needs a small code change to
honour the config there. That is a separate, larger item.

---

### L2 — Tune the gate-sparsity weight ★★★★☆

**Is `task_loss_scale` doing what the recipe intends?** Yes, but only by accident
of the optimizer, and it does two *different* things in the two phases:

- **Under AdamW** (arm fine-tune and all PAI phases): Adam normalises by the RMS
  of the gradient, so multiplying the task loss by 100 leaves the task update
  unchanged up to `eps`. Its *only* effect is to set the task:sparsity ratio to
  100:1 — which is exactly the intent. `task_loss_scale: 100 + sparsity_weight: 1.0`
  is numerically equivalent to `task_loss_scale: 1 + sparsity_weight: 0.01` here.
- **Under SGD** (the scratch baselines): it is a **100× effective-LR multiplier**.
  `lr: 0.01` becomes an effective 1.0 on the CE term, and the fixed weight decay
  becomes ~100× weaker in relative terms. Visible in the log: scratch C12 seed1
  epoch 1 reports `train_loss=195.13` (`outputs/sparknet-dendritic-study-v2/study.log:9`).

So the scale is *working*, but it is also why the two phases sit in wildly
different optimisation regimes (→ L1).

**Is the sparsity weight tuned?** **No.** The 100:1 ratio is inherited verbatim
from the paper and has never been swept here. It is a direct capacity knob: the
gate (`sparknet.py:115-122`) is a 32-channel stochastic gate and the sparsity
term is `mean(Φ((tanh_out + 0.5)/0.5))` (`sparknet.py:124-136`), i.e. the mean
probability a gate is open. Measured occupancy sits around **0.30–0.61**
depending on run (`train_gate_sparsity` in the arm and phase_b JSONLs) — so
roughly 40–70 % of the gate is being shut off. At C2, where the `.fc` head is
396 of 1,150 parameters, that is a lot of capacity being discarded.

- **Expected gain:** ±0.3–0.8 pp; likely positive at narrow widths where
  capacity is scarce. *SPECULATION* on sign; the occupancy numbers are measured.
- **Cost:** 3 ratios × 1 width × 3 seeds ≈ 9 × 10.2 min ≈ **1.5 h**.
- **Risk:** medium. Changes effective capacity → changes what a `gate_conv`
  dendrite attaches to. **Must be frozen across arms.**
- **Validate:** sweep 100:1, 300:1, 1000:1 at C6 (the knee of the frontier).

---

### L3 — Turn the augmentation back on ★★★★☆

Current augmentation is, in practice, **time-shift only**
(`configs/train/sparknet_c16_dendritic_prune_no_kd.yaml:32-42`, identical in
`..._fast_io.yaml:22-34`):

```yaml
time_shift_ms: 100, probability: 0.8     # active
white_noise_db_range: [-90, -46], p 0.8  # -90..-46 dB is inaudible; near no-op
background_noise_probability: 0.0        # OFF
speed_factor_range: null                 # OFF
spec_augment: false                      # OFF
```

Both disabled paths are **already implemented and tested**:
`mix_background_noise` (`src/kws/data/augment.py:67`) and `SpecAugmenter`
(`src/kws/data/augment.py:210-219`, time-mask 30 / freq-mask 10 / 2 masks). The
project's own documented default is far stronger — `README.md` describes
"±150 ms shifts, 0.85–1.15 speed changes, 75% noise mixing down to −5 dB SNR,
and two larger SpecAugment masks". The weak setting exists to be paper-faithful,
not because it was found better.

- **Expected gain:** +0.5–1.5 pp at C8–C12. *SPECULATION* on magnitude.
- **Important asymmetry:** at C2 (63.3 %) and C4 (82.9 %) the model is badly
  **under**-fitting; more augmentation will likely *hurt* there. Do not apply
  uniformly across the width sweep without measuring.
- **Cost:** config-only + 3 widths × 3 seeds ≈ 9 × 10.2 min ≈ **1.5 h**.
- **Risk:** **high for study validity.** Augmentation sets how much headroom
  exists, and a dendrite's marginal value depends on whether the model is under-
  or over-fitting. **Must be frozen across arms.**

---

### L4 — Fix the `_unknown_` class, the measured bottleneck ★★★☆☆

Per-class F1 on the held-out test set for the C16 replication
(`outputs/sparknet-paper-replication/c16-seed0/test_report_fixedseed0.json`,
overall accuracy 95.07 %):

```
_silence_ 0.995 | stop 0.979 | right 0.969 | yes 0.969 | left 0.964 |
up 0.952 | on 0.946 | off 0.943 | no 0.940 | go 0.936 | down 0.934 |
_unknown_ 0.878   <-- 5.6 pp below the next-worst class
```

The same pattern holds for the teacher (`_unknown_` F1 0.958 vs ≥ 0.966
elsewhere, `reports/ds_cnn_l_teacher_current.json`). `_unknown_` is a pooled
class of ~25 distinct words sized at 1× the keyword-class mean
(`configs/data/speech_commands_v2_mfcc32_paper.yaml`, `target_ratio_to_avg_keyword_count: 1.0`),
so it has the widest intra-class variance and the *least* data per underlying word.

Lever: oversample `_unknown_` in training only (ratio 1.5–2.0) while leaving the
val/test prior fixed, or apply a class-balanced loss.

- **Expected gain:** +0.3–0.8 pp overall, concentrated in the bottleneck.
- **Cost:** config-only + 3 seeds ≈ **0.5 h**.
- **Risk:** changes the class prior; breaks direct comparability with the paper
  number. **Must be frozen across arms.**

---

### L5 — Best-checkpoint selection policy / EMA ★★★☆☆

`train.py:549` selects on strict improvement of raw epoch validation accuracy:

```python
if not best_state or val_acc > best_val_acc:
```

With val n = 4445 the binomial sd of a single evaluation at p ≈ 0.93 is
**0.38 pp**. Taking the max over 200 correlated epochs adds an optimistic bias
of roughly **+0.5–0.9 pp** — *larger than every effect this study is trying to
measure* (dendrite −0.21 pp, identity-FT −0.55 pp).

There is **no EMA, no SWA, no weight averaging anywhere** in `src/`
(`grep -rn "ema\|AveragedModel\|swa"` → no matches). This is the standard
cheap win for tiny models.

Two changes, independent:
1. **EMA of weights** (decay 0.999), evaluate and checkpoint the EMA copy.
   ~20 lines in `train.py`. Typical gain on models this size: +0.2–0.5 pp, and
   it *reduces* seed variance.
2. **Select on a smoothed criterion** (e.g. mean of the last 3 validation
   scores) instead of the raw max, to strip the selection bias.

- **Expected gain:** +0.2–0.5 pp real, plus a large gain in *measurement validity*.
- **Cost:** ~1 h engineering, zero extra compute.
- **Risk:** **must be frozen** — the study picks arms on best val, so changing
  the selection rule can change which arm wins.

---

### L6 — Raise the accuracy-per-parameter frontier at the narrow end ★★★☆☆

This is the study's stated purpose, and the frontier has a cliff:

| | C6→C4 | C4→C2 |
| --- | ---: | ---: |
| Δ params | −402 | −354 |
| Δ accuracy | −5.00 pp | **−19.53 pp** |

C2 loses 19.5 pp for 354 parameters. Two concrete, measured causes:

1. **The `.fc` head does not scale with width.** `gate_channels: 32` is hardcoded
   in every model config regardless of `channels` — the comment says *"The
   released checkpoints hard-code 32"* (`configs/model/sparknet_c2_paper.yaml:32-33`).
   So the head is `32 × 12 + 12 = 396` parameters at *every* width: 11.6 % of
   C12 but **34.4 % of C2**. The study's own C2 config already flags this
   (`sparknet_c2_paper.yaml:22-28`): a `.fc` dendrite copies ~50× more parameters
   than a pointwise one at C2, so cross-arm ranking at C2 largely measures which
   arm got the bigger budget.
   **Lever:** scale `gate_channels` with width (e.g. 16 at C2/C4). At C2 that cuts
   the head to 204 params — 17 % of the budget freed for width.
2. **The LR is not width-scaled.** All widths use `lr: 0.01` × 100 scale =
   effective 1.0 under SGD. That is plausible for 4,636 parameters and probably
   too aggressive for 1,150. C2's seed sd of **1.49 pp** (vs 0.32 at C8) is
   consistent with an unstable optimisation, not just a capacity limit.

- **Expected gain:** *SPECULATION* on magnitude, but the parameter accounting is
  exact. A width-scaled LR alone plausibly recovers several pp at C2/C4.
- **Cost:** 2 LR values × 2 widths × 5 seeds ≈ 20 × 10.2 min ≈ **3.4 h**.
- **Risk:** re-ranks the dendrite arms at narrow widths. **Must be frozen.**

---

### L7 — Lengthen the post-dendrite fine-tune ★★☆☆☆

`resume_epochs: 8` (`..._no_kd.yaml:59`). The resume phase trains only the base
parameters with the dendrite frozen, and it does a **full LR restart to 1e-3 with
cosine decay to zero over 8 epochs** — a mini-cycle, not a settling phase.
Measured at `arms/pointwise/c10-seed0`: 8 epochs, +0.16 pp, status
`no_improvement`; at `c10-seed1` it *lost* 0.36 pp against the pre-resume best.
Every completed arm run has a populated `resume` block; most read `no_improvement`.

- **Expected gain:** +0.1–0.3 pp from 24–32 epochs at a lower peak LR.
- **Cost:** 0.7 min per 8 epochs → ~+2 min/run × 150 runs ≈ **5 h**.
- **Risk:** the `control` arm config states the invariant explicitly — *"This
  value must equal the dendritic arm's"* (`..._control.yaml:81-85`).
  **Must be frozen and changed in lockstep.**

---

### L8 — Recognise what `improvement_threshold` is actually measuring ★★☆☆☆

`improvement_threshold: [0.005, 0.002, 0.001]` with `max_dendrite_tries: 3`
(`..._no_kd.yaml:97,100`). PAI must observe a **0.5 pp** validation improvement
to accept the first dendrite. The measured dendrite effect is **−0.21 pp**. So
the threshold is essentially never met and all three tries are consumed.

This is not a bug — it is a correctly conservative search. But it means the study
currently reports *"PAI rarely accepts a dendrite under a 0.5 pp bar"*, which is a
weaker claim than *"dendrites do not help"*. Worth stating explicitly in the
write-up. Changing it **changes the dendrite conclusion directly** and must be
frozen.

- **Expected gain:** none (it is a reporting/interpretation item).
- **Cost:** zero.

---

### Ranked summary

| # | Lever | Expected gain | Eng. cost | Compute | Risk to study |
| --- | --- | --- | --- | --- | --- |
| **L1** | Repair fine-tune handoff | **+0.5 pp mean, +2.4 pp @C2** | config only | **0 h** | high — freeze |
| **L2** | Tune gate-sparsity ratio | ±0.3–0.8 pp | config only | 1.5 h | medium — freeze |
| **L3** | Restore augmentation | +0.5–1.5 pp (wide widths only) | config only | 1.5 h | high — freeze |
| **L4** | Oversample `_unknown_` | +0.3–0.8 pp | config only | 0.5 h | medium — freeze |
| **L5** | EMA + smoothed selection | +0.2–0.5 pp, big validity win | ~1 h code | 0 h | high — freeze |
| **L6** | Width-scaled `gate_channels` + LR | several pp @C2/C4 (spec.) | new configs | 3.4 h | high — freeze |
| **L7** | Longer post-dendrite resume | +0.1–0.3 pp | config only | 5 h | high — freeze |
| **L8** | Reinterpret `improvement_threshold` | 0 (reporting) | zero | 0 h | freeze |

**KD sits below all eight**: best measured effect 0.00 pp, and it costs a 7 h
teacher retrain before it can even be tested on the study's frontend.

---

## 4. Question 4 — which levers must be frozen across arms

The study's claim is *"dendrites, at placement X, are worth Y at width W."* A
lever invalidates it if it changes the *dendrite* conclusion rather than raising
all boats equally.

**Every lever in §3 must be frozen.** That is the honest answer, and the reasons
differ:

| Lever | Why it changes the dendrite conclusion, not just the level |
| --- | --- |
| **L1 handoff** | Its damage is **width-dependent** (−0.16 pp @C12 → −2.40 pp @C2). It differentially depresses the baseline at exactly the narrow widths where dendrites are hypothesised to help most. This is the most dangerous one. |
| **L2 sparsity ratio** | Sets effective gate capacity. The `gate_conv` arm attaches a dendrite *to the gate*. Changing the ratio changes that arm's headroom specifically. |
| **L3 augmentation** | Determines whether a model is under- or over-fitting. A dendrite adds capacity, so its marginal value flips sign across that boundary. At C2 (63 %, underfitting) extra capacity should help; at C12 it may not. |
| **L4 class balance** | Changes the prior and the per-class difficulty profile, which changes what extra capacity buys. |
| **L5 selection policy** | Arms are **selected on best validation accuracy** (`selection_split: validation` in every arm report). Changing the selection rule can change the winner without changing any model. |
| **L6 `gate_channels`** | Sets the `.fc` dendrite's parameter cost (396) relative to `pointwise` (288) and `depthwise` (576). Changing it **re-ranks the arms mechanically**, independent of any dendrite effect. |
| **L7 `resume_epochs`** | The `control` arm exists solely to be budget-matched; its own config states equality is the invariant (`..._control.yaml:81-85`). |
| **L8 PAI search block** | `improvement_threshold`, `max_dendrites`, `max_dendrite_tries`, `n_epochs_to_switch`, `history_lookback`, `dendritic_schedule_epochs`, `post_integration_lr_multiplier`, `candidate_weight_initialization_multiplier`, `initial_correlation_batches` — these *are* the dendrite search. |

**Also freeze:** the five seeds (0–4), the data config, and the source scratch
checkpoints.

**Practical consequence:** the running study-v2 matrix must **finish on the
current recipe**. None of these can be changed mid-flight without splitting the
matrix into two incomparable halves. They are the **v3 recipe**, and v3 needs its
own fresh scratch baselines. The only genuinely safe knobs are IO
(`num_workers`, `cache_features`) — and even those are not free: the repo already
documents that worker count perturbs augmentation RNG and therefore results
(`configs/train/sparknet_narrow_paper_fast_io.yaml:12-16`).

---

## 5. Question 5 — the proposed next experiment

### 5.1 Timing basis (measured, from `study.log`)

Extracted by pairing each run marker in
`outputs/sparknet-dendritic-study-v2/study.log` with its bracketing log
timestamps (the file is 38,393 lines / 3.3 MB; parsed programmatically, never
`cat`-ed):

| Unit | Median wall-clock |
| --- | ---: |
| Scratch baseline (200 ep, SGD) | **10.2 min** |
| Full arm run | **11.0 min** |

Per-phase breakdown for `arms/pointwise/c12-seed0` (from its
`metrics/sparsity/**/*.jsonl` `elapsed_seconds`):

| Phase | Epochs | Time |
| --- | ---: | ---: |
| `prune_supervised` (identity FT) | 40 | **2.3 min** |
| `pai` | 128 | 9.4 min |
| `resume_supervised` | 8 | 0.7 min |

Remaining study-v2 budget: 28 of 150 arm runs are complete, so
**122 × 11.0 min ≈ 22.4 h** remain. Do not add to that queue.

### 5.2 The experiment: the identity-fine-tune ablation

**One question, decisively:** *is the −0.55 pp identity-fine-tune loss caused by
the recipe change at the handoff, and can it be removed?*

This is the right next experiment because it targets the largest measured effect
in the study (−0.55 pp, 26/28 runs), it is **5× the size of the dendrite effect it
is currently masking**, and it costs almost nothing — because **it does not run
PAI at all**. Only the 2.3-minute fine-tune phase is needed.

**Design.** 3 fine-tune recipes × 3 widths × 5 seeds = **45 runs**, each a
40-epoch identity fine-tune from the *existing* scratch checkpoints (which are
already on disk and free).

| Recipe | Definition |
| --- | --- |
| **A** (current) | `sparknet_c16_dendritic_prune_no_kd.yaml` as-is: AdamW, lr 1e-3, wd 1e-4, LS 0.1, cosine, 5 % warmup |
| **B** (repaired) | The L1 diff: SGD+momentum 0.9, lr 5e-4, wd 1e-3, LS 0.0, polynomial_hold, no warmup |
| **C** (null) | `epochs: 0` — no fine-tune at all; pass the scratch checkpoint straight through |

Widths **C12, C6, C2** — the two ends and the knee, where the damage ranges from
−0.16 pp to −2.40 pp. Seeds **0–4**.

Everything else is held at the current arm values: `task_loss_scale: 100.0`,
`batch_size: 256`, the data config, the augmentation block, and the entire
`perforatedai` block (inert here, since PAI never runs).

**Cost: 45 × 2.3 min ≈ 1.7 h**, plus ~15 min of orchestration. Scratch baselines
are already on disk.

**Decision rule.**
- If **B ≈ C > A**: confirmed, the handoff recipe is the cause. Adopt B as the v3
  fine-tune for all five arms and re-run the arm matrix.
- If **C > B ≈ A**: no fine-tune recipe helps; drop the identity fine-tune
  entirely for `method: identity` and feed PAI the scratch checkpoint directly.
  This is the cheapest possible outcome and also the most likely one at C2.
- If **A ≈ B ≈ C**: the loss is not the recipe; the next suspect is the
  augmentation RNG / dataloader-worker difference between the two entry points.

**Follow-on, conditional on B or C winning.** Re-run only `{control, pointwise}`
× `{C12, C6, C2}` × 5 seeds under the winning recipe: **30 runs × 11.0 min
≈ 5.5 h**. That re-establishes the dendrite verdict on a baseline that is not
already 0.55 pp in the hole.

**Total: ~7.2 h**, versus ~22.4 h to finish the current matrix and ~7 h just to
retrain a teacher that comparison B suggests will buy 0.00 pp.

### 5.3 What this experiment deliberately does not do

- It does not launch anything now. The study-v2 process must finish first, or
  this must run on separate capacity.
- It does not touch KD. Per §1, KD is the wrong place to spend the next 7 hours.
- It does not use the test split. Arm selection stays on validation
  (`objective.use_test: false`, `..._no_kd.yaml:61-63`), and test is touched only
  after the arms are frozen.

---

## 6. Provenance

- Prior KD investigation recovered from git: `git show e0ecadd:KWS_Model/KD_DIAGNOSIS.md`
  (7,216 bytes). `PLAN.md`, `ERRORS.md`, `SPARK_NET_FIXES.md` also exist in
  history at `9c1b179^`. None are in the working tree at HEAD despite
  `SPARKNET_DENDRITE_FIXES.md` claiming they were restored, and despite
  `README.md` still referencing `PLAN.md` — worth re-restoring.
- Diagnostic tool: `scripts/diagnose_kd.py` (paired clean/augmented teacher views,
  KD/CE logit-gradient ratio and cosine). Raw prior reports were retained under
  `outputs/diagnostics/kd_20260914/`.
- Study-v2 aggregates in §3.1 and §5.1 were computed from the on-disk YAML/JSONL
  artifacts and the study log; no run was re-executed.
