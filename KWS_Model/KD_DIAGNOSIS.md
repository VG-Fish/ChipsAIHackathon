# SparkNet response-KD diagnosis

Measured on 2026-09-14 UTC using the existing DS-CNN-L teacher and the running
SparkNet C12 student's best checkpoint. No teacher retraining or test-set
evaluation was used. The replacement recipe is an evidence-motivated experiment,
not a demonstrated accuracy improvement.

## Findings

Two paired-view training probes give the same picture. Each pair uses the same
underlying waveform (including synthesized silence); only the student's light
augmentation differs. Samples are selected without replacement within each
probe, but the two probes can overlap and are not independent full training runs.

| Measurement | 1,024 pairs, seed 31415 | 2,048 pairs, seed 27182 |
| --- | ---: | ---: |
| Teacher accuracy, clean | 98.54% | 98.68% |
| Teacher accuracy, augmented | 98.83% | 98.49% |
| Teacher mean confidence, augmented | 89.65% | 89.60% |
| Teacher entropy, augmented (nats) | 0.511 | 0.514 |
| Old KD target change, mean total variation | 0.00977 | 0.01003 |
| Old weighted KD / CE logit-gradient norm | 0.148 | 0.146 |
| Old KD / CE logit-gradient cosine | 0.904 | 0.885 |
| T=2, equal weights: weighted gradient norm ratio | 1.233 | 1.209 |
| T=2, equal weights: gradient cosine | 0.737 | 0.708 |

This does **not** support a substantial teacher failure on the student's
augmentation. On the first validation subset the teacher reached 97.85%, close
to its checkpoint's recorded 97.78%. The first training probe contained 89
student mistakes; the teacher was correct on 80 of them.

The concrete recipe weakness is a small, largely redundant teacher contribution.
At temperature 1, the student-logit gradient of the old loss is exactly the
cross-entropy gradient against

`effective_target = (6/7) * smoothed_label + (1/7) * teacher_probability`

(up to the config's decimal rounding; teacher entropy contributes only a constant
to the objective). Teacher probabilities are already close to the labels
smoothed at 0.1. The old mixture therefore moves only about 1% probability mass
on average. Its weighted gradient is about 15% of the CE gradient and points
mostly in the same direction. The code's KL direction, batch reduction, and
temperature-squared scaling are correct; the weakness is in the recipe, not a
reversed KL or detached student gradient.

The teacher stays frozen in evaluation mode, receives exactly the same input
tensor as the student, and has the same feature shape and class ordering.
Feature matching is disabled, so the loss imposes no equality between DS-CNN
and SparkNet internal representations. Student capacity can still limit the
benefit: these probes do not isolate architectural capacity or prove that a
different temperature will improve final accuracy.

## Change and verification

`configs/train/light_kd.yaml` now uses temperature 2, response weight 0.5,
classification weight 0.5, and feature weight 0. All non-KD settings stay fixed,
including label smoothing 0.1, augmentation, stochastic gates, and 200 epochs.
The measured logit-gradient ratio becomes roughly 1.2, with a less redundant
direction. This is a bounded next experiment, not a tuned optimum.

Live training now records teacher accuracy, confidence, true-class probability,
entropy, weighted loss components, and analytical KD/CE logit-gradient
balance/alignment. Those diagnostics are detached and do not change the loss or
backpropagation. Graphs retain and display them. Online ratios/cosines are
sample-weighted means of batch statistics, whereas this table uses each entire
probe. The probe uses student eval mode; live training includes gate noise and
training-mode batch normalization, so their numbers need not match exactly.

Regression coverage checks gradients against autograd, diagnostic detachment,
zero-norm behavior, graph visibility, recipe settings, and exact training resume.

## Limits and provenance

- Teacher SHA256:
  `058c53bb99b1edf22d10f179aca18e0b16925bb80740051182b6b7d66b8e8048`.
- The external teacher checkpoint contains no training-config provenance. Its
  weights do **not** match the teacher in `outputs/full-run-20260912T063022Z`.
  That run's saved strong-augmentation/cached-view recipe cannot establish how
  this particular teacher was trained. Teacher label smoothing is therefore
  not asserted as a verified cause.
- The completed no-KD baseline peaked at 92.15%. The old KD run was stopped
  after 169 epochs, peaking at 92.05% at epoch 160. This small difference
  between an incomplete and completed run does not establish a reliable loss.
  Identical nominal seed values also do not guarantee matched initialization:
  the plain and KD entry points consume RNG differently before constructing
  the student. Multi-seed, matched comparisons are needed for an accuracy claim.
- The first probe's clean/augmented teacher prediction flip rate was 5/1,024.
  The small clean-versus-augmented accuracy differences should not be interpreted
  as augmentation improving or hurting the teacher.

Raw reports, per-class probabilities, and logits are retained under
`outputs/diagnostics/kd_20260914/` and its `replicate_27182/` subfolder. They are
outside the replaced run so cleanup does not remove the diagnosis.

## Replacement run

The superseded run `step2_sparknet_c12_light_kd_20260913T233255Z` was gracefully
interrupted and moved from `outputs/phase_b/` to macOS Trash (recoverable).
The existing teacher, completed no-KD baseline, and diagnostic reports were
preserved. The new stage-2 run starts from a newly initialized student, without
`--resume-dir` or `--student-checkpoint`, under
`outputs/phase_b/step2_sparknet_c12_light_kd_t2_20260914T013331Z/`, with `--graphs`.
It does not rerun stage 1 or continue the old optimizer state.

Startup verification completed after epoch 1: both best/latest checkpoints and
the HTML/CSV graphs were written; the saved recipe contains T=2, equal weights,
and no student warm start. All logged diagnostics were finite. Teacher accuracy
on the full epoch's augmented inputs was 98.65%, with weighted KD/CE gradient
norm ratio 1.090. This verifies the new signal is active, not an accuracy gain.

Final code verification: 264 tests passed; `ty check` passed for all changed
Python files. Repository-wide `ty check` still reports 32 pre-existing
diagnostics outside those files.

## Reproduction

Reproduce on a current student checkpoint:

```bash
uv run python scripts/diagnose_kd.py \
  --student-checkpoint outputs/YOUR_RUN/models/checkpoints/student/best.pt \
  --samples 1024 --seed 31415 \
  --output-dir outputs/diagnostics/YOUR_AUDIT
```

The general rationale for temperature-scaled response KD comes from
[Hinton et al.](https://arxiv.org/abs/1503.02531). Teacher label smoothing can
reduce transfer in some settings ([Müller et al.](https://arxiv.org/abs/1906.02629)),
but that literature does not establish the cause for this checkpoint.
