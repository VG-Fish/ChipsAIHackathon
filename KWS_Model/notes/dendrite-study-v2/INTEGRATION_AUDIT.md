# SparkNet / PerforatedAI integration audit

**Audited:** 2026-09-17  
**Scope:** the live `sparknet-dendritic-study-v2` configuration and the code
that creates, trains, resumes, and exports its PAI-wrapped models.  This is an
audit only: it makes no code/config changes and does not start, stop, or alter
the running study.

## Bottom line

The live integration is functionally sound enough for the current matrix to be
interpretable: PB is enabled, the intended modules are actually wrapped, base
weights and BatchNorm statistics are handled correctly in the dendrite phase,
the model is moved back to the device after PAI restructuring, and optimizer /
scheduler state is rebuilt and checkpointed at restructuring boundaries.

There is no verified implementation bug that justifies changing study-v2 while
it is running.  Three forward-looking hazards merit a v3 ablation or explicit
configuration: hidden dependence on PAI's output-dimension default,
nonzero AdamW weight decay despite the vendor caution, and enabling PAI's
`unwrapped_modules_confirmed` escape hatch without a permanent coverage test.
The much larger *experimental* confound is the identity-fine-tune loss, which
is documented in `RESULTS_ANALYSIS.md`; it is not a PAI lifecycle defect.

## Evidence gathered

### Static code audit

| Requirement / hazard | Evidence | Verdict |
| --- | --- | --- |
| Configure PAI before wrapping | `configure_perforatedai` sets device, PB, phase controls, and module filters before `UPA.perforate_model` in `src/kws/optimize/dendritic.py` | Pass |
| PB actually enabled | `set_perforated_backpropagation(True)` and live log prints `Building dendrites with Perforated Backpropagation`; installed versions recorded in journal §4.2 | Pass |
| Exact arm placement | Built-in class defaults are cleared; `module_ids` are passed directly to PAI | Pass |
| PAI needs a validation score and may restructure | Each epoch calls `GPA.pai_tracker.add_validation_score(val_acc, model)`, then `.to(device)` | Pass |
| Optimizer after restructuring | A new AdamW + scheduler are created and registered with `set_optimizer_instance` when `restructured` | Pass, with semantic caveat below |
| Base frozen during P mode | The code measures optimizer membership and pins base BatchNorm running statistics in p mode | Pass |
| Resumption / artifact safety | Paired native PAI + KWS sidecar state is saved at validation boundaries and after integrations; final clean graph is exported | Pass |
| Headless PAI debugger | Vendor `pdb.set_trace()` calls are converted to useful exceptions before configuration | Pass |

Relevant code anchors: `configure_perforatedai` (lines 1849--1929),
`_make_optimizer_and_scheduler` (1932--1978), the p-mode guard
(2420--2461), validation/restructure handling (2516--2602), and paired
checkpoint/final export handling (2633--2734) in
`src/kws/optimize/dendritic.py`.

### Empirical wrapping verification

I ran the repository's read-only converter probe:

```bash
cd KWS_Model
uv run python notes/dendrite-study-v2/verify_placements.py \
  notes/dendrite-study-v2/placements.json
```

The probe builds each study width and calls the same placement resolver,
`configure_perforatedai`, and `UPA.perforate_model` path used by training.
It does not train or touch `outputs/`.  Its complete machine-readable result
is [`placements.json`](placements.json).

At C12, the observed wrapped modules and one-dendrite accounting are:

| Arm | Intended / observed PAI wrapper(s) | Base params | Copy + residual params | Projected one-dendrite params |
| --- | --- | ---:| ---:| ---:|
| `pointwise` | `blocks.2.pointwise`, `blocks.3.pointwise` | 3,400 | 288 + 24 | 3,712 |
| `fc` | `fc` | 3,400 | 396 + 12 | 3,808 |
| `gate_conv` | `gate_conv` | 3,400 | 416 + 32 | 3,848 |
| `depthwise` | `blocks.2.depthwise`, `blocks.3.depthwise` | 3,400 | 576 + 24 | 4,000 |
| `control` | none | 3,400 | 0 + 0 | 3,400 |

The immediate `params_after_wrap` value is smaller than the projection by the
residual scale terms because no dendrite has been accepted at conversion time;
that is expected.  The control arm's module list and parameter count are empty
and unchanged, so it is a real scheduled no-dendrite control rather than an
accidentally perforated model.

## Correctness details

### Placement and model semantics

The study uses `conversion: module_ids`, validated by
`dendritic_config.py`.  The two pointwise targets are exact Conv2d modules;
they are wrapped as such.  This isolates cross-channel mixing in the late
DSConv blocks.  The depthwise targets retain their `groups=C`, so a depthwise
dendrite cannot create cross-channel mixing; its arm measures a materially
different function class.  `fc` and `gate_conv` each have one target.

The module-id resolver rejects missing modules and overlaps but intentionally
admits an empty list for the control arm.  This is correct and is documented
in `placement_module_names` / `pai_module_ids`.

### PAI lifecycle

The code follows the important vendor lifecycle:

1. configure global PAI state and wrap after the base checkpoint/fine-tune;
2. train, validate, and pass a validation score to PAI at each boundary;
3. move the returned model to the active device after a potential rewrite;
4. rebuild optimizer/scheduler if the module graph changes;
5. export a clean model once PAI reports completion.

The live pointwise C12 artifact confirms a normal `n → p → n → n` style
cycle, one accepted integration, and zero base parameters in the p-phase
optimizer after the optimizer rebuild.  This is the critical invariant: a
candidate's correlation is not being measured against a moving base model.

### Checkpointing and deployment

`pai_saves` is disabled in the vendor configuration, so native vendor
`load_pai_model` convenience reloads are not the deployment contract here.
That is intentional rather than a failure: the pipeline stores paired native
and KWS state for restart and calls `export_final_pai_model` for the clean graph
used by profile/resume.  A deployment consumer should use the clean exported
artifact documented in each run report, not assume a vendor save file exists.

## Issues and recommended disposition

| Priority | Finding | Why it matters | Disposition |
| --- | --- | --- | --- |
| P0 | No verified lifecycle fault | Changing a live matrix would destroy comparability with no demonstrated fix | **Do not alter study-v2.** |
| P1 | `weight_decay: 1e-4` is passed to AdamW in every arm | Vendor documentation cautions against weight decay for dendrites; a zero-initialised branch gain is especially vulnerable to decay | Run a post-v2, paired `wd=0` ablation with matching control/pointwise arms. |
| P1 | Output dimensions rely on the installed PAI default | Current artifact shows `[-1, 0, -1, -1]`, the desired Conv2d channel axis for SparkNet, but `configure_perforatedai` does not set it explicitly | Add an explicit `set_output_dimensions([-1, 0, -1, -1])` plus a test before a future run; do not rewrite an in-flight configuration. |
| P2 | `set_optimizer_instance` is the fallback integration path | It works and is exercised, but PAI does not own the external scheduler; package defaults such as internal LR-search options must not be interpreted as active | Document this choice and explicitly disable inert package options if/when PAI exposes them in config. |
| P2 | `set_unwrapped_modules_confirmed(True)` suppresses a vendor guard | The converter probe proves today's intended wrappers, but a future SparkNet parameter/module could be untracked without prompting | Turn the probe into a CI test or assert a complete accounting of model parameters before future runs. |
| P3 | Module-local pointwise wrapping leaves adjacent BatchNorm outside the copied module | This is the correct isolation design for study-v2, but differs from vendor advice to group normalisation with a preceding layer | Keep it frozen in v2; evaluate whole-block placement only as a separately budgeted v3 arm. |
| P3 | In-place activations and residual paths are structural PAI risk areas | No failure is present in conversions or completed pointwise runs | Keep their conversion probe and phase artifacts as a regression check. |

## Follow-up plan, in order

1. Let the current five-arm matrix finish unchanged.  Continue to treat its
   output as validation-only until arm/width selection is frozen.
2. Compare each arm to the scheduled `control` and to its own paired scratch
   checkpoint; do not compare raw means across differently sized models.
3. First post-v2 code/config work: make output dimensions explicit and add a
   conversion/parameter-accounting regression test.  Then rerun the smallest
   `control`/`pointwise` weight-decay ablation.
4. Address the identity-fine-tune handoff before inferring a negative or
   positive PB effect as intrinsic.  That intervention must receive fresh
   scratch baselines.

## Reproduction notes

- `verify_placements.py` is safe to rerun; it uses a temporary directory for
  PAI's conversion side effects and restores the working directory.
- `aggregate_study.py` is read-only with respect to `outputs/` and refreshes
  only `notes/dendrite-study-v2/aggregate_study.json`.
- Never run `scripts/report_test_accuracy.py` before selection is frozen: it
  consumes the study's held-out-test budget.
