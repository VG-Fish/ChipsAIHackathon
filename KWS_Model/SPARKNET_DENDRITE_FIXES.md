# SparkNet dendritic audit progress

This file is the live progress record for the review of the five-seed,
three-dendrite, `.fc` student-SparkNet prune run and the preceding agent's
changes.

## Scope checked

- Read `AGENTS.md` and all of `KWS_Model/README.md`.
- Loaded the PerforatedAI integration, results-analysis, and debugging
  guidance.
- Inspected the five seed reports and their prune/PAI/resume metrics under
  `KWS_Model/outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/`.
- Compared the local recipe with the SparkNet paper and reference repository.
- Audited the uncommitted implementation, configs, tests, and helper scripts.

## Initial evidence

- The five seed directories contain C12, C10, and C8 candidates, not one
  single result. The aggregate reports identify the PAI placement as `.fc`,
  `max_dendrites: 3`, validation-only, and no KD.
- The reports use paper-replication C16 checkpoints as sources, with 4,636
  deployed parameters and about 396K local MACs; the paper's C16 reference is
  4,636 parameters and 454.5K MACs on SC2. The local run therefore needs a
  careful split/frontend/metric caveat before claiming a paper reproduction.
- The PAI JSONL records contain `seed: 0` in every seed directory. This is a
  finding to resolve: it may be vendor metadata, but it must not be presented
  as proof that the five runs used independent PAI seeds.
- The current config files were changed to a new one-dendrite recipe with a
  lifted threshold, while the analyzed outputs were produced by the prior
  three-dendrite recipe. Those are historical results and must not be mixed
  with the new default recipe.

## Confirmed problems and fixes

- The five historical directories are not five independently seeded downstream
  runs: their prune/PAI sidecars all report seed 0, and their manifests report
  null. Only the source checkpoints differ by directory seed. The runner now
  resolves an experiment-level seed, passes it to every cycle, records it in
  the aggregate report, and the CLI session manifest; all experiment seed
  configs now declare their seed explicitly.
- The PAI no-KD loop previously optimized unscaled cross-entropy even though
  the C16 train recipe declares `task_loss_scale: 100` plus SparkNet gate
  sparsity. It now uses the shared loss-scale validator and applies the scale
  before auxiliary losses, matching ordinary fine-tuning.
- The prior report headline changed meaning whenever a PAI minimum-parameter
  row was present: final post-resume accuracy was subtracted from that row and
  called a dendrite gain. The headline now remains the final end-to-end delta
  versus the prune/fine-tune baseline. PAI-row deltas are separate, explicitly
  labeled within-search comparisons; they are not causal no-dendrite controls.
- The backfill utility previously reinstated that same overclaim. It now keeps
  the final-minus-prune pipeline delta as the headline, records the PAI
  minimum-parameter row only as descriptive context, and the five historical
  reports (15 candidates) have been normalized to the corrected schema.
- The held-out-test reporting utility no longer treats a `seedN` directory
  name as stochastic provenance. It reports run count separately, computes a
  sample SD only for distinct seeds recorded in report/checkpoint metadata,
  and therefore cannot present the historical downstream artifacts as a
  verified five-seed result.
- Placement-config comments no longer transfer the classifier's 396 copied
  parameters or historical `.fc` search behavior to unrun placements. The C12
  copied-module projections are labeled per placement (pointwise 288,
  gate-conv 416, depthwise 576), before PAI residual scale terms.
- Direct cost projection now accepts the `-1` unlimited sentinel and uses the
  documented one-dendrite lower bound. Runtime resume-boundary checks translate
  unlimited to a positive PAI cap, avoiding invalid sentinel comparisons.
- Restored deleted historical audit notes (`PLAN.md`, `ERRORS.md`,
  `KD_DIAGNOSIS.md`, `PAI_TORN_CHECKPOINT_PAIR.md`, and `SPARK_NET_FIXES.md`)
  because README/source references still point to them.

## Paper comparison and current interpretation

- The SparkNet paper reports C16 at 4,636 parameters, 454.5K MACs, and 95.7%
  SC2 test accuracy; local C16 source artifacts are validation-only and use a
  local MAC counter (396,304), so they are not a direct paper-test replication.
- Historical `.fc` d3 final validation means across the five directory-labeled
  outputs are C12 92.65% +/- 0.32%, C10 90.98% +/- 0.26%, and C8 88.04% +/-
  0.62% (sample SD). The corresponding final-minus-prune deltas are +2.50,
  +5.19, and +8.12 percentage points, but those include the longer PAI and
  post-PAI resume schedules. Final-minus-PAI-minimum-row deltas average only
  +0.23, +0.45, and +0.88 points; these are search-row comparisons, not causal
  dendrite effects. No test accuracy is present.
- The historical d3 recipe uses `.fc`, max dendrites 3, thresholds
  `[0.001, 0.0001, 0.0]`, AdamW/batch 256/40 prune epochs plus 130 PAI epochs
  and 8 resume epochs. Current configs target a separate d1 rerun with lifted
  thresholds; no d1 training was launched during this audit.
- PerforatedAI guidance favors testing `.fc`, pre-tanh `gate_conv`, backbone
  pointwise, and depthwise placements under matched parameter/MAC budgets.
  Future runs should also use an independently trained no-dendrite arm and
  test evaluation only after validation-based arm selection.

## Verification

- Fresh isolated reviewers independently checked seed/loss propagation,
  report semantics, documentation, and unlimited-dendrite boundary behavior.
  One reviewer found a CLI-manifest fallback gap when only the train config
  records a seed; the CLI now resolves that seed before opening the session.
- Reporting and resume regression tests cover: no path-derived seed
  provenance, SD suppression for unknown/repeated seeds, corrected historical
  backfill semantics, and rejection of aggregate resume across a changed or
  unrecorded seed.
- Final focused suite passed: `95 passed in 13.55s` across the SparkNet
  experiment, reporting utilities, compression projection, dendritic loop,
  and resume-regression tests.
- `python -m compileall -q src tests scripts` passed.
- `git diff --check` passed.
- The three new reporting files pass `ruff check` and `ruff format --check`;
  the final post-format reporting/experiment regression run passed `17
  passed`. Repository-wide lint still contains pre-existing import-order and
  unused-import findings in older optimization modules.
- A post-backfill scan confirmed all 15 historical candidate records use
  `validation_accuracy_gain_basis: prune_finetune_baseline` and carry
  `pai_zero_architecture_comparison_basis:
  minimum_parameter_row_in_best_arch_scores`.

## Type-check follow-up

- `ty check` initially reported five diagnostics: three caused by an
  over-broad `object` annotation in the PAI restart-pair monitor and two caused
  by the dynamically wrapped PAI model not being narrowed to the existing
  `FeatureModel` protocol in the KD branch.
- The monitor now carries `PaiPairStatus` through its suspect list and display
  helper. The PAI KD loop now uses the same explicit `FeatureModel` narrowing
  already used by the ordinary training loop; runtime behavior is unchanged.
- Final `ty check` passed with no diagnostics. The focused dendritic and
  restart-pair suites passed: `68 passed in 12.26s`.
