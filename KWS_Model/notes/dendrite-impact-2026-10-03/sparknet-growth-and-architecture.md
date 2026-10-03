# SparkNet, PerforatedAI growth, and architecture audit

Research begun 2026-10-03. This note is being written throughout the artifact audit. All accuracy differences are percentage points unless stated otherwise. Existing summaries are reference context; numerical conclusions will be recomputed from original reports and epoch logs.

## Scope and evidence policy

Assigned families: `sparknet-dendritic-study-v2`, `sparknet-grow-dendrites-v3`, `sparknet-pai-documented`, `sparknet-lowdata02`, `sparknet-lowdata10`, `sparknet-lowdata20`, `sparknet-native-dendrites`, and SparkNet paper replication. No training, code change, licensed PAI import, or private environment-file read is needed. Dendrites in `sparknet-native-dendrites` are classified separately because the user's subject is PerforatedAI.

Read [root AGENTS.md](../../../AGENTS.md), [root README](../../../README.md), [KWS README](../../README.md), and the complete existing [SparkNet/Pico research](../dendrite-study-v2/agent-sparknet-pico-deployment-research.md) and [paper MAC audit](../dendrite-study-v2/sparknet-paper-macs-audit.md). Older notes identify a real counting-convention risk: the published paper uses THOP including BN costs, while local reports count conv/linear products and explicit learned skip operations. Local C16 is 4,636 parameters / 396,304 local MACs, versus published 4,636 / 454.5K THOP MACs. We must preserve those labels.

## Initial observations to verify

- Raw v3 reports explicitly label `selection_split: validation` and `test_split_used: false`. Their best validation accuracy is an epoch-selected statistic, not held-out test accuracy.
- A real C10g16 block-2 BN dendrite has 130 additional deployed parameters and 11,110 additional local MACs. Its report records 0.0 candidate-phase base-weight drift, zero integration output difference, restored momentum, and zero clean-export parity difference.
- A `fc-sham` report still says one dendrite was added and includes its branch cost, while its skip weights remain zero and its on/off accuracy is identical. Counting `num_dendrites_added` alone would misclassify this control as evidence of an active dendrite.
- Candidate phase accuracy can be constant by construction while base weights are frozen and the candidate has not been integrated. This does not establish failure of candidate training; PB scores and post-integration results must be examined.

## Audit progress

2026-10-03: inventoried report schemas and assigned output families; investigating seed independence and paired controls next.

## Results established from complete report/epoch inventory

The audit scripts now preserve every original report row and aggregate paired differences in [all run inventory](sparknet-all-runs.json), [paired growth rows](sparknet-paired-growth.json), [growth aggregates](sparknet-growth-aggregates.csv), and [post-hoc aggregates](sparknet-posthoc-aggregates.json). Reproduction commands are appended below once the audit is final.

### Seed and cohort checks

All 150 v2 reports carry the intended directory seed; every emitted epoch seed agrees. For example, [C8 seed4 classifier manifest](../../outputs/sparknet-dendritic-study-v2/arms/fc/c8-seed4/manifest.yaml) and its [identity fine-tune metrics](../../outputs/sparknet-dendritic-study-v2/arms/fc/c8-seed4/metrics/sparsity/sparknet_c8/prune_supervised.jsonl) both record seed4. This family avoids the earlier C16 sweep's downstream-seed0 problem. Its original source checkpoints are the matching same-width same-seed scratch checkpoints.

For growth runs, 262 complete report/scratch pairs are available. In 261, **every pre-switch validation epoch matches scratch exactly**; the sole exception is the explicitly KD-trained C4 pilot, whose objective differs from ordinary supervised scratch. Candidate phases add 15 wall-clock epochs to the 200 base epochs, and can advance loader/augmentation random streams after the switch. Hence same base schedule and pre-switch equality establish a strong paired experiment, but shams are still needed to quantify the training perturbation from the candidate pause. The 10% training subset uses a fixed subset seed0 while unknown/silence construction still varies with the run seed; comparisons are within same-seed cohorts.

### v2 outcome and limitation

All 150 arm/control runs completed: 6 widths × 5 seeds × 5 arms (control, fc, gate_conv, pointwise on blocks2+3, depthwise on blocks2+3). Every one of the 24 placement×width mean differences against conventional continuation is negative, with its unadjusted paired 95% t interval below zero. This is a negative result for the executed workflow. It does not prove dendrites intrinsically hurt SparkNet: source→40-epoch identity fine-tune→PAI uses a different optimizer and restarted schedule, and the conventional control's continuation horizon differs from PAI's history-driven search. Most accepted branches are retained, so integration failure is an inadequate explanation for the family as a whole.

### Low-data result

The strongest repeated evidence is 20 same-seed pairs for 10%-data C16g16 with one block1 BN-grouped dendrite. Its mean best-validation gain is **+0.00225 pp**, paired t interval approximately **[-0.082,+0.087] pp**. The active dendrite raises parameters from 4,140 to 4,444 and local MACs from 370,256 to 397,728. Under this experiment there is no evidence of a useful average accuracy gain compensating for +7.34% parameters/+7.42% local MACs. More placements (block1+2 or block1+2+3) have negative five-seed means and higher costs. The 2%/20% experiments show small positive C16 means with intervals including zero, not a demonstrated broad advantage from reducing training data.

### Test-report semantics discovered

The broader v3 test file [all evaluations](../../outputs/plots/sparknet-dendritic-comparison/broader_v3_test_accuracy_all.json) has 11 unique test evaluations and no ordinary matched scratch tests. Its shorter counterpart repeats 8 of those rows. These evaluations use `final_clean_pai.pt`; the attached `validation_accuracy` is selected/best validation accuracy, so the two values describe different checkpoint selection semantics.

**Its FAR and FRR fields are wrong for the 10-keyword task.** The [test-report script](../../scripts/report_test_accuracy.py) falls back from missing `num_keywords` to `num_classes` (12). [Metric computation](../../src/kws/utils/metrics.py) then treats unknown and silence as keywords, yielding FAR=0 and FRR=1-accuracy despite substantial unknown→keyword errors in the stored confusion matrices. Top-1 accuracy remains valid. This audit will recompute 10-keyword FAR/FRR from those matrices in new research artifacts and preserve the originals.

### Hardware notes fully read

Read the complete 4,478-line [ReRAM integration guide](../ReRAM%20Simulation/ReRAM%20Simulation%20Implementation.md) and [implementation journal](../ReRAM%20Simulation/IMPLEMENTATION_JOURNAL.md). The guide is a plan and accounting policy; the journal explicitly says no actual upstream NeuroSim compilation/simulation was claimed because the external checkout/interface was unavailable. Parameter/MAC savings cannot be translated directly into ReRAM area/energy: tiny dendritic matrices, array granularity, and peripheral overhead can dominate. SparkNet depthwise connectivity must be represented correctly, and tanh/add/BN/frontend costs can sit outside a matrix-only simulator. Host CPU timing, modeled ReRAM timing, and physical Pico/ESP32 timing remain separate evidence categories.
