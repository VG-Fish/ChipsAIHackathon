# Historical DS-CNN evidence and primary paper context

Audit date: 2026-10-03. All percentages below are validation accuracy unless explicitly marked test. This review reads raw artifacts; no model training or licensed package import was performed.

## Direct DS-CNN PAI runs

The canonical PAI CSV is exactly `<candidate-directory-name>_best_arch_scores.csv`; retries and `beforeSwitch` snapshots are not additional independent runs. Values recomputed by [historical_dscnn_audit.py](historical_dscnn_audit.py), with complete numerical records in [historical-dscnn-evidence.json](historical-dscnn-evidence.json).

| Family/candidate | Prune fine-tune best | PAI zero-dendrite best | PAI selected best | Increment after zero-dendrite | Base → clean deployed parameters | Status |
|---|---:|---:|---:|---:|---:|---|
| compression-run/w18_classifier | 65.207% | 68.698% | 70.164% | +1.466 pp | 1,830 → 2,070 | candidate complete; parent interrupted |
| compression-run/w14_classifier | 64.687% | 66.133% | 66.442% | +0.309 pp | 1,522 → 1,714 | candidate complete; parent interrupted |
| compression-run/w10_classifier | 62.064% | 64.262% | 64.262% | no grown architecture in canonical CSV | 1,246 → 1,246 observed | partial PAI; no completed dendritic result |
| full-run/candidate_w18 | 79.383% | 82.584% | 82.681% | +0.096 pp | 1,830 → 2,070 | candidate recorded; parent failed |
| full-run/candidate_w17 | 78.939% | 79.286% | 79.286% | no grown architecture | 1,750 → 1,750 observed | PAI interrupted after three epochs |
| standalone dendritic_framework_w18 | not associated with a completed current report | 79.769% | 79.769% | no grown architecture | 1,830 | zero-dendrite-only artifact; empty switch file |

Sources: [compression report](../../outputs/compression-run/reports/compression_experiment.yaml), [later sparsity report](../../outputs/full-run-20260912T063022Z/reports/sparsity.yaml), each family's `pai/candidates/*/*_best_arch_scores.csv`, and `metrics/sparsity/**/*.jsonl`. The legacy w6 conventional row completed at 54.426%; its dendritic placements remained planned. Several broader placements were rejected before training by the 1.5M-MAC budget. Rejected or unexecuted placements are not negative dendrite results.

The apparent +4.957 pp legacy w18 improvement over pruning fine-tuning contains +3.491 pp of pre-dendrite continuation and +1.466 pp after the zero architecture. The later, stronger recipe reduces that last increment to +0.096 pp. This supports recipe and baseline quality as major moderators. It does not prove the remaining increment is caused exclusively by dendritic capacity: there is no matched independent continuation control, no equal-cost architecture control, and no held-out test evaluation of these dendritic candidates.

The eight resume-KD epochs did not raise the selected PAI maximum: legacy w18/w14 resume best were 70.145/65.921%; later w18 resume best was 82.527%. The later w18 row was rejected by the 85% validation admission floor.

There is substantial training cost even when inference growth is small: legacy w18 records 249 PAI epochs totaling 10,492 seconds; w14 records 140 totaling 5,845 seconds; later w18 records 301 totaling 19,168 seconds. These are sums of recorded per-epoch elapsed times, not reliable end-to-end wall-clock estimates across restarts. The later report's 589-second elapsed field describes a resumed invocation and must not be mistaken for the whole 301-epoch search.

## MAC accounting correction

Historical deployed costs claim w18 1,450,656 → 1,450,668 MACs and w14 1,209,888 → 1,209,900. That +12 is only the scale term. The copied classifier performs 216/168 extra dot-product MACs respectively. At the same input shape and with the current counter's convention, corrected totals are **1,450,884** and **1,210,068**, excluding activation nonlinearities. These are analytical corrections, not newly profiled historical models.

The mechanism is visible in installed `clean_perforatedai.c`: the main branch invokes `.forward()` directly, bypassing module hooks. The current [profile helper](../../src/kws/utils/profile.py) routes those branches through hookable calls during profiling. Historical raw reports are preserved; their low reported MAC increments do not establish effectively free dendrites.

The 2,058/1,702 native CSV counts omit 12 scale parameters present in the 2,070/1,714 clean graph counts. Thus even within one run the PAI architecture count and deployed count can differ. Use actual exported graph costs for deployment claims.

Recorded latency values are host CPU timings. Recorded one-byte weight memory is projected logical precision; it is not proof those artifacts are quantized. There is no measured ESP32/MRAM latency, RAM arena, power, or complete clustering/quantization/export chain for these direct historical PAI candidates.

## Baseline context

The 469,604-parameter log-mel DS-CNN-L teacher has 97.686% test accuracy; the 4,096-parameter XS student has 85.111% validation and 83.643% test ([reports](../../reports/phase_a/), [teacher report](../../reports/ds_cnn_l_teacher_current.json)). The older 146,902/467,942-parameter sanity runs had six outputs; they cannot be compared directly with the later 12-class counts.

The earlier audit's explicit correction establishes `(40,101)` as the trained log-mel input shape, rather than synthetic `(40,98)` profiles ([audit correction](../dendrite-study-v2/agent-dscnn-results-audit.md)). SparkNet and newer faithful batches use other frontend details. Cross-family numerical differences do not isolate architecture or dendrite mechanism unless frontend, balance, split, loss, and training recipe agree.

## Primary PerforatedAI papers checked

The [method preprint](https://arxiv.org/html/2501.18018v2) specifies local correlation training with the downstream gradient path through dendrites suppressed, followed by frozen dendrites and ordinary neuron optimization. It evaluates several tasks and includes a parameter-controlled mTAN width study. Its larger mTAN models rejected the first dendrite; a smaller model incurred substantially longer training. These results motivate a capacity-limitation hypothesis, rather than universal improvement.

The [follow-up experiments paper](https://arxiv.org/pdf/2506.00356) reports BERT, protein, and MobileNet experiments. It explicitly lacks repeated-run error bars because it arose from a hackathon. Its embedding-heavy model comparisons can have low dendrite overhead because embeddings were excluded. That cost distribution is quite different from copying most of a tiny convolutional network.

The [KWS preprint](https://arxiv.org/html/2605.15647v1) reports an 800-trial architecture and hyperparameter search using Edge Impulse's tutorial pipeline: approximately 93.3% test at 1,556 parameters versus a 92.1% baseline with 3,859 parameters. Selected endpoints of a large search are not fixed-recipe effect estimates. The current repository's SC2 frontend, balancing, class task, and architecture controls need their own comparison.

The [official repository](https://github.com/PerforatedAI/PerforatedAI) distinguishes the full PB system behind its headline results from the open-source wrapper. This review classifies actual per-run PB/GD settings and physical architectures, instead of assuming the installed package or a dendrite-named directory identifies the mechanism.

The supplied [pruning paper](../../53132_Pruning_Then_Perforating.pdf) is read in full and discussed in the [main synthesis](../DENDRITE_IMPACT_RESEARCH_2026-10-03.md). Its attractive classifier-only cost on ResNet/Pets must be compared with the much larger relative classifier cost on SparkNet before transferring its conclusions.
