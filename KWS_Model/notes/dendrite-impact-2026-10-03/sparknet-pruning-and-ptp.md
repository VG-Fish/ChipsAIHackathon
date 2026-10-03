# SparkNet pruning and Pruning Then Perforating audit — 2026-10-03

This is an evidence notebook being written during a fresh audit. Scope: all artifacts in `outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3`, `sparknet-c12-dendritic-prune-no-kd-unlimited`, `sparknet-c16-dendritic-prune-no-kd-fc-only-d3`, `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3`, `sparknet-c16-ptp`, `sparknet-c16g16-ptp`, and `sparknet-c18g8-ptp` (all paths relative to `KWS_Model/`). Historical saved configs and reports take precedence over the currently edited training code when reconstructing experiments.

## Audit approach and early cautions

- Read the repo `AGENTS.md`, both READMEs, the PerforatedAI analysis skill, previous project notes, and the supplied Pruning Then Perforating PDF (parent-provided text extraction).
- Do not interpret run names or `max_dendrites` as proof that the final architecture contains dendrites. Reconcile deployed parameter count, saved PAI architecture scores, switches, config snapshots, and final reports.
- Separate initial pruned accuracy, conventional fine-tuning, PAI phase, and any subsequent supervised resume. A PAI phase can train the unexpanded backbone before adding branches; its entire gain is not a clean causal estimate for dendrites.
- Examine incomplete artifacts and rejected candidates; do not summarize only completed successes.
- All experiments assigned here are intended to be no-KD runs; saved snapshots will verify this.
- Parameter/MAC reductions must compare the final deployed graph to its relevant trained parent. A branch adds capacity to a smaller base; it does not prune itself.

## Work log

- Created the notebook before extracting results. Memorable authentication/consent and recall were handled by the parent agent: configured, read-write consent, no matching procedure.

## First verified inventory and protocol findings

A read-only parser (`pruning_audit.py`) enumerated report rows, all phase JSONL, canonical architecture/PB/switch CSVs, manifests, and input hashes. Machine-readable evidence is `pruning_inventory.json` and `pruning_candidates.csv` beside this notebook.

- 28 actual run roots: 2 earlier C12 runs, 5 earlier C16 fc runs, 5 earlier C16 gate-conv runs, 6 C16 PTP runs, 5 C16g16 PTP runs, and 5 C18g8 PTP runs. Top-level `seedN.log` files are logs, not extra runs.
- C12 fc: run remains interrupted/running, C10 completed; C8 has 40 pruning-FT plus 132 PAI epochs but no final result; C6 has no epoch artifacts. C12 unlimited: C10 has 40 FT plus 1,788 PAI epoch records but no final clean result; remaining widths not started.
- Earlier C16 fc: all 15 candidate rows complete. C16 gate-conv: 15 planned rows, only C12 started; seeds0–2 have 537–557 PAI epochs; seeds3–4 stop during initial FT after 6/1 epochs. Do not call these completed placement comparisons.
- Completed PTP: C16 has 42 candidate rows (7 rates × 6 seeds), C16g16 and C18g8 each have 30 (6 rates × 5 seeds). All use `.fc` only, up to 3 retained classifier dendrites, sigmoid, history lookback1, 10-epoch switch window, three tries, AdamW at 5e-4, validation plateau scheduling, no label smoothing, CE scale100, and no post-PAI resume. Unlike earlier one-offs, PTP logs TEST and clean eval-mode TRAIN every PAI epoch. Selection uses validation. This is an extension of the supplied paper's vision protocol to SparkNet KWS, not an exact replication of its ResNet task.
- **Accounting finding requiring corrected analysis:** `scripts/analyze_ptp.py` places a selected epoch at `pruned_params + evaluated_parameter_count - first_epoch_count`. PAI-native counts add one copied classifier (396 at gate32; 204 at gate16; 108 at gate8), while the exported graph also carries trainable combination scales. Reported clean costs add 408/216/120 for one dendrite and 1,260/684/396 for three. Thus the script's advertised “true parameter count” undercounts deployed parameters by 12 for one, 36 for two, and 72 for three. Use final clean costs when making deployment/Pareto claims. Detailed correction and recomputed interpolated gains follow.
- Earlier C16 directory labels have distinct source checkpoints but their epoch logs record `seed: 0` even for `seed1`. This is not proof of duplicate results: it indicates the continuation RNG was not swept with the directory/source seed. Exact hash and configuration reconciliation is in progress.

## How the supplied paper frames the claim

The supplied `53132_Pruning_Then_Perforating.pdf` is an anonymous ICLR2027 submission marked under review. Its Study A uses half-width ImageNet-pretrained ResNet18 on Oxford-IIIT Pets, group-L2 structured pruning, six seeds × nine prune rates, classifier-only dendrites, and best-validation selection within each dendrite budget0–3. It claims +1.15pp mean test gain over an interpolated, parameter-matched within-run zero-dendrite reference; Study B (Taylor pruning) claims +1.60pp; production segmentation is observational without intervals. It acknowledges dendrites add parameters and PAI retrains the base before growing branches.

The relevant transfer question is whether a similarly parameter-matched gain survives on SparkNet's extremely small head. Its gate remains fixed when the backbone is pruned, so classifier cost does not shrink with pruning. The supplied paper's head does shrink with channel pruning; its reported budget3 parameter overhead is 0.8–10.4%. This architectural difference can make SparkNet's raw recovery positive while parameter-matched recovery remains negative.

A within-run zero-dendrite row controls initial pre-growth retraining only partially: later neuron training, optimizer restarts, candidate retries, variable epoch budgets, and base BatchNorm updates remain intertwined with growth. It should be described as the paper's reference construction, not an independent time-matched causal control.

## Earlier C16 prune/FC runs: five sources, one continuation RNG

The five source checkpoints are distinct paper-replication seeds0–4, whose recorded source validation averages 94.974% ± 0.283pp at 4,636 parameters / 396,304 MACs. The pruning, PAI, and resumed-supervised epochs all log `seed: 0` in each directory. A safe `torch.load(..., weights_only=True)` inspection also confirms checkpoint `seed: 0` in seed1. These are five independent source initializations, **not five independently seeded continuations**. No two candidate pruning JSONLs, PAI JSONLs, or final clean artifacts in the assigned inventory share an SHA256 digest.

| Pruned width | Prune FT val mean ± sd | PAI zero-row val | PAI best val | Final val mean ± sd | Final−prune FT | PAI best−zero row | Resume contribution | Deployed params mean (range) | Retained counts across 5 sources |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C12 | 90.146 ± 0.683% | 92.418% | 92.612% | 92.648 ± 0.317% | +2.502pp | +0.193pp | +0.036pp | 4,060 (3,808–4,228) | d1×2, d2×3 |
| C10 | 85.791 ± 1.370% | 90.533% | 90.880% | 90.979 ± 0.265% | +5.188pp | +0.346pp | +0.099pp | 3,518.8 (2,854–4,114) | d0×1, d1×1, d2×2, d3×1 |
| C8 | 79.928 ± 1.983% | 87.163% | 87.825% | 88.045 ± 0.625% | +8.117pp | +0.661pp | +0.220pp | 3,188.8 (2,764–3,616) | d1×2, d2×1, d3×2 |

Sample standard deviations above describe differences between source models. Gains are percentage points, not relative percentages. Each source-width has 40 FT epochs, 520–854 PAI epochs, and 8 resume epochs. Larger gains versus the short pruned fine-tune are largely reproduced by ordinary zero-dendrite training inside PAI: +2.272/+4.742/+7.235pp at C12/C10/C8. This is the main confound in the old +2.5/+5.2/+8.1pp recovery headline.

The C10 seed0 final contains **zero dendrites** at exactly 2,854 params/224,806 MACs, yet rises from 86.884% pruning-FT to 91.249% final val. Its PAI zero/best row are both91.159%, and supervised resume adds0.090pp. Thus a substantial “dendritic pipeline” improvement can occur with no retained dendrite.

These final models all trail the existing same-width gate32 scratch means (C12 93.88%, C10 92.80%, C8 91.23%) while usually spending more parameters. Direct exact-pair checks follow. No test accuracy is stored for these old candidate runs.

Every old run's current experiment/train config hashes differ from its manifest; data/model/source hashes still match. The current configs describe later v2 recipes and must not be used to reconstruct the historical d3 setup. Old reports/native snapshots specify tanh, history8, n_epochs_to_switch25, threshold `[0.001,0.0001,0]`, init0.01, 40 correlation batches, three tries, LR restart multiplier0.25, and frozen base BN during p mode. PTP switches to the documented-paper sigmoid/history1/plateau recipe and has independent continuation seeds.

## PTP cost correction and source-specific comparators

A safe tensor inspection of C16 PTP seed0 C13 final (`pai/candidates/sparknet_c13_multilayer/final_clean_pai.pt`) finds four `fc.layer_array` branches: one base and three copied classifiers, each `[12,32]` weight plus12 biases. `fc.skip_weights.0/.1/.2` have shapes `[1,12]`, `[2,12]`, `[3,12]`, totaling72 learned scale coefficients. This proves the clean-graph overhead rather than inferring it from the library integration counter.

For D retained classifier dendrites and G gate channels, the native copied weight/bias overhead is `D*(12*G+12)`. The clean overhead is that plus `12*D*(D+1)/2`. This triangular formula yields exactly the reports' D1/D2/D3 costs; older notes' `D²*C` formula is inconsistent with these artifacts.

Recomputed PTP comparisons place every validation-selected epoch at that clean cost. The paper's zero-dendrite reference uses seed means of budget0 TEST accuracy at each pruned width, linearly interpolates in log10(params), and extends flat above the least-pruned point. At the lightest prune rate this flat extension assigns zero parameter penalty, so the resulting tiny positive gain is **extrapolated**, not a measured same-cost advantage. Strict comparisons will mark these outside-range cases unavailable.

The PTP pipeline evaluates test and clean train each epoch only for logging; code feeds validation alone to PAI, plateau scheduling, and checkpoint selection. The offline analysis selects the highest logged validation epoch within budgets0–3. The deployed report independently reads canonical PAI architecture scores and exports its selected checkpoint. These choices do not always agree: **14/102 cells differ on val or retained cost**, including one equal-val tie carrying different branch count. Test results in the analysis therefore describe logged budget-selected epochs; they are not verified final-clean-checkpoint test evaluations.

`build_datasets(seed=train_cfg.seed)` also samples pooled unknown clips and materialized silence separately for each seed. Within a PTP seed, budget0/final share the cohort. Across seeds, all target-keyword recordings share official split lists but unknown/silence subsets differ. The older seed0 continuations additionally make source-seed validation deltas imperfectly paired. Historical fixed-eval-seed0 test baselines are not a perfectly identical cohort for every PTP seed.

## Completed PTP: raw gain versus the parameter bill

These tables summarize budget3 epochs selected offline on validation with corrected clean costs. `Raw` is budget3 TEST minus budget0 TEST; `Cost` is the zero-curve accuracy increase associated with spending those added parameters; `Matched` is TEST minus the zero-curve at the final clean count. All differences are pp. Entries with `strict n=0` are wholly above the observed zero-curve domain and use the paper's flat extrapolation. Test cohorts are seed-specific. No unpruned parent is included in Z by the analysis definition.


### sparknet-c16-ptp

| Target prune | Width | Base params | Selected clean params mean (range) | B0 test % | B3 test mean ± sd % | Raw pp | Cost pp | Matched pp | Strict n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | C15 | 4309 | 5073.0 (4309–5569) | 94.206 | 94.250 ± 0.302 | +0.044 | +0.000 | +0.044 | 1/6 |
| 20% | C13 | 3691 | 4735.0 (4519–4951) | 92.972 | 93.098 ± 0.550 | +0.126 | +1.234 | -1.108 | 0/6 |
| 30% | C11 | 3121 | 4381.0 (4381–4381) | 90.832 | 91.350 ± 0.620 | +0.518 | +3.374 | -2.856 | 0/6 |
| 40% | C10 | 2854 | 4114.0 (4114–4114) | 89.492 | 90.092 ± 0.586 | +0.600 | +4.345 | -3.745 | 6/6 |
| 50% | C8 | 2356 | 3544.0 (3184–3616) | 85.061 | 86.322 ± 0.836 | +1.261 | +7.378 | -6.117 | 6/6 |
| 60% | C6 | 1906 | 3166.0 (3166–3166) | 80.055 | 81.057 ± 1.951 | +1.002 | +10.960 | -9.958 | 6/6 |
| 70% | C3 | 1321 | 2509.0 (2149–2581) | 52.795 | 58.337 ± 5.152 | +5.542 | +33.661 | -28.119 | 6/6 |

The final report and offline budget selection disagree in some cells; reported final architecture costs are in `pruning_candidates.csv`, selected epoch costs in `pruning_summaries.json`.


### sparknet-c16g16-ptp

| Target prune | Width | Base params | Selected clean params mean (range) | B0 test % | B3 test mean ± sd % | Raw pp | Cost pp | Matched pp | Strict n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | C15 | 3829 | 4136.2 (4045–4273) | 93.988 | 94.020 ± 0.224 | +0.033 | +0.000 | +0.033 | 0/5 |
| 20% | C13 | 3243 | 3927.0 (3927–3927) | 92.094 | 92.446 ± 0.275 | +0.352 | +1.894 | -1.542 | 0/5 |
| 30% | C12 | 2968 | 3652.0 (3652–3652) | 90.638 | 91.121 ± 0.779 | +0.483 | +2.810 | -2.328 | 5/5 |
| 40% | C10 | 2454 | 3090.0 (2898–3138) | 88.200 | 88.744 ± 0.576 | +0.544 | +3.109 | -2.565 | 5/5 |
| 50% | C8 | 1988 | 2672.0 (2672–2672) | 84.434 | 85.706 ± 0.880 | +1.272 | +4.858 | -3.586 | 5/5 |
| 60% | C6 | 1570 | 2254.0 (2254–2254) | 77.575 | 79.014 ± 0.591 | +1.440 | +9.105 | -7.665 | 5/5 |

The final report and offline budget selection disagree in some cells; reported final architecture costs are in `pruning_candidates.csv`, selected epoch costs in `pruning_summaries.json`.


### sparknet-c18g8-ptp

| Target prune | Width | Base params | Selected clean params mean (range) | B0 test % | B3 test mean ± sd % | Raw pp | Cost pp | Matched pp | Strict n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | C17 | 4207 | 4406.2 (4327–4459) | 94.237 | 94.258 ± 0.396 | +0.020 | +0.000 | +0.020 | 0/5 |
| 20% | C15 | 3589 | 3927.4 (3841–3985) | 92.773 | 92.990 ± 0.229 | +0.217 | +0.829 | -0.612 | 5/5 |
| 30% | C14 | 3298 | 3694.0 (3694–3694) | 92.012 | 92.217 ± 0.472 | +0.204 | +1.026 | -0.822 | 5/5 |
| 40% | C12 | 2752 | 3119.2 (3004–3148) | 90.229 | 90.544 ± 0.676 | +0.315 | +1.232 | -0.917 | 5/5 |
| 50% | C10 | 2254 | 2621.2 (2506–2650) | 87.018 | 87.890 ± 1.001 | +0.871 | +2.423 | -1.552 | 5/5 |
| 60% | C8 | 1804 | 2171.2 (2056–2200) | 81.198 | 82.601 ± 1.440 | +1.403 | +4.832 | -3.430 | 5/5 |

The final report and offline budget selection disagree in some cells; reported final architecture costs are in `pruning_candidates.csv`, selected epoch costs in `pruning_summaries.json`.


### Uncertainty using independent parent seeds

Rate rows from the same source seed are correlated, so the pooled t intervals over42/30/30 rows from `analyze_ptp.py` are not used as independent evidence. The next table first averages rates within each source seed, then computes a95% t interval across6/5/5 seed units. The bootstrap resamples **entire source seeds**, refits the seed-mean zero curve in each replicate, and recomputes the overall gain10,000 times; its interval also reflects uncertainty in Z. Small seed counts and a coarse interpolated curve limit precision.

| Source | Seed units | Mean gain pp with flat extrapolation | Seed-mean 95% t CI | Refitted-Z bootstrap 95% CI | Strict common-rate mean pp | Strict common rates | Strict refitted-Z bootstrap CI |
|---|---:|---:|---|---|---:|---|---|
| sparknet-c16-ptp | 6 | -7.408 | [-8.703, -6.114] | [-8.280, -6.652] | -11.985 | 40%, 50%, 60%, 70% | [-13.374, -10.727] |
| sparknet-c16g16-ptp | 5 | -2.942 | [-3.347, -2.537] | [-3.144, -2.768] | -4.036 | 30%, 40%, 50%, 60% | [-4.274, -3.822] |
| sparknet-c18g8-ptp | 5 | -1.219 | [-1.749, -0.688] | [-1.403, -1.008] | -1.467 | 20%, 30%, 40%, 50%, 60% | [-1.708, -1.207] |

At C16 the severe C3 pruning cases account for much of the large pooled deficit; do not generalize−7.41pp to a typical light-pruning candidate. The C3 pruning-FT validation mean is42.07% ±22.78pp; seed3 stops after9FT epochs at7.58%, seed4 after12 at18.99%. Ordinary pre-dendrite PAI retraining raises them to54.11% and32.98%. These failures stay in the inventory and comparison rather than being censored. For common in-range C16 rates40–70%, average clean-cost-matched deficit is−11.985pp, because the added classifier budget crosses the steep ordinary pruning curve.


### Comparison with scratch models and conventional parents

Validation comparisons below use currently completed same-gate scratch runs. They are independent architecture controls, not perfectly time-matched continued-training controls. Missing exact widths use an explicitly labeled log-parameter interpolation of the measured same-gate mean scratch curve. Some curve widths have3 rather than5 seeds; exact run lists and seed counts are stored in `pruning_extended.json`. No g16/g8 scratch test curve is available, so these comparisons are validation only.

| Source | Width | Final report−same-width, same-seed scratch val pp | Final report−same-gate scratch parameter curve val pp |
|---|---:|---:|---:|
| sparknet-c16-ptp | C15 | unavailable | -0.587 |
| sparknet-c16-ptp | C13 | unavailable | -1.533 |
| sparknet-c16-ptp | C11 | unavailable | -2.906 |
| sparknet-c16-ptp | C10 | -2.286 (n=5) | -3.830 |
| sparknet-c16-ptp | C8 | -4.472 (n=5) | -7.018 |
| sparknet-c16-ptp | C6 | -6.718 (n=5) | -11.968 |
| sparknet-c16-ptp | C3 | unavailable | -31.971 |
| sparknet-c16g16-ptp | C15 | unavailable | -0.522 |
| sparknet-c16g16-ptp | C13 | -0.828 (n=5) | -1.603 |
| sparknet-c16g16-ptp | C12 | -1.134 (n=5) | -2.087 |
| sparknet-c16g16-ptp | C10 | -2.820 (n=3) | -4.083 |
| sparknet-c16g16-ptp | C8 | -3.744 (n=5) | -6.069 |
| sparknet-c16g16-ptp | C6 | -6.802 (n=3) | -11.388 |
| sparknet-c18g8-ptp | C17 | +0.037 (n=3) | -0.135 |
| sparknet-c18g8-ptp | C15 | unavailable | -0.687 |
| sparknet-c18g8-ptp | C14 | unavailable | -0.988 |
| sparknet-c18g8-ptp | C12 | unavailable | -2.064 |
| sparknet-c18g8-ptp | C10 | -3.525 (n=3) | -4.045 |
| sparknet-c18g8-ptp | C8 | -5.924 (n=3) | -8.058 |

Same-gate conventional scratch mean rows dominate the final **individual** report candidate in accuracy/parameters for40/42 C16,28/30 C16g16,17/30 C18g8 cases; requiring no greater MAC count reduces these counts to35/42,26/30,16/30. These are descriptive comparisons to an architecture mean, not paired statistical tests, and favorable seeds should not be mistaken for a method-level frontier. Parent-level examples and exact nondominated candidates follow.

## Final export versus candidate-training snapshots

The discrepancy has a specific cause: **all14 discrepant budget3 selections are p-mode epochs**, while all88 selections that agree with export are n-mode epochs. There are13 different validation values and9 different parameter counts (union14). `analyze_ptp.py` currently searches every epoch; the runner's canonical architecture CSV records neuron-mode maxima and exports that architecture. During p mode the current candidate is being trained and ordinary base weights are absent from the optimizer; PTP does not freeze BatchNorm running statistics. A transient p-mode snapshot is therefore not interchangeable with the retained exported model.

A read-only sensitivity recomputes all four budgets using n-mode epochs, assigns the corrected clean cost, and refits the zero reference. It matches101/102 exports. The exceptional C18g8 seed0 C17 records a higher zero-dendrite score at the n→p switch boundary (epoch12) that is omitted from its canonical architecture row. A stronger reconstruction maps **each canonical CSV row uniquely to an n-mode epoch using native count + validation + recorded TRAIN accuracy**, then selects budgets only among those rows. All102 reconstructed budget3 costs and validation scores now match final reports exactly. The mapped TEST scores reconstruct the selected architecture epoch; final-clean test inference has not been rerun. Detailed unique epoch mappings and both sensitivities are in `pruning_neuron_mode.json`.

| Source | N-only mean matched TEST pp | Seed-clustered 95% t CI | All-epoch mean pp |
|---|---:|---|---:|
| sparknet-c16-ptp | -7.514 | [-8.659, -6.369] | -7.408 |
| sparknet-c16g16-ptp | -2.945 | [-3.343, -2.546] | -2.942 |
| sparknet-c18g8-ptp | -1.309 | [-1.943, -0.675] | -1.219 |

The earlier per-rate tables and bootstrap retain the historical **all-epoch** analysis definition. The n-only check changes small numbers and resolves export identity; it preserves the unfavorable same-cost conclusion. No global zero-pruning source checkpoint is added to either Z curve.

### Canonical export-aligned TEST reconstruction (preferred evidence)

| Source | Mean matched TEST pp | Seed-clustered t CI | Refitted-Z bootstrap CI | Strict common mean pp | Strict common rates | Strict bootstrap CI |
|---|---:|---|---|---:|---|---|
| sparknet-c16-ptp | -7.497 | [-8.642, -6.352] | [-8.257, -6.885] | -12.189 | 40%, 50%, 60%, 70% | [-13.379, -11.203] |
| sparknet-c16g16-ptp | -2.924 | [-3.325, -2.523] | [-3.133, -2.753] | -4.028 | 30%, 40%, 50%, 60% | [-4.275, -3.834] |
| sparknet-c18g8-ptp | -1.228 | [-1.840, -0.615] | [-1.343, -1.124] | -1.476 | 20%, 30%, 40%, 50%, 60% | [-1.646, -1.337] |

#### sparknet-c16-ptp canonical budgets

| Prune / width | B0 params | Final clean mean (range) | B0 test % | Final mapped test mean ± SD % | Raw pp | Cost pp | Matched pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% / C15 | 4309 | 5145.0 (4309–5569) | 94.151 | 94.260 ± 0.292 | +0.109 | +0.000 | +0.109 |
| 20% / C13 | 3691 | 4807.0 (4519–4951) | 92.900 | 93.119 ± 0.541 | +0.218 | +1.251 | -1.033 |
| 30% / C11 | 3121 | 4381.0 (4381–4381) | 90.855 | 91.350 ± 0.620 | +0.494 | +3.296 | -2.802 |
| 40% / C10 | 2854 | 4114.0 (4114–4114) | 89.485 | 90.092 ± 0.586 | +0.607 | +4.292 | -3.685 |
| 50% / C8 | 2356 | 3616.0 (3616–3616) | 85.078 | 86.329 ± 0.843 | +1.251 | +7.572 | -6.321 |
| 60% / C6 | 1906 | 3166.0 (3166–3166) | 79.976 | 81.057 ± 1.951 | +1.080 | +11.054 | -9.973 |
| 70% / C3 | 1321 | 2581.0 (2581–2581) | 52.522 | 58.398 ± 5.191 | +5.876 | +34.652 | -28.777 |

#### sparknet-c16g16-ptp canonical budgets

| Prune / width | B0 params | Final clean mean (range) | B0 test % | Final mapped test mean ± SD % | Raw pp | Cost pp | Matched pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% / C15 | 3829 | 4229.8 (4045–4513) | 93.988 | 94.098 ± 0.186 | +0.110 | +0.000 | +0.110 |
| 20% / C13 | 3243 | 3927.0 (3927–3927) | 92.049 | 92.446 ± 0.275 | +0.397 | +1.939 | -1.542 |
| 30% / C12 | 2968 | 3652.0 (3652–3652) | 90.491 | 91.121 ± 0.779 | +0.630 | +2.945 | -2.315 |
| 40% / C10 | 2454 | 3090.0 (2898–3138) | 88.278 | 88.695 ± 0.613 | +0.417 | +2.941 | -2.524 |
| 50% / C8 | 1988 | 2672.0 (2672–2672) | 84.429 | 85.706 ± 0.880 | +1.276 | +4.839 | -3.563 |
| 60% / C6 | 1570 | 2254.0 (2254–2254) | 77.436 | 79.014 ± 0.591 | +1.579 | +9.289 | -7.710 |

#### sparknet-c18g8-ptp canonical budgets

| Prune / width | B0 params | Final clean mean (range) | B0 test % | Final mapped test mean ± SD % | Raw pp | Cost pp | Matched pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% / C17 | 4207 | 4461.4 (4327–4603) | 94.217 | 94.229 ± 0.365 | +0.012 | +0.000 | +0.012 |
| 20% / C15 | 3589 | 3927.4 (3841–3985) | 92.769 | 93.018 ± 0.255 | +0.249 | +0.820 | -0.570 |
| 30% / C14 | 3298 | 3694.0 (3694–3694) | 91.984 | 92.217 ± 0.472 | +0.233 | +1.048 | -0.815 |
| 40% / C12 | 2752 | 3119.2 (3004–3148) | 90.016 | 90.503 ± 0.683 | +0.487 | +1.360 | -0.873 |
| 50% / C10 | 2254 | 2650.0 (2650–2650) | 86.789 | 87.845 ± 1.043 | +1.055 | +2.616 | -1.561 |
| 60% / C8 | 1804 | 2200.0 (2200–2200) | 81.166 | 82.618 ± 1.444 | +1.452 | +5.011 | -3.559 |
## Complete disposition of the 28 run roots

Folders named `seedN.log` are launcher logs, not runs. A report may say running after interruption; the artifact disposition below takes precedence. The138 serialized/planned candidate rows contain118 final clean exports. Gate-conv manifests plan candidates without serializing completed result rows. No incomplete arm is treated as an accuracy failure or silently dropped from this inventory.

| Run root under outputs/ | Report / manifest | Actual result disposition |
|---|---|---|
| `sparknet-c12-dendritic-prune-no-kd-fc-only-d3` | running / interrupted | 1/3 final clean exports (C10); partial PAI C8; remaining candidates uncompleted |
| `sparknet-c12-dendritic-prune-no-kd-unlimited` | running / running | 0/3 final clean exports; partial PAI C10; remaining candidates uncompleted |
| `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed0` | complete / completed | 3/3 final clean exports (C8, C10, C12) |
| `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed1` | complete / completed | 3/3 final clean exports (C8, C10, C12) |
| `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed2` | complete / completed | 3/3 final clean exports (C8, C10, C12) |
| `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed3` | complete / completed | 3/3 final clean exports (C8, C10, C12) |
| `sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed4` | complete / completed | 3/3 final clean exports (C8, C10, C12) |
| `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed0` | running / running | C12 PAI partial; no final clean export; later widths not completed |
| `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed1` | running / running | C12 PAI partial; no final clean export; later widths not completed |
| `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed2` | running / running | C12 PAI partial; no final clean export; later widths not completed |
| `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed3` | running / running | C12 pruning FT only (6 epochs); no PAI or final export |
| `sparknet-c16-dendritic-prune-no-kd-gate-conv-d3/seed4` | running / running | C12 pruning FT only (1 epochs); no PAI or final export |
| `sparknet-c16-ptp/seed0` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16-ptp/seed1` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16-ptp/seed2` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16-ptp/seed3` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16-ptp/seed4` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16-ptp/seed5` | complete / completed | 7/7 final clean exports (C3, C6, C8, C10, C11, C13, C15) |
| `sparknet-c16g16-ptp/seed0` | complete / completed | 6/6 final clean exports (C6, C8, C10, C12, C13, C15) |
| `sparknet-c16g16-ptp/seed1` | complete / completed | 6/6 final clean exports (C6, C8, C10, C12, C13, C15) |
| `sparknet-c16g16-ptp/seed2` | complete / completed | 6/6 final clean exports (C6, C8, C10, C12, C13, C15) |
| `sparknet-c16g16-ptp/seed3` | complete / completed | 6/6 final clean exports (C6, C8, C10, C12, C13, C15) |
| `sparknet-c16g16-ptp/seed4` | complete / completed | 6/6 final clean exports (C6, C8, C10, C12, C13, C15) |
| `sparknet-c18g8-ptp/seed0` | complete / completed | 6/6 final clean exports (C8, C10, C12, C14, C15, C17) |
| `sparknet-c18g8-ptp/seed1` | complete / completed | 6/6 final clean exports (C8, C10, C12, C14, C15, C17) |
| `sparknet-c18g8-ptp/seed2` | complete / completed | 6/6 final clean exports (C8, C10, C12, C14, C15, C17) |
| `sparknet-c18g8-ptp/seed3` | complete / completed | 6/6 final clean exports (C8, C10, C12, C14, C15, C17) |
| `sparknet-c18g8-ptp/seed4` | complete / completed | 6/6 final clean exports (C8, C10, C12, C14, C15, C17) |

### What the partial runs establish

- C12 FC-only: C10 completed at92.3240% validation,4114 clean parameters, three retained dendrites. C8 has40 pruning-FT epochs and132 PAI epochs, still at its zero-dendrite native count; no final export exists. The source C12 is92.6133% at3400 parameters, so the completed C10 candidate is larger and less accurate than its parent.
- C12 unlimited: C10 has1788 unique PAI epochs and native architecture counts2854 through13078 in eight1278-parameter increments. Canonical validation rises from91.5718% at the zero row to93.2305% at the largest native architecture. That is+0.6172pp above its source validation, using at least3.85× the parent native parameter count; exact clean cost, retained final architecture and final test are unavailable. This is a partial high-compute observation, not a completed frontier result.
- C16 gate-conv: seeds0–2 have557/552/537 PAI epochs on C12. Canonical zero-to-best improvements are+0.2475/+0.3600/+0.3825pp; native counts3400/3816/4232/4648 show attempted growth up to three copies. Seeds3–4 stop during pruning fine-tuning. Every seed lacks a final clean export; no placement-level comparison to completed FC arms is valid.

## Parent-relative deployment tradeoffs

These comparisons use the **final reports**, not offline snapshots. Positive size reductions mean smaller. Every completed PTP width has lower mean validation than its own unpruned source. More capacity and more training recover some post-pruning accuracy, but recovery is not preservation of the original model. Gate-width families change both encoder width and gate width, pruning criterion/recipe families also differ, and this is not a randomized placement study.

| Source family | Final width | Final−source val pp | Mean parameter reduction | Mean MAC reduction |
|---|---:|---:|---:|---:|
| sparknet-c16-ptp | C15 | -0.585 | -10.98% | +7.77% |
| sparknet-c16-ptp | C13 | -1.646 | -3.69% | +22.74% |
| sparknet-c16-ptp | C11 | -3.270 | +5.50% | +36.52% |
| sparknet-c16-ptp | C10 | -4.364 | +11.26% | +42.97% |
| sparknet-c16-ptp | C8 | -7.904 | +22.00% | +54.94% |
| sparknet-c16-ptp | C6 | -13.461 | +31.71% | +65.70% |
| sparknet-c16-ptp | C3 | -34.927 | +44.33% | +79.54% |
| sparknet-c16g16-ptp | C15 | -0.580 | -2.17% | +8.00% |
| sparknet-c16g16-ptp | C13 | -1.867 | +5.14% | +23.15% |
| sparknet-c16g16-ptp | C12 | -2.713 | +11.79% | +30.27% |
| sparknet-c16g16-ptp | C10 | -5.417 | +25.36% | +43.54% |
| sparknet-c16g16-ptp | C8 | -8.229 | +35.46% | +55.47% |
| sparknet-c16g16-ptp | C6 | -14.898 | +45.56% | +66.11% |
| sparknet-c18g8-ptp | C17 | -0.220 | +1.60% | +7.49% |
| sparknet-c18g8-ptp | C15 | -1.359 | +13.38% | +21.68% |
| sparknet-c18g8-ptp | C14 | -1.962 | +18.53% | +28.34% |
| sparknet-c18g8-ptp | C12 | -3.874 | +31.20% | +40.83% |
| sparknet-c18g8-ptp | C10 | -6.659 | +41.55% | +52.14% |
| sparknet-c18g8-ptp | C8 | -11.708 | +51.48% | +62.31% |

The unpruned C16/g16 parent is itself a strong conventional option:94.9966% ±0.3067pp validation,4140 parameters,370256 MACs, versus C16/g32 at94.9306% ±0.2743pp,4636 parameters,396304 MACs and C18/g8 at94.9921% ±0.3595pp,4534 parameters,419246 MACs. Its mean accuracy difference is within seed variation, while its parameter and MAC advantage is exact. Thus a comparison restricted to one original gate32 parent misses a conventional architectural improvement available before dendrites.

### Best individual completed candidates, with exact evidence paths

These are family-specific maximum final validation values and are **selected seed envelopes**, not matched-seed mean wins. Report and checkpoint paths are relative to `KWS_Model/`.

| Family / seed / width | Final val % | Parameters | MACs | Retained D | Evidence |
|---|---:|---:|---:|---:|---|
| sparknet-c12-dendritic-prune-no-kd-fc-only-d3 / None / C10 | 92.3240 | 4114 | 225646 | 3.0 | report `outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3/reports/sparknet_dendritic_prune_experiment.yaml`; checkpoint `outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3/pai/candidates/sparknet_c10_multilayer/final_clean_pai.pt` |
| sparknet-c16-dendritic-prune-no-kd-fc-only-d3 / 4 / C12 | 93.1159 | 3808 | 277520 | 1.0 | report `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed4/reports/sparknet_dendritic_prune_experiment.yaml`; checkpoint `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed4/pai/candidates/sparknet_c12_multilayer/final_clean_pai.pt` |
| sparknet-c16-ptp / 3 / C15 | 95.0281 | 5569 | 365915 | 3.0 | report `outputs/sparknet-c16-ptp/seed3/reports/sparknet_dendritic_prune_experiment.yaml`; checkpoint `outputs/sparknet-c16-ptp/seed3/pai/candidates/sparknet_c15_multilayer/final_clean_pai.pt` |
| sparknet-c16g16-ptp / 3 / C15 | 94.7357 | 4045 | 340463 | 1.0 | report `outputs/sparknet-c16g16-ptp/seed3/reports/sparknet_dendritic_prune_experiment.yaml`; checkpoint `outputs/sparknet-c16g16-ptp/seed3/pai/candidates/sparknet_c15_multilayer/final_clean_pai.pt` |
| sparknet-c18g8-ptp / 0 / C17 | 95.0506 | 4459 | 387861 | 2.0 | report `outputs/sparknet-c18g8-ptp/seed0/reports/sparknet_dendritic_prune_experiment.yaml`; checkpoint `outputs/sparknet-c18g8-ptp/seed0/pai/candidates/sparknet_c17_multilayer/final_clean_pai.pt` |

## Accounting, controls, and limits

- The completed118 clean checkpoints were inspected using safetensors without importing PerforatedAI. Every clean branch/skip parameter overhead matches its report. All42 C16,30 g16 and30 g8 PTP cells completed; the only PTP zero-dendrite final is C16 seed1 C15. A budget of three means at most three, not exactly three.
- Canonical `noImprove_lr` artifacts record failed retries, not whole-run zero retention. Thirteen of15 completed historical C16 FC cells have these artifacts although14 retain dendrites. Every placement claim must inspect final branch structure or clean overhead. PB correlation magnitudes describe candidate fit, and no fixed threshold or two-observation trend proves a useful placement.
- Recovery versus immediate pruning FT conflates a long ordinary zero-dendrite continuation with branch growth. Old C16 FC runs use40 FT +520–854 PAI +8 resume epochs; zero-phase recovery dominates. Their final branch-associated canonical best−zero changes are only+0.193/+0.346/+0.661pp at C12/C10/C8. A C10 seed0 zero-branch model recovers+4.364pp before resume. A matched ordinary continuation at identical optimizer, data cohort, wall time and checkpoint selection is absent.
- PTP uses validation for search, logs clean train and test every epoch, and has no post-PAI resume. Its zero reference has already received substantial ordinary PAI neuron training. It is a meaningful within-search zero control, although shorter than the full growing lifecycle. Held-out TEST is never an optimizer/PAI score input in current code; repeated observational access still differs from a one-time frozen test evaluation.
- No configured hardware budget is enforced in any assigned run. FC PTP head overhead is1260/684/396 clean parameters at D3 for gate32/16/8, a large fraction of these tiny models. The supplied paper's largest classification budget adds at most10.4%, under3% through70% pruning; that scale assumption does not transfer to SparkNet. The paper's use of a cheap head on larger models is relevant context, not positive evidence for these tiny models.
- Reports contain logical8-bit weight bytes, conservative activation accounting and host CPU latency. They are not board deployment measurements or evidence of quantized dendritic equivalence. The old singleton C12 FC reported MAC225646 is below the clean formula226030 for C10/D3/g32, a384-MAC missing base-branch discrepancy; preserve it as historical reported cost and avoid precise cross-family MAC attribution.
- Group pruning in PTP physically rebuilds a narrower dense network, manually propagating producer and consumer dependency groups with fixed gate width. Old pruning ranks producer L1 magnitudes; PTP uses dependency-group L2 scores. Target rates round to realizable widths and achieved rates differ. The severe C3 early-stopped fits remain included. No claim here treats masks, nominal requested rates, or stopped poor fits as equivalent achieved dense cost.

## Reading coverage and corrections to historical notes

Read the root AGENTS/README and KWS README; supplied Pruning Then Perforating text; `PROJECT_FINDINGS.md`; and `dendrite-study-v2/{DATA_INVENTORY,INTEGRATION_AUDIT,DSCNN_DENDRITIC_PIPELINES,agent-dscnn-pipeline-designs,ENHANCEMENTS,PICO_COMPRESSION_PIPELINE_JOURNAL,review-next-runs-evidence,review-next-runs-pai,agent-repo-results-audit}.md`. Parent and sibling agents cover core PAI theory, primary papers and the other output families. The historical notes are useful development records, not independently verified results.

Relevant lessons: an optimizer handoff can harm a previously good checkpoint; identity-prune controls expose that before any branch is added; sham switches preserve lifecycle while adding no branch and are stronger controls for switch effects; scratch equal-cost curves and actual no-branch controls answer different questions; class-output scales and direct module forward calls affect clean counting/profiling; normalized export equivalence should be measured rather than inferred; heldout performance, hardware cost and timing must be reported separately. The historical journal's proposal to apply a DSCNN prune/KD/resume recipe to SparkNet is motivation, not causal evidence.

Corrections made in this audit: current configs are not authoritative for historical saved manifests; native counts omit clean skip scales; the clean scale count is triangular, not D²×classes; noImprove files do not prove zero retained branches; a PB score threshold is not a performance guarantee; old C16 pruning directories do not provide five independent continuation seeds; all-epoch offline selection can choose candidate p-mode snapshots; and endpoint-flat extrapolation cannot establish a measured cost-matched win. Prior notes that predate late-September PTP cannot be treated as a complete current inventory.

## Audit completion

Read-only analyses completed with standard Python/YAML/scipy/safetensors. No training, licensed PAI import, original artifact mutation, or `.env` read was performed. The machine-readable inventory and all computations are retained beside this notebook. Tests of model behavior were not run because this is an artifact analysis; exact clean overheads were independently checked against every saved final tensor graph, and reconstructed canonical epoch costs/validation match all102 PTP final reports.
