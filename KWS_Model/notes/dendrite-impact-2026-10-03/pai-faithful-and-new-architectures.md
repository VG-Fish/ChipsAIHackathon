# PerforatedAI faithful-loop and newer architecture audit

Working notes, 2026-10-03. This is a new analysis of existing artifacts; no training is performed. Coverage assigned: every `outputs/pai-faithful*` run, `students-dnn`, `students-dscnn`, `students-dtnet`, `students-ei`, `students-ei-adamw`, `students-msd`, and `research-*` queue provenance.

## Completed audit

Final results and corrected live parameter counts are in the tables below. Earlier raw clean counts have been corrected because the faithful cleanup copy was not evaluated and loses required single-dendrite coefficients. Final coverage: 267 evaluated faithful roots, 24 partial roots, one explicit failure, and 211 completed native architecture roots.

## Evidence rules

- A readable `result.json` means the faithful driver reached final model selection and held-out test evaluation. `epoch_cap` is an evaluated cap, not convergence. `pai_training_complete` is the tracker completion criterion. Interrupted folders with epochs but no result are partial, regardless of a saved checkpoint.
- PAI's attempted/integrated counts are distinct from dendrites retained in the selected model. Use `structural_max_per_module` and recorded live unique parameter counts (`numel_best`) to describe the scored candidate. The driver's unverified cleaned copy drops single-dendrite coefficients; see the accounting correction below.
- Each run's test evaluation uses the fixed evaluation seed 0 and 4,890 examples. Budget snapshots reuse the same training trajectory and test set; they are not independent seeds.
- The faithful driver uses an Edge Impulse style 2D CNN rather than SparkNet. Its input keeps the first 13 MFCCs from the MFCC-32 data configuration, applies per-utterance CMVN, and adds Gaussian noise during training. Do not compare its accuracy directly to log-mel runs as an isolated architectural effect.
- Native manually constructed dendritic DTNet/MSD/EI/DNN/DS-CNN architectures are separate from PerforatedAI's adaptive dendrite training, even if their names contain dendrite shorthand.
- `notes/PROJECT_FINDINGS.md` was compiled 2026-09-23; its claim that the documented PAI recipe never won predates the September 29–October 2 faithful-loop experiments audited here.

## Initial census

Faithful directories: main 80/80 results; b21 85/85; b22 45/45; b23 25/25; b24 25/25. Aborted/interrupted sibling directories: f10-aborted 7 results, 3 partials and 1 failed PB run; b21-interrupted 6 partials; b22-interrupted 7; b23-interrupted 3; main-interrupted 5. Total 267 results and 24 interrupted/partial directories and one failed directory. These are artifact counts, not independent scientific replicates.

Newer architecture directories: DNN 110, DS-CNN 24, DTNet 30, EI 12, EI AdamW 5, MSD 30 (211 total). Completion and provenance are being checked.

`research-1003a/jobs.all.txt` queues a b25 expansion to seeds 5–9 plus a new four-layer c12x32x64x61 depth control. At the initial census no b25 output directory existed: this is a plan, not evidence.

## Updated interpretation of the positive faithful-loop evidence

The new faithful driver produced real replicated PAI improvements against its own smaller base, unlike the old SparkNet documented-loop generation. On the main MPS generation, two-layer `c8x16` PB/tanh reaches 82.937% test versus 80.000% for the same base, both five seeds. It increases scored live parameters from 21,228 to 42,492 (approximately doubles them). Plain two-layer `c16x32` reaches 82.654% at 44,748 parameters: the 2.94 pp base gain shrinks to 0.28 pp against width. PB's low-width `c4x8` reaches 72.798%, while GD/tanh reaches 72.814% at the same 20,688 parameters. Both exceed the 10,332-parameter base (67.963%), but the slightly larger plain `c8x16` reaches 80.000%. PB/ReLU at `c4x8` is weaker (69.804%) and rejects one selected dendrite.

The CPU b22 generation reveals a major depth confound in a parameter-only comparison: a plain third convolution (`c12x32x64`, 42,084 parameters) reaches 91.644% test, while two-layer `c8x16`+PB reaches 83.276% at 42,492. The deeper model uses more Conv/Linear MACs, so this is not a joint accuracy/parameter/compute Pareto domination.

CPU b24 tests PAI on a stronger three-layer base. All-layer PB at `c12x32x64` reaches 92.139% test at 84,288 parameters; the near-matched widened three-layer control `c17x50x100` reaches 90.806% at 84,182 (+1.33 pp PB). Linear-only PB reaches 91.898% at 62,076, compared with widened `c16x41x82` at 91.284%, 62,041 (+0.61 pp PB). However a near-matched four-layer control `c12x32x64x94` reaches 92.773% at 84,250, 38 parameters fewer than all-layer PB. The best conclusion is that PAI can beat width under this training recipe, while depth remains a strong competitor. Exact paired seed intervals and MAC proxies follow below.

## Provenance and measurement cautions discovered during census

- All 267 faithful result files are SC-v2 **12-class** results, not the original published Edge Impulse task. All use 4,890 held-out test samples, selected by validation. Main/b21 use MPS; b22/b23/b24 use CPU.
- Main, b22 and b23 controls run for 600 epochs; b24 controls use an LR stop below 1e-6 (144–220 epochs for its width control; 145–219 for its four-layer control). Selected checkpoints can be compared, but training cost is unequal. A b24 no-dendrite small-base control is not repeated locally: its exact architecture control exists in b22 on CPU, same recipe, seeds 0–4, 600 epochs. Cross-batch attribution must disclose this.
- b21 explicitly separates full-data and 20% training data; its detached optimizer variants train phase 0 at a constant LR because the tracker scheduler points at a different optimizer. These cannot be pooled with tracked variants. The aborted 10% branch has seven completed no-dendrite seed-0 controls, three partial dendritic runs and one failed PB run; it supplies no completed PB/GD test outcome.
- There are 94 tracker-completed faithful runs, 158 evaluated epoch-cap runs and 15 evaluated LR-stop runs. Of the 95 dendritic runs, 93 retain at least one dendrite; 2 select zero-dendrite models after attempting a candidate. This is why attempt counters and retry filenames are not retention evidence.
- Native architecture census is complete: all 211 manifests say completed. No test JSON lives under these folders. Their best-validation means are validation evidence, not test accuracy.
- All 211 current native model configurations exactly match the SHA256 recorded in their run manifests. A separate read-only shape probe constructed 62 unique configurations; every measured parameter count agreed with the logged epoch count. This imports ordinary local model code, not the licensed PAI package.
- Faithful results do not report measured MACs. `faithful-run-inventory.csv` reports an explicitly labeled analytical Conv2d/Linear proxy (one retained PAI dendrite doubles copied modules). The original affine-only column excludes normalization, pooling, nonlinearities, bias additions, PAI skip-edge multiplications and frontend cost; an additional proxy column includes the one-dendrite top-edge multiplications. Multi-dendrite rows have no proxy until their exact graph is profiled.
- `faithful-native-model-costs.csv` uses forward hooks for Conv1d/Conv2d/Linear. DTNet executes dense tensor matmuls directly, so hooks would undercount it: its cost is separately labeled ideal streaming recurrence plus IDCT, not the current MPS training execution.

## Critical correction: the scored live graph differs from the unverified cleaned copy

The faithful script calls `clean_param_count` on a deep copy, performs `blockwise_network` and `refresh_net`, records its parameter count, then evaluates validation/test on the original live selected model. It never checks score parity of the cleaned graph. The installed single-dendrite clean wrapper omits its only `skip_weights` entry. The main pipeline explicitly repairs this through `_collect_dendrite_top_weights` and `_restore_single_dendrite_skip_weights` in `src/kws/optimize/dendritic.py:1195`; the faithful script does not call that repair. The parent audit checked the installed clean-wrapper implementation: without this attribute its forward falls through to the original base output, discarding the single dendrite's contribution. Therefore `clean_best` is an informational count of an unverified graph, not the cost of the graph that achieved the reported accuracy.

The result files already provide a better accounting field: `numel_best`, the unique live parameters excluding `parent_module` aliases. For b24 classifier-only PB, that is **62,076**, not raw cleaned **62,064**: 12 trained dendrite-to-output coefficients were omitted. For all-layer PB at the same three-convolution base it is **84,288**, not raw cleaned **84,168**: `12 + 32 + 64 + 12 = 120` coefficients were omitted. Main `c8x16` PB is **42,492**, not **42,456**; main `c4x8` PB/GD is **20,688**, not **20,664**.

All tables and paired cost CSVs below use the live unique parameter count. `faithful-run-inventory.csv` preserves both raw clean and live counts. It also provides a separate live analytical proxy including the top-edge multiplications. The correct b24 linear proxy is **3,332,472 MACs**; all-layer PB is **6,658,832**. These proxies remain architecture estimates rather than hardware measurements. Multi-dendrite live `numel_best` can include superseded top arrays, so its inventory value is explicitly a recorded count rather than an asserted minimal deployment cost.

This correction strengthens the four-layer comparison: the plain four-layer control has **84,250 parameters**, 38 fewer than all-layer PB's **84,288**, and **4,708,164** affine MACs versus PB's **6,658,832** including top coefficients. It also has higher mean test accuracy by **0.634 pp**. That is domination under the scored parameter counts and analytical operation convention; measured device latency/power and verified clean exports are unavailable for this family.

A safe `torch.load(..., weights_only=True)` of a representative faithful best checkpoint was rejected by its serialization. No unsafe load or licensed PAI import was attempted; this audit relies on explicit recorded unique counts, structural metadata, existing source, and epoch records. It does not repair or evaluate the exported faithful graph.

## What the first 13 MFCCs and frontend mean here

The common train/validation split sizes in faithful logs are **36,923 / 4,445**, with **4,890** fixed-seed test examples. It is the 12-class balanced SC-v2 task. The original PAI result described in the driver is a recipe reference, not the same benchmark. Faithful `EICNN` treats the MFCC coefficient index and time as the two spatial axes of a `Conv2d` and keeps only 13 coefficients. The older `students-ei` family uses `Conv1d` over time with all 32 MFCC coefficients as input channels. DTNet optionally applies IDCT to recover frequency-ordered log-mel energies before CMVN. SparkNet uses MFCC bins as channels. These representations and receptive fields are different.

The phase-0 optimizer bug is explicitly exposed, not hidden: `--phase0-optimizer detached` reproduces the source block's scheduler being attached to an unused Adam optimizer, so the actual first-phase optimizer stays at LR 0.005; `tracked` uses the optimizer returned by the PAI tracker. Comparisons below preserve that distinction. Faithful `none` controls still use the PAI tracker and scheduler, but `doing_pai=False`, `max_dendrites=0` and `DOING_NO_SWITCH`; having the package installed is not evidence of adaptive dendrites.

`max_dendrites=1` means one candidate branch **per selected module**. In the two-convolution all-layer model it copies `conv1`, `conv2`, and `fc`; in a three-convolution model it copies four modules. Linear-only adds a branch at `fc` alone. This is why all-layer perforation approximately doubles the network. `pb` explicitly enables Perforated Backpropagation; `gd` explicitly disables it but retains the PAI adaptive architecture machinery. The native student architectures use ordinary joint gradient descent with fixed branches and do not invoke PAI architecture search.

## Attribution checks and statistical limits

Same-base faithful pairs really do share their initial trajectories. On the main MPS `c8x16` pair, `train_loss`, `val_loss`, `val_acc`, and LR are identical from epoch 1 until the first restructuring, for **137, 95, 113, 104, 76** epochs at seeds 0–4. For CPU b23 `c6x12x24`, the common prefixes are **97, 82, 125, 92, 85** epochs. For b24 classifier-only PB versus b22's CPU `c12x32x64` base, the common prefixes are **105, 99, 114, 100, 98** epochs. This validates the cross-batch small-base control as the same initial training trajectory; the dendritic run diverges only at the first PAI restructuring.

This still estimates the combined effect of adding/training dendrites, freezing/releasing phases, optimizer resets and the PAI stopping/selection policy. Fixed every-25-epoch restart controls test one reset schedule, not an exact event-matched sham. Against same-size main `c8x16`, the raw PB gain is +2.937 pp; against its restart control it is +1.472 pp. Against doubled-width `c16x32`, +0.282 pp falls to +0.094 pp against its restart control. Thus some gains depend on the optimization policy.

The paired intervals below use Student t intervals across matched **seed labels**, with n=5 (or n=3 for native comparisons), and are descriptive. No correction for the many architecture/activation/placement comparisons is applied. Width/depth models have different shapes and initial draws; a shared seed does not make them the same initial model. Multiple batch repeats with seeds 0–4 are not ten independent seeds. The same fixed test set is reused across all architectures and budget snapshots; it is held out from per-run checkpoint selection, but the overall research program has observed it repeatedly. Without per-example predictions, this audit cannot perform a prediction-paired McNemar comparison.

All 267 final result files expose both selected reevaluation accuracy and the largest epoch accuracy. In 44 cases they differ slightly, by a few validation examples. This audit uses `val_acc_selected`, the evaluated selected checkpoint, not an optimistic maximum sampled along the training loop.

## Low-data failure and missing planned work

At 10% training data, there are only **29 batches per epoch** with the faithful batch size 128. The PB default waits for **40 initial correlation batches**. The failed `f10-pb-all-max1-tanh-sw25-c4x8-seed0` log reaches its first restructure at epoch 133, then raises the guarded PAI debugger exception; the diagnostic explicitly reports `29 < 40` and identifies `conv1`. Its failure is a configuration/batch-count incompatibility, not evidence that dendrites inherently fail with scarce data. The three other 10% dendritic folders are partial; the GD run had one dendrite at epoch 173, while the two `c8x16` PB runs had not reached addition by epochs 115/123. No completed 10% dendritic test result exists.

All 25 `research-*` queue folders were audited. Twenty-one retain `jobs.all.txt`; four consumed native queues retain empty `jobs.txt` plus status logs and the targets' manifests. The combined queue provenance has 1,000 rows before deduplicating repeated schedules, including requeues and unexecuted plans. `research-1003a` contains 40 b25 jobs, an empty status log, and no target output directories at audit time. It requests new seeds 5–9 and a new four-convolution `c12x32x64x61` control. These would help independent replication and the classifier-only depth comparison, but they are not run results.

## Every faithful arm, separated by batch, training fraction and phase-0 optimizer

All faithful accuracies in this table are selected-checkpoint percentages. MAC columns are analytical live affine-plus-top proxies; a dash denotes multi-dendrite topology not reconstructed. Parameters are recorded live unique counts; multi-dendrite counts may include inactive historical arrays. The five-seed spread is the sample standard deviation across runs, not test-sampling uncertainty.

| Batch | Train fraction | Phase 0 | Arm | n | Live params | Live MAC proxy | Validation mean | Test mean ± SD | Retained counts (seed order) | Mean epochs |
|---|---:|---|---|---:|---|---|---:|---:|---|---:|
| pai-faithful | 1.0 | tracked | gd-all-max1-tanh-sw25-c4x8 | 5 | 20688 | 328256 | 75.748 | 72.814 ± 2.279 | 1,1,1,1,1 | 166.400 |
| pai-faithful | 1.0 | tracked | none-c12x24 | 5 | 32700 | 1097100 | 84.562 | 82.818 ± 0.788 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c16x32 | 5 | 44748 | 1874064 | 84.346 | 82.654 ± 0.951 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c20x40 | 5 | 57372 | 2856660 | 83.118 | 81.419 ± 2.212 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c24x48 | 5 | 70572 | 4044888 | 80.670 | 78.859 ± 3.848 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c28x56 | 5 | 84348 | 5438748 | 82.448 | 80.376 ± 2.851 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c4x8 | 5 | 10332 | 160068 | 71.226 | 67.963 ± 1.214 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c6x12 | 5 | 15708 | 317214 | 81.098 | 78.634 ± 1.108 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-c8x16 | 5 | 21228 | 525768 | 82.929 | 80.000 ± 0.853 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-r25-c16x32 | 5 | 44748 | 1874064 | 84.468 | 82.843 ± 1.107 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-r25-c28x56 | 5 | 84348 | 5438748 | 80.553 | 77.914 ± 2.669 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | none-r25-c8x16 | 5 | 21228 | 525768 | 83.235 | 81.464 ± 1.178 | 0,0,0,0,0 | 600.000 |
| pai-faithful | 1.0 | tracked | pb-all-max1-relu-sw25-c4x8 | 5 | 10332,20688 | 160068,328256 | 72.832 | 69.804 ± 3.026 | 1,1,1,0,1 | 267.800 |
| pai-faithful | 1.0 | tracked | pb-all-max1-tanh-sw25-c4x8 | 5 | 20688 | 328256 | 76.684 | 72.798 ± 1.380 | 1,1,1,1,1 | 205.600 |
| pai-faithful | 1.0 | tracked | pb-all-max1-tanh-sw25-c8x16 | 5 | 42492 | 1067764 | 85.548 | 82.937 ± 0.844 | 1,1,1,1,1 | 279.800 |
| pai-faithful | 1.0 | tracked | pb-linear-max3-tanh-sw25-c8x16 | 5 | 41220,61248,81300 | 545748,— | 84.940 | 82.303 ± 0.789 | 3,1,3,2,3 | 518.400 |
| pai-faithful-b21 | 0.2 | detached | none-c8x16 | 5 | 21228 | 525768 | 76.900 | 73.992 ± 1.089 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | detached | pb-all-max1-relu-sw25-c8x16 | 5 | 42492 | 1067764 | 77.296 | 74.339 ± 0.823 | 1,1,1,1,1 | 227.800 |
| pai-faithful-b21 | 0.2 | tracked | gd-all-max1-tanh-sw25-c8x16 | 5 | 42492 | 1067764 | 77.003 | 74.286 ± 1.017 | 1,1,1,1,1 | 239.200 |
| pai-faithful-b21 | 0.2 | tracked | none-c12x24 | 5 | 32700 | 1097100 | 77.795 | 75.350 ± 1.253 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-c16x32 | 5 | 44748 | 1874064 | 77.872 | 75.309 ± 0.189 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-c24x48 | 5 | 70572 | 4044888 | 77.350 | 75.063 ± 0.837 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-c4x8 | 5 | 10332 | 160068 | 68.292 | 64.924 ± 1.202 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-c8x16 | 5 | 21228 | 525768 | 76.036 | 73.215 ± 1.117 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-r25-c16x32 | 5 | 44748 | 1874064 | 78.443 | 75.746 ± 0.713 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | none-r25-c8x16 | 5 | 21228 | 525768 | 77.017 | 74.376 ± 0.686 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 0.2 | tracked | pb-all-max1-relu-sw25-c8x16 | 5 | 42492 | 1067764 | 77.786 | 74.879 ± 0.633 | 1,1,1,1,1 | 283.200 |
| pai-faithful-b21 | 0.2 | tracked | pb-all-max1-tanh-sw25-c4x8 | 5 | 20688 | 328256 | 69.836 | 66.446 ± 1.190 | 1,1,1,1,1 | 238.200 |
| pai-faithful-b21 | 0.2 | tracked | pb-all-max1-tanh-sw25-c8x16 | 5 | 42492 | 1067764 | 77.219 | 74.434 ± 1.216 | 1,1,1,1,1 | 264.600 |
| pai-faithful-b21 | 1.0 | detached | none-c16x32 | 5 | 44748 | 1874064 | 83.843 | 81.329 ± 1.097 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 1.0 | detached | none-c8x16 | 5 | 21228 | 525768 | 83.307 | 81.117 ± 1.047 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b21 | 1.0 | detached | pb-all-max1-relu-sw25-c8x16 | 5 | 21228,42492 | 1067764,525768 | 83.204 | 80.622 ± 1.466 | 1,0,1,1,1 | 337.200 |
| pai-faithful-b21 | 1.0 | tracked | pb-all-max1-relu-sw25-c8x16 | 5 | 42492 | 1067764 | 83.631 | 81.076 ± 1.171 | 1,1,1,1,1 | 312.200 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-c12x24 | 1 | 32700 | 1097100 | 70.619 | 67.607 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-c16x32 | 1 | 44748 | 1874064 | 71.991 | 68.262 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-c24x48 | 1 | 70572 | 4044888 | 70.304 | 68.589 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-c4x8 | 1 | 10332 | 160068 | 63.937 | 60.982 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-c8x16 | 1 | 21228 | 525768 | 68.684 | 66.912 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-r25-c16x32 | 1 | 44748 | 1874064 | 73.386 | 70.798 ± — | 0 | 600.000 |
| pai-faithful-b21-f10-aborted | 0.1 | tracked | none-r25-c8x16 | 1 | 21228 | 525768 | 70.641 | 67.464 ± — | 0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c12x24 | 5 | 32700 | 1097100 | 83.955 | 82.728 ± 1.549 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c12x32x64 | 5 | 42084 | 3312492 | 92.265 | 91.644 ± 0.511 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c16x32 | 5 | 44748 | 1874064 | 84.526 | 83.108 ± 1.522 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c16x42x84 | 5 | 64306 | 5676624 | 92.229 | 91.456 ± 0.415 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c20x40 | 5 | 57372 | 2856660 | 83.181 | 81.710 ± 4.087 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | none-c24x48 | 5 | 70572 | 4044888 | 81.588 | 79.395 ± 3.198 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b22 | 1.0 | tracked | pb-all-max1-sigmoid-sw25-c8x16 | 5 | 42492 | 1067764 | 85.093 | 83.346 ± 0.810 | 1,1,1,1,1 | 246.800 |
| pai-faithful-b22 | 1.0 | tracked | pb-all-max1-tanh-sw25-c12x24 | 5 | 65448 | 2218536 | 86.443 | 84.601 ± 0.870 | 1,1,1,1,1 | 323.800 |
| pai-faithful-b22 | 1.0 | tracked | pb-all-max1-tanh-sw25-c8x16 | 5 | 42492 | 1067764 | 85.057 | 83.276 ± 0.496 | 1,1,1,1,1 | 249.600 |
| pai-faithful-b23 | 1.0 | tracked | none-c12x24x48 | 5 | 28140 | 2160396 | 91.195 | 90.384 ± 0.515 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b23 | 1.0 | tracked | none-c6x12x24 | 5 | 10836 | 579294 | 86.884 | 84.969 ± 1.369 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b23 | 1.0 | tracked | none-c8x20x40 | 5 | 21272 | 1369896 | 90.367 | 89.509 ± 0.738 | 0,0,0,0,0 | 600.000 |
| pai-faithful-b23 | 1.0 | tracked | pb-all-max1-tanh-sw25-c6x12x24 | 5 | 21726 | 1173258 | 88.522 | 87.121 ± 0.791 | 1,1,1,1,1 | 298.600 |
| pai-faithful-b23 | 1.0 | tracked | pb-all-max1-tanh-sw25-c8x20x40 | 5 | 42624 | 2761608 | 91.145 | 90.540 ± 0.796 | 1,1,1,1,1 | 285.200 |
| pai-faithful-b24 | 1.0 | tracked | none-c12x32x64x94 | 5 | 84250 | 4708164 | 93.525 | 92.773 ± 0.358 | 0,0,0,0,0 | 175.000 |
| pai-faithful-b24 | 1.0 | tracked | none-c16x41x82 | 5 | 62041 | 5469216 | 91.888 | 91.284 ± 0.535 | 0,0,0,0,0 | 177.000 |
| pai-faithful-b24 | 1.0 | tracked | none-c17x50x100 | 5 | 84182 | 7643139 | 91.658 | 90.806 ± 0.824 | 0,0,0,0,0 | 162.000 |
| pai-faithful-b24 | 1.0 | tracked | pb-all-max1-tanh-sw25-c12x32x64 | 5 | 84288 | 6658832 | 92.630 | 92.139 ± 0.324 | 1,1,1,1,1 | 328.600 |
| pai-faithful-b24 | 1.0 | tracked | pb-linear-max1-tanh-sw25-c12x32x64 | 5 | 62076 | 3332472 | 92.481 | 91.898 ± 0.421 | 1,1,1,1,1 | 281.800 |

## Native architectural experiments: complete census

These 211 runs are fixed native architectures trained from scratch for 200 epochs, not PerforatedAI adaptive dendrites. Only validation accuracy is available. DTNet costs are ideal causal recurrence plus IDCT; other MACs are new dummy-forward Conv1d/Conv2d/Linear hook counts on SHA-verified unchanged model configs. No frontend extraction cost, activation, normalization, bias adds or pooling cost is included. The training configuration varies by group as recorded in the inventory.

| Group | Model | n | Native branches? | Params | MACs | Validation mean ± SD |
|---|---|---:|---|---:|---:|---:|
| students-dnn | dnn_h16 | 5 | False | 13532 | 13504 | 69.183 ± 0.638 |
| students-dnn | dnn_h16_d2f32 | 5 | True | 15052 | 14968 | 71.465 ± 0.310 |
| students-dnn | dnn_h16_d2f32lin | 5 | True | 15052 | 14968 | 70.124 ± 0.649 |
| students-dnn | dnn_h16_d4f32 | 5 | True | 16572 | 16432 | 73.966 ± 0.832 |
| students-dnn | dnn_h16_d4f32head | 5 | True | 14396 | 14320 | 71.793 ± 0.803 |
| students-dnn | dnn_h16_d4f32hid | 5 | True | 15708 | 15616 | 71.483 ± 0.440 |
| students-dnn | dnn_h16_d4f32lin | 5 | True | 16572 | 16432 | 69.494 ± 1.028 |
| students-dnn | dnn_h16_d4f32rnd | 5 | True | 16572 | 16432 | 73.156 ± 0.418 |
| students-dnn | dnn_h16x112 | 5 | False | 16588 | 16448 | 76.103 ± 0.298 |
| students-dnn | dnn_h16x59 | 5 | False | 15051 | 14964 | 74.349 ± 0.596 |
| students-dnn | dnn_h16x59_hd2 | 5 | True | 15867 | 15756 | 74.704 ± 0.510 |
| students-dnn | dnn_h16x59_hd4 | 5 | True | 16683 | 16548 | 75.366 ± 0.642 |
| students-dnn | dnn_h16x59x21 | 5 | False | 15855 | 15747 | 75.438 ± 0.467 |
| students-dnn | dnn_h16x59x32 | 5 | False | 16647 | 16528 | 75.640 ± 0.483 |
| students-dnn | dnn_h16x87 | 5 | False | 15863 | 15748 | 75.699 ± 0.381 |
| students-dnn | dnn_h18 | 5 | False | 15222 | 15192 | 70.038 ± 0.413 |
| students-dnn | dnn_h20 | 5 | False | 16912 | 16880 | 71.600 ± 0.883 |
| students-dnn | dnn_h24 | 5 | False | 20292 | 20256 | 74.232 ± 0.499 |
| students-dnn | dnn_h32 | 5 | False | 27052 | 27008 | 77.156 ± 0.654 |
| students-dnn | dnn_h32_d2f32 | 5 | True | 30044 | 29912 | 78.785 ± 0.408 |
| students-dnn | dnn_h35 | 5 | False | 29587 | 29540 | 77.755 ± 0.624 |
| students-dnn | dnn_h48 | 5 | False | 40572 | 40512 | 80.117 ± 0.626 |
| students-dscnn | ds_cnn_w20_mfcc | 3 | False | 2112 | 1354800 | 91.339 ± 0.354 |
| students-dscnn | ds_cnn_w20d1f4_mfcc | 3 | True | 2352 | 1518000 | 91.219 ± 0.057 |
| students-dscnn | ds_cnn_w20d2f4_mfcc | 3 | True | 2592 | 1681200 | 91.076 ± 0.630 |
| students-dscnn | ds_cnn_w24_mfcc | 3 | False | 2724 | 1782432 | 92.268 ± 0.305 |
| students-dscnn | ds_cnn_w24d2f4_mfcc | 3 | True | 3300 | 2174112 | 92.771 ± 0.332 |
| students-dscnn | ds_cnn_w24fcd4f8_mfcc | 3 | True | 3204 | 1782864 | 92.313 ± 0.594 |
| students-dscnn | ds_cnn_w28_mfcc | 3 | False | 3400 | 2262288 | 93.018 ± 0.163 |
| students-dscnn | ds_cnn_w32_mfcc | 3 | False | 4140 | 2794368 | 93.446 ± 0.332 |
| students-dtnet | dtnet_a_het | 3 | True | 4160 | 419760 | 89.036 ± 0.715 |
| students-dtnet | dtnet_a_lin | 3 | True | 4160 | 419760 | 87.454 ± 0.677 |
| students-dtnet | dtnet_a_mfcc | 3 | True | 4160 | 316336 | 79.768 ± 1.543 |
| students-dtnet | dtnet_a_notau | 3 | True | 4167 | 384030 | 89.794 ± 0.806 |
| students-dtnet | dtnet_a_point | 3 | False | 4183 | 455723 | 89.794 ± 0.249 |
| students-dtnet | dtnet_a_point3 | 3 | False | 4137 | 460451 | 91.429 ± 0.265 |
| students-dtnet | dtnet_a_rnd | 3 | True | 4160 | 419760 | 89.756 ± 0.289 |
| students-dtnet | dtnet_a_shared | 3 | True | 4176 | 438356 | 88.871 ± 0.447 |
| students-dtnet | dtnet_b_het | 3 | True | 2019 | 256934 | 86.322 ± 0.749 |
| students-dtnet | dtnet_b_point | 3 | False | 2029 | 265639 | 88.219 ± 0.182 |
| students-ei | ei_c10x20 | 1 | False | 7842 | 133800 | 86.232 ± — |
| students-ei | ei_c4x8 | 1 | False | 3000 | 46176 | 68.571 ± — |
| students-ei | ei_c4x8_hd1f32 | 1 | True | 3408 | 46572 | 65.692 ± — |
| students-ei | ei_c4x8_hd2f32 | 1 | True | 3816 | 46968 | 62.002 ± — |
| students-ei | ei_c4x8_hd4f32 | 1 | True | 4632 | 47760 | 65.984 ± — |
| students-ei | ei_c5x10 | 1 | False | 3777 | 59250 | 69.291 ± — |
| students-ei | ei_c6x12 | 1 | False | 4566 | 72936 | 78.943 ± — |
| students-ei | ei_c7x14 | 1 | False | 5367 | 87234 | 82.317 ± — |
| students-ei | ei_c8x16 | 1 | False | 6180 | 102144 | 82.182 ± — |
| students-ei | ei_c8x16_hd1f64 | 1 | True | 6972 | 102924 | 83.802 ± — |
| students-ei | ei_c8x16_hd2f64 | 1 | True | 7764 | 103704 | 83.307 ± — |
| students-ei | ei_c9x18 | 1 | False | 7005 | 117666 | 84.882 ± — |
| students-ei-adamw | ei_c10x20 | 1 | False | 7842 | 133800 | 85.849 ± — |
| students-ei-adamw | ei_c4x8 | 1 | False | 3000 | 46176 | 65.692 ± — |
| students-ei-adamw | ei_c4x8_hd2f32 | 1 | True | 3816 | 46968 | 64.342 ± — |
| students-ei-adamw | ei_c8x16 | 1 | False | 6180 | 102144 | 81.260 ± — |
| students-ei-adamw | ei_c8x16_hd2f64 | 1 | True | 7764 | 103704 | 83.510 ± — |
| students-msd | sparknet_msd_a13_dense | 3 | False | 4189 | 391204 | 93.963 ± 0.779 |
| students-msd | sparknet_msd_a13_lin | 3 | True | 4134 | 369044 | 93.506 ± 0.352 |
| students-msd | sparknet_msd_a13_relu | 3 | True | 4134 | 369044 | 94.031 ± 0.143 |
| students-msd | sparknet_msd_a_1scale | 3 | True | 4114 | 350864 | 93.558 ± 0.175 |
| students-msd | sparknet_msd_a_dense | 3 | False | 4101 | 379043 | 93.108 ± 0.364 |
| students-msd | sparknet_msd_a_lin | 3 | True | 4116 | 369448 | 89.486 ± 0.496 |
| students-msd | sparknet_msd_a_relu | 3 | True | 4116 | 369448 | 90.731 ± 0.228 |
| students-msd | sparknet_msd_b_dense | 3 | False | 1999 | 181593 | 88.046 ± 0.238 |
| students-msd | sparknet_msd_b_lin | 3 | True | 1999 | 177048 | 86.839 ± 1.216 |
| students-msd | sparknet_msd_b_relu | 3 | True | 1999 | 177048 | 87.829 ± 0.487 |

## Paired faithful test effects

Values are arm minus named control in percentage points. See [faithful-paired-effects.csv](faithful-paired-effects.csv) for paired per-seed values, validation effects, exact arm keys and recorded costs. These descriptive 95% intervals are uncorrected for exploratory comparisons.

| Comparison | n | Test delta pp | Descriptive 95% interval | Positive seeds | Arm / control live params |
|---|---:|---:|---|---:|---|
| pb-all-max1-tanh-sw25-c4x8 vs same base | 5 | 4.834 | [3.648, 6.021] | 5/5 | 20688 / 10332 |
| pb-all-max1-tanh-sw25-c4x8 vs width | 5 | -7.202 | [-9.697, -4.708] | 0/5 | 20688 / 21228 |
| gd-all-max1-tanh-sw25-c4x8 vs same base | 5 | 4.851 | [2.512, 7.190] | 5/5 | 20688 / 10332 |
| gd-all-max1-tanh-sw25-c4x8 vs width | 5 | -7.186 | [-9.841, -4.531] | 0/5 | 20688 / 21228 |
| pb-all-max1-relu-sw25-c4x8 vs same base | 5 | 1.840 | [-0.812, 4.493] | 4/5 | 10332,20688 / 10332 |
| pb-all-max1-relu-sw25-c4x8 vs width | 5 | -10.196 | [-14.518, -5.875] | 0/5 | 10332,20688 / 21228 |
| PB vs GD low width | 5 | -0.016 | [-2.696, 2.663] | 4/5 | 20688 / 20688 |
| main PB c8x16 vs none-c8x16 | 5 | 2.937 | [1.995, 3.878] | 5/5 | 42492 / 21228 |
| main PB c8x16 vs none-r25-c8x16 | 5 | 1.472 | [0.079, 2.866] | 4/5 | 42492 / 21228 |
| main PB c8x16 vs none-c16x32 | 5 | 0.282 | [-0.251, 0.816] | 4/5 | 42492 / 44748 |
| main PB c8x16 vs none-r25-c16x32 | 5 | 0.094 | [-1.062, 1.250] | 3/5 | 42492 / 44748 |
| main linear max3 vs none-c8x16 | 5 | 2.303 | [1.594, 3.011] | 5/5 | 41220,61248,81300 / 21228 |
| main linear max3 vs none-r25-c8x16 | 5 | 0.838 | [-1.153, 2.830] | 4/5 | 41220,61248,81300 / 21228 |
| main linear max3 vs none-c16x32 | 5 | -0.352 | [-1.664, 0.960] | 2/5 | 41220,61248,81300 / 44748 |
| main linear max3 vs none-c28x56 | 5 | 1.926 | [-2.310, 6.162] | 4/5 | 41220,61248,81300 / 84348 |
| full tracked ReLU vs base | 5 | 1.076 | [0.038, 2.113] | 5/5 | 42492 / 21228 |
| full detached ReLU vs detached base | 5 | -0.495 | [-3.059, 2.069] | 2/5 | 21228,42492 / 21228 |
| full ReLU detached minus tracked | 5 | -0.454 | [-1.736, 0.828] | 1/5 | 21228,42492 / 42492 |
| 20pct pb-all-max1-tanh-sw25-c8x16 vs none-c8x16 | 5 | 1.219 | [0.721, 1.717] | 5/5 | 42492 / 21228 |
| 20pct pb-all-max1-tanh-sw25-c8x16 vs none-r25-c8x16 | 5 | 0.057 | [-1.203, 1.318] | 2/5 | 42492 / 21228 |
| 20pct pb-all-max1-tanh-sw25-c8x16 vs none-c16x32 | 5 | -0.875 | [-2.566, 0.815] | 1/5 | 42492 / 44748 |
| 20pct pb-all-max1-tanh-sw25-c8x16 vs none-r25-c16x32 | 5 | -1.313 | [-3.201, 0.575] | 1/5 | 42492 / 44748 |
| 20pct pb-all-max1-relu-sw25-c8x16 vs none-c8x16 | 5 | 1.665 | [0.698, 2.631] | 5/5 | 42492 / 21228 |
| 20pct pb-all-max1-relu-sw25-c8x16 vs none-r25-c8x16 | 5 | 0.503 | [-0.560, 1.566] | 4/5 | 42492 / 21228 |
| 20pct pb-all-max1-relu-sw25-c8x16 vs none-c16x32 | 5 | -0.429 | [-1.367, 0.509] | 1/5 | 42492 / 44748 |
| 20pct pb-all-max1-relu-sw25-c8x16 vs none-r25-c16x32 | 5 | -0.867 | [-1.906, 0.172] | 1/5 | 42492 / 44748 |
| 20pct gd-all-max1-tanh-sw25-c8x16 vs none-c8x16 | 5 | 1.072 | [0.554, 1.589] | 5/5 | 42492 / 21228 |
| 20pct gd-all-max1-tanh-sw25-c8x16 vs none-r25-c8x16 | 5 | -0.090 | [-0.843, 0.663] | 1/5 | 42492 / 21228 |
| 20pct gd-all-max1-tanh-sw25-c8x16 vs none-c16x32 | 5 | -1.022 | [-2.373, 0.328] | 1/5 | 42492 / 44748 |
| 20pct gd-all-max1-tanh-sw25-c8x16 vs none-r25-c16x32 | 5 | -1.460 | [-3.438, 0.518] | 1/5 | 42492 / 44748 |
| 20pct PB vs GD tanh | 5 | 0.147 | [-0.780, 1.075] | 4/5 | 42492 / 42492 |
| 20pct detached ReLU vs detached base | 5 | 0.348 | [-1.427, 2.123] | 2/5 | 42492 / 21228 |
| 20pct PB c4 vs base | 5 | 1.521 | [0.207, 2.836] | 5/5 | 20688 / 10332 |
| 20pct PB c4 vs width | 5 | -6.769 | [-8.497, -5.040] | 0/5 | 20688 / 21228 |
| CPU b22 sigmoid minus tanh | 5 | 0.070 | [-1.041, 1.180] | 2/5 | 42492 / 42492 |
| CPU b22 PB c8 vs width c16x32 | 5 | 0.168 | [-2.257, 2.592] | 3/5 | 42492 / 44748 |
| CPU b22 PB c8 vs 3conv same params | 5 | -8.368 | [-9.587, -7.149] | 0/5 | 42492 / 42084 |
| CPU b22 PB c12 vs base | 5 | 1.873 | [0.727, 3.019] | 5/5 | 65448 / 32700 |
| CPU b22 PB c12 vs 3conv same params | 5 | -6.855 | [-7.573, -6.137] | 0/5 | 65448 / 64306 |
| CPU b23 PB c6x12x24 vs base | 5 | 2.151 | [0.792, 3.510] | 5/5 | 21726 / 10836 |
| CPU b23 PB c6x12x24 vs width | 5 | -2.389 | [-4.072, -0.705] | 0/5 | 21726 / 21272 |
| CPU b23 PB c8x20x40 vs base | 5 | 1.031 | [0.683, 1.378] | 5/5 | 42624 / 21272 |
| CPU b23 PB c8x20x40 vs width | 5 | -1.104 | [-1.940, -0.268] | 0/5 | 42624 / 42084 |
| CPU b24 all PB vs base b22 | 5 | 0.495 | [-0.020, 1.010] | 5/5 | 84288 / 42084 |
| CPU b24 all PB vs near-matched width | 5 | 1.333 | [-0.035, 2.701] | 4/5 | 84288 / 84182 |
| CPU b24 linear PB vs base b22 | 5 | 0.254 | [0.041, 0.466] | 5/5 | 62076 / 42084 |
| CPU b24 linear PB vs near-matched width | 5 | 0.613 | [0.159, 1.068] | 5/5 | 62076 / 62041 |
| CPU b24 all PB vs 4conv near-matched params | 5 | -0.634 | [-1.031, -0.237] | 0/5 | 84288 / 84250 |
| CPU b24 all PB minus linear PB | 5 | 0.241 | [-0.224, 0.707] | 3/5 | 84288 / 62076 |

## Paired native validation effects

These compare manually constructed architectures and do not estimate the effect of PerforatedAI. See [faithful-native-paired-validation-effects.csv](faithful-native-paired-validation-effects.csv) for per-seed differences and MAC comparisons.

| Group | Arm minus control | n | Validation delta pp | Descriptive 95% interval | Arm / control params |
|---|---|---:|---:|---|---|
| students-dnn | dnn_h16_d2f32 minus dnn_h16 | 5 | 2.281 | [1.325, 3.238] | 15052 / 13532 |
| students-dnn | dnn_h16_d2f32 minus dnn_h18 | 5 | 1.426 | [0.821, 2.032] | 15052 / 15222 |
| students-dnn | dnn_h16_d2f32 minus dnn_h16x59 | 5 | -2.884 | [-3.924, -1.844] | 15052 / 15051 |
| students-dnn | dnn_h16_d4f32 minus dnn_h16 | 5 | 4.783 | [3.692, 5.874] | 16572 / 13532 |
| students-dnn | dnn_h16_d4f32 minus dnn_h20 | 5 | 2.367 | [1.014, 3.719] | 16572 / 16912 |
| students-dnn | dnn_h16_d4f32 minus dnn_h16x112 | 5 | -2.137 | [-3.278, -0.996] | 16572 / 16588 |
| students-dnn | dnn_h16_d2f32 minus dnn_h16_d2f32lin | 5 | 1.341 | [0.630, 2.052] | 15052 / 15052 |
| students-dnn | dnn_h16_d4f32 minus dnn_h16_d4f32lin | 5 | 4.472 | [2.297, 6.648] | 16572 / 16572 |
| students-dnn | dnn_h16_d4f32 minus dnn_h16_d4f32rnd | 5 | 0.810 | [-0.316, 1.936] | 16572 / 16572 |
| students-dnn | dnn_h16_d4f32head minus dnn_h16 | 5 | 2.610 | [1.861, 3.358] | 14396 / 13532 |
| students-dnn | dnn_h16_d4f32head minus dnn_h18 | 5 | 1.755 | [0.462, 3.048] | 14396 / 15222 |
| students-dnn | dnn_h16_d4f32hid minus dnn_h16 | 5 | 2.299 | [1.853, 2.746] | 15708 / 13532 |
| students-dnn | dnn_h32_d2f32 minus dnn_h32 | 5 | 1.629 | [0.853, 2.405] | 30044 / 27052 |
| students-dnn | dnn_h32_d2f32 minus dnn_h35 | 5 | 1.030 | [0.173, 1.887] | 30044 / 29587 |
| students-dnn | dnn_h16x59_hd2 minus dnn_h16x59 | 5 | 0.355 | [-0.409, 1.120] | 15867 / 15051 |
| students-dnn | dnn_h16x59_hd2 minus dnn_h16x87 | 5 | -0.994 | [-2.016, 0.027] | 15867 / 15863 |
| students-dnn | dnn_h16x59_hd2 minus dnn_h16x59x21 | 5 | -0.733 | [-1.631, 0.164] | 15867 / 15855 |
| students-dnn | dnn_h16x59_hd4 minus dnn_h16x59 | 5 | 1.017 | [-0.068, 2.102] | 16683 / 15051 |
| students-dnn | dnn_h16x59_hd4 minus dnn_h16x112 | 5 | -0.738 | [-1.458, -0.018] | 16683 / 16588 |
| students-dnn | dnn_h16x59_hd4 minus dnn_h16x59x32 | 5 | -0.274 | [-1.339, 0.790] | 16683 / 16647 |
| students-dscnn | ds_cnn_w20d1f4_mfcc minus ds_cnn_w20_mfcc | 3 | -0.120 | [-1.071, 0.831] | 2352 / 2112 |
| students-dscnn | ds_cnn_w20d2f4_mfcc minus ds_cnn_w20_mfcc | 3 | -0.262 | [-2.192, 1.667] | 2592 / 2112 |
| students-dscnn | ds_cnn_w24d2f4_mfcc minus ds_cnn_w24_mfcc | 3 | 0.502 | [-1.045, 2.050] | 3300 / 2724 |
| students-dscnn | ds_cnn_w24d2f4_mfcc minus ds_cnn_w28_mfcc | 3 | -0.247 | [-0.852, 0.357] | 3300 / 3400 |
| students-dscnn | ds_cnn_w24fcd4f8_mfcc minus ds_cnn_w24_mfcc | 3 | 0.045 | [-2.117, 2.207] | 3204 / 2724 |
| students-dscnn | ds_cnn_w24fcd4f8_mfcc minus ds_cnn_w28_mfcc | 3 | -0.705 | [-1.962, 0.553] | 3204 / 3400 |
| students-dtnet | dtnet_a_het minus dtnet_a_point | 3 | -0.757 | [-2.616, 1.101] | 4160 / 4183 |
| students-dtnet | dtnet_a_het minus dtnet_a_point3 | 3 | -2.392 | [-4.237, -0.548] | 4160 / 4137 |
| students-dtnet | dtnet_b_het minus dtnet_b_point | 3 | -1.897 | [-3.321, -0.473] | 2019 / 2029 |
| students-dtnet | dtnet_a_het minus dtnet_a_lin | 3 | 1.582 | [0.305, 2.860] | 4160 / 4160 |
| students-dtnet | dtnet_a_het minus dtnet_a_shared | 3 | 0.165 | [-0.981, 1.311] | 4160 / 4176 |
| students-dtnet | dtnet_a_het minus dtnet_a_notau | 3 | -0.757 | [-1.751, 0.237] | 4160 / 4167 |
| students-dtnet | dtnet_a_het minus dtnet_a_rnd | 3 | -0.720 | [-2.278, 0.838] | 4160 / 4160 |
| students-dtnet | dtnet_a_het minus dtnet_a_mfcc | 3 | 9.269 | [6.509, 12.029] | 4160 / 4160 |
| students-msd | sparknet_msd_a_relu minus sparknet_msd_a_lin | 3 | 1.245 | [-0.255, 2.745] | 4116 / 4116 |
| students-msd | sparknet_msd_a_relu minus sparknet_msd_a_dense | 3 | -2.377 | [-2.714, -2.040] | 4116 / 4101 |
| students-msd | sparknet_msd_a_relu minus sparknet_msd_a_1scale | 3 | -2.827 | [-3.302, -2.352] | 4116 / 4114 |
| students-msd | sparknet_msd_a13_relu minus sparknet_msd_a13_lin | 3 | 0.525 | [-0.460, 1.509] | 4134 / 4134 |
| students-msd | sparknet_msd_a13_relu minus sparknet_msd_a13_dense | 3 | 0.067 | [-2.221, 2.356] | 4134 / 4189 |
| students-msd | sparknet_msd_b_relu minus sparknet_msd_b_lin | 3 | 0.990 | [-2.369, 4.349] | 1999 / 1999 |
| students-msd | sparknet_msd_b_relu minus sparknet_msd_b_dense | 3 | -0.217 | [-1.815, 1.380] | 1999 / 1999 |
| students-ei | ei_c4x8_hd2f32 minus ei_c4x8 | 1 | -6.569 | single seed | 3816 / 3000 |
| students-ei | ei_c8x16_hd2f64 minus ei_c8x16 | 1 | 1.125 | single seed | 7764 / 6180 |
| students-ei | ei_c8x16_hd2f64 minus ei_c10x20 | 1 | -2.925 | single seed | 7764 / 7842 |
| students-ei-adamw | ei_c4x8_hd2f32 minus ei_c4x8 | 1 | -1.350 | single seed | 3816 / 3000 |
| students-ei-adamw | ei_c8x16_hd2f64 minus ei_c8x16 | 1 | 2.250 | single seed | 7764 / 6180 |
| students-ei-adamw | ei_c8x16_hd2f64 minus ei_c10x20 | 1 | -2.340 | single seed | 7764 / 7842 |
| students-ei | ei_c4x8_hd1f32 minus ei_c4x8 | 1 | -2.880 | single seed | 3408 / 3000 |
| students-ei | ei_c4x8_hd4f32 minus ei_c4x8 | 1 | -2.587 | single seed | 4632 / 3000 |
| students-ei | ei_c8x16_hd1f64 minus ei_c8x16 | 1 | 1.620 | single seed | 6972 / 6180 |
| students-ei | ei_c8x16_hd1f64 minus ei_c9x18 | 1 | -1.080 | single seed | 6972 / 7005 |

## What the native ablations add to our understanding

These native experiments answer architectural questions adjacent to PAI; they should not be counted as successful or failed PerforatedAI training. `DendriticPointwise` implements restricted local fan-in, a learned affine branch, ReLU and a learned soma sum. It is fixed at initialization and optimized jointly with the entire network. In native DNN and EI it adds a residual branch to a conventional affine layer. In DS-CNN it adds before the pointwise BatchNorm. In MSD SparkNet it replaces a pointwise mix while reading multiple temporal dilations. DTNet adds branch-specific causal low-pass filters and optional soma filters.

The native DNN study is a clear counterexample to the broad claim that dendritic nonlinearities never beat width. Five-seed validation means for `h16+d2` are **71.465%**, versus **70.038%** for a conventional `h18`, with **15,052 vs 15,222 parameters** and **14,968 vs 15,192 affine MACs**. The gain is **+1.426 pp**, descriptive paired interval **[+0.821,+2.032]**. `h16+d4` reaches **73.966%** versus `h20` **71.600%**, **16,572 vs 16,912 parameters** and **16,432 vs 16,880 affine MACs**, with **+2.367 pp [+1.014,+3.719]**.

But the corresponding conventional depth controls are stronger. `h16x59` has **15,051 parameters**, **14,964 affine MACs** and **74.349%** validation: its score is **2.884 pp** above `h16+d2` at practically identical or slightly smaller cost. `h16x112` has **16,588 parameters**, **16,448 affine MACs** and **76.103%** validation, beating `h16+d4` by **2.137 pp** while spending 16 extra parameters. On the stronger two-hidden-layer base `h16x59`, adding head branches gives only **+0.355 pp (d2)** or **+1.017 pp (d4)**, with wide paired intervals; other depth controls still reach similar or higher scores.

The DNN identity-activation control is useful mechanistic evidence. ReLU branches exceed identity branches by **+1.341 pp** at d2 and **+4.472 pp** at d4 with exactly the same parameters/MACs. An identity branch is linear in the original input and can be absorbed into the parent affine map; it changes parameterization without adding a new nonlinear function. ReLU expands the function class, which matters here. The local-versus-random window comparison is only **+0.810 pp [-0.316,+1.936]**, so these runs do not strongly establish locality as the cause.

Native DS-CNN is weaker evidence. At w20, one/two branch additions give **−0.120 / −0.262 pp** validation over the base. At w24, two pointwise branches give **+0.502 pp [-1.045,+2.050]**, but conventional w28 is **0.247 pp** higher while costing only 100 more parameters. Classifier-only w24 branches add **+0.045 pp** at 480 extra parameters, with an interval spanning roughly ±2 pp. None has held-out test evaluation in these folders.

The original MSD all-block design spends a large part of its budget making the 32-channel stem multiscale, which pushes its main width down to C8. It reaches **90.731%** validation, below dense multiscale C7 **93.108%** and a wider single-scale native dendritic C15 control **93.558%**. Restricting multiscale branches to blocks 1–3 leaves the stem conventional and permits C10: `a13_relu` reaches **94.031 ± 0.143%** validation at 4,134 parameters, versus its dense control **93.963 ± 0.779%** at 4,189. The matched seed delta is only **+0.067 pp [-2.221,+2.356]**. The ReLU-versus-identity comparison is **+0.525 pp [-0.460,+1.509]**. This is a promising validation-level architecture, but these runs do not isolate a reliable nonlinear dendrite advantage over dense mixing. The dense control also changes gate width to G11 versus G16, so it is a resource comparison rather than a clean one-knob intervention.

DTNet's heterogeneous timescale hypothesis is not supported as a winner. The ~4.2K-parameter heterogeneous two-layer branch model gets **89.036%** validation. Its equal-budget point-neuron two-layer control gets **89.794%**; a three-layer point control at slightly fewer parameters gets **91.429%**, beating it by **2.392 pp [-4.237,−0.548]**. At the ~2K budget, the point control wins by **1.897 pp [0.473,3.321]**. ReLU branch activation beats linear activation (+1.582 pp), yet heterogeneous branch time constants do not beat no-branch-time-constant or random-layout budget controls. Some controls retune widths, so they test usefulness under a budget rather than the isolated effect of changing a time-constant flag.

The DTNet frontend ablation is much larger than these branch effects: IDCT to frequency-ordered log-mel features versus retaining MFCC order changes validation by **+9.269 pp [+6.509,+12.029]** at the same parameters. Since a local window in MFCC space is not a local frequency band, this changes the meaning of locality as well as normalization. The IDCT also adds **103,424 affine MACs** under the ideal streaming estimate. Per-utterance CMVN uses a whole utterance and prevents claiming the full pipeline is strictly causal; the branch recurrence itself can be streamed.

Native EI head runs have only seed 0. The small c4x8 branches lose approximately **2.6–6.6 pp** under the SGD recipe. c8x16 head additions give **+1.1–1.6 pp** over the base, but conventional width controls outperform them at similar parameter budgets. AdamW's c8x16 d2 head gives +2.250 pp over its base, still 2.340 pp below the wider plain c10x20. These are preliminary single-seed validation comparisons and use all 32 MFCC coefficients, unlike faithful EICNN's first 13.

## Budget and training-policy findings

The stored budget snapshots demonstrate that longer training is needed to realize some PB benefits. For main MPS c8x16, the test gain over the same base is **−0.082 pp at epoch 100**, **+2.626 at 200**, and **+2.937 at 300**. Against the widened c16x32 control it is **−2.499**, **−0.029**, then **+0.282 pp** at those same caps. The b24 classifier-only PB advantage over its widened control is **+0.389 at 100**, **+0.491 at 200**, **+0.511 at 300**, and **+0.613 by 500**. All-layer b24 PB remains below the four-layer control at every stored cap (−0.982 at 100, −1.235 at 200, −0.634 finally).

These snapshots belong to their parent training trajectories. They are 294 descriptive comparison rows, not 294 new experiments. An equal epoch count also does not imply equal FLOPs or wall time once the dendritic model grows. A PAI restructuring resets Adam and ReduceLROnPlateau, and the driver can reject/retry a candidate even under a one-dendrite cap. b24 seed 2 runs its candidate phase twice: linear-only training ends at epoch 426 and all-layer at 502; this is search effort, not a second independent replicate.

PB superiority over ordinary GD is not established by the existing direct controls. At full-data c4x8, PB versus GD is **−0.016 pp [-2.696,+2.663]** test. At 20%-data c8x16, PB/tanh versus GD/tanh is **+0.147 pp [-0.780,+1.075]**. The three-layer b24 classifier-only result has no corresponding GD arm; it establishes that this PAI/PB recipe works in that setting, not that PB beats GD there. ReLU, tanh and sigmoid choices cannot be pooled: full-data low-width ReLU is weaker, and CPU b22 sigmoid versus tanh is only **+0.070 pp [-1.041,+1.180]**.

## Coverage, reproducibility and remaining boundaries

Final coverage is **292 faithful run roots = 267 evaluated, 24 interrupted/partial, 1 failed**; **211 completed native student run roots**; **59 faithful arm groups**, **67 native group/model arms** (62 unique model configurations); **49 seed-paired faithful comparisons** (98 validation/test rows); **51 seed-paired native validation comparisons**; and **25 research queue folders**. `research-0929c/status.log` also records 25 failed DNN queue attempts before the current completed artifacts. Those historical attempts are not 25 additional independent trained models; the current manifests no longer retain those failed invocations. The queue status CSV preserves their evidence.

Read-only verification performed: census all result files/epoch histories; recorded result and epoch SHA256 checksums; matched native model configurations to all 211 manifest checksums; constructed 62 ordinary model architectures and verified every parameter count against logged epochs; computed Conv1d/Conv2d/Linear shape-probe costs and separately labeled DTNet recurrence estimates; checked exact pre-restructure trajectory identity for the key same-base pairs. All saved analysis CSVs are generated by the accompanying scripts. No training, source edit, licensed PAI import, unsafe checkpoint unpickling, environment-file read or export repair was performed.

The supported conclusion is specific: adaptive PAI dendrites reliably improve several constrained faithful bases, and the three-layer classifier-only configuration provides a five-seed test gain against a nearly equal-parameter widened model at substantially lower affine compute. Whole-network growth often loses to plain depth, and the raw cleaned faithful graph is not parity-verified. Native nonlinear branches sometimes beat width, but conventional depth and frontend design explain much of the stronger architectural performance. These results do not justify a universal dendrite benefit, a universal PB advantage, a successful reproduction of a different published benchmark, or a deployed MCU claim for the faithful family.

Reproduce the read-only analysis from the repository root:

```bash
KWS_Model/.venv/bin/python KWS_Model/notes/dendrite-impact-2026-10-03/faithful_audit.py
KWS_Model/.venv/bin/python KWS_Model/notes/dendrite-impact-2026-10-03/faithful_native_costs.py
KWS_Model/.venv/bin/python KWS_Model/notes/dendrite-impact-2026-10-03/faithful_pairs.py
KWS_Model/.venv/bin/python KWS_Model/notes/dendrite-impact-2026-10-03/faithful_native_pairs.py
```
