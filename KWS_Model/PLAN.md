# SparkNet Student Evaluation Plan

**Status:** Phase A done (2026-09-13; see "Phase A results"). Every finding was rechecked on 2026-09-13
against the paper text, the vendored NeMo code, the released checkpoints'
tensors and stored hyperparameters, the project source, and the run outputs.
Corrections from that pass are folded in below.

The previous contents of this file, the unified run-outputs, metrics, and
resume plan, had already been removed from the working tree before this rewrite.
They remain available with `git show a0eebe9:KWS_Model/PLAN.md`.

## Question

Should the framework's step-2 deployment student change from DS-CNN-XS to
SparkNet (Svirsky, Shaham, and Lindenbaum, "Sparse Binarization for Fast
Keyword Spotting," Interspeech 2024)?

## Recommendation

Yes, but only after checks. The evidence below makes SparkNet the
highest-leverage change available to the student: at a similar parameter count
it reports about ten more points of 12-class accuracy than this project's XS
student, and it needs about eight times fewer MACs. The paper's accuracy was
measured on a different class balance, feature front end, and training recipe,
though, and the pipeline is wired to DS-CNN in about eight modules. So:

1. **Phase A:** score the released checkpoints on this project's split, with no
   training.
2. **Phase B:** run a short, recipe-controlled step-2 A/B.
3. **Phase C:** refactor the framework only if Phase B passes.

## Sources and how they were examined

- **Paper:** <https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf>.
- **Reference code:** <https://github.com/jsvir/sparknet> at commit `e66915e`,
  MIT license. Read `model.py`, `inference.py`,
  `conf/cfg_labels_12_channels_16_data_v2.yaml`, and the vendored NeMo
  `parts/submodules/jasper.py` and `modules/audio_preprocessing.py`. Inspected
  the tensors and the stored hyperparameters and callbacks of the released
  `ckpt/kws_C_{4,8,16,32}.ckpt`. Copies of the checkpoints and the license are
  in `models/checkpoints/external/sparknet/`, which is gitignored.
- **12-class run:** metrics JSONL and logs in
  `outputs/full-run-20260912T063022Z`, read 2026-09-13. The run is **not
  running**. It stopped on a `KeyboardInterrupt` at 12:36:40 on 2026-09-13,
  three epochs into candidate w17's PAI phase (best 79.29% so far), although
  `reports/sparsity.yaml` still says `status: running`.
- **Cost measurements:** a scratch SparkNet reimplementation, not committed and
  sketched in Finding 7, measured with this project's `kws.utils.profile`
  hooks.

## Findings

### 1. Where the 12-class pipeline currently stands

Values are best validation accuracy unless noted. Test accuracy comes from
`ERRORS.md`.

| Stage | Params | Accuracy |
|---|---|---|
| Teacher DS-CNN-L | 469,604 | 97.69% val (epoch 57) |
| Student DS-CNN-XS, distilled from scratch | 4,096 | 85.07% val (epoch 167), 83.64% test |
| w18 prune + KD | 1,830 | 79.38% (best at epoch 39 of 40, still rising) |
| w18 PAI (`.fc` dendrite) | 2,058 | 82.68% (epoch 276) |
| w18 resume KD | 2,070 | 82.53% |
| w17 prune + KD | 1,750 | 78.94% |
| w17 PAI | — | 79.29% after 3 epochs, then interrupted |

The project's reference target is Edge Impulse's ~3,800 parameters at ~92%.
Both w18 and w17 are below the search's 0.85 accuracy floor.

**The w18 PAI gain is not a dendrite effect.** PAI starts in neuron mode,
which is plain continued KD on the base model. Before the first restructure at
epoch 105, that base-only phase had already reached **82.58% at epoch 80**. The
best model with the dendrite reached 82.68%, a difference of 0.10 points, about
5 of 5,185 validation clips. One validation standard error is about 0.53
points. The +3.3 points over prune + KD came from more training epochs, which
also means prune + KD's 40-epoch budget is too short.

The historical six-class sparsity report that recorded 91.9% at 2,946 deployed
parameters was removed during the stale-artifact cleanup. It was not a
12-class baseline: its source was `ds_cnn_xs_distilled_warm.pt` (not the
12-class checkpoint), and its 0.90 accuracy floor belonged to the older task.

### 2. Deployment cost measured with the project's profiler

Input is the real feature shape, 40 log-mel bins × 101 frames. MACs count only
Conv2d and Linear layers, exactly as the Pareto search does. Peak activations
use the profiler's liveness estimate, in elements, which equals bytes at int8.

| Model | Params | MACs | Peak activations |
|---|---|---|---|
| DS-CNN-XS | 4,096 | 3,358,320 | 81,600 |
| DS-CNN-XXS (w18 base) | 1,830 | 1,450,656 | 36,720 |
| DS-CNN w15 base | 1,596 | 1,267,020 | 36,720 |
| SparkNet C=16, log-mel 40 | 4,852 | 418,120 | 8,080 |
| SparkNet C=12, log-mel 40 | 3,584 | 295,708 | 8,080 |
| SparkNet C=8, log-mel 40 | 2,508 | 192,688 | 8,080 |
| SparkNet C=16, MFCC 32 (paper input) | 4,636 | 396,304 | 6,464 |
| SparkNet C=8, MFCC 32 (paper input) | 2,356 | 177,336 | 6,464 |
| SparkNet C=4, MFCC 32 (paper input) | 1,504 | 96,940 | 6,464 |

- **Counting method:** the paper reports 454.5K MACs for C=16 using `thop`.
  `thop` also charges every BatchNorm 4 × its input elements and ignores conv
  bias. A `thop`-style count of the scratch model reproduces the paper exactly
  for C=16 (454,480) and approximately for C=32 (1,170,368 against "1.2M").
  This project's hooks count 396,304 for the same C=16 model. Compare ratios,
  not absolute MACs across the two methods.
- **Size relative to XS:** SparkNet C=16 on log-mel 40 has 18% *more*
  parameters than XS (4,852 against 4,096) and 8.0× fewer MACs. C=12 is the
  closest match under XS's parameter budget, with 3,584 parameters and 11.4×
  fewer MACs.
- **Stale shape comments:** model config comments say the input is (40, 98),
  but `FeatureExtractor` emits 101 frames. Models derive `input_shape` from
  data, so only the comments are affected.
- **Receptive field** in feature frames, at a 10 ms hop:
  - DS-CNN-XS sees **13 frames (~130 ms)** before global average pooling: the
    5-wide stride-2 stem, then two 3×3 depthwise convs at jump 2.
  - DS-CNN-L sees 34 frames.
  - SparkNet sees **71 frames (~710 ms)**: 1 + 10 + 14 + 18 + 28.

  Keywords last roughly 300–800 ms. BC-ResNet-0.625 (4,585 parameters) reports
  95.4% on SC2 and SparkNet C=16 (4,636) reports 95.7%, while XS sits at 83.6%
  test here. Those papers use 1× unknown/silence balance and this project uses
  2× (Finding 5), so the gap is suggestive, not controlled. It still points to
  XS's temporal reach as a likely cause of its deficit, more than its parameter
  budget.

### 3. The reference architecture, verified from the released checkpoints

SparkNet processes MFCC frames as 1D sequences, with frequency bins as channels:

| Layer | Kernel | Output | Residual |
|---|---|---|---|
| Time-channel separable conv 1 | 11 | C | no |
| Time-channel separable conv 2 | 15 | C | yes |
| Time-channel separable conv 3 | 19 | C | yes |
| Time-channel separable conv 4 | 29 | C | yes |
| 1×1 gate conv (with bias) → BN → tanh | 1 | 32 | no |
| Mean over time → Linear | — | 12 | no |

Each separable conv is a depthwise conv, a pointwise conv (neither has a bias),
BN, and ReLU, with `same` padding K // 2. The residual path is **not an
identity**: NeMo's `JasperBlock` adds a 1×1 conv plus BN, and the ReLU follows
the residual sum.

**BatchNorm epsilon differs by layer.** The block and residual BNs are built
as `BatchNorm1d(C, eps=1e-3, momentum=0.1)` (vendored `jasper.py:962`). The
gate BN is a default `BatchNorm1d(32)` with eps 1e-5. A port that uses the
default eps everywhere will not reproduce the checkpoints' outputs.

**Gate width.** The paper defines the gate z as "the same size as its input x"
(F channels), so that z can mask x. The released `model.py` hard-codes 32
(`Conv1d(C, 32, 1)`, `Linear(32, 12)`), which equals F for the paper's MFCC-32
input. The released inference path never multiplies z by x, so the width is
free in practice, and SparkNet can consume this project's 40-bin log-mel
features directly. At 40 bins the paper-faithful width would be 40: at C=16
that is 5,100 parameters and 431,144 project MACs, against 4,852 and 418,120
with width 32.

Parameter counts, excluding the checkpoint's preprocessor buffers:

| C | Checkpoint | Paper | Paper MACs (`thop`) | Paper SC2 accuracy |
|---|---|---|---|---|
| 32 | 11,500 | 11,500 | 1.2M | 97.0–97.1% |
| 16 | 4,636 | 4,636 | 454.5K | 95.7% |
| 8 | 2,356 | 2,292 | 190K | 92.1% |
| 4 | 1,504 | 1,416 | 105K | 83.5% |

The paper's C=8 and C=4 rows are below the released checkpoints in parameters
(2,292 against 2,356; 1,416 against 1,504) and in MACs. The `thop`-style count
gives 212,888 and 121,180 against the paper's 190K and 105K. The same counting
method reproduces C=16 exactly, so the released C=8 and C=4 checkpoints are
probably not the models behind those rows, and **their accuracy may not match
92.1% and 83.5%**. The C=32 accuracy appears as 97.0 in Table 3 and 97.1 in
Table 4.

Each released checkpoint is one training run, selected by minimum
`val_loss_gates`: C=16 at epoch 195 (run `seed_4`) and C=8 at epoch 191.

Checkpoint tensor names, needed for the Phase A port:

- **Separable convs:** `fs.encoder.{i}.mconv.0` is the depthwise conv
  `(C_in, 1, K)`; `mconv.1` is the pointwise conv `(C_out, C_in, 1)`; `mconv.2`
  is the BN.
- **Residual path:** `fs.encoder.{i}.res.0.0` is the 1×1 conv and `res.0.1` its
  BN, for i = 1..3.
- **Gate:** `output_layer.0` is the gate conv (with bias) and `output_layer.1`
  its BN.
- **Classifier:** `freq_linear_proj`.
- **Front end:** the preprocessor stores `dct_mat (32, 32)`, a mel filterbank
  `(257, 32)`, and a `window (400,)`.

The Lightning checkpoint pickles OmegaConf/NeMo objects. Loading it needs those
packages installed, or a restricted unpickler. The restricted version resolves
only `torch`, `torch._utils`, `torch.storage`, `collections.OrderedDict`, and
plain builtin containers and scalars, and replaces every other class with an
inert stub. This loads `state_dict` from all four checkpoints (verified).

### 4. What "sparse binarization" does and does not buy

- **Training:** `z = clamp(tanh(u) + 0.5 + ε, 0, 1)` with `ε ~ N(0, 0.5²)`, and
  the classifier sees `fc(mean_t z)`.
- **Inference:** the noise is disabled, so `z = clamp(tanh(u) + 0.5, 0, 1)`.
  This is a continuous hard sigmoid, **not a binary tensor**, and nothing in
  `inference.py` thresholds it. The deployed graph has no binary or sparse
  kernel to exploit.
- **Where the speedup comes from:** entirely from the 1D time-channel separable
  architecture.
- **Loss as actually implemented:** `100·CE + 1·mean(Φ((0.5 + μ)/σ))`, with σ =
  0.5. The paper's "λ = 1e+2" multiplies the cross-entropy, not the sparsity
  term. AdamW is invariant to the overall loss scale, so start a port with a
  sparsity weight of about 0.01 × the CE weight.
- **The ×100 also rescales the SGD recipe.** Relative to a loss with CE weight
  1, SGD on `100·CE + reg` is SGD with 100× the learning rate, 1/100 of the
  weight decay, and a 0.01 sparsity weight. The configured lr 1e-2 and weight
  decay 1e-3 are therefore an effective lr of 1.0 and weight decay of 1e-5 on
  the CE scale. Copying the numbers onto an unscaled loss does not reproduce
  the reference.
- **Ablation:** removing the noise, clipping, and sparsity term costs one point
  (94.7% versus 95.7%). The stochastic gate is a modest regularizer, not the
  source of the efficiency.

### 5. Why the published accuracy will not transfer directly

- **Class balance:** the reference manifests rebalance `_unknown_` and
  `_silence_` to 1× the keyword-class mean. This project uses 2×, a harder task
  and one closer to deployment.
- **Evaluation data:** according to the checkpoint hyperparameters, the
  reference trains and tests on
  `google_speech_commands_v2_bcresnet/{train,valid,test}_manifest_balanced.json`.
  That is a BC-ResNet-style preparation whose script is not released. This
  project samples its own unknown pool from the official lists and synthesizes
  silence with gain 0.1–1.0.
- **Front end:** NeMo `AudioToMFCCPreprocessor`, which wraps
  `torchaudio.transforms.MFCC` with:
  - `log_mels=True`, meaning `ln(mel_power + 1e-6)`;
  - 32 mel bins and 32 coefficients, DCT-II with `ortho` norm;
  - `n_fft` 512, a 400-sample periodic Hann window, and a 160-sample hop;
  - torchaudio defaults for the rest (`center=True`, reflect padding, power 2,
    HTK mel scale, no mel norm).

  With torchaudio 2.11, this configuration reproduces the checkpoint's stored
  `dct_mat`, mel `fb`, and `window` buffers to within 4.3e-6 (verified).

  NeMo right-pads each batch's waveforms with zeros to the longest clip, then
  symmetrically zero-pads or randomly crops the features to 101 frames. Almost
  every batch contains a full 1 s clip, so this matches this project's
  right-pad of every clip to 16,000 samples, which yields 101 frames.

  **This project's existing `type: mfcc` path is not equivalent.** It leaves
  torchaudio's default `log_mels=False`, which is dB with `top_db` 80. The
  project's default front end is log-mel with 40 bins and a 30 ms window.
- **Recipe:**

  | Setting | Reference config | This project |
  |---|---|---|
  | Optimizer | SGD, momentum 0.9, lr 1e-2, weight decay 1e-3, on `100·CE` (see Finding 4) | AdamW, lr 1e-3 |
  | Schedule | warmup 5%, hold 40%, then poly-2 decay to 1e-6 | cosine |
  | Epochs / batch | 200 / 128 | 200 / 256 |
  | Time shift | ±100 ms, p = 0.8 | ±150 ms circular roll |
  | Noise | white noise −90 to −46 dB, p = 0.8 | background noise, 75%, down to −5 dB SNR |
  | Other augmentation | none (no SpecAugment) | speed 0.85–1.15, two SpecAugment mask pairs |
  | View caching | new augmentation every epoch | `cache_train_features` fixes one augmented view per entry for the whole run |

  Notes on the reference column:
  - The paper, the repository YAML, and the C=4, C=8, and C=32 checkpoints
    agree on it.
  - The released **C=16 checkpoint differs:** its stored hyperparameters show
    lr 0.1 and hold ratio 0.
  - C=32 additionally used Freesound background noise at 0–20 dB SNR.
  - The paper's prose describes the decay as covering "the remaining 85%",
    which does not add up with 5% warmup and 40% hold.

- **Seeds:** the paper reports mean ± std but does not say over how many runs,
  and each released checkpoint is a single run. This project's comparisons are
  single-seed.

### 6. Confound: tiny students underfit the current recipe

Last-epoch training accuracy, measured in train mode with dropout on and the
fixed augmented views, compared with validation accuracy:

| Model | Train accuracy | Val accuracy |
|---|---|---|
| XS student | 64.7% | 84.6% |
| w18 PAI | 60.0% | 82.1% |
| Teacher | 96.1% | 97.4% |

The augmentation and dropout strength suits DS-CNN-L, not a few-thousand-parameter
student. A SparkNet A/B that changes architecture and recipe together will
not show which one mattered. A lighter recipe is also an independent,
architecture-free lever for the existing DS-CNN students.

### 7. Fit with the compression framework

**Implementation shape that keeps most of the framework working:** build the 1D
convs as `nn.Conv2d` with `(1, K)` kernels on a `(B, F, 1, T)` view of the
`(B, 1, F, T)` input. Use `BatchNorm2d`, name the classifier `.fc`, and expose
`forward_features` (returning `mean_t z`, shape `(B, 32)`) and
`classify_features`. Keep the 40-bin log-mel input, so the fixed teacher and
the student share one feature view. This is the shape used for Finding 2:

```python
class TCSBlock(nn.Module):  # register each Conv2d directly before its own BN
    def __init__(self, cin, cout, k, residual, bn_eps=1e-3):  # 1e-3 matches NeMo
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, (1, k), padding=(0, k // 2), groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=bn_eps)
        self.res_conv = nn.Conv2d(cin, cout, 1, bias=False) if residual else None
        self.res_bn = nn.BatchNorm2d(cout, eps=bn_eps) if residual else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.bn(self.pointwise(self.depthwise(x)))
        if self.res_conv is not None:
            y = y + self.res_bn(self.res_conv(x))
        return self.relu(y)

# SparkNet.forward: x.permute(0, 2, 1, 3) -> 4 TCSBlocks (K = 11, 15, 19, 29;
# residual on blocks 2-4) -> gate_conv (bias) -> gate_bn (default eps) -> tanh
# -> (+ N(0, 0.5^2) noise in training) -> clamp(+0.5, 0, 1) -> mean over time -> fc
```

**Expected to carry over with that shape:**

- **Cost profiling:** `count_macs` hooks only `Conv2d`/`Linear` and reads
  `kernel_size[0] * kernel_size[1]` (`src/kws/utils/profile.py:64`, `:86`). A
  `Conv1d` implementation would silently contribute zero MACs, which is the
  main reason to use `Conv2d`. `weight_memory_bytes` and
  `deployed_parameter_count` count every float parameter. After quantization,
  though, they unpack only `nnq.Conv2d`/`nnq.Linear` (`:186`, `:208`), so a
  quantized `Conv1d` would be dropped as well.
- **Clustering:** `CLUSTERABLE_TYPES` is `Conv2d`/`Linear`. Every layer has ≥ 64
  weights at C=16. At C=8 the 8×8 pointwise and residual convs sit exactly at
  the `min_weights: 64` threshold.
- **N:M masks:** the depthwise kernels have input dimensions 11, 15, 19, and 29,
  so they are skipped as indivisible. The pointwise, gate, and classifier layers
  are eligible.
- **QAT fusion:** Conv+BN fusion discovers adjacent `Conv2d` → `BatchNorm2d`
  children in registration order (`src/kws/optimize/quantize_qat.py:186`). The
  block above must keep each conv registered immediately before its own BN.
- **KD:** the feature adapter becomes `Linear(32, 276)`, and
  `supports_pooled_features` accepts a 2D feature tensor.
- **PAI:** the `fc_only` conversion targets the module id `.fc`.
- **Noise handling:** `evaluate_loss_acc` runs in eval mode, so validation sees
  the noise-free gate.
- **Export:** ONNX export and TorchScript benchmarking consume a live module.

**Needs new code:**

- **Model registry.** `build_ds_cnn` is called directly for the student in
  `src/kws/train.py:632`, `src/kws/optimize/distill.py:154`,
  `src/kws/evaluate.py:26`, `src/kws/optimize/prune.py:391`,
  `src/kws/optimize/dendritic.py:577`, and
  `src/kws/optimize/quantize_qat.py:576`. The teacher builds in
  `src/kws/optimize/kd.py:162` and `src/kws/pipeline.py:259` can stay DS-CNN.
- **Auxiliary-loss hook.** Both loss sites compute the loss inline:
  `run_finetune` (`src/kws/train.py:168`) computes plain CE at `:355` and the
  KD loss at `:366`, and the PAI loop does so at
  `src/kws/optimize/dendritic.py:1990`. The sparsity term must reach all of
  them.
- **Residual-aware structured pruning.** `prune_ds_cnn`
  (`src/kws/optimize/prune.py:69`) threads keep-masks through a plain chain. In
  SparkNet, block 1's output and every residual branch share one channel set,
  so a single keep-mask must cover all four blocks plus the residual convs. The
  gate width is a second pruning axis, and kernel length a possible third.
- **Sweep plumbing keyed on DS-CNN structure:**
  - `_source_block_width` and `_target_model_cfg`
    (`src/kws/optimize/dendritic_prune_loop.py:273`, `:286`) read
    `block_channels`;
  - `build_cycle_base` and `estimate_one_dendrite_params`
    (`src/kws/optimize/dendritic.py:568`, `:605`) assume `.stem` and `.blocks`;
  - `configure_perforatedai` tracks `DSConvBlock`
    (`src/kws/optimize/dendritic.py:1460`–`1468`);
  - the pipeline reads `block_channels[0]` (`src/kws/pipeline.py:630`).
- **Eager int8 conversion.** The residual `+` needs `FloatFunctional.add`.
  The `+ 0.5` can be removed from the graph instead of using `add_scalar`,
  because `clamp(t + 0.5, 0, 1) = clamp(t, -0.5, 0.5) + 0.5`. After the mean,
  the constant becomes `0.5 · W.sum(dim=1)`, which folds into the `fc` bias.
  Tanh uses fixed quantization parameters. Whether this torch version's eager
  `convert` handles `Tanh`, the clamp, and the mean has **not** been tested.
  Add a convert-and-parity test.
- **Activation-memory estimate.** The leaf-pair estimate does not charge the
  residual input a block holds while its main branch runs, which undercounts
  by up to C×T elements per block. That changes the reported peak only when
  3·C·T exceeds the 2·F·T input peak, which means C ≥ 27 at 40 bins. For
  example, C=32 on MFCC-32 needs 9,696 elements, not the 6,464 reported. The
  C=16 peak of 8,080 is unaffected. Charge it explicitly anyway.
- **Target runtime.** Whether ESP-DL or TFLite Micro supports int8 Tanh, Clip
  (or Min/Max), Add, and Mean on the ESP32 has **not** been verified.

### 8. Effect on the dendrite story

**The current DS-CNN evidence for dendrites is unproven.** Finding 1 shows that
w18's PAI gain matches what PAI's own base-only phase reached before any
dendrite was added. Any dendrite claim, on DS-CNN or SparkNet, needs a
no-dendrite control trained for the same number of epochs.

At C=16, SparkNet's reported accuracy leaves little gap for dendrites to close.
The PAI claim becomes "SparkNet C=8 or C=12 plus dendrites approaches C=16 at
fewer parameters or MACs," and it has to beat that matched-epoch control.

An `.fc`-only dendrite on SparkNet only makes a 32→12 head nonlinear. The
reference ablation does not settle whether the head is a bottleneck. Its
"auxiliary larger classifier" was a separate MatchboxNet-4x1x64 that read the
gated input x⊙z and added a second CE loss during training (95.6% against
95.7%), not a larger replacement head. Whether an `.fc` dendrite helps
SparkNet, or whether the gate conv is the better perforation target, has to be
measured.

## Plan

### Phase A: score the released checkpoints on this split (no training)

Scope: no training, no PAI, and no `.env`. Do not touch the pruning, PAI,
clustering, or QAT code. Inputs are the checkpoints in
`models/checkpoints/external/sparknet/` (see Sources) and the run's XS student
at
`outputs/full-run-20260912T063022Z/models/checkpoints/student/ds_cnn_xs_distilled_warm_12class.pt`
(12 classes, val_acc 0.8507).

- [x] **A1. Front-end option.**
  - Add `log_mels: bool = False` to `FeatureExtractor`
    (`src/kws/data/features.py`) and pass it to `torchaudio.transforms.MFCC`.
    The default must leave current behavior unchanged.
  - Both construction sites, `build_feature_extractor` and
    `src/kws/data/dataset.py:214`, read `features.get("log_mels", False)`.
  - Add `configs/data/speech_commands_v2_mfcc32.yaml`, a copy of
    `speech_commands_v2.yaml` whose `features` block is `type: mfcc`,
    `n_mels: 32`, `win_length_ms: 25`, `hop_length_ms: 10`, `log_mels: true`.
    This yields `(1, 32, 101)` for a 16,000-sample clip.
  - Test: the extractor matches `torchaudio.transforms.MFCC(sample_rate=16000,
    n_mfcc=32, log_mels=True, melkwargs=dict(n_fft=512, win_length=400,
    hop_length=160, n_mels=32))` and differs from `log_mels=False`.
- [x] **A2. Model.** Add `src/kws/models/sparknet.py`:
  - `SparkNet(n_feat, num_classes, channels=16, gate_channels=32,
    kernels=(11, 15, 19, 29), block_bn_eps=1e-3)` in the Finding 7 shape.
    Modules are `blocks` (four `TCSBlock`s, residual on the last three),
    `gate_conv` (`Conv2d(C, G, 1)` with bias), `gate_bn` (`BatchNorm2d(G)` with
    the default eps), and `fc` (`Linear(G, num_classes)`).
  - Input is `(B, 1, F, T)`. `forward_features` returns `mean_t z` with shape
    `(B, G)`. `classify_features` is `fc`, and `forward` composes the two. The
    N(0, 0.5²) noise is added only in training mode.
  - Add `build_sparknet(model_cfg, input_shape, num_classes)` mirroring
    `build_ds_cnn`, with `model_cfg` keys `name`, `channels`, and
    `gate_channels`.
- [x] **A3. Port.** Add `src/kws/models/sparknet_port.py`, run as
      `python -m kws.models.sparknet_port --ckpt … --data-config … --out …`.
  - Load with the restricted unpickler from Finding 3.
  - Map `fs.encoder.{i}.mconv.0` → `blocks.{i}.depthwise`, `mconv.1` →
    `pointwise`, `mconv.2` → `bn`, `res.0.0` → `res_conv`, `res.0.1` →
    `res_bn`, `output_layer.0` → `gate_conv`, `output_layer.1` → `gate_bn`,
    and `freq_linear_proj` → `fc`. Conv weights take `unsqueeze(2)`.
  - Infer C and G from the tensor shapes. Load with `strict=True`, and fail if
    any non-`preprocessor.*` source key is left unused.
  - Build the `FeatureExtractor` from `--data-config` and fail unless its MFCC
    `dct_mat`, `MelSpectrogram.mel_scale.fb`, and
    `MelSpectrogram.spectrogram.window` match the checkpoint's
    `preprocessor.featurizer.*` buffers (max abs diff < 1e-4).
  - The reference label order is `yes no up down left right on off stop go
    _unknown_ _silence_`. Assert that it equals `build_label_map` for the data
    config.
  - Write a project-style checkpoint with `model_family: sparknet`,
    `model_cfg`, `input_shape: [32, 101]`, `num_classes: 12`,
    `num_keywords: 10`, `label_map`, `model_state_dict`, and `source`
    (path, upstream commit, sha256).
- [x] **A4. Evaluate.** In `src/kws/evaluate.py`:
  - `load_model_from_checkpoint` dispatches on
    `ckpt.get("model_family", "ds_cnn")`, so existing checkpoints are
    unaffected.
  - Add `--split {test,val}`, defaulting to `test`.
  - Fail clearly if the data config's `(n_mels, frames)` differs from the
    checkpoint's `input_shape`.
  - Add `split` to the JSON report.
- [x] **A5. Tests** in `tests/test_sparknet.py`, none of which may need the
      gitignored checkpoints:
  - **Parity.** Build a random reference-format state dict for C=8 and G=32,
    with positive running variances. Compute an independent reference forward
    with `F.conv1d` and `F.batch_norm` (eps 1e-3 in blocks, 1e-5 at the gate,
    ReLU after the residual sum). Port it through the A3 mapping and match the
    eval-mode `SparkNet` output to 1e-5.
  - **Costs.** Pin `kws.utils.profile.count_macs` and parameter counts to the
    Finding 2 numbers: C=16 log-mel 40 is 4,852 params and 418,120 MACs; C=16
    MFCC-32 is 4,636 and 396,304; C=8 MFCC-32 is 2,356 and 177,336.
  - **Dispatch.** A SparkNet checkpoint saved to `tmp_path` loads through
    `load_model_from_checkpoint`.
  - `uv run python -m pytest tests/` must pass. (`uv run pytest` fails to spawn
    in this environment.)
- [x] **A6. Runs.** Port C=16 and C=8 to
      `models/checkpoints/sparknet_c{16,8}_ported.pt`. Evaluate each SparkNet
      checkpoint with the MFCC-32 config, and the XS student with the default
      config, on `val` and on `test`.
  - Use one split per process, because synthesized silence draws from the
    global RNG in order.
  - Keep the default seed 0.
  - Write the JSON reports to `reports/phase_a/`.
- [x] **A7. Record** `reports/sparknet_phase_a.yaml`, which is not gitignored.
      For each model and split, record:
  - accuracy, FAR, FRR, and per-class F1;
  - keyword-only accuracy (the diagonal over the first 10 confusion rows);
  - parameters, and project MACs.

  This is a one-off characterization of an external model, and later model
  selection stays on validation only.
- **Harness check:** the XS test accuracy should be close to the recorded
  0.8364. Record any gap larger than 0.5 points. All Gate A comparisons use XS
  numbers from this same harness.
- **Port check:** the paper reports 95.7% for C=16. If C=16's keyword-only test
  accuracy is below 0.90, suspect the port or the front end before the task
  difference.
- **Gate A:** proceed if C=16 is clearly above XS under the same harness. The
  proposed bar is ≥ 90% test accuracy. If it falls short, read the per-class
  confusion before concluding anything about the architecture.

#### Phase A results (2026-09-13)

Phase A is done. Numbers come from `reports/sparknet_phase_a.yaml`, seed 0.

| Model | Params | MACs | Val acc | Test acc | Test keyword-only | Test acc excl. silence | Test FAR (unknown / silence) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SparkNet C=16 | 4,636 | 396,304 | 0.8330 | 0.8454 | 0.9416 | 0.9149 | 0.216 / 0.571 |
| SparkNet C=8 | 2,356 | 177,336 | 0.7290 | 0.7344 | 0.8746 | 0.8216 | 0.404 / 0.788 |
| DS-CNN-XS | 4,096 | 3,358,320 | 0.8511 | 0.8364 | 0.8471 | 0.8096 | 0.368 / 0.002 |

- **Checks.**
  - The harness reproduces XS at 0.8364.
  - The port is exact: the raw upstream ckpt run through a functional forward
    gives a max logit difference of 0.0.
  - The full 4,074-clip keyword test set, scored outside the harness with
    soundfile and torchaudio MFCC, matches the harness per-class diagonal
    exactly for both checkpoints.
- **Gate A: not met as written, and not decidable from the released
  checkpoints.** The whole shortfall is the synthesized silence class:
  - C=16 labels 451 of 815 test silence clips as `up`.
  - A gain sweep over the six noise files shows the cause:
    - Quiet noise stays silence: gain ≤ 0.001 always, gain 0.1 about 77% of
      the time.
    - Loud noise becomes `up`: about 73% at gain 1.0, and `pink_noise.wav`
      at every gain ≥ 0.1.
  - The released model never saw this project's gain 0.1–1.0 noise labelled as
    silence, so this is a data mismatch.
  - Class balance does not explain it, because FAR is a rate.
  - On everything else, C=16 clearly beats XS:
    - keyword-only accuracy +9.5 points;
    - unknown false accepts 0.216 vs 0.368;
    - accuracy excluding silence +10.5 points;
    - 8.5× fewer MACs.
  - C=8 is not competitive.
- **Decision.** Proceed to Phase B. Phase B trains on this project's silence,
  so the confound goes away there. Gate B, not Gate A, is the real test.
  - C=16 is over XS's parameter budget, so C=12 is the gated candidate and C=16
    is the reference.
  - In Phase B, watch the `_silence_` and `up` F1 scores.

### Phase B: controlled step-2 A/B in this project

- **Front end: 40-bin log-mel for every run**, the teacher's and XS's input.
  - `kd.py:315` feeds the teacher and the student the same input tensor, so a
    KD run cannot give SparkNet MFCC-32.
  - The shared input also keeps the XS comparison on identical features.
  - Phase A's MFCC-32 path exists only to verify the port.
  - Gate width stays at the released 32; see Open questions.
- **Size: C=12 is the gated candidate.**
  - Gate B caps parameters at XS's 4,096.
  - At log-mel 40, C=16 has 4,852 parameters and is over the cap. It is trained
    only as a reference point.
  - C=12 has 3,584 parameters (3,800 with gate width 40) and 295,708 MACs, so
    it fits.
- [x] **Minimum code.** Reuse Phase A's `src/kws/models/sparknet.py`. Replace
      the evaluate dispatch with a model registry covering only the train,
      distill, and evaluate build sites.
  - Done in `src/kws/models/registry.py`. A model config selects its builder
    with `family` (default `ds_cnn`). Checkpoints written by train and distill
    also record `model_family`, the key the Phase A port uses.
  - `kws.train` gained `--stage` (default `teacher`) so a student run does not
    write under `teacher/`.
  - Model configs: `configs/model/sparknet_c{8,12,16}.yaml`.
  - Still DS-CNN only: the KD teacher, pipeline, prune, PAI, and QAT build
    sites (Phase C).
- [x] **Sparsity loss.** Add the auxiliary-loss hook to `run_finetune`, with the
      weight configurable and starting near 0.01 × the CE weight.
  - `SparkNet.auxiliary_losses()` returns the reference term
    `mean(Φ((μ + 0.5) / 0.5))` over the pre-noise gate μ, with its weight.
  - `run_finetune` adds the weighted term to the total in both the plain and
    KD paths, and logs the unweighted value as `train_gate_sparsity`.
  - The weight is `sparsity_weight: 0.01` in the SparkNet model config, not the
    train config, so SparkNet and XS runs share every recipe file verbatim.
  - The PAI loop (`src/kws/optimize/dendritic.py:1990`) does not call the hook
    yet (Phase C).
- [x] **Light recipe.** Add a train config that approximates the reference
      recipe: ±100 ms shift, low-level white noise, no SpecAugment, no speed
      perturbation, and no fixed augmented-view cache. Keep AdamW. If SGD is
      ever tried, apply Finding 4's ×100 rescaling rather than copying the
      reference lr and weight decay.
  - `configs/train/light.yaml`: ±100 ms zero-fill shift and −90 to −46 dB
    white noise, each with p = 0.8, as in the reference YAML. No background
    noise mixing, speed perturbation, or SpecAugment, and
    `cache_train_features: false`. `configs/train/light_kd.yaml` is the same
    plus response + CE KD with weights 1/7 and 6/7 (Song et al.'s 0.1 / 0.6
    renormalized).
  - Train configs take an optional `augmentation` block
    (`kws.data.augment.build_augmenters`). Without one, the recipe and its
    random draws are unchanged (tested). Unknown keys are rejected. All six
    augmented `build_datasets` call sites pass the block through.
  - Measured cost on the M3 Pro (MPS): C=12 about 5 s per epoch without KD
    and 42 s with KD, where the DS-CNN-L teacher forward dominates.
- [ ] **Runs, in priority order,** selected on validation:
  1. SparkNet C=12, light recipe, no KD.
  2. SparkNet C=12, light recipe, response + CE KD (`feature_weight: 0`).
  3. DS-CNN-XS, light recipe (architecture control).
  4. SparkNet C=12 with the current `distill_imc.yaml` recipe (recipe control).
  5. SparkNet C=16 (over-budget reference) and C=8, with the best recipe found
     above.
  6. The three-term Song et al. KD, only if run 2 helped.
  7. Optional: SparkNet C=12 on MFCC-32, light recipe, no KD. Only this no-KD
     run can use MFCC-32, and it answers the front-end open question.
- **Gate B:** SparkNet C=12 (or any width ≤ 4,096 parameters) beats DS-CNN-XS
  **under the same recipe** by a material validation margin. Report test
  accuracy once for the selected models.

### Phase C: move the whole framework to SparkNet (only if Gate B passes)

- [ ] Route every student build site through the registry.
- [ ] Add residual-aware structured pruning with a shared keep-mask, plus
      regression tests for output equivalence at `keep_ratio: 1.0`.
- [ ] Generalize the sweep variable from `block_channels` to a width spec
      covering C, and optionally gate width.
- [ ] Lengthen prune + KD, which Finding 1 shows was still improving at
      epoch 40.
- [ ] Before any dendrite claim, add a no-dendrite control that trains for the
      same number of epochs as the PAI run.
- [ ] Make PAI module tracking architecture-aware, and add the sparsity term to
      the PAI loss.
- [ ] Make QAT work: `FloatFunctional` for the residual add, the `+ 0.5`
      folded into the `fc` bias, a convert test covering Tanh, clamp, and mean,
      and a TorchScript and ONNX parity test.
- [ ] Charge residual liveness in the activation estimate, with a test.
- [ ] Confirm the int8 operator coverage of the ESP32 target runtime.
- [ ] Update the README model table, configs, and pipeline defaults.
- [ ] Rerun step 3 as a width sweep and compare its Pareto frontier with the
      DS-CNN frontier.

### Phase D: deployment follow-ups

- **Streaming.** SparkNet has no striding and pools globally over time, so live
  microphone inference can cache per-layer ring buffers and pay roughly MACs/T
  per 10 ms hop. The `same` zero padding used in clip-level training differs
  from a streaming context. Measure that gap before relying on streaming output.

## Open questions

- **Front end:** if Phase B run 7 (MFCC-32, no KD) clearly beats run 1
  (log-mel 40, no KD), then either retrain the teacher on MFCC or teach the
  dataset and cache to return two feature views so KD can use MFCC-32.
- **KD with stochastic gates:** does feature KD (pooled gate occupancy projected
  onto the teacher embedding) conflict with the sparsity objective?
- **Dendrite target:** `.fc` or the gate conv, and at which C?
- **Gate width on log-mel 40:** keep the released 32, or use the
  paper-faithful 40? Gate width 40 adds 216 parameters at C=12 (3,800, still
  under XS) and 248 at C=16.
- **Class balance:** 2× unknown/silence stays the task definition here. Record
  1× numbers only as a bridge to the literature.
