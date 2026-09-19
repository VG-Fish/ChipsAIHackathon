# SparkNet paper MAC audit

Primary sources: Svirsky, Shaham, and Lindenbaum, [“Sparse Binarization for Fast Keyword Spotting”](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf), Interspeech 2024, pp. 3010–3014; and the [author-linked repository](https://github.com/jsvir/sparknet) at commit [`e66915e8ad3c2c6be96781fa1516953dd4b50718`](https://github.com/jsvir/sparknet/tree/e66915e8ad3c2c6be96781fa1516953dd4b50718). The ISCA record itself links that repository.

## Architecture and input counted by the paper

SparkNet consumes an MFCC matrix `F × T`. The experiment uses 16 kHz, one-second Speech Commands clips and `F=32` MFCC coefficients. Four time-channel-separable 1-D blocks use kernels `11, 15, 19, 29`; each is depthwise temporal convolution followed by pointwise channel convolution, batch normalization, and ReLU. Every block has stride 1 and dilation 1, and the final three have residual branches. A `1×1` convolution maps `C` channels back to `F`, followed by BN and tanh; the resulting gate matrix is clipped after the training-time Gaussian relaxation, averaged over time, and projected from `F` to 12 classes. `C` is the width varied in the reported models. See [paper §2.2 and Table 1, PDF pp. 2–3 (paper pp. 3011–3012)](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf).

The released configuration makes the paper's omitted shape details concrete: `T=101`, 25 ms Hann windows, 10 ms stride, FFT 512, 32 Mel bins, 32 MFCC coefficients, and crop/pad to 101 frames ([configuration, lines 1–119](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/conf/cfg_labels_12_channels_16_data_v2.yaml#L1-L119)). The released model hard-codes the output convolution to 32 channels and the classifier to `32 → 12` ([model.py, lines 260–270](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/model.py#L260-L270)); its forward averages the clipped gates over time ([model.py, lines 125–145](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/model.py#L125-L145)).

## Published Pareto data

These are the paper's values, not recomputed values. Accuracy is top-1 percent on the official 12-class Speech Commands test split; `±` is the reported standard deviation.

| Model | Params | MACs | SC1 accuracy | SC2 accuracy |
|---|---:|---:|---:|---:|
| TinySpeech-X | 10.8K | 10.9M | 94.6 ± 0.00 | — |
| res8-narrow | 19.9K | 5.65M | 90.1 ± 0.98 | — |
| DS-ResNet10 | 10K | 5.8M | 95.2 ± 0.36 | — |
| BC-ResNet-1 | 9,232 | 3.6M | 96.6 ± 0.21 | 96.9 ± 0.30 |
| **SparkNet C=32** | **11,500** | **1.2M** | **96.2 ± 0.19** | **97.0 ± 0.18** |
| TinySpeech-Z | 2.7K | 2.6M | 92.4 ± 0.00 | — |
| BC-ResNet-0.625 | 4,585 | 1.9M | 95.2 ± 0.37 | 95.4 ± 0.31 |
| **SparkNet C=16** | **4,636** | **454.5K** | **95.3 ± 0.33** | **95.7 ± 0.17** |

Source: [paper Table 3, PDF p. 4 (paper p. 3013)](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf). Table 4 reports the width sweep:

| SparkNet width | Params | MACs | SC1 accuracy | SC2 accuracy |
|---:|---:|---:|---:|---:|
| C=32 | 11,500 | 1.2M | 96.2 ± 0.19 | 97.1 ± 0.30 |
| C=16 | 4,636 | 454.5K | 95.3 ± 0.33 | 95.7 ± 0.30 |
| C=8 | 2,292 | 190K | 91.6 ± 0.76 | 92.1 ± 0.33 |
| C=4 | 1,416 | 105K | 82.3 ± 1.91 | 83.5 ± 0.60 |

Source: [paper Table 4, PDF p. 4 (paper p. 3013)](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf). Table 3 and Table 4 disagree slightly for the SC2 uncertainty of C=16 (`±0.17` versus `±0.30`) and for the C=32 SC2 result (`97.0 ± 0.18` versus `97.1 ± 0.30`); comparisons should preserve the table being cited rather than silently combine them.

For the equal-size noise comparison, Table 2 reports BC-ResNet-0.625 versus SparkNet C=16 at 0/5/10/15/20 dB and clean as `75.72/84.47/90.68/92.67/94.18/95.40` versus `75.98/85.05/91.37/93.58/94.63/95.70`, with standard deviations shown in the paper. This supports the robustness claim but does not define another size/MAC point ([paper Table 2, PDF p. 4](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf)).

## Counting convention and comparison to this repository

The paper explicitly says MACs were calculated with PyTorch-OpCounter/THOP (paper §3.1 footnote 1). The released code confirms `thop.profile` and pins `thop==0.1.1.post2209072238` ([requirements.txt, line 22](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/requirements.txt#L22)). It profiles a batch-one `1 × 32 × 101` feature tensor plus length 101 ([train.py, lines 40–44](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/train.py#L40-L44)). The profiled wrapper starts at the four-block feature selector and includes the output convolution and linear classifier, so the published MACs **exclude waveform-to-MFCC preprocessing** ([train.py, lines 13–26](https://github.com/jsvir/sparknet/blob/e66915e8ad3c2c6be96781fa1516953dd4b50718/train.py#L13-L26)). They are theoretical operation counts, not measured latency.

This repository's [`count_macs`](../../src/kws/utils/profile.py#L114) uses forward hooks and counts only convolution and linear products, plus explicit learned dendritic skip-edge operations. It does not count BN, ReLU/tanh, clipping, pooling, ordinary residual addition, MFCC extraction, memory traffic, or hardware-specific kernel behavior. The two counters therefore do not share an operation registry even though both use batch 1 and `32 × 101` features.

For this repository's scalar-width SparkNet at 101 frames, the graph gives
`parameters(C) = 6C² + 141C + 844` and local-hook
`MACs(C) = 606C² + 12,827C + 35,936`. Wrapping one `C → C`
pointwise module adds `C² + C` parameters and `101(C² + C)` local MACs:
the copied convolution contributes `C²` parameters and `101C²` MACs, and the
learned channelwise skip contributes `C` parameters and `101C` operations.
The completed pointwise arm wraps two modules, so its overhead is twice this.

The mismatch is observable on the conventional graph: this repository reports C=16 as 396,304 MACs versus the paper's 454.5K, and C=8 as 177,336 versus 190K ([local regression assertions](../../tests/test_sparknet.py#L109)). The offsets are width-dependent, so a single global conversion factor is not defensible. The recent C8 pointwise-input-scale-75 clean export is 2,500 parameters and 191,880 local-counter MACs, versus its 2,356-parameter/177,336-MAC local base: **+144 parameters (+6.11%) and +14,544 MACs (+8.20%)**. Its raw 191,880 happens to sit 0.99% above the paper's C8 `190K`, but that cross-counter comparison is misleading; the valid statement is the within-counter +8.20% cost.

As a direct counter check, profiling this repository's generated graphs with the
paper's pinned `thop==0.1.1.post2209072238` gives:

| Local graph | Parameters seen by THOP | THOP MACs | Paper row |
|---|---:|---:|---:|
| C=4 base | 1,504 | 121,180 | 1,416 / 105K |
| C=8 base | 2,356 | 212,888 | 2,292 / 190K |
| C=16 base | 4,636 | 454,480 | 4,636 / 454.5K |
| C=32 base | 11,500 | 1,170,368 | 11,500 / 1.2M |
| C=8 pointwise-input-scale-75 | 2,484 | 225,816 | no paper row |

This was a batch-one `1 × 1 × 32 × 101` forward pass over the local models.
The C=16 result reproduces the published value exactly before decimal
rounding, and C=32 rounds to the published precision. The low-width rows do
not reproduce the paper in parameters or MACs, so their discrepancy is in the
represented graph or published table rather than in a global THOP-versus-hook
conversion. THOP does not discover the clean dendritic wrapper's 16
`ParameterList` skip weights or its 1,616 custom elementwise skip operations.
Adding those explicit deployment costs makes the clean dendritic point
2,500 parameters / 227,432 THOP-plus-skip MACs, still **+144 parameters and
+14,544 MACs** over the matching local C=8 base.

There is also a parameter-count caveat. The local C=16 graph matches the paper at 4,636 parameters, while the local C=8 graph has 2,356 versus the paper's 2,292. Both use 32 MFCC bins, so the 64-parameter gap cannot be attributed to a different feature-bin count. The released repository supplies only a C=16 YAML even though it publishes C=4/8/16/32 checkpoints; the primary sources do not explain the low-width discrepancy. Until the checkpoint variants are reconstructed and profiled under both tools, cite local and paper parameter counts separately.

For a defensible frontier plot, use one of two treatments: (1) keep the paper points labeled “published THOP MACs” and local points labeled “project hook MACs,” without ordering across conventions; or (2) run pinned THOP and the local counter on the same conventional and clean dendritic graphs at `1 × 32 × 101`, then use one convention for every plotted point. Add frontend MACs only as a separately defined end-to-end metric for all models.
