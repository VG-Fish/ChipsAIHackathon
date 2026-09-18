# DS-CNN + dendrite pipeline research

**Scope.** Primary-source research for the proposed replacement of SparkNet with
depthwise-separable CNNs (DS-CNNs), retaining dendritic training and targeting a
Cortex-M0+/RP2040-class deployment through TFLite Micro (TFLM). Research-only:
no training run and no model/source-code edits were made. Existing project notes
were consulted, especially `agent-perforatedai-primary-research.md` and
`agent-sparknet-pico-deployment-research.md`; general RP2040 facts are therefore
not repeated here.

Evidence labels used below:

- **Fact:** directly stated by or observable in the cited primary source (paper,
  first-party repository, or official API/specification).
- **Inference:** engineering recommendation derived from those facts and/or the
  current repository; it still needs a run or export test.
- **Unknown:** no primary-source evidence was found, or the cited source does
  not establish the stronger claim.

## Executive recommendation

**Inference:** use two experimental arms:

1. **Deployability-first:** start from DS-CNN-XXS or XS, prune by channel width,
   distill from the existing DS-CNN-L teacher, then add at most one PAI dendrite
   to a late pointwise convolution and/or the classifier. Export a clean fixed
   graph, require full int8 conversion, and validate it with a minimal TFLM
   resolver. Do not initially perforate depthwise convolutions.
2. **Accuracy-first:** start from DS-CNN-S, prune to a small set of widths, use
   KD before and after a one-dendrite PAI phase, and place the dendrite on the
   last pointwise block plus `fc`. Keep the same clean-graph, int8, and TFLM
   gate. This arm tests whether extra dendritic capacity is useful when the
   base network is not already extremely narrow.

The reason for avoiding depthwise perforation in the first pass is not that it
is impossible. PAI documents generic `Conv2d` support, and PyTorch represents a
depthwise convolution as `Conv2d(groups=in_channels)` ([PyTorch Conv2d
documentation](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L46-L85)). However, no official PAI
example or test was found for grouped/depthwise Conv2d, and PAI dendrites are
copies of selected modules. **Unknown:** whether every PAI version preserves
the expected grouped-convolution behavior and can later be cleaned/exported.

## 1. DS-CNN architecture and MCU evidence

### 1.1 Architecture owned by the reference paper/code

**Fact:** Arm's *Hello Edge: Keyword Spotting on Microcontrollers* defines DS-CNN
as a regular first convolution followed by depthwise-separable convolutions;
each separable layer performs per-channel spatial filtering and then a 1x1
pointwise convolution to combine channels. The paper finishes with average
pooling and a fully connected layer. [Paper, §3.5](https://arxiv.org/abs/1711.07128)

**Fact:** the paper's reference implementation makes the topology explicit:
the first layer is regular `convolution2d`; later layers call
`separable_convolution2d` with `depth_multiplier=1`, followed by batch
normalization and a 1x1 pointwise convolution; global average pooling and a
final fully connected classifier follow. [Arm `models.py`,
`create_ds_cnn_model`](https://raw.githubusercontent.com/ARM-software/ML-KWS-for-MCU/master/models.py#L995-L1103)

**Fact:** the paper's best 8-bit-weight/activation points for 10-keyword KWS
were:

| Budget class | Test accuracy | Stored memory | Ops/inference |
| --- | ---: | ---: | ---: |
| Small | 94.4% | 38.6 KB | 5.4 M |
| Medium | 94.9% | 189.2 KB | 19.8 M |
| Large | 95.4% | 497.6 KB | 56.9 M |

These are the authors' search points, not predictions for this repository's
frontend, class count, or RP2040 runtime. [Hello Edge Table 5](https://arxiv.org/pdf/1711.07128#page=7)

**Fact:** the paper expands the search below the three headline budgets and
reports scaled-down DS-CNNs below 8 KB and 500 K operations; it says these
models outperform DNNs at similar operations while using more than 10x less
memory. [Hello Edge, §4.3/Figure 7](https://arxiv.org/pdf/1711.07128#page=8)

**Inference:** the paper supports using an intentionally small DS-CNN backbone
before adding dendrites. The project should compare a genuinely tiny base plus
one learned residual branch against a wider conventional base at the same
measured clean-graph cost, rather than comparing a dendritic model to an
unconstrained DS-CNN.

### 1.2 Match to this repository

**Fact (repository):** `KWS_Model/src/kws/models/ds_cnn.py` implements a fixed
feature-map graph: regular Conv2d stem, a sequence of `DSConvBlock`s, fixed
average pooling, and Linear classification. `layers.py` defines each block as
depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU. The fixed pool was
chosen to avoid dynamic shapes.

**Fact (repository):** the supplied configurations intentionally cover a useful
size ladder: DS-CNN-XXS (1,830 parameters), XS (4,096), S (24,188), M (147,940),
and L (469,604), for the current 12-class, (40, 98)-feature task. These counts
are configuration comments and should be re-measured after any class/frontend
change: [`configs/model/ds_cnn_*.yaml`](../../configs/model).

**Fact (repository):** `prune_ds_cnn` performs actual width surgery. It selects
pointwise output channels by L1 norm, threads the surviving indices into the
next depthwise convolution and BN, then slices the final classifier input:
[`src/kws/optimize/prune.py`](../../src/kws/optimize/prune.py).

**Inference:** this existing channel-surgery path is the right pruning primitive
for a dense TFLM deployment. It changes tensor shapes and therefore can reduce
weights, MACs, activation storage, and scratch buffers. A one-time zero mask
without shape surgery should not be counted as a speed or memory win unless a
target sparse kernel is actually used.

## 2. Quantization and operator compatibility

### 2.1 int8 specification and CMSIS-NN

**Fact:** the TensorFlow Lite 8-bit specification represents values as
`real = (int8 - zero_point) * scale`; convolution weights support per-axis
(per-output-channel) quantization, and the specification explicitly lists both
Conv2D and DepthwiseConv2D as per-axis-supported operations. [TensorFlow Lite
8-bit quantization specification](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_spec.md)

**Fact:** official TensorFlow guidance says full integer post-training
quantization requires a representative dataset. To force an integer-only
model, set `target_spec.supported_ops` to `TFLITE_BUILTINS_INT8` and choose int8
inference input/output types. [TensorFlow post-training quantization guide](https://www.tensorflow.org/lite/performance/post_training_quantization)

**Fact:** CMSIS-NN states that it follows the TFLite/TFLM int8 and int16
specifications and aims to be bit-exact with TFLite reference kernels. Its
operator table lists int8 C implementations for Conv2D, DepthwiseConv2D,
Fully Connected, Add, Mul, Average Pooling, and Softmax. [CMSIS-NN README](https://github.com/ARM-software/CMSIS-NN/blob/main/README.md#supported-framework)

**Fact:** Arm's DS-CNN C++ deployment code calls a regular convolution for the
stem, a depthwise-separable kernel for each block, a 1x1 convolution for each
pointwise stage, and a final fully connected layer. It also states that batch
normalization parameters are folded into convolution weights/biases. [Arm
DS-CNN deployment source](https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/NN/DS_CNN/ds_cnn.cpp#L515-L557)

**Inference:** the export contract for this repository should be a fixed graph
containing only the stem Conv2D, DepthwiseConv2D, 1x1 Conv2D, fused ReLU,
AveragePool2D, Reshape/Squeeze, and FullyConnected operations. BatchNorm and
Dropout must disappear or be folded before the TFLM gate. The exact graph must
be inspected after conversion; do not assume PyTorch module names imply TFLite
operator names.

### 2.2 TFLM evidence and resolver gate

**Fact:** the official TFLM `micro_speech` test registers a deliberately small
resolver containing Reshape, FullyConnected, DepthwiseConv2D, and Softmax, then
allocates a static tensor arena and invokes the int8 model. The same test reads
int8 input and dequantizes int8 output using the tensor scale/zero point. [TFLM
`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc#L893-L1045)

**Fact:** the TFLM example README documents a Cortex-M0 QEMU test path using
`OPTIMIZED_KERNEL_DIR=cmsis_nn`. [TFLM micro_speech README](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/README.md#run-the-c-tests-on-a-development-machine)

**Inference:** a DS-CNN TFLM resolver should start with:

```cpp
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddAveragePool2D();
resolver.AddFullyConnected();
resolver.AddReshape();       // if emitted by the converter
resolver.AddSqueeze();       // if emitted instead of Reshape
resolver.AddSoftmax();       // only if probabilities are part of the model
```

The resolver is a test gate, not a claim that every converted graph uses exactly
these ops. The converter must be asked to emit full int8, then the generated
FlatBuffer's op list must determine the final resolver. Arena usage and output
parity should be checked on the actual target.

**Unknown:** neither the TFLM micro_speech example nor the Arm DS-CNN C++
example proves that this repository's PyTorch model will convert without graph
rewrites. In particular, layout conversion (PyTorch NCHW versus TFLite/TFLM
deployment layout), fixed global pooling, fused BN, and per-axis depthwise
scales require an actual conversion test.

**Fact:** Arm notes that CMSIS-NN can run on earlier processors such as Cortex-M0,
but the SIMD performance benefits are for Cortex-M4/M7/M33/M35P-class devices.
[Arm CMSIS-NN deployment guidance](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Image%20recognition%20on%20Arm%20Cortex-M%20with%20CMSIS-NN.pdf)

**Inference:** on Cortex-M0+, measure scalar TFLM/CMSIS-NN latency and arena
peak rather than extrapolating Arm Cortex-M7 numbers from Hello Edge. This is
one reason to keep the first arm at one dendrite and small widths.

## 3. Pruning implications for DS-CNN

**Fact:** TensorFlow Model Optimization's structured-pruning guide says its
2-by-4 pattern is applied to the last dimension of the TFLite weight tensor; it
can improve inference time only on hardware supporting that pattern and may
reduce accuracy because of the restriction. [TFMOT structured-pruning guide](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_sparsity_2_by_4)

**Inference:** use structured channel pruning as the primary arm for a scalar
M0+/TFLM deployment. It directly reduces dense operator dimensions and avoids
relying on an unverified sparse kernel. Keep the project's N:M arm as a
secondary experiment only if the intended accelerator/MRAM path has a measured
zero-skipping implementation.

**Inference:** prune a DS-CNN block as a coupled unit:

1. rank/slice pointwise output channels;
2. apply the same surviving channel indices to the next block's depthwise
   filters and first BN;
3. slice the next pointwise input dimension and its BN;
4. slice the final classifier input;
5. fine-tune with KD before any dendrite phase.

This is exactly the dependency pattern implemented by the repository's
`prune_ds_cnn`; independently pruning only depthwise kernels would break the
channel contract or fail to remove the corresponding pointwise compute.

**Unknown:** I found no primary source demonstrating that PAI's dendritic
residual branches survive structured channel surgery automatically. Therefore,
pruning must finish and be checkpointed before `UPA.perforate_model`; never
prune a live PAI-wrapped model in the first experiment.

## 4. PerforatedAI evidence and limits

**Fact:** PAI's official README documents the training lifecycle as
`UPA.perforate_model` -> register optimizer -> report validation scores -> handle
restructuring and reinitialize the optimizer. [PAI README quick start](https://github.com/PerforatedAI/PerforatedAI#quick-start)

**Fact:** the same README lists `Conv2d`/`Linear`-style PyTorch integration and
an Edge Impulse keyword-spotting example, claiming better accuracy at every
parameter count across 800 sweeps. The README does not identify that example's
architecture, exact graph, or TFLM export. [PAI README examples](https://github.com/PerforatedAI/PerforatedAI#examples)

**Unknown:** no official PAI repository/API example, paper result, or test was
found that specifically combines PAI with a depthwise-separable KWS network,
`groups=in_channels` depthwise Conv2d, channel-pruned DS-CNN, or a TFLite/TFLM
export.

**Fact:** the public PAI source describes its open-source release separately
from the proprietary Perforated Backpropagation package and says the headline
compression results come from the full suite, not the open-source release.
[PAI README, alternative training mechanisms](https://github.com/PerforatedAI/PerforatedAI#alternative-training-mechanisms)

**Inference:** report the experiment as a PAI/DS-CNN architecture study, not as
a reproduction of the vendor's headline compression or PB results. The clean
fixed graph—not the live PAI wrapper, tracker state, or native checkpoint—is the
only candidate that should be sent to TFLite/TFLM conversion.

**Inference:** rank placement risk as follows:

| Placement | Why test | Initial recommendation |
| --- | --- | --- |
| `fc` | Lowest graph risk; output is 2-D and classifier is already a supported Linear/FC shape. | Always include as a control. |
| Last block's `pointwise` Conv2d | Adds feature capacity while preserving depthwise kernel shape and operator family. | Primary dendritic placement. |
| Stem Conv2d | Larger copied branch and early activation cost. | Secondary arm only. |
| `depthwise` Conv2d | Grouped copy and depthwise per-channel semantics are not explicitly documented by PAI. | Unknown/high-risk; test only after the clean path works. |
| Whole `DSConvBlock` | Composite module includes BN/ReLU and may cause recursive wrapping or export complexity. | Avoid as the first placement. |

## 5. Proposed pipelines

### Pipeline A: XXS/XS deployability-first, one late residual branch

**Inference design:**

```text
DS-CNN-L teacher
  -> KD train DS-CNN-XS (or XXS if the post-dendrite budget is very tight)
  -> structured channel-prune / width sweep
  -> KD fine-tune each clean narrow student
  -> PAI on {last pointwise Conv2d, fc}, max_dendrites=1
  -> clean/finalize fixed model
  -> fuse BN and run full-int8 PTQ or QAT
  -> verify TFLite op set, TFLM arena, and Cortex-M0+ latency
```

Suggested first candidates are the repository's XS (4,096 parameter) and XXS
(1,830 parameter) configs, with measured projected dendrite cost as the admission
rule. A one-dendrite cap is deliberate: PAI branches copy selected modules and
add combination parameters, while the edge graph must remain inside the same
budget as the no-dendrite control.

**Pass criteria:** validation accuracy above the existing acceptance floor;
clean exported parameter/MAC/weight-byte/activation-peak budgets; complete int8
conversion with no float fallback; TFLM `AllocateTensors()` and `Invoke()`;
and accuracy parity against the host TFLite interpreter within a predeclared
tolerance. Compare against the exact pruned no-dendrite checkpoint.

### Pipeline B: S-to-narrow accuracy-first, late-block plus classifier

**Inference design:**

```text
DS-CNN-L teacher
  -> train DS-CNN-S student with feature/response KD
  -> sweep structured keep ratios (for example 0.75, 0.50, 0.35)
  -> KD fine-tune each pruned width
  -> PAI on last pointwise Conv2d + fc, max_dendrites=1
  -> resume KD/supervised fine-tuning on the fixed dendrites
  -> clean/finalize, fuse BN, full-int8 QAT or calibrated PTQ
  -> TFLM resolver/arena/latency gate on Cortex-M0+
```

This arm asks whether dendrites recover accuracy lost by channel surgery when
the base has enough representational room. Keep the same deployment gate so a
larger, more accurate PAI model cannot win merely by exceeding the memory or
MAC budget. If one late pointwise branch is unhelpful, the next controlled
comparison is `fc`-only, not depthwise perforation; that isolates placement
from the grouped-convolution compatibility question.

### Optional Pipeline C: depthwise-placement feasibility probe

This is **not** the first production candidate. Take one already-trained XXS
checkpoint, perforate exactly one depthwise Conv2d (one layer, one dendrite),
and measure: PAI forward correctness, parameter/MAC growth, clean-finalization
success, ONNX/TFLite conversion, per-channel int8 scales, and TFLM parity. Stop
if any check fails. The probe turns the present PAI depthwise support question
into a bounded compatibility experiment without contaminating the main
accuracy/efficiency comparison.

## 6. Required measurements and unknowns to resolve

For every arm, record:

- clean base and clean PAI parameter counts, MACs, int8 weight bytes;
- peak activation/tensor-arena bytes after conversion;
- FlatBuffer operator list and operator versions;
- input/output dtype, scales, zero points, and per-channel weight scales;
- host PyTorch -> clean PyTorch -> TFLite -> TFLM output parity;
- Cortex-M0+ wall-clock latency for one inference and the full audio/frontend
  pipeline;
- validation/test accuracy separately (test only after the recipe is frozen).

Open questions that primary sources did not settle:

1. Does the installed PAI build correctly wrap and clean a grouped depthwise
   `Conv2d`? **Unknown;** use Pipeline C.
2. Does `prepare_final_model` produce a graph accepted by this repository's
   exporter after the chosen PAI placement? **Unknown;** test before claiming
   TFLM deployment.
3. Will a given PyTorch/TFLite conversion retain only the resolver ops listed
   above, or emit extra `Quantize`, `Dequantize`, `Pad`, `Mean`, or `Squeeze`
   nodes? **Unknown;** inspect the FlatBuffer and register only the exact set.
4. Does the claimed PAI Edge Impulse KWS advantage transfer to this DS-CNN,
   frontend, class count, and RP2040 budget? **Unknown;** the official example
   does not publish enough architecture/deployment detail to infer this.

## Sources consulted

1. Zhang et al., *Hello Edge: Keyword Spotting on Microcontrollers*, arXiv
   1711.07128: <https://arxiv.org/abs/1711.07128> and PDF tables/appendix:
   <https://arxiv.org/pdf/1711.07128>.
2. Arm, `ML-KWS-for-MCU` reference implementation and DS-CNN deployment:
   <https://github.com/ARM-software/ML-KWS-for-MCU>.
3. TensorFlow Lite 8-bit quantization specification:
   <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_spec.md>.
4. TensorFlow post-training quantization guidance:
   <https://www.tensorflow.org/lite/performance/post_training_quantization>.
5. TensorFlow Lite Micro `micro_speech` README and test:
   <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/README.md>
   and
   <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc>.
6. Arm CMSIS-NN operator/quantization support:
   <https://github.com/ARM-software/CMSIS-NN/blob/main/README.md>.
7. TensorFlow Model Optimization structural pruning:
   <https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_sparsity_2_by_4>.
8. PerforatedAI official README and API/source repository:
   <https://github.com/PerforatedAI/PerforatedAI>.
