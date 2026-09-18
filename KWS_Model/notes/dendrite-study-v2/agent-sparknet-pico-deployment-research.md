# SparkNet / RP2040 deployment research

**Scope.** Research-only deployment assessment for the SparkNet + dendrite study. No
code, model, or training run was changed or executed for this note. Published numbers
are kept separate from RP2040 feasibility inferences. The intended target is an
RP2040/Pico-class board: dual-core Cortex-M0+, up to 133 MHz, and 264 kB on-chip
SRAM.

## Decision in one page

* **Use TensorFlow Lite Micro (TFLM) as the first deployment route, with CMSIS-NN
  only as an optional kernel path.** Raspberry Pi maintains a Pico-specific TFLM
  port that includes voice-recognition examples and can use both RP2040 cores for
  some operations ([Raspberry Pi `pico-tflmicro`](https://github.com/raspberrypi/pico-tflmicro)).
  TFLM is designed for no OS, no dynamic allocation, and very small memory; its
  documented limitations (manual arena management and a limited operator subset)
  are exactly the constraints that should be tested here
  ([TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)).
* **The realistic first candidate is float-trained SparkNet C16 or C32 converted to
  fully integer int8.** The paper reports C16 at 4,636 parameters, 454.5 kMACs,
  95.3%/95.7% SC1/SC2 accuracy; C32 at 11,500 parameters, 1.2M MACs,
  96.2%/97.0%. These are attractive compute/accuracy points, but they are paper
  measurements on a host GPU/software stack—not RP2040 latency or SRAM measurements.
* **C4 is a feasibility probe, not an accuracy target.** Published C4 has 1,416
  parameters, 105 kMACs, and only 82.3%/83.5% accuracy. C8 (2,292 parameters,
  190 kMACs, 91.6%/92.1%) is the more credible lower-cost point.
* **Do not equate SparkNet sparsity with sparse MCU execution.** SparkNet learns a
  sample-wise binary/sparse gate representation, but its convolution weights remain
  dense. Standard TFLM/CMSIS-NN kernels will still execute the dense convolution MACs
  unless a custom sparse operator is implemented; the paper’s MAC count therefore
  remains the relevant first deployment estimate.
* **The supplied Edge Impulse ExecuTorch post is useful context, not RP2040 evidence.**
  It reports `.pte` file sizes and an int8 size example, but explicitly says its
  Python harness does not run on microcontrollers, its `.pte` excludes the DSP/MFCC
  stage, and the Cortex-M route is not yet its mature path. Treat its numbers as
  secondary/vendor claims and prefer TFLM/CMSIS-NN for this board.

## Source hierarchy and caveat labels

* **[PRIMARY/PAPER]** peer-reviewed SparkNet or BC-ResNet paper, with the exact
  benchmark protocol and reported metrics.
* **[PRIMARY/OFFICIAL]** Raspberry Pi, TensorFlow, Arm, or PyTorch documentation and
  source repositories.
* **[SECONDARY/MARKETING]** Edge Impulse blog claims. They are useful for workflow
  context but are not independent RP2040 measurements.
* **[INFERENCE]** a conclusion made here from the published architecture and RP2040
  constraints. It must be confirmed with a Pico build and timer/memory logs.

## 1. What SparkNet actually is

### Architecture and inference graph

The source is Svirsky, Shaham, and Lindenbaum, “Sparse Binarization for Fast Keyword
Spotting,” Interspeech 2024 ([paper PDF](https://www.isca-archive.org/interspeech_2024/svirsky24_interspeech.pdf),
[released code](https://github.com/jsvir/sparknet)). **[PRIMARY/PAPER]** The evaluated
input is a one-second, 16-kHz Speech Commands clip represented by 32 MFCC bins. The
project’s matching frontend produces `(1, 32, 101)` features using a 25-ms window and
10-ms hop (`KWS_Model/configs/data/speech_commands_v2_mfcc32_paper.yaml`), so the
host-to-device frontend must reproduce those exact choices.

The model is a small 1-D time-channel-separable CNN:

1. Four depthwise-over-time + pointwise-across-channel blocks, each followed by batch
   normalization and ReLU; temporal kernel widths are 11, 15, 19, and 29.
2. Residual connections in the last three blocks.
3. A 1x1 output/gate convolution, batch normalization, and `tanh`.
4. Average pooling over time and a linear classifier for 12 classes (10 keywords,
   unknown, silence).

The paper’s Table 1 describes the 1x1 output as producing `F` channels, whereas this
repository’s port exposes a configurable `gate_channels` (the study commonly uses
32). **[PRIMARY/PAPER + LOCAL SOURCE]** Treat the checked-out model/export graph as
authoritative for deployment dimensions; do not infer the gate tensor shape solely
from the paper table.

During training, the gate output is perturbed with Gaussian noise (`sigma=0.5`),
shifted/clipped into `[0,1]`, and regularized by an approximate L0/sparsity loss. The
classifier consumes the pooled gate representation rather than the original MFCC
values. At inference, stochastic training noise is absent. **[PRIMARY/PAPER]** This
is a learned input/activation representation, not binary neural-network weights.

### Published benchmark numbers

The paper evaluates official Speech Commands v1/v2 test splits, top-1 accuracy, and
host-computed MACs. It trains for 200 epochs on a GTX 1080 Ti; no MCU, TFLM, CMSIS-NN,
quantized, arena, or embedded latency result is reported. **[PRIMARY/PAPER]**

| Model | Parameters | MACs | SC1 accuracy | SC2 accuracy | Interpretation |
|---|---:|---:|---:|---:|---|
| SparkNet C32 | 11,500 | 1.2M | 96.2 ± 0.19% | 97.0 ± 0.18% (Table 3; 97.1 ± 0.30% in Table 4) | High-accuracy candidate; still tiny weights |
| SparkNet C16 | 4,636 | 454.5K | 95.3 ± 0.33% | 95.7 ± 0.17/0.30% | Best paper cost/accuracy starting point |
| SparkNet C8 | 2,292 | 190K | 91.6 ± 0.76% | 92.1 ± 0.33% | Low-cost probe |
| SparkNet C4 | 1,416 | 105K | 82.3 ± 1.91% | 83.5 ± 0.60% | Feasibility floor, likely unacceptable quality |
| BC-ResNet-0.625 | 4,585 | 1.9M | 95.2 ± 0.37% | 95.4 ± 0.31% | Paper’s matched C16 comparison |
| BC-ResNet-1 | 9,232 | 3.6M | 96.6 ± 0.21% | 96.9 ± 0.30% | Larger compute baseline |

The SparkNet paper’s direct claim is therefore **C16 has approximately the same
accuracy as BC-ResNet-0.625 at about one quarter of its MACs**, and C32 is roughly
three times faster than BC-ResNet-1 and five times faster than DS-ResNet10 in its
table. Those are cross-paper host MAC comparisons under the paper’s protocol, not
Pico speedups. **[PRIMARY/PAPER]**

For a second primary comparison, the BC-ResNet paper describes its broadcasted
residual architecture (frequency-depthwise convolution, frequency averaging,
temporal depthwise separable convolution, and a broadcast residual) and reports
BC-ResNet-1 at 9.2K parameters/3.1M multiplies and 96.6%/96.9% on SC1/SC2, while
BC-ResNet-8 reaches 321K/89.1M and 98.0%/98.7%
([BC-ResNet Interspeech 2021 PDF](https://www.isca-archive.org/interspeech_2021/kim21l_interspeech.pdf)).
**[PRIMARY/PAPER]** This supports the useful comparison axis: SparkNet is unusually
MAC-efficient, while BC-ResNet has a stronger accuracy/compute curve at larger sizes.

## 2. RP2040 constraints and what they imply

Raspberry Pi’s official specification lists a dual-core Arm Cortex-M0+ at up to
133 MHz and 264 kB on-chip SRAM ([RP2040 specifications](https://www.raspberrypi.com/products/rp2040/specifications/);
[RP2040 datasheet](https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf)).
**[PRIMARY/OFFICIAL]** The SRAM is shared by application code’s mutable state,
TFLM arena, audio buffers, MFCC scratch, stack, and any other firmware data. Flash
can hold model constants, but weight size alone does not establish deployability.

Arm’s CMSIS-NN documentation says every supported operator has a pure-C path for
Cortex-M0/M3; DSP/SIMD optimizations target cores such as Cortex-M4/M33, and Helium
targets M55/M85 ([CMSIS-NN README](https://github.com/ARM-software/CMSIS-NN)).
**[PRIMARY/OFFICIAL]** Therefore RP2040 should be expected to use the scalar C path,
not the faster DSP kernels often quoted for Cortex-M4 benchmarks. CMSIS-NN does list
int8 convolution, depthwise convolution, fully-connected, pooling, and activation
coverage, but the exact TFLM operator layout and compiler flags must be checked in
the built binary.

**[INFERENCE]** Approximate int8 weight storage is very small: 4,636 parameters are
about 4.6 kB of int8 values (18.5 kB at fp32), and 11,500 are about 11.5 kB int8
(46 kB fp32), before biases, scales, alignment, and runtime metadata. This leaves
substantial theoretical SRAM headroom, but it says nothing about peak activation
buffers, MFCC computation, or firmware code size. The 32x101 input alone is 3,232
bytes at int8 and 12,928 bytes at fp32; an audio ring buffer for 16,000 samples is
another 32 kB at int16.

### Recommended runtime path

TFLM’s official microcontroller guide states that the runtime is designed for only
some kilobytes, does not require an OS, standard C/C++ libraries, or dynamic memory,
and uses a manually managed tensor arena; it also warns that only a limited
operator subset is supported ([TFLM microcontrollers guide](https://www.tensorflow.org/lite/microcontrollers)).
The upstream porting guide shows how to generate a minimal static library and how to
select optimized CMSIS-NN kernels through `OPTIMIZED_KERNEL_DIR=cmsis_nn`
([TFLM new-platform support](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md)).
The Raspberry Pi port is a practical starting point, but it is read-only/generated
and maintained on a best-effort basis; pin the tested revision in any reproducible
experiment.

For quantization, TensorFlow’s official guidance distinguishes weight-only/dynamic
range quantization from full integer quantization. Full integer conversion requires
a representative dataset for activation ranges and enables integer-only hardware
execution ([post-training quantization guide](https://www.tensorflow.org/model_optimization/guide/quantization/post_training);
[RepresentativeDataset API](https://www.tensorflow.org/api_docs/python/tf/lite/RepresentativeDataset)).
Use representative MFCC tensors from the same frontend, not arbitrary random input.
The representative-data API describes a few hundred samples as the usual calibration
scale. **[PRIMARY/OFFICIAL]**

SparkNet’s `tanh`, residual adds, depthwise 1-D convolutions represented as 2-D
`(1,K)` kernels, average pooling, and final linear layer should be checked against
the target TFLM/CMSIS-NN operator set. **[INFERENCE]** BatchNorm should be folded
into adjacent convolution/linear parameters before export. A first experiment should
be fully int8 weights *and* activations; if tanh or gate quantization loses too much
accuracy, test int16 activations/int8 weights as a fallback, then compare its arena
and scalar-C latency rather than assuming it is faster.

## 3. Is ExecuTorch + Edge Impulse suitable for RP2040?

The supplied Edge Impulse article is dated September 2026 and should be labeled
**[SECONDARY/MARKETING]** ([article](https://www.edgeimpulse.com/blog/from-pytorch-to-the-edge-getting-started-with-executorch-and-edge-impulse/)).
Its useful, directly stated facts are:

* A learning block emits both `model.onnx` (ingested by Edge Impulse, then converted
  to TFLite) and `model.pte` (native ExecuTorch artifact); the `.pte` is not used by
  the Edge Impulse SDK path.
* Its baseline examples are float32, and reported `.pte` sizes include a program
  overhead above raw weights. The article later gives a ~98 kB float32 model becoming
  32.6 kB with int8 in its deployment block.
* It explicitly says the `.pte` contains only the neural network, not the DSP/MFCC
  stage; exact feature parameters must be reproduced separately.
* It explicitly says the current Python harness needs PyTorch and cannot run on a
  microcontroller; a no-Python C++ executor is a prerequisite.
* It says the Cortex-M route is not the mature route and identifies Edge Impulse’s
  EON Compiler path as the mature Cortex-M option. The article reports end-to-end
  CPU execution on an Arduino UNO Q, not an RP2040.

These admissions make the article useful for understanding artifact boundaries, but
none is evidence that SparkNet `.pte` will fit, compile, or run on an RP2040. Its
projected “quarter the weights” arithmetic is not a measured RP2040 result. Do not
use the article’s XNNPACK numbers as a Pico benchmark.

Official ExecuTorch documentation is more cautious: the runtime overview says the
core runtime can be under 50 kB *without kernels/backends*, supports user-provided
memory, and may run bare-metal, but warns that individual kernels/backends can have
additional requirements ([ExecuTorch runtime overview](https://github.com/pytorch/executorch/blob/main/docs/source/runtime-overview.md)).
The official examples page marks the Arm microcontroller/Cortex-M backend as beta
and says examples are representative, not a compatibility list
([ExecuTorch examples](https://github.com/pytorch/executorch/blob/main/examples/README.md)).
That is a poor risk profile for a <30-minute RP2040 feasibility check compared with
the existing Pico TFLM path. ExecuTorch could be revisited after a working C++
microcontroller backend and operator map are demonstrated, but should not be the
primary study deliverable.

## 4. A test that can finish in under 30 minutes

This is a deployment smoke test, not training. It should use one already available
SparkNet C16 checkpoint and a fixed bundle of precomputed `(32,101)` MFCC tensors so
frontend implementation does not consume the whole time budget.

1. **0–5 min: host export inventory.** Export C16 (then C8 only if time permits) to
   TFLite. Record input/output shapes, dtypes, all operators, parameter count, raw
   FlatBuffer bytes, and whether BatchNorm has been folded. Reject any float or
   unsupported op if the target is intended to be integer-only.
2. **5–12 min: calibration and host parity.** Calibrate full-int8 with 100–300
   representative MFCC tensors. Evaluate the official validation/test split through
   the quantized host interpreter. Record accuracy, unknown/silence behavior, and
   max absolute/logit or top-1 disagreement against fp32. If PTQ loses too much
   accuracy, record that result; do not silently substitute it with fp32.
3. **12–18 min: minimal Pico build.** Build the Raspberry Pi TFLM example/application
   with reference kernels first. Add CMSIS-NN only if the generated tree and target
   compile cleanly. Link the model as a C array in flash. Use a fixed input tensor,
   so the first run tests model execution independent of microphone hardware.
4. **18–25 min: flash and time.** Flash the UF2, run at a known 133 MHz clock, and
   time at least 100 warm batch-1 invocations using a hardware timer. Record mean,
   p50, p90, p99, and minimum/maximum cycles or microseconds. Run once per core mode
   (single-core and the port’s dual-core option) if the port exposes it.
5. **25–30 min: memory and correctness log.** Use TFLM’s recording allocator (or an
   equivalent high-water mark) to record tensor-arena bytes. Log firmware flash
   bytes, static RAM, stack high-water mark if available, model FlatBuffer bytes,
   and output agreement on a small fixed vector set. A model “fits” only when arena,
   stack, audio/frontend buffers, and application reserve all fit inside 264 kB.

For an end-to-end audio test, reserve a separate follow-up: the paper/project’s exact
MFCC frontend needs a 16-kHz audio capture path, FFT/mel/DCT implementation, and
matching quantization. A model-only success with precomputed MFCCs does not establish
real microphone deployment.

## 5. Exact evidence required for a Pareto improvement

The repository already defines deployment-cost fields in
[`src/kws/utils/profile.py`](../../src/kws/utils/profile.py) and a Pareto comparator
in [`src/kws/optimize/pareto.py`](../../src/kws/optimize/pareto.py). Use the same
matched control, frontend, split, and seed policy for every candidate. The minimum
artifact row should contain:

| Axis | Required artifact metric | Measurement rule |
|---|---|---|
| Quality | `test_accuracy`, `FAR`, `FRR`, per-class F1/confusion matrix | Quantized target graph, official held-out test split; report mean ± sample SD over at least 5 seeds for a study claim |
| Model size | `deployed_params`, `weight_bytes`, FlatBuffer/C-array bytes | Count actual deployed graph, including dendrite branches, biases/scales, alignment, and metadata; do not report only trainable base params |
| Compute | `macs` and operator breakdown | Count actual batch-1 exported graph; SparkNet gate sparsity does not reduce this unless a sparse kernel is actually used |
| Mutable memory | `activation_peak_bytes` / tensor-arena high-water mark | Record on TFLM target; host hook estimate is only a preflight bound |
| Runtime | Pico cycles/us mean, p50, p90, p99 | Same clock, compiler flags, core mode, warmup, and input; report model-only and, later, full frontend latency |
| Quantization validity | fp32-vs-int8 accuracy delta and output parity | State PTQ/QAT, calibration set, input/output dtypes, scales/zero points, and any float fallback |
| Reproducibility | firmware commit, TFLM/CMSIS-NN revisions, toolchain, linker map | Store the exact build command and binary hash |

For the current repo’s host-side benchmark, preserve the existing fields `accuracy`,
`far`, `frr`, `latency_ms_mean/p50/p90/p99`, `onnx_bytes`, and parity; then add
target-side arena/flash/cycle fields. The host ONNX latency is a proxy only, not the
RP2040 result.

Call candidate **A** a matched-control Pareto improvement over **B** only if:

* `A.accuracy` is no worse than B by the predeclared tolerance (recommended first
  smoke-test tolerance: 0.2 percentage points; final claim should use seed-wise
  uncertainty), and FAR/FRR do not regress beyond a predeclared bound;
* A is no worse on every required cost axis (`deployed_params`, MACs, p50 latency,
  weight bytes, arena bytes) and is materially better on at least one; or A gains a
  predeclared accuracy margin (for example ≥0.5 pp) while staying within the cost
  budget; and
* the result survives the same held-out test protocol across at least five seeds,
  with the raw per-seed rows retained. One lucky validation seed or one host MAC
  estimate is not enough.

The most defensible initial frontier is therefore: fp32 C16 host baseline → int8 C16
host baseline → int8 C16 TFLM reference kernel → int8 C16 CMSIS-NN/pico build → C8
only if C16 fails a memory or latency budget. Report C4 as a quality/feasibility
boundary, not as evidence of a useful keyword spotter.

## Bottom line

SparkNet’s published C16 point is a strong candidate for an RP2040 experiment because
its 454.5 kMAC host cost and 4.6K parameters are far below the paper’s BC-ResNet
baseline. **[INFERENCE]** It is plausible that an int8 C16 graph fits in the 264 kB
SRAM budget, but feasibility is unproven until the Pico arena and cycle measurements
exist. The first 30-minute test should use TFLM, scalar/reference kernels first and
CMSIS-NN second, and log the exact artifacts above. The Edge Impulse ExecuTorch post
does not establish RP2040 suitability and should remain a secondary comparison, not
the deployment basis.
