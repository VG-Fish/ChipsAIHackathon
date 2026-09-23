# NeuroSim ReRAM Hardware-Evaluation Integration

## 0. Objective

Add a hardware-simulation subsystem to the existing KWS repository that accepts an already-trained, inference-ready PyTorch model—including models trained or compressed using PerforatedAI—and estimates its compute-in-memory hardware characteristics using **DNN+NeuroSim V2.1 configured for ReRAM**.

The subsystem must answer:

* How much modeled chip/array area does this network require?
* What is estimated inference latency?
* What is estimated inference energy?
* What is estimated leakage power?
* What is estimated throughput?
* What are the major latency contributors?
* What are the major energy contributors?
* Which neural-network layers dominate hardware cost?
* How efficiently does the model map onto NeuroSim's ReRAM arrays?
* Do PerforatedAI parameter/MAC reductions translate into modeled hardware savings?
* What assumptions went into each result?
* Is a result exact within the chosen backend model, approximate, or an upper bound?

This is an **inference hardware-estimation tool**.

It must:

* use trained models;
* operate on the clean inference graph;
* generate NeuroSim-compatible inputs;
* execute NeuroSim externally;
* parse results into structured reports;
* preserve complete experiment provenance.

It must **not**:

* retrain models;
* modify checkpoints;
* modify the user's NeuroSim checkout in place;
* pretend NeuroSim estimates are physical hardware measurements;
* silently omit unsupported neural-network operations;
* silently make dense approximations of sparse/grouped connectivity.

---

# 1. Scope of version 1

Version 1 supports exactly one hardware backend:

```text
DNN_NeuroSim_V2.1
```

and exactly one memory technology exposed through our interface:

```text
ReRAM
```

Backend identifier:

```text
neurosim_v21_reram
```

Do not build a generic multi-device abstraction yet.

Do not introduce unused device-selection machinery.

The first implementation should optimize for:

```text
correctness
reproducibility
clear provenance
minimal ambiguity
```

rather than broad simulator support.

---

# 2. Existing repository integration

The current project already separates:

```text
src/kws/models/
src/kws/optimize/
src/kws/export/
src/kws/utils/profile.py
src/kws/pipeline.py
tests/
```

The new functionality should live beside the export/profiling infrastructure rather than inside training.

Create:

```text
src/kws/hardware/
    __init__.py

    neurosim/
        __init__.py
        __main__.py

        cli.py
        config.py

        model_loader.py
        graph_capture.py
        ir.py

        quantization.py
        traces.py
        network_csv.py

        source.py
        build.py
        runner.py

        parser.py
        report.py

        backend.py

        patches/
            v21/
```

Add tests:

```text
tests/test_neurosim_config.py
tests/test_neurosim_ir.py
tests/test_neurosim_graph_capture.py
tests/test_neurosim_quantization.py
tests/test_neurosim_traces.py
tests/test_neurosim_network_csv.py
tests/test_neurosim_parser.py
tests/test_neurosim_report.py
tests/test_neurosim_grouped_conv.py
tests/test_neurosim_integration.py
```

Do **not** modify `kws.pipeline` until the standalone subsystem is working end-to-end.

The repository already has project-owned profiling/export modules, so this new subsystem should complement them rather than replace them.

---

# 3. Core data flow

The complete execution path must be:

```text
trained PyTorch checkpoint
        ↓
reconstruct clean inference model
        ↓
run deterministic real input
        ↓
capture executed weighted layers
        ↓
build simulator-independent hardware IR
        ↓
quantize copied weights/activations for simulation
        ↓
generate NeuroSim weight matrices
        ↓
generate NeuroSim input bit traces
        ↓
generate NeuroSim network description
        ↓
prepare isolated NeuroSim source copy
        ↓
apply deterministic compatibility patch
        ↓
generate ReRAM hardware configuration
        ↓
compile NeuroSim
        ↓
execute inference estimation
        ↓
capture raw outputs
        ↓
parse normalized metrics
        ↓
generate JSON / CSV / Markdown report
```

Every arrow in this flow should have an explicit implementation boundary.

---

# 4. Command-line interface

Entry point:

```bash
uv run python -m kws.hardware.neurosim
```

Implement four useful commands initially:

```text
inspect
export
run
validate
```

Do not add a comparison command yet.

---

# 5. `inspect`

Usage:

```bash
uv run python -m kws.hardware.neurosim inspect \
  --checkpoint PATH \
  --model-config PATH
```

`inspect` must not require NeuroSim to be installed.

It must:

1. reconstruct the model;
2. load the checkpoint;
3. put the model in evaluation mode;
4. run one deterministic sample;
5. identify every weighted operation executed;
6. identify grouped/depthwise convolutions;
7. identify unsupported weighted operators;
8. report input/output tensor shapes;
9. report parameter count;
10. report MAC count using existing project utilities where possible;
11. report whether the model is eligible for hardware export.

Example output structure:

```text
Model: SparkNetC12
Checkpoint: ...
Parameters: ...
MACs: ...

Weighted executions:
[0] stem.conv
    Conv2d
    input:  [1, 1, 32, 32]
    output: [1, 12, 16, 16]
    kernel: 3x3
    groups: 1

[1] blocks.0.depthwise
    Conv2d
    ...
    groups: 12
    depthwise: yes

...

Eligibility:
✓ checkpoint reconstructed
✓ forward pass succeeded
✓ all weighted layers recognized
✓ grouped convolutions require patched mapping
```

If something is unsupported:

```text
Eligibility:
FAILED

Reason:
Unsupported weighted operator:
foo.custom_projection
type: CustomFooLayer
```

Do not continue silently.

---

# 6. `export`

Usage:

```bash
uv run python -m kws.hardware.neurosim export \
  --checkpoint PATH \
  --model-config PATH \
  --data-config PATH \
  --hardware-config configs/hardware/neurosim/reram_v21.yaml \
  --output-dir outputs/neurosim/my_run
```

This command must perform everything necessary to generate simulator inputs but must **not** compile or execute NeuroSim.

It generates:

```text
model IR
network description
quantized weight matrices
activation bit traces
activity statistics
manifest
effective configuration
```

This command is essential for debugging.

A user must be able to inspect the exact simulator inputs before running NeuroSim.

---

# 7. `run`

Usage:

```bash
uv run python -m kws.hardware.neurosim run \
  --checkpoint PATH \
  --model-config PATH \
  --data-config PATH \
  --hardware-config configs/hardware/neurosim/reram_v21.yaml \
  --output-dir outputs/neurosim/my_run
```

Equivalent conceptual flow:

```text
validate
→ inspect
→ export
→ prepare NeuroSim
→ compile
→ execute
→ parse
→ report
```

---

# 8. `validate`

Usage:

```bash
uv run python -m kws.hardware.neurosim validate \
  --hardware-config configs/hardware/neurosim/reram_v21.yaml
```

It must validate:

```text
configuration schema
NeuroSim root
expected upstream files
supported source revision structure
compiler availability
hardware parameter ranges
```

It must not require loading a model.

---

# 9. Hardware configuration location

Create:

```text
configs/hardware/neurosim/
    reram_v21.yaml
```

Do not spread device assumptions across Python constants.

All relevant hardware assumptions must be visible in configuration.

---

# 10. Configuration schema

Recommended structure:

```yaml
schema_version: 1

backend: neurosim_v21_reram

source:
  root_env: NEUROSIM_V21_ROOT

simulation:
  inference_only: true
  mapping: novel
  pipeline: false

precision:
  weight_bits: 8
  activation_bits: 8
  cell_bits: 2
  adc_bits: 6

array:
  rows: 128
  cols: 128
  columns_per_adc: 8

technology:
  node_nm: 32
  temperature_k: 300
  clock_hz: 1000000000

memory:
  access_type: 1t1r

  resistance_on_ohm: 240000
  resistance_off_ohm: 24000000

  read_voltage_v: 0.5
  read_pulse_width_s: 1.0e-8

  write_voltage_v: 4.0
  write_pulse_width_s: 5.0e-8

  access_resistance_ohm: 15000

trace:
  split: test
  samples: 16
  seed: 0

graph:
  grouped_conv_mode: patched
  reject_unsupported_ops: true

runtime:
  timeout_seconds: 300
```

These values are **reference simulation defaults**, not universal physical properties.

The configuration file must contain comments stating this.

---

# 11. Configuration implementation

Create immutable configuration dataclasses.

Example conceptual hierarchy:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    weight_bits: int
    activation_bits: int
    cell_bits: int
    adc_bits: int


@dataclass(frozen=True)
class ArrayConfig:
    rows: int
    cols: int
    columns_per_adc: int


@dataclass(frozen=True)
class TechnologyConfig:
    node_nm: int
    temperature_k: float
    clock_hz: int


@dataclass(frozen=True)
class MemoryConfig:
    access_type: str
    resistance_on_ohm: float
    resistance_off_ohm: float
    read_voltage_v: float
    read_pulse_width_s: float
    write_voltage_v: float
    write_pulse_width_s: float
    access_resistance_ohm: float
```

Avoid passing raw nested dictionaries around the implementation.

---

# 12. Configuration validation

Fail before simulation if:

```text
backend != neurosim_v21_reram
```

or:

```text
simulation.inference_only != true
```

The tool is not implementing training estimation.

---

# 13. Precision validation

Require:

```text
1 <= weight_bits <= 16
1 <= activation_bits <= 16
1 <= cell_bits <= weight_bits
1 <= adc_bits <= 16
```

Do not silently modify requested precision.

---

# 14. Array validation

Initial accepted values:

```text
rows ∈ {32, 64, 128, 256}
cols ∈ {32, 64, 128, 256}
```

Require:

```text
columns_per_adc >= 1
columns_per_adc <= cols
cols % columns_per_adc == 0
```

Fail if invalid.

Do not auto-correct.

---

# 15. Memory validation

Require:

```text
resistance_on_ohm > 0
resistance_off_ohm > resistance_on_ohm

read_voltage_v > 0
read_pulse_width_s > 0

write_voltage_v > 0
write_pulse_width_s > 0

access_resistance_ohm >= 0
```

---

# 16. External NeuroSim installation

NeuroSim must remain external to this repository.

User supplies a local clone.

Expected environment variable:

```bash
export NEUROSIM_V21_ROOT=/absolute/path/DNN_NeuroSim_V2.1
```

Do not automatically clone anything.

Do not download anything during a normal run.

---

# 17. Source validation

`source.py` must verify:

```text
root exists
expected repository files exist
NeuroSIM source directory exists
Makefile/build files exist
expected configuration code exists
```

If Git metadata is available, capture:

```text
remote URL
branch
commit SHA
dirty status
```

If Git metadata is unavailable, do not fail solely for that reason.

Instead record:

```text
git_commit: null
```

with a warning.

---

# 18. Source provenance

Every simulation must record:

```text
NeuroSim absolute source path
Git commit if available
dirty true/false if determinable
hashes of patched source inputs where practical
```

This is necessary because simulator changes can materially change estimates.

---

# 19. Never modify the user's original NeuroSim source

This is mandatory.

Never:

```text
edit Param.cpp in place
apply patches directly to source checkout
run destructive cleanup in original source tree
```

Instead create an isolated prepared source copy.

---

# 20. Build cache

Use a cache directory such as:

```text
.cache/neurosim/
    v21/
        <source-hash>/
            <hardware-config-hash>/
                source/
                build/
                executable
                build_manifest.json
```

Cache key must include:

```text
NeuroSim source identity
our patch version
effective hardware config
backend code version
```

Compiler identity may also be included if easy.

---

# 21. Model loading

Do not create a second independent interpretation of project checkpoints.

Reuse the repository's existing model reconstruction and checkpoint-loading machinery.

The project already contains model registry, checkpoint utilities, and multiple deployment/export paths.

Output should be normalized into something like:

```python
@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module

    checkpoint_path: Path
    checkpoint_sha256: str

    model_config_path: Path
    model_config_sha256: str

    run_id: str | None

    metadata: Mapping[str, Any]
```

---

# 22. Model evaluation state

After loading:

```python
model.cpu()
model.eval()
```

All graph/export operations must run under:

```python
torch.inference_mode()
```

Do not require CUDA or MPS.

Hardware export is not computationally intensive enough to justify device-specific behavior.

CPU gives more reproducible behavior.

---

# 23. Which PerforatedAI artifact to use

Evaluate the **clean deployed inference graph**.

Do not evaluate a PerforatedAI training wrapper simply because it is the newest checkpoint.

For example, if the run contains multiple artifacts such as:

```text
best_model.pt
best_model_pai.pt
latest.pt
latest_pai.pt
```

the implementation must use existing project metadata/export logic to determine which artifact represents the actual deployment graph.

If this cannot be determined unambiguously:

```text
FAIL
```

with a message such as:

```text
Could not determine the clean inference artifact for this
PerforatedAI checkpoint family.

Provide an explicit clean deployment checkpoint.
```

Do not guess.

---

# 24. Pre-export integrity check

Before generating hardware inputs:

1. hash checkpoint;
2. reconstruct model;
3. load weights strictly;
4. record model parameter tensors;
5. run one deterministic real sample;
6. verify output is finite;
7. save output checksum;
8. verify parameters are unchanged after capture.

At the end:

```text
checkpoint SHA before == checkpoint SHA after
```

must hold trivially because the checkpoint file should never be modified.

Also verify the in-memory model parameters have not changed.

---

# 25. Real input requirement

Use a real deterministic sample from the project's dataset pipeline.

Do not use:

```python
torch.randn(...)
```

for final export.

Synthetic tensors are allowed only in unit tests.

---

# 26. Hardware intermediate representation

Do not directly convert PyTorch modules to NeuroSim CSVs.

Create a simulator-independent IR first.

Create:

```python
@dataclass(frozen=True)
class HardwareLayer:
    execution_index: int
    execution_name: str

    module_name: str
    module_type: str

    op_type: str

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    in_channels: int | None
    out_channels: int | None

    kernel_h: int | None
    kernel_w: int | None

    stride_h: int | None
    stride_w: int | None

    padding_h: int | None
    padding_w: int | None

    dilation_h: int | None
    dilation_w: int | None

    groups: int

    weight_shape: tuple[int, ...]
    has_bias: bool

    parameter_count: int
    macs: int

    weight_key: str
```

---

# 27. Whole-model IR

Create:

```python
@dataclass(frozen=True)
class HardwareModelIR:
    schema_version: int

    model_name: str

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    layers: tuple[HardwareLayer, ...]

    non_cim_ops: tuple[NonCIMOperation, ...]

    total_parameters: int
    total_macs: int
```

Write:

```text
ir/model_ir.json
```

The IR is the authoritative representation consumed by later export steps.

---

# 28. Why use an IR

This avoids coupling:

```text
PyTorch model inspection
```

directly to:

```text
NeuroSim's historical CSV format
```

Benefits:

```text
easier testing
clear debugging
future backend changes
explicit grouped-convolution support
clean provenance
```

---

# 29. Graph-capture strategy

Do not rely solely on `torch.fx`.

PerforatedAI and custom project modules may contain control flow or wrappers that make symbolic tracing unreliable.

Primary mechanism:

```text
runtime forward hooks
```

Attach hooks to supported weighted leaf operations.

Initially:

```python
nn.Conv2d
nn.Linear
```

Capture what **actually executes** during inference.

---

# 30. Runtime-capture fields

For every weighted invocation capture:

```text
module path
module instance identity
execution number
operator type

input tensor shape
output tensor shape

weight tensor shape

stride
padding
dilation
groups

bias yes/no
```

Use runtime tensor shapes instead of inferring output shapes only from configuration.

---

# 31. Shared modules

A module can theoretically execute more than once in one forward pass.

Therefore:

```text
module identity != execution identity
```

Example:

```text
shared_projection#0
shared_projection#1
```

Both executions must appear separately in the execution IR.

They may reference the same `weight_key`.

---

# 32. Supported CIM operations

Initial mapping:

```text
Conv2d
Linear
```

These are the only operations that should be translated into ReRAM matrix operations in v1.

---

# 33. Non-CIM operations

Capture operations such as:

```text
BatchNorm
activation functions
pooling
flatten
reshape
dropout
elementwise addition
sigmoid
softmax
```

as:

```text
non_cim_ops
```

Do not silently claim that every neural-network operation is represented by NeuroSim.

The final report must state:

```text
NeuroSim estimates primarily describe the mapped matrix operations
and supported peripheral circuitry. Neural-network operations not
represented by the backend may not be fully reflected in the reported
latency and energy.
```

---

# 34. Unsupported weighted operators

If the model contains a weighted operation outside supported types, fail.

Examples:

```text
Conv1d
ConvTranspose2d
custom learned matrix operator
custom learned filter not represented by Conv2d/Linear
```

Do not ignore it.

---

# 35. Standard convolution weight conversion

PyTorch `Conv2d` weight shape:

```text
[out_channels,
 in_channels / groups,
 kernel_h,
 kernel_w]
```

For:

```text
groups = 1
```

flatten to:

```text
[out_channels,
 in_channels * kernel_h * kernel_w]
```

Then transpose:

```text
[in_channels * kernel_h * kernel_w,
 out_channels]
```

This becomes the logical crossbar weight matrix.

---

# 36. Linear weight conversion

PyTorch:

```text
[out_features, in_features]
```

Export:

```text
[in_features, out_features]
```

Implementation:

```python
weight_matrix = (
    weight
    .detach()
    .cpu()
    .numpy()
    .T
)
```

---

# 37. Depthwise and grouped convolution

This is one of the highest-risk portions of the integration.

Do not represent:

```text
groups > 1
```

as a normal dense convolution.

That would inflate connectivity, MACs, array dimensions, and potentially hardware cost.

Configuration must support:

```yaml
graph:
  grouped_conv_mode: patched
```

Also permit:

```text
reject
dense_upper_bound
```

for debugging.

---

# 38. `grouped_conv_mode = reject`

If:

```python
layer.groups != 1
```

raise:

```text
Grouped convolution encountered.

Set:
graph.grouped_conv_mode: patched

to use the validated grouped-convolution mapping.
```

---

# 39. `grouped_conv_mode = dense_upper_bound`

This exists only for early debugging.

If selected, treat the grouped convolution as dense.

Every resulting report must be classified:

```text
UPPER_BOUND
```

and contain:

```text
Grouped convolutions were mapped as dense convolutions.
Hardware area/energy/latency may be materially overestimated.
```

Do not treat these results as final.

---

# 40. `grouped_conv_mode = patched`

This is the target mode for real KWS experiments.

For:

```text
groups = G
input channels = Cin
output channels = Cout
```

define:

```text
effective_input_channels = Cin / G
```

Logical matrix rows:

```text
effective_input_channels
× kernel_h
× kernel_w
```

Logical matrix columns:

```text
Cout
```

before any bit-slicing/cell expansion performed by the simulator.

---

# 41. Depthwise convolution

For depthwise convolution:

```text
groups = Cin
```

therefore:

```text
effective_input_channels = 1
```

Logical matrix rows:

```text
kernel_h × kernel_w
```

not:

```text
Cin × kernel_h × kernel_w
```

This must have an explicit regression test.

---

# 42. Grouped-convolution MAC count

For grouped Conv2d:

```text
MACs =
output_h
× output_w
× output_channels
× (input_channels / groups)
× kernel_h
× kernel_w
```

Use actual runtime `output_h` and `output_w`.

---

# 43. Extended internal network format

Do not force all required information into NeuroSim's legacy network CSV immediately.

Generate an internal extended representation containing:

```text
input_h
input_w
input_channels

kernel_h
kernel_w

output_channels

stride_h
stride_w

groups

output_h
output_w

pool_flag
```

Then adapt the copied NeuroSim source to consume it.

---

# 44. Backwards compatibility in patched NeuroSim

If the parser encounters a legacy line without the new fields:

```text
groups = 1
```

must be assumed.

Existing upstream-compatible networks should continue to work.

---

# 45. Output geometry

Never recalculate output size using a simplified convolution formula when runtime output is already known.

Use captured PyTorch output:

```text
output_h
output_w
```

This avoids errors from:

```text
padding
stride
border behavior
```

---

# 46. Padding

Capture:

```text
padding_h
padding_w
```

Store it in IR.

Even if NeuroSim does not model padding as a hardware operation, it matters for verifying the relationship between:

```text
input geometry
output geometry
number of convolution positions
```

---

# 47. Stride

Support:

```text
stride_h
stride_w
```

separately.

Do not assume square stride.

If the NeuroSim patch cannot correctly represent asymmetric stride:

```text
FAIL
```

rather than using one dimension for both.

---

# 48. Dilation

Initial implementation supports only:

```text
dilation_h = 1
dilation_w = 1
```

Otherwise fail:

```text
Dilated convolution is not supported by the current
NeuroSim export path.
```

Do not approximate.

---

# 49. Pooling

Pooling should not automatically be treated as part of a weighted layer.

If the selected NeuroSim network format supports an explicit pooling flag, set it only when graph capture reliably establishes that pooling directly follows that weighted operation.

If association is uncertain:

```text
pool_flag = false
```

and represent pooling under:

```text
non_cim_ops
```

Being conservative is preferable to inventing execution structure.

---

# 50. Quantization philosophy

Hardware simulation precision must be explicit.

Do not assume:

```text
checkpoint dtype == hardware precision
```

A float32 checkpoint can be simulated using:

```text
8-bit weights
8-bit activations
```

without modifying the checkpoint.

---

# 51. Existing quantization metadata

If the exact deployment artifact already contains authoritative quantization metadata such as:

```text
bit width
scale
zero point
```

prefer it.

Do not invent a different quantization representation when the deployed graph already specifies one.

---

# 52. Floating-point checkpoint quantization

For a floating-point layer and `b` weight bits:

```python
qmax = 2 ** (b - 1) - 1

max_abs = max(abs(weight))

if max_abs == 0:
    scale = 1.0
else:
    scale = max_abs / qmax

q = round(weight / scale)
q = clamp(q, -qmax, qmax)
```

Use:

```text
per-tensor symmetric quantization
```

for v1.

Do not implement per-channel quantization yet unless required by an authoritative existing deployment artifact.

---

# 53. Weight quantization metadata

Record for each layer:

```text
bit width
quantization scheme
scale

original minimum
original maximum

quantized minimum
quantized maximum

exported matrix SHA-256
```

---

# 54. Never mutate model weights

This is prohibited:

```python
module.weight.data = quantized_weight
```

Instead:

```python
export_weight = quantize(
    module.weight.detach()
)
```

The in-memory model remains untouched.

---

# 55. Activation calibration

Collect activation inputs from real dataset samples.

Configuration:

```yaml
trace:
  split: test
  samples: 16
  seed: 0
```

Requirements:

```text
augmentation disabled
shuffle false
deterministic sample ordering
known sample IDs
```

---

# 56. Activation capture

Register temporary hooks on each weighted operation.

For each selected sample:

```text
run inference
capture actual tensor entering each weighted layer
```

Do not capture only model input.

Every weighted operation requires its own input trace.

---

# 57. Activation quantization

Initial scheme:

```text
per-layer
per-tensor
symmetric
max-absolute scaling
```

Given layer activation tensor:

```python
max_abs = max(abs(activation))
```

derive scale using requested activation bit width.

Record the scale and original range.

---

# 58. Activation calibration consistency

For v1, determine a layer's activation scale across the configured calibration samples.

Do not choose a completely different scale for every sample unless upstream NeuroSim specifically requires that interpretation.

Preferred:

```text
one deterministic calibration scale per layer
```

derived from all selected trace samples.

This makes samples comparable.

---

# 59. Trace encoder

Create a standalone pure function:

```python
encode_fixed_point_bits(
    values: np.ndarray,
    bits: int,
    scale: float,
) -> np.ndarray
```

It must contain no filesystem behavior.

It must be independently testable.

---

# 60. Bit representation

Before implementing trace export, verify NeuroSim V2.1's expected bit ordering and signed-value representation.

Do not invent the bit convention.

Create golden examples against upstream behavior.

Required values:

```text
0
1
-1

small positive
small negative

maximum representable positive
minimum representable negative
```

---

# 61. Activity factor

Calculate activity from the exact generated bit trace.

Define:

```text
activity =
number of active/high bits
/
total bit positions
```

Do not use:

```text
fraction of nonzero floating-point activations
```

as a substitute.

---

# 62. Weight files

Write:

```text
weights/
    layer_000.csv
    layer_001.csv
    layer_002.csv
    ...
```

Weight matrices are shared across all trace samples.

Do not duplicate them per sample.

---

# 63. V2.1 old/current weight interface

If the V2.1 executable requires both:

```text
current weight
previous weight
```

for its historical training-oriented interface, then in inference-only mode use:

```text
previous_weight = current_weight
```

Document this explicitly.

There is no training update.

Do not create fictitious prior weights.

---

# 64. Activation trace files

Organize:

```text
traces/
    sample_000/
        layer_000_input.csv
        layer_001_input.csv
        ...
        activity.json

    sample_001/
        ...
```

Example activity file:

```json
{
  "layer_000": 0.38471,
  "layer_001": 0.29103
}
```

---

# 65. Dataset provenance

Record exact selected examples.

Do not record only:

```text
samples = 16
```

Record:

```text
dataset split
dataset item index
sample identifier if available
class label
```

for every trace.

This allows exact regeneration.

---

# 66. Export determinism

These must be sufficient to reproduce simulator inputs:

```text
checkpoint hash
model-config hash
data-config hash
hardware-config hash
sample IDs
trace seed
quantization algorithm/version
exporter version
```

---

# 67. NeuroSim inference-only configuration

The backend must explicitly configure V2.1 for inference estimation.

In particular, ensure the equivalent upstream setting is:

```text
trainingEstimation = false
```

Do not expose a CLI option that allows turning training estimation on in v1.

This integration is inference-only.

---

# 68. ReRAM configuration

Set the upstream memory type to its ReRAM option.

All relevant ReRAM parameters must be generated from YAML rather than manually editing upstream files.

---

# 69. Generated configuration mechanism

Avoid fragile search-and-replace over dozens of lines in `Param.cpp`.

Instead, in the **copied NeuroSim source only**, apply a deterministic patch once so that an auto-generated configuration include can override user-controlled values.

For example:

```cpp
#include "GeneratedUserConfig.inc"
```

Then generate:

```text
GeneratedUserConfig.inc
```

for each build.

The exact syntax must match upstream variable ownership.

---

# 70. Patch versioning

Give every local upstream patch a version.

Example:

```text
NEUROSIM_PATCH_SCHEMA = 1
```

Include this value in:

```text
build cache key
manifest
build manifest
```

Whenever the upstream patch changes materially, increment the version.

---

# 71. ADC precision

If NeuroSim expects:

```text
number of levels
```

rather than:

```text
ADC bits
```

convert:

```text
adc_levels = 2 ** adc_bits
```

Record:

```text
adc_bits
adc_levels
```

in manifest.

Never conflate the two.

---

# 72. ReRAM assumptions

The checked-in default hardware config must state clearly:

```text
These values describe one configured reference ReRAM model.
They are not universal ReRAM device characteristics.
```

Users must be able to change:

```text
Ron
Roff

read voltage
write voltage

read pulse width
write pulse width

access resistance

technology node

subarray dimensions

precision
```

without editing source code.

---

# 73. Build preparation

Process:

```text
validate original source
        ↓
determine source identity
        ↓
calculate build cache key
        ↓
copy upstream source if cache miss
        ↓
apply known patch
        ↓
generate hardware configuration
        ↓
compile
        ↓
record build manifest
```

---

# 74. Compilation

Use Python subprocess APIs.

Example:

```python
subprocess.run(
    command,
    cwd=build_directory,
    check=True,
    capture_output=True,
    text=True,
    timeout=...,
)
```

Do not default to:

```python
shell=True
```

Save compiler stdout/stderr.

---

# 75. Build failures

If compilation fails, retain:

```text
compiler command
stdout
stderr
return code
prepared source path
```

and present an actionable error.

Do not delete evidence automatically.

---

# 76. Simulation execution

Generate executable arguments programmatically.

Do not blindly execute an upstream shell script if the same command can be expressed explicitly.

Use:

```python
subprocess.run(...)
```

with:

```text
explicit cwd
timeout
captured stdout
captured stderr
```

---

# 77. One trace sample per simulator invocation

Prefer isolated sample runs:

```text
raw/
    sample_000/
    sample_001/
    ...
```

Each directory contains:

```text
stdout.txt
stderr.txt
return_code.txt
upstream result files
```

This simplifies:

```text
resume
debugging
aggregation
failure isolation
```

---

# 78. Resume behavior

If:

```text
sample_000
sample_001
sample_002
```

completed successfully and sample 3 failed, rerunning the same configuration should be able to reuse completed results.

Reuse is permitted only if hashes match:

```text
checkpoint
hardware config
trace
prepared simulator build
```

Do not reuse based only on directory names.

---

# 79. Normalized hardware result

Create:

```python
@dataclass(frozen=True)
class HardwareResult:
    backend: str

    chip_area_m2: float | None

    forward_latency_s: float

    forward_dynamic_energy_j: float

    leakage_power_w: float | None

    leakage_energy_per_inference_j: float | None

    total_energy_per_inference_j: float

    fps: float

    tops: float | None
    tops_per_w: float | None

    per_layer: tuple["LayerHardwareResult", ...]

    component_breakdown: Mapping[str, float]

    warnings: tuple[str, ...]
```

Use SI units internally.

---

# 80. Per-layer result

Suggested:

```python
@dataclass(frozen=True)
class LayerHardwareResult:
    execution_index: int
    execution_name: str

    area_m2: float | None

    latency_s: float | None

    dynamic_energy_j: float | None

    mapped_rows: int | None
    mapped_cols: int | None

    num_subarrays: int | None

    latency_fraction: float | None
    energy_fraction: float | None
```

---

# 81. Raw versus normalized metrics

Preserve raw simulator output.

Generate two conceptual layers:

```text
raw NeuroSim metrics
normalized project metrics
```

Do not destroy the upstream field names.

This makes parser errors auditable.

---

# 82. Inference energy definition

Headline:

```text
dynamic_energy_per_inference
```

comes only from forward inference.

If leakage power is available:

```text
leakage_energy_per_inference =
    leakage_power
    × forward_latency
```

Then:

```text
total_energy_per_inference =
    dynamic_energy_per_inference
    + leakage_energy_per_inference
```

Do not include training-related energy.

---

# 83. V2.1 training-oriented outputs

Because V2.1 contains training-estimation functionality, parser code must not automatically assume any upstream field named:

```text
Total Energy
```

means inference energy.

Explicitly identify forward-pass fields.

Add parser tests based on real V2.1 output.

---

# 84. Throughput

For one serial inference:

```text
FPS = 1 / forward_latency_s
```

Only apply this if the simulator's reported forward latency corresponds to one complete inference under the chosen configuration.

Document the interpretation.

---

# 85. Operation convention

Use:

```text
1 MAC = 2 operations
```

for project-normalized TOPS.

Therefore:

```text
operations =
2 × model_MACs
```

and:

```text
TOPS =
operations
/
forward_latency_s
/
1e12
```

Record this convention.

---

# 86. Average power

Calculate:

```text
average_power =
total_energy_per_inference
/
forward_latency_s
```

---

# 87. TOPS/W

Calculate:

```text
TOPS_per_W =
TOPS
/
average_power
```

Store any upstream NeuroSim efficiency metric separately.

Do not overwrite one with the other.

---

# 88. Multi-sample aggregation

For activity-dependent metrics across `N` samples calculate:

```text
mean
standard deviation
minimum
maximum
```

At minimum:

```text
latency
dynamic energy
total energy
FPS
TOPS/W
```

Headline uses:

```text
mean ± standard deviation
```

and always includes:

```text
n = N
```

---

# 89. Area aggregation

Area should normally be invariant across input traces.

If parsed area differs materially between samples:

```text
FAIL OR WARN LOUDLY
```

depending on upstream semantics.

Do not simply average inconsistent structural areas without investigation.

---

# 90. Per-layer report

For each mapped layer include:

```text
execution index
module name
op type

input shape
output shape

weight shape
groups

parameter count
MAC count

weight bits
activation bits

logical matrix rows
logical matrix cols

mapped array rows
mapped array cols
number of subarrays if available

latency
dynamic energy

latency percentage
energy percentage
```

---

# 91. Hotspot summaries

Generate:

```text
Top 5 layers by latency
Top 5 layers by energy
```

Also report:

```text
fraction of total latency from top 5
fraction of total energy from top 5
```

---

# 92. Run output structure

Use:

```text
outputs/neurosim/<run>/
    manifest.json

    configs/
        hardware.yaml
        model.yaml
        data.yaml

    ir/
        model_ir.json
        network.csv

    weights/
        layer_000.csv
        layer_001.csv
        ...

    traces/
        sample_000/
        sample_001/
        ...

    build/
        build_manifest.json

    raw/
        sample_000/
        sample_001/
        ...

    reports/
        hardware.json
        hardware.csv
        hardware.md
        layers.csv
```

Do not write generated files into repository root.

---

# 93. Manifest contents

At minimum:

```json
{
  "schema_version": 1,

  "checkpoint": {
    "path": "...",
    "sha256": "...",
    "run_id": "..."
  },

  "model_config_sha256": "...",
  "data_config_sha256": "...",
  "hardware_config_sha256": "...",

  "backend": "neurosim_v21_reram",

  "neurosim": {
    "root": "...",
    "git_commit": "...",
    "dirty": false
  },

  "trace": {
    "split": "test",
    "samples": 16,
    "seed": 0,
    "sample_ids": []
  },

  "precision": {
    "weight_bits": 8,
    "activation_bits": 8,
    "cell_bits": 2,
    "adc_bits": 6
  },

  "grouped_conv_mode": "patched",

  "patch_schema": 1,

  "warnings": [],
  "approximations": []
}
```

---

# 94. Report validity classification

Every report must contain exactly one validity label:

```text
EXACT_WITHIN_BACKEND_MODEL
APPROXIMATE
UPPER_BOUND
INVALID
```

---

# 95. `EXACT_WITHIN_BACKEND_MODEL`

Use when:

```text
every weighted op supported
grouped layers mapped using validated patch
no intentionally dense approximations
parser fields verified
```

This does **not** mean physical measurement.

---

# 96. `UPPER_BOUND`

Use when:

```text
grouped_conv_mode = dense_upper_bound
```

or another deliberate structural overestimate is used.

---

# 97. `INVALID`

Use internally when a completed result should not be interpreted.

Normally a failed simulation should not generate a normal hardware report.

---

# 98. Mandatory report disclaimer

Every Markdown report must say:

```text
This is a NeuroSim model estimate under the specified ReRAM,
circuit, mapping, precision, and activity assumptions.

It is not a physical-device measurement.
```

---

# 99. Do not conflate simulator output with target-device runtime

Your existing README already distinguishes software-side/host proxy measurements from actual deployment measurements.

The new subsystem must preserve the same discipline.

Use wording:

```text
estimated CIM accelerator inference latency
```

not:

```text
device runtime latency
```

unless measured on the actual target system.

---

# 100. Accuracy is outside v1 hardware estimation

Version 1 covers:

```text
area
latency
energy
leakage
throughput
mapping
per-layer hardware cost
```

Do not initially implement:

```text
conductance noise accuracy
read-noise accuracy
variation-induced accuracy loss
ADC-induced accuracy loss
```

Those belong in a later phase.

---

# 101. Do not use NeuroSim's historical PyTorch layers as the main model

Do not replace current project modules with old simulator-specific layers.

Do not transform:

```text
nn.Conv2d
```

into an old custom NeuroSim layer inside the training/deployment model.

Instead:

```text
current PyTorch model
      ↓
our exporter
      ↓
NeuroSim trace/network inputs
```

This isolates the simulator from model training.

---

# 102. Why this architecture matters for PerforatedAI

PerforatedAI can change:

```text
network structure
weighted-operation placement
parameter count
execution graph
```

Therefore simulator integration should evaluate the **resulting inference graph**, not assumptions about how the model was trained.

This is exactly why runtime capture plus a normalized IR is preferable.

---

# 103. Test: configuration

`tests/test_neurosim_config.py`

Test:

```text
valid ReRAM config accepted

wrong backend rejected

training_estimation=true rejected

cell_bits > weight_bits rejected

activation_bits < 1 rejected

invalid ADC width rejected

invalid array rows rejected

invalid array cols rejected

columns_per_adc > cols rejected

cols not divisible by columns_per_adc rejected

Roff <= Ron rejected

negative pulse width rejected
```

---

# 104. Test: graph capture

Build toy models:

```text
Conv2d

Linear

Conv2d → ReLU → Linear

depthwise Conv2d

grouped Conv2d

shared Linear used twice
```

Verify exact execution order and shapes.

---

# 105. Test: ordinary convolution matrix

Use:

```python
nn.Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=2,
)
```

Fill weights with sequential integers.

Manually create expected flattened/transposed matrix.

Assert exact equality.

Do not test only matrix dimensions.

---

# 106. Test: Linear matrix

Use a small known matrix.

Verify:

```text
PyTorch [out, in]
```

becomes:

```text
NeuroSim [in, out]
```

exactly.

---

# 107. Test: depthwise matrix

Use:

```python
nn.Conv2d(
    in_channels=4,
    out_channels=4,
    kernel_size=3,
    groups=4,
)
```

Assert:

```text
effective input channels = 1
logical rows = 9
```

not:

```text
36
```

---

# 108. Test: ordinary convolution parity

For:

```text
groups = 1
```

the patched mapping calculation must exactly match the legacy calculation.

This test is mandatory.

It protects against breaking normal convolutions while adding group support.

---

# 109. Test: grouped convolution

Example:

```python
nn.Conv2d(
    in_channels=8,
    out_channels=12,
    kernel_size=3,
    groups=4,
)
```

Expected:

```text
effective input channels = 2
logical matrix rows = 18
logical matrix cols = 12
```

before simulator bit/cell expansion.

---

# 110. Test: MAC counts

Validate against an independent formula for:

```text
standard Conv2d
depthwise Conv2d
grouped Conv2d
Linear
```

Where possible also compare against the project's existing profiler.

---

# 111. Test: zero-weight quantization

All-zero tensor:

```text
scale = 1
all quantized values = 0
```

No divide-by-zero.

---

# 112. Test: quantization determinism

Identical tensor/config must create byte-identical exported matrix content.

---

# 113. Test: signed quantization

Cover:

```text
positive values
negative values
symmetric range
clamping
rounding
```

---

# 114. Test: bit encoding

Golden-vector test exact bitstrings for:

```text
0
1
-1
max
min
```

Do not settle for shape-only assertions.

---

# 115. Test: activity calculation

Use a manually specified bit matrix.

Example:

```text
1 0 1 0
1 1 0 0
```

Expected:

```text
activity = 4 / 8 = 0.5
```

Assert exact or appropriately tight floating-point equality.

---

# 116. Test: parser

Create fixture files representing a real or minimized NeuroSim V2.1 output.

Verify parsing of:

```text
forward latency
forward energy
area
leakage
component breakdown
layer data
```

Parser unit tests must not run the simulator.

---

# 117. Optional integration test

If:

```text
NEUROSIM_V21_ROOT
```

is defined:

1. prepare simulator;
2. compile;
3. run tiny model;
4. parse output;
5. assert positive latency/energy.

Otherwise:

```python
pytest.skip(...)
```

Normal CI must not require external NeuroSim installation.

---

# 118. Upstream compatibility test

Before real KWS models, create a very small network compatible with the original simulator assumptions.

Generate its inputs using:

```text
upstream reference path
our exporter
```

Compare:

```text
network dimensions
weight matrix layout
input trace layout
activity factor
```

Any differences must be explained.

---

# 119. Existing-project regression tests

After each implementation phase:

```bash
uv run python -m pytest tests/
```

Existing tests must continue passing.

Do not weaken unrelated assertions.

---

# 120. First end-to-end model

Do not begin with the most complicated PerforatedAI artifact.

Use a small, straightforward existing model.

Possible candidate:

```text
SparkNet C8
```

or another model already cleanly reconstructed by the repository.

Acceptance:

```text
checkpoint loads
forward pass succeeds
IR generated
all weighted ops accounted for
traces generated
NeuroSim compiles
NeuroSim exits zero
report generated
checkpoint unchanged
```

---

# 121. First PerforatedAI acceptance test

Use one clean post-PerforatedAI deployment model.

Acceptance:

```text
all weighted operations appear in IR
no training wrapper remains
parameter count agrees with project profiler
MAC count agrees with project profiler or documented convention
NeuroSim run succeeds
checkpoint unchanged
```

---

# 122. Parameter-count cross-check

For every real-model run calculate parameters via:

```text
existing project profiler
hardware IR
```

They need not match total model parameters perfectly if the IR intentionally excludes non-weighted state, but any difference must be explainable.

Report:

```text
project total params
IR weighted params
difference
reason
```

---

# 123. MAC cross-check

Compare:

```text
existing project MAC count
IR MAC count
```

If relative difference exceeds a small threshold such as:

```text
0.1%
```

fail the export unless the difference has an explicitly recognized cause.

Do not silently continue.

---

# 124. Why MAC cross-check matters

A wrong grouped-convolution interpretation can still produce:

```text
valid CSV
valid C++
successful simulator output
plausible-looking numbers
```

Therefore simulator execution success is **not** sufficient evidence that mapping is correct.

Independent MAC verification is a required correctness gate.

---

# 125. Reporting existing software metrics

Hardware reports should also include existing model metrics where available:

```text
validation/test accuracy
parameter count
MAC count
weight bytes
```

This enables later analysis such as:

```text
accuracy vs area
accuracy vs energy
accuracy vs latency
parameters vs energy
MACs vs energy
```

Your existing pipeline already records model cost metrics and Pareto information, so hardware metrics can extend that analysis rather than replace it.

---

# 126. PerforatedAI experiment table

Eventually generate data suitable for:

| Model     | PAI | Accuracy | Params | MACs | Area | Latency | Energy |
| --------- | --: | -------: | -----: | ---: | ---: | ------: | -----: |
| control   |  No |      ... |    ... |  ... |  ... |     ... |    ... |
| dendritic | Yes |      ... |    ... |  ... |  ... |     ... |    ... |

Do not automatically declare one architecture better.

The purpose is to measure whether software-level compression produces modeled hardware-level savings.

---

# 127. Pareto analysis

Later, reuse or extend the existing Pareto infrastructure.

Useful frontiers:

```text
accuracy vs energy
accuracy vs latency
accuracy vs area

accuracy vs parameter count
accuracy vs MAC count
```

A model that reduces parameters may not reduce hardware area proportionally because:

```text
array utilization
subarray granularity
ADC/peripheral overhead
mapping fragmentation
```

can dominate.

That is an important experimental result.

---

# 128. Fail-closed conditions

Fail rather than guess if:

```text
checkpoint cannot be reconstructed

clean inference artifact is ambiguous

forward pass fails

output contains NaN/Inf

weighted op unsupported

grouped convolution unsupported under selected mode

dilation unsupported

unexpected NeuroSim source structure

NeuroSim source unavailable

patch fails

C++ build fails

simulator exits nonzero

expected parser metric unavailable

weight-file count differs from mapped layer count

trace-file count differs from mapped layer count

MAC cross-check materially disagrees

model parameters change during export
```

---

# 129. Warnings instead of errors

Warnings are appropriate for:

```text
dirty NeuroSim checkout

unmodeled non-CIM operators

small trace sample count

reference ReRAM device parameters

dense upper-bound mapping when explicitly requested

missing Git metadata
```

Warnings must appear in both:

```text
manifest.json
hardware.md
```

---

# 130. Logging

Use concise structured progress logging:

```text
Loading checkpoint...
Capturing inference graph...
Found 18 weighted executions.
Collecting 16 calibration samples...
Quantizing weights...
Writing NeuroSim network description...
Preparing NeuroSim build...
Compiling NeuroSim...
Running trace 1/16...
...
Parsing results...
Writing report...
```

Do not log full matrices or tensors.

---

# 131. Determinism requirements

Given identical:

```text
checkpoint
model config
data config
hardware config
sample IDs
source revision
```

the tool should generate identical:

```text
IR
weight CSVs
activation CSVs
activity factors
network description
generated simulator configuration
```

Hash generated inputs.

---

# 132. Build manifest

Create:

```text
build/build_manifest.json
```

containing:

```text
source identity
source commit
dirty status
patch schema
hardware-config hash
generated-config hash
compiler command
compiler version if easy
executable SHA-256
```

---

# 133. Hardware JSON report

`reports/hardware.json` should contain machine-readable normalized results.

Suggested structure:

```json
{
  "validity": "EXACT_WITHIN_BACKEND_MODEL",

  "model": {
    "name": "...",
    "checkpoint_sha256": "...",
    "parameters": 12345,
    "macs": 456789,
    "accuracy": 0.94
  },

  "hardware": {
    "backend": "neurosim_v21_reram",
    "technology_node_nm": 32,
    "array_rows": 128,
    "array_cols": 128,
    "weight_bits": 8,
    "activation_bits": 8,
    "cell_bits": 2
  },

  "metrics": {
    "area_m2": 0.0,
    "latency_s_mean": 0.0,
    "latency_s_std": 0.0,
    "dynamic_energy_j_mean": 0.0,
    "total_energy_j_mean": 0.0,
    "fps_mean": 0.0,
    "tops_mean": 0.0,
    "tops_per_w_mean": 0.0
  },

  "warnings": []
}
```

---

# 134. Hardware CSV

Create one summary row per simulation configuration.

Useful columns:

```text
run_id
checkpoint
checkpoint_sha256

model_name
accuracy
parameters
macs

backend

technology_node_nm

array_rows
array_cols

weight_bits
activation_bits
cell_bits
adc_bits

trace_count

area_mm2

latency_ms_mean
latency_ms_std

dynamic_energy_uj_mean
dynamic_energy_uj_std

total_energy_uj_mean
total_energy_uj_std

fps_mean

tops_mean
tops_per_w_mean

validity
```

---

# 135. Layers CSV

Create:

```text
reports/layers.csv
```

Columns:

```text
execution_index
module_name
op_type

input_shape
output_shape

weight_shape
groups

parameters
macs

logical_rows
logical_cols

mapped_rows
mapped_cols

subarrays

latency_s
latency_fraction

dynamic_energy_j
energy_fraction
```

---

# 136. Markdown report structure

`hardware.md` should contain:

```text
# NeuroSim ReRAM Hardware Estimate

## Validity

## Model

## Simulation configuration

## Headline metrics

## Trace statistics

## Per-layer hotspots

## Layer table

## Unmodeled operations

## ReRAM assumptions

## Provenance

## Warnings

## Interpretation
```

---

# 137. Report interpretation section

Include wording similar to:

```text
These numbers estimate the behavior of the mapped neural-network
operations under the configured NeuroSim ReRAM model.

They are not physical measurements.

Results depend on the chosen memory parameters, array geometry,
precision, circuit assumptions, mapping strategy, and activation
traces.
```

---

# 138. Pipeline integration must come later

Only after standalone commands are stable should:

```text
src/kws/pipeline.py
```

be modified.

Then add an optional stage:

```text
hardware
```

Possible command:

```bash
uv run python -m kws.pipeline \
  --stages cluster,quantize,benchmark,hardware
```

---

# 139. Pipeline-stage semantics

The hardware stage must:

```text
consume an existing deployment artifact
```

It must not:

```text
train
fine-tune
re-run PerforatedAI
change model architecture
```

---

# 140. Relationship to existing profiler

Do not replace:

```text
kws.utils.profile
```

Existing profiler answers:

```text
parameters
MACs
software memory estimates
other architecture-level costs
```

NeuroSim answers:

```text
modeled ReRAM CIM area
modeled latency
modeled energy
modeled circuit/peripheral costs
```

Both are useful.

---

# 141. Relationship to existing host benchmark

Do not replace:

```text
kws.export.benchmark
```

That benchmark and NeuroSim answer different questions.

Keep:

```text
actual software-runtime measurements
```

and:

```text
modeled CIM hardware estimates
```

as separate evidence.

---

# 142. Implementation order

Implement in this exact sequence.

---

## Commit 1 — package skeleton and configuration

Add:

```text
hardware/__init__.py
neurosim/__init__.py
neurosim/__main__.py
neurosim/cli.py
neurosim/config.py
```

Implement:

```text
inspect --help
export --help
run --help
validate --help
```

Implement config parsing and validation.

Tests:

```text
test_neurosim_config.py
```

No model loading.

No simulator.

---

## Commit 2 — model loading

Add:

```text
model_loader.py
```

Reuse existing repository reconstruction/checkpoint utilities.

Test:

```text
DS-CNN checkpoint
SparkNet checkpoint
```

Output normalized `LoadedModel`.

---

## Commit 3 — runtime graph capture

Add:

```text
graph_capture.py
ir.py
```

Support:

```text
Conv2d
Linear
```

Generate:

```text
model_ir.json
```

Use synthetic toy-model tests first.

---

## Commit 4 — model integrity/cost checks

Add:

```text
parameter counting
MAC counting
existing-profiler cross-check
```

Do not proceed to simulator export until these checks pass.

---

## Commit 5 — standard convolution export

Add:

```text
network_csv.py
```

Initially support only:

```text
groups = 1
```

Do not implement grouped convolution yet.

Test exact matrix geometry.

---

## Commit 6 — quantization

Add:

```text
quantization.py
```

Implement deterministic per-tensor symmetric quantization.

Add golden tests.

---

## Commit 7 — activation traces

Add:

```text
traces.py
```

Implement:

```text
real sample collection
layer activation capture
activation calibration
bit encoding
activity calculation
```

No simulator yet.

At this point:

```text
export
```

should work fully.

---

## Commit 8 — upstream source validator

Add:

```text
source.py
```

Validate:

```text
NEUROSIM_V21_ROOT
expected files
source identity
Git metadata
```

Implement:

```text
validate
```

fully.

---

## Commit 9 — isolated build system

Add:

```text
build.py
backend.py
patches/v21/
```

Implement:

```text
source copy
patch application
generated config
build cache
compilation
```

Do not run a real KWS model yet.

---

## Commit 10 — toy end-to-end ReRAM simulation

Add:

```text
runner.py
```

Run a tiny:

```text
Conv2d + Linear
```

network.

Confirm:

```text
NeuroSim compiles
NeuroSim executes
raw outputs retained
```

No parser normalization beyond what is necessary.

---

## Commit 11 — parser

Add:

```text
parser.py
```

Normalize:

```text
forward latency
dynamic energy
area
leakage
```

Create fixture-based parser tests.

---

## Commit 12 — report generation

Add:

```text
report.py
```

Generate:

```text
hardware.json
hardware.csv
hardware.md
layers.csv
```

---

## Commit 13 — grouped/depthwise patch

Now implement:

```text
groups
effective input channels
extended network format
actual output dimensions
```

Patch local NeuroSim copy.

This is deliberately postponed until the ordinary path works.

---

## Commit 14 — grouped-convolution parity testing

Verify:

```text
groups=1
```

patched output exactly matches pre-patch behavior.

Verify depthwise dimensions mathematically.

Do not continue until this passes.

---

## Commit 15 — first real KWS model

Run one straightforward model.

Verify:

```text
IR
MACs
parameters
weights
traces
simulation
report
```

---

## Commit 16 — clean PerforatedAI model

Run one post-PerforatedAI deployment graph.

Resolve all differences between:

```text
project profiler
hardware IR
```

before accepting results.

---

## Commit 17 — multi-sample aggregation

Run:

```text
16 deterministic test samples
```

Add:

```text
mean
std
min
max
```

for activity-dependent hardware metrics.

---

## Commit 18 — pipeline integration

Only now modify:

```text
kws.pipeline
```

Add optional:

```text
hardware
```

stage.

---

# 143. First milestone

Stop after Commit 12 and review.

At that point the system should support:

```text
ordinary Conv2d
Linear

real checkpoint
real activations

V2.1 ReRAM
inference-only

area
latency
energy
leakage

normalized report
```

for networks without grouped convolutions.

Do not let a weaker coding model immediately continue into grouped convolution without review.

---

# 144. Second milestone

Commits 13–16.

Goal:

```text
correct depthwise/grouped mapping
+
real KWS architecture
+
real clean PerforatedAI artifact
```

This is the scientifically important milestone.

---

# 145. Third milestone

Add:

```text
multi-sample activity statistics
pipeline integration
model-family evaluation
```

At this point you can run a complete PAI hardware study.

---

# 146. Suggested model-family study

For each selected architecture:

```text
baseline/control checkpoint
clean PerforatedAI checkpoint
```

run the same ReRAM configuration.

Hold fixed:

```text
technology node
array size
precision
ADC precision
trace selection procedure
trace count
```

Then compare:

```text
accuracy
parameters
MACs
area
latency
energy
TOPS/W
```

---

# 147. Do not assume parameter savings equal area savings

One important research question is precisely whether:

```text
parameter reduction
```

maps to:

```text
physical array reduction
```

It may not.

Potential causes:

```text
array granularity
fragmentation
underutilized subarrays
peripheral overhead
ADC overhead
small layers
irregular dendritic additions
```

The simulator is useful partly because it exposes this distinction.

---

# 148. PerforatedAI-specific hardware question

For each PAI model, ask:

```text
Did the new dendritic structure reduce total modeled hardware cost?

Or did it reduce model parameters while creating inefficiently
mapped small matrices?
```

This is potentially much more informative than parameter count alone.

---

# 149. Array-utilization metric

If the simulator provides enough information, calculate:

```text
logical used cells
/
allocated physical cells
```

per layer.

Call it:

```text
array_utilization
```

Do not invent this number if allocation information is unavailable.

---

# 150. Potential PAI fragmentation metric

Optional after core implementation works:

```text
total allocated crossbar capacity
total used logical weight cells
unused allocated capacity
```

This could identify architectures that are parameter-efficient but array-inefficient.

Do not include in first implementation if upstream mapping details are unclear.

---

# 151. Scientific evidence hierarchy

Keep these distinct:

```text
PyTorch evaluation
    → actual model accuracy

project profiling
    → software/architecture cost

NeuroSim
    → modeled ReRAM hardware behavior

physical target
    → actual hardware measurement
```

Do not collapse these into one metric.

---

# 152. Reproducibility requirement

A hardware result is not complete unless another person can determine:

```text
exact checkpoint
exact model config
exact trace samples
exact NeuroSim source revision
exact NeuroSim patch version
exact ReRAM assumptions
exact precision
exact array geometry
```

from the output directory alone.

---

# 153. Licensing architecture

Keep NeuroSim as an externally supplied dependency.

Do not vendor the full simulator into this repository.

The adapter may contain:

```text
configuration generation
small compatibility patches
input/output translators
```

without requiring the upstream simulator source to live permanently inside this project.

Review upstream licensing separately before any future redistribution/commercial use.

---

# 154. Things the implementation model must NOT do

Do not:

* retrain the model;
* fine-tune the model;
* modify checkpoint files;
* modify weights in-place;
* replace current project Conv2d layers with historical simulator layers;
* downgrade the project's PyTorch installation;
* import an old simulator PyTorch environment into the project;
* vendor the complete NeuroSim repository;
* edit the user's NeuroSim checkout;
* hard-code one checkpoint;
* hard-code one model architecture;
* assume all convolutions are 3×3;
* assume all convolutions have stride 1;
* assume all convolutions have groups 1;
* assume output geometry from input geometry when runtime output exists;
* treat depthwise convolution as dense without warning;
* omit unsupported weighted operations;
* continue after a meaningful MAC mismatch;
* use random tensors for final activity estimates;
* call nonzero activation fraction the bit activity factor;
* include training energy in inference energy;
* silently change user hardware settings;
* hide simulator warnings;
* describe NeuroSim numbers as physical measurements.

---

# 155. Code-quality requirements

Use:

```text
small pure functions
immutable dataclasses
explicit units
typed interfaces
minimal global state
```

Avoid:

```text
giant functions
implicit dictionaries
magic column numbers
mutable module-level config
string parsing scattered across files
```

---

# 156. Units

Use SI internally:

```text
seconds
joules
watts
meters²
ohms
volts
hertz
```

Convert only for display.

Examples:

```text
s → ms
J → µJ
m² → mm²
```

Do not store an unlabeled float such as:

```python
latency = 2.4
```

without a defined unit.

Prefer names:

```python
latency_s
energy_j
area_m2
```

---

# 157. Parser robustness

Do not scatter regexes throughout the project.

Put all upstream output interpretation inside:

```text
parser.py
```

If NeuroSim's textual output changes, this should be the primary module requiring modification.

---

# 158. Backend isolation

Put all assumptions specific to V2.1 inside:

```text
backend.py
source.py
build.py
parser.py
patches/v21/
```

Generic modules such as:

```text
graph_capture.py
ir.py
quantization.py
traces.py
```

must not depend unnecessarily on V2.1 internals.

This keeps future changes possible without premature abstraction.

---

# 159. Final definition of done

The initial ReRAM integration is complete only when all of the following hold:

1. A normal project PyTorch checkpoint can be reconstructed.

2. A clean PerforatedAI-derived checkpoint can be reconstructed.

3. Neither checkpoint is modified.

4. No training occurs.

5. Real deterministic dataset inputs are used.

6. Every weighted operation is accounted for.

7. Unsupported weighted operations fail closed.

8. Standard convolutions export correctly.

9. Linear layers export correctly.

10. Depthwise convolutions export correctly.

11. General grouped convolutions export correctly.

12. Grouped mapping passes independent mathematical tests.

13. MAC totals agree with the project's independent profiler.

14. Weight quantization is deterministic.

15. Activation quantization is deterministic.

16. Input bit encoding is validated against known examples.

17. Activity factors are calculated from exported bits.

18. NeuroSim is configured for inference-only estimation.

19. NeuroSim source is never edited in place.

20. Simulator source revision is recorded.

21. ReRAM assumptions are recorded.

22. Simulator builds reproducibly from an isolated copy.

23. Build failures preserve diagnostic output.

24. Simulation failures preserve diagnostic output.

25. Raw simulator outputs are retained.

26. Parsed hardware results use explicit units.

27. Area is reported.

28. Forward latency is reported.

29. Forward dynamic energy is reported.

30. Leakage is reported when available.

31. Energy per inference is calculated consistently.

32. FPS is calculated consistently.

33. TOPS is calculated using a documented operation convention.

34. TOPS/W is calculated consistently.

35. Multiple activation traces can be aggregated.

36. Per-layer costs are reported where supported.

37. Report validity is prominently labeled.

38. Unmodeled neural-network operations are disclosed.

39. Existing project tests continue to pass.

40. New NeuroSim tests pass.

41. A real SparkNet/DS-CNN run succeeds.

42. A clean PerforatedAI run succeeds.

43. Results can be reproduced from the run directory and external simulator checkout.

44. No result is presented as a physical hardware measurement.

---

# 160. Final implementation principle

Optimize for **traceable correctness**.

A simulator run completing successfully does not prove that the neural network was mapped correctly.

The implementation must independently validate:

```text
graph structure
parameters
MACs
group connectivity
matrix geometry
quantization
trace representation
simulator configuration
parsed metric semantics
```

before treating a hardware estimate as valid.

A clearly labeled limitation is acceptable.

A plausible-looking number produced from an incorrect mapping is not.
