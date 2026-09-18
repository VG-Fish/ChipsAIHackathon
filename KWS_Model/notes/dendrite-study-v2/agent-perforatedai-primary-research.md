# PerforatedAI for parameter-efficient keyword spotting

Primary-source research, audited 2026-09-18. Sources are limited to PerforatedAI's
official site/repository/API reference and the official PerforatedAI paper hosted at
`perforatedai.com`. Claims are labelled **Fact**, **Inference**, or **Unknown**.

## Bottom line

- **Fact:** Open-source PerforatedAI (PAI) is a PyTorch training-time architecture
  growth system. It wraps selected modules, trains a candidate dendrite, and then
  integrates accepted dendrite modules into the forward path. The public README's
  quick-start flow is `UPA.perforate_model` -> register optimizer -> report
  validation score -> reinitialize the optimizer after restructuring.
  [README](https://github.com/PerforatedAI/PerforatedAI#quick-start)
- **Fact:** A dendrite is an additional learned copy of the wrapped module plus
  learned combination weights. Dendrites therefore add parameters and compute while
  training and, unless a later model redesign/export removes them, at inference.
  [module implementation](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L976-L1040)
- **Fact:** PAI can improve the accuracy/parameter Pareto frontier by starting with a
  smaller base network and adding targeted dendrites; this is not the same as taking a
  trained model and pruning or quantizing it. The official README explicitly says
  its headline compression numbers come from the separate Perforated
  Backpropagation (PB) system, not this open-source release.
  [README](https://github.com/PerforatedAI/PerforatedAI#key-results),
  [README: alternative training mechanisms](https://github.com/PerforatedAI/PerforatedAI#alternative-training-mechanisms)
- **Inference:** For this repository, the practical KWS experiment should be a
  small Conv2d/Linear (or Conv1d/Linear) baseline plus a tightly bounded number of
  dendrites, followed by an explicit clean-model/export experiment. Do not assume
  that a PAI checkpoint is a TFLite/TFLM artifact.
- **Unknown:** I found no official PAI statement or tested example promising
  TorchScript, ONNX, TFLite, or TFLM export of a perforated model. The documented
  inference path is still a Python/PyTorch PAI loader; see deployment below.

## 1. What a dendrite computes

### Wrapped neuron forward

- **Fact:** `PAINeuronModule` retains the original module as `main_module`, creates a
  `PAIDendriteModule`, and keeps `dendrites_to_top` and candidate combination weights
  as `ParameterList`s. The wrapper requires exactly one `0` in
  `output_dimensions`; that index identifies the neuron/channel axis. `-1` means a
  variable dimension. [PAINeuronModule constructor](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L276-L396)
- **Fact:** On each forward, PAI computes the original output `out = main_module(...)`,
  computes every accepted dendrite output, and adds each dendrite output multiplied
  by a learned `to_top` vector. The vector is unsqueezed/expanded across all axes
  except the neuron axis, so the output retains the original tensor format.
  [forward implementation](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L695-L797)
- **Fact:** Internally, `PAIDendriteModule` stores accepted copies in a `ModuleList`.
  A later dendrite can also receive a learned weighted contribution from previously
  created dendrites before its nonlinear output is returned. The source comments call
  this dendrite-to-dendrite computation.
  [dendrite forward](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L976-L1040),
  [dendrite-to-dendrite path](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L1267-L1374)

One useful abstraction for a selected module is:

```text
y = main(x) + sum_i broadcast(D_i(x, D_<i) * W_i)
```

where `D_i` is a copied module with the PAI dendrite processing/nonlinearity and
`W_i` is the learned dendrite-to-neuron (top) weight for the output channels. This
equation is a compact restatement of the implementation, not a separate PAI API.

### Parameter impact

- **Fact:** The parent dendrite module is a deep copy of the original selected module.
  Candidate copies are initialized from that parent copy; initialization can use a
  random multiplier (default `0.01`) and optionally scale by the main module's mean
  absolute weight. [parent copy and config](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L976-L1050),
  [candidate initialization](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L936-L974)
- **Fact:** A `Conv1d`, `Conv2d`, or `Linear` dendrite copy has approximately the
  same trainable parameter count as its parent (including bias when present), plus a
  learned top-combination vector of length `out_channels` for each accepted
  dendrite. After the first dendrite, the implementation also allocates learned
  dendrite-to-dendrite weights shaped `(previous_dendrites, out_channels)`.
  [dendrite-to-top allocation](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L549-L681),
  [dendrite-to-dendrite allocation](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L1100-L1168)
- **Inference:** If a selected layer has `P` parent parameters, `C` output channels,
  and `n` accepted dendrites, a useful lower-bound estimate for persistent added
  parameters is `n*P + n*C`, before counting any inter-dendrite weights and
  framework buffers. The exact count should be measured from the resulting
  `state_dict`, because processors, biases, and the current PAI version affect it.
- **Fact:** The tracker records both `num_dendrites_added`,
  `num_dendrites_integrated`, and parameter counts at each network structure, which
  makes the actual before/after count observable.
  [tracker state](https://docs.perforatedai.com/perforatedai/tracker_perforatedai.html)
- **Fact:** `retain_all_dendrites=False` by default, so failed candidates are not
  intended to remain as accepted architecture. That does not mean PAI prunes an
  already-trained base network: it means candidate dendrites that do not meet the
  score criterion can be discarded during the growth search.
  [configuration defaults](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L919-L934)

## 2. Training phases, freezing, and restructuring

- **Fact:** The official PerforatedAI paper describes alternating a **neuron phase**
  (ordinary gradient descent until validation plateaus) and a **dendrite phase** in
  which neuron weights are frozen, dendrite nodes are trained using a modified
  Cascade-Correlation rule, and the best node per neuron is frozen and incorporated
  into the forward pass. [official paper, sections 2.1-2.2](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf)
- **Fact:** In the open-source implementation, switching to `p` mode sets up the
  dendrite training state; switching back to `n` mode appends the accepted candidate
  to the permanent layer list, appends combination weights, increments the dendrite
  count, and deletes the temporary candidate modules.
  [mode switch](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L549-L681),
  [candidate integration](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L1190-L1266)
- **Fact:** The normal public training loop is intentionally open-ended. After each
  validation score, `add_validation_score` returns `(model, restructured,
  training_complete)`. The model must be moved back to its device, and the optimizer
  must be recreated when `restructured` is true because new parameters appeared.
  [README quick start](https://github.com/PerforatedAI/PerforatedAI#quick-start)
- **Fact:** The default switch policy is history-based (`n_epochs_to_switch=10`),
  with a fixed-switch option (`fixed_switch_num=250`), and the default test flag is
  `testing_dendrite_capacity=True`. The same configuration exposes no-switch and
  switch-every-time modes. [configuration defaults](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L571-L681)
- **Inference:** A run constrained to under 30 minutes should use a small base model,
  restrict perforation to a few high-value layers, cap `max_dendrites`, and use a
  deliberately short fixed-switch schedule only for a smoke experiment. The normal
  history mode can spend multiple epochs waiting for plateaus and then restart the
  optimizer after each architecture change.

### Minimal current API shape

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

model = UPA.perforate_model(model, save_name="kws_pai", maximizing_score=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
GPA.pai_tracker.set_optimizer_instance(optimizer)

while True:
    train_one_epoch(model, optimizer)
    val_score = validate(model)
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_score, model
    )
    model = model.to(device)
    if training_complete:
        break
    if restructured:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
```

This follows the official README. Preserve the project's actual optimizer and
scheduler arguments in a real integration.

## 3. Validation/PB scoring and selection

- **Fact:** The public tracker compares a running validation score against the current
  best. For a maximizing score, a new score must beat both the relative threshold and
  the absolute threshold; minimizing scores use the corresponding lower-than tests.
  [score comparison](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/tracker_perforatedai.py#L2522-L2569)
- **Fact:** Defaults are `improvement_threshold=[0.001, 0.0001, 0.0]` and
  `improvement_threshold_raw=1e-5`; the active value is selected by the tracker’s
  schedule. The running history length defaults to one, so validation is not smoothed
  by a long window unless configured.
  [threshold defaults](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L611-L640)
- **Fact:** The official paper describes PB/candidate training as modified
  Cascade-Correlation: existing neuron weights are frozen, candidate dendrites are
  trained to correlate with the target error/gradient signal, and the best candidate
  is frozen and incorporated. [official paper, section 2.2](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf)
- **Fact:** The open-source code currently defaults to `global_candidates=1`; when
  candidates greater than one are encountered, the integration path explicitly warns
  that multi-candidate ranking is not set up and drops into a debugger. Its current
  integration also selects `plane_max_index = 0`. [candidate configuration and
  selection](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L900-L918),
  [integration code](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L1229-L1266)
- **Fact:** The repository states that the PB library/algorithm is not part of the
  open-source release and that its `perforated_backpropagation` flag is false in this
  release. [README alternative mechanisms](https://github.com/PerforatedAI/PerforatedAI#alternative-training-mechanisms)
- **Inference:** Do not advertise the open-source run as reproducing the proprietary
  PB candidate-scoring method or the README’s headline compression numbers. The OSS
  path is useful for testing the wrapped architecture and ordinary backprop training;
  PB-specific score behavior requires the separately licensed package.
- **Unknown:** The exact PB correlation normalization, candidate ranking across more
  than one candidate, and any proprietary PB optimizer details are not reproducible
  from the public repository.

## 4. Layers, shapes, and model compatibility

- **Fact:** Current default module names selected for perforation are
  `PAISequential`, `Conv1d`, `Conv2d`, `Conv3d`, and `Linear`. Module IDs can narrow
  this to particular layers. [defaults](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L724-L768)
- **Fact:** PAI's default `output_dimensions` is `[-1, 0, -1, -1]` (batch, channel,
  height, width). A sequence format is `[-1, -1, 0]` (batch, time, features); exactly
  one `0` is required. [shape configuration](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L881-L891)
- **Fact:** The wrapper automatically shortens the default shape for `Linear` to
  batch/neuron dimensions and for `Conv1d` to batch/time-or-channel/neuron dimensions.
  Conv2d uses the four-dimensional format; Conv3d is listed as supported by name but
  has no analogous shape-shortening branch in this code. [shape adaptation](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L369-L394)
- **Fact:** The default path assumes a module has one tensor input and one tensor
  output. Tuple outputs fail when PAI attempts to combine them; custom processors are
  provided for modules with nontrivial input/output signatures.
  [configuration comment](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L791-L824),
  [tuple-output failure](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L735-L797)
- **Inference for this KWS tree:** MFCC/spectrogram DS-CNN/SparkNet blocks using
  ordinary Conv2d + Linear are the lowest-risk target. A raw waveform Conv1d model is
  also plausible, but its `output_dimensions` and temporal axis must be explicitly
  checked. Do not perforate pooling, reshape, recurrent, or tuple-producing helper
  modules without a processor or tracking configuration.

## 5. Checkpoints, clean model, and inference

- **Fact:** `UPA.perforate_model` initializes tracker/scaffolding and saves a PAI
  configuration alongside a named run when full training mode is used.
  [perforate_model API](https://docs.perforatedai.com/perforatedai/utils_perforatedai.html)
- **Fact:** The PAI-specific loader `NPA.load_pai_model(net, filename)` first converts
  a freshly constructed base model into PAI wrappers, loads a SafeTensors state dict,
  recreates the saved dendrite cycles, and then loads the weights. This is the
  documented direct inference/deployment path and requires the same base architecture.
  [network loader](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/network_perforatedai.py#L899-L1060)
- **Fact:** `prepare_final_model` is documented as a cleanup path that deep-copies the
  network, removes PAI scaffolding, removes `tracker_string`, and makes parameters
  contiguous. In the current source it delegates to internal blockwise/cleanup
  helpers. [final-model API](https://docs.perforatedai.com/perforatedai/utils_perforatedai.html)
- **Fact:** `using_safe_tensors=True` and `pai_saves=False` are the current defaults;
  PAI saves are a separate optimized-save option. [file/save defaults](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L571-L609)
- **Inference:** For a KWS artifact, keep both (1) the PAI checkpoint needed to
  reproduce/load the trained dendritic architecture and (2) a cleaned model exported
  from `prepare_final_model` if the cleanup path succeeds. Record parameter count,
  operator graph, dtype, and accuracy for each artifact.
- **Unknown:** The repository does not specify a stable ABI for the cleaned model,
  guarantee that every custom processor can be removed, or provide a C/C++ runtime.
  A consumer should not assume that shipping a `.safetensors` file alone is enough;
  the PAI Python package and model class are part of the documented loader path.

## 6. Quantization and export compatibility

- **Fact:** PAI exposes a configurable `d_type` for newly created dendrite and
  connection weights, defaulting to `torch.float`; this is a dtype setting, not an
  advertised quantization/export API. [dtype setting](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py#L775-L790)
- **Fact:** Candidate initialization has a special branch for `torch.uint8`, but the
  normal random initialization and PAI combination paths are written for floating
  tensors. This branch alone is not evidence of end-to-end int8 inference support.
  [initialization](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L936-L974)
- **Unknown:** I found no official PAI API or example for post-training quantization,
  quantization-aware training, TorchScript, ONNX, TFLite, or TFLM export. The official
  docs describe PyTorch/SafeTensors loading and a Python cleanup helper, but no edge
  runtime contract. [API index](https://docs.perforatedai.com/perforatedai/)
- **Inference:** A safe export experiment would first call the cleanup helper, then
  verify the resulting ordinary PyTorch module with `torch.jit`/ONNX and only then
  attempt int8 conversion. Exporting the live PAI wrapper directly is high risk due to
  `ModuleList`-based dynamic dendrite loops, runtime restructuring, processor
  delegation, and training-time gradient hooks. This is an engineering inference from
  the source, not an official compatibility claim.
- **Inference for TFLM:** TFLM cannot consume a Python PAI checkpoint or execute PAI's
  dynamic growth/tracker. It would require a cleaned fixed graph and a separately
  validated conversion of every operator. No official PAI source documents such a
  conversion, so TFLM support should be treated as unproven.

## 7. Evidence relevant to small models and KWS

- **Fact:** The official README names an “Edge Impulse Block” keyword-spotting
  example and says that across 800 hyperparameter sweeps dendritic models were more
  accurate at every parameter count. [README examples](https://github.com/PerforatedAI/PerforatedAI#examples)
- **Fact:** The official PerforatedAI semantic-segmentation paper says prior PB work
  included microcontroller-class keyword spotting and that the KWS study dominated
  the accuracy/parameter Pareto frontier across 800 hyperparameter trials.
  [official paper, related work/discussion](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf)
- **Fact:** The same paper reports the pattern in a different edge task: its best
  perforated MobileNetV2 achieved 79.34% Person F1 at 5.88M parameters versus 79.53%
  for a 12.19M traditional ResNet-18, and the best perforated ShuffleNetV2 was about
  25 ms/frame. These are segmentation results, not KWS results.
  [official paper, results](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf)
- **Unknown:** The PerforatedAI repo does not contain the KWS paper, the 800-run raw
  sweep table, a reproducible KWS model/checkpoint, or exact KWS model-size/latency
  numbers. The paper's reference points to an Edge Impulse winners blog, not a full
  paper in this repository. Therefore the KWS claim is useful motivation but not a
  reproducible benchmark for this tree.
- **Inference:** The strongest local test is a controlled Pareto comparison: same
  MFCC frontend, split, seed set, and tiny base model; compare zero dendrites against
  one/two dendrites at matched or recorded parameter counts, then separately measure
  clean-model latency and int8 conversion.

## 8. Failure modes and a <30-minute experimental plan

### Failure modes directly visible in official code

- **Shape mismatch:** exactly one zero is required in `output_dimensions`; incorrect
  rank or channel index triggers an error/exit during backward setup. Use the PAI
  debugging output-dimension setting before a real run.
  [dimension checks](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L160-L242)
- **Tuple/non-single-tensor outputs:** default PAI combination cannot combine a tuple;
  use a processor or mark the module tracked.
  [forward failure](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L735-L797)
- **Unused/frozen modules:** switching to dendrite mode requires gradient-derived
  output-channel state; frozen, unused, or non-gradient modules can fail this setup.
  [mode-switch diagnostics](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py#L640-L681)
- **Optimizer stale after growth:** restructuring adds parameters; the official quick
  start explicitly recreates and re-registers the optimizer.
  [README loop](https://github.com/PerforatedAI/PerforatedAI#quick-start)
- **DataParallel wrapper:** the tracker expects the underlying model, not a
  `DataParallel` wrapper, when validation is registered.
  [tracker validation check](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/tracker_perforatedai.py#L2380-L2435)
- **Checkpoint architecture mismatch:** the PAI loader reconstructs wrappers and
  dendrite cycles based on the saved state dict; instantiate the same base model and
  preserve the relevant config.
  [loader reconstruction](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/network_perforatedai.py#L972-L1047)

### Recommended time-boxed smoke plan (inference)

1. Establish a non-PAI tiny KWS baseline and count parameters/MACs.
2. Perforate only the final one or two Conv2d blocks and classifier (`Conv2d`,
   `Linear`), set exact output dimensions, and cap total dendrites to 1-2.
3. Use a short fixed-switch smoke run only to verify that a candidate can be trained,
   integrated, saved, and reloaded. Keep the real validation protocol and seed fixed.
4. Measure wall time per epoch and stop if PAI spends the budget in repeated switch/LR
   attempts. The default history schedule (`n_epochs_to_switch=10`) and up-to-100
   dendrite cap are not appropriate assumptions for a 30-minute budget.
5. Export/clean only after training. Compare baseline, live PAI, and cleaned model on
   the same KWS test set; then attempt quantization/export as a separate experiment.

The first smoke run should be treated as an integration test, not evidence of
compression. Since dendrites add capacity, a credible compression claim must compare
against a smaller non-PAI baseline and report the final cleaned artifact's actual
parameter count, flash/RAM estimate, latency, and KWS false-accept/false-reject
metrics.

## Source inventory

- [PerforatedAI official repository README](https://github.com/PerforatedAI/PerforatedAI)
- [Configuration source](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/globals_perforatedai.py)
- [Dendrite/module source](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/modules_perforatedai.py)
- [Tracker source](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/tracker_perforatedai.py)
- [Network/checkpoint source](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/network_perforatedai.py)
- [Utility/save/load source](https://github.com/PerforatedAI/PerforatedAI/blob/main/perforatedai/utils_perforatedai.py)
- [Official generated API: globals](https://docs.perforatedai.com/perforatedai/globals_perforatedai.html)
- [Official generated API: utils](https://docs.perforatedai.com/perforatedai/utils_perforatedai.html)
- [Official generated API: network](https://docs.perforatedai.com/perforatedai/network_perforatedai.html)
- [Official generated API: tracker](https://docs.perforatedai.com/perforatedai/tracker_perforatedai.html)
- [Official PerforatedAI paper hosted at perforatedai.com](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf)

