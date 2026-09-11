"""Deployment cost measurement: compute, memory, and latency.

Step 3e of the framework records latency, memory, and compute alongside
accuracy, because a search that only records accuracy cannot tell whether a
candidate extends the Pareto frontier.

Everything here works by forward hooks rather than by walking a config, so a
PerforatedAI-wrapped model, a codebook-parametrized model, and a plain DS-CNN
are all measured the same way -- dendrite copies contribute their real
convolutions to the MAC count exactly as they will on the device.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn


@dataclass(frozen=True)
class DeploymentCost:
    """What a candidate costs on the target, at batch size 1."""

    params: int
    macs: int
    weight_bytes: int
    activation_peak_bytes: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    device: str
    bits_per_weight: int
    weight_memory_method: str = "projected_logical_precision"
    activation_memory_method: str = "forward_hook_liveness_estimate"

    def as_dict(self) -> dict:
        return asdict(self)


def _leaf_modules(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if not list(module.children())]


def count_macs(model: nn.Module, input_shape: tuple[int, int]) -> int:
    """Multiply-accumulate operations for one batch-1 inference.

    Counted from the shapes actually seen in a forward pass, so grouped and
    depthwise convolutions, and any dendrite copies present, are all correct
    without the caller describing the topology.
    """
    total = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        output_elements = output.numel()
        kernel_macs = (
            module.in_channels // module.groups
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        total += output_elements * kernel_macs

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    def residual_hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        # Each learned skip edge performs one per-element multiply/add. The
        # branch convolutions/linears are counted by their own hooks.
        if hasattr(module, "pai_skip_connection_count"):
            edges = int(module.pai_skip_connection_count)
        else:
            edges = sum(weight.shape[0] for weight in module.skip_weights)
        total += output.numel() * edges

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nnq.Conv2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, (nn.Linear, nnq.Linear)):
            handles.append(module.register_forward_hook(linear_hook))
        elif hasattr(module, "pai_skip_connection_count") or (
            hasattr(module, "layer_array") and hasattr(module, "skip_weights")
        ):
            handles.append(module.register_forward_hook(residual_hook))

    _run_probe(model, input_shape)
    for handle in handles:
        handle.remove()
    return total


# Modules that reinterpret or pass through their input without allocating a new
# buffer. Counting them would inflate the arena estimate by a tensor that the
# deployed graph never materializes.
NON_ALLOCATING_MODULES = (nn.Flatten, nn.Dropout, nn.Dropout2d, nn.Identity)


def measure_peak_activation_bytes(
    model: nn.Module, input_shape: tuple[int, int], *, bytes_per_activation: int = 1,
) -> int:
    """Peak live activation footprint, approximated as the largest adjacent pair.

    A sequential MCU runtime holds one layer's input and its output at once and
    frees everything earlier, so the largest input+output pair over the graph
    is the arena size the deployment has to provision. Views and in-place ops
    are charged for their output only, because they write into a buffer that is
    already live rather than asking for a second one.
    """
    pairs: list[int] = []
    residual_peaks: list[int] = []
    handles = []

    def hook(module, inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            return
        if getattr(module, "inplace", False):
            pairs.append(output.numel())
            return
        input_elements = sum(
            tensor.numel() for tensor in inputs if isinstance(tensor, torch.Tensor)
        )
        pairs.append(input_elements + output.numel())

    for module in _leaf_modules(model):
        if isinstance(module, NON_ALLOCATING_MODULES):
            continue
        handles.append(module.register_forward_hook(hook))

    # A residual/dendritic wrapper retains earlier branch outputs while it
    # computes later branches. The largest leaf pair undercounts that live set,
    # so observe direct branch outputs and charge their conservative sum plus
    # the wrapper input. This remains an estimate because a backend allocator
    # may reuse storage internally, but it is safe for arena sizing.
    branch_handles = []
    for parent in model.modules():
        if not (hasattr(parent, "layer_array") and hasattr(parent, "skip_weights")):
            continue
        branch_sizes: list[int] = []

        def branch_hook(_module, _inputs, output, sizes=branch_sizes):
            if isinstance(output, torch.Tensor):
                sizes.append(output.numel())

        for branch in parent.layer_array:
            branch_handles.append(branch.register_forward_hook(branch_hook))

        branch_handles.append(
            parent.register_forward_pre_hook(
                lambda _module, _inputs, sizes=branch_sizes: sizes.clear()
            )
        )

        def residual_hook(module, inputs, _output, sizes=branch_sizes):
            input_elements = sum(
                tensor.numel() for tensor in inputs if isinstance(tensor, torch.Tensor)
            )
            residual_peaks.append(input_elements + sum(sizes))

        handles.append(parent.register_forward_hook(residual_hook))

    try:
        _run_probe(model, input_shape)
    finally:
        for handle in [*branch_handles, *handles]:
            handle.remove()
    return max([*pairs, *residual_peaks], default=0) * bytes_per_activation


def weight_memory_bytes(model: nn.Module, *, bits_per_weight: int = 32) -> int:
    """Flash/MRAM footprint of the parameters at the deployed precision."""
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    quantized_types = (nnq.Conv2d, nnq.Linear)
    if any(isinstance(module, quantized_types) for module in model.modules()):
        # Packed int8 weights are not exposed through .parameters(). Count the
        # tensors held by the quantized operators, plus their FP32 biases and
        # ordinary parameters retained by custom residual arithmetic.
        total = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        for module in model.modules():
            if isinstance(module, quantized_types):
                total += (module.weight().numel() * bits_per_weight + 7) // 8
                bias = module.bias()
                if bias is not None:
                    total += bias.numel() * 4
        return total
    numel = sum(parameter.numel() for parameter in model.parameters())
    return (numel * bits_per_weight + 7) // 8


def deployed_parameter_count(model: nn.Module) -> int:
    """Count logical weights/biases for both float and packed int8 modules."""
    quantized_types = (nnq.Conv2d, nnq.Linear)
    if any(isinstance(module, quantized_types) for module in model.modules()):
        count = 0
        for module in model.modules():
            if isinstance(module, quantized_types):
                count += module.weight().numel()
                bias = module.bias()
                if bias is not None:
                    count += bias.numel()
        # Other non-quantized parameters (for example BatchNorm in a graph
        # that was not fused) are still logical deployment parameters.
        count += sum(parameter.numel() for parameter in model.parameters())
        return count
    return sum(parameter.numel() for parameter in model.parameters())


def _run_probe(model: nn.Module, input_shape: tuple[int, int]) -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = _model_device(model)
    try:
        with torch.no_grad():
            return model(torch.zeros(1, 1, *input_shape, device=device))
    finally:
        model.train(was_training)


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int],
    *,
    device: torch.device | None = None,
    iterations: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    """Batch-1 wall-clock latency, reported as mean/p50/p90 milliseconds.

    Batch 1 is the only meaningful shape for an always-on wake-word device;
    batched throughput would flatter every candidate equally and rank none of
    them correctly.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    sample = torch.zeros(1, 1, *input_shape, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    try:
        with torch.no_grad():
            for _ in range(max(warmup, 0)):
                model(sample)
            synchronize()

            timings: list[float] = []
            for _ in range(iterations):
                started = perf_counter()
                model(sample)
                synchronize()
                timings.append((perf_counter() - started) * 1000.0)
    finally:
        model.train(was_training)

    timings.sort()
    return {
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p50": timings[len(timings) // 2],
        "latency_ms_p90": timings[min(int(len(timings) * 0.9), len(timings) - 1)],
    }


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int],
    *,
    device: torch.device | None = None,
    bits_per_weight: int = 32,
    bytes_per_activation: int = 1,
    latency_iterations: int = 50,
    latency_warmup: int = 10,
) -> DeploymentCost:
    """Measure every deployment cost the Pareto search compares candidates on."""
    device = device or _model_device(model)
    latency = measure_latency(
        model,
        input_shape,
        device=device,
        iterations=latency_iterations,
        warmup=latency_warmup,
    )
    return DeploymentCost(
        params=deployed_parameter_count(model),
        macs=count_macs(model, input_shape),
        weight_bytes=weight_memory_bytes(model, bits_per_weight=bits_per_weight),
        activation_peak_bytes=measure_peak_activation_bytes(
            model, input_shape, bytes_per_activation=bytes_per_activation,
        ),
        device=str(device),
        bits_per_weight=bits_per_weight,
        weight_memory_method=(
            "packed_quantized_operators_plus_dense_parameters"
            if any(isinstance(module, (nnq.Conv2d, nnq.Linear)) for module in model.modules())
            else "projected_logical_precision"
        ),
        activation_memory_method=(
            "forward_hook_conservative_branch_liveness_estimate"
            if any(
                hasattr(module, "layer_array") and hasattr(module, "skip_weights")
                for module in model.modules()
            )
            else "forward_hook_sequential_liveness_estimate"
        ),
        **latency,
    )
