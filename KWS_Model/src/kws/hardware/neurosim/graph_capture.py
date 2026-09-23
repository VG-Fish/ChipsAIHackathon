"""Runtime capture of the weighted operations actually executed by a model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .ir import HardwareLayer, HardwareModelIR, NonCIMOperation
from kws.utils.profile import _hookable_branch_calls


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _shape(value: Any) -> tuple[int, ...]:
    tensor = _first_tensor(value)
    if tensor is None:
        raise ValueError("runtime operation did not expose a tensor shape")
    return tuple(int(dimension) for dimension in tensor.shape)


def _module_type(module: nn.Module) -> str:
    return module.__class__.__name__


def _is_known_non_cim(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.Sigmoid,
            nn.Tanh,
            nn.Softmax,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.Flatten,
            nn.Identity,
        ),
    )


def _validate_weighted_modules(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if (
            isinstance(module, (nn.Conv2d, nn.Linear, nn.ParameterList, nn.ParameterDict))
            or _is_known_non_cim(module)
            or _skip_edge_count(module) > 0
        ):
            continue
        direct_parameters = list(module.named_parameters(recurse=False))
        if direct_parameters:
            label = name or _module_type(module)
            raise ValueError(f"unsupported weighted operator {label} ({_module_type(module)})")


def _skip_edge_count(module: nn.Module) -> int:
    if hasattr(module, "pai_skip_connection_count"):
        return int(getattr(module, "pai_skip_connection_count"))
    skip_weights = getattr(module, "skip_weights", None)
    if isinstance(skip_weights, nn.ParameterList):
        return sum(int(weight.shape[0]) for weight in skip_weights)
    return 0


@dataclass(frozen=True)
class CapturedGraph:
    ir: HardwareModelIR
    execution_inputs: tuple[torch.Tensor, ...]
    model_output: Any


def _conv_geometry(module: nn.Conv2d) -> tuple[int, int, int, int, int, int]:
    return (
        int(module.in_channels),
        int(module.out_channels),
        int(module.kernel_size[0]),
        int(module.kernel_size[1]),
        int(module.stride[0]),
        int(module.stride[1]),
    )


def capture_graph(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    model_name: str | None = None,
    grouped_conv_mode: str = "patched",
    reject_unsupported_ops: bool = True,
) -> CapturedGraph:
    """Run one CPU sample and build an IR from runtime execution.

    Hooks are attached only to leaf modules.  A module's object identity is
    retained through its ``weight_key``, while each call receives a distinct
    execution index/name.
    """

    if grouped_conv_mode not in {"reject", "dense_upper_bound", "patched"}:
        raise ValueError(f"unknown grouped convolution mode: {grouped_conv_mode}")
    if reject_unsupported_ops:
        _validate_weighted_modules(model)

    leaf_modules = [
        (name or "__root__", module)
        for name, module in model.named_modules()
        if not any(True for _ in module.children())
        and (bool(name) or isinstance(module, (nn.Conv2d, nn.Linear)))
    ]
    residual_modules = [
        (name or "__root__", module)
        for name, module in model.named_modules()
        if _skip_edge_count(module) > 0
    ]
    capture_modules = leaf_modules + [
        item for item in residual_modules if item not in leaf_modules
    ]
    handles = []
    counters: defaultdict[str, int] = defaultdict(int)
    layers: list[HardwareLayer] = []
    non_cim_ops: list[NonCIMOperation] = []
    execution_inputs: list[torch.Tensor] = []
    execution_index = 0

    def hook(name: str, module: nn.Module):
        nonlocal execution_index

        def on_forward(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal execution_index
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return
            call_index = counters[name]
            counters[name] += 1
            execution_name = f"{name}#{call_index}"
            input_shape = tuple(int(dimension) for dimension in input_tensor.shape)
            output_shape = tuple(int(dimension) for dimension in output_tensor.shape)

            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    if any(int(value) != 1 for value in module.dilation):
                        raise ValueError(
                            f"dilated convolution is not supported: {execution_name}"
                        )
                    if module.groups != 1 and grouped_conv_mode == "reject":
                        raise ValueError(
                            f"grouped convolution rejected by configuration: {execution_name}"
                        )
                    in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w = (
                        _conv_geometry(module)
                    )
                    padding_h, padding_w = (int(module.padding[0]), int(module.padding[1]))
                    dilation_h, dilation_w = (int(module.dilation[0]), int(module.dilation[1]))
                    macs = (
                        int(output_tensor.numel())
                        * (in_channels // int(module.groups))
                        * kernel_h
                        * kernel_w
                    )
                    weight_shape = tuple(int(dimension) for dimension in module.weight.shape)
                    parameter_count = int(module.weight.numel()) + (
                        int(module.bias.numel()) if module.bias is not None else 0
                    )
                else:
                    in_channels = int(module.in_features)
                    out_channels = int(module.out_features)
                    kernel_h = kernel_w = None
                    stride_h = stride_w = None
                    padding_h = padding_w = None
                    dilation_h = dilation_w = None
                    macs = int(output_tensor.numel()) * in_channels
                    weight_shape = tuple(int(dimension) for dimension in module.weight.shape)
                    parameter_count = int(module.weight.numel()) + (
                        int(module.bias.numel()) if module.bias is not None else 0
                    )
                layer = HardwareLayer(
                    execution_index=execution_index,
                    execution_name=execution_name,
                    module_name=name,
                    module_type=_module_type(module),
                    op_type="conv2d" if isinstance(module, nn.Conv2d) else "linear",
                    input_shape=input_shape,
                    output_shape=output_shape,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    padding_h=padding_h,
                    padding_w=padding_w,
                    dilation_h=dilation_h,
                    dilation_w=dilation_w,
                    groups=int(module.groups) if isinstance(module, nn.Conv2d) else 1,
                    weight_shape=weight_shape,
                    has_bias=module.bias is not None,
                    parameter_count=parameter_count,
                    macs=macs,
                    weight_key=f"{name}.weight",
                )
                layers.append(layer)
                execution_inputs.append(input_tensor.detach().cpu().clone())
            else:
                residual_macs = int(output_tensor.numel()) * _skip_edge_count(module)
                non_cim_ops.append(
                    NonCIMOperation(
                        execution_name=execution_name,
                        module_type=_module_type(module),
                        output_shape=output_shape,
                        execution_index=execution_index,
                        input_shape=input_shape,
                        macs=residual_macs,
                    )
                )
            execution_index += 1

        return on_forward

    try:
        for name, module in capture_modules:
            handles.append(module.register_forward_hook(hook(name, module)))
        with _hookable_branch_calls(model):
            with torch.inference_mode():
                model_output = model(sample.cpu())
    finally:
        for handle in handles:
            handle.remove()

    if not layers:
        raise ValueError("model executed no supported weighted operations")
    return CapturedGraph(
        ir=HardwareModelIR(
            schema_version=1,
            model_name=model_name or model.__class__.__name__,
            input_shape=tuple(int(dimension) for dimension in sample.shape),
            output_shape=_shape(model_output),
            layers=tuple(layers),
            non_cim_ops=tuple(non_cim_ops),
            total_parameters=sum(int(parameter.numel()) for parameter in model.parameters()),
            total_macs=sum(layer.macs for layer in layers)
            + sum(operation.macs for operation in non_cim_ops),
        ),
        execution_inputs=tuple(execution_inputs),
        model_output=model_output,
    )
