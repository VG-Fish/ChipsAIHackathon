"""Adapter for the legacy command and file format in NeuroSim V2.1.

The official V2.1 wrapper does not consume the project's extended IR CSV.  It
expects one legacy network row and four files per weighted execution.  This
module keeps that translation explicit and records the assumptions in the
export manifest.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .ir import HardwareLayer, HardwareModelIR
from .traces import activity_factor, encode_fixed_point_bits


@dataclass(frozen=True)
class V21LayerFiles:
    """Files and legacy row for one NeuroSim V2.1 weighted execution."""

    layer_index: int
    source_execution: str
    group_index: int
    row: tuple[int, ...]
    weight: Path
    old_weight: Path
    input_by_sample: dict[str, Path]
    activity_by_sample: dict[str, float]


def _group_count(layer: HardwareLayer) -> int:
    return int(layer.groups) if layer.op_type == "conv2d" else 1


def _legacy_row(layer: HardwareLayer, group_index: int) -> list[int]:
    if layer.op_type == "linear":
        return [1, 1, int(layer.in_channels or 0), 1, 1, int(layer.out_channels or 0), 0, 1]
    if len(layer.input_shape) != 4 or len(layer.output_shape) != 4:
        raise ValueError(f"NeuroSim V2.1 requires 4-D convolution tensors: {layer.execution_name}")
    groups = _group_count(layer)
    if not 0 <= group_index < groups:
        raise ValueError("group index is out of range")
    input_h, input_w = int(layer.input_shape[2]), int(layer.input_shape[3])
    output_h, output_w = int(layer.output_shape[2]), int(layer.output_shape[3])
    kernel_h, kernel_w = int(layer.kernel_h or 1), int(layer.kernel_w or 1)
    stride_h, stride_w = int(layer.stride_h or 1), int(layer.stride_w or 1)
    padding_h, padding_w = int(layer.padding_h or 0), int(layer.padding_w or 0)
    # V2.1 has no padding fields.  Encoding the padded input dimensions keeps
    # its valid-convolution geometry equal to the model's padded convolution.
    effective_h = input_h + 2 * padding_h
    effective_w = input_w + 2 * padding_w
    expected_h = (effective_h - kernel_h) // stride_h + 1
    expected_w = (effective_w - kernel_w) // stride_w + 1
    if (expected_h, expected_w) != (output_h, output_w):
        raise ValueError(
            f"cannot encode convolution geometry for {layer.execution_name}: "
            f"V2.1 gives {(expected_h, expected_w)}, model gives {(output_h, output_w)}"
        )
    in_channels = int(layer.in_channels or 0) // groups
    out_channels = int(layer.out_channels or 0) // groups
    return [effective_h, effective_w, in_channels, kernel_h, kernel_w, out_channels, 0, stride_w]


def legacy_network_rows(ir: HardwareModelIR) -> list[list[int]]:
    """Return one official V2.1 eight-column row per mapped execution."""

    rows: list[list[int]] = []
    for layer in ir.layers:
        rows.extend(_legacy_row(layer, group) for group in range(_group_count(layer)))
    return rows


def build_v21_command(
    *,
    executable: str | Path,
    network_csv: str | Path,
    weight_bits: int,
    activation_bits: int,
    layer_files: Iterable[tuple[str | Path, str | Path, str | Path]],
    epoch: int = 0,
) -> list[str]:
    """Build the ``main epoch network weight_bits activation_bits ...`` command."""

    command = [
        str(executable),
        str(epoch),
        str(network_csv),
        str(weight_bits),
        str(activation_bits),
    ]
    for weight, old_weight, input_path in layer_files:
        command.extend([str(weight), str(old_weight), str(input_path), "1"])
    return command


def _weight_values(module: nn.Module, layer: HardwareLayer, group_index: int) -> np.ndarray:
    values = module.weight.detach().cpu().numpy()
    if layer.op_type == "linear":
        return values.T.copy()
    groups = _group_count(layer)
    outputs_per_group = values.shape[0] // groups
    start = group_index * outputs_per_group
    end = start + outputs_per_group
    return values[start:end].reshape(outputs_per_group, -1).T.copy()


def _v21_weight_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Match the official WAGE wrapper's signed normalized weight convention."""

    if not np.all(np.isfinite(values)):
        raise ValueError("weights contain NaN or Inf")
    if bits == 1:
        return np.where(values >= 0, 1.0, -1.0)
    delta = 1.0 / (2 ** (bits - 1))
    clipped = np.clip(values, -1.0 + delta, 1.0 - delta)
    return np.round(clipped * (2 ** (bits - 1))) / (2 ** (bits - 1))


def _conv_patches(values: np.ndarray, layer: HardwareLayer, group_index: int) -> np.ndarray:
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("V2.1 convolution traces require a single 4-D sample")
    _, channels, height, width = values.shape
    groups = _group_count(layer)
    channels_per_group = channels // groups
    kernel_h, kernel_w = int(layer.kernel_h or 1), int(layer.kernel_w or 1)
    stride_h, stride_w = int(layer.stride_h or 1), int(layer.stride_w or 1)
    padding_h, padding_w = int(layer.padding_h or 0), int(layer.padding_w or 0)
    padded = np.pad(
        values,
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
        mode="constant",
    )
    patches: list[np.ndarray] = []
    channel_start = group_index * channels_per_group
    channel_end = channel_start + channels_per_group
    output_h, output_w = int(layer.output_shape[2]), int(layer.output_shape[3])
    for row in range(output_h):
        for col in range(output_w):
            patch = padded[
                0,
                channel_start:channel_end,
                row * stride_h : row * stride_h + kernel_h,
                col * stride_w : col * stride_w + kernel_w,
            ]
            patches.append(patch.reshape(-1))
    return np.asarray(patches, dtype=np.float64)


def _activation_matrix(
    values: np.ndarray,
    layer: HardwareLayer,
    group_index: int,
    *,
    bits: int,
    scale: float,
) -> tuple[np.ndarray, float]:
    if layer.op_type == "conv2d":
        patches = _conv_patches(values, layer, group_index)
        encoded = encode_fixed_point_bits(patches, bits=bits, scale=scale)
        # V2.1 stores features by row and activation bit planes by column,
        # with all bit planes for one output position adjacent.
        matrix = encoded.transpose(1, 0, 2).reshape(encoded.shape[1], -1)
    else:
        flat = values.reshape(-1)
        encoded = encode_fixed_point_bits(flat, bits=bits, scale=scale)
        matrix = encoded
    return matrix, activity_factor(matrix)


def export_v21_inputs(
    *,
    model: nn.Module,
    ir: HardwareModelIR,
    modules: dict[str, nn.Module],
    arrays: dict[str, list[np.ndarray]],
    scales: dict[str, float],
    selected: list[tuple[str, int, torch.Tensor, int | None]],
    config: Any,
    root: str | Path,
) -> dict[str, Any]:
    """Write the official V2.1 network, weights, and per-sample bitstreams."""

    destination = Path(root).resolve() / "v21"
    weights_dir = destination / "weights"
    traces_dir = destination / "traces"
    weights_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    rows = legacy_network_rows(ir)
    with (destination / "NetWork.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)

    layer_files: list[V21LayerFiles] = []
    row_index = 0
    for layer_index, layer in enumerate(ir.layers):
        module = model if layer.module_name == "__root__" else modules.get(layer.module_name)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"could not resolve weight for {layer.module_name}")
        for group_index in range(_group_count(layer)):
            stem = f"layer_{row_index:03d}"
            weight_path = weights_dir / f"{stem}.csv"
            weight = _v21_weight_quantize(
                _weight_values(module, layer, group_index), config.precision.weight_bits
            )
            np.savetxt(weight_path, weight, fmt="%.8g", delimiter=",")
            old_weight_path = weights_dir / f"{stem}_old.csv"
            np.savetxt(old_weight_path, weight, fmt="%.8g", delimiter=",")
            input_by_sample: dict[str, Path] = {}
            activity_by_sample: dict[str, float] = {}
            for sample_number, (sample_id, _item_index, _tensor, _label) in enumerate(selected):
                matrix, activity = _activation_matrix(
                    arrays[layer.execution_name][sample_number],
                    layer,
                    group_index,
                    bits=config.precision.activation_bits,
                    scale=scales[layer.execution_name],
                )
                sample_dir = traces_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                input_path = sample_dir / f"{stem}_input.csv"
                np.savetxt(input_path, matrix, fmt="%d", delimiter=",")
                input_by_sample[sample_id] = input_path
                activity_by_sample[sample_id] = activity
            layer_files.append(
                V21LayerFiles(
                    layer_index=row_index,
                    source_execution=layer.execution_name,
                    group_index=group_index,
                    row=tuple(rows[row_index]),
                    weight=weight_path,
                    old_weight=old_weight_path,
                    input_by_sample=input_by_sample,
                    activity_by_sample=activity_by_sample,
                )
            )
            row_index += 1

    manifest = {
        "format": "DNN_NeuroSim_V2.1 Training_pytorch/NeuroSIM",
        "network_csv": "v21/NetWork.csv",
        "layer_count": len(layer_files),
        "layers": [
            {
                "layer_index": item.layer_index,
                "source_execution": item.source_execution,
                "group_index": item.group_index,
                "row": list(item.row),
                "weight": str(item.weight.relative_to(Path(root).resolve())),
                "old_weight": str(item.old_weight.relative_to(Path(root).resolve())),
            }
            for item in layer_files
        ],
        "assumptions": [
            "Weights use the official WAGE signed normalized quantization convention.",
            "Bias, batch normalization, ReLU, residual adds, and global average pooling are outside the CIM weighted-array rows.",
            "Convolution padding is represented as explicit zero-padded input dimensions in legacy NetWork.csv.",
            "Grouped convolutions are split into one legacy row per group, preserving connectivity.",
        ],
        "validity": "APPROXIMATE",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"directory": destination, "layers": layer_files, "manifest": manifest}


def v21_command_for_sample(
    *,
    executable: str | Path,
    v21_export: dict[str, Any],
    sample_id: str,
    weight_bits: int,
    activation_bits: int,
) -> list[str]:
    files = [
        (item.weight, item.old_weight, item.input_by_sample[sample_id])
        for item in v21_export["layers"]
    ]
    command = build_v21_command(
        executable=executable,
        network_csv=Path(v21_export["directory"]) / "NetWork.csv",
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        layer_files=files,
    )
    for index, item in enumerate(v21_export["layers"]):
        command[8 + index * 4] = str(item.activity_by_sample[sample_id])
    return command
