"""Extended network description consumed by the isolated backend build."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .ir import HardwareLayer, HardwareModelIR


_FIELDS = (
    "execution_index",
    "execution_name",
    "module_name",
    "op_type",
    "input_channels",
    "input_h",
    "input_w",
    "kernel_h",
    "kernel_w",
    "output_channels",
    "stride_h",
    "stride_w",
    "output_h",
    "output_w",
    "groups",
    "dilation_h",
    "dilation_w",
    "padding_h",
    "padding_w",
    "weight_rows",
    "weight_cols",
    "has_bias",
    "parameter_count",
    "macs",
    "weight_key",
)


def _dimensions(layer: HardwareLayer) -> tuple[int | None, int | None, int | None]:
    if len(layer.input_shape) == 4:
        return layer.input_shape[2], layer.input_shape[3], layer.input_shape[1]
    return None, None, layer.input_shape[-1] if layer.input_shape else None


def _row(layer: HardwareLayer) -> dict[str, object]:
    input_h, input_w, input_channels = _dimensions(layer)
    output_h = layer.output_shape[2] if len(layer.output_shape) == 4 else None
    output_w = layer.output_shape[3] if len(layer.output_shape) == 4 else None
    if layer.op_type == "linear":
        output_h = output_w = None
    weight_rows = (
        (layer.in_channels or 0) * (layer.kernel_h or 1) * (layer.kernel_w or 1)
        if layer.op_type == "conv2d"
        else layer.in_channels
    )
    weight_cols = layer.out_channels
    return {
        "execution_index": layer.execution_index,
        "execution_name": layer.execution_name,
        "module_name": layer.module_name,
        "op_type": layer.op_type,
        "input_channels": input_channels,
        "input_h": input_h,
        "input_w": input_w,
        "kernel_h": layer.kernel_h or 1,
        "kernel_w": layer.kernel_w or 1,
        "output_channels": layer.out_channels,
        "stride_h": layer.stride_h or 1,
        "stride_w": layer.stride_w or 1,
        "output_h": output_h,
        "output_w": output_w,
        "groups": layer.groups,
        "dilation_h": layer.dilation_h or 1,
        "dilation_w": layer.dilation_w or 1,
        "padding_h": layer.padding_h or 0,
        "padding_w": layer.padding_w or 0,
        "weight_rows": weight_rows,
        "weight_cols": weight_cols,
        "has_bias": int(layer.has_bias),
        "parameter_count": layer.parameter_count,
        "macs": layer.macs,
        "weight_key": layer.weight_key,
    }


def network_csv_text(ir: HardwareModelIR) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    for layer in ir.layers:
        writer.writerow(_row(layer))
    return stream.getvalue()


def write_network_csv(ir: HardwareModelIR, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(network_csv_text(ir), encoding="utf-8")
