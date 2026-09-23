"""Simulator-independent hardware intermediate representation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _as_array(weight: Any) -> np.ndarray:
    if hasattr(weight, "detach"):
        weight = weight.detach().cpu().numpy()
    return np.asarray(weight)


def conv_weight_matrix(weight: Any, *, groups: int, in_channels: int) -> np.ndarray:
    """Convert PyTorch convolution weights to a logical input-by-output matrix.

    Grouped convolutions are represented as a block diagonal matrix.  This
    keeps disconnected channels explicit instead of silently turning them
    into dense connectivity.
    """

    array = _as_array(weight)
    if array.ndim != 4:
        raise ValueError("Conv2d weights must have four dimensions")
    if groups < 1:
        raise ValueError("groups must be positive")
    out_channels, channels_per_group, kernel_h, kernel_w = array.shape
    if in_channels % groups:
        raise ValueError("in_channels must be divisible by groups")
    if out_channels % groups:
        raise ValueError("out_channels must be divisible by groups")
    if channels_per_group != in_channels // groups:
        raise ValueError("weight shape is inconsistent with in_channels and groups")

    local_rows = channels_per_group * kernel_h * kernel_w
    if groups == 1:
        return array.reshape(out_channels, local_rows).T.copy()

    matrix = np.zeros(
        (in_channels * kernel_h * kernel_w, out_channels),
        dtype=array.dtype,
    )
    outputs_per_group = out_channels // groups
    for group in range(groups):
        row_start = group * local_rows
        row_end = row_start + local_rows
        col_start = group * outputs_per_group
        col_end = col_start + outputs_per_group
        matrix[row_start:row_end, col_start:col_end] = array[
            col_start:col_end
        ].reshape(outputs_per_group, local_rows).T
    return matrix


def linear_weight_matrix(weight: Any) -> np.ndarray:
    """Convert PyTorch ``[out_features, in_features]`` to its transpose."""

    array = _as_array(weight)
    if array.ndim != 2:
        raise ValueError("Linear weights must have two dimensions")
    return array.T.copy()


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


@dataclass(frozen=True)
class NonCIMOperation:
    execution_name: str
    module_type: str
    output_shape: tuple[int, ...]
    execution_index: int = 0
    input_shape: tuple[int, ...] | None = None
    macs: int = 0


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "layers": [asdict(layer) for layer in self.layers],
            "non_cim_ops": [asdict(op) for op in self.non_cim_ops],
            "total_parameters": self.total_parameters,
            "total_macs": self.total_macs,
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HardwareModelIR":
        layers = tuple(
            HardwareLayer(
                **{
                    **item,
                    "input_shape": tuple(item["input_shape"]),
                    "output_shape": tuple(item["output_shape"]),
                    "weight_shape": tuple(item["weight_shape"]),
                }
            )
            for item in raw["layers"]
        )
        operations = tuple(
            NonCIMOperation(
                **{
                    **item,
                    "output_shape": tuple(item["output_shape"]),
                    "input_shape": (
                        tuple(item["input_shape"])
                        if item.get("input_shape") is not None
                        else None
                    ),
                }
            )
            for item in raw["non_cim_ops"]
        )
        return cls(
            schema_version=int(raw["schema_version"]),
            model_name=str(raw["model_name"]),
            input_shape=tuple(raw["input_shape"]),
            output_shape=tuple(raw["output_shape"]),
            layers=layers,
            non_cim_ops=operations,
            total_parameters=int(raw["total_parameters"]),
            total_macs=int(raw["total_macs"]),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "HardwareModelIR":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
