"""Activation trace encoding and deterministic trace export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .quantization import symmetric_quantize
from kws.utils.profile import _hookable_branch_calls


def encode_fixed_point_bits(
    values: np.ndarray,
    *,
    bits: int,
    scale: float,
) -> np.ndarray:
    """Encode signed symmetric integers as MSB-first two's-complement bits."""

    quantized = symmetric_quantize(values, bits=bits, scale=scale).values
    modulus = 1 << bits
    unsigned = np.where(quantized < 0, quantized + modulus, quantized)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    return ((unsigned[..., None] >> shifts) & 1).astype(np.uint8)


def activity_factor(bits: np.ndarray) -> float:
    array = np.asarray(bits)
    if array.size == 0:
        return 0.0
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("activity input must contain only binary values")
    return float(np.mean(array))


@dataclass(frozen=True)
class TraceSample:
    sample_id: str
    item_index: int
    label: int | None
    layer_bits: dict[str, np.ndarray]
    activity: dict[str, float]


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return True


def collect_activation_arrays(
    model: nn.Module,
    samples: Iterable[tuple[str, int, torch.Tensor, int | None]],
    layer_names: tuple[str, ...],
) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, int, int | None]]]:
    """Capture weighted-layer inputs for deterministic calibration/export."""

    wanted = set(layer_names)
    module_names = {name.split("#", 1)[0] for name in layer_names}
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    provenance: list[tuple[str, int, int | None]] = []
    handles = []
    counters: dict[str, int] = {}

    for name, module in model.named_modules():
        capture_module_name = name or "__root__"
        if capture_module_name not in module_names:
            continue

        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            capture_name: str = capture_module_name,
        ) -> None:
            tensor = _first_tensor(inputs)
            if tensor is not None:
                if not _all_finite(tensor):
                    raise ValueError("activation input is not finite (contains NaN or Inf)")
                call_index = counters.get(capture_name, 0)
                counters[capture_name] = call_index + 1
                execution_name = f"{capture_name}#{call_index}"
                key = execution_name if execution_name in wanted else capture_name
                if key in arrays:
                    arrays[key].append(tensor.detach().cpu().numpy().copy())

        handles.append(module.register_forward_hook(hook))
    try:
        with _hookable_branch_calls(model):
            with torch.inference_mode():
                for sample_id, item_index, tensor, label in samples:
                    counters.clear()
                    before = {name: len(values) for name, values in arrays.items()}
                    output = model(tensor.cpu())
                    if not _all_finite(output):
                        raise ValueError("model output is not finite (contains NaN or Inf)")
                    if any(len(values) == before[name] for name, values in arrays.items()):
                        missing = [name for name, values in arrays.items() if len(values) == before[name]]
                        raise ValueError(f"weighted layers did not execute: {missing}")
                    provenance.append((sample_id, item_index, label))
    finally:
        for handle in handles:
            handle.remove()
    return arrays, provenance


def calibration_scales(
    arrays: dict[str, list[np.ndarray]], *, bits: int
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name, values in arrays.items():
        if not values:
            raise ValueError(f"no activation samples captured for {name}")
        merged = np.concatenate([value.reshape(-1) for value in values])
        scales[name] = symmetric_quantize(merged, bits=bits).scale
    return scales


def write_trace_sample(
    sample: TraceSample,
    destination: str | Path,
) -> None:
    root = Path(destination) / sample.sample_id
    root.mkdir(parents=True, exist_ok=True)
    for name, bits in sample.layer_bits.items():
        safe_name = name.replace(".", "_")
        np.savetxt(root / f"{safe_name}_input.csv", bits.reshape(-1, bits.shape[-1]), fmt="%d", delimiter=",")
    (root / "activity.json").write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "item_index": sample.item_index,
                "label": sample.label,
                "activity": sample.activity,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
