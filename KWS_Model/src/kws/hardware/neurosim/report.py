"""Normalized SI-unit result and report writers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


VALIDITY_LABELS = {
    "EXACT_WITHIN_BACKEND_MODEL",
    "APPROXIMATE",
    "UPPER_BOUND",
    "INVALID",
}


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
    latency_fraction: float | None = None
    energy_fraction: float | None = None


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
    per_layer: tuple[LayerHardwareResult, ...]
    component_breakdown: Mapping[str, float]
    warnings: tuple[str, ...]
    validity: str = "EXACT_WITHIN_BACKEND_MODEL"
    model_macs: int | None = None
    raw_fields: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.validity not in VALIDITY_LABELS:
            raise ValueError(f"unknown report validity {self.validity}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_result(
    *,
    backend: str,
    model_macs: int,
    forward_latency_s: float,
    forward_dynamic_energy_j: float,
    leakage_power_w: float | None,
    chip_area_m2: float | None,
    per_layer: tuple[LayerHardwareResult, ...] = (),
    component_breakdown: Mapping[str, float] | None = None,
    warnings: tuple[str, ...] = (),
    validity: str = "EXACT_WITHIN_BACKEND_MODEL",
    raw_fields: Mapping[str, str] | None = None,
) -> HardwareResult:
    if forward_latency_s <= 0:
        raise ValueError("forward latency must be positive")
    leakage_energy = (
        leakage_power_w * forward_latency_s if leakage_power_w is not None else None
    )
    total_energy = forward_dynamic_energy_j + (leakage_energy or 0.0)
    tops = (2.0 * model_macs / forward_latency_s) / 1e12
    average_power = total_energy / forward_latency_s
    tops_per_w = tops / average_power if average_power > 0 else None
    total_layer_latency = sum(layer.latency_s or 0.0 for layer in per_layer)
    total_layer_energy = sum(layer.dynamic_energy_j or 0.0 for layer in per_layer)
    normalized_layers = tuple(
        replace(
            layer,
            latency_fraction=(layer.latency_s / total_layer_latency)
            if layer.latency_s is not None and total_layer_latency > 0
            else None,
            energy_fraction=(layer.dynamic_energy_j / total_layer_energy)
            if layer.dynamic_energy_j is not None and total_layer_energy > 0
            else None,
        )
        for layer in per_layer
    )
    breakdown = dict(component_breakdown or {})
    breakdown.setdefault("dynamic_energy_j", forward_dynamic_energy_j)
    if leakage_energy is not None:
        breakdown.setdefault("leakage_energy_j", leakage_energy)
    return HardwareResult(
        backend=backend,
        chip_area_m2=chip_area_m2,
        forward_latency_s=forward_latency_s,
        forward_dynamic_energy_j=forward_dynamic_energy_j,
        leakage_power_w=leakage_power_w,
        leakage_energy_per_inference_j=leakage_energy,
        total_energy_per_inference_j=total_energy,
        fps=1.0 / forward_latency_s,
        tops=tops,
        tops_per_w=tops_per_w,
        per_layer=normalized_layers,
        component_breakdown=breakdown,
        warnings=tuple(warnings),
        validity=validity,
        model_macs=model_macs,
        raw_fields=raw_fields,
    )


def write_hardware_json(result: HardwareResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_layers_csv(result: HardwareResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(result.per_layer[0]).keys()) if result.per_layer else [
        "execution_index", "execution_name", "area_m2", "latency_s", "dynamic_energy_j",
        "mapped_rows", "mapped_cols", "num_subarrays", "latency_fraction", "energy_fraction",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in result.per_layer:
            writer.writerow(asdict(layer))


def write_hardware_csv(result: HardwareResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "backend": result.backend,
        "validity": result.validity,
        "chip_area_m2": result.chip_area_m2,
        "forward_latency_s": result.forward_latency_s,
        "forward_dynamic_energy_j": result.forward_dynamic_energy_j,
        "leakage_power_w": result.leakage_power_w,
        "leakage_energy_per_inference_j": result.leakage_energy_per_inference_j,
        "total_energy_per_inference_j": result.total_energy_per_inference_j,
        "fps": result.fps,
        "tops": result.tops,
        "tops_per_w": result.tops_per_w,
    }
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(values))
        writer.writeheader()
        writer.writerow(values)


def write_hardware_markdown(result: HardwareResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NeuroSim ReRAM Hardware Estimate",
        "",
        "## Validity",
        "",
        f"`{result.validity}`",
        "",
        "This is a NeuroSim model estimate under the specified ReRAM, circuit, mapping, precision, and activity assumptions.",
        "It is not a physical-device measurement.",
        "",
        "## Headline metrics",
        "",
        f"- Forward latency: `{result.forward_latency_s:.6g} s`",
        f"- Dynamic energy per inference: `{result.forward_dynamic_energy_j:.6g} J`",
        f"- Total energy per inference: `{result.total_energy_per_inference_j:.6g} J`",
        f"- Throughput: `{result.fps:.6g} FPS`",
        f"- TOPS: `{result.tops:.6g}`" if result.tops is not None else "- TOPS: unavailable",
        "",
        "## Per-layer hotspots",
        "",
        "| Execution | Latency (s) | Dynamic energy (J) | Latency fraction | Energy fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in result.per_layer:
        lines.append(
            f"| {layer.execution_name} | {layer.latency_s!s} | {layer.dynamic_energy_j!s} | "
            f"{layer.latency_fraction!s} | {layer.energy_fraction!s} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in result.warnings)
    if not result.warnings:
        lines.append("- None")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
