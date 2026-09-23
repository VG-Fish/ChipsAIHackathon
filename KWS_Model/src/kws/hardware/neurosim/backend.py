"""Backend-independent export orchestration for NeuroSim inputs."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import nn

from .config import HardwareConfig
from .graph_capture import capture_graph
from .ir import HardwareModelIR, conv_weight_matrix, linear_weight_matrix
from .network_csv import write_network_csv
from .parser import parse_neurosim_output
from .quantization import symmetric_quantize
from .report import (
    HardwareResult,
    normalize_result,
    write_hardware_csv,
    write_hardware_json,
    write_hardware_markdown,
    write_layers_csv,
)
from .runner import run_simulator
from .traces import (
    TraceSample,
    activity_factor,
    calibration_scales,
    collect_activation_arrays,
    encode_fixed_point_bits,
    write_trace_sample,
)
from .v21 import export_v21_inputs


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    ir: HardwareModelIR
    manifest: Mapping[str, Any]


def _format_command(command: Iterable[str], root: Path, sample_id: str) -> list[str]:
    values = {
        "export_dir": str(root),
        "network": str(root / "ir/network.csv"),
        "weights": str(root / "weights"),
        "config": str(root / "configs/hardware.yaml"),
        "sample_id": sample_id,
        "sample_dir": str(root / "traces" / sample_id),
    }
    return [str(argument).format(**values) for argument in command]


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path | None) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path is not None and path.exists() else None


def _sample_tuple(value: tuple[Any, ...]) -> tuple[str, int, torch.Tensor, int | None]:
    if len(value) == 3:
        sample_id, item_index, tensor = value
        label = None
    elif len(value) == 4:
        sample_id, item_index, tensor, label = value
    else:
        raise ValueError("samples must contain (sample_id, item_index, tensor[, label])")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("sample tensor must be a torch.Tensor")
    return str(sample_id), int(item_index), tensor, int(label) if label is not None else None


def _weight_matrix(module: nn.Module, layer) -> np.ndarray:
    if layer.op_type == "conv2d":
        return conv_weight_matrix(
            module.weight,
            groups=layer.groups,
            in_channels=layer.in_channels or 0,
        )
    return linear_weight_matrix(module.weight)


def export_inputs(
    model: nn.Module,
    samples: Iterable[tuple[Any, ...]],
    config: HardwareConfig,
    output_dir: str | Path,
    *,
    model_metadata: Mapping[str, Any] | None = None,
) -> ExportResult:
    """Export model IR, quantized weights, and real-input activation traces."""

    selected = [_sample_tuple(sample) for sample in samples]
    if not selected:
        raise ValueError("at least one real sample is required for export")
    model.cpu().eval()
    parameters_before = [parameter.detach().cpu().clone() for parameter in model.parameters()]

    first_output = capture_graph(
        model,
        selected[0][2].cpu(),
        model_name=str((model_metadata or {}).get("model_name", model.__class__.__name__)),
        grouped_conv_mode=config.graph.grouped_conv_mode,
        reject_unsupported_ops=config.graph.reject_unsupported_ops,
    )
    if not _finite(first_output.model_output):
        raise ValueError("model output contains NaN or Inf")
    ir = first_output.ir
    mac_cross_check: dict[str, Any] = {"reference_macs": None, "matches": None}
    if len(ir.input_shape) == 4 and ir.input_shape[0] == 1:
        from kws.utils.profile import count_macs

        reference_macs = int(count_macs(model, tuple(ir.input_shape[-2:])))
        mac_cross_check = {
            "reference_macs": reference_macs,
            "export_macs": ir.total_macs,
            "matches": reference_macs == ir.total_macs,
        }
        if reference_macs != ir.total_macs:
            raise RuntimeError(
                "hardware IR MAC count disagrees with the project profiler: "
                f"IR={ir.total_macs}, profiler={reference_macs}"
            )
    root = Path(output_dir).resolve()
    (root / "ir").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "traces").mkdir(parents=True, exist_ok=True)
    ir.write_json(root / "ir/model_ir.json")
    write_network_csv(ir, root / "ir/network.csv")

    modules = dict(model.named_modules())
    weight_metadata: list[dict[str, Any]] = []
    for layer_number, layer in enumerate(ir.layers):
        module = model if layer.module_name == "__root__" else modules.get(layer.module_name)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"could not resolve weight for {layer.module_name}")
        matrix = _weight_matrix(module, layer)
        quantized = symmetric_quantize(matrix, bits=config.precision.weight_bits)
        path = root / "weights" / f"layer_{layer_number:03d}.csv"
        np.savetxt(path, quantized.values, fmt="%d", delimiter=",")
        weight_metadata.append(
            {
                "execution_name": layer.execution_name,
                "weight_key": layer.weight_key,
                "bits": quantized.bits,
                "scale": quantized.scale,
                "original_min": quantized.original_min,
                "original_max": quantized.original_max,
                "quantized_min": quantized.quantized_min,
                "quantized_max": quantized.quantized_max,
                "matrix_shape": list(matrix.shape),
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        )

    layer_names = tuple(layer.execution_name for layer in ir.layers)
    arrays, _provenance = collect_activation_arrays(model, selected, layer_names)
    scales = calibration_scales(arrays, bits=config.precision.activation_bits)
    trace_manifest: list[dict[str, Any]] = []
    for sample_number, (sample_id, item_index, _tensor, label) in enumerate(selected):
        layer_bits: dict[str, np.ndarray] = {}
        activity: dict[str, float] = {}
        for layer_name in layer_names:
            values = arrays[layer_name][sample_number]
            bits = encode_fixed_point_bits(
                values,
                bits=config.precision.activation_bits,
                scale=scales[layer_name],
            )
            layer_bits[layer_name] = bits
            activity[layer_name] = activity_factor(bits)
        write_trace_sample(
            TraceSample(sample_id, item_index, label, layer_bits, activity),
            root / "traces",
        )
        trace_manifest.append(
            {
                "sample_id": sample_id,
                "item_index": item_index,
                "label": label,
                "activity": activity,
            }
        )

    v21_export = export_v21_inputs(
        model=model,
        ir=ir,
        modules=modules,
        arrays=arrays,
        scales=scales,
        selected=selected,
        config=config,
        root=root,
    )

    parameters_after = [parameter.detach().cpu() for parameter in model.parameters()]
    if len(parameters_before) != len(parameters_after) or any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    ):
        raise RuntimeError("model parameters changed during hardware export")

    hardware_config = {
        key: value for key, value in config.to_dict().items() if key != "config_path"
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/hardware.yaml").write_text(
        yaml.safe_dump(hardware_config, sort_keys=False), encoding="utf-8"
    )
    warnings = [
        "Reference ReRAM parameters are simulation assumptions, not physical measurements.",
    ]
    if ir.non_cim_ops:
        warnings.append("Non-CIM operations are recorded but not fully modeled by mapped arrays.")
    if config.graph.grouped_conv_mode == "dense_upper_bound":
        warnings.append("Grouped convolutions use an explicitly dense upper-bound mapping.")
    warnings.append(
        "Official NeuroSim V2.1 models weighted executions; bias, normalization, activation, residual, and pooling operations are not CIM rows."
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "backend": config.backend,
        "model": dict(model_metadata or {}),
        "checkpoint": {
            "path": (model_metadata or {}).get("checkpoint_path"),
            "sha256": (model_metadata or {}).get("checkpoint_sha256"),
            "run_id": (model_metadata or {}).get("run_id"),
        },
        "model_config_sha256": (model_metadata or {}).get("model_config_sha256"),
        "data_config": {
            "path": (model_metadata or {}).get("data_config_path"),
            "sha256": (model_metadata or {}).get("data_config_sha256"),
        },
        "hardware_config_sha256": _sha256_path(config.config_path),
        "model_ir": {
            "path": "ir/model_ir.json",
            "total_parameters": ir.total_parameters,
            "total_macs": ir.total_macs,
        },
        "mac_cross_check": mac_cross_check,
        "precision": {
            "weight_bits": config.precision.weight_bits,
            "activation_bits": config.precision.activation_bits,
            "cell_bits": config.precision.cell_bits,
            "adc_bits": config.precision.adc_bits,
        },
        "grouped_conv_mode": config.graph.grouped_conv_mode,
        "patch_schema": 1,
        "weights": weight_metadata,
        "activation_scales": scales,
        "trace": {
            "split": config.trace.split,
            "samples": len(selected),
            "seed": config.trace.seed,
            "sample_ids": [item[0] for item in selected],
            "examples": trace_manifest,
        },
        "warnings": warnings,
        "approximations": (
            ["dense_upper_bound_grouped_connectivity"]
            if config.graph.grouped_conv_mode == "dense_upper_bound"
            else []
        ),
        "validity": (
            "APPROXIMATE"
            if config.graph.grouped_conv_mode == "dense_upper_bound"
            else "APPROXIMATE"
        ),
        "v21": v21_export["manifest"],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExportResult(output_dir=root, ir=ir, manifest=manifest)


def _finalize_simulation_result(
    exported: ExportResult,
    parsed: list[Any],
) -> HardwareResult:
    """Normalize parsed runs and write the common report set."""

    latencies = [item.forward_latency_s for item in parsed]
    energies = [item.forward_dynamic_energy_j for item in parsed]
    leakage_values = [item.leakage_power_w for item in parsed]
    area_values = [item.chip_area_m2 for item in parsed]
    warnings = list(exported.manifest.get("warnings", []))
    if any(value is None for value in area_values) or not area_values:
        area = None
    else:
        known_area = [value for value in area_values if value is not None]
        area = statistics.fmean(known_area)
        if max(known_area) - min(known_area) > max(abs(area) * 1e-9, 1e-30):
            warnings.append("Parsed chip area differs across trace samples.")
    leakage = (
        statistics.fmean([value for value in leakage_values if value is not None])
        if all(value is not None for value in leakage_values)
        else None
    )
    result = normalize_result(
        backend=str(exported.manifest["backend"]),
        model_macs=int(exported.ir.total_macs),
        forward_latency_s=statistics.fmean(latencies),
        forward_dynamic_energy_j=statistics.fmean(energies),
        leakage_power_w=leakage,
        chip_area_m2=area,
        warnings=tuple(warnings),
        validity=str(exported.manifest.get("validity", "EXACT_WITHIN_BACKEND_MODEL")),
        raw_fields={key: value for item in parsed for key, value in item.raw_fields.items()},
    )
    reports = exported.output_dir / "reports"
    write_hardware_json(result, reports / "hardware.json")
    write_hardware_csv(result, reports / "hardware.csv")
    write_hardware_markdown(result, reports / "hardware.md")
    write_layers_csv(result, reports / "layers.csv")
    total_energies = [
        energy + ((leakage_value or 0.0) * latency)
        for energy, leakage_value, latency in zip(energies, leakage_values, latencies)
    ]
    fps_values = [1.0 / latency for latency in latencies]
    tops = (2.0 * exported.ir.total_macs) / 1e12
    tops_per_w_values = [
        tops / (energy / latency) if energy > 0 else None
        for energy, latency in zip(total_energies, latencies)
    ]

    def stats(values: list[float | None]) -> dict[str, float | None]:
        known = [value for value in values if value is not None]
        return {
            "mean": statistics.fmean(known) if known else None,
            "stdev": statistics.stdev(known) if len(known) > 1 else 0.0 if known else None,
            "minimum": min(known) if known else None,
            "maximum": max(known) if known else None,
        }

    (reports / "sample_statistics.json").write_text(
        json.dumps(
            {
                "n": len(parsed),
                "forward_latency_s": stats(latencies),
                "forward_dynamic_energy_j": stats(energies),
                "total_energy_per_inference_j": stats(total_energies),
                "fps": stats(fps_values),
                "tops_per_w": stats(tops_per_w_values),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def run_exported_v21_simulation(
    exported: ExportResult,
    executable: str | Path,
    *,
    config: HardwareConfig,
    timeout_seconds: float,
) -> HardwareResult:
    """Run the official V2.1 executable against every exported trace."""

    from .v21 import v21_command_for_sample

    v21_manifest = exported.manifest.get("v21")
    if not isinstance(v21_manifest, Mapping):
        raise ValueError("export does not contain official NeuroSim V2.1 inputs")
    v21_export = {
        "directory": exported.output_dir / "v21",
        "layers": [],
    }
    raw_manifest = json.loads(
        (exported.output_dir / "v21/manifest.json").read_text(encoding="utf-8")
    )
    from .v21 import V21LayerFiles

    for item in raw_manifest["layers"]:
        input_by_sample = {
            sample_id: exported.output_dir / "v21" / "traces" / sample_id / f"layer_{item['layer_index']:03d}_input.csv"
            for sample_id in exported.manifest["trace"]["sample_ids"]
        }
        v21_export["layers"].append(
            V21LayerFiles(
                layer_index=int(item["layer_index"]),
                source_execution=str(item["source_execution"]),
                group_index=int(item["group_index"]),
                row=tuple(int(value) for value in item["row"]),
                weight=exported.output_dir / item["weight"],
                old_weight=exported.output_dir / item["old_weight"],
                input_by_sample=input_by_sample,
                activity_by_sample={},
            )
        )
    # Activities are deterministic trace metadata, and are not needed to
    # reconstruct the command's file paths; use the per-sample values stored
    # in each trace directory when rebuilding the final argument list.
    parsed = []
    sample_ids = list(exported.manifest["trace"]["sample_ids"])
    for sample_id in sample_ids:
        activity = json.loads(
            (exported.output_dir / "traces" / sample_id / "activity.json").read_text(
                encoding="utf-8"
            )
        )["activity"]
        for item in v21_export["layers"]:
            item.activity_by_sample[sample_id] = float(
                activity[item.source_execution]
            )
        run = run_simulator(
            v21_command_for_sample(
                executable=executable,
                v21_export=v21_export,
                sample_id=sample_id,
                weight_bits=config.precision.weight_bits,
                activation_bits=config.precision.activation_bits,
            ),
            cwd=exported.output_dir / "v21",
            sample_id=sample_id,
            output_root=exported.output_dir / "raw",
            timeout_seconds=timeout_seconds,
        )
        if run.return_code != 0:
            raise RuntimeError(
                f"NeuroSim V2.1 failed for {sample_id} with return code {run.return_code}; "
                f"see {run.output_directory}"
            )
        parsed.append(parse_neurosim_output(run.stdout))
    return _finalize_simulation_result(exported, parsed)


def run_exported_simulation(
    exported: ExportResult,
    command: Iterable[str],
    *,
    timeout_seconds: float,
) -> HardwareResult:
    """Run one explicit command per trace and normalize its forward metrics.

    Command arguments may use ``{network}``, ``{weights}``, ``{config}``,
    ``{sample_id}``, ``{sample_dir}``, or ``{export_dir}``.  The command is
    intentionally supplied by the caller because NeuroSim executable flags
    vary across upstream source revisions.
    """

    sample_ids = list(exported.manifest["trace"]["sample_ids"])
    if not sample_ids:
        raise ValueError("export contains no trace samples")
    parsed = []
    for sample_id in sample_ids:
        run = run_simulator(
            _format_command(command, exported.output_dir, sample_id),
            cwd=exported.output_dir,
            sample_id=sample_id,
            output_root=exported.output_dir / "raw",
            timeout_seconds=timeout_seconds,
        )
        if run.return_code != 0:
            raise RuntimeError(
                f"NeuroSim failed for {sample_id} with return code {run.return_code}; "
                f"see {run.output_directory}"
            )
        parsed.append(parse_neurosim_output(run.stdout))

    latencies = [item.forward_latency_s for item in parsed]
    energies = [item.forward_dynamic_energy_j for item in parsed]
    leakage_values = [item.leakage_power_w for item in parsed]
    area_values = [item.chip_area_m2 for item in parsed]
    warnings = list(exported.manifest.get("warnings", []))
    if any(value is None for value in area_values) or not area_values:
        area = None
    else:
        known_area = [value for value in area_values if value is not None]
        area = statistics.fmean(known_area)
        if max(known_area) - min(known_area) > max(abs(area) * 1e-9, 1e-30):
            warnings.append("Parsed chip area differs across trace samples.")
    leakage = (
        statistics.fmean([value for value in leakage_values if value is not None])
        if all(value is not None for value in leakage_values)
        else None
    )
    result = normalize_result(
        backend=str(exported.manifest["backend"]),
        model_macs=int(exported.ir.total_macs),
        forward_latency_s=statistics.fmean(latencies),
        forward_dynamic_energy_j=statistics.fmean(energies),
        leakage_power_w=leakage,
        chip_area_m2=area,
        warnings=tuple(warnings),
        validity=str(exported.manifest.get("validity", "EXACT_WITHIN_BACKEND_MODEL")),
        raw_fields={key: value for item in parsed for key, value in item.raw_fields.items()},
    )
    reports = exported.output_dir / "reports"
    write_hardware_json(result, reports / "hardware.json")
    write_hardware_csv(result, reports / "hardware.csv")
    write_hardware_markdown(result, reports / "hardware.md")
    write_layers_csv(result, reports / "layers.csv")
    total_energies = [
        energy + ((leakage_value or 0.0) * latency)
        for energy, leakage_value, latency in zip(energies, leakage_values, latencies)
    ]
    fps_values = [1.0 / latency for latency in latencies]
    tops = (2.0 * exported.ir.total_macs) / 1e12
    tops_per_w_values = [
        tops / (energy / latency) if energy > 0 else None
        for energy, latency in zip(total_energies, latencies)
    ]

    def stats(values: list[float | None]) -> dict[str, float | None]:
        known = [value for value in values if value is not None]
        return {
            "mean": statistics.fmean(known) if known else None,
            "stdev": statistics.stdev(known) if len(known) > 1 else 0.0 if known else None,
            "minimum": min(known) if known else None,
            "maximum": max(known) if known else None,
        }

    (reports / "sample_statistics.json").write_text(
        json.dumps(
            {
                "n": len(parsed),
                "forward_latency_s": stats(latencies),
                "forward_dynamic_energy_j": stats(energies),
                "total_energy_per_inference_j": stats(total_energies),
                "fps": stats(fps_values),
                "tops_per_w": stats(tops_per_w_values),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
