"""Command line entry points for the standalone NeuroSim estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml

from .backend import export_inputs, run_exported_v21_simulation
from .build import compile_neurosim, prepare_isolated_build
from .config import load_hardware_config
from .graph_capture import capture_graph
from .model_loader import load_inference_model, sha256_file
from .source import resolve_source_root, validate_neurosim_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kws.hardware.neurosim")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="inspect the executed weighted graph")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--model-config")
    inspect.add_argument("--data-config")

    export = commands.add_parser("export", help="export NeuroSim inputs without executing it")
    _add_run_inputs(export)

    run = commands.add_parser("run", help="export, build, execute, and report")
    _add_run_inputs(run)
    run.add_argument("--cache-dir", default=".cache/neurosim")
    run.add_argument("--executable", help="prepared NeuroSim executable")
    run.add_argument("--simulator-arg", action="append", default=[])

    validate = commands.add_parser("validate", help="validate hardware config and external source")
    validate.add_argument("--hardware-config", required=True)
    validate.add_argument("--skip-source-check", action="store_true")
    return parser


def _add_run_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--output-dir", required=True)


def _load_samples(data_config: str | Path, config, model) -> list[tuple[str, int, torch.Tensor, int | None]]:
    from kws.data.dataset import build_datasets
    from kws.data.splits import TEST, TRAIN, VAL
    from kws.utils.seed import set_seed

    data_path = Path(data_config)
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    set_seed(config.trace.seed)
    datasets, _label_map = build_datasets(
        data_cfg,
        augment=False,
        seed=config.trace.seed,
    )
    split_names = {"train": TRAIN, "val": VAL, "test": TEST}
    if config.trace.split not in split_names:
        raise ValueError(f"unsupported trace split {config.trace.split!r}")
    dataset = datasets[split_names[config.trace.split]]
    selected = []
    for item_index in range(min(config.trace.samples, len(dataset))):
        features, label = dataset[item_index]
        if features.ndim == 3:
            features = features.unsqueeze(0)
        selected.append((f"sample_{item_index:03d}", item_index, features, int(label)))
    if len(selected) < config.trace.samples:
        raise ValueError(
            f"trace requested {config.trace.samples} samples but split contains only {len(dataset)}"
        )
    return selected


def _inspect(args: argparse.Namespace) -> int:
    loaded = load_inference_model(args.checkpoint, model_config_path=args.model_config)
    if args.data_config:
        # Inspection uses the first real example when a data config is given;
        # final export always requires this path.
        class _InspectionConfig:
            trace = type("Trace", (), {"seed": 0, "split": "test", "samples": 1})()

        sample = _load_samples(args.data_config, _InspectionConfig, loaded.model)[0][2]
    else:
        shape = loaded.metadata.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("inspect needs --data-config when checkpoint input_shape is unavailable")
        sample = torch.zeros(1, 1, int(shape[0]), int(shape[1]))
        print("Warning: inspect used a zero structural sample; export requires real dataset samples.")
    captured = capture_graph(loaded.model, sample)
    print(f"Model: {captured.ir.model_name}")
    print(f"Checkpoint: {loaded.checkpoint_path}")
    print(f"Parameters: {captured.ir.total_parameters}")
    print(f"MACs: {captured.ir.total_macs}")
    print("\nWeighted executions:")
    for layer in captured.ir.layers:
        depthwise = layer.op_type == "conv2d" and layer.groups == (layer.in_channels or -1)
        print(f"[{layer.execution_index}] {layer.execution_name}")
        print(f"    {layer.module_type}")
        print(f"    input:  {list(layer.input_shape)}")
        print(f"    output: {list(layer.output_shape)}")
        print(f"    groups: {layer.groups}{' (depthwise)' if depthwise else ''}")
    print("\nEligibility: PASS")
    return 0


def _export(args: argparse.Namespace):
    config = load_hardware_config(args.hardware_config)
    loaded = load_inference_model(args.checkpoint, model_config_path=args.model_config)
    samples = _load_samples(args.data_config, config, loaded.model)
    return export_inputs(
        loaded.model,
        samples,
        config,
        args.output_dir,
        model_metadata={
            "checkpoint_path": str(loaded.checkpoint_path),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "run_id": loaded.run_id,
            "model_config_path": str(loaded.model_config_path)
            if loaded.model_config_path
            else None,
            "model_config_sha256": loaded.model_config_sha256,
            "data_config_path": str(Path(args.data_config).resolve()),
            "data_config_sha256": sha256_file(args.data_config),
            "model_name": loaded.model.__class__.__name__,
        },
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "validate":
        config = load_hardware_config(args.hardware_config)
        if not args.skip_source_check:
            identity = validate_neurosim_source(resolve_source_root(config))
            print(f"NeuroSim source: {identity.root}")
            print(f"Git commit: {identity.git_commit or 'unavailable'}")
        print(f"Configuration valid: {config.backend}")
        return 0
    if args.command == "inspect":
        return _inspect(args)
    exported = _export(args)
    print(f"Exported NeuroSim inputs to {exported.output_dir}")
    if args.command == "export":
        return 0

    config = load_hardware_config(args.hardware_config)
    source_root = resolve_source_root(config)
    artifact = prepare_isolated_build(source_root, config, args.cache_dir)
    artifact = compile_neurosim(artifact, timeout_seconds=config.runtime.timeout_seconds)
    executable = args.executable or artifact.executable
    if executable is None:
        raise ValueError("NeuroSim build completed without an executable")
    exported.manifest["neurosim"] = {
        "root": str(artifact.source_identity.root),
        "git_remote": artifact.source_identity.git_remote,
        "git_branch": artifact.source_identity.git_branch,
        "git_commit": artifact.source_identity.git_commit,
        "dirty": artifact.source_identity.dirty,
        "source_sha256": artifact.source_identity.source_sha256,
    }
    exported.manifest["build_manifest_path"] = str(artifact.manifest_path)
    (exported.output_dir / "manifest.json").write_text(
        json.dumps(dict(exported.manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = run_exported_v21_simulation(
        exported,
        executable,
        config=config,
        timeout_seconds=config.runtime.timeout_seconds,
    )
    print(f"Wrote hardware report to {exported.output_dir / 'reports/hardware.json'}")
    return 0
