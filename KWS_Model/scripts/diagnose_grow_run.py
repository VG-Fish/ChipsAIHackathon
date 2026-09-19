"""Re-evaluate a finished grow run's clean exports and diagnose its dendrite.

    uv run --env-file .env python scripts/diagnose_grow_run.py \\
        --run-dir outputs/sparknet-grow-dendrites-v3/fc/c8-seed0 [--export both] [--device cpu]

For each requested export (``final``: the last epoch, ``best``: the best
post-switch epoch) this rebuilds the clean inference model from
``final_clean_pai.pt`` with :func:`kws.optimize.grow_clean_rebuild.rebuild_clean_model`
(plain PyTorch, no PerforatedAI licence needed), checks its validation accuracy
against the grow summary, and runs
:func:`kws.optimize.grow_diagnostics.dendrite_diagnostics`: accuracy with the
dendrite on and with every skip weight zeroed, and per-module linearity,
saturation and correlation-with-base metrics.

The validation split is rebuilt exactly as the driver builds it:
``build_datasets(..., splits=(TRAIN, VAL))`` with the run's seed.  The TRAIN
split has to be requested even though only VAL is used: one ``random.Random``
seeded stream picks every split's unknown-word examples in order, so building
VAL on its own draws a different validation set.  VAL is never augmented
(``build_datasets`` only gives the training split augmenters).  The loader is
``build_data_loader(VAL, shuffle=False)``'s single-process equivalent: same
batch size, same order.

Writes ``<run>/reports/grow_diagnostics.yaml`` (atomically; refuses to replace
an existing one without ``--force``).  ``grow_summary.yaml`` is only read.
Exit status: 0 on success, 2 when refused (run not complete, still locked,
output exists, or a requested export is missing).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import sys
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TRAIN, VAL
from kws.optimize.grow_clean_rebuild import load_clean_state, rebuild_clean_model
from kws.optimize.grow_diagnostics import dendrite_diagnostics
from kws.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path("reports/grow_summary.yaml")
OUTPUT_PATH = Path("reports/grow_diagnostics.yaml")
FORMAT_VERSION = 1
SOURCE = "diagnose_grow_run"
# export -> (summary artifacts key, summary results key it should reproduce)
EXPORTS = {
    "final": ("final_clean", "final_val_acc"),
    "best": ("best_clean", "best_val_acc_post_switch"),
}
# CPU vs MPS numerics can flip a borderline example or two.
WARN_SAMPLES = 3
EXIT_REFUSED = 2


class Refused(Exception):
    """A precondition failed; nothing was written."""


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise Refused(f"{path} is not a YAML mapping")
    return value


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def with_repo_dataset_root(data_cfg: dict) -> dict:
    """``data_cfg`` with a relative dataset root resolved against the repo, as the driver sees it."""
    dataset = dict(data_cfg["dataset"])
    dataset["root"] = str(_repo_path(str(dataset["root"])))
    return {**data_cfg, "dataset": dataset}


def run_is_live(run_dir: Path) -> bool:
    """True while a process still holds the run's advisory lock."""
    lock_path = run_dir / ".run.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def config_hash_warnings(run_dir: Path, paths: Mapping[str, Path]) -> list[str]:
    """Configs whose bytes differ from what the run's manifest recorded."""
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return ["no manifest.yaml; cannot check that the configs are the ones the run used"]
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    recorded = {
        entry.get("role"): entry.get("sha256")
        for entry in manifest.get("inputs", []) or []
        if isinstance(entry, dict)
    }
    warnings = []
    for role, path in paths.items():
        expected = recorded.get(role)
        if expected is None:
            warnings.append(f"manifest has no sha256 for {role}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            warnings.append(f"{path} changed since the run (sha256 differs from the manifest)")
    return warnings


def build_val_batches(
    data_cfg: dict, train_cfg: dict, seed: int
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[int, int], int]:
    """The driver's validation split as in-memory batches, input shape, class count."""
    set_seed(seed)
    datasets, label_map = build_datasets(
        data_cfg,
        augment=train_cfg["augment"],
        seed=seed,
        cache_features=bool(train_cfg.get("cache_features", True)),
        cache_train_features=bool(train_cfg.get("cache_train_features", False)),
        augmentation=train_cfg.get("augmentation"),
        splits=(TRAIN, VAL),
    )
    val = datasets[VAL]
    sample, _ = val[0]
    input_shape = (int(sample.shape[-2]), int(sample.shape[-1]))
    loader = DataLoader(val, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)
    batches = [(features, labels) for features, labels in loader]
    return batches, input_shape, len(label_map)


@torch.no_grad()
def accuracy(model: torch.nn.Module, batches: Iterable, device: torch.device) -> tuple[float, int]:
    model.eval()
    correct = total = 0
    for features, labels in batches:
        logits = model(features.to(device))
        correct += int((logits.argmax(1).cpu() == labels).sum())
        total += int(labels.numel())
    return correct / total, total


def diagnose_export(
    name: str,
    clean_path: Path,
    artifact: str,
    summary: Mapping[str, Any],
    model_cfg: dict,
    input_shape: tuple[int, int],
    num_classes: int,
    forward_function,
    batches: list,
    device: torch.device,
) -> tuple[dict, list[str]]:
    """Diagnostics of one clean export, plus warnings."""
    warnings: list[str] = []
    state, metadata = load_clean_state(clean_path)
    if metadata.get("format") != "perforatedai_clean_inference":
        warnings.append(f"{clean_path.name}: metadata format is {metadata.get('format')!r}")
    model = rebuild_clean_model(model_cfg, input_shape, num_classes, state, forward_function)
    model = model.to(device)
    rebuilt_acc, n_samples = accuracy(model, batches, device)
    diagnostics = dendrite_diagnostics(model, batches, device)
    if diagnostics["val_acc_dendrite_on"] != rebuilt_acc:
        warnings.append(
            f"{name}: dendrite-on accuracy {diagnostics['val_acc_dendrite_on']:.6f} differs from "
            f"the plain pass {rebuilt_acc:.6f}"
        )
    _, results_key = EXPORTS[name]
    summary_acc = (summary.get("results") or {}).get(results_key)
    record: dict[str, Any] = {
        "artifact": artifact,
        **diagnostics,
        "rebuilt_val_acc": rebuilt_acc,
        "summary_val_acc_field": f"results.{results_key}",
        "summary_val_acc": summary_acc,
    }
    if summary_acc is None:
        record["rebuilt_val_acc_minus_summary"] = None
        record["rebuilt_val_acc_minus_summary_samples"] = None
        warnings.append(f"{name}: summary has no results.{results_key} to compare with")
    else:
        difference = rebuilt_acc - float(summary_acc)
        samples = int(round(difference * n_samples))
        record["rebuilt_val_acc_minus_summary"] = difference
        record["rebuilt_val_acc_minus_summary_samples"] = samples
        if abs(samples) > WARN_SAMPLES:
            warnings.append(
                f"{name}: rebuilt accuracy is {samples:+d} samples from the summary's "
                f"results.{results_key} (more than the {WARN_SAMPLES} device numerics explain)"
            )
    return record, warnings


def atomic_write_yaml(path: Path, value: Any) -> None:
    """Write via a sibling temporary file and a rename (mode follows the umask)."""
    text = yaml.safe_dump(value, sort_keys=False, default_flow_style=False)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def format_report(report: Mapping[str, Any]) -> str:
    lines = []
    for name in EXPORTS:
        record = report.get(name)
        if record is None:
            continue
        on, off = record["val_acc_dendrite_on"], record["val_acc_dendrite_off"]
        delta = record["rebuilt_val_acc_minus_summary_samples"]
        lines.append(
            f"{name:5s}  dendrite on {on:.5f}  off {off:.5f}  off-drop {100 * (on - off):+.2f} pp  "
            f"(n={record['n_samples']}; rebuilt vs summary "
            f"{'n/a' if delta is None else f'{delta:+d} samples'})"
        )
        for module, stats in record["modules"].items():
            if "linear_r2_vs_preactivation" in stats:
                detail = (
                    f"R2 {stats['linear_r2_vs_preactivation']:.4f}  "
                    f"corr {stats['corr_with_base_output']:+.4f}  "
                    f"std ratio {stats['dendrite_to_base_std_ratio']:.4f}  "
                    f"tanh saturated {stats['tanh_saturated_fraction']:.3f} "
                    f"linear {stats['tanh_linear_fraction']:.3f}  "
                    f"|skip| {stats['skip_weight_mean_abs']:.4f}"
                )
            else:
                detail = (
                    f"corr {stats.get('corr_with_base_output', float('nan')):+.4f}  "
                    f"|skip| {stats['skip_weight_mean_abs']:.4f}  ({stats.get('note', '')})"
                )
            lines.append(f"       {module}: {detail}")
    return "\n".join(lines)


def diagnose_run(run_dir: Path, exports: list[str], device: torch.device, force: bool) -> dict:
    summary_path = run_dir / SUMMARY_PATH
    if not summary_path.is_file():
        raise Refused(f"no {SUMMARY_PATH} in {run_dir}")
    summary = _load_yaml(summary_path)
    if summary.get("status") != "complete":
        raise Refused(f"{summary_path} has status {summary.get('status')!r}, not 'complete'")
    if run_is_live(run_dir):
        raise Refused(f"{run_dir} is still locked by a running process")
    output_path = run_dir / OUTPUT_PATH
    if output_path.exists() and not force:
        raise Refused(f"{output_path} exists; pass --force to replace it")

    artifacts = summary.get("artifacts") or {}
    clean_paths: dict[str, Path] = {}
    for name in exports:
        relative = artifacts.get(EXPORTS[name][0])
        if relative is None:
            if len(exports) == 1:
                raise Refused(f"summary has no artifacts.{EXPORTS[name][0]}")
            print(f"note: summary has no artifacts.{EXPORTS[name][0]}; skipping {name}", file=sys.stderr)
            continue
        path = run_dir / relative
        if not path.is_file():
            raise Refused(f"missing export {path}")
        clean_paths[name] = path
    if not clean_paths:
        raise Refused("no clean export to diagnose")

    config_paths = {
        role: _repo_path(str(summary[role]))
        for role in ("data_config", "model_config", "train_config")
    }
    for warning in config_hash_warnings(run_dir, config_paths):
        print(f"WARN  {warning}", file=sys.stderr)
    data_cfg = with_repo_dataset_root(_load_yaml(config_paths["data_config"]))
    model_cfg = _load_yaml(config_paths["model_config"])
    train_cfg = _load_yaml(config_paths["train_config"])
    seed = int(summary["seed"])
    function_name = str((train_cfg.get("perforatedai") or {}).get("forward_function", "tanh"))
    forward_function = getattr(torch, function_name)

    batches, input_shape, num_classes = build_val_batches(data_cfg, train_cfg, seed)
    recorded_shape = summary.get("input_shape")
    if recorded_shape is not None and list(recorded_shape) != list(input_shape):
        raise Refused(f"validation input shape {list(input_shape)} != summary {recorded_shape}")
    if summary.get("num_classes") not in (None, num_classes):
        raise Refused(f"{num_classes} classes != summary num_classes {summary['num_classes']}")

    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "source": SOURCE,
        "device": str(device),
        "run_id": summary.get("run_id"),
        "forward_function": function_name,
    }
    warnings: list[str] = []
    for name, path in clean_paths.items():
        record, export_warnings = diagnose_export(
            name, path, str(path.relative_to(run_dir)), summary, model_cfg, input_shape,
            num_classes, forward_function, batches, device,
        )
        report[name] = record
        warnings.extend(export_warnings)

    if output_path.exists() and not force:
        raise Refused(f"{output_path} appeared while diagnosing; pass --force to replace it")
    atomic_write_yaml(output_path, report)
    print(format_report(report))
    for warning in warnings:
        print(f"WARN  {warning}", file=sys.stderr)
    print(f"wrote {output_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--export", choices=("final", "best", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="replace an existing grow_diagnostics.yaml")
    args = parser.parse_args(argv)
    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    exports = ["final", "best"] if args.export == "both" else [args.export]
    run_dir = args.run_dir.resolve()
    try:
        diagnose_run(run_dir, exports, device, args.force)
    except Refused as error:
        print(f"refusing: {error}", file=sys.stderr)
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
