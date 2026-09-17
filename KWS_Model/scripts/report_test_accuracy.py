"""Evaluate already-selected checkpoints on the held-out test split.

Everything in these experiments is both selected *and* reported on validation.
That is the right way to select -- the test split must not influence which
checkpoint, which width or which dendrite placement wins -- but it means none
of the reported numbers are comparable to the SparkNet paper's 95.7 +/- 0.17,
which is a test figure.  Comparing a validation accuracy to a test accuracy is
not a small approximation: validation here is the split the checkpoint was
chosen on, so it is optimistically biased by exactly the selection that makes
it useful.

This script closes that gap in the only way that does not compromise the
experiments: it takes checkpoints that have *already* been selected -- by a
finished run's own report, on validation -- and evaluates each one once on
test.  It never chooses a checkpoint, a width, a seed or an arm by test
accuracy, and it has no flag that would let it.  Selection stays on
validation; this script only measures.

Two things follow from that and are enforced below:

* The evaluation seed is fixed (``--eval-seed``, default 0).  The data config
  synthesizes its silence clips, so an unseeded test build is a different test
  set every time and the run-to-run spread would be noise dressed up as a
  result.  ``outputs/sparknet-paper-replication/c16-seed0/test_report_fixedseed0.json``
  is an example of the artifact this produces.
* Test and validation are never printed in the same column.  A sample standard
  deviation is reported only when every result carries a distinct, recorded
  training seed. Directory names are labels, not seed provenance.

Checkpoint loading takes one of two paths, chosen per checkpoint by what is in
the file.  A plain SparkNet checkpoint (the pruned baselines, the from-scratch
runs) carries ``model_cfg`` and ``model_family``, so ``kws.evaluate``'s own
``load_model_from_checkpoint`` rebuilds it.  A dendritic checkpoint does not:
PAI's exported graph has no registry family, and after a supervised resume the
report's selected checkpoint is the resume's ``best.pt``, whose state dict fits
the exported PAI graph and nothing else.  Those go through the graph
reconstruction ``scripts/backfill_dendritic_macs.py`` already does -- rebuild
from the candidate's ``cycle_metadata.yaml`` and ``final_clean_pai.pt``, then
load the selected weights over it -- rather than a second, divergent copy of
that logic here.

Usage:

    uv run --env-file .env python scripts/report_test_accuracy.py \\
        --run-glob 'outputs/sparknet-c16-dendritic-*/seed*' --output test.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TEST
from kws.evaluate import load_model_from_checkpoint, run_inference
from kws.models.registry import checkpoint_input_shape
from kws.utils.device import get_device
from kws.utils.metrics import compute_metrics
from kws.utils.seed import set_seed

# scripts/ is on sys.path when this file is run directly, but not when it is
# imported; keep the on-disk layout knowledge in one place either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_dendritic_macs import load_clean_graph  # noqa: E402
from backfill_zero_dendrite_control import candidate_pai_dir  # noqa: E402

REPORT_NAME = "sparknet_dendritic_prune_experiment.yaml"
DEFAULT_DATA_CONFIG = "configs/data/speech_commands_v2_mfcc32_paper.yaml"
# SparkNet, Speech Commands v2, 12-class, test split. The number every arm here
# is ultimately trying to be comparable to.
PAPER_TEST_ACCURACY = 0.957
PAPER_TEST_SD = 0.0017
_SEED_SUFFIX = re.compile(r"[-_]?seed\d+$")


@dataclass
class Target:
    """One checkpoint to evaluate, plus the labels needed to group it."""

    checkpoint: Path
    arm: str
    width: int | None = None
    seed: int | None = None
    # The accuracy the checkpoint was selected on. Reported beside the test
    # number, never merged with it.
    validation_accuracy: float | None = None
    pai_dir: Path | None = None
    run_root: Path | None = None


@dataclass
class Result:
    """One finished evaluation."""

    checkpoint: str
    arm: str
    width: int | None
    seed: int | None
    validation_accuracy: float | None
    test_accuracy: float = 0.0
    test_far: float = 0.0
    test_frr: float = 0.0
    num_params: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


def _recorded_seed(mapping: dict[str, Any]) -> int | None:
    """Return an explicitly recorded seed; never infer provenance from a path."""
    value = mapping.get("seed")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_seed(run_root: Path | None) -> int | None:
    if run_root is None:
        return None
    manifest = run_root / "manifest.yaml"
    if not manifest.exists():
        return None
    payload = yaml.safe_load(manifest.read_text()) or {}
    return _recorded_seed(payload) if isinstance(payload, dict) else None


def _experiment_name(run_root: Path) -> str:
    """Name the arm a run belongs to, with the seed folded out of the name.

    Seeds are laid out two ways here: the dendritic experiments put each seed
    in its own ``seed3`` subdirectory, while the sweeps name the directory
    itself ``c12-seed3``. Either way the seed must not survive into the arm
    label, or every seed becomes its own group of one and there is nothing left
    to take a standard deviation over.
    """
    stem = _SEED_SUFFIX.sub("", run_root.name)
    if not stem:
        return run_root.parent.name
    if stem == run_root.name:
        return stem
    return f"{run_root.parent.name}/{stem}"


def _describe(target: Target) -> str:
    width = "C?" if target.width is None else f"C{target.width}"
    seed = "seed?" if target.seed is None else f"seed{target.seed}"
    return f"{target.arm} {width} {seed}"


def _run_root_of(checkpoint: Path) -> Path | None:
    for parent in checkpoint.resolve().parents:
        if (parent / "manifest.yaml").exists():
            return parent
    return None


def targets_from_report(run_root: Path) -> list[Target]:
    """Read a finished run's already-selected candidates out of its report.

    Selection has happened by the time this runs: the report records which
    checkpoint each arm won with, chosen on validation. This only reads that
    decision.
    """
    report_path = run_root / "reports" / REPORT_NAME
    if not report_path.exists():
        print(f"{run_root}: no {REPORT_NAME} report, skipped")
        return []
    report = yaml.safe_load(report_path.read_text())
    experiment = _experiment_name(run_root)
    # Historical reports may live under ``seedN`` directories even when every
    # downstream stage actually ran with seed 0. Only the report itself can
    # establish the stochastic provenance of this selected checkpoint.
    seed = _recorded_seed(report) if isinstance(report, dict) else None

    targets: list[Target] = []
    for record in report.get("candidates", []):
        if record.get("status") != "complete":
            continue
        width = int(record["width"])
        for kind in ("dendritic", "baseline"):
            block = record.get(kind) or {}
            raw = block.get("checkpoint")
            if not raw:
                continue
            checkpoint = Path(str(raw))
            if not checkpoint.is_absolute():
                checkpoint = run_root / checkpoint
            if not checkpoint.exists():
                print(f"{run_root} C{width} {kind}: {checkpoint} is missing, skipped")
                continue
            targets.append(
                Target(
                    checkpoint=checkpoint,
                    arm=f"{experiment} {kind}",
                    width=width,
                    seed=seed,
                    validation_accuracy=(
                        float(block["validation_accuracy"])
                        if block.get("validation_accuracy") is not None
                        else None
                    ),
                    pai_dir=(
                        candidate_pai_dir(run_root, record)
                        if kind == "dendritic"
                        else None
                    ),
                    run_root=run_root,
                )
            )
    return targets


def targets_from_checkpoints(paths: Iterable[Path]) -> list[Target]:
    """Describe bare checkpoint paths well enough to group them."""
    targets: list[Target] = []
    for path in paths:
        checkpoint = Path(path)
        run_root = _run_root_of(checkpoint)
        if checkpoint.name == "final_clean_pai.pt":
            # A PAI export cannot be unpickled to describe itself, so take what
            # its directory says and let the loader read the weights later.
            targets.append(
                Target(
                    checkpoint=checkpoint,
                    arm=(
                        f"{_experiment_name(run_root)} dendritic"
                        if run_root is not None
                        else checkpoint.parent.name
                    ),
                    seed=_manifest_seed(run_root),
                    pai_dir=checkpoint.parent,
                    run_root=run_root,
                )
            )
            continue
        checkpoint_meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_cfg = checkpoint_meta.get("model_cfg") or {}
        checkpoint_seed = _recorded_seed(checkpoint_meta)
        targets.append(
            Target(
                checkpoint=checkpoint,
                arm=_experiment_name(run_root)
                if run_root is not None
                else checkpoint.parent.name,
                width=int(model_cfg["channels"]) if model_cfg.get("channels") else None,
                seed=(
                    checkpoint_seed
                    if checkpoint_seed is not None
                    else _manifest_seed(run_root)
                ),
                validation_accuracy=(
                    float(checkpoint_meta["val_acc"])
                    if checkpoint_meta.get("val_acc") is not None
                    else None
                ),
                run_root=run_root,
            )
        )
    return targets


def targets_from_selection_manifest(path: Path) -> list[Target]:
    """Load targets frozen by validation without performing any selection."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"selection manifest must be an object: {path}")
    if payload.get("selection_split") != "validation":
        raise ValueError("selection manifest must use the validation split")
    if payload.get("test_split_used") is not False:
        raise ValueError("selection manifest indicates prior test use")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("selection manifest contains no targets")

    targets: list[Target] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise ValueError("selection target must be an object")
        checkpoint = Path(str(raw["checkpoint"]))
        if not checkpoint.exists():
            raise ValueError(f"selected checkpoint is missing: {checkpoint}")
        targets.append(
            Target(
                checkpoint=checkpoint,
                arm=str(raw["arm"]),
                width=int(raw["width"]),
                seed=int(raw["seed"]),
                validation_accuracy=float(raw["validation_accuracy"]),
                pai_dir=Path(str(raw["pai_dir"])) if raw.get("pai_dir") else None,
                run_root=(
                    Path(str(raw["run_root"])) if raw.get("run_root") else None
                ),
            )
        )
    return targets


def _pai_metadata(pai_dir: Path, input_shape: tuple[int, int]) -> dict[str, Any]:
    """Recover class counts for a PAI export, which carries none of its own.

    PAI's export is weights and nothing else. The counts come from the source
    checkpoint the cycle recorded pruning from, which is the same network with
    the same head, so its class and keyword counts are the candidate's.
    """
    metadata = yaml.safe_load((pai_dir / "cycle_metadata.yaml").read_text())
    source = torch.load(
        metadata["source_checkpoint"], map_location="cpu", weights_only=False
    )
    return {
        "num_keywords": int(source["num_keywords"]),
        "num_classes": int(source["num_classes"]),
        "input_shape": list(input_shape),
    }


def load_target_model(
    target: Target, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    """Return the evaluable model and the metadata describing its head.

    Three shapes turn up among the already-selected checkpoints:

    * A plain SparkNet checkpoint names its model family, so the registry
      rebuilds it -- ``kws.evaluate.load_model_from_checkpoint`` unchanged.
    * PAI's own ``final_clean_pai.pt`` is not a torch pickle at all and must be
      read through PAI's loader, which the graph reconstruction already does.
    * A dendritic candidate whose supervised resume ran is selected on that
      resume's ``best.pt``: an ordinary pickle whose state dict fits the
      exported PAI graph and nothing else, so the graph is rebuilt first and
      the selected weights loaded over it.

    The reconstruction is the one ``scripts/backfill_dendritic_macs.py``
    already performs, rather than a second copy here that could drift from it.
    """
    if target.pai_dir is None and target.checkpoint.name == "final_clean_pai.pt":
        raise ValueError(
            f"{target.checkpoint} is a PAI export but no candidate directory "
            "was resolved for it, so its graph cannot be rebuilt"
        )

    if target.checkpoint.name == "final_clean_pai.pt":
        assert target.pai_dir is not None
        # load_clean_graph loads this exact file into the graph it rebuilds and
        # verifies the state dict matches, so there is nothing further to load.
        model, input_shape = load_clean_graph(target.pai_dir, device)
        return model, _pai_metadata(target.pai_dir, input_shape)

    checkpoint = torch.load(target.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_cfg") is not None:
        model, checkpoint = load_model_from_checkpoint(str(target.checkpoint), device)
        return model, checkpoint

    if target.pai_dir is None:
        raise ValueError(
            f"{target.checkpoint} carries no model_cfg and no PAI candidate "
            "directory was resolved for it, so its graph cannot be rebuilt"
        )
    model, _ = load_clean_graph(target.pai_dir, device)
    # strict=False only to tolerate PAI's non-parameter ``tracker_string``;
    # anything else means this is not the graph these weights belong to.
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if unexpected or set(missing) - {"tracker_string"}:
        raise RuntimeError(
            f"{target.checkpoint} did not match the graph rebuilt from "
            f"{target.pai_dir}: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval(), checkpoint


def evaluate(
    targets: Sequence[Target], data_config: Path, eval_seed: int
) -> list[Result]:
    """Evaluate every target once on the test split, under one fixed seed."""
    data_cfg = yaml.safe_load(Path(data_config).read_text())
    device = get_device()

    # Re-seeded before the dataset build and before every model load so the
    # synthesized silence clips, and therefore the test set itself, are
    # identical for each checkpoint in the sweep.
    set_seed(eval_seed)
    datasets, label_map = build_datasets(
        data_cfg, augment=False, seed=eval_seed, splits={TEST}
    )
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    loader = DataLoader(datasets[TEST], batch_size=128, shuffle=False, num_workers=0)
    # One probe for the whole sweep: the feature shape is a property of the
    # data config, not of any checkpoint.
    actual_shape = tuple(datasets[TEST][0][0].shape[-2:])

    results: list[Result] = []
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {_describe(target)}")
        set_seed(eval_seed)
        model, checkpoint = load_target_model(target, device)

        expected_shape = tuple(checkpoint_input_shape(checkpoint))
        if expected_shape != actual_shape:
            raise ValueError(
                f"{data_config} produces features shaped {actual_shape} but "
                f"{target.checkpoint} expects {expected_shape}"
            )

        y_true, y_pred = run_inference(model, loader, device)
        metrics = compute_metrics(
            y_true, y_pred, int(checkpoint["num_keywords"]), label_names
        )
        metrics["num_params"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        metrics["split"] = "test"
        results.append(
            Result(
                checkpoint=str(target.checkpoint),
                arm=target.arm,
                width=target.width,
                seed=target.seed,
                validation_accuracy=target.validation_accuracy,
                test_accuracy=float(metrics["accuracy"]),
                test_far=float(metrics["far"]),
                test_frr=float(metrics["frr"]),
                num_params=int(metrics["num_params"]),
                metrics=metrics,
            )
        )
        print(
            f"    test accuracy {metrics['accuracy']:.4f}  params {metrics['num_params']}"
        )
    return results


def summarize(results: Sequence[Result]) -> list[dict[str, Any]]:
    """Aggregate results, exposing spread only for verified distinct seeds."""
    groups: dict[tuple[str, int | None], list[Result]] = {}
    for result in results:
        groups.setdefault((result.arm, result.width), []).append(result)

    summary: list[dict[str, Any]] = []
    for (arm, width), members in sorted(
        groups.items(), key=lambda item: (item[0][0], -(item[0][1] or 0))
    ):
        test = [member.test_accuracy for member in members]
        recorded_seeds = [member.seed for member in members if member.seed is not None]
        distinct_seeds = sorted(set(recorded_seeds))
        seed_provenance_verified = len(recorded_seeds) == len(members) and len(
            distinct_seeds
        ) == len(members)
        validation = [
            member.validation_accuracy
            for member in members
            if member.validation_accuracy is not None
        ]
        summary.append(
            {
                "arm": arm,
                "width": width,
                "n_results": len(test),
                "n_seeds": len(distinct_seeds),
                "seeds": distinct_seeds,
                "seed_provenance_verified": seed_provenance_verified,
                "test_accuracy_mean": statistics.fmean(test),
                "test_accuracy_sd": (
                    statistics.stdev(test)
                    if seed_provenance_verified and len(test) > 1
                    else None
                ),
                "validation_accuracy_mean": (
                    statistics.fmean(validation) if validation else None
                ),
                "validation_accuracy_sd": (
                    statistics.stdev(validation)
                    if seed_provenance_verified and len(validation) > 1
                    else None
                ),
                "mean_params": statistics.fmean(m.num_params for m in members),
            }
        )
    return summary


def print_summary(summary: Sequence[dict[str, Any]]) -> None:
    def cell(mean: float | None, sd: float | None) -> str:
        if mean is None:
            return f"{'--':>17}"
        spread = "   n/a" if sd is None else f"{sd * 100:5.2f}"
        return f"{mean * 100:6.2f} +/- {spread}"

    print()
    print("All accuracies are percentages. TEST and VALIDATION are different splits:")
    print("validation is the split every checkpoint here was SELECTED on, so it is")
    print("optimistically biased; only the TEST column is comparable to the paper.")
    print()
    header = (
        f"{'arm':<48} {'width':>5} {'runs':>4} {'seeds':>5} "
        f"{'TEST acc (sample sd)':>21} {'VALIDATION acc (sd)':>21} {'params':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        width = "--" if row["width"] is None else f"C{row['width']}"
        print(
            f"{row['arm']:<48} {width:>5} {row['n_results']:>4} {row['n_seeds']:>5} "
            f"{cell(row['test_accuracy_mean'], row['test_accuracy_sd']):>21} "
            f"{cell(row['validation_accuracy_mean'], row['validation_accuracy_sd']):>21} "
            f"{row['mean_params']:>8.0f}"
        )
    print("-" * len(header))
    print(
        f"{'paper reference (SparkNet C16, TEST)':<48} {'C16':>5} {'':>4} {'':>5} "
        f"{PAPER_TEST_ACCURACY * 100:6.2f} +/- {PAPER_TEST_SD * 100:5.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="Checkpoint paths to evaluate on the test split.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Glob of experiment run directories whose reports name the "
            "already-selected candidate checkpoints; repeatable."
        ),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "Evaluate only checkpoints frozen by validation in this manifest; "
            "cannot be combined with checkpoint paths or --run-glob."
        ),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path(DEFAULT_DATA_CONFIG),
        help=f"Data config for the test split (default: {DEFAULT_DATA_CONFIG}).",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help=(
            "Fixed seed for test-set construction. The data config synthesizes "
            "silence clips, so an unseeded build is not reproducible."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write a JSON summary here."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be evaluated without loading any data.",
    )
    args = parser.parse_args()

    targets: list[Target] = []
    if args.selection_manifest is not None:
        if args.run_glob or args.checkpoints:
            parser.error(
                "--selection-manifest cannot be combined with checkpoints or --run-glob"
            )
        targets = targets_from_selection_manifest(args.selection_manifest)
    else:
        for pattern in args.run_glob:
            for run_root in sorted(Path().glob(pattern)):
                targets.extend(targets_from_report(run_root))
        if args.checkpoints:
            targets.extend(targets_from_checkpoints(args.checkpoints))

    if not targets:
        print("nothing to evaluate: pass checkpoint paths or --run-glob")
        return 1

    if args.dry_run:
        print(f"would evaluate {len(targets)} checkpoint(s) on the TEST split")
        print(f"  data config: {args.data_config}")
        print(f"  eval seed:   {args.eval_seed}")
        for target in targets:
            validation = (
                "--"
                if target.validation_accuracy is None
                else f"{target.validation_accuracy:.4f}"
            )
            print(f"  {_describe(target):<62} val={validation}  {target.checkpoint}")
        return 0

    if args.selection_manifest is not None and args.output is None:
        parser.error("--selection-manifest requires --output for a durable test report")
    if args.selection_manifest is not None and args.output.exists():
        parser.error(
            f"test report already exists: {args.output}; refusing repeated evaluation"
        )

    results = evaluate(targets, args.data_config, args.eval_seed)
    summary = summarize(results)
    print_summary(summary)

    if args.output is not None:
        payload = {
            "split": "test",
            "selection_split": "validation",
            "eval_seed": args.eval_seed,
            "data_config": str(args.data_config),
            "paper_reference": {
                "test_accuracy": PAPER_TEST_ACCURACY,
                "test_accuracy_sd": PAPER_TEST_SD,
            },
            "groups": summary,
            "evaluations": [asdict(result) for result in results],
        }
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
