#!/usr/bin/env python3
"""Freeze SparkNet dendrite-arm selection using validation results only.

The winner is chosen once per width from the mean validation accuracy across
the complete seed set.  Test metrics are neither accepted nor read here.  The
resulting JSON is the sole input to the held-out-test reporting step.

The frozen target set is deliberately wider than "the winners".  Three things
go into it, all fixed before a single test batch is loaded:

* the winning arm at each width, chosen on cross-seed validation means;
* every ``REFERENCE_ARMS`` arm -- the empty-placement control -- at every
  width, because it is the only budget-matched no-dendrite comparison there
  is.  The winner alone would leave the headline claim untestable: the
  from-scratch baseline never saw the 40-epoch adaptation, the PAI schedule or
  the resume, so an arm beating it on test says as much about optimizer steps
  as about dendrites;
* the from-scratch baselines, which are the honest floor at matched width.

Carrying the control forward is not test-driven selection.  Nothing here reads
a test metric, and which checkpoints get evaluated is settled by this file
before the evaluator runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
import torch

from kws.models.registry import build_model_from_checkpoint
from kws.utils.profile import deployed_parameter_count

# scripts/ is on sys.path when this file is run directly, but not when it is
# imported as ``scripts.select_sparknet_arms``; keep the sibling import working
# either way rather than depending on some other module having done this first.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_zero_dendrite_control import candidate_pai_dir  # noqa: E402

REPORT_NAME = "sparknet_dendritic_prune_experiment.yaml"
ARM_MODULE_IDS = {
    "pointwise": frozenset({".blocks.2.pointwise", ".blocks.3.pointwise"}),
    "fc": frozenset({".fc"}),
    "gate_conv": frozenset({".gate_conv"}),
    "depthwise": frozenset({".blocks.2.depthwise", ".blocks.3.depthwise"}),
    "control": frozenset(),
}
# Arms that ride along to test at every width whether or not they win, because
# the comparison the study exists to make needs them there.
REFERENCE_ARMS = ("control",)


def _arm_name(report: dict[str, Any]) -> str:
    pai = report.get("perforatedai")
    if not isinstance(pai, dict) or not isinstance(pai.get("module_ids"), list):
        raise ValueError("report must record perforatedai.module_ids")
    selected = frozenset(str(value) for value in pai["module_ids"])
    for name, module_ids in ARM_MODULE_IDS.items():
        if selected == module_ids:
            return name
    raise ValueError(f"unsupported SparkNet arm module_ids: {sorted(selected)}")


def _checkpoint_path(run_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{run_root}: selected candidate has no checkpoint")
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = run_root / checkpoint
    if not checkpoint.exists():
        raise ValueError(f"{run_root}: selected checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


def scratch_checkpoint_path(scratch_root: Path, width: int, seed: int) -> Path:
    """Locate one from-scratch baseline in the sweep's directory layout."""
    return (
        Path(scratch_root)
        / f"c{int(width)}-seed{int(seed)}"
        / "models/checkpoints/paper_replication/best.pt"
    )


def _scratch_deployed_params(state: dict[str, Any], checkpoint: Path) -> int:
    """Count a scratch baseline the way every arm row is counted.

    Summing the raw state dict would also count BatchNorm running statistics
    and ``num_batches_tracked``, so the scratch rows would not be commensurable
    with the arms' ``deployed_params``, which come from
    ``deployed_parameter_count`` on a built model.  Rebuilding the model is
    also the cheapest real check that the checkpoint we are about to freeze
    actually loads.
    """
    try:
        model = build_model_from_checkpoint(state)
        model.load_state_dict(state["model_state_dict"], strict=True)
    except Exception as exc:  # pragma: no cover - depends on a corrupt file
        raise ValueError(
            f"scratch checkpoint {checkpoint} could not be rebuilt: {exc}"
        ) from exc
    return int(deployed_parameter_count(model))


def _verify_scratch_source(
    report: dict[str, Any], report_path: Path, scratch_root: Path, width: int, seed: int
) -> None:
    """Refuse an arm run that did not start from its own seed's baseline.

    This is the invariant the whole five-seed design rests on.  A matrix whose
    runs all grew from one scratch checkpoint reports a spread over PAI's own
    randomness, not over independent training runs, and nothing downstream can
    detect that from the numbers alone.
    """
    source = report.get("source")
    recorded = (source or {}).get("checkpoint") if isinstance(source, dict) else None
    if not recorded:
        raise ValueError(f"report does not record its source checkpoint: {report_path}")
    expected = scratch_checkpoint_path(scratch_root, width, seed)
    if Path(str(recorded)).resolve() != expected.resolve():
        raise ValueError(
            f"{report_path}: C{width} seed {seed} grew from {recorded}, not from "
            f"its own from-scratch baseline {expected}"
        )


def build_selection(
    run_roots: Iterable[Path],
    *,
    expected_seeds: Sequence[int] = tuple(range(5)),
    scratch_root: Path | None = None,
    reference_arms: Sequence[str] = REFERENCE_ARMS,
) -> dict[str, Any]:
    """Select one arm per width from complete, independent validation runs."""
    seeds = tuple(int(seed) for seed in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must be a non-empty set of distinct seeds")
    reference_arms = tuple(dict.fromkeys(str(arm) for arm in reference_arms))
    unknown = [arm for arm in reference_arms if arm not in ARM_MODULE_IDS]
    if unknown:
        raise ValueError(f"unknown reference arms: {unknown}")

    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    widths: set[int] = set()
    for raw_root in run_roots:
        run_root = Path(raw_root)
        report_path = run_root / "reports" / REPORT_NAME
        if not report_path.exists():
            raise ValueError(f"missing completed report: {report_path}")
        report = yaml.safe_load(report_path.read_text())
        if not isinstance(report, dict) or report.get("status") != "complete":
            raise ValueError(f"report is not complete: {report_path}")
        if report.get("selection_split") != "validation" or report.get(
            "test_split_used"
        ) is not False:
            raise ValueError(f"report is not validation-only: {report_path}")
        seed = report.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds:
            raise ValueError(f"report has unexpected or missing seed: {report_path}")
        arm = _arm_name(report)
        candidates = report.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"study report must contain exactly one width: {report_path}")
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get("status") != "complete":
            raise ValueError(f"candidate is not complete: {report_path}")
        dendritic = candidate.get("dendritic")
        if not isinstance(dendritic, dict):
            raise ValueError(f"candidate has no dendritic result: {report_path}")
        width = int(candidate["width"])
        key = (width, arm, seed)
        if key in rows:
            raise ValueError(f"duplicate run for C{width} {arm} seed {seed}")
        accuracy = dendritic.get("validation_accuracy")
        params = dendritic.get("deployed_params")
        if accuracy is None or params is None:
            raise ValueError(f"candidate has incomplete validation metrics: {report_path}")
        if scratch_root is not None:
            _verify_scratch_source(report, report_path, Path(scratch_root), width, seed)
        rows[key] = {
            "width": width,
            "arm": arm,
            "seed": seed,
            "validation_accuracy": float(accuracy),
            "deployed_params": int(params),
            "checkpoint": str(_checkpoint_path(run_root, dendritic.get("checkpoint"))),
            "run_root": str(run_root.resolve()),
            "pai_dir": str(candidate_pai_dir(run_root, candidate).resolve()),
        }
        widths.add(width)

    summaries_by_width: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for width in sorted(widths, reverse=True):
        for arm in ARM_MODULE_IDS:
            members = [rows.get((width, arm, seed)) for seed in seeds]
            if any(member is None for member in members):
                missing = [
                    seed for seed, member in zip(seeds, members) if member is None
                ]
                raise ValueError(f"missing C{width} {arm} seeds: {missing}")
            complete = [member for member in members if member is not None]
            summaries_by_width[width].append(
                {
                    "arm": arm,
                    "validation_accuracy_mean": statistics.fmean(
                        member["validation_accuracy"] for member in complete
                    ),
                    "validation_accuracy_sample_sd": (
                        statistics.stdev(
                            member["validation_accuracy"] for member in complete
                        )
                        if len(complete) > 1
                        else None
                    ),
                    "mean_deployed_params": statistics.fmean(
                        member["deployed_params"] for member in complete
                    ),
                }
            )
    frozen_widths: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for width in sorted(widths, reverse=True):
        summaries = summaries_by_width[width]
        winner = min(
            summaries,
            key=lambda row: (
                -row["validation_accuracy_mean"],
                row["mean_deployed_params"],
                row["arm"],
            ),
        )
        selected_arm = str(winner["arm"])
        # The winner is the answer to "which placement"; the reference arms are
        # what that answer has to be read against.  Both are fixed here, on
        # validation, so carrying the reference forward costs nothing in
        # selection integrity and is the difference between a test table that
        # can support the study's claim and one that cannot.
        carried = [selected_arm] + [
            arm for arm in reference_arms if arm != selected_arm
        ]
        frozen_widths.append(
            {
                "width": width,
                "selected_arm": selected_arm,
                "reference_arms": [arm for arm in carried if arm != selected_arm],
                "selection_metric": "mean_validation_accuracy_across_seeds",
                "arm_summaries": summaries,
            }
        )
        for arm in carried:
            targets.extend(rows[(width, arm, seed)] for seed in seeds)

    if scratch_root is not None:
        for width in sorted(widths, reverse=True):
            for seed in seeds:
                checkpoint = scratch_checkpoint_path(scratch_root, width, seed)
                if not checkpoint.exists():
                    raise ValueError(f"missing fixed scratch baseline: {checkpoint}")
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if state.get("seed") != seed:
                    raise ValueError(
                        f"scratch checkpoint {checkpoint} records seed "
                        f"{state.get('seed')!r}, expected {seed}"
                    )
                model_cfg = state.get("model_cfg") or {}
                if int(model_cfg.get("channels", -1)) != width:
                    raise ValueError(
                        f"scratch checkpoint {checkpoint} does not record C{width}"
                    )
                if state.get("val_acc") is None:
                    raise ValueError(
                        f"scratch checkpoint {checkpoint} has no validation accuracy"
                    )
                targets.append(
                    {
                        "width": width,
                        "arm": "scratch",
                        "seed": seed,
                        "validation_accuracy": float(state["val_acc"]),
                        "deployed_params": _scratch_deployed_params(
                            state, checkpoint
                        ),
                        "checkpoint": str(checkpoint.resolve()),
                        "run_root": str(
                            (Path(scratch_root) / f"c{width}-seed{seed}").resolve()
                        ),
                        "pai_dir": None,
                    }
                )

    return {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_used": False,
        "expected_seeds": list(seeds),
        "reference_arms": list(reference_arms),
        "widths": frozen_widths,
        "targets": targets,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"selection is already frozen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="*", type=Path)
    parser.add_argument("--run-glob", action="append", default=[])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help=(
            "Include the fixed from-scratch baselines in the frozen test target "
            "set, and verify each arm run grew from its own seed's baseline."
        ),
    )
    parser.add_argument(
        "--reference-arm",
        action="append",
        default=None,
        dest="reference_arms",
        metavar="ARM",
        help=(
            "Arm carried to test at every width whether or not it wins "
            f"(default: {', '.join(REFERENCE_ARMS)}); repeatable. "
            "--reference-arm none carries the winners alone."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = list(args.run_roots)
    for pattern in args.run_glob:
        roots.extend(sorted(Path().glob(pattern)))
    if not roots:
        parser.error("pass run roots or at least one --run-glob")
    requested = args.reference_arms
    reference_arms = (
        REFERENCE_ARMS
        if requested is None
        else tuple(arm for arm in requested if arm != "none")
    )
    try:
        selection = build_selection(
            roots,
            expected_seeds=args.expected_seeds,
            scratch_root=args.scratch_root,
            reference_arms=reference_arms,
        )
    except ValueError as error:
        parser.error(str(error))
    _atomic_json(args.output, selection)
    print(f"froze validation-only arm selection at {args.output}")
    for width in selection["widths"]:
        carried = ", ".join(width["reference_arms"]) or "none"
        print(
            f"  C{width['width']}: {width['selected_arm']} "
            f"(carried to test alongside: {carried})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
