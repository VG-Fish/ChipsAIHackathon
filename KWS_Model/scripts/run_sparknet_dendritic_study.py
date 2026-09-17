#!/usr/bin/env python3
"""Run the scratch-start, five-seed SparkNet dendrite placement study.

Order is deliberate: paper-recipe scratch baselines first, then pointwise,
classifier, gate, depthwise, and empty-placement arms.  Pointwise leads the
arms because it is the placement that targets what narrowing SparkNet actually
removes -- cross-channel mixing -- and it is the cheapest of the four, so the
most informative result lands first if the matrix has to be cut short.

Every arm starts from its own seed's from-scratch checkpoint rather than from a
pruned C16, and each (width, seed) pair trains its own baseline, so the spread
this study reports is over independent training runs instead of over one
network's PAI randomness.

Arm selection is frozen from cross-seed validation means.  The frozen set that
goes to test is the winner at each width, the empty-placement control at every
width, and the fixed scratch baselines -- the control rides along because it is
the only no-dendrite comparison that saw the same epoch budget, and without it
a test table cannot separate dendrites from extra optimization.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (12, 10, 8, 6, 4, 2)
SEEDS = (0, 1, 2, 3, 4)
ARM_ORDER = ("pointwise", "fc", "gate_conv", "depthwise", "control")
# The empty-placement arm rides to test at every width whether or not it wins;
# it is the only budget-matched no-dendrite comparison the study has.
REFERENCE_ARMS = ("control",)
SCRATCH_TRAIN_CONFIG = "configs/train/sparknet_narrow_paper_fast_io.yaml"
ARM_TRAIN_CONFIG = {
    "pointwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_pointwise.yaml",
    "fc": "configs/train/sparknet_c16_dendritic_prune_no_kd.yaml",
    "gate_conv": "configs/train/sparknet_c16_dendritic_prune_no_kd_gate_conv.yaml",
    "depthwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_depthwise.yaml",
    "control": "configs/train/sparknet_c16_dendritic_prune_no_kd_control.yaml",
}


@dataclass(frozen=True)
class ArmRun:
    arm: str
    width: int
    seed: int
    source_checkpoint: Path
    output_dir: Path
    config_path: Path
    config: dict[str, Any]


def _load_yaml(path: str) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def build_arm_runs(study_root: Path) -> list[ArmRun]:
    """Build the complete 5-arm x 3-width x 5-seed identity-start matrix."""
    study_root = Path(study_root)
    runs: list[ArmRun] = []
    for arm in ARM_ORDER:
        train_config = ARM_TRAIN_CONFIG[arm]
        train = _load_yaml(train_config)
        pai = copy.deepcopy(train["perforatedai"])
        for width in WIDTHS:
            for seed in SEEDS:
                output_dir = study_root / "arms" / arm / f"c{width}-seed{seed}"
                source = (
                    study_root
                    / "scratch"
                    / f"c{width}-seed{seed}"
                    / "models/checkpoints/paper_replication/best.pt"
                )
                config = {
                    "source_checkpoint": str(source),
                    "source_channels": width,
                    "widths": [width],
                    "teacher_checkpoint": None,
                    "data_config": "configs/data/speech_commands_v2_mfcc32_paper.yaml",
                    "model_config": f"configs/model/sparknet_c{width}_paper.yaml",
                    "train_config": train_config,
                    "output_dir": str(output_dir),
                    "seed": seed,
                    "objective": {
                        "metric": "validation_accuracy",
                        "use_test": False,
                    },
                    "perforatedai": pai,
                    "pruning": {"method": "identity"},
                }
                runs.append(
                    ArmRun(
                        arm=arm,
                        width=width,
                        seed=seed,
                        source_checkpoint=source,
                        output_dir=output_dir,
                        config_path=output_dir / "study_experiment.yaml",
                        config=config,
                    )
                )
    return runs


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _write_config(run: ArmRun) -> None:
    serialized = yaml.safe_dump(run.config, sort_keys=False)
    if run.config_path.exists():
        if run.config_path.read_text() != serialized:
            raise ValueError(
                f"refusing to change the recorded recipe for {run.output_dir}"
            )
        return
    run.config_path.parent.mkdir(parents=True, exist_ok=True)
    run.config_path.write_text(serialized)


def _report_complete(run: ArmRun) -> bool:
    report = run.output_dir / "reports/sparknet_dendritic_prune_experiment.yaml"
    if not report.exists():
        return False
    payload = yaml.safe_load(report.read_text())
    return isinstance(payload, dict) and payload.get("status") == "complete"


def _missing_baselines(scratch_root: Path) -> list[tuple[int, int]]:
    """Return the (width, seed) baselines the sweep has not finished.

    Both markers are required, exactly as the sweep's own skip test requires
    them: an interrupted run leaves a best.pt from a schedule that never ran
    to completion, and treating that as a baseline would quietly seed the
    study with a half-trained network.
    """
    missing: list[tuple[int, int]] = []
    for width in WIDTHS:
        for seed in SEEDS:
            run = scratch_root / f"c{width}-seed{seed}"
            if not (
                (run / "models/checkpoints/paper_replication/best.pt").exists()
                and (run / "metrics/summaries.yaml").exists()
            ):
                missing.append((width, seed))
    return missing


def run_baselines(study_root: Path, *, dry_run: bool) -> None:
    scratch_root = Path(study_root) / "scratch"
    if dry_run:
        print(
            f"plan {len(WIDTHS) * len(SEEDS)} paper-recipe scratch runs "
            f"-> {scratch_root} "
            f"(widths={WIDTHS}, seeds={SEEDS})"
        )
        return
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(scratch_root)
    # The sweep script carries its own WIDTHS/SEEDS defaults.  Left unset they
    # silently disagree with the constants above, and every arm at a missing
    # width then fails on a baseline that was never trained -- so pass them
    # rather than trusting two copies to stay in step.
    env["WIDTHS"] = " ".join(str(width) for width in WIDTHS)
    env["SEEDS"] = " ".join(str(seed) for seed in SEEDS)
    # The paper recipe with its dataloader turned up: same optimizer, same
    # schedule, same 200 epochs, but the MFCCs are computed by 8 workers
    # instead of the training process.  That is the difference between a 2.7 h
    # sweep and a 16 h one, and it blocks all 150 arm runs behind it.
    env["TRAIN_CONFIG"] = SCRATCH_TRAIN_CONFIG

    # The sweep continues past a failed run and exits non-zero at the end.
    # Propagating that would kill the launcher over one bad baseline -- after
    # it had already paid for all the others -- so take the exit code as a
    # hint and the filesystem as the truth.  Completed runs are skipped on a
    # second pass, so the retry only costs whatever is actually missing.
    command = ["bash", "scripts/run_sparknet_scratch_sweep.sh"]
    for attempt in (1, 2):
        try:
            _run(command, env=env)
        except subprocess.CalledProcessError as error:
            print(
                f"scratch sweep attempt {attempt} exited {error.returncode}",
                file=sys.stderr,
            )
        missing = _missing_baselines(scratch_root)
        if not missing:
            return
        if attempt == 1:
            print(
                f"retrying {len(missing)} incomplete scratch run(s)",
                file=sys.stderr,
            )

    # Do not raise: the widths that did train should still get their arms, and
    # run_arms already fails each cell whose baseline is absent.  main() then
    # refuses to freeze a selection over the incomplete matrix.
    print(f"\n{len(missing)} scratch baseline(s) did not complete:", file=sys.stderr)
    for width, seed in missing:
        run = scratch_root / f"c{width}-seed{seed}"
        state = "partial output present" if run.exists() else "never started"
        print(f"  C{width} seed{seed}: {state} ({run})", file=sys.stderr)
    print(
        "a partial run cannot be restarted in place -- kws.train refuses to "
        "write a fresh phase over existing metrics.  Remove the directory to "
        "retrain it cleanly, or pass --resume-dir to continue it (which gives "
        "that seed a different schedule history than the others).",
        file=sys.stderr,
    )


def run_arms(study_root: Path, *, dry_run: bool) -> list[str]:
    """Drive every arm run, returning the labels of the ones that did not finish.

    One arm that dies at hour six must not cost the seventy-four runs queued
    behind it -- the scratch sweep already works this way, and a matrix this
    long is exactly where it matters.  Failures are collected and reported at
    the end; re-invoking the launcher retries only what is still incomplete.
    """
    failures: list[str] = []
    for run in build_arm_runs(study_root):
        label = f"{run.arm} C{run.width} seed{run.seed}"
        if _report_complete(run):
            print(f"skip {label}: complete")
            continue
        if dry_run:
            print(f"plan {run.arm:<10} C{run.width} seed{run.seed} <- {run.source_checkpoint}")
            continue
        if not run.source_checkpoint.exists():
            print(
                f"FAIL {label}: scratch baseline is missing: {run.source_checkpoint}",
                file=sys.stderr,
            )
            failures.append(label)
            continue
        try:
            _write_config(run)
            command = [
                "uv",
                "run",
                "--env-file",
                ".env",
                "python",
                "-m",
                "kws.optimize.sparknet_dendritic_prune_experiment",
                "--config",
                str(run.config_path),
                "--no-KD",
            ]
            report = run.output_dir / "reports/sparknet_dendritic_prune_experiment.yaml"
            if report.exists():
                command.append("--resume")
            _run(command)
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            print(f"FAIL {label}: {error}", file=sys.stderr)
            failures.append(label)
            continue
        if not _report_complete(run):
            # A zero exit that left no completed report is a failure, not a
            # finished run; saying so now stops the next stage from freezing a
            # selection over a matrix with a hole in it.
            print(f"FAIL {label}: exited 0 without a complete report", file=sys.stderr)
            failures.append(label)
    return failures


def freeze_selection(study_root: Path, *, dry_run: bool) -> Path:
    study_root = Path(study_root)
    selection = study_root / "selection/selected_arms.json"
    if selection.exists():
        print(f"skip validation selection: already frozen at {selection}")
        return selection
    if dry_run:
        print(f"plan freeze cross-seed validation winners -> {selection}")
        return selection
    command = [
        sys.executable,
        "scripts/select_sparknet_arms.py",
        "--expected-seeds",
        *(str(seed) for seed in SEEDS),
        "--scratch-root",
        str(study_root / "scratch"),
        "--output",
        str(selection),
    ]
    for arm in REFERENCE_ARMS:
        command.extend(("--reference-arm", arm))
    command.extend(str(run.output_dir) for run in build_arm_runs(study_root))
    _run(command)
    return selection


def evaluate_frozen_selection(study_root: Path, *, dry_run: bool) -> Path:
    study_root = Path(study_root)
    selection = study_root / "selection/selected_arms.json"
    report = study_root / "selection/test_report.json"
    if report.exists():
        print(f"skip held-out test: receipt already exists at {report}")
        return report
    if dry_run:
        print(f"plan one-time held-out test from {selection} -> {report}")
        return report
    if not selection.exists():
        raise FileNotFoundError(f"frozen validation selection is missing: {selection}")
    _run(
        [
            "uv",
            "run",
            "--env-file",
            ".env",
            "python",
            "scripts/report_test_accuracy.py",
            "--selection-manifest",
            str(selection),
            "--output",
            str(report),
        ]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("baselines", "arms", "select", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("outputs/sparknet-dendritic-study-v2"),
        help=(
            "Study directory; a relative path is resolved against the repo "
            "root, which is where every stage's subprocess runs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Every stage below runs its subprocesses with cwd=ROOT, so a relative
    # study root has to mean the same directory here as it does there; left
    # alone it would silently follow whatever directory the launcher was
    # invoked from and the existence checks would disagree with the runs.
    study_root = args.study_root
    if not study_root.is_absolute():
        study_root = ROOT / study_root

    if args.stage in ("baselines", "all"):
        run_baselines(study_root, dry_run=args.dry_run)
    if args.stage in ("arms", "all"):
        failures = run_arms(study_root, dry_run=args.dry_run)
        if failures:
            print(
                f"\n{len(failures)} arm run(s) did not complete:",
                file=sys.stderr,
            )
            for label in failures:
                print(f"  {label}", file=sys.stderr)
            # Selection demands a complete matrix, so stopping here is the
            # honest outcome: re-invoke to retry only what is missing.
            print(
                "refusing to freeze a selection over an incomplete matrix; "
                "re-run --stage arms to retry the failures",
                file=sys.stderr,
            )
            return 1
    if args.stage in ("select", "all"):
        freeze_selection(study_root, dry_run=args.dry_run)
    if args.stage in ("test", "all"):
        evaluate_frozen_selection(study_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
