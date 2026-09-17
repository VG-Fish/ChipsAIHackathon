"""Retrofit descriptive PAI minimum-parameter rows into finished reports.

Completed ``sparknet_dendritic_prune_experiment.yaml`` reports carry a
``validation_accuracy_gain`` measured against the prune-fine-tune baseline.
That remains the report headline because it describes the complete pipeline,
but it is not a causal estimate of what dendrites add: the two checkpoints
received different training schedules.

PerforatedAI also wrote ``<candidate>_best_arch_scores.csv`` for each completed
candidate.  Its minimum-parameter row is useful context for the architecture
search, but it is not an independently trained, schedule-matched no-dendrite
counterfactual.  ``kws.optimize.dendritic`` owns the parsing
(``read_pai_zero_dendrite_score``); this script only adds that descriptive
comparison to reports produced before the live experiment did so.

The prune-fine-tune delta remains the headline and cost basis.  Separate fields
record final and search accuracy differences versus the minimum-parameter PAI
row without calling those differences a dendrite effect.

Dry run by default.  ``--apply`` writes, and refuses any run whose lock is
still held so it cannot race a live experiment's own report writes.
"""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from typing import Any, Iterable

import yaml

from kws.optimize.dendritic import read_pai_zero_dendrite_score
from kws.utils.artifacts import ArtifactLayout

REPORT_NAME = "sparknet_dendritic_prune_experiment.yaml"
DEFAULT_SCAN_ROOT = Path("outputs")


def discover_run_roots(
    run_dirs: Iterable[Path], patterns: Iterable[str], scan_root: Path
) -> list[Path]:
    """Collect every run root that owns a dendritic experiment report.

    Explicit ``--run-dir`` values are taken as given so a caller can name a run
    whose report is missing and get a clear message about it, rather than
    having it silently vanish from an auto-discovered set.
    """
    roots: list[Path] = [Path(run_dir) for run_dir in run_dirs]
    globbed: list[Path] = []
    for pattern in patterns:
        globbed.extend(sorted(Path().glob(pattern)))
    if not roots and not globbed:
        globbed = [
            report.parent.parent
            for report in sorted(scan_root.glob(f"**/reports/{REPORT_NAME}"))
        ]
    for candidate in globbed:
        if (candidate / "reports" / REPORT_NAME).exists():
            roots.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def candidate_pai_dir(run_root: Path, record: dict[str, Any]) -> Path:
    """Locate the PAI save directory that produced a completed candidate.

    ``dendritic.checkpoint`` is not one shape.  A candidate whose supervised
    resume never ran keeps PAI's own export, stored relative to the run root as
    ``pai/candidates/<name>/final_clean_pai.pt``; a candidate whose resume did
    run has the field replaced by the absolute path of that resume's best
    checkpoint under ``models/checkpoints/sparsity/<name>/<phase>/best.pt``.
    Both spellings carry the candidate's name, which is the leaf of its PAI
    directory, so the name is recovered from whichever shape is present and the
    width is used only as a last resort.
    """
    checkpoint = Path(str(record["dendritic"]["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = run_root / checkpoint
    if checkpoint.name == "final_clean_pai.pt":
        return checkpoint.parent

    parts = checkpoint.parts
    if "sparsity" in parts:
        name = parts[parts.index("sparsity") + 1]
    else:
        name = f"sparknet_c{int(record['width'])}_multilayer"
    return run_root / "pai" / "candidates" / name


def _insert_after(
    mapping: dict[str, Any], anchor: str, additions: dict[str, Any]
) -> None:
    """Splice keys in after ``anchor`` so a backfilled report reads like a fresh one.

    Plain assignment would append the new keys at the end of the block, which
    leaves the control sitting far from the number it is the control for and
    makes a backfilled report diff noisily against one a current run wrote.
    """
    if anchor not in mapping:
        mapping.update(additions)
        return
    rebuilt: dict[str, Any] = {}
    for key, value in mapping.items():
        rebuilt[key] = value
        if key == anchor:
            rebuilt.update(additions)
    for key in additions:
        rebuilt.setdefault(key, additions[key])
    mapping.clear()
    mapping.update(rebuilt)


class Row:
    """One candidate's descriptive comparison, for the dry-run table."""

    def __init__(
        self,
        run_root: Path,
        width: int,
        pipeline_gain: float,
        min_row_gain: float,
        min_row_accuracy: float,
        min_row_params: int,
    ) -> None:
        self.run_root = run_root
        self.width = width
        self.pipeline_gain = pipeline_gain
        self.min_row_gain = min_row_gain
        self.min_row_accuracy = min_row_accuracy
        self.min_row_params = min_row_params

    def format(self) -> str:
        return (
            f"{str(self.run_root):<62} C{self.width:<3} "
            f"{self.pipeline_gain:+13.6f} {self.min_row_gain:+13.6f}"
        )


def backfill(run_root: Path, *, apply: bool) -> list[Row]:
    """Add a descriptive minimum-parameter PAI-row comparison to one report."""
    layout = ArtifactLayout(run_root)
    report_path = layout.report_path(REPORT_NAME)
    if not report_path.exists():
        print(f"{run_root}: no {REPORT_NAME} report, skipped")
        return []
    report = yaml.safe_load(report_path.read_text())

    rows: list[Row] = []
    for record in report.get("candidates", []):
        if record.get("status") != "complete":
            continue
        comparison = record.get("comparison")
        if not isinstance(comparison, dict):
            print(
                f"{run_root} C{record['width']}: complete but has no comparison, skipped"
            )
            continue
        if comparison.get("pai_zero_architecture_comparison_basis"):
            print(f"{run_root} C{record['width']}: already has the PAI-row comparison")
            continue

        pai_dir = candidate_pai_dir(run_root, record)
        try:
            min_row_accuracy, min_row_params = read_pai_zero_dendrite_score(
                str(pai_dir)
            )
        except (FileNotFoundError, ValueError) as exc:
            # The row lives only in PAI's own summary. Without it there is
            # nothing to write, and guessing one would invent evidence.
            print(f"{run_root} C{record['width']}: no PAI minimum row ({exc}), skipped")
            continue

        dendritic = record["dendritic"]
        accuracy = float(dendritic["validation_accuracy"])
        baseline_accuracy = float(record["baseline"]["validation_accuracy"])
        pipeline_gain = accuracy - baseline_accuracy
        min_row_gain = accuracy - min_row_accuracy
        rows.append(
            Row(
                run_root,
                int(record["width"]),
                pipeline_gain,
                min_row_gain,
                min_row_accuracy,
                min_row_params,
            )
        )
        if not apply:
            continue

        _insert_after(
            dendritic,
            "pai_search_validation_accuracy",
            {
                "zero_dendrite_validation_accuracy": min_row_accuracy,
                "zero_dendrite_params": int(min_row_params),
            },
        )
        comparison["validation_accuracy_gain"] = pipeline_gain
        comparison["validation_accuracy_gain_basis"] = "prune_finetune_baseline"
        comparison["validation_accuracy_gain_vs_prune_finetune"] = pipeline_gain
        comparison["final_validation_accuracy_gain_vs_pai_zero_architecture"] = (
            min_row_gain
        )
        comparison["pai_zero_architecture_comparison_basis"] = (
            "minimum_parameter_row_in_best_arch_scores"
        )
        search_accuracy = dendritic.get("pai_search_validation_accuracy")
        if search_accuracy is not None:
            search_accuracy = float(search_accuracy)
            comparison["pai_search_accuracy_gain_vs_prune_finetune"] = (
                search_accuracy - baseline_accuracy
            )
            comparison["pai_search_accuracy_gain"] = search_accuracy - min_row_accuracy
            comparison["pai_search_accuracy_gain_vs_pai_zero_architecture"] = (
                search_accuracy - min_row_accuracy
            )
            comparison["pai_search_accuracy_gain_basis"] = (
                "best_architecture_row_minus_minimum_parameter_row"
            )

    if apply and rows:
        layout.atomic_yaml(report_path, report)
        print(f"{run_root}: rewrote {report_path}")
    return rows


def run_is_live(run_root: Path) -> bool:
    """True while a training process still holds the run's advisory lock."""
    lock_path = run_root / ".run.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="An experiment run directory; repeatable.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Glob of run directories, e.g. "
            "'outputs/sparknet-c16-dendritic-*/seed*'; repeatable."
        ),
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help=(
            "Directory scanned for runs carrying a dendritic report when "
            "neither --run-dir nor --glob is given (default: outputs)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the rebased reports; without it nothing is written.",
    )
    args = parser.parse_args()

    run_roots = discover_run_roots(args.run_dir, args.glob, args.scan_root)
    if not run_roots:
        print("no runs carrying a dendritic experiment report were found")
        return

    rows: list[Row] = []
    for run_root in run_roots:
        if args.apply and run_is_live(run_root):
            print(f"{run_root}: still running, refusing to rewrite its report")
            continue
        rows.extend(backfill(run_root, apply=args.apply))

    if not rows:
        print("\nevery completed candidate already has the PAI-row comparison")
        return

    print()
    print(f"{'run':<62} {'width':<4} {'pipeline gain':>13} {'vs min row':>13}")
    print(f"{'-' * 62} {'-' * 4} {'-' * 13} {'-' * 13}")
    for row in rows:
        print(row.format())
    print()
    print("pipeline gain: final validation accuracy - prune-fine-tune baseline")
    print("vs min row: final validation accuracy - minimum-parameter PAI row")
    if not args.apply:
        print(
            f"\n{len(rows)} candidate(s) would change; rerun with --apply to write them"
        )


if __name__ == "__main__":
    main()
