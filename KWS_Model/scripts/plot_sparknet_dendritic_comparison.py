#!/usr/bin/env python3
"""Generate editable SparkNet-versus-dendritic comparison graphics.

The default input is the completed study under
``outputs/sparknet-dendritic-study-v2``.  Each report contains a matched
SparkNet baseline and dendritic model for one arm, width, and seed.  The
script averages seed runs for the headline plots, keeps the individual seed
points faintly visible, and writes a CSV summary alongside the figures.

Run from ``KWS_Model`` with:

    uv run python scripts/plot_sparknet_dendritic_comparison.py

The constants near the top and all command-line options are intentionally
simple to edit when preparing a different figure or selecting another arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import yaml


# Edit these defaults for a different study or output location.
DEFAULT_INPUT_ROOT = Path("outputs/sparknet-dendritic-study-v2")
DEFAULT_BROADER_INPUT_ROOT = Path("outputs/sparknet-grow-dendrites-v3")
DEFAULT_PAPER_REPLICATION_ROOT = Path("outputs/sparknet-paper-replication")
DEFAULT_EXPORT_ROOT = Path("outputs/rp2040")
DEFAULT_OUTPUT_DIR = Path("outputs/plots/sparknet-dendritic-comparison")
DEFAULT_BROADER_TEST_INPUT = DEFAULT_OUTPUT_DIR / "broader_v3_test_accuracy_all.json"
# C2 and C4 are excluded from the generated figures by default.  Edit this set
# when preparing a different width range, or pass the matching include flag.
DEFAULT_EXCLUDED_WIDTHS = frozenset({2, 4})
PARETO_ARM = "control"
# Explicit trained models used for the new frontier.  The requested
# ``g16g16`` is interpreted as the repository's ``C16g16`` model name.
NEW_TRAINED_FRONTIER_MODEL_NAMES = (
    "sparknet_c6g16_paper",
    "sparknet_c9g8_paper",
    "sparknet_c10g8_paper",
    "sparknet_c10g16_paper",
    "sparknet_c12_paper",
    "sparknet_c16g16_paper",
    "sparknet_c18g16_paper",
)
# Edit this default if a different deployment point is preferred in practice.
PRACTICAL_MODEL_NAME = "sparknet_c10g8_paper"
MODEL_LABEL_SYNTAX_NOTE = "cXgY: X = channel width, Y = gate width (e.g. C10g8)"
OUTPUT_FILENAMES = {
    "summary": "comparison_summary.csv",
    "report": "model_stats.md",
    "accuracy_params_errorbars": "accuracy_vs_params_errorbars.png",
    "accuracy_macs_errorbars": "accuracy_vs_macs_errorbars.png",
    "accuracy_params_lines": "accuracy_vs_params_lines.png",
    "accuracy_macs_lines": "accuracy_vs_macs_lines.png",
    "accuracy_params_annotated": "accuracy_vs_params_annotated.png",
    "accuracy_macs_annotated": "accuracy_vs_macs_annotated.png",
    "accuracy_gain": "accuracy_gain_vs_parameter_overhead.png",
    "cost_overhead": "cost_overhead_by_width.png",
    "dashboard": "metrics_dashboard.png",
    "best_models_summary": "best_models_summary.csv",
    "faithful_sparknet_summary": "faithful_sparknet_summary.csv",
    "best_vs_faithful_params": "best_vs_faithful_accuracy_vs_params.png",
    "best_vs_faithful_macs": "best_vs_faithful_accuracy_vs_macs.png",
}
LEGACY_OUTPUT_FILENAMES = (
    "pareto_accuracy_vs_params.png",
    "pareto_accuracy_vs_macs.png",
    "pareto_colored_lines_params.png",
    "pareto_colored_lines_macs.png",
    "pareto_all_models_labeled_params.png",
    "pareto_all_models_labeled_macs.png",
)

ARM_LABELS = {
    "control": "Control",
    "fc": "Classifier",
    "depthwise": "Depthwise",
    "gate_conv": "Gate convolution",
    "pointwise": "Pointwise",
}
ARM_COLORS = {
    "control": "#4C78A8",
    "fc": "#F58518",
    "depthwise": "#54A24B",
    "gate_conv": "#E45756",
    "pointwise": "#72B7B2",
}
MODEL_COLORS = {
    "SparkNet": "#4C78A8",
    "Dendritic": "#F58518",
}


@dataclass(frozen=True)
class ComparisonRecord:
    """One completed SparkNet/dendritic pair from one seed."""

    arm: str
    width: int
    seed: int | None
    sparknet_accuracy: float
    dendritic_accuracy: float
    sparknet_params: float
    dendritic_params: float
    sparknet_macs: float
    dendritic_macs: float
    dendritic_latency_ms: float | None
    dendritic_activation_bytes: float | None
    report_path: Path


@dataclass(frozen=True)
class ComparisonSummary:
    """Mean comparison for one arm and width across available seeds."""

    arm: str
    width: int
    n_seeds: int
    sparknet_accuracy: float
    dendritic_accuracy: float
    sparknet_accuracy_std: float
    dendritic_accuracy_std: float
    sparknet_params: float
    dendritic_params: float
    sparknet_macs: float
    dendritic_macs: float
    accuracy_gain_pp: float
    parameter_overhead_pct: float
    mac_overhead_pct: float
    dendritic_latency_ms: float | None
    dendritic_activation_bytes: float | None
    seed_ids: tuple[int | None, ...]


@dataclass(frozen=True)
class BroaderModelRecord:
    """One completed v3 grow-dendrites validation run."""

    model_name: str
    arm: str
    width: int
    seed: int | None
    validation_accuracy_pct: float
    params: float
    macs: float
    report_path: Path


@dataclass(frozen=True)
class BroaderModelSummary:
    """A seed summary for one model name and training variant."""

    model_name: str
    arm: str
    width: int
    n_seeds: int
    mean_validation_accuracy_pct: float
    best_validation_accuracy_pct: float
    validation_accuracy_std_pct: float
    params: float
    macs: float
    seed_ids: tuple[int | None, ...]


@dataclass(frozen=True)
class ExportedModelRecord:
    """One exported PAI model with an INT8 test result."""

    model: str
    width: int
    seed: int | None
    test_accuracy_pct: float
    params: float
    macs: float | None
    report_path: Path


@dataclass(frozen=True)
class BroaderTestRecord:
    """Held-out test result for a selected v3 grow run."""

    model_name: str
    arm: str
    width: int
    seed: int | None
    test_accuracy_pct: float
    params: float
    report_path: Path


@dataclass(frozen=True)
class FaithfulSparkNetRecord:
    """One exact paper-recipe SparkNet before dendrites are integrated."""

    model_name: str
    arm: str
    width: int
    seed: int | None
    validation_accuracy_pct: float
    params: float
    macs: float
    report_path: Path


@dataclass(frozen=True)
class FaithfulSparkNetSummary:
    """A seed summary for one exact paper-recipe SparkNet model."""

    model_name: str
    width: int
    n_seeds: int
    mean_validation_accuracy_pct: float
    best_validation_accuracy_pct: float
    validation_accuracy_std_pct: float
    params: float
    macs: float
    seed_ids: tuple[int | None, ...]


@dataclass(frozen=True)
class PublishedReference:
    """Published SC2 reference point used for the external comparison."""

    model: str
    width: int
    accuracy_pct: float
    params: float
    macs: float


PUBLISHED_SC2_REFERENCES = (
    PublishedReference("Paper C4", 4, 83.5, 1416, 105000),
    PublishedReference("Paper C8", 8, 92.1, 2292, 190000),
    PublishedReference("Paper C16", 16, 95.7, 4636, 454500),
)


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _accuracy_fraction(value: object) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    # Reports currently store fractions; accepting percentages makes the
    # script convenient for hand-edited reports too.
    return number / 100 if number > 1 else number


def _cost(section: dict, key: str) -> float | None:
    value = _as_number(section.get(key))
    if value is not None:
        return value
    full_cost = section.get("full_cost")
    if isinstance(full_cost, dict):
        return _as_number(full_cost.get(key))
    return None


def _seed_from_run_name(run_name: str) -> int | None:
    match = re.search(r"seed(-?\d+)", run_name)
    return int(match.group(1)) if match else None


def _report_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("**/reports/sparknet_dendritic_prune_experiment.yaml"))


def _arm_and_run(root: Path, report_path: Path) -> tuple[str, str]:
    relative = report_path.relative_to(root) if root.is_dir() else report_path
    parts = relative.parts
    if "arms" in parts:
        arm_index = parts.index("arms") + 1
        if len(parts) > arm_index + 1:
            return parts[arm_index], parts[arm_index + 1]
    # A direct report is less common but still useful when editing or testing.
    run_name = report_path.parent.parent.name
    return run_name.split("-", 1)[0], run_name


def load_records(root: Path, arms: Sequence[str] | None = None) -> list[ComparisonRecord]:
    """Load completed, accuracy-and-cost-complete pairs from YAML reports."""

    requested_arms = set(arms) if arms else None
    records: list[ComparisonRecord] = []
    for report_path in _report_paths(root):
        report = yaml.safe_load(report_path.read_text()) or {}
        arm, run_name = _arm_and_run(root, report_path)
        if requested_arms and arm not in requested_arms:
            continue
        seed = _seed_from_run_name(run_name)
        for candidate in report.get("candidates", []):
            if candidate.get("status") != "complete":
                continue
            baseline = candidate.get("baseline") or {}
            dendritic = candidate.get("dendritic") or {}
            width = _as_number(candidate.get("width"))
            sparknet_accuracy = _accuracy_fraction(baseline.get("validation_accuracy"))
            dendritic_accuracy = _accuracy_fraction(dendritic.get("validation_accuracy"))
            sparknet_params = _cost(baseline, "deployed_params")
            dendritic_params = _cost(dendritic, "deployed_params")
            sparknet_macs = _cost(baseline, "macs")
            dendritic_macs = _cost(dendritic, "macs")
            full_cost = dendritic.get("full_cost") or {}
            latency = _as_number(full_cost.get("latency_ms_mean"))
            activation_bytes = _as_number(full_cost.get("activation_peak_bytes"))
            required = (
                width,
                sparknet_accuracy,
                dendritic_accuracy,
                sparknet_params,
                dendritic_params,
                sparknet_macs,
                dendritic_macs,
            )
            if any(value is None for value in required):
                continue
            records.append(
                ComparisonRecord(
                    arm=arm,
                    width=int(width),
                    seed=seed,
                    sparknet_accuracy=float(sparknet_accuracy),
                    dendritic_accuracy=float(dendritic_accuracy),
                    sparknet_params=float(sparknet_params),
                    dendritic_params=float(dendritic_params),
                    sparknet_macs=float(sparknet_macs),
                    dendritic_macs=float(dendritic_macs),
                    dendritic_latency_ms=latency,
                    dendritic_activation_bytes=activation_bytes,
                    report_path=report_path,
                )
            )
    return sorted(records, key=lambda record: (record.arm, record.width, record.seed or -1))


def _grow_report_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(root.glob("**/reports/grow_summary.yaml"))


def load_broader_model_records(root: Path) -> list[BroaderModelRecord]:
    """Load completed v3 grow-dendrites runs for the broader model scan."""

    records: list[BroaderModelRecord] = []
    for report_path in _grow_report_paths(root):
        report = yaml.safe_load(report_path.read_text()) or {}
        if report.get("status") != "complete":
            continue
        results = report.get("results") or {}
        deployed = (report.get("cost") or {}).get("deployed") or {}
        width = _as_number(report.get("width"))
        validation_accuracy = _accuracy_fraction(results.get("best_val_acc_overall"))
        params = _as_number(deployed.get("params"))
        macs = _as_number(deployed.get("macs"))
        model_name = report.get("model_name")
        arm = report.get("arm") or report.get("placement")
        if any(value is None for value in (width, validation_accuracy, params, macs)):
            continue
        if not isinstance(model_name, str) or not isinstance(arm, str):
            continue
        seed = _as_number(report.get("seed"))
        records.append(
            BroaderModelRecord(
                model_name=model_name,
                arm=arm,
                width=int(width),
                seed=None if seed is None else int(seed),
                validation_accuracy_pct=float(validation_accuracy) * 100,
                params=float(params),
                macs=float(macs),
                report_path=report_path,
            )
        )
    return sorted(records, key=lambda record: (record.params, -record.validation_accuracy_pct, record.model_name, record.seed or -1))


def _is_faithful_sparknet_model(model_name: object) -> bool:
    """Return whether a model uses the exact C-width paper SparkNet config.

    The ``g`` model names use alternate gate widths (for example C8g16).
    Exact paper-recipe models are named ``sparknet_cN_paper`` and use the
    standard 32 gate channels.
    """

    return isinstance(model_name, str) and re.fullmatch(r"sparknet_c\d+_paper", model_name) is not None


def load_faithful_sparknet_records(
    root: Path, paper_replication_root: Path | None = None
) -> list[FaithfulSparkNetRecord]:
    """Load exact paper-recipe SparkNet bases from completed training reports.

    When dedicated paper-replication summaries exist, they are included with
    the same base cost as the matching grow report.  The later best-per-model
    selection therefore chooses the dedicated replication when it is stronger.
    """

    records: list[FaithfulSparkNetRecord] = []
    for report_path in _grow_report_paths(root):
        report = yaml.safe_load(report_path.read_text()) or {}
        if report.get("status") != "complete":
            continue
        model_name = report.get("model_name")
        if not _is_faithful_sparknet_model(model_name):
            continue
        results = report.get("results") or {}
        base_cost = (report.get("cost") or {}).get("base") or {}
        width = _as_number(report.get("width"))
        validation_accuracy = _accuracy_fraction(results.get("best_val_acc_pre_switch"))
        params = _as_number(base_cost.get("params"))
        macs = _as_number(base_cost.get("macs"))
        arm = report.get("arm") or report.get("placement")
        seed = _as_number(report.get("seed"))
        if any(value is None for value in (width, validation_accuracy, params, macs)):
            continue
        if not isinstance(arm, str):
            continue
        records.append(
            FaithfulSparkNetRecord(
                model_name=model_name,
                arm=arm,
                width=int(width),
                seed=None if seed is None else int(seed),
                validation_accuracy_pct=float(validation_accuracy) * 100,
                params=float(params),
                macs=float(macs),
                report_path=report_path,
            )
        )
    if paper_replication_root is not None and paper_replication_root.exists():
        costs_by_model = {
            record.model_name: (record.params, record.macs)
            for record in records
        }
        for summary_path in sorted(paper_replication_root.glob("**/metrics/summaries.yaml")):
            summary = yaml.safe_load(summary_path.read_text()) or {}
            phases = summary.get("phases") or {}
            if not isinstance(phases, dict):
                continue
            run_name = summary_path.parent.parent.name
            seed = _seed_from_run_name(run_name)
            for phase_name, phase in phases.items():
                model_name = str(phase_name).rsplit("/", 1)[-1]
                if not _is_faithful_sparknet_model(model_name) or not isinstance(phase, dict):
                    continue
                validation_accuracy = _accuracy_fraction(phase.get("best_val_acc"))
                costs = costs_by_model.get(model_name)
                width_match = re.search(r"sparknet_c(\d+)_paper", model_name)
                if validation_accuracy is None or costs is None or width_match is None:
                    continue
                records.append(
                    FaithfulSparkNetRecord(
                        model_name=model_name,
                        arm="paper_replication",
                        width=int(width_match.group(1)),
                        seed=seed,
                        validation_accuracy_pct=float(validation_accuracy) * 100,
                        params=costs[0],
                        macs=costs[1],
                        report_path=summary_path,
                    )
                )
    return sorted(records, key=lambda record: (record.params, -record.validation_accuracy_pct, record.model_name, record.seed or -1))


def summarize_faithful_sparknet_records(
    records: Iterable[FaithfulSparkNetRecord],
) -> list[FaithfulSparkNetSummary]:
    """Aggregate faithful SparkNet bases by model, training variant, and width."""

    grouped: dict[tuple[str, str, int], list[FaithfulSparkNetRecord]] = {}
    for record in records:
        if not _is_faithful_sparknet_model(record.model_name):
            continue
        grouped.setdefault((record.model_name, record.arm, record.width), []).append(record)

    summaries: list[FaithfulSparkNetSummary] = []
    for (model_name, _arm, width), group in sorted(grouped.items()):
        accuracies = [record.validation_accuracy_pct for record in group]
        summaries.append(
            FaithfulSparkNetSummary(
                model_name=model_name,
                width=width,
                n_seeds=len(group),
                mean_validation_accuracy_pct=mean(accuracies),
                best_validation_accuracy_pct=max(accuracies),
                validation_accuracy_std_pct=_std(accuracies),
                params=mean(record.params for record in group),
                macs=mean(record.macs for record in group),
                seed_ids=tuple(record.seed for record in sorted(group, key=lambda item: item.seed or -1)),
            )
        )
    return summaries


def best_faithful_sparknet_summary(
    summaries: Iterable[FaithfulSparkNetSummary],
) -> list[FaithfulSparkNetSummary]:
    """Keep the best exact paper-recipe training variant for each model name."""

    best_by_model: dict[str, FaithfulSparkNetSummary] = {}
    for summary in summaries:
        current = best_by_model.get(summary.model_name)
        if current is None or (
            summary.best_validation_accuracy_pct,
            -summary.params,
        ) > (
            current.best_validation_accuracy_pct,
            -current.params,
        ):
            best_by_model[summary.model_name] = summary
    return sorted(best_by_model.values(), key=lambda summary: (summary.params, -summary.best_validation_accuracy_pct, summary.model_name))


def select_new_frontier_summaries(
    summaries: Iterable[BroaderModelSummary],
) -> list[BroaderModelSummary]:
    """Select the explicitly requested trained models for the new frontier."""

    summaries_by_name = {summary.model_name: summary for summary in summaries}
    return [
        summaries_by_name[model_name]
        for model_name in NEW_TRAINED_FRONTIER_MODEL_NAMES
        if model_name in summaries_by_name
    ]


def most_parameter_efficient_summary(
    summaries: Iterable[BroaderModelSummary],
) -> BroaderModelSummary | None:
    """Return the trained model with the highest validation accuracy per parameter."""

    summaries = list(summaries)
    if not summaries:
        return None
    return max(
        summaries,
        key=lambda summary: (
            summary.best_validation_accuracy_pct / summary.params,
            -summary.params,
        ),
    )


def filter_best_models_by_faithful_dominance(
    best_summaries: Iterable[BroaderModelSummary],
    faithful_summaries: Iterable[FaithfulSparkNetSummary],
    metric: str,
) -> list[BroaderModelSummary]:
    """Remove best models beaten by a faithful model at the plotted cost.

    A best trained model is omitted when a faithful SparkNet has strictly
    higher validation accuracy and no greater value for the requested cost
    metric.  The comparison is intentionally metric-specific: a model can be
    retained on the parameter graph but omitted from the MAC graph when the
    faithful model trades fewer MACs for more parameters.
    """

    if metric not in {"params", "macs"}:
        raise ValueError(f"unsupported best-vs-faithful metric: {metric}")
    faithful_cost_attribute = metric
    faithful = list(faithful_summaries)
    retained: list[BroaderModelSummary] = []
    for candidate in best_summaries:
        candidate_cost = float(getattr(candidate, metric))
        is_dominated = any(
            faithful_model.best_validation_accuracy_pct > candidate.best_validation_accuracy_pct
            and float(getattr(faithful_model, faithful_cost_attribute)) <= candidate_cost
            for faithful_model in faithful
        )
        if not is_dominated:
            retained.append(candidate)
    return retained


def summarize_broader_model_records(records: Iterable[BroaderModelRecord]) -> list[BroaderModelSummary]:
    """Aggregate v3 records by model name and training variant."""

    grouped: dict[tuple[str, str, int], list[BroaderModelRecord]] = {}
    for record in records:
        grouped.setdefault((record.model_name, record.arm, record.width), []).append(record)

    summaries: list[BroaderModelSummary] = []
    for (model_name, arm, width), group in sorted(grouped.items()):
        accuracies = [record.validation_accuracy_pct for record in group]
        summaries.append(
            BroaderModelSummary(
                model_name=model_name,
                arm=arm,
                width=width,
                n_seeds=len(group),
                mean_validation_accuracy_pct=mean(accuracies),
                best_validation_accuracy_pct=max(accuracies),
                validation_accuracy_std_pct=_std(accuracies),
                params=mean(record.params for record in group),
                macs=mean(record.macs for record in group),
                seed_ids=tuple(record.seed for record in sorted(group, key=lambda item: item.seed or -1)),
            )
        )
    return summaries


def best_broader_summary_per_model(summaries: Iterable[BroaderModelSummary]) -> list[BroaderModelSummary]:
    """Keep the best completed v3 variant for each model name.

    This intentionally produces a compact table while retaining every model
    width, including widths that do not have a published SparkNet equivalent.
    """

    best_by_model: dict[str, BroaderModelSummary] = {}
    for summary in summaries:
        current = best_by_model.get(summary.model_name)
        if current is None or (
            summary.best_validation_accuracy_pct,
            -summary.params,
        ) > (
            current.best_validation_accuracy_pct,
            -current.params,
        ):
            best_by_model[summary.model_name] = summary
    return sorted(best_by_model.values(), key=lambda summary: (summary.params, -summary.best_validation_accuracy_pct, summary.model_name))


def _exported_report_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(root.glob("**/reports/rp2040.yaml"))


def _grow_cost_from_export(report: dict, export_root: Path) -> float | None:
    source = report.get("source") or {}
    source_path = source.get("path")
    if not isinstance(source_path, str):
        return None
    grow_root = Path(source_path)
    if not grow_root.is_absolute():
        grow_root = export_root / grow_root
    grow_report = grow_root / "reports/grow_summary.yaml"
    if not grow_report.is_file():
        return None
    grow_summary = yaml.safe_load(grow_report.read_text()) or {}
    deployed = (grow_summary.get("cost") or {}).get("deployed") or {}
    return _as_number(deployed.get("macs"))


def load_exported_model_records(root: Path, arm: str = "pointwise_b2-bn") -> list[ExportedModelRecord]:
    """Load exported PAI INT8 test results for the matched pointwise arm."""

    records: list[ExportedModelRecord] = []
    for report_path in _exported_report_paths(root):
        report = yaml.safe_load(report_path.read_text()) or {}
        source = report.get("source") or {}
        if source.get("arm") != arm:
            continue
        test_int = ((report.get("test") or {}).get("int") or {}).get("accuracy")
        params = _as_number(report.get("float_parameters"))
        width = _as_number(source.get("width"))
        accuracy = _as_number(test_int)
        if any(value is None for value in (width, accuracy, params)):
            continue
        width_int = int(width)
        match = re.search(r"c\d+(g\d+)?", str(report.get("name", "")))
        width_label = "" if not match or match.group(1) is None else match.group(1)
        seed_value = _as_number(source.get("seed", report.get("seed")))
        records.append(
            ExportedModelRecord(
                model=f"PAI C{width_int}{width_label}",
                width=width_int,
                seed=None if seed_value is None else int(seed_value),
                test_accuracy_pct=float(accuracy),
                params=float(params),
                macs=_grow_cost_from_export(report, root),
                report_path=report_path,
            )
        )
    return sorted(records, key=lambda record: (record.width, record.seed or -1))


def load_broader_test_records(
    path: Path, broader_records: Sequence[BroaderModelRecord]
) -> list[BroaderTestRecord]:
    """Load held-out test results and join them to their v3 grow runs."""

    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(evaluations, list):
        return []

    by_run_root = {
        record.report_path.parent.parent.resolve(): record
        for record in broader_records
    }
    test_records: list[BroaderTestRecord] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        checkpoint = Path(str(evaluation.get("checkpoint", "")))
        run_root = next(
            (parent for parent in checkpoint.resolve().parents if parent in by_run_root),
            None,
        )
        if run_root is None:
            continue
        source = by_run_root[run_root]
        test_accuracy = _accuracy_fraction(evaluation.get("test_accuracy"))
        params = _as_number(evaluation.get("num_params"))
        if test_accuracy is None or params is None:
            continue
        seed = _as_number(evaluation.get("seed"))
        test_records.append(
            BroaderTestRecord(
                model_name=source.model_name,
                arm=source.arm,
                width=source.width,
                seed=None if seed is None else int(seed),
                test_accuracy_pct=float(test_accuracy) * 100,
                params=float(params),
                report_path=path,
            )
        )
    return sorted(test_records, key=lambda record: (record.width, record.seed or -1))


def _display_model_name(model_name: str) -> str:
    display = model_name.removeprefix("sparknet_")
    display = display.replace("_paper", " paper")
    return re.sub(r"^c", "C", display)


def _display_variant(arm: str) -> str:
    return arm.replace("_", " ") if arm else "—"


def filter_widths(records: Iterable[ComparisonRecord], excluded_widths: Iterable[int]) -> list[ComparisonRecord]:
    """Remove excluded SparkNet widths from all downstream plot data."""

    excluded = set(excluded_widths)
    return [record for record in records if record.width not in excluded]


def filter_pareto_data(
    records: Iterable[ComparisonRecord], summaries: Iterable[ComparisonSummary]
) -> tuple[list[ComparisonRecord], list[ComparisonSummary]]:
    """Keep only the control arm for the less-cluttered Pareto figures."""

    return (
        [record for record in records if record.arm == PARETO_ARM],
        control_only_summaries(summaries),
    )


def control_only_summaries(summaries: Iterable[ComparisonSummary]) -> list[ComparisonSummary]:
    """Keep only the control arm for every generated graphic."""

    return [summary for summary in summaries if summary.arm == PARETO_ARM]


def _mean_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_records(records: Iterable[ComparisonRecord]) -> list[ComparisonSummary]:
    """Average seed runs by arm and width."""

    grouped: dict[tuple[str, int], list[ComparisonRecord]] = {}
    for record in records:
        grouped.setdefault((record.arm, record.width), []).append(record)

    summaries: list[ComparisonSummary] = []
    for (arm, width), group in sorted(grouped.items()):
        sparknet_accuracy = mean(record.sparknet_accuracy for record in group) * 100
        dendritic_accuracy = mean(record.dendritic_accuracy for record in group) * 100
        sparknet_params = mean(record.sparknet_params for record in group)
        dendritic_params = mean(record.dendritic_params for record in group)
        sparknet_macs = mean(record.sparknet_macs for record in group)
        dendritic_macs = mean(record.dendritic_macs for record in group)
        summaries.append(
            ComparisonSummary(
                arm=arm,
                width=width,
                n_seeds=len(group),
                sparknet_accuracy=sparknet_accuracy,
                dendritic_accuracy=dendritic_accuracy,
                sparknet_accuracy_std=_std(record.sparknet_accuracy * 100 for record in group),
                dendritic_accuracy_std=_std(record.dendritic_accuracy * 100 for record in group),
                sparknet_params=sparknet_params,
                dendritic_params=dendritic_params,
                sparknet_macs=sparknet_macs,
                dendritic_macs=dendritic_macs,
                accuracy_gain_pp=dendritic_accuracy - sparknet_accuracy,
                parameter_overhead_pct=(dendritic_params / sparknet_params - 1) * 100,
                mac_overhead_pct=(dendritic_macs / sparknet_macs - 1) * 100,
                dendritic_latency_ms=_mean_optional(record.dendritic_latency_ms for record in group),
                dendritic_activation_bytes=_mean_optional(record.dendritic_activation_bytes for record in group),
                seed_ids=tuple(record.seed for record in group),
            )
        )
    return summaries


def pareto_frontier(points: Iterable[tuple[float, float, object]]) -> list[tuple[float, float, object]]:
    """Return points with minimum x cost and maximum y accuracy."""

    ordered = sorted(points, key=lambda point: (point[0], -point[1]))
    frontier: list[tuple[float, float, object]] = []
    best_y = -math.inf
    for point in ordered:
        if point[1] > best_y:
            frontier.append(point)
            best_y = point[1]
    return frontier


def _metric_values(summary: ComparisonSummary, metric: str) -> tuple[float, float, str]:
    if metric == "params":
        return summary.sparknet_params, summary.dendritic_params, "deployed parameters"
    if metric == "macs":
        return summary.sparknet_macs, summary.dendritic_macs, "MACs"
    raise ValueError(f"unsupported Pareto metric: {metric}")


def _model_points(summaries: Iterable[ComparisonSummary], metric: str):
    for summary in summaries:
        sparknet_x, dendritic_x, _ = _metric_values(summary, metric)
        yield sparknet_x, summary.sparknet_accuracy, summary, "SparkNet"
        yield dendritic_x, summary.dendritic_accuracy, summary, "Dendritic"


def colored_line_series(summaries: Iterable[ComparisonSummary], metric: str) -> dict[str, list[tuple[float, float]]]:
    """Return mean model points for clean variant-colored Pareto lines."""

    ordered = sorted(summaries, key=lambda summary: _metric_values(summary, metric)[0])
    return {
        "SparkNet": [(_metric_values(summary, metric)[0], summary.sparknet_accuracy) for summary in ordered],
        "Dendritic": [(_metric_values(summary, metric)[1], summary.dendritic_accuracy) for summary in ordered],
    }


def labeled_pareto_points(
    summaries: Iterable[ComparisonSummary], metric: str
) -> list[tuple[float, float, str, ComparisonSummary, str]]:
    """Return all arm/variant points with compact C-width labels."""

    ordered = sorted(summaries, key=lambda summary: (summary.width, summary.arm))
    points: list[tuple[float, float, str, ComparisonSummary, str]] = []
    for summary in ordered:
        sparknet_x, dendritic_x, _ = _metric_values(summary, metric)
        points.append((sparknet_x, summary.sparknet_accuracy, f"C{summary.width}", summary, "SparkNet"))
        points.append((dendritic_x, summary.dendritic_accuracy, f"C{summary.width}", summary, "Dendritic"))
    return points


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def _arm_color(arm: str) -> str:
    palette = list(ARM_COLORS.values())
    if arm in ARM_COLORS:
        return ARM_COLORS[arm]
    return palette[hash(arm) % len(palette)]


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_pareto(
    records: Sequence[ComparisonRecord],
    summaries: Sequence[ComparisonSummary],
    metric: str,
    output: Path,
    dpi: int = 220,
    show_seed_points: bool = True,
) -> Path:
    """Plot the control arm against deployment cost."""

    plt, Line2D = _pyplot()
    records, summaries = filter_pareto_data(records, summaries)
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout="constrained")
    x_label = "deployed parameters" if metric == "params" else "MACs"

    if show_seed_points:
        for record in records:
            color = _arm_color(record.arm)
            if metric == "params":
                sparknet_x, dendritic_x = record.sparknet_params, record.dendritic_params
            else:
                sparknet_x, dendritic_x = record.sparknet_macs, record.dendritic_macs
            ax.plot(
                [sparknet_x, dendritic_x],
                [record.sparknet_accuracy * 100, record.dendritic_accuracy * 100],
                color=color,
                alpha=0.14,
                linewidth=0.8,
                zorder=1,
            )
            ax.scatter(sparknet_x, record.sparknet_accuracy * 100, color=color, marker="o", s=18, alpha=0.18, zorder=2)
            ax.scatter(dendritic_x, record.dendritic_accuracy * 100, color=color, marker="D", s=18, alpha=0.18, zorder=2)

    for summary in summaries:
        sparknet_x, dendritic_x, _ = _metric_values(summary, metric)
        color = _arm_color(summary.arm)
        ax.plot(
            [sparknet_x, dendritic_x],
            [summary.sparknet_accuracy, summary.dendritic_accuracy],
            color=color,
            alpha=0.45,
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            sparknet_x,
            summary.sparknet_accuracy,
            yerr=summary.sparknet_accuracy_std,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=2,
            zorder=4,
        )
        ax.errorbar(
            dendritic_x,
            summary.dendritic_accuracy,
            yerr=summary.dendritic_accuracy_std,
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=2,
            zorder=4,
        )

    point_data = [(x, y, (summary, variant)) for x, y, summary, variant in _model_points(summaries, metric)]
    ax.set_xlabel(x_label)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"SparkNet vs dendritic models: accuracy vs {x_label}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_ylim(bottom=max(0, min(point[1] for point in point_data) - 3, 0))
    _style_axis(ax)

    variant_handles = [
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="SparkNet", markersize=7),
        Line2D([0], [0], marker="D", color="#555555", linestyle="None", label="Dendritic", markersize=7),
    ]
    arm_handles = [
        Line2D([0], [0], marker="o", color=_arm_color(arm), linestyle="None", label=ARM_LABELS.get(arm, arm), markersize=6)
        for arm in sorted({summary.arm for summary in summaries})
    ]
    ax.legend(handles=variant_handles + arm_handles, loc="best", frameon=False, ncol=2)
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_colored_pareto(
    summaries: Sequence[ComparisonSummary], metric: str, output: Path, dpi: int = 220
) -> Path:
    """Plot clean control-arm comparison lines with color-coded variants."""

    plt, _ = _pyplot()
    _, summaries = filter_pareto_data([], summaries)
    if not summaries:
        raise ValueError("colored Pareto plots require at least one control-arm summary")

    series = colored_line_series(summaries, metric)
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout="constrained")
    x_label = "deployed parameters" if metric == "params" else "MACs"
    for variant, marker in (("SparkNet", "o"), ("Dendritic", "D")):
        points = series[variant]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=MODEL_COLORS[variant],
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.2,
            label=f"{variant} (control)",
            zorder=3,
        )

    point_data = [(x, y, (summary, variant)) for x, y, summary, variant in _model_points(summaries, metric)]

    ax.set_xlabel(x_label)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"Control-arm comparison: accuracy vs {x_label}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_ylim(bottom=max(0, min(point[1] for point in point_data) - 3, 0))
    _style_axis(ax)
    ax.legend(loc="best", frameon=False)
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_labeled_pareto(
    summaries: Sequence[ComparisonSummary], metric: str, output: Path, dpi: int = 220
) -> Path:
    """Plot the control arm without error bars and annotate each width."""

    plt, Line2D = _pyplot()
    summaries = control_only_summaries(summaries)
    points = labeled_pareto_points(summaries, metric)
    if not points:
        raise ValueError("labeled Pareto plots require at least one summary")

    fig, ax = plt.subplots(figsize=(11.0, 7.0), layout="constrained")
    x_label = "deployed parameters" if metric == "params" else "MACs"
    arms = sorted({summary.arm for summary in summaries})
    for arm in arms:
        rows = sorted((summary for summary in summaries if summary.arm == arm), key=lambda summary: summary.width)
        color = _arm_color(arm)
        sparknet_x = [_metric_values(summary, metric)[0] for summary in rows]
        dendritic_x = [_metric_values(summary, metric)[1] for summary in rows]
        ax.plot(
            sparknet_x,
            [summary.sparknet_accuracy for summary in rows],
            color=color,
            linestyle=":",
            marker="o",
            linewidth=1.8,
            markersize=6,
            label=ARM_LABELS.get(arm, arm),
            zorder=2,
        )
        ax.plot(
            dendritic_x,
            [summary.dendritic_accuracy for summary in rows],
            color=color,
            linestyle="-",
            marker="D",
            linewidth=2.0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )

    point_data = [(x, y, (summary, variant)) for x, y, _, summary, variant in points]
    for x, y, label, _, variant in points:
        ax.annotate(
            label,
            (x, y),
            xytext=(0, -12 if variant == "SparkNet" else -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
            zorder=5,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"Control-arm comparison with widths: accuracy vs {x_label} (C2/C4 excluded)")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_ylim(bottom=max(0, min(point[1] for point in point_data) - 4, 0))
    _style_axis(ax)
    variant_handles = [
        Line2D([0], [0], color="#555555", linestyle=":", marker="o", label="SparkNet", markersize=6),
        Line2D([0], [0], color="#555555", linestyle="-", marker="D", label="Dendritic", markersize=6),
    ]
    arm_handles = [
        Line2D([0], [0], color=_arm_color(arm), linestyle="-", label=ARM_LABELS.get(arm, arm))
        for arm in arms
    ]
    ax.legend(handles=variant_handles + arm_handles, loc="best", frameon=False, ncol=2)
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def _compact_model_label(model_name: str) -> str:
    """Make a short graph label while preserving gate-width variants."""

    return _display_model_name(model_name).removesuffix(" paper").replace(" ", "")


def plot_best_vs_faithful(
    best_summaries: Sequence[BroaderModelSummary],
    faithful_summaries: Sequence[FaithfulSparkNetSummary],
    metric: str,
    output: Path,
    dpi: int = 220,
) -> Path:
    """Compare best trained v3 models with their faithful SparkNet bases.

    The two point series are overlaid with separate Pareto frontiers: the old
    faithful SparkNet frontier and the new best-trained-model frontier.
    """

    plt, Line2D = _pyplot()
    if not best_summaries and not faithful_summaries:
        raise ValueError("best-vs-faithful plots require at least one model")

    x_attribute = "params" if metric == "params" else "macs"
    x_label = "parameters" if metric == "params" else "MACs"
    best_summaries = filter_best_models_by_faithful_dominance(best_summaries, faithful_summaries, metric)
    best_summaries = select_new_frontier_summaries(best_summaries)
    series = (
        (
            "Best trained dendritic models",
            best_summaries,
            "best_validation_accuracy_pct",
            "D",
            "#F58518",
        ),
        (
            "Faithful Sparknet (previous SOTA)",
            faithful_summaries,
            "best_validation_accuracy_pct",
            "o",
            "#4C78A8",
        ),
    )
    fig, ax = plt.subplots(figsize=(10.4, 6.8), layout="constrained")
    all_y: list[float] = []
    for legend_label, summaries, accuracy_attribute, marker, color in series:
        ordered = sorted(summaries, key=lambda item: (getattr(item, x_attribute), item.width))
        for summary in ordered:
            x = float(getattr(summary, x_attribute))
            y = float(getattr(summary, accuracy_attribute))
            all_y.append(y)
            ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
                s=72,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                zorder=3,
            )
            ax.annotate(
                _compact_model_label(summary.model_name),
                (x, y),
                xytext=(-8, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8,
                color=color,
                zorder=4,
            )

    frontier_series = (
        ("Old faithful SparkNet frontier", faithful_summaries, "#4C78A8", "--"),
        (
            "New best trained frontier",
            select_new_frontier_summaries(best_summaries),
            "#F58518",
            "-",
        ),
    )
    for frontier_label, summaries, color, linestyle in frontier_series:
        frontier = pareto_frontier(
            (
                float(getattr(summary, x_attribute)),
                float(getattr(summary, "best_validation_accuracy_pct")),
                summary,
            )
            for summary in summaries
        )
        if frontier:
            ax.plot(
                [point[0] for point in frontier],
                [point[1] for point in frontier],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=frontier_label,
                zorder=2,
            )

    parameter_efficient = most_parameter_efficient_summary(best_summaries)
    if parameter_efficient is not None:
        ax.scatter(
            float(getattr(parameter_efficient, x_attribute)),
            float(parameter_efficient.best_validation_accuracy_pct),
            marker="*",
            s=190,
            facecolor="#FFD166",
            edgecolor="#8C5A00",
            linewidth=0.9,
            zorder=5,
        )

    practical = next(
        (summary for summary in best_summaries if summary.model_name == PRACTICAL_MODEL_NAME),
        None,
    )
    if practical is not None:
        ax.scatter(
            float(getattr(practical, x_attribute)),
            float(practical.best_validation_accuracy_pct),
            marker="o",
            s=190,
            facecolors="none",
            edgecolors="#333333",
            linewidth=1.8,
            zorder=5,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("best validation accuracy (%)")
    ax.set_title(f"Best trained models vs faithful SparkNet: accuracy vs {x_label}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_ylim(bottom=max(0, min(all_y) - 3))
    _style_axis(ax)
    ax.text(
        0.99,
        0.015,
        MODEL_LABEL_SYNTAX_NOTE,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
        zorder=6,
    )
    legend_handles = [
            Line2D(
                [0],
                [0],
                marker="D",
                color="#F58518",
                linestyle="None",
                label="Best trained dendritic models",
                markersize=7,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#4C78A8",
                linestyle="None",
                label="Faithful Sparknet (previous SOTA)",
                markersize=7,
            ),
            Line2D([0], [0], color="#4C78A8", linestyle="--", label="Old faithful SparkNet frontier"),
            Line2D([0], [0], color="#F58518", linestyle="-", label="New best trained frontier"),
    ]
    if parameter_efficient is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="#8C5A00",
                markerfacecolor="#FFD166",
                linestyle="None",
                label="Most parameter-efficient",
                markersize=11,
            )
        )
    if practical is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#333333",
                markerfacecolor="none",
                linestyle="None",
                label=f"Practical model ({_compact_model_label(PRACTICAL_MODEL_NAME)})",
                markersize=9,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=False,
    )
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_gain_vs_overhead(summaries: Sequence[ComparisonSummary], output: Path, dpi: int = 220) -> Path:
    """Show accuracy gain against the parameter overhead of adding dendrites."""

    plt, _ = _pyplot()
    summaries = control_only_summaries(summaries)
    fig, ax = plt.subplots(figsize=(8.8, 6.0), layout="constrained")
    for summary in summaries:
        ax.scatter(
            summary.parameter_overhead_pct,
            summary.accuracy_gain_pp,
            color=_arm_color(summary.arm),
            s=36 + 12 * summary.width,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.annotate(
            f"C{summary.width}",
            (summary.parameter_overhead_pct, summary.accuracy_gain_pp),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("dendritic parameter overhead (%)")
    ax.set_ylabel("dendritic validation accuracy gain (percentage points)")
    ax.set_title("Accuracy gain versus dendritic parameter overhead")
    _style_axis(ax)
    handles = [
        plt.Line2D([0], [0], marker="o", color=_arm_color(arm), linestyle="None", label=ARM_LABELS.get(arm, arm), markersize=6)
        for arm in sorted({summary.arm for summary in summaries})
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_overhead_by_width(summaries: Sequence[ComparisonSummary], output: Path, dpi: int = 220) -> Path:
    """Plot parameter and MAC overhead as two small multiples."""

    plt, _ = _pyplot()
    summaries = control_only_summaries(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, layout="constrained")
    for ax, attribute, title, ylabel in (
        (axes[0], "parameter_overhead_pct", "Parameter overhead", "parameters added (%)"),
        (axes[1], "mac_overhead_pct", "MAC overhead", "MACs added (%)"),
    ):
        for arm in sorted({summary.arm for summary in summaries}):
            rows = sorted((summary for summary in summaries if summary.arm == arm), key=lambda summary: summary.width)
            ax.plot(
                [summary.width for summary in rows],
                [getattr(summary, attribute) for summary in rows],
                marker="o",
                color=_arm_color(arm),
                linewidth=1.8,
                label=ARM_LABELS.get(arm, arm),
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("SparkNet width")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Dendritic cost overhead by SparkNet width")
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_dashboard(summaries: Sequence[ComparisonSummary], output: Path, dpi: int = 220) -> Path:
    """Create a compact overview of accuracy, gain, overhead, and latency."""

    plt, _ = _pyplot()
    summaries = control_only_summaries(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
    axes = axes.ravel()
    arms = sorted({summary.arm for summary in summaries})

    for arm in arms:
        rows = sorted((summary for summary in summaries if summary.arm == arm), key=lambda summary: summary.width)
        color = _arm_color(arm)
        widths = [summary.width for summary in rows]
        axes[0].plot(widths, [summary.sparknet_accuracy for summary in rows], "o--", color=color, alpha=0.55, linewidth=1, label=f"{ARM_LABELS.get(arm, arm)} SparkNet")
        axes[0].plot(widths, [summary.dendritic_accuracy for summary in rows], "D-", color=color, linewidth=1.8, label=f"{ARM_LABELS.get(arm, arm)} dendritic")
        axes[1].plot(widths, [summary.accuracy_gain_pp for summary in rows], "o-", color=color, linewidth=1.8, label=ARM_LABELS.get(arm, arm))
        axes[2].plot(widths, [summary.parameter_overhead_pct for summary in rows], "o-", color=color, linewidth=1.8, label=ARM_LABELS.get(arm, arm))
        latency_rows = [row for row in rows if row.dendritic_latency_ms is not None]
        if latency_rows:
            axes[3].plot([row.width for row in latency_rows], [row.dendritic_latency_ms for row in latency_rows], "o-", color=color, linewidth=1.8, label=ARM_LABELS.get(arm, arm))

    axes[0].set_title("Validation accuracy")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].set_title("Dendritic accuracy gain")
    axes[1].set_ylabel("gain (percentage points)")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set_title("Parameter overhead")
    axes[2].set_ylabel("overhead (%)")
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[3].set_title("Dendritic runtime proxy")
    axes[3].set_ylabel("mean latency (ms)")
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("SparkNet width")
        _style_axis(ax)
    fig.suptitle("SparkNet and dendritic study metrics", fontsize=15)
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def write_summary_csv(summaries: Sequence[ComparisonSummary], output: Path) -> Path:
    fields = [
        "arm",
        "width",
        "n_seeds",
        "sparknet_accuracy_pct",
        "dendritic_accuracy_pct",
        "sparknet_accuracy_std_pct",
        "dendritic_accuracy_std_pct",
        "sparknet_params",
        "dendritic_params",
        "sparknet_macs",
        "dendritic_macs",
        "accuracy_gain_pp",
        "parameter_overhead_pct",
        "mac_overhead_pct",
        "dendritic_latency_ms",
        "dendritic_activation_peak_bytes",
        "seed_ids",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "arm": summary.arm,
                    "width": summary.width,
                    "n_seeds": summary.n_seeds,
                    "sparknet_accuracy_pct": f"{summary.sparknet_accuracy:.6f}",
                    "dendritic_accuracy_pct": f"{summary.dendritic_accuracy:.6f}",
                    "sparknet_accuracy_std_pct": f"{summary.sparknet_accuracy_std:.6f}",
                    "dendritic_accuracy_std_pct": f"{summary.dendritic_accuracy_std:.6f}",
                    "sparknet_params": f"{summary.sparknet_params:.3f}",
                    "dendritic_params": f"{summary.dendritic_params:.3f}",
                    "sparknet_macs": f"{summary.sparknet_macs:.3f}",
                    "dendritic_macs": f"{summary.dendritic_macs:.3f}",
                    "accuracy_gain_pp": f"{summary.accuracy_gain_pp:.6f}",
                    "parameter_overhead_pct": f"{summary.parameter_overhead_pct:.6f}",
                    "mac_overhead_pct": f"{summary.mac_overhead_pct:.6f}",
                    "dendritic_latency_ms": "" if summary.dendritic_latency_ms is None else f"{summary.dendritic_latency_ms:.6f}",
                    "dendritic_activation_peak_bytes": "" if summary.dendritic_activation_bytes is None else f"{summary.dendritic_activation_bytes:.3f}",
                    "seed_ids": ",".join("unknown" if seed is None else str(seed) for seed in summary.seed_ids),
                }
            )
    return output


def write_model_inventory_csv(
    summaries: Sequence[BroaderModelSummary] | Sequence[FaithfulSparkNetSummary],
    output: Path,
    category: str,
) -> Path:
    """Write the exact rows used by the best-model comparison figures."""

    fields = [
        "category",
        "model_name",
        "display_name",
        "width",
        "n_seeds",
        "mean_validation_accuracy_pct",
        "best_validation_accuracy_pct",
        "validation_accuracy_std_pct",
        "params",
        "macs",
        "seed_ids",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "category": category,
                    "model_name": summary.model_name,
                    "display_name": _compact_model_label(summary.model_name),
                    "width": summary.width,
                    "n_seeds": summary.n_seeds,
                    "mean_validation_accuracy_pct": f"{summary.mean_validation_accuracy_pct:.6f}",
                    "best_validation_accuracy_pct": f"{summary.best_validation_accuracy_pct:.6f}",
                    "validation_accuracy_std_pct": f"{summary.validation_accuracy_std_pct:.6f}",
                    "params": f"{summary.params:.3f}",
                    "macs": f"{summary.macs:.3f}",
                    "seed_ids": ",".join("unknown" if seed is None else str(seed) for seed in summary.seed_ids),
                }
            )
    return output


def _markdown_accuracy(value: float, standard_deviation: float) -> str:
    return f"{value:.2f}% ± {standard_deviation:.2f}"


def _markdown_signed(value: float, suffix: str) -> str:
    return f"{value:+.2f} {suffix}"


def _markdown_optional(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.3f}{suffix}"


def write_markdown_report(
    summaries: Sequence[ComparisonSummary],
    output: Path,
    broader_summaries: Sequence[BroaderModelSummary] | None = None,
    exported_records: Sequence[ExportedModelRecord] | None = None,
    broader_test_records: Sequence[BroaderTestRecord] | None = None,
    faithful_summaries: Sequence[FaithfulSparkNetSummary] | None = None,
) -> Path:
    """Write a readable Markdown report from the completed model summaries."""

    ordered = sorted(summaries, key=lambda summary: (summary.arm, summary.width))
    broader_summaries = list(broader_summaries or [])
    exported_records = list(exported_records or [])
    broader_test_records = list(broader_test_records or [])
    faithful_summaries = list(faithful_summaries or [])
    control_rows = [summary for summary in ordered if summary.arm == PARETO_ARM]
    arms = sorted({summary.arm for summary in ordered})
    widths = sorted({summary.width for summary in ordered})
    total_seed_pairs = sum(summary.n_seeds for summary in ordered)
    seed_label = "seed pair" if total_seed_pairs == 1 else "seed pairs"

    best_accuracy = max(ordered, key=lambda summary: summary.dendritic_accuracy)
    best_gain = max(ordered, key=lambda summary: summary.accuracy_gain_pp)
    noncontrol = [summary for summary in ordered if summary.arm != PARETO_ARM]
    mean_overhead_by_arm = {
        arm: mean(summary.parameter_overhead_pct for summary in ordered if summary.arm == arm)
        for arm in arms
    }
    lowest_overhead_arm = min(noncontrol, key=lambda summary: mean_overhead_by_arm[summary.arm]) if noncontrol else None

    lines = [
        "# SparkNet vs Dendritic Model Statistics",
        "",
        f"> Aggregated from {total_seed_pairs} matched {seed_label} across {len(ordered)} completed arm/width configurations.",
        "> The matched v2, broader v3, and external reference sections label validation and held-out test accuracy separately; `±` values are standard deviation across seeds.",
        "",
        "## Coverage",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Completed matched seed pairs | {total_seed_pairs} |",
        f"| Arm/width configurations | {len(ordered)} |",
        f"| Arms | {', '.join(ARM_LABELS.get(arm, arm) for arm in arms)} |",
        f"| Widths | {', '.join(f'C{width}' for width in widths)} |",
        "| Excluded widths | C2, C4 |",
        "",
        "## Highlights",
        "",
        f"- Highest dendritic accuracy: **{best_accuracy.dendritic_accuracy:.2f}%** ({ARM_LABELS.get(best_accuracy.arm, best_accuracy.arm)}, C{best_accuracy.width}).",
        f"- Largest dendritic accuracy gain: **{_markdown_signed(best_gain.accuracy_gain_pp, 'pp')}** ({ARM_LABELS.get(best_gain.arm, best_gain.arm)}, C{best_gain.width}).",
    ]
    if lowest_overhead_arm is not None:
        arm = lowest_overhead_arm.arm
        lines.append(
            f"- Lowest mean parameter overhead among dendritic placements: **{mean_overhead_by_arm[arm]:.2f}%** ({ARM_LABELS.get(arm, arm)})."
        )

    lines.extend(
        [
            "",
            "## Control-arm results",
            "",
            "| Width | Seeds | SparkNet accuracy | Dendritic accuracy | Gain | SparkNet params | Dendritic params | SparkNet MACs | Dendritic MACs | Latency (ms) | Activation peak (KB) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in control_rows:
        activation_kb = None if summary.dendritic_activation_bytes is None else summary.dendritic_activation_bytes / 1024
        lines.append(
            f"| C{summary.width} | {summary.n_seeds} | {summary.sparknet_accuracy:.2f}% | "
            f"{summary.dendritic_accuracy:.2f}% | {_markdown_signed(summary.accuracy_gain_pp, 'pp')} | "
            f"{summary.sparknet_params:,.0f} | {summary.dendritic_params:,.0f} | "
            f"{summary.sparknet_macs:,.0f} | {summary.dendritic_macs:,.0f} | "
            f"{_markdown_optional(summary.dendritic_latency_ms)} | "
            f"{'—' if activation_kb is None else f'{activation_kb:.1f}'} |"
        )

    lines.extend(
        [
            "",
            "## Placement-arm summary",
            "",
            "| Placement | Configurations | Seed pairs | Best dendritic accuracy | Best width | Mean gain | Mean parameter overhead | Mean MAC overhead | Mean latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        rows = [summary for summary in ordered if summary.arm == arm]
        best = max(rows, key=lambda summary: summary.dendritic_accuracy)
        latency_values = [summary.dendritic_latency_ms for summary in rows if summary.dendritic_latency_ms is not None]
        mean_latency = None if not latency_values else mean(latency_values)
        lines.append(
            f"| {ARM_LABELS.get(arm, arm)} | {len(rows)} | {sum(summary.n_seeds for summary in rows)} | "
            f"{best.dendritic_accuracy:.2f}% | C{best.width} | "
            f"{_markdown_signed(mean(summary.accuracy_gain_pp for summary in rows), 'pp')} | "
            f"{mean(summary.parameter_overhead_pct for summary in rows):.2f}% | "
            f"{mean(summary.mac_overhead_pct for summary in rows):.2f}% | "
            f"{_markdown_optional(mean_latency)} |"
        )

    lines.extend(
        [
            "",
            "## Complete results",
            "",
            "| Placement | Width | Seeds | SparkNet accuracy | Dendritic accuracy | Gain | Parameter overhead | MAC overhead | Latency (ms) | Activation peak (KB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in ordered:
        activation_kb = None if summary.dendritic_activation_bytes is None else summary.dendritic_activation_bytes / 1024
        lines.append(
            f"| {ARM_LABELS.get(summary.arm, summary.arm)} | C{summary.width} | {summary.n_seeds} | "
            f"{_markdown_accuracy(summary.sparknet_accuracy, summary.sparknet_accuracy_std)} | "
            f"{_markdown_accuracy(summary.dendritic_accuracy, summary.dendritic_accuracy_std)} | "
            f"{_markdown_signed(summary.accuracy_gain_pp, 'pp')} | {summary.parameter_overhead_pct:.2f}% | "
            f"{summary.mac_overhead_pct:.2f}% | {_markdown_optional(summary.dendritic_latency_ms)} | "
            f"{'—' if activation_kb is None else f'{activation_kb:.1f}'} |"
        )

    if broader_summaries:
        test_by_key: dict[tuple[str, str, int], list[BroaderTestRecord]] = {}
        for test_record in broader_test_records:
            test_by_key.setdefault(
                (test_record.model_name, test_record.arm, test_record.width), []
            ).append(test_record)
        lines.extend(
            [
                "",
                "## Broader trained-model scan (v3 validation)",
                "",
                "> One best completed v3 training variant is shown for each model name. These are validation results (`best_val_acc_overall`), not test results; the mean and `±` value summarize the completed seeds for that selected variant.",
                "",
                "| Model | Width | Best trained variant | Seeds | Mean best validation | Best validation | Held-out test | Params | MACs | Seed IDs |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for broader in sorted(broader_summaries, key=lambda item: (item.params, -item.best_validation_accuracy_pct, item.model_name)):
            seed_ids = ", ".join("?" if seed is None else str(seed) for seed in broader.seed_ids)
            test_rows = test_by_key.get((broader.model_name, broader.arm, broader.width), [])
            test_accuracy = "—"
            if test_rows:
                test_values = [row.test_accuracy_pct for row in test_rows]
                test_accuracy = f"{mean(test_values):.2f}%"
                if len(test_values) > 1:
                    test_accuracy += f" ± {_std(test_values):.2f}"
            lines.append(
                f"| {_display_model_name(broader.model_name)} | C{broader.width} | {_display_variant(broader.arm)} | "
                f"{broader.n_seeds} | {_markdown_accuracy(broader.mean_validation_accuracy_pct, broader.validation_accuracy_std_pct)} | "
                f"{broader.best_validation_accuracy_pct:.2f}% | {test_accuracy} | {broader.params:,.0f} | {broader.macs:,.0f} | {seed_ids} |"
            )

    if broader_summaries and faithful_summaries:
        lines.extend(
            [
                "",
                "## Best trained models vs faithful SparkNet bases",
                "",
                "> The new comparison figures use the best completed v3 model for each model name and the pre-dendrite base from exact `sparknet_cN_paper` runs. Both columns are validation accuracy; each graph overlays the old faithful SparkNet frontier and the new best-trained frontier through C6g16, C9g8, C10g8, C10g16, C12, C16g16, and C18g16. The star marks the highest validation-accuracy-per-parameter model, and the ring marks the practical-model default (C10g8). Each graph omits best-trained points that have lower validation accuracy and no lower plotted cost than a faithful SparkNet.",
                "",
                "| Category | Model | Accuracy | Params | MACs | Seeds |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for category, rows in (("Best trained", broader_summaries), ("Faithful SparkNet", faithful_summaries)):
            for row in sorted(rows, key=lambda item: (item.params, -item.best_validation_accuracy_pct, item.model_name)):
                seed_ids = ", ".join("?" if seed is None else str(seed) for seed in row.seed_ids)
                lines.append(
                    f"| {category} | {_compact_model_label(row.model_name)} | {row.best_validation_accuracy_pct:.2f}% | "
                    f"{row.params:,.0f} | {row.macs:,.0f} | {seed_ids} |"
                )

    lines.extend(
        [
            "",
            "## Published SC2 and exported PAI references",
            "",
            "> Published rows are SC2 test results from the paper. PAI rows are exported INT8 test results from the RP2040 reports; the parenthetical values preserve the individual seed results.",
            "",
            "| Model | Accuracy | Params | MACs | Seeds | Comparison |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for reference in PUBLISHED_SC2_REFERENCES:
        lines.append(
            f"| {reference.model} | {reference.accuracy_pct:.3f}% SC2 test | {reference.params:,.0f} | "
            f"{reference.macs:,.0f} | Published | Reference |"
        )

    exported_by_model: dict[str, list[ExportedModelRecord]] = {}
    for exported in exported_records:
        exported_by_model.setdefault(exported.model, []).append(exported)
    for model, rows in sorted(exported_by_model.items(), key=lambda item: (item[1][0].width, item[0])):
        accuracies = [row.test_accuracy_pct for row in rows]
        mean_accuracy = mean(accuracies)
        accuracy_text = f"{mean_accuracy:.3f}% INT8 test"
        if len(accuracies) > 1:
            accuracy_text += f" ± {_std(accuracies):.3f} ({', '.join(f'{value:.3f}' for value in accuracies)})"
        params = mean(row.params for row in rows)
        mac_values = [row.macs for row in rows if row.macs is not None]
        macs = "—" if not mac_values else f"{mean(mac_values):,.0f}"
        seed_ids = ", ".join("?" if row.seed is None else str(row.seed) for row in rows)
        lower_references = [reference for reference in PUBLISHED_SC2_REFERENCES if reference.width < rows[0].width]
        if lower_references:
            nearest = max(lower_references, key=lambda reference: reference.width)
            comparison = f"{mean_accuracy - nearest.accuracy_pct:+.3f} pp vs {nearest.model}"
        else:
            comparison = "No lower published point"
        lines.append(
            f"| {model} | {accuracy_text} | {params:,.0f} | {macs} | {seed_ids} | {comparison} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Control` is the no-dendrite baseline; the other placements indicate where the dendritic module was attached.",
            "- Parameter and MAC overhead are relative to the matched SparkNet baseline at the same width.",
            "- The source CSV is available as [`comparison_summary.csv`](comparison_summary.csv).",
            "- The best-model comparison data are available as [`best_models_summary.csv`](best_models_summary.csv) and [`faithful_sparknet_summary.csv`](faithful_sparknet_summary.csv).",
            "- The broader v3 scan intentionally includes C2/C4 and non-paper widths such as C9g8, C14, and C17g8; the C2/C4 exclusion applies to the graphs, not this inventory table.",
            "- Source records: `outputs/sparknet-grow-dendrites-v3/**/reports/grow_summary.yaml` and `outputs/rp2040/**/reports/rp2040.yaml`.",
            "- Selected v3 held-out results come from `report_test_accuracy.py` with a fixed evaluation seed; RP2040 INT8 results are reported separately and are not mixed into the v3 validation column.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return output


def _remove_legacy_outputs(output_dir: Path) -> None:
    """Remove graph filenames superseded by the standardized output names."""

    for filename in LEGACY_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_ROOT, help="study directory or one report YAML")
    parser.add_argument("--broader-input", type=Path, default=DEFAULT_BROADER_INPUT_ROOT, help="completed v3 grow-dendrites directory used by the Markdown inventory")
    parser.add_argument("--paper-replication-input", type=Path, default=DEFAULT_PAPER_REPLICATION_ROOT, help="dedicated paper-replication directory used for faithful SparkNet bases")
    parser.add_argument("--export-input", type=Path, default=DEFAULT_EXPORT_ROOT, help="RP2040 export directory used by the Markdown test references")
    parser.add_argument("--broader-test-input", type=Path, default=DEFAULT_BROADER_TEST_INPUT, help="JSON produced by report_test_accuracy.py for selected v3 checkpoints")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", nargs="+", help="only plot these arm directory names; default: all arms")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-seed-points", action="store_true", help="hide faint individual seed points")
    parser.add_argument("--include-c2", action="store_true", help="include C2, which is excluded by default")
    parser.add_argument("--include-c4", action="store_true", help="include C4, which is excluded by default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input, args.arms)
    excluded_widths = set(DEFAULT_EXCLUDED_WIDTHS)
    if args.include_c2:
        excluded_widths.discard(2)
    if args.include_c4:
        excluded_widths.discard(4)
    records = filter_widths(records, excluded_widths)
    if not records:
        raise SystemExit(f"no completed paired reports found under {args.input}")
    summaries = aggregate_records(records)
    broader_records = load_broader_model_records(args.broader_input)
    broader_summaries = best_broader_summary_per_model(summarize_broader_model_records(broader_records))
    best_model_summaries = [summary for summary in broader_summaries if summary.width not in excluded_widths]
    faithful_records = load_faithful_sparknet_records(args.broader_input, args.paper_replication_input)
    faithful_summaries = best_faithful_sparknet_summary(summarize_faithful_sparknet_records(faithful_records))
    faithful_graph_summaries = [summary for summary in faithful_summaries if summary.width not in excluded_widths]
    exported_records = load_exported_model_records(args.export_input)
    broader_test_records = load_broader_test_records(args.broader_test_input, broader_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_outputs(args.output_dir)
    outputs = [
        write_summary_csv(summaries, args.output_dir / OUTPUT_FILENAMES["summary"]),
        write_markdown_report(
            summaries,
            args.output_dir / OUTPUT_FILENAMES["report"],
            broader_summaries=broader_summaries,
            exported_records=exported_records,
            broader_test_records=broader_test_records,
            faithful_summaries=faithful_graph_summaries,
        ),
        plot_pareto(records, summaries, "params", args.output_dir / OUTPUT_FILENAMES["accuracy_params_errorbars"], args.dpi, not args.no_seed_points),
        plot_pareto(records, summaries, "macs", args.output_dir / OUTPUT_FILENAMES["accuracy_macs_errorbars"], args.dpi, not args.no_seed_points),
        plot_colored_pareto(summaries, "params", args.output_dir / OUTPUT_FILENAMES["accuracy_params_lines"], args.dpi),
        plot_colored_pareto(summaries, "macs", args.output_dir / OUTPUT_FILENAMES["accuracy_macs_lines"], args.dpi),
        plot_labeled_pareto(summaries, "params", args.output_dir / OUTPUT_FILENAMES["accuracy_params_annotated"], args.dpi),
        plot_labeled_pareto(summaries, "macs", args.output_dir / OUTPUT_FILENAMES["accuracy_macs_annotated"], args.dpi),
        plot_gain_vs_overhead(summaries, args.output_dir / OUTPUT_FILENAMES["accuracy_gain"], args.dpi),
        plot_overhead_by_width(summaries, args.output_dir / OUTPUT_FILENAMES["cost_overhead"], args.dpi),
        plot_dashboard(summaries, args.output_dir / OUTPUT_FILENAMES["dashboard"], args.dpi),
    ]
    if best_model_summaries or faithful_graph_summaries:
        outputs.extend(
            [
                write_model_inventory_csv(
                    best_model_summaries,
                    args.output_dir / OUTPUT_FILENAMES["best_models_summary"],
                    "best_trained_model",
                ),
                write_model_inventory_csv(
                    faithful_graph_summaries,
                    args.output_dir / OUTPUT_FILENAMES["faithful_sparknet_summary"],
                    "faithful_sparknet",
                ),
                plot_best_vs_faithful(
                    best_model_summaries,
                    faithful_graph_summaries,
                    "params",
                    args.output_dir / OUTPUT_FILENAMES["best_vs_faithful_params"],
                    args.dpi,
                ),
                plot_best_vs_faithful(
                    best_model_summaries,
                    faithful_graph_summaries,
                    "macs",
                    args.output_dir / OUTPUT_FILENAMES["best_vs_faithful_macs"],
                    args.dpi,
                ),
            ]
        )
    print(
        f"loaded {len(records)} seed pairs; aggregated {len(summaries)} arm/width summaries; "
        f"added {len(broader_records)} broader v3 runs, {len(broader_test_records)} v3 test results, "
        f"{len(faithful_records)} faithful SparkNet bases, and {len(exported_records)} exported PAI results"
    )
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
