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
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import yaml


# Edit these defaults for a different study or output location.
DEFAULT_INPUT_ROOT = Path("outputs/sparknet-dendritic-study-v2")
DEFAULT_OUTPUT_DIR = Path("outputs/plots/sparknet-dendritic-comparison")

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
    """Plot accuracy against a deployment cost and highlight its frontier."""

    plt, Line2D = _pyplot()
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
    frontier = pareto_frontier(point_data)
    if frontier:
        ax.plot(
            [point[0] for point in frontier],
            [point[1] for point in frontier],
            color="#222222",
            linestyle="--",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Pareto frontier",
            zorder=5,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"SparkNet vs dendritic models: accuracy vs {x_label}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_ylim(bottom=max(0, min(point[1] for point in point_data) - 3, 0))
    _style_axis(ax)

    variant_handles = [
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="SparkNet", markersize=7),
        Line2D([0], [0], marker="D", color="#555555", linestyle="None", label="Dendritic", markersize=7),
        Line2D([0], [0], color="#222222", linestyle="--", label="Pareto frontier"),
    ]
    arm_handles = [
        Line2D([0], [0], marker="o", color=_arm_color(arm), linestyle="None", label=ARM_LABELS.get(arm, arm), markersize=6)
        for arm in sorted({summary.arm for summary in summaries})
    ]
    ax.legend(handles=variant_handles + arm_handles, loc="best", frameon=False, ncol=2)
    result = _save(fig, output, dpi)
    plt.close(fig)
    return result


def plot_gain_vs_overhead(summaries: Sequence[ComparisonSummary], output: Path, dpi: int = 220) -> Path:
    """Show accuracy gain against the parameter overhead of adding dendrites."""

    plt, _ = _pyplot()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_ROOT, help="study directory or one report YAML")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", nargs="+", help="only plot these arm directory names; default: all arms")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-seed-points", action="store_true", help="hide faint individual seed points")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input, args.arms)
    if not records:
        raise SystemExit(f"no completed paired reports found under {args.input}")
    summaries = aggregate_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_summary_csv(summaries, args.output_dir / "comparison_summary.csv"),
        plot_pareto(records, summaries, "params", args.output_dir / "pareto_accuracy_vs_params.png", args.dpi, not args.no_seed_points),
        plot_pareto(records, summaries, "macs", args.output_dir / "pareto_accuracy_vs_macs.png", args.dpi, not args.no_seed_points),
        plot_gain_vs_overhead(summaries, args.output_dir / "accuracy_gain_vs_parameter_overhead.png", args.dpi),
        plot_overhead_by_width(summaries, args.output_dir / "cost_overhead_by_width.png", args.dpi),
        plot_dashboard(summaries, args.output_dir / "metrics_dashboard.png", args.dpi),
    ]
    print(f"loaded {len(records)} seed pairs; aggregated {len(summaries)} arm/width summaries")
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
