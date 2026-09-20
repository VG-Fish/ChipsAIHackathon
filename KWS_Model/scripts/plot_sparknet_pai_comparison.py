#!/usr/bin/env python3
"""Plot SparkNet scratch models against their PerforatedAI counterparts.

The input is the JSON emitted by ``report_sparknet_grow.py``.  Only the
architecture-matched ``c<W>g16`` PAI cells are plotted; the scratch series is
the report's paired scratch control.  Accuracy is the mean best validation
accuracy, and cost is deployed parameter count (use ``--cost macs`` for MACs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import yaml


def collect(report: dict, arm: str = "pointwise_b2-bn", metric: str = "best", cost: str = "params"):
    """Return ``(width, scratch, pai, cost)`` rows for the g16 real arm."""
    rows = []
    for cell in report["cells"]:
        if cell.get("arm") != arm or cell.get("sham") or not str(cell.get("model_name", "")).endswith("g16"):
            continue
        if metric not in cell.get("grow_mean", {}):
            continue
        scratch = cell.get("scratch_mean_paired", {}).get(metric)
        deployed = (cell.get("deployed") or {}).get(cost)
        if scratch is None or deployed is None:
            continue
        rows.append((int(cell["width"]), float(scratch), float(cell["grow_mean"][metric]), int(deployed)))
    return sorted(rows)


def collect_exports(root: Path, names: list[str]):
    """Read exported RP2040 reports as ``(label, test_pct, params, flash)``."""
    rows = []
    for name in names:
        run = root / name
        report = yaml.safe_load((run / "reports/rp2040.yaml").read_text())
        accuracy = report["test"]["int"]["accuracy"]
        params = report["float_parameters"]
        firmware = run / "models/exported/rp2040/firmware/kws_rp2040.uf2"
        stem = name.split("-seed", 1)[0]
        width = stem.split("g", 1)[0].lstrip("c")
        label = ("PAI " if "g16" in name else "SparkNet ") + "C" + width
        rows.append((label, float(accuracy), int(params), os.stat(firmware).st_size))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="pointwise_b2-bn")
    parser.add_argument("--metric", choices=("best", "final", "last5", "last10"), default="best")
    parser.add_argument("--cost", choices=("params", "macs"), default="params")
    parser.add_argument("--export-root", type=Path, help="RP2040 export root; enables hardware comparison mode")
    parser.add_argument("--export-names", nargs="+", default=["c16-seed0-scratch", "c6g16-b2-seed0", "c10g16-b2-seed0", "c18g16-b2-seed0"])
    args = parser.parse_args()
    if args.export_root:
        rows = collect_exports(args.export_root, args.export_names)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
        for label, accuracy, params, flash in rows:
            color = "#f58518" if label.startswith("PAI") else "#4c78a8"
            ax.scatter(flash, accuracy, s=70, color=color, label=label)
            ax.annotate(label, (flash, accuracy), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("RP2040 firmware size (UF2 bytes)")
        ax.set_ylabel("INT8 test accuracy (%)")
        ax.set_title("Exported SparkNet versus PerforatedAI")
        ax.grid(True, alpha=0.25)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"wrote {args.output} ({len(rows)} exports)")
        return
    report = json.loads(args.report.read_text())
    rows = collect(report, args.arm, args.metric, args.cost)
    if not rows:
        raise SystemExit("no architecture-matched g16 rows found")

    import matplotlib.pyplot as plt

    widths, scratch, pai, costs = zip(*rows)
    scratch = tuple(value * 100 for value in scratch)
    pai = tuple(value * 100 for value in pai)
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.plot(costs, scratch, "o-", label="SparkNet scratch", color="#4c78a8")
    ax.plot(costs, pai, "D-", label="SparkNet + PAI", color="#f58518")
    for x, y, width in zip(costs, pai, widths):
        ax.annotate(f"C{width}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Deployed parameters" if args.cost == "params" else "MACs")
    ax.set_ylabel(f"Mean {args.metric} validation accuracy (%)")
    ax.set_title("SparkNet versus PerforatedAI")
    ax.set_ylim(bottom=max(0, min(scratch + pai) - 2))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output} ({len(rows)} widths)")


if __name__ == "__main__":
    main()
