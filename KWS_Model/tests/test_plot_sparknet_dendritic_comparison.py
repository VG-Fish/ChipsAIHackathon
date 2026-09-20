import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_sparknet_dendritic_comparison.py"
SPEC = importlib.util.spec_from_file_location("plot_sparknet_dendritic_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _report(status="complete", baseline_accuracy=0.90, dendritic_accuracy=0.92):
    return f"""
status: {status}
candidates:
  - width: 8
    status: {status}
    baseline:
      validation_accuracy: {baseline_accuracy}
      deployed_params: 2000
      macs: 100000
    dendritic:
      validation_accuracy: {dendritic_accuracy}
      deployed_params: 2200
      macs: 101000
      full_cost:
        latency_ms_mean: 1.5
        activation_peak_bytes: 4096
    comparison:
      validation_accuracy_gain: {dendritic_accuracy - baseline_accuracy}
"""


def test_load_records_reads_completed_pairs_and_skips_planned_runs(tmp_path):
    complete = tmp_path / "arms/fc/c8-seed0/reports/sparknet_dendritic_prune_experiment.yaml"
    planned = tmp_path / "arms/fc/c10-seed0/reports/sparknet_dendritic_prune_experiment.yaml"
    complete.parent.mkdir(parents=True)
    planned.parent.mkdir(parents=True)
    complete.write_text(_report())
    planned.write_text(_report(status="planned"))

    records = MODULE.load_records(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.arm == "fc"
    assert record.width == 8
    assert record.seed == 0
    assert record.sparknet_accuracy == 0.90
    assert record.dendritic_params == 2200
    assert record.dendritic_latency_ms == 1.5


def test_aggregate_records_reports_mean_gain_and_overhead():
    first = MODULE.ComparisonRecord("fc", 8, 0, 0.90, 0.92, 2000, 2200, 100000, 101000, 1.5, 4096, Path("a"))
    second = MODULE.ComparisonRecord("fc", 8, 1, 0.94, 0.93, 2000, 2200, 100000, 101000, 1.7, 4096, Path("b"))

    [summary] = MODULE.aggregate_records([first, second])

    assert summary.n_seeds == 2
    assert summary.sparknet_accuracy == 92.0
    assert summary.dendritic_accuracy == 92.5
    assert summary.accuracy_gain_pp == 0.5
    assert summary.parameter_overhead_pct == pytest.approx(10.0)
    assert summary.mac_overhead_pct == pytest.approx(1.0)
    assert summary.dendritic_latency_ms == pytest.approx(1.6)


def test_pareto_frontier_minimizes_cost_and_maximizes_accuracy():
    points = [
        (100, 80, "dominated"),
        (80, 81, "frontier-a"),
        (120, 83, "frontier-b"),
        (150, 82, "dominated"),
    ]

    frontier = MODULE.pareto_frontier(points)

    assert [point[2] for point in frontier] == ["frontier-a", "frontier-b"]
