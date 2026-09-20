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
    planned.parent.mkdir(parents=True, exist_ok=True)
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


def test_filter_widths_removes_c2_from_plot_data():
    records = [
        MODULE.ComparisonRecord("fc", 2, 0, 0.90, 0.91, 1000, 1100, 50000, 51000, None, None, Path("c2")),
        MODULE.ComparisonRecord("fc", 8, 0, 0.90, 0.92, 2000, 2200, 100000, 101000, None, None, Path("c8")),
    ]

    filtered = MODULE.filter_widths(records, {2})

    assert [record.width for record in filtered] == [8]


def test_default_plot_widths_exclude_c2_and_c4():
    assert MODULE.DEFAULT_EXCLUDED_WIDTHS == frozenset({2, 4})


def test_control_only_summaries_remove_noncontrol_arms_from_graph_data():
    records = [
        MODULE.ComparisonRecord("control", 8, 0, 0.90, 0.91, 2000, 2100, 100000, 101000, None, None, Path("control")),
        MODULE.ComparisonRecord("fc", 8, 0, 0.90, 0.92, 2000, 2200, 100000, 101000, None, None, Path("fc")),
        MODULE.ComparisonRecord("depthwise", 8, 0, 0.90, 0.92, 2000, 2200, 100000, 101000, None, None, Path("depthwise")),
    ]

    summaries = MODULE.aggregate_records(records)

    assert {summary.arm for summary in MODULE.control_only_summaries(summaries)} == {"control"}


def test_filter_pareto_data_keeps_only_control_arm():
    records = [
        MODULE.ComparisonRecord("control", 8, 0, 0.90, 0.91, 2000, 2100, 100000, 101000, None, None, Path("control")),
        MODULE.ComparisonRecord("fc", 8, 0, 0.90, 0.92, 2000, 2200, 100000, 101000, None, None, Path("fc")),
    ]
    summaries = MODULE.aggregate_records(records)

    pareto_records, pareto_summaries = MODULE.filter_pareto_data(records, summaries)

    assert {record.arm for record in pareto_records} == {"control"}
    assert {summary.arm for summary in pareto_summaries} == {"control"}


def test_colored_line_series_separates_sparknet_and_dendritic_without_error_bars():
    records = [
        MODULE.ComparisonRecord("control", 4, 0, 0.82, 0.83, 1500, 1500, 97000, 97000, None, None, Path("c4")),
        MODULE.ComparisonRecord("control", 8, 0, 0.91, 0.90, 2350, 2350, 177000, 177000, None, None, Path("c8")),
    ]
    summaries = MODULE.aggregate_records(records)

    series = MODULE.colored_line_series(summaries, "params")

    assert series["SparkNet"] == [(1500.0, 82.0), (2350.0, 91.0)]
    assert series["Dendritic"] == [(1500.0, 83.0), (2350.0, 90.0)]


def test_labeled_pareto_points_include_width_indicator_for_each_model_point():
    records = [
        MODULE.ComparisonRecord("control", 6, 0, 0.87, 0.88, 1906, 1906, 134714, 134714, None, None, Path("c6")),
        MODULE.ComparisonRecord("pointwise", 8, 0, 0.91, 0.90, 2356, 2500, 177336, 191880, None, None, Path("c8")),
    ]

    points = MODULE.labeled_pareto_points(MODULE.aggregate_records(records), "params")

    assert [point[2] for point in points] == ["C6", "C6", "C8", "C8"]


def test_comparison_plots_omit_pareto_frontier_overlay(tmp_path, monkeypatch):
    records = [
        MODULE.ComparisonRecord("control", 6, 0, 0.87, 0.88, 1906, 1950, 134714, 136000, None, None, Path("c6")),
        MODULE.ComparisonRecord("control", 8, 0, 0.91, 0.90, 2356, 2400, 177336, 180000, None, None, Path("c8")),
    ]
    summaries = MODULE.aggregate_records(records)
    saved_figures = []

    def capture_figure(fig, path, dpi):
        saved_figures.append(fig)
        return path

    monkeypatch.setattr(MODULE, "_save", capture_figure)

    MODULE.plot_pareto(records, summaries, "params", tmp_path / "frontier.png", show_seed_points=False)
    MODULE.plot_colored_pareto(summaries, "params", tmp_path / "colored.png")
    MODULE.plot_labeled_pareto(summaries, "params", tmp_path / "labeled.png")

    assert len(saved_figures) == 3
    for figure in saved_figures:
        axis = figure.axes[0]
        assert all(line.get_label() != "Pareto frontier" for line in axis.lines)
        legend = axis.get_legend()
        assert legend is not None
        assert "Pareto frontier" not in {text.get_text() for text in legend.get_texts()}


def test_annotated_plot_shows_only_control_and_restores_width_labels(tmp_path, monkeypatch):
    records = [
        MODULE.ComparisonRecord("control", 6, 0, 0.87, 0.88, 1906, 1950, 134714, 136000, None, None, Path("control-c6")),
        MODULE.ComparisonRecord("control", 8, 0, 0.91, 0.90, 2356, 2400, 177336, 180000, None, None, Path("control-c8")),
        MODULE.ComparisonRecord("fc", 6, 0, 0.87, 0.89, 2100, 2200, 145000, 150000, None, None, Path("fc-c6")),
        MODULE.ComparisonRecord("fc", 8, 0, 0.91, 0.92, 2550, 2650, 185000, 190000, None, None, Path("fc-c8")),
    ]
    summaries = MODULE.aggregate_records(records)
    saved_figures = []

    def capture_figure(fig, path, dpi):
        saved_figures.append(fig)
        return path

    monkeypatch.setattr(MODULE, "_save", capture_figure)

    MODULE.plot_labeled_pareto(summaries, "macs", tmp_path / "annotated.png")

    axis = saved_figures[0].axes[0]
    legend_labels = {text.get_text() for text in axis.get_legend().get_texts()}
    assert legend_labels == {"SparkNet", "Dendritic", "Control"}
    assert [text.get_text() for text in axis.texts] == ["C6", "C6", "C8", "C8"]


def test_output_graph_names_use_standardized_accuracy_prefixes():
    assert MODULE.OUTPUT_FILENAMES["accuracy_params_errorbars"] == "accuracy_vs_params_errorbars.png"
    assert MODULE.OUTPUT_FILENAMES["accuracy_macs_errorbars"] == "accuracy_vs_macs_errorbars.png"
    assert MODULE.OUTPUT_FILENAMES["accuracy_params_lines"] == "accuracy_vs_params_lines.png"
    assert MODULE.OUTPUT_FILENAMES["accuracy_macs_lines"] == "accuracy_vs_macs_lines.png"
    assert MODULE.OUTPUT_FILENAMES["accuracy_params_annotated"] == "accuracy_vs_params_annotated.png"
    assert MODULE.OUTPUT_FILENAMES["accuracy_macs_annotated"] == "accuracy_vs_macs_annotated.png"
    assert all(not name.startswith("pareto_") for name in MODULE.OUTPUT_FILENAMES.values())


def test_markdown_report_contains_readable_summary_tables(tmp_path):
    records = [
        MODULE.ComparisonRecord("control", 6, 0, 0.87, 0.88, 1906, 1906, 134714, 134714, 1.24, 6464, Path("control-c6")),
        MODULE.ComparisonRecord("fc", 8, 0, 0.91, 0.92, 2356, 2764, 177336, 177732, 1.41, 6464, Path("fc-c8")),
    ]

    report_path = MODULE.write_markdown_report(MODULE.aggregate_records(records), tmp_path / "model_stats.md")
    report = report_path.read_text()

    assert report.startswith("# SparkNet vs Dendritic Model Statistics")
    assert "## Control-arm results" in report
    assert "## Placement-arm summary" in report
    assert "## Complete results" in report
    assert "| C6 | 1 | 87.00% | 88.00% | +1.00 pp |" in report
    assert "| Classifier | C8 | 1 | 91.00% ± 0.00 | 92.00% ± 0.00 | +1.00 pp |" in report
    assert "2 matched seed pairs" in report


def test_markdown_report_has_standardized_output_name():
    assert MODULE.OUTPUT_FILENAMES["report"] == "model_stats.md"


def test_load_broader_model_records_reads_completed_v3_runs(tmp_path):
    complete = tmp_path / "pointwise_b2-bn/c14-seed0/reports/grow_summary.yaml"
    planned = tmp_path / "pointwise_b2-bn/c15-seed0/reports/grow_summary.yaml"
    complete.parent.mkdir(parents=True)
    planned.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text(
        """
status: complete
width: 14
seed: 0
model_name: sparknet_c14_paper
arm: pointwise_b2-bn
results:
  best_val_acc_overall: 0.94803
cost:
  deployed:
    params: 4232
    macs: 355500
"""
    )
    planned.parent.mkdir(parents=True, exist_ok=True)
    planned.write_text(
        """
status: planned
width: 15
seed: 0
model_name: sparknet_c15_paper
results: {}
cost: {}
"""
    )

    records = MODULE.load_broader_model_records(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.model_name == "sparknet_c14_paper"
    assert record.width == 14
    assert record.seed == 0
    assert record.validation_accuracy_pct == pytest.approx(94.803)
    assert record.params == 4232
    assert record.macs == 355500


def test_load_faithful_sparknet_records_reads_pre_dendrite_model(tmp_path):
    report = tmp_path / "pointwise_b2-bn/c8-seed0/reports/grow_summary.yaml"
    report.parent.mkdir(parents=True)
    report.write_text(
        """
status: complete
width: 8
seed: 0
model_name: sparknet_c8_paper
arm: pointwise_b2-bn
results:
  best_val_acc_pre_switch: 0.914
  best_val_acc_overall: 0.921
cost:
  base:
    params: 2356
    macs: 177336
  deployed:
    params: 2444
    macs: 184608
"""
    )

    [record] = MODULE.load_faithful_sparknet_records(tmp_path)

    assert record.model_name == "sparknet_c8_paper"
    assert record.width == 8
    assert record.validation_accuracy_pct == pytest.approx(91.4)
    assert record.params == 2356
    assert record.macs == 177336


def test_load_faithful_sparknet_records_prefers_dedicated_paper_replication(tmp_path):
    grow_report = tmp_path / "grow/pointwise_b2-bn/c16-seed0/reports/grow_summary.yaml"
    grow_report.parent.mkdir(parents=True)
    grow_report.write_text(
        """
status: complete
width: 16
seed: 0
model_name: sparknet_c16_paper
arm: pointwise_b2-bn
results:
  best_val_acc_pre_switch: 0.945
cost:
  base:
    params: 4636
    macs: 396304
"""
    )
    paper_summary = tmp_path / "paper/c16-seed0/metrics/summaries.yaml"
    paper_summary.parent.mkdir(parents=True)
    paper_summary.write_text(
        """
phases:
  paper_replication/sparknet_c16_paper:
    best_val_acc: 0.9532
"""
    )

    records = MODULE.load_faithful_sparknet_records(tmp_path / "grow", tmp_path / "paper")

    assert len(records) == 2
    assert max(record.validation_accuracy_pct for record in records) == pytest.approx(95.32)
    assert all(record.params == 4636 for record in records)


def test_faithful_sparknet_summaries_keep_exact_paper_model_names(tmp_path):
    faithful = MODULE.FaithfulSparkNetRecord(
        "sparknet_c8_paper", "pointwise_b2-bn", 8, 0, 91.4, 2356, 177336, Path("c8")
    )
    gated = MODULE.FaithfulSparkNetRecord(
        "sparknet_c8g16_paper", "pointwise_b2-bn", 8, 0, 90.8, 2268, 190880, Path("c8g16")
    )

    summaries = MODULE.best_faithful_sparknet_summary(
        MODULE.summarize_faithful_sparknet_records([faithful, gated])
    )

    assert [summary.model_name for summary in summaries] == ["sparknet_c8_paper"]
    assert summaries[0].params == 2356


def test_best_vs_faithful_filters_models_dominated_on_the_plotted_metric():
    best = [
        MODULE.BroaderModelSummary(
            "sparknet_c6g16_paper", "pointwise_b2-bn", 6, 1, 87.0, 87.0, 0.0, 1624, 129068, (0,)
        ),
        MODULE.BroaderModelSummary(
            "sparknet_c12g16_paper", "pointwise_b2-bn", 12, 1, 93.0, 93.0, 0.0, 3388, 297536, (0,)
        ),
        MODULE.BroaderModelSummary(
            "sparknet_c17g16_paper", "pointwise_b2-bn", 17, 1, 94.7, 94.7, 0.0, 4803, 432371, (0,)
        ),
        MODULE.BroaderModelSummary(
            "sparknet_c18g16_paper", "pointwise_b2-bn", 18, 1, 96.1, 96.1, 0.0, 5176, 468428, (0,)
        ),
    ]
    faithful = [
        MODULE.FaithfulSparkNetSummary(
            "sparknet_c6_paper", 6, 1, 88.0, 88.0, 0.0, 1906, 134714, (0,)
        ),
        MODULE.FaithfulSparkNetSummary(
            "sparknet_c12_paper", 12, 1, 94.0, 94.0, 0.0, 3400, 277124, (0,)
        ),
        MODULE.FaithfulSparkNetSummary(
            "sparknet_c16_paper", 16, 1, 95.3, 95.3, 0.0, 4636, 396304, (0,)
        ),
    ]

    params = MODULE.filter_best_models_by_faithful_dominance(best, faithful, "params")
    macs = MODULE.filter_best_models_by_faithful_dominance(best, faithful, "macs")

    assert [summary.width for summary in params] == [6, 12, 18]
    assert [summary.width for summary in macs] == [6, 18]


def test_new_frontier_selects_requested_trained_models():
    requested = {
        "sparknet_c6g16_paper",
        "sparknet_c9g8_paper",
        "sparknet_c10g8_paper",
        "sparknet_c10g16_paper",
        "sparknet_c12_paper",
        "sparknet_c16g16_paper",
        "sparknet_c18g16_paper",
    }
    summaries = [
        MODULE.BroaderModelSummary(
            model_name,
            "pointwise_b2-bn",
            width,
            1,
            90.0,
            90.0,
            0.0,
            2000 + width,
            100000 + width,
            (0,),
        )
        for width, model_name in (
            (6, "sparknet_c6g16_paper"),
            (9, "sparknet_c9g8_paper"),
            (10, "sparknet_c10g8_paper"),
            (10, "sparknet_c10g16_paper"),
            (12, "sparknet_c12_paper"),
            (16, "sparknet_c16g16_paper"),
            (18, "sparknet_c18g16_paper"),
            (8, "sparknet_c8g16_paper"),
        )
    ]

    selected = MODULE.select_new_frontier_summaries(summaries)

    assert {summary.model_name for summary in selected} == requested


def test_best_vs_faithful_plot_has_two_labeled_series_and_frontiers(tmp_path, monkeypatch):
    best = [
        MODULE.BroaderModelSummary(
            "sparknet_c6g16_paper", "pointwise_b2-bn", 6, 1, 91.0, 91.0, 0.0, 2268, 190880, (0,)
        ),
        MODULE.BroaderModelSummary(
            "sparknet_c10g8_paper", "pointwise_b2-bn", 10, 1, 92.0, 92.0, 0.0, 2384, 211388, (0,)
        ),
        MODULE.BroaderModelSummary(
            "sparknet_c11g16_paper", "pointwise_b2-bn", 11, 1, 93.0, 93.0, 0.0, 2700, 240000, (0,)
        ),
    ]
    faithful = [
        MODULE.FaithfulSparkNetSummary(
            "sparknet_c8_paper", 8, 1, 91.4, 91.4, 0.0, 2356, 177336, (0,)
        )
    ]
    saved_figures = []

    def capture_figure(fig, path, dpi):
        saved_figures.append(fig)
        return path

    monkeypatch.setattr(MODULE, "_save", capture_figure)

    MODULE.plot_best_vs_faithful(best, faithful, "params", tmp_path / "models.png")

    axis = saved_figures[0].axes[0]
    legend_labels = {text.get_text() for text in axis.get_legend().get_texts()}
    assert legend_labels == {
        "Best trained dendritic models",
        "Faithful Sparknet (previous SOTA)",
        "Old faithful SparkNet frontier",
        "New best trained frontier",
        "Most parameter-efficient",
        "Practical model (C10g8)",
    }
    assert {text.get_text() for text in axis.texts} == {
        "C6g16",
        "C10g8",
        "C8",
        "cXgY: X = channel width, Y = gate width (e.g. C10g8)",
    }
    syntax_note = next(
        text
        for text in axis.texts
        if text.get_text().startswith("cXgY:")
    )
    assert syntax_note.get_position() == (0.99, 0.015)
    assert syntax_note.get_horizontalalignment() == "right"
    assert syntax_note.get_verticalalignment() == "bottom"
    assert {line.get_label() for line in axis.lines} == {
        "Old faithful SparkNet frontier",
        "New best trained frontier",
    }
    assert len(axis.collections) == 5
    model_annotations = [text for text in axis.texts if not text.get_text().startswith("cXgY:")]
    assert all(annotation.get_position() == (-8, 8) for annotation in model_annotations)
    assert all(annotation.get_horizontalalignment() == "right" for annotation in model_annotations)
    assert all(annotation.get_verticalalignment() == "bottom" for annotation in model_annotations)


def test_markdown_report_includes_broader_and_exported_sections(tmp_path):
    records = [
        MODULE.ComparisonRecord("control", 6, 0, 0.87, 0.88, 1906, 1906, 134714, 134714, 1.24, 6464, Path("control-c6")),
    ]
    broader = [
        MODULE.BroaderModelSummary(
            model_name="sparknet_c14_paper",
            arm="pointwise_b2-bn",
            width=14,
            n_seeds=1,
            mean_validation_accuracy_pct=94.803,
            best_validation_accuracy_pct=94.803,
            validation_accuracy_std_pct=0.0,
            params=4232,
            macs=355500,
            seed_ids=(0,),
        )
    ]
    exported = [
        MODULE.ExportedModelRecord(
            model="PAI C6g16",
            width=6,
            seed=0,
            test_accuracy_pct=85.787,
            params=1624,
            macs=129068,
            report_path=Path("rp2040.yaml"),
        )
    ]

    report_path = MODULE.write_markdown_report(
        MODULE.aggregate_records(records),
        tmp_path / "model_stats.md",
        broader_summaries=broader,
        exported_records=exported,
    )
    report = report_path.read_text()

    assert "## Broader trained-model scan (v3 validation)" in report
    assert "C14 paper" in report
    assert "## Published SC2 and exported PAI references" in report
    assert "PAI C6g16" in report


def test_load_broader_test_records_matches_saved_v3_checkpoints(tmp_path):
    run_root = tmp_path / "pointwise_b2-bn" / "c14-seed0"
    summary_path = run_root / "reports/grow_summary.yaml"
    checkpoint = run_root / "pai/candidates/model/final_clean_pai.pt"
    summary_path.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    broader = [
        MODULE.BroaderModelRecord(
            model_name="sparknet_c14_paper",
            arm="pointwise_b2-bn",
            width=14,
            seed=0,
            validation_accuracy_pct=94.803,
            params=4232,
            macs=355500,
            report_path=summary_path,
        )
    ]
    test_path = tmp_path / "test.json"
    test_path.write_text(
        __import__("json").dumps(
            {
                "evaluations": [
                    {
                        "checkpoint": str(checkpoint),
                        "test_accuracy": 0.9438,
                        "validation_accuracy": 0.94803,
                        "seed": 0,
                        "num_params": 4232,
                    }
                ]
            }
        )
    )

    [record] = MODULE.load_broader_test_records(test_path, broader)

    assert record.model_name == "sparknet_c14_paper"
    assert record.test_accuracy_pct == pytest.approx(94.38)
    assert record.params == 4232
