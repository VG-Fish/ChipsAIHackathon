import pytest
import yaml

from scripts import backfill_zero_dendrite_control as backfill_script
from scripts.report_test_accuracy import Result, summarize, targets_from_report


def test_test_summary_does_not_infer_seed_replication_from_result_count():
    results = [
        Result("a.pt", "arm", 12, None, 0.90, test_accuracy=0.91),
        Result("b.pt", "arm", 12, None, 0.92, test_accuracy=0.93),
    ]

    row = summarize(results)[0]

    assert row["n_results"] == 2
    assert row["n_seeds"] == 0
    assert row["seed_provenance_verified"] is False
    assert row["test_accuracy_sd"] is None
    assert row["validation_accuracy_sd"] is None


def test_test_summary_reports_spread_for_distinct_recorded_seeds():
    results = [
        Result("a.pt", "arm", 12, 0, 0.90, test_accuracy=0.91),
        Result("b.pt", "arm", 12, 1, 0.92, test_accuracy=0.93),
    ]

    row = summarize(results)[0]

    assert row["n_seeds"] == 2
    assert row["seed_provenance_verified"] is True
    assert row["test_accuracy_sd"] == pytest.approx(0.0141421356)


def test_report_seed_must_be_recorded_not_inferred_from_directory(tmp_path):
    run_root = tmp_path / "seed4"
    report_path = run_root / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    checkpoint = run_root / "baseline.pt"
    report_path.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    report = {
        "status": "complete",
        "candidates": [
            {
                "status": "complete",
                "width": 12,
                "baseline": {
                    "checkpoint": str(checkpoint),
                    "validation_accuracy": 0.9,
                },
            }
        ],
    }
    report_path.write_text(yaml.safe_dump(report))

    assert targets_from_report(run_root)[0].seed is None

    report["seed"] = 4
    report_path.write_text(yaml.safe_dump(report))
    assert targets_from_report(run_root)[0].seed == 4


def test_backfill_keeps_pipeline_headline_and_labels_pai_row_descriptively(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    report_path = run_root / "reports" / backfill_script.REPORT_NAME
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        yaml.safe_dump(
            {
                "candidates": [
                    {
                        "status": "complete",
                        "width": 12,
                        "baseline": {"validation_accuracy": 0.80},
                        "dendritic": {
                            "checkpoint": "models/checkpoints/sparsity/c12/resume/best.pt",
                            "validation_accuracy": 0.92,
                            "pai_search_validation_accuracy": 0.919,
                        },
                        "comparison": {
                            "validation_accuracy_gain": 0.005,
                            "validation_accuracy_gain_basis": "zero_dendrite",
                            "pai_search_accuracy_gain": 0.004,
                        },
                    }
                ]
            },
            sort_keys=False,
        )
    )
    monkeypatch.setattr(
        backfill_script,
        "read_pai_zero_dendrite_score",
        lambda _path: (0.915, 100),
    )

    rows = backfill_script.backfill(run_root, apply=True)
    comparison = yaml.safe_load(report_path.read_text())["candidates"][0]["comparison"]

    assert rows[0].pipeline_gain == pytest.approx(0.12)
    assert rows[0].min_row_gain == pytest.approx(0.005)
    assert comparison["validation_accuracy_gain"] == pytest.approx(0.12)
    assert comparison["validation_accuracy_gain_basis"] == "prune_finetune_baseline"
    assert comparison["validation_accuracy_gain_vs_prune_finetune"] == pytest.approx(
        0.12
    )
    assert comparison[
        "final_validation_accuracy_gain_vs_pai_zero_architecture"
    ] == pytest.approx(0.005)
    assert comparison["pai_zero_architecture_comparison_basis"] == (
        "minimum_parameter_row_in_best_arch_scores"
    )
    assert comparison["pai_search_accuracy_gain"] == pytest.approx(0.004)
    assert comparison["pai_search_accuracy_gain_vs_prune_finetune"] == pytest.approx(
        0.119
    )
