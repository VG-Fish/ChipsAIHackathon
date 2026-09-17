import importlib
import importlib.util
import json

import pytest
import torch
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


SCRATCH_BEST = "models/checkpoints/paper_replication/best.pt"


def _scratch_path(scratch_root, width, seed):
    return scratch_root / f"c{width}-seed{seed}" / SCRATCH_BEST


def _write_scratch_checkpoint(scratch_root, width, seed, val_acc):
    """Write a real, loadable from-scratch baseline at the sweep's layout."""
    from kws.models.registry import build_model

    model_cfg = {
        "family": "sparknet",
        "name": f"sparknet_c{width}_paper",
        "channels": width,
        "gate_channels": 32,
        "sparsity_weight": 1.0,
    }
    model = build_model(model_cfg, input_shape=(32, 101), num_classes=12)
    checkpoint = _scratch_path(scratch_root, width, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_family": "sparknet",
            "model_cfg": model_cfg,
            "input_shape": (32, 101),
            "num_classes": 12,
            "num_keywords": 10,
            "label_map": {},
            "val_acc": val_acc,
            "seed": seed,
        },
        checkpoint,
    )
    return checkpoint, sum(p.numel() for p in model.parameters())


def _write_arm_report(
    root, arm_ids, seed, width, accuracy, params=100, source=None
):
    report_path = root / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    checkpoint = root / "selected.pt"
    report_path.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    report_path.write_text(
        yaml.safe_dump(
            {
                "status": "complete",
                "seed": seed,
                "selection_split": "validation",
                "test_split_used": False,
                "source": {"checkpoint": str(source) if source else None},
                "perforatedai": {"module_ids": arm_ids},
                "candidates": [
                    {
                        "width": width,
                        "status": "complete",
                        "dendritic": {
                            "checkpoint": str(checkpoint),
                            "validation_accuracy": accuracy,
                            "deployed_params": params,
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )


def test_validation_selector_freezes_one_cross_seed_arm_per_width(tmp_path):
    """A per-seed winner would leak seed noise into the held-out test set."""
    spec = importlib.util.find_spec("scripts.select_sparknet_arms")
    assert spec is not None
    selector = importlib.import_module("scripts.select_sparknet_arms")

    run_roots = []
    arms = {
        "pointwise": [".blocks.3.pointwise", ".blocks.2.pointwise"],
        "fc": [".fc"],
        "gate_conv": [".gate_conv"],
        "depthwise": [".blocks.3.depthwise", ".blocks.2.depthwise"],
        "control": [],
    }
    for seed in (0, 1):
        for arm, module_ids in arms.items():
            root = tmp_path / arm / f"c12-seed{seed}"
            # Pointwise wins on the mean even though fc wins seed 0.
            scores = {
                "pointwise": (0.90, 0.94),
                "fc": (0.92, 0.89),
                "gate_conv": (0.88, 0.88),
                "depthwise": (0.87, 0.87),
                "control": (0.86, 0.86),
            }
            _write_arm_report(root, module_ids, seed, 12, scores[arm][seed])
            run_roots.append(root)

    selection = selector.build_selection(run_roots, expected_seeds=(0, 1))

    assert selection["selection_split"] == "validation"
    assert selection["test_split_used"] is False
    assert selection["widths"][0]["selected_arm"] == "pointwise"
    # The winner goes to test, and so does the budget-matched control: without
    # it the test table has no no-dendrite arm that saw the same schedule.
    assert selection["widths"][0]["reference_arms"] == ["control"]
    assert [
        (target["arm"], target["seed"]) for target in selection["targets"]
    ] == [("pointwise", 0), ("pointwise", 1), ("control", 0), ("control", 1)]


def test_a_winning_control_is_not_carried_to_test_twice(tmp_path):
    selector = importlib.import_module("scripts.select_sparknet_arms")

    run_roots = []
    for seed in (0, 1):
        for arm, module_ids in selector.ARM_MODULE_IDS.items():
            root = tmp_path / arm / f"c12-seed{seed}"
            _write_arm_report(
                root,
                sorted(module_ids),
                seed,
                12,
                0.95 if arm == "control" else 0.90,
            )
            run_roots.append(root)

    selection = selector.build_selection(run_roots, expected_seeds=(0, 1))

    assert selection["widths"][0]["selected_arm"] == "control"
    assert selection["widths"][0]["reference_arms"] == []
    assert [target["arm"] for target in selection["targets"]] == ["control"] * 2


def test_selection_refuses_arms_that_did_not_grow_from_their_own_seed(tmp_path):
    """Five runs off one baseline is not five seeds, and only this can see it."""
    selector = importlib.import_module("scripts.select_sparknet_arms")
    scratch_root = tmp_path / "scratch"

    run_roots = []
    for seed in (0, 1):
        _write_scratch_checkpoint(scratch_root, 12, seed, 0.88 + seed / 100)
        for arm, module_ids in selector.ARM_MODULE_IDS.items():
            root = tmp_path / "arms" / arm / f"c12-seed{seed}"
            # Every arm claims seed 0's baseline, whatever seed it reports.
            _write_arm_report(
                root,
                sorted(module_ids),
                seed,
                12,
                0.90,
                source=_scratch_path(scratch_root, 12, 0),
            )
            run_roots.append(root)

    with pytest.raises(ValueError, match="grew from"):
        selector.build_selection(
            run_roots, expected_seeds=(0, 1), scratch_root=scratch_root
        )


def test_selection_manifest_is_the_only_source_for_one_time_test_targets(tmp_path):
    """Test reporting must consume frozen targets, not re-select from run globs."""
    from scripts import report_test_accuracy as reporting

    manifest = tmp_path / "selection.json"
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "test_split_used": False,
                "targets": [
                    {
                        "checkpoint": str(checkpoint),
                        "arm": "pointwise",
                        "width": 12,
                        "seed": 0,
                        "validation_accuracy": 0.9,
                        "run_root": str(tmp_path),
                        "pai_dir": str(tmp_path / "pai"),
                    }
                ],
            }
        )
    )

    loader = getattr(reporting, "targets_from_selection_manifest", None)
    assert loader is not None
    targets = loader(manifest)

    assert len(targets) == 1
    assert targets[0].checkpoint == checkpoint
    assert targets[0].arm == "pointwise"
    assert targets[0].validation_accuracy == pytest.approx(0.9)


def test_validation_selection_includes_fixed_scratch_baselines(tmp_path):
    selector = importlib.import_module("scripts.select_sparknet_arms")
    scratch_root = tmp_path / "scratch"
    run_roots = []
    expected_params = None
    for seed in (0, 1):
        _, expected_params = _write_scratch_checkpoint(
            scratch_root, 12, seed, 0.88 + seed / 100
        )
        for arm, module_ids in selector.ARM_MODULE_IDS.items():
            root = tmp_path / "arms" / arm / f"c12-seed{seed}"
            _write_arm_report(
                root,
                list(module_ids),
                seed,
                12,
                0.9,
                source=_scratch_path(scratch_root, 12, seed),
            )
            run_roots.append(root)

    selection = selector.build_selection(
        run_roots,
        expected_seeds=(0, 1),
        scratch_root=scratch_root,
    )

    scratch = [target for target in selection["targets"] if target["arm"] == "scratch"]
    assert [target["seed"] for target in scratch] == [0, 1]
    assert [target["validation_accuracy"] for target in scratch] == pytest.approx(
        [0.88, 0.89]
    )
    assert all(target["pai_dir"] is None for target in scratch)
    # Counted the way every arm row is counted: parameters only.  Summing the
    # state dict would fold in BatchNorm buffers and num_batches_tracked, and
    # the scratch rows would stop being comparable with the arms'.
    assert all(target["deployed_params"] == expected_params for target in scratch)
