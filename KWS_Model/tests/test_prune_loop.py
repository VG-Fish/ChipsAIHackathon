import pytest
import torch
import yaml

from kws.models.ds_cnn import build_ds_cnn
from kws.optimize.dendritic import (
    DendriticCycleResult,
    FRAMEWORK_CYCLE_VERSION,
    cycle_fingerprint,
)
from kws.optimize.dendritic_prune_loop import (
    ParetoSearch,
    _load_completed_result,
    _target_model_cfg,
    candidate_costs,
    candidate_widths,
    judge_candidate,
)
from kws.optimize.pareto import ParetoPoint


def _result(width, val_acc, deployed, cost=None):
    return DendriticCycleResult(
        save_name=f"run_w{width}",
        block_channels=[width, width],
        base_params=width * 100,
        deployed_params=deployed,
        best_val_acc=val_acc,
        epochs=10,
        elapsed_seconds=1.0,
        cost=cost,
    )


def test_candidate_costs_fall_back_to_parameters_when_nothing_was_profiled():
    assert candidate_costs(_result(16, 0.91, 2654)) == {"deployed_params": 2654.0}


def test_candidate_costs_include_every_profiled_axis():
    costs = candidate_costs(
        _result(16, 0.91, 2654, cost={"macs": 1.2e6, "latency_ms_p50": 0.3,
                                      "weight_bytes": 2654, "params": 2654})
    )
    assert costs == {
        "deployed_params": 2654.0,
        "macs": 1.2e6,
        "latency_ms_p50": 0.3,
        "weight_bytes": 2654.0,
    }


def test_search_compares_only_axes_every_candidate_recorded():
    """A reused legacy run must not make an unprofiled axis look free."""
    search = ParetoSearch(minimum_accuracy=0.9, patience=2)
    search.add(ParetoPoint("w18", 0.92, {"deployed_params": 2946.0, "macs": 3e6}))
    assert search.cost_keys == ("deployed_params", "macs")

    search.add(ParetoPoint("w17", 0.91, {"deployed_params": 2798.0}))
    assert search.cost_keys == ("deployed_params",)
    assert {p.label for p in search.frontier.front} == {"w18", "w17"}


def test_search_preserves_patience_when_frontier_axes_are_rebuilt():
    search = ParetoSearch(minimum_accuracy=0.9, patience=2)
    search.add(ParetoPoint("w18", 0.92, {"deployed_params": 2946.0, "macs": 3e6}))
    search.record_inadmissible("below floor")
    # The missing MACs shrink the shared-axis set and force a frontier rebuild.
    # That rebuild must not erase the already-counted rejected candidate.
    update = search.add(ParetoPoint("w17", 0.91, {"deployed_params": 3000.0}))
    assert not update.extended_frontier
    assert search.stagnant_streak == 2
    assert search.should_stop()


def test_search_keeps_going_past_a_cheaper_less_accurate_candidate():
    search = ParetoSearch(minimum_accuracy=0.9, patience=2)
    search.add(ParetoPoint("w18", 0.918, {"deployed_params": 2946.0}))
    update = search.add(ParetoPoint("w16", 0.901, {"deployed_params": 2654.0}))

    assert update.extended_frontier
    assert not search.should_stop()


def test_search_stops_after_two_stalled_candidates():
    search = ParetoSearch(minimum_accuracy=0.9, patience=2)
    search.add(ParetoPoint("w18", 0.918, {"deployed_params": 2946.0}))
    search.add(ParetoPoint("w17", 0.905, {"deployed_params": 2960.0}))  # dearer and worse
    assert not search.should_stop()
    search.add(ParetoPoint("w16", 0.903, {"deployed_params": 2950.0}))
    assert search.should_stop()


def test_search_reports_accuracy_and_pareto_stalls_separately():
    search = ParetoSearch(minimum_accuracy=0.9, patience=2)
    search.add(ParetoPoint("w18", 0.918, {"deployed_params": 2946.0}))
    search.record_inadmissible("below floor")
    search.record_inadmissible("below floor")

    assert search.should_stop()
    assert search.stop_cause == "accuracy_inadmissible"

    search = ParetoSearch(
        minimum_accuracy=0.9,
        patience=2,
        relative_cost_tolerance=0.05,
    )
    search.add(ParetoPoint("w18", 0.918, {"deployed_params": 2946.0}))
    search.add(ParetoPoint("w17", 0.917, {"deployed_params": 2900.0}))
    search.add(ParetoPoint("w16", 0.916, {"deployed_params": 2890.0}))

    assert search.should_stop()
    assert search.stop_cause == "pareto_stagnation"


def test_admission_rule_still_enforces_the_floor_and_the_optional_drop():
    assert judge_candidate(0.901, 0.90, 0.906, None).accepted
    assert not judge_candidate(0.899, 0.90, 0.901, None).accepted
    assert not judge_candidate(0.902, 0.90, 0.910, 0.005).accepted


def test_widths_descend_to_the_configured_minimum():
    assert candidate_widths(18, 14, 1) == [18, 17, 16, 15, 14]
    assert candidate_widths(18, 13, 3) == [18, 15, 13]
    with pytest.raises(ValueError):
        candidate_widths(4, 8, 1)


def test_completed_run_reuse_requires_framework_provenance_and_keeps_resume_score(
    tmp_path,
):
    source_cfg = {
        "name": "tiny",
        "initial_channels": 2,
        "initial_kernel": 3,
        "initial_stride": 1,
        "block_channels": [4, 4],
        "dropout": 0.0,
    }
    source = build_ds_cnn(source_cfg, (8, 8), 3)
    checkpoint_path = tmp_path / "student.pt"
    torch.save(
        {
            "model_cfg": source_cfg,
            "model_state_dict": source.state_dict(),
            "input_shape": (8, 8),
            "num_classes": 3,
        },
        checkpoint_path,
    )

    run_dir = tmp_path / "candidate_w2"
    run_dir.mkdir()
    (run_dir / "final_clean_pai.pt").write_bytes(b"complete")
    (run_dir / f"{run_dir.name}Scores.csv").write_text("epoch,score\n1,0.91\n")
    (run_dir / f"{run_dir.name}_best_arch_scores.csv").write_text(
        "Param Counts,Max Valid Scores,Train\n100,0.9100,0.90\n"
    )

    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"fixed teacher")
    teacher = str(teacher_path)
    data_cfg = {"dataset": "synthetic"}
    train_cfg = {"pruning": {"kind": "structured", "keep_ratio": 1.0}}
    candidate_train_cfg = dict(train_cfg)
    candidate_train_cfg["pruning"] = {"kind": "structured", "keep_ratio": 0.5}
    target_cfg = _target_model_cfg(str(checkpoint_path), 2)
    metadata = {
        "framework_cycle_version": FRAMEWORK_CYCLE_VERSION,
        "fingerprint": cycle_fingerprint(
            str(checkpoint_path),
            teacher,
            data_cfg,
            target_cfg,
            candidate_train_cfg,
        ),
        "source_checkpoint": str(checkpoint_path),
        "teacher_checkpoint": teacher,
        "base_model_cfg": target_cfg,
        "result": {
            "best_val_acc": 0.93,
            "pai_deployed_params": 100,
            "prune_finetune": {"status": "complete"},
            "resume": {"status": "complete"},
            "phase_trail": [{"mode": "n"}],
            "cost": {
                "params": 102,
                "macs": 1000,
                "latency_ms_p50": 1.0,
                "weight_bytes": 102,
                "activation_peak_bytes": 256,
            },
        },
    }
    (run_dir / "cycle_metadata.yaml").write_text(yaml.safe_dump(metadata))

    result = _load_completed_result(
        str(run_dir),
        2,
        str(checkpoint_path),
        teacher,
        data_cfg,
        train_cfg,
    )
    assert result.best_val_acc == 0.93
    assert result.deployed_params == 102

    metadata.pop("framework_cycle_version")
    (run_dir / "cycle_metadata.yaml").write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="legacy run"):
        _load_completed_result(
            str(run_dir),
            2,
            str(checkpoint_path),
            teacher,
            data_cfg,
            train_cfg,
        )
