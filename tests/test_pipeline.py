import pytest
import yaml

from kws.optimize.distill import distillation_fingerprint
from kws.pipeline import (
    STAGES,
    select_deployment_candidate,
    validate_sparsity_report,
)
from kws.optimize.dendritic_prune_loop import search_fingerprint


def _sweep():
    return {
        "pareto": {
            "frontier": [
                {"label": "w18", "accuracy": 0.918,
                 "costs": {"deployed_params": 2946.0, "latency_ms_p50": 0.30},
                 "save_name": "dendritic_prune_w18"},
                {"label": "w15", "accuracy": 0.900,
                 "costs": {"deployed_params": 2514.0, "latency_ms_p50": 0.24},
                 "save_name": "dendritic_prune_w15"},
            ]
        }
    }


def test_stages_cover_the_six_framework_steps():
    assert STAGES == ("teacher", "student", "sparsity", "cluster", "quantize", "benchmark")


def test_selection_minimizes_the_configured_cost_axis():
    config = {"sparsity": {"select_by": "deployed_params"}}
    assert select_deployment_candidate(_sweep(), config)["label"] == "w15"

    config = {"sparsity": {"select_by": "latency_ms_p50"}}
    assert select_deployment_candidate(_sweep(), config)["label"] == "w15"


def test_selection_can_prefer_accuracy_instead():
    config = {"sparsity": {"select_by": "accuracy"}}
    assert select_deployment_candidate(_sweep(), config)["label"] == "w18"


def test_selection_refuses_an_empty_frontier():
    with pytest.raises(ValueError, match="no Pareto frontier"):
        select_deployment_candidate({"pareto": {"frontier": []}}, {"sparsity": {}})


def test_selection_refuses_an_unrecorded_cost_axis():
    config = {"sparsity": {"select_by": "activation_peak_bytes"}}
    with pytest.raises(ValueError, match="unrecorded cost"):
        select_deployment_candidate(_sweep(), config)


def test_student_recipe_fingerprint_covers_training_inputs(tmp_path):
    teacher = tmp_path / "teacher.pt"
    warm_start = tmp_path / "warm.pt"
    teacher.write_bytes(b"teacher-v1")
    warm_start.write_bytes(b"warm-v1")
    model_cfg = {"name": "student", "block_channels": [2]}
    data_cfg = {"classes": ["yes", "no"]}
    train_cfg = {"seed": 0, "epochs": 2}

    first = distillation_fingerprint(
        str(teacher), model_cfg, data_cfg, train_cfg, str(warm_start)
    )
    changed_train = distillation_fingerprint(
        str(teacher), model_cfg, data_cfg, {"seed": 1, "epochs": 2}, str(warm_start)
    )
    warm_start.write_bytes(b"warm-v2")
    changed_artifact = distillation_fingerprint(
        str(teacher), model_cfg, data_cfg, train_cfg, str(warm_start)
    )

    assert first != changed_train
    assert first != changed_artifact


def test_downstream_stages_reject_stale_sparsity_fingerprint(tmp_path):
    teacher = tmp_path / "teacher.pt"
    student = tmp_path / "student.pt"
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    search_path = tmp_path / "search.yaml"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    data_cfg = {"dataset": "synthetic"}
    train_cfg = {"seed": 0}
    search_cfg = {
        "start_channels": 2,
        "minimum_channels": 1,
        "channel_step": 1,
        "summary_path": str(tmp_path / "summary.yaml"),
    }
    for path, payload in (
        (data_path, data_cfg),
        (train_path, train_cfg),
        (search_path, search_cfg),
    ):
        path.write_text(yaml.safe_dump(payload))
    config = {
        "teacher": {"checkpoint": str(teacher)},
        "student": {"checkpoint": str(student)},
        "data_config": str(data_path),
        "sparsity": {
            "train_config": str(train_path),
            "search_config": str(search_path),
        },
    }
    fingerprint = search_fingerprint(
        str(student), str(teacher), data_cfg, train_cfg, search_cfg
    )
    sweep = {
        "status": "complete",
        "fingerprint": fingerprint,
        "pareto": {"frontier": [{"label": "w2"}]},
    }

    validate_sparsity_report(sweep, config)
    train_path.write_text(yaml.safe_dump({"seed": 1}))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_sparsity_report(sweep, config)
