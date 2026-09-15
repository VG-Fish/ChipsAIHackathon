from pathlib import Path

import pytest
import torch
import yaml

from kws.models.sparknet import SparkNet
from kws.optimize.sparknet_dendritic_prune_experiment import (
    dendritic_delta,
    interpolate_pruning_curve,
    load_config,
    run_experiment,
    validate_config,
)


CONFIG = Path(__file__).parents[1] / "configs/experiment/sparknet_c12_dendritic_prune_no_kd.yaml"


@pytest.fixture
def experiment_config(tmp_path):
    """A complete local C12 input set; ordinary tests never need Phase B artifacts."""
    input_shape = (32, 5)
    source_cfg = {
        "family": "sparknet",
        "name": "sparknet_c12",
        "channels": 12,
        "gate_channels": 32,
        "sparsity_weight": 0.01,
    }
    source = SparkNet(32, 12, channels=12, gate_channels=32, input_shape=input_shape)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "model_cfg": source_cfg,
            "input_shape": input_shape,
            "num_classes": 12,
            "val_acc": 0.9,
        },
        checkpoint,
    )
    data_config = tmp_path / "data.yaml"
    model_config = tmp_path / "model.yaml"
    train_config = tmp_path / "train.yaml"
    data_config.write_text(yaml.safe_dump({"features": {"type": "mfcc", "n_mels": 32}}))
    model_config.write_text(yaml.safe_dump(source_cfg))
    pai = {
        "conversion": "module_ids",
        "module_ids": [".blocks.3", ".gate_conv", ".fc"],
        "max_dendrites": -1,
        "testing_dendrite_capacity": False,
        "switch_mode": "history",
        "history_lookback": 8,
        "max_dendrite_tries": 3,
    }
    train_config.write_text(
        yaml.safe_dump({"objective": {"metric": "validation_accuracy", "use_test": False}, "perforatedai": pai})
    )
    return {
        "source_checkpoint": str(checkpoint),
        "source_channels": 12,
        "widths": [10, 8, 6],
        "teacher_checkpoint": None,
        "data_config": str(data_config),
        "model_config": str(model_config),
        "train_config": str(train_config),
        "output_dir": str(tmp_path / "default-run"),
        "objective": {"metric": "validation_accuracy", "use_test": False},
        "perforatedai": pai,
        "pruning": {"method": "l1_filter"},
    }


def test_recipe_is_validation_only_and_dry_run(tmp_path, experiment_config):
    config = experiment_config
    result = run_experiment(config, dry_run=True, output_dir=tmp_path / "run")
    assert result["status"] == "planned"
    assert config["perforatedai"]["max_dendrites"] == -1
    assert config["perforatedai"]["switch_mode"] == "history"
    assert config["perforatedai"]["history_lookback"] == 8
    assert config["objective"]["use_test"] is False
    assert result["budget_enforced"] is False
    assert [candidate["width"] for candidate in result["candidates"]] == [10, 8, 6]
    assert all(candidate["baseline"]["deployed_params"] > 0 for candidate in result["candidates"])
    assert all(
        "prune_supervised" in candidate["baseline"]["checkpoint"]
        for candidate in result["candidates"]
    )


def test_recipe_rejects_teacher():
    config = load_config(CONFIG)
    config["teacher_checkpoint"] = "teacher.pt"
    with pytest.raises(ValueError, match="no-KD"):
        validate_config(config)


def test_dendritic_delta_is_against_matched_baseline():
    assert dendritic_delta(
        {"deployed_params": 120, "macs": 220},
        {"deployed_params": 100, "macs": 200},
    ) == {"parameter_delta": 20, "mac_delta": 20}


def test_pruning_curve_interpolation_refuses_extrapolation():
    points = [
        {"deployed_params": 100, "validation_accuracy": 0.7},
        {"deployed_params": 200, "validation_accuracy": 0.9},
    ]
    assert interpolate_pruning_curve(150, points) == pytest.approx(0.8)
    assert interpolate_pruning_curve(250, points) is None


def test_execution_dispatches_no_kd_cycle_from_each_exact_width(tmp_path, experiment_config):
    config = experiment_config
    calls = []

    def fake_cycle(
        checkpoint_path,
        data_cfg,
        model_cfg,
        train_cfg,
        save_name,
        **kwargs,
    ):
        width = model_cfg["channels"]
        calls.append((checkpoint_path, data_cfg, model_cfg, train_cfg, save_name, kwargs))
        return {
            "best_val_acc": 0.80 + width / 1000,
            "pai_best_val_acc": 0.79 + width / 1000,
            "base_params": 100 * width,
            "deployed_params": 100 * width + 10,
            "pai_deployed_params": 100 * width + 9,
            "cost": {"macs": 1000 * width + 20},
            "prune_finetune": {
                "best_val_acc": 0.78 + width / 1000,
                "checkpoint": str(tmp_path / f"c{width}.pt"),
                "reused": False,
                "distillation": None,
            },
            "resume": {"status": "skipped"},
        }

    report = run_experiment(
        config,
        output_dir=tmp_path / "run",
        resume=True,
        cycle_runner=fake_cycle,
    )

    assert report["status"] == "complete"
    assert len(calls) == 3
    for expected_width, call in zip([10, 8, 6], calls):
        _, _, model_cfg, train_cfg, _, kwargs = call
        assert model_cfg["channels"] == expected_width
        assert train_cfg["pruning"]["target_channels"] == expected_width
        assert train_cfg["perforatedai"]["module_ids"] == [
            ".blocks.3", ".gate_conv", ".fc"
        ]
        assert kwargs["teacher_checkpoint"] is None
        # Aggregate resume is tri-state: a resumed aggregate lets each cycle
        # auto-select its canonical sidecar.
        assert kwargs["resume"] is None
        assert kwargs["prune_finetune_candidate"] == f"sparknet_c{expected_width}"
    assert all(
        candidate["comparison"]["validation_accuracy_gain"] == pytest.approx(0.02)
        for candidate in report["candidates"]
    )


def test_refuses_to_replace_existing_report_without_resume(tmp_path, experiment_config):
    report_path = tmp_path / "run" / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("status: running\n")

    with pytest.raises(FileExistsError, match="pass resume=True"):
        run_experiment(experiment_config, dry_run=True, output_dir=tmp_path / "run")

    assert report_path.read_text() == "status: running\n"


def test_resume_keeps_completed_widths_and_only_runs_remaining(tmp_path, experiment_config):
    output_dir = tmp_path / "run"
    initial_calls = []

    def interrupted_cycle(*args, **kwargs):
        width = args[2]["channels"]
        initial_calls.append(width)
        assert kwargs["resume"] is False
        if width == 8:
            raise RuntimeError("simulated interruption")
        return _cycle_result(width, tmp_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_experiment(experiment_config, output_dir=output_dir, cycle_runner=interrupted_cycle)
    assert initial_calls == [10, 8]

    resumed_calls = []

    def resumed_cycle(*args, **kwargs):
        width = args[2]["channels"]
        resumed_calls.append(width)
        return _cycle_result(width, tmp_path)

    report = run_experiment(
        experiment_config, output_dir=output_dir, resume=True, cycle_runner=resumed_cycle
    )

    assert resumed_calls == [8, 6]
    assert report["status"] == "complete"
    assert [candidate["status"] for candidate in report["candidates"]] == ["complete"] * 3


def test_existing_completed_candidate_with_empty_comparison_is_rejected(
    tmp_path, experiment_config
):
    output_dir = tmp_path / "run"
    report = run_experiment(experiment_config, dry_run=True, output_dir=output_dir)
    candidate = report["candidates"][0]
    candidate["status"] = "complete"
    candidate["baseline"].update(
        {"validation_accuracy": 0.8, "deployed_params": 100, "macs": 200}
    )
    candidate["dendritic"].update(
        {"validation_accuracy": 0.8, "deployed_params": 90, "macs": 180}
    )
    candidate["comparison"] = {}
    report_path = output_dir / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))

    with pytest.raises(ValueError, match="invalid comparison"):
        run_experiment(experiment_config, dry_run=True, output_dir=output_dir, resume=True)


def _cycle_result(width, tmp_path):
    return {
        "best_val_acc": 0.80 + width / 1000,
        "pai_best_val_acc": 0.79 + width / 1000,
        "deployed_params": 100 * width + 10,
        "pai_deployed_params": 100 * width + 9,
        "cost": {"macs": 1000 * width + 20},
        "prune_finetune": {
            "best_val_acc": 0.78 + width / 1000,
            "checkpoint": str(tmp_path / f"c{width}.pt"),
            "reused": False,
            "distillation": None,
        },
        "resume": {"status": "skipped"},
    }
