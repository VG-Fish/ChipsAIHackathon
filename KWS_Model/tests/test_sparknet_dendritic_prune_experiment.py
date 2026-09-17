from pathlib import Path

import pytest
import torch
import yaml

from kws.models.sparknet import SparkNet
from kws.optimize.sparknet_dendritic_prune_experiment import (
    dendritic_delta,
    interpolate_pruning_curve,
    load_config,
    report_metrics,
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


def test_from_scratch_source_can_run_an_identity_width_cycle(
    tmp_path, experiment_config
):
    """A trained C12 source must reach PAI without being pruned to C11 first."""
    experiment_config["widths"] = [12]
    experiment_config["pruning"] = {"method": "identity"}

    report = run_experiment(
        experiment_config,
        dry_run=True,
        output_dir=tmp_path / "identity-run",
    )

    candidate = report["candidates"][0]
    assert candidate["width"] == 12
    assert candidate["prune_fraction"] == 0.0
    assert candidate["baseline"]["deployed_params"] == report["source"][
        "deployed_params"
    ]


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
    config["seed"] = 7
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
        assert kwargs["seed"] == 7
        assert train_cfg["seed"] == 7
        # Aggregate resume is tri-state: a resumed aggregate lets each cycle
        # auto-select its canonical sidecar.
        assert kwargs["resume"] is None
        assert kwargs["prune_finetune_candidate"] == f"sparknet_c{expected_width}"
    assert all(
        candidate["comparison"]["validation_accuracy_gain"] == pytest.approx(0.02)
        for candidate in report["candidates"]
    )
    assert report["seed"] == 7


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


def test_resume_rejects_a_different_or_unrecorded_seed(tmp_path, experiment_config):
    output_dir = tmp_path / "run"
    experiment_config["seed"] = 3
    run_experiment(
        experiment_config,
        output_dir=output_dir,
        cycle_runner=lambda *args, **kwargs: _cycle_result(
            args[2]["channels"], tmp_path
        ),
    )

    experiment_config["seed"] = 4
    with pytest.raises(ValueError, match="different or unrecorded seed"):
        run_experiment(
            experiment_config,
            dry_run=True,
            output_dir=output_dir,
            resume=True,
        )

    report_path = output_dir / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    report = yaml.safe_load(report_path.read_text())
    report.pop("seed")
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    with pytest.raises(ValueError, match="different or unrecorded seed"):
        run_experiment(
            experiment_config,
            dry_run=True,
            output_dir=output_dir,
            resume=True,
        )


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


def _retarget_module_ids(config, module_ids):
    """Point both the experiment and its train config at one placement.

    ``_load_inputs`` requires the two PAI blocks to match exactly, so a test
    that changes the placement has to rewrite the train config file too.
    """
    pai = dict(config["perforatedai"])
    pai["module_ids"] = list(module_ids)
    config["perforatedai"] = pai
    Path(config["train_config"]).write_text(
        yaml.safe_dump(
            {
                "objective": {"metric": "validation_accuracy", "use_test": False},
                "perforatedai": pai,
            }
        )
    )
    return config


def test_backbone_and_empty_placements_are_accepted(experiment_config):
    """Backbone convolutions are legal targets; ``[]`` is the control arm."""
    assert validate_config(_retarget_module_ids(experiment_config, []))[
        "perforatedai"
    ]["module_ids"] == []
    for module_id in (
        ".blocks.3.pointwise",
        ".blocks.0.depthwise",
        ".blocks.2",
        ".gate_conv",
        ".fc",
    ):
        validate_config(_retarget_module_ids(experiment_config, [module_id]))
    with pytest.raises(ValueError, match="unsupported"):
        validate_config(_retarget_module_ids(experiment_config, [".blocks.4.pointwise"]))
    with pytest.raises(ValueError, match="unsupported"):
        validate_config(_retarget_module_ids(experiment_config, [".gate_bn"]))


def test_pointwise_dendrites_are_cheaper_than_a_classifier_dendrite(
    tmp_path, experiment_config
):
    """Two C12-derived pointwise copies cost far less than one ``.fc`` copy."""
    pointwise = run_experiment(
        _retarget_module_ids(
            experiment_config, [".blocks.3.pointwise", ".blocks.2.pointwise"]
        ),
        dry_run=True,
        output_dir=tmp_path / "pointwise",
    )
    classifier = run_experiment(
        _retarget_module_ids(experiment_config, [".fc"]),
        dry_run=True,
        output_dir=tmp_path / "classifier",
    )
    control = run_experiment(
        _retarget_module_ids(experiment_config, []),
        dry_run=True,
        output_dir=tmp_path / "control",
    )

    for pointwise_candidate, classifier_candidate, control_candidate in zip(
        pointwise["candidates"], classifier["candidates"], control["candidates"]
    ):
        pointwise_cost = pointwise_candidate["dendritic"]["one_dendrite_cost_projection"]
        classifier_cost = classifier_candidate["dendritic"]["one_dendrite_cost_projection"]
        control_cost = control_candidate["dendritic"]["one_dendrite_cost_projection"]
        width = pointwise_candidate["width"]
        assert pointwise_cost["copied_params_per_dendrite"] == 2 * width * width
        assert pointwise_cost["copied_params_per_dendrite"] < classifier_cost[
            "copied_params_per_dendrite"
        ]
        # The control arm copies nothing, so its projection is the base cost.
        assert tuple(control_cost["module_names"]) == ()
        assert control_cost["projected_params"] == control_cost["base_params"]
        assert control_cost["projected_macs"] == control_cost["base_macs"]


def test_headline_gain_switches_basis_with_the_zero_dendrite_control():
    pai = {"validation_accuracy": 0.92, "deployed_params": 120, "macs": 220}
    baseline = {"validation_accuracy": 0.80, "deployed_params": 100, "macs": 200}

    controlled = report_metrics(
        pai,
        baseline,
        zero_dendrite_accuracy=0.915,
        pai_search_accuracy=0.919,
    )
    uncontrolled = report_metrics(pai, baseline)

    # The headline is always the final end-to-end delta; a PAI architecture row
    # is not a matched no-dendrite counterfactual.
    assert controlled["validation_accuracy_gain"] == pytest.approx(0.12)
    assert controlled["validation_accuracy_gain_basis"] == "prune_finetune_baseline"
    assert controlled["final_validation_accuracy_gain_vs_pai_zero_architecture"] == pytest.approx(0.005)
    assert controlled["pai_search_accuracy_gain"] == pytest.approx(0.004)
    assert controlled["pai_search_accuracy_gain_vs_pai_zero_architecture"] == pytest.approx(0.004)
    # Without a PAI-row comparison the headline still keeps its baseline meaning.
    assert uncontrolled["validation_accuracy_gain"] == pytest.approx(0.12)
    assert uncontrolled["validation_accuracy_gain_basis"] == "prune_finetune_baseline"
    for metrics in (controlled, uncontrolled):
        assert metrics["validation_accuracy_gain_vs_prune_finetune"] == pytest.approx(0.12)
        # Cost deltas stay measured against the pruned baseline either way.
        assert metrics["parameter_delta"] == 20
        assert metrics["mac_delta"] == 20
        assert metrics["parameter_growth_fraction"] == pytest.approx(0.2)
        assert metrics["mac_growth_fraction"] == pytest.approx(0.1)


def test_cycle_zero_dendrite_row_becomes_the_reported_control(tmp_path, experiment_config):
    def fake_cycle(checkpoint_path, data_cfg, model_cfg, train_cfg, save_name, **kwargs):
        width = model_cfg["channels"]
        result = _cycle_result(width, tmp_path)
        result["zero_dendrite_val_acc"] = 0.795 + width / 1000
        result["zero_dendrite_params"] = 100 * width - 5
        return result

    report = run_experiment(
        experiment_config, output_dir=tmp_path / "run", cycle_runner=fake_cycle
    )

    for candidate in report["candidates"]:
        width = candidate["width"]
        dendritic = candidate["dendritic"]
        comparison = candidate["comparison"]
        assert dendritic["zero_dendrite_validation_accuracy"] == pytest.approx(
            0.795 + width / 1000
        )
        assert dendritic["zero_dendrite_params"] == 100 * width - 5
        assert comparison["validation_accuracy_gain_basis"] == "prune_finetune_baseline"
        assert comparison["validation_accuracy_gain"] == pytest.approx(0.02)
        assert comparison["final_validation_accuracy_gain_vs_pai_zero_architecture"] == pytest.approx(0.005)
        assert comparison["validation_accuracy_gain_vs_prune_finetune"] == pytest.approx(0.02)
        assert comparison["pai_search_accuracy_gain"] == pytest.approx(-0.005)
        assert comparison["pai_search_accuracy_gain_vs_prune_finetune"] == pytest.approx(0.01)

    # A report written with the control must survive its own resume check.
    resumed = run_experiment(
        experiment_config,
        dry_run=True,
        output_dir=tmp_path / "run",
        resume=True,
    )
    assert [candidate["status"] for candidate in resumed["candidates"]] == ["complete"] * 3
