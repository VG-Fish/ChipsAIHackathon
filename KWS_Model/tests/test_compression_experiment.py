import pytest
import torch
import yaml

from kws.models.ds_cnn import DSCNN
from kws.optimize.compression_experiment import (
    CompressionBudget,
    _finalize_comparisons,
    resolve_keep_ratios,
    run_compression_experiment,
)
from kws.optimize.dendritic_config import (
    PAI_EFFECTIVELY_UNLIMITED_DENDRITES,
    UNLIMITED_DENDRITES,
    normalize_module_ids,
    pai_runtime_dendrite_limit,
    placement_module_names,
    projected_dendrite_count,
    project_dendritic_cost,
    validate_max_dendrites,
)


def _model() -> DSCNN:
    return DSCNN(
        input_shape=(8, 8),
        num_classes=3,
        initial_channels=4,
        initial_kernel=3,
        initial_stride=1,
        block_channels=[6, 6],
        dropout=0.0,
    )


def test_width_generation_supports_absolute_multiplier_and_sweep_forms():
    assert resolve_keep_ratios([40, 40], {"widths": [18, 10]}) == [0.45, 0.25]
    assert resolve_keep_ratios(
        [40, 40], {"width_multipliers": [0.5, 0.25]}
    ) == [0.5, 0.25]
    assert resolve_keep_ratios(
        [40, 40], {"sweep": {"start": 18, "minimum": 13, "step": 3}}
    ) == [0.45, 0.375, 0.325]


def test_exact_placement_ids_are_normalized_validated_and_non_overlapping():
    model = _model()
    assert normalize_module_ids(["fc", ".blocks.0"]) == (".fc", ".blocks.0")
    assert placement_module_names(
        model, {"conversion": "module_ids", "module_ids": ["blocks.1"]}
    ) == ("blocks.1",)

    try:
        normalize_module_ids(["blocks.0", "blocks.0.pointwise"])
    except ValueError as error:
        assert "may not overlap" in str(error)
    else:
        raise AssertionError("nested placement IDs should be rejected")


def test_all_feature_blocks_placement_covers_every_block_and_classifier():
    assert placement_module_names(
        _model(), {"conversion": "blocks_and_linear"}
    ) == ("blocks.0", "blocks.1", "fc")


def test_projection_charges_selected_copy_and_residual_against_budget():
    model = _model().eval()
    fc = project_dendritic_cost(
        model, (8, 8), {"conversion": "fc_only"}, max_dendrites=1
    )
    block = project_dendritic_cost(
        model,
        (8, 8),
        {"conversion": "module_ids", "module_ids": ["blocks.0"]},
        max_dendrites=1,
    )

    assert fc.projected_params > fc.base_params
    assert fc.projected_macs > fc.base_macs
    assert block.copied_macs_per_dendrite > fc.copied_macs_per_dendrite
    budget = CompressionBudget(fc.projected_params, fc.projected_macs)
    assert budget.violations(params=fc.projected_params, macs=fc.projected_macs) == []
    assert budget.violations(
        params=block.projected_params, macs=block.projected_macs
    )


def test_projection_charges_triangular_skip_cost_for_multiple_dendrites():
    projection = project_dendritic_cost(
        _model().eval(), (8, 8), {"conversion": "fc_only"}, max_dendrites=3
    )

    assert projection.scale_connections == 6
    assert projection.projected_params - projection.base_params == (
        3 * projection.copied_params_per_dendrite
        + 6 * projection.residual_params_per_dendrite
    )
    assert projection.projected_macs - projection.base_macs == (
        3 * projection.copied_macs_per_dendrite
        + 6 * projection.residual_macs_per_dendrite
    )


def test_unlimited_dendrites_uses_one_dendrite_for_cost_admission():
    assert validate_max_dendrites(UNLIMITED_DENDRITES) == -1
    assert projected_dendrite_count(UNLIMITED_DENDRITES) == 1
    assert (
        pai_runtime_dendrite_limit(UNLIMITED_DENDRITES)
        == PAI_EFFECTIVELY_UNLIMITED_DENDRITES
    )
    assert pai_runtime_dendrite_limit(3) == 3
    for invalid in (True, False, 0, -2, 1.5, "3"):
        with pytest.raises(ValueError, match="positive integer or -1"):
            validate_max_dendrites(invalid)


def test_comparison_matches_conventional_models_at_no_greater_final_cost():
    records = [
        {
            "label": "conventional/w8",
            "arm": "conventional",
            "status": "complete",
            "budget_admitted": True,
            "accuracy": 0.80,
            "costs": {"deployed_params": 800, "macs": 8000},
        },
        {
            "label": "conventional/w10",
            "arm": "conventional",
            "status": "complete",
            "budget_admitted": True,
            "accuracy": 0.84,
            "costs": {"deployed_params": 1000, "macs": 10000},
        },
        {
            "label": "dendritic/w8_fc",
            "arm": "dendritic",
            "status": "complete",
            "budget_admitted": True,
            "accuracy": 0.86,
            "costs": {"deployed_params": 900, "macs": 9000},
        },
    ]

    comparison = _finalize_comparisons(records, ("deployed_params", "macs"))

    assert comparison["best_accuracy_delta"] == pytest.approx(0.02)
    matched = comparison["matched_budget_comparisons"][0]
    assert matched["dendritic"] == "dendritic/w8_fc"
    assert matched["conventional"] == "conventional/w8"
    assert matched["accuracy_delta"] == pytest.approx(0.06)
    assert {point["label"] for point in comparison["frontier"]} == {
        "conventional/w8",
        "dendritic/w8_fc",
    }


def test_dry_run_writes_a_plan_without_importing_perforatedai(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    model = _model()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "name": "tiny",
                "initial_channels": 4,
                "initial_kernel": 3,
                "initial_stride": 1,
                "block_channels": [6, 6],
                "dropout": 0.0,
            },
            "input_shape": (8, 8),
            "num_classes": 3,
            "num_keywords": 1,
        },
        source,
    )
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text("target_keywords: [yes]\n", encoding="utf-8")
    train_path.write_text("seed: 0\nperforatedai: {max_dendrites: 1}\n", encoding="utf-8")
    output = tmp_path / "run"
    config = {
        "source_checkpoint": str(source),
        "teacher_checkpoint": str(tmp_path / "not-needed-in-dry-run.pt"),
        "data_config": str(data_path),
        "train_config": str(train_path),
        "output_dir": str(output),
        "budget": {"max_deployed_params": 10000, "max_macs": 1000000},
        "backbones": {"widths": [4]},
        "placements": [{"name": "classifier", "conversion": "fc_only"}],
        "perforatedai": {
            "enabled": True,
            "on_unavailable": "error",
            "max_dendrites": -1,
        },
    }

    def reject_import(name):
        raise AssertionError(f"dry-run imported {name}")

    monkeypatch.setattr(
        "kws.optimize.compression_experiment.importlib.import_module", reject_import
    )
    report = run_compression_experiment(config, dry_run=True)

    assert report["status"] == "planned"
    assert [candidate["arm"] for candidate in report["candidates"]] == [
        "conventional",
        "dendritic",
    ]
    assert all(candidate["accuracy"] is None for candidate in report["candidates"])
    dendritic = report["candidates"][1]
    assert report["max_dendrites"] == -1
    assert dendritic["perforatedai"]["max_dendrites"] == -1
    assert dendritic["cost_projection"]["max_dendrites"] == 1
    assert dendritic["cost_projection"]["configured_max_dendrites"] == -1
    assert dendritic["cost_projection"]["basis"] == "one_dendrite_lower_bound"
    assert dendritic["budget_admission_basis"] == "one_dendrite_lower_bound"
    persisted = yaml.safe_load(
        (output / "reports" / "compression_experiment.yaml").read_text()
    )
    assert persisted["dry_run"] is True


def test_unavailable_pai_can_skip_cleanly_after_conventional_arm(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pt"
    model = _model()
    model_cfg = {
        "name": "tiny",
        "initial_channels": 4,
        "initial_kernel": 3,
        "initial_stride": 1,
        "block_channels": [6, 6],
        "dropout": 0.0,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "input_shape": (8, 8),
            "num_classes": 3,
            "num_keywords": 1,
        },
        source,
    )
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text("target_keywords: [yes]\n", encoding="utf-8")
    train_path.write_text("seed: 0\nperforatedai: {max_dendrites: 1}\n", encoding="utf-8")
    output = tmp_path / "run"
    config = {
        "source_checkpoint": str(source),
        "teacher_checkpoint": str(tmp_path / "teacher.pt"),
        "data_config": str(data_path),
        "train_config": str(train_path),
        "output_dir": str(output),
        "budget": {"max_deployed_params": 10000, "max_macs": 1000000},
        "backbones": {"widths": [4]},
        "placements": [{"name": "classifier", "conversion": "fc_only"}],
        "perforatedai": {"enabled": True, "on_unavailable": "skip"},
    }

    best_path = output / "models" / "checkpoints" / "sparsity" / "w4" / "prune_kd" / "best.pt"

    def fake_prune(*_args, **_kwargs):
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"stand-in")
        return {"best_val_acc": 0.75, "checkpoint": str(best_path)}

    monkeypatch.setattr(
        "kws.optimize.compression_experiment.prune_and_fine_tune", fake_prune
    )
    monkeypatch.setattr(
        "kws.optimize.compression_experiment._profile_checkpoint",
        lambda *_args, **_kwargs: {"deployed_params": 250, "macs": 2500},
    )

    def unavailable(_name):
        raise ModuleNotFoundError("perforatedai is not installed")

    monkeypatch.setattr(
        "kws.optimize.compression_experiment.importlib.import_module", unavailable
    )
    report = run_compression_experiment(config)

    conventional, dendritic = report["candidates"]
    assert conventional["status"] == "complete"
    assert conventional["accuracy"] == 0.75
    assert dendritic["status"] == "skipped_pai_unavailable"
    assert "ModuleNotFoundError" in dendritic["skip_reason"]
    assert report["status"] == "complete"
