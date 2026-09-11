import torch
import yaml

from kws.models.ds_cnn import build_ds_cnn
from kws.optimize.dendritic import (
    build_cycle_base,
    estimate_one_dendrite_params,
    read_pai_architecture_results,
)
from kws.optimize.dendritic_prune_loop import candidate_widths, judge_candidate


def test_cycle1_base_hits_expected_size_and_widths(tmp_path):
    source_cfg = {
        "name": "ds_cnn_xs",
        "initial_channels": 18,
        "initial_kernel": 5,
        "initial_stride": 2,
        "block_channels": [40, 40],
        "dropout": 0.2,
    }
    input_shape = (40, 98)
    source = build_ds_cnn(source_cfg, input_shape, num_classes=6)
    checkpoint_path = tmp_path / "xs.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "model_cfg": source_cfg,
            "input_shape": input_shape,
            "num_classes": 6,
        },
        checkpoint_path,
    )

    with open("configs/model/ds_cnn_xxs.yaml") as f:
        target_model_cfg = yaml.safe_load(f)
    model, checkpoint, model_cfg = build_cycle_base(
        str(checkpoint_path),
        keep_ratio=0.45,
        target_model_cfg=target_model_cfg,
    )

    assert checkpoint["model_cfg"]["block_channels"] == [40, 40]
    assert model_cfg["name"] == "ds_cnn_xxs"
    assert model_cfg["block_channels"] == [18, 18]
    assert sum(parameter.numel() for parameter in model.parameters()) == 1716
    assert estimate_one_dendrite_params(model) == 2946

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, *checkpoint["input_shape"]))
    assert output.shape == (2, checkpoint["num_classes"])


def test_read_pai_architecture_results_selects_best_deployable_row(tmp_path):
    run_dir = tmp_path / "pai_w18"
    run_dir.mkdir()
    (run_dir / "pai_w18_best_arch_scores.csv").write_text(
        "Param Counts,Max Valid Scores,Train\n"
        "1716,0.8895,0.69\n"
        "2946,0.9061,0.75\n"
    )

    accuracy, parameters = read_pai_architecture_results(str(run_dir))

    assert accuracy == 0.9061
    assert parameters == 2946


def test_pruning_widths_descend_to_configured_minimum():
    assert candidate_widths(18, 14, 1) == [18, 17, 16, 15, 14]
    assert candidate_widths(18, 12, 3) == [18, 15, 12]
    assert candidate_widths(18, 13, 3) == [18, 15, 13]


def test_pruning_stops_below_accuracy_floor():
    accepted = judge_candidate(0.901, 0.90, 0.906, None)
    degraded = judge_candidate(0.899, 0.90, 0.901, None)

    assert accepted.accepted
    assert not degraded.accepted
    assert "below" in degraded.reason


def test_optional_relative_drop_rule():
    decision = judge_candidate(0.902, 0.90, 0.910, 0.005)

    assert not decision.accepted
    assert "dropped" in decision.reason
