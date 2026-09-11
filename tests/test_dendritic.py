import torch
import yaml

from kws.models.ds_cnn import build_ds_cnn
from kws.optimize.dendritic import build_cycle_base, estimate_one_dendrite_params


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
