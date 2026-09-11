import yaml
import torch

from kws.export.to_onnx import export_to_onnx
from kws.models.ds_cnn import build_ds_cnn


def test_onnx_export_matches_pytorch(tmp_path):
    with open("configs/model/ds_cnn_xs.yaml") as f:
        model_cfg = yaml.safe_load(f)

    input_shape = (40, 98)
    num_classes = 6
    model = build_ds_cnn(model_cfg, input_shape, num_classes)
    model.eval()

    checkpoint_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": model_cfg,
        "input_shape": input_shape,
        "num_classes": num_classes,
    }, checkpoint_path)

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(str(checkpoint_path), str(onnx_path), atol=1e-4)
    assert onnx_path.exists()
