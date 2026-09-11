import glob

import pytest
import torch
import yaml

from kws.models.ds_cnn import build_ds_cnn

MODEL_CONFIGS = sorted(glob.glob("configs/model/ds_cnn_*.yaml"))


@pytest.mark.parametrize("config_path", MODEL_CONFIGS)
def test_forward_pass_shape(config_path):
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    input_shape = (40, 98)
    num_classes = 6
    model = build_ds_cnn(model_cfg, input_shape, num_classes)
    model.eval()

    x = torch.randn(2, 1, *input_shape)
    with torch.no_grad():
        out = model(x)
        encoded = model.forward_features(x)
        out_from_features = model.classify_features(encoded)

    assert out.shape == (2, num_classes)
    assert torch.isfinite(out).all()
    assert encoded.shape == (2, model.fc.in_features)
    assert torch.equal(out, out_from_features)


@pytest.mark.parametrize("config_path", MODEL_CONFIGS)
def test_no_dynamic_pool(config_path):
    """AvgPool2d with a fixed kernel size (not AdaptiveAvgPool2d) is required for
    clean ONNX export and dendrite-hook compatibility -- see plan §Model Architecture.
    """
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    model = build_ds_cnn(model_cfg, (40, 98), 6)
    assert isinstance(model.pool, torch.nn.AvgPool2d)
    assert not isinstance(model.pool, torch.nn.AdaptiveAvgPool2d)
