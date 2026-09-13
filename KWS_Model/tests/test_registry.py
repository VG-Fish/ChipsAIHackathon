import pytest
import torch
from torch.utils.data import TensorDataset

import kws.train as train_module
from kws.data.splits import TRAIN, VAL
from kws.evaluate import load_model_from_checkpoint
from kws.models.ds_cnn import DSCNN
from kws.models.registry import (
    build_model,
    build_model_from_checkpoint,
    checkpoint_model_family,
    model_family,
)
from kws.models.sparknet import SparkNet

DS_CNN_CFG = {
    "name": "tiny_ds_cnn",
    "initial_channels": 4,
    "initial_kernel": 3,
    "initial_stride": 2,
    "block_channels": [4],
}
SPARKNET_CFG = {"family": "sparknet", "name": "tiny_sparknet", "channels": 4, "gate_channels": 8}


def test_config_without_family_builds_ds_cnn():
    assert model_family(DS_CNN_CFG) == "ds_cnn"
    assert isinstance(build_model(DS_CNN_CFG, (8, 12), 3), DSCNN)
    assert isinstance(build_model(SPARKNET_CFG, (8, 12), 3), SparkNet)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown model family"):
        build_model({**SPARKNET_CFG, "family": "matchboxnet"}, (8, 12), 3)


def test_checkpoint_family_accepts_either_spelling_and_rejects_disagreement():
    ported = {"model_family": "sparknet", "model_cfg": {"channels": 4, "gate_channels": 8}}
    assert checkpoint_model_family(ported) == "sparknet"
    assert checkpoint_model_family({"model_cfg": SPARKNET_CFG}) == "sparknet"
    assert checkpoint_model_family({"model_cfg": DS_CNN_CFG}) == "ds_cnn"
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_model_family({"model_family": "ds_cnn", "model_cfg": SPARKNET_CFG})

    model = build_model_from_checkpoint(
        {**ported, "input_shape": [8, 12], "num_classes": 3}
    )
    assert isinstance(model, SparkNet)


def test_train_writes_a_sparknet_checkpoint_that_evaluate_loads(tmp_path, monkeypatch):
    torch.manual_seed(0)
    input_shape = (8, 12)
    features = torch.randn(16, 1, *input_shape)
    labels = torch.arange(16) % 3
    datasets = {TRAIN: TensorDataset(features, labels), VAL: TensorDataset(features, labels)}
    label_map = {"a": 0, "_unknown_": 1, "_silence_": 2}
    monkeypatch.setattr(
        train_module, "build_datasets", lambda *args, **kwargs: (datasets, label_map)
    )
    monkeypatch.setattr(train_module, "get_device", lambda: torch.device("cpu"))

    train_cfg = {
        "augment": False,
        "epochs": 2,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 0.01,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_fraction": 0.0,
        "seed": 0,
    }
    checkpoint = tmp_path / "student.pt"
    result = train_module.train(
        {"target_keywords": ["a"]}, SPARKNET_CFG, train_cfg, checkpoint, stage="student",
    )
    assert "train_gate_sparsity" in result.history[-1]

    model, ckpt = load_model_from_checkpoint(str(checkpoint), torch.device("cpu"))
    assert isinstance(model, SparkNet)
    assert ckpt["model_family"] == "sparknet"
    assert tuple(ckpt["input_shape"]) == input_shape
    with torch.no_grad():
        assert model(features).shape == (16, 3)
