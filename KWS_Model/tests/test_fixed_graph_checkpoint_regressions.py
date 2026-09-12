import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from kws.data.splits import TRAIN, VAL
from kws.optimize import quantize_qat
from kws.optimize.cluster import (
    ClusterSpec,
    apply_weight_clustering,
    codebook_finetune,
)
from kws.utils.artifacts import ArtifactLayout


def _train_config() -> dict:
    return {
        "epochs": 1,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "warmup_fraction": 0.0,
        "label_smoothing": 0.0,
        "seed": 0,
        "augment": False,
    }


def _zero_validation(_model, _loader, _device, _criterion):
    return 1.0, 0.0


def test_cluster_best_is_a_loadable_model_checkpoint_not_latest_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("kws.train.evaluate_loss_acc", _zero_validation)
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    apply_weight_clustering(model, ClusterSpec(bits=2, min_weights=1))
    dataset = TensorDataset(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    output_dir = tmp_path / "cluster-run"

    result = codebook_finetune(
        model,
        loader,
        loader,
        torch.device("cpu"),
        _train_config(),
        epochs=1,
        output_dir=output_dir,
    )

    layout = ArtifactLayout(output_dir)
    best_path = layout.checkpoint_path("cluster", "codebook", "best")
    latest_path = layout.checkpoint_path("cluster", "codebook", "latest")
    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    latest = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=False,
    )
    run_id = yaml.safe_load((output_dir / "manifest.yaml").read_text())["run_id"]

    assert result["best_val_acc"] == 0.0
    assert best["kind"] == "kws_best_model"
    assert best["model_format"] == "cluster-parametrized-state-dict"
    assert best["best_metric_value"] == 0.0
    assert best["best_epoch"] == 1
    assert "optimizer_state_dict" not in best
    assert latest["kind"] == "kws_training_state"
    assert best["run_id"] == latest["run_id"] == run_id
    assert best_path != latest_path
    assert all(
        torch.equal(value, latest["best_model_state_dict"][name])
        for name, value in best["model_state_dict"].items()
    )
    model.load_state_dict(best["model_state_dict"], strict=True)


class _TinyQATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(4, 2)

    def forward(self, features):
        return self.classifier(self.flatten(features))


class _Cost:
    def as_dict(self):
        return {"parameters": 10}


def test_qat_persists_distinct_best_and_resumable_latest_on_zero_accuracy(
    tmp_path, monkeypatch
):
    dataset = TensorDataset(
        torch.randn(4, 1, 2, 2), torch.zeros(4, dtype=torch.long)
    )
    monkeypatch.setattr(
        quantize_qat,
        "build_datasets",
        lambda *_args, **_kwargs: ({TRAIN: dataset, VAL: dataset}, {"zero": 0}),
    )
    monkeypatch.setattr(
        quantize_qat,
        "build_data_loader",
        lambda value, _config, *, shuffle, generator=None: DataLoader(
            value, batch_size=2, shuffle=shuffle, generator=generator
        ),
    )
    monkeypatch.setattr("kws.train.evaluate_loss_acc", _zero_validation)
    monkeypatch.setattr(quantize_qat, "profile_model", lambda *_args, **_kwargs: _Cost())
    output_dir = tmp_path / "qat-run"

    result = quantize_qat.quantize_aware_distill_model(
        _TinyQATModel(),
        (2, 2),
        {},
        _train_config(),
        output_dir / "models" / "exported" / "requested-int8.pt",
        output_dir=output_dir,
    )

    layout = ArtifactLayout(output_dir)
    best_path = layout.checkpoint_path("quantize", "qat", "best")
    latest_path = layout.checkpoint_path("quantize", "qat", "latest")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    run_id = yaml.safe_load((output_dir / "manifest.yaml").read_text())["run_id"]

    assert result["best_val_acc"] == 0.0
    assert torch.jit.load(result["checkpoint"])
    assert result["best_checkpoint"] == str(best_path)
    assert result["latest_checkpoint"] == str(latest_path)
    assert best["kind"] == "kws_best_model"
    assert best["model_format"] == "prepared-qat-state-dict"
    assert best["best_metric_value"] == 0.0
    assert best["best_epoch"] == 1
    assert "optimizer_state_dict" not in best
    assert latest["kind"] == "kws_training_state"
    assert best["run_id"] == latest["run_id"] == run_id
    extra_files = {"run_id": ""}
    torch.jit.load(result["checkpoint"], _extra_files=extra_files)
    assert extra_files["run_id"].decode() == run_id
    assert best_path != latest_path
    assert all(
        torch.equal(value, latest["best_model_state_dict"][name])
        for name, value in best["model_state_dict"].items()
    )
    assert latest["stage_specific_state"]["qat_backend"] == result["backend"]
    assert any("activation_post_process" in name for name in latest["model_state_dict"])
    result["fake_quantized_model"].load_state_dict(
        best["model_state_dict"], strict=True
    )
