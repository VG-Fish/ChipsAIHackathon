import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kws.optimize.kd import DistillationCriterion, KDWeights
from kws.train import run_finetune
from kws.utils.checkpointing import MetricsRecorder
from kws.utils.seed import set_seed


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TinyFeatureClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(2, 2)
        self.fc = torch.nn.Linear(2, 2)

    def forward_features(self, x):
        return self.features(x)

    def classify_features(self, features):
        return self.fc(features)

    def forward(self, x):
        return self.classify_features(self.forward_features(x))


class TinyTeacher:
    feature_dim = 3

    def __call__(self, inputs):
        teacher_features = torch.cat((inputs, inputs[:, :1]), dim=1)
        return teacher_features[:, :2], teacher_features


def _loaders(seed):
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(features, labels)
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    return (
        DataLoader(dataset, batch_size=2, shuffle=True, generator=train_generator),
        DataLoader(dataset, batch_size=2, shuffle=False, generator=val_generator),
    )


def _cfg():
    return {
        "epochs": 4,
        "batch_size": 2,
        "label_smoothing": 0.0,
        "lr": 0.05,
        "weight_decay": 0.0,
        "warmup_fraction": 0.0,
        "seed": 7,
    }


def _run(model, recorder, latest, *, resume_state=None, stop_after=None, run_id=None):
    train_loader, val_loader = _loaders(7)
    def on_epoch_end(record):
        if stop_after is not None and record["epoch"] == stop_after:
            raise RuntimeError("simulated interruption")
    return run_finetune(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        _cfg(),
        recorder=recorder,
        latest_path=latest,
        resume_state=resume_state,
        stage="train",
        phase="tiny",
        recipe={"model": "tiny", "train": _cfg()},
        on_epoch_end=on_epoch_end,
        run_id=run_id,
    )


def test_resume_matches_uninterrupted_cpu_run(tmp_path):
    set_seed(123)
    full_model = TinyClassifier()
    full_recorder = MetricsRecorder(tmp_path / "full.jsonl", stage="train", phase="tiny")
    full = _run(full_model, full_recorder, tmp_path / "full.latest.pt")
    full_next = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    set_seed(123)
    interrupted_model = TinyClassifier()
    interrupted_recorder = MetricsRecorder(tmp_path / "split.jsonl", stage="train", phase="tiny")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(
            interrupted_model,
            interrupted_recorder,
            tmp_path / "split.latest.pt",
            stop_after=2,
        )
    state = torch.load(tmp_path / "split.latest.pt", map_location="cpu", weights_only=False)

    set_seed(999)
    resumed_model = TinyClassifier()
    resumed_recorder = MetricsRecorder(tmp_path / "split.jsonl", stage="train", phase="tiny")
    resumed = _run(
        resumed_model,
        resumed_recorder,
        tmp_path / "split.latest.pt",
        resume_state=state,
    )
    resumed_next = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert resumed.completed_epoch == full.completed_epoch == 4
    assert resumed.global_step == full.global_step
    assert len(resumed_recorder.records) == 4
    for left, right in zip(full.history, resumed.history):
        assert {k: v for k, v in left.items() if k != "elapsed_seconds"} == {
            k: v for k, v in right.items() if k != "elapsed_seconds"
        }
    for name, tensor in full_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name])
    assert full_next == resumed_next


def test_legacy_best_only_checkpoint_is_rejected_for_resume(tmp_path):
    from kws.utils.checkpointing import require_training_state

    with pytest.raises(ValueError, match="not a resumable training-state"):
        require_training_state({"model_state_dict": {}}, source="best.pt")


def test_resume_rejects_a_checkpoint_from_another_manifest_run(tmp_path):
    latest = tmp_path / "latest.pt"
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl", stage="train", phase="tiny")
    _run(TinyClassifier(), recorder, latest, run_id="manifest-run")
    state = torch.load(latest, map_location="cpu", weights_only=False)

    with pytest.raises(ValueError, match="expected 'other-run'"):
        _run(
            TinyClassifier(),
            MetricsRecorder(tmp_path / "metrics.jsonl", stage="train", phase="tiny"),
            latest,
            resume_state=state,
            run_id="other-run",
        )


def test_fresh_run_refuses_to_append_to_existing_metrics(tmp_path):
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl", stage="train", phase="tiny")
    recorder.append({"epoch": 1, "val_acc": 0.5})

    with pytest.raises(ValueError, match="metrics already contain"):
        _run(TinyClassifier(), recorder, tmp_path / "latest.pt")


def test_kd_adapter_is_checkpointed_and_restored(tmp_path):
    config = {**_cfg(), "epochs": 1}
    recipe = {"model": "tiny-feature", "train": config, "kd": KDWeights().as_dict()}
    model = TinyFeatureClassifier()
    kd = DistillationCriterion(
        TinyTeacher(), KDWeights(), 0.0, torch.device("cpu"),
        student_feature_dim=2,
    )
    train_loader, val_loader = _loaders(7)
    latest = tmp_path / "kd.latest.pt"

    run_finetune(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        config,
        kd=kd,
        latest_path=latest,
        stage="student",
        phase="distill",
        recipe=recipe,
    )
    state = torch.load(latest, map_location="cpu", weights_only=False)
    saved_adapter = state["stage_specific_state"]["kd_state_dict"]
    assert saved_adapter["adapter_state_dict"] is not None

    resumed_model = TinyFeatureClassifier()
    resumed_kd = DistillationCriterion(
        TinyTeacher(), KDWeights(), 0.0, torch.device("cpu"),
        student_feature_dim=2,
    )
    train_loader, val_loader = _loaders(7)
    result = run_finetune(
        resumed_model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        config,
        kd=resumed_kd,
        latest_path=latest,
        resume_state=state,
        stage="student",
        phase="distill",
        recipe=recipe,
    )

    assert result.completed_epoch == 1
    assert torch.equal(resumed_kd.adapter.weight, kd.adapter.weight)
