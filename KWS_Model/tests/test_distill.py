from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import kws.optimize.distill as distill_module
from kws.optimize.distill import imc_distillation_losses


def test_imc_distillation_loss_matches_paper_weighting():
    student_logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    teacher_logits = torch.tensor([[1.5, -0.5], [0.0, 1.0]])
    student_features = torch.tensor([[0.5, 0.2], [0.1, 0.9]], requires_grad=True)
    teacher_features = torch.tensor([[0.4, 0.3], [0.3, 0.7]])
    labels = torch.tensor([0, 1])

    losses = imc_distillation_losses(
        student_logits,
        teacher_logits,
        student_features,
        teacher_features,
        labels,
        temperature=1.0,
        feature_weight=0.3,
        response_weight=0.1,
        classification_weight=0.6,
        label_smoothing=0.0,
    )

    expected = (
        0.3 * F.mse_loss(student_features, teacher_features)
        + 0.1
        * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )
        + 0.6 * F.cross_entropy(student_logits, labels)
    )
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert student_logits.grad is not None
    assert student_features.grad is not None


@pytest.mark.parametrize(
    ("temperature", "weights"),
    [
        (0.0, (0.3, 0.1, 0.6)),
        (1.0, (-0.1, 0.5, 0.6)),
        (1.0, (0.3, 0.3, 0.3)),
    ],
)
def test_imc_distillation_rejects_invalid_settings(temperature, weights):
    logits = torch.zeros(2, 2)
    features = torch.zeros(2, 3)
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        imc_distillation_losses(
            logits,
            logits,
            features,
            features,
            labels,
            temperature=temperature,
            feature_weight=weights[0],
            response_weight=weights[1],
            classification_weight=weights[2],
            label_smoothing=0.0,
        )


def test_distill_best_checkpoint_callback_uses_atomic_save_contract(
    tmp_path, monkeypatch,
):
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"teacher")
    out_checkpoint = tmp_path / "student-best.pt"

    class FakeTeacher:
        input_shape = (2, 2)
        num_classes = 2
        num_keywords = 1
        feature_dim = 2
        model_cfg = {"name": "teacher"}
        val_acc = 0.75

        def __init__(self, checkpoint_path, device):
            self.checkpoint_path = str(checkpoint_path)
            self.checkpoint_sha256 = "teacher-digest"

    class FakeStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    def fake_run_finetune(model, *args, on_best, **kwargs):
        on_best(model, 0.5)
        return SimpleNamespace(best_val_acc=0.5)

    monkeypatch.setattr(distill_module, "FrozenTeacher", FakeTeacher)
    monkeypatch.setattr(distill_module, "build_ds_cnn", lambda *args: FakeStudent())
    monkeypatch.setattr(
        distill_module,
        "build_datasets",
        lambda *args, **kwargs: (
            {"training": object(), "validation": object()},
            {"a": 0, "b": 1},
        ),
    )
    monkeypatch.setattr(distill_module, "build_data_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(distill_module, "run_finetune", fake_run_finetune)

    distill_module.distill(
        str(teacher_checkpoint),
        {"name": "student"},
        {"dataset": "tiny"},
        {
            "seed": 1,
            "augment": False,
            "label_smoothing": 0.0,
            "distillation": {},
        },
        out_checkpoint,
    )

    checkpoint = torch.load(out_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["val_acc"] == 0.5
    assert checkpoint["distillation"]["recipe_fingerprint"]
