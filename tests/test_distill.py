import pytest
import torch
import torch.nn.functional as F

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
