import pytest
import torch
import torch.nn.functional as F
from typing import Any, cast

from kws.models.ds_cnn import DSCNN
from kws.optimize.kd import (
    DistillationCriterion,
    KDWeights,
    distillation_diagnostics,
    distillation_losses,
    supports_pooled_features,
)


def test_weights_must_form_a_convex_mix():
    KDWeights(feature_weight=0.3, response_weight=0.1, classification_weight=0.6)
    with pytest.raises(ValueError):
        KDWeights(feature_weight=0.5, response_weight=0.5, classification_weight=0.5)
    with pytest.raises(ValueError):
        KDWeights(feature_weight=-0.1, response_weight=0.5, classification_weight=0.6)
    with pytest.raises(ValueError):
        KDWeights(temperature=0.0)


def test_dropping_features_renormalizes_instead_of_shrinking_the_loss():
    weights = KDWeights(feature_weight=0.3, response_weight=0.1, classification_weight=0.6)
    reduced = weights.without_features()

    assert reduced.feature_weight == 0.0
    assert reduced.response_weight + reduced.classification_weight == pytest.approx(1.0)
    # The two survivors keep their relative proportions (1:6).
    assert reduced.classification_weight / reduced.response_weight == pytest.approx(6.0)


def test_dropping_features_is_refused_when_it_carries_all_the_weight():
    with pytest.raises(ValueError):
        KDWeights(feature_weight=1.0, response_weight=0.0, classification_weight=0.0).without_features()


def test_feature_loss_is_required_when_it_is_weighted():
    logits = torch.zeros(2, 3)
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="without_features"):
        distillation_losses(
            logits, logits, labels,
            weights=KDWeights(), label_smoothing=0.0,
        )


def test_mismatched_feature_widths_are_rejected_rather_than_broadcast():
    logits = torch.zeros(2, 3)
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="after projection"):
        distillation_losses(
            logits, logits, labels,
            weights=KDWeights(), label_smoothing=0.0,
            student_features=torch.zeros(2, 4),
            teacher_features=torch.zeros(2, 8),
        )


def test_feature_less_losses_match_a_hand_computed_two_term_objective():
    student = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    teacher = torch.tensor([[1.5, -0.5], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = KDWeights(feature_weight=0.3, response_weight=0.1,
                        classification_weight=0.6).without_features()

    losses = distillation_losses(
        student, teacher, labels, weights=weights, label_smoothing=0.0,
    )
    expected = (
        weights.response_weight
        * F.kl_div(F.log_softmax(student, dim=1), F.softmax(teacher, dim=1),
                   reduction="batchmean")
        + weights.classification_weight * F.cross_entropy(student, labels)
    )
    assert torch.allclose(losses["total"], expected)
    assert losses["feature"].item() == 0.0


def test_temperature_scaling_squares_back_into_the_response_gradient():
    student = torch.tensor([[2.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])
    weights = KDWeights(temperature=4.0, feature_weight=0.0,
                        response_weight=1.0, classification_weight=0.0)

    losses = distillation_losses(student, teacher, labels, weights=weights, label_smoothing=0.0)
    expected = F.kl_div(
        F.log_softmax(student / 4.0, dim=1), F.softmax(teacher / 4.0, dim=1),
        reduction="batchmean",
    ) * 16.0
    assert torch.allclose(losses["response"], expected)


def test_feature_support_is_probed_on_the_live_object():
    model = DSCNN(input_shape=(40, 98), num_classes=6, initial_channels=18,
                  initial_kernel=5, initial_stride=2, block_channels=[18, 18], dropout=0.2)
    assert supports_pooled_features(model, (40, 98))
    assert not supports_pooled_features(torch.nn.Linear(4, 4), (40, 98))


def test_probing_leaves_training_mode_untouched():
    model = DSCNN(input_shape=(40, 98), num_classes=6, initial_channels=18,
                  initial_kernel=5, initial_stride=2, block_channels=[18, 18], dropout=0.2)
    model.train()
    supports_pooled_features(model, (40, 98))
    assert model.training


class _TeacherDescription:
    feature_dim = 5


def test_distillation_criterion_state_round_trips_feature_adapter():
    weights = KDWeights()
    source = DistillationCriterion(
        cast(Any, _TeacherDescription()), weights, 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    with torch.no_grad():
        source_adapter = source.adapter
        assert source_adapter is not None
        source_adapter.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))

    restored = DistillationCriterion(
        cast(Any, _TeacherDescription()), weights, 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    restored.load_state_dict(source.state_dict(), strict=True)

    restored_adapter = restored.adapter
    source_adapter = source.adapter
    assert restored_adapter is not None and source_adapter is not None
    assert torch.equal(restored_adapter.weight, source_adapter.weight)


def test_distillation_criterion_rejects_missing_required_adapter_state():
    criterion = DistillationCriterion(
        cast(Any, _TeacherDescription()), KDWeights(), 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    state = criterion.state_dict()
    state["adapter_state_dict"] = None

    with pytest.raises(RuntimeError, match="missing the feature adapter"):
        criterion.load_state_dict(state, strict=True)


@pytest.mark.parametrize("temperature,response_weight", [(1.0, 1 / 7), (2.0, 0.5), (4.0, 0.8)])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_diagnostics_match_teacher_probabilities_and_autograd(
    temperature, response_weight, label_smoothing, monkeypatch,
):
    student = torch.tensor([[1.0, -0.5, 0.2], [-0.8, 1.4, 0.6]], requires_grad=True)
    teacher = torch.tensor([[2.0, 0.3, -0.9], [-1.0, 0.5, 1.8]], requires_grad=True)
    labels = torch.tensor([0, 1])
    weights = KDWeights(
        temperature=temperature, feature_weight=0.0,
        response_weight=response_weight, classification_weight=1 - response_weight,
    )
    losses = distillation_losses(
        student, teacher, labels, weights=weights, label_smoothing=label_smoothing,
    )
    assert set(losses) == {"total", "feature", "response", "classification"}
    response_gradient = torch.autograd.grad(
        weights.response_weight * losses["response"], student, retain_graph=True,
    )[0].flatten()
    classification_gradient = torch.autograd.grad(
        weights.classification_weight * losses["classification"], student,
    )[0].flatten()

    def forbid_sync(*args, **kwargs):
        raise AssertionError("diagnostics must not synchronize device to CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbid_sync)
        patch.setattr(torch.Tensor, "cpu", forbid_sync)
        metrics = distillation_diagnostics(
            student, teacher, labels, weights=weights, label_smoothing=label_smoothing,
        )
    probabilities = teacher.detach().softmax(dim=1)
    assert metrics["teacher_accuracy"].item() == 0.5
    torch.testing.assert_close(metrics["teacher_confidence"], probabilities.max(dim=1).values.mean())
    torch.testing.assert_close(
        metrics["teacher_true_class_probability"], probabilities[torch.arange(2), labels].mean(),
    )
    torch.testing.assert_close(
        metrics["teacher_entropy_nats"], -(probabilities * probabilities.log()).sum(dim=1).mean(),
    )
    torch.testing.assert_close(
        metrics["kd_logit_grad_norm_ratio"], response_gradient.norm() / classification_gradient.norm(),
    )
    torch.testing.assert_close(
        metrics["kd_logit_grad_cosine"],
        F.cosine_similarity(response_gradient, classification_gradient, dim=0),
    )
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
    assert all(not value.requires_grad and value.grad_fn is None for value in metrics.values())
    assert all(value.device == student.device for value in metrics.values())


@pytest.mark.parametrize("response_weight,classification_weight", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
def test_diagnostics_stay_finite_with_zero_gradients_or_saturated_probabilities(
    response_weight, classification_weight,
):
    weights = KDWeights(
        feature_weight=1 - response_weight - classification_weight,
        response_weight=response_weight, classification_weight=classification_weight,
    )
    metrics = distillation_diagnostics(
        torch.tensor([[10000.0, -10000.0], [10000.0, -10000.0]]),
        torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]]),
        torch.tensor([0, 0]), weights=weights, label_smoothing=0.0,
    )
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["kd_logit_grad_cosine"].item() == 0.0


class _FixedTeacher:
    feature_dim = 3

    def __call__(self, inputs):
        return inputs.detach(), inputs.detach()


@pytest.mark.parametrize("feature_weight", [0.0, 0.3])
def test_criterion_diagnostics_do_not_change_loss_or_gradients(feature_weight):
    student = torch.tensor([[1.0, 0.2, -0.5], [-0.3, 1.5, 0.5]], requires_grad=True)
    teacher = torch.tensor([[2.0, 0.0, 0.5], [-1.0, 1.0, 0.0]])
    features = student * 2
    labels = torch.tensor([0, 1])
    weights = KDWeights(
        temperature=2.0, feature_weight=feature_weight,
        response_weight=(1 - feature_weight) / 2, classification_weight=(1 - feature_weight) / 2,
    )
    criterion = DistillationCriterion(
        cast(Any, _FixedTeacher()), weights, 0.1, torch.device("cpu"),
        student_feature_dim=3,
    )
    projected = criterion.adapter(features) if criterion.adapter is not None else None
    expected = distillation_losses(
        student, teacher, labels, weights=weights, label_smoothing=0.1,
        student_features=projected, teacher_features=teacher,
    )
    losses = criterion(teacher, student, labels, features)
    for key in expected:
        torch.testing.assert_close(losses[key], expected[key])
    parameters = [student, *criterion.extra_parameters()]
    expected_gradients = torch.autograd.grad(expected["total"], parameters, retain_graph=True)
    actual_gradients = torch.autograd.grad(losses["total"], parameters)
    for actual, wanted in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted)
    for key in set(losses) - set(expected):
        assert not losses[key].requires_grad
    torch.testing.assert_close(losses["weighted_response_loss"], weights.response_weight * expected["response"])
    torch.testing.assert_close(
        losses["weighted_classification_loss"], weights.classification_weight * expected["classification"],
    )
