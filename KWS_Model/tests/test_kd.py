import pytest
import torch
import torch.nn.functional as F

from kws.models.ds_cnn import DSCNN
from kws.optimize.kd import (
    DistillationCriterion,
    KDWeights,
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
        _TeacherDescription(), weights, 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    with torch.no_grad():
        source.adapter.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))

    restored = DistillationCriterion(
        _TeacherDescription(), weights, 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    restored.load_state_dict(source.state_dict(), strict=True)

    assert torch.equal(restored.adapter.weight, source.adapter.weight)


def test_distillation_criterion_rejects_missing_required_adapter_state():
    criterion = DistillationCriterion(
        _TeacherDescription(), KDWeights(), 0.0, torch.device("cpu"),
        student_feature_dim=3,
    )
    state = criterion.state_dict()
    state["adapter_state_dict"] = None

    with pytest.raises(RuntimeError, match="missing the feature adapter"):
        criterion.load_state_dict(state, strict=True)
