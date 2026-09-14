"""Formula checks for the read-only, paired-view KD diagnostic script."""

import runpy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/diagnose_kd.py"))


def test_old_kd_target_change_is_total_variation_of_effective_target():
    logits = torch.tensor([[2.0, -1.0], [-0.2, 0.3]])
    labels = torch.tensor([0, 1])
    result = _SCRIPT["summarize"](logits, labels, {"first": 0, "second": 1})
    target = 0.9 * F.one_hot(labels, 2) + 0.05
    mixed = (6 / 7) * target + (1 / 7) * logits.softmax(1)
    expected = 0.5 * (mixed - target).abs().sum(1).mean()
    assert result["old_kd_mean_target_total_variation"] == pytest.approx(expected.item())
    assert result["accuracy"] == 1
    assert result["classes"]["first"]["samples"] == 1


def test_diagnostic_gradient_ratio_matches_autograd():
    student = torch.tensor([[0.1, 0.6, 0.3], [-1.0, 0.4, 1.2]], requires_grad=True)
    teacher = torch.tensor([[1.4, 0.5, -0.3], [0.0, 0.3, 1.7]])
    labels = torch.tensor([0, 2])
    ce = 0.5 * F.cross_entropy(student, labels, label_smoothing=0.1)
    kd = 0.5 * 4 * F.kl_div((student / 2).log_softmax(1), (teacher / 2).softmax(1), reduction="batchmean")
    ce_grad = torch.autograd.grad(ce, student, retain_graph=True)[0]
    kd_grad = torch.autograd.grad(kd, student)[0]
    result = _SCRIPT["gradient_summary"](student.detach(), teacher, labels, 2, 0.5, 0.1)
    assert result["weighted_kd_to_ce_norm_ratio"] == pytest.approx((kd_grad.norm() / ce_grad.norm()).item())
    expected_cosine = F.cosine_similarity(ce_grad.flatten(), kd_grad.flatten(), dim=0)
    assert result["ce_kd_cosine"] == pytest.approx(expected_cosine.item())


def test_diagnostic_zero_gradients_are_finite():
    result = _SCRIPT["gradient_summary"](torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0, 1]), 1, 0, 1)
    assert result["weighted_kd_to_ce_norm_ratio"] == 0
    assert result["ce_kd_cosine"] == 0
