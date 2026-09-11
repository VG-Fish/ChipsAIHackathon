"""Knowledge distillation shared by every stage of the compression framework.

The framework distills from one fixed full-precision teacher at four different
points: the initial student baseline, pruning fine-tuning, the post-dendrite
resume, and quantization-aware fine-tuning.  All four want the same three
losses from Song et al., Interspeech 2022 (feature + response + ground truth),
so they live here once and each stage passes the criterion into its loop.

The teacher is loaded once and frozen.  A stage that cannot expose comparable
pooled features -- a PerforatedAI-wrapped or fake-quantized student, for
example -- uses :meth:`KDWeights.without_features`, which renormalizes the
response and classification weights instead of silently dropping a term.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.models.ds_cnn import build_ds_cnn
from kws.utils.logging import get_logger

logger = get_logger(__name__)


def file_sha256(path: str | Path) -> str:
    """Content identity for a checkpoint used as training provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class KDWeights:
    """Temperature and the three loss weights, which must form a convex mix."""

    temperature: float = 1.0
    feature_weight: float = 0.3
    response_weight: float = 0.1
    classification_weight: float = 0.6

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        weights = (
            self.feature_weight,
            self.response_weight,
            self.classification_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("distillation weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("distillation weights must sum to 1")

    @classmethod
    def from_config(cls, config: dict | None) -> "KDWeights":
        config = config or {}
        return cls(
            temperature=float(config.get("temperature", 1.0)),
            feature_weight=float(config.get("feature_weight", 0.3)),
            response_weight=float(config.get("response_weight", 0.1)),
            classification_weight=float(config.get("classification_weight", 0.6)),
        )

    def without_features(self) -> "KDWeights":
        """Redistribute the feature weight over the two surviving losses.

        Renormalizing keeps the total loss on the same scale as a stage that
        does have comparable features, so learning rates carry over between
        stages unchanged.
        """
        remaining = self.response_weight + self.classification_weight
        if remaining <= 0:
            raise ValueError(
                "cannot drop the feature loss when it carries all the weight"
            )
        return replace(
            self,
            feature_weight=0.0,
            response_weight=self.response_weight / remaining,
            classification_weight=self.classification_weight / remaining,
        )

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "feature_weight": self.feature_weight,
            "response_weight": self.response_weight,
            "classification_weight": self.classification_weight,
        }


def distillation_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weights: KDWeights,
    label_smoothing: float,
    student_features: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the weighted feature, response, and classification losses."""
    if weights.feature_weight > 0:
        if student_features is None or teacher_features is None:
            raise ValueError(
                "feature distillation is weighted but features were not supplied; "
                "use KDWeights.without_features() for feature-less stages"
            )
        if student_features.shape != teacher_features.shape:
            raise ValueError(
                "student and teacher features must match after projection: "
                f"{student_features.shape} != {teacher_features.shape}"
            )
        feature_loss = F.mse_loss(student_features, teacher_features)
    else:
        feature_loss = student_logits.new_zeros(())

    classification_loss = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing,
    )
    soft_student = F.log_softmax(student_logits / weights.temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / weights.temperature, dim=1)
    response_loss = (
        F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        * (weights.temperature ** 2)
    )

    total = (
        weights.feature_weight * feature_loss
        + weights.response_weight * response_loss
        + weights.classification_weight * classification_loss
    )
    return {
        "total": total,
        "feature": feature_loss,
        "response": response_loss,
        "classification": classification_loss,
    }


class FrozenTeacher(nn.Module):
    """The fixed full-precision teacher, loaded once and never updated.

    Every stage that distills reads logits and pooled features from the same
    instance, which is what makes "KD loss from the fixed teacher" literally
    true across pruning, dendrite, clustering, and quantization fine-tuning.
    """

    def __init__(self, checkpoint_path: str | Path, device: torch.device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_ds_cnn(
            checkpoint["model_cfg"],
            tuple(checkpoint["input_shape"]),
            checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.model_cfg = checkpoint["model_cfg"]
        self.input_shape = tuple(checkpoint["input_shape"])
        self.num_classes = int(checkpoint["num_classes"])
        self.num_keywords = int(checkpoint["num_keywords"])
        self.label_map = checkpoint.get("label_map")
        self.val_acc = checkpoint.get("val_acc")

    @property
    def feature_dim(self) -> int:
        return self.model.fc.in_features

    def train(self, mode: bool = True) -> "FrozenTeacher":
        # A fixed teacher must never leave eval mode; batch-norm statistics
        # drifting mid-distillation would silently change the target.
        return super().train(False)

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.model.forward_features(features)
        return self.model.classify_features(pooled), pooled


class DistillationCriterion:
    """Task loss + KD loss against a fixed teacher, usable by any training loop.

    When ``student_feature_dim`` is given, a training-only linear adapter maps
    the student's pooled features to the teacher width so the feature loss is
    well defined for unequal encoder widths.  The adapter is never part of the
    deployed model; ``extra_parameters`` hands it to the caller's optimizer.
    """

    def __init__(
        self,
        teacher: FrozenTeacher,
        weights: KDWeights,
        label_smoothing: float,
        device: torch.device,
        *,
        student_feature_dim: int | None = None,
    ):
        self.teacher = teacher
        self.label_smoothing = label_smoothing
        self.device = device
        self.adapter: nn.Linear | None = None

        if student_feature_dim is None:
            self.weights = weights.without_features()
            if weights.feature_weight > 0:
                logger.info(
                    "Student features unavailable; redistributing the %.2f feature "
                    "weight -> response %.3f / classification %.3f",
                    weights.feature_weight,
                    self.weights.response_weight,
                    self.weights.classification_weight,
                )
        else:
            self.weights = weights
            if weights.feature_weight > 0:
                self.adapter = nn.Linear(
                    student_feature_dim, teacher.feature_dim, bias=False,
                ).to(device)

    @property
    def uses_features(self) -> bool:
        return self.weights.feature_weight > 0

    def extra_parameters(self) -> list[nn.Parameter]:
        """Adapter parameters the caller must add to its optimizer."""
        return list(self.adapter.parameters()) if self.adapter is not None else []

    def train(self, mode: bool = True) -> None:
        if self.adapter is not None:
            self.adapter.train(mode)

    def eval(self) -> None:
        self.train(False)

    def __call__(
        self,
        inputs: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        teacher_logits, teacher_features = self.teacher(inputs)
        projected = None
        if self.uses_features:
            if student_features is None:
                raise ValueError(
                    "this criterion distills features but none were supplied"
                )
            projected = (
                self.adapter(student_features)
                if self.adapter is not None
                else student_features
            )
        return distillation_losses(
            student_logits,
            teacher_logits,
            labels,
            weights=self.weights,
            label_smoothing=self.label_smoothing,
            student_features=projected,
            teacher_features=teacher_features if self.uses_features else None,
        )

    def describe(self) -> dict:
        return {
            "teacher_checkpoint": self.teacher.checkpoint_path,
            "teacher_sha256": self.teacher.checkpoint_sha256,
            "teacher_model": self.teacher.model_cfg["name"],
            "teacher_val_acc": self.teacher.val_acc,
            "weights": self.weights.as_dict(),
            "feature_adapter": (
                "training_only_linear_not_saved" if self.adapter is not None else None
            ),
        }


def supports_pooled_features(model: nn.Module, input_shape: tuple[int, int]) -> bool:
    """Probe whether ``model`` can still expose the pooled encoder features.

    PerforatedAI wrapping, ``prepare_qat`` module swaps, and codebook
    parametrization all rewrite the module tree, so whether feature
    distillation is available is a property of the live object, not of the
    architecture.  Probing once is cheaper than discovering it mid-epoch.
    """
    if not hasattr(model, "forward_features"):
        return False
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            features = model.forward_features(torch.zeros(1, 1, *input_shape, device=device))
        return features.dim() == 2
    except Exception:  # noqa: BLE001 - any failure means "not usable", not a crash
        return False
    finally:
        model.train(was_training)
