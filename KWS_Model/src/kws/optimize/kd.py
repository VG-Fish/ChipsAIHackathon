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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.models.ds_cnn import FeatureModel, build_ds_cnn
from kws.models.registry import checkpoint_input_shape
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


@dataclass(frozen=True)
class KDAnnealing:
    """Linear schedule for introducing the teacher response target.

    The feature weight stays fixed.  The response weight is interpolated from
    ``response_weight_start`` through ``response_weight_end`` (the configured
    KD response weight by default), and the classification weight is the
    complement so that every effective recipe remains a convex mix.  Epochs
    are zero-based: ``start_epoch=0`` applies to the first training epoch.
    """

    start_epoch: int = 0
    end_epoch: int = 0
    response_weight_start: float = 0.0
    response_weight_end: float | None = None
    schedule_type: str = "linear"

    @classmethod
    def from_config(
        cls, config: Mapping | None, base_weights: KDWeights,
    ) -> "KDAnnealing | None":
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise TypeError("distillation.anneal must be a mapping")
        allowed = {
            "type", "start_epoch", "end_epoch",
            "response_weight_start", "response_weight_end",
        }
        unknown = sorted(set(config).difference(allowed))
        if unknown:
            raise ValueError(f"unknown distillation.anneal options: {unknown}")
        schedule_type = str(config.get("type", "linear"))
        if schedule_type != "linear":
            raise ValueError(
                f"unsupported distillation anneal type {schedule_type!r}; "
                "expected 'linear'"
            )
        start_epoch = int(config.get("start_epoch", 0))
        end_epoch = int(config.get("end_epoch", 0))
        if start_epoch < 0 or end_epoch < 0:
            raise ValueError("distillation anneal epochs must be non-negative")
        if end_epoch < start_epoch:
            raise ValueError(
                "distillation anneal end_epoch must be >= start_epoch"
            )
        start = float(config.get("response_weight_start", 0.0))
        end_value = config.get("response_weight_end", base_weights.response_weight)
        end = float(end_value)
        maximum = 1.0 - base_weights.feature_weight
        if not 0.0 <= start <= maximum:
            raise ValueError(
                "distillation anneal response_weight_start must be between "
                f"0 and {maximum:g}"
            )
        if not 0.0 <= end <= maximum:
            raise ValueError(
                "distillation anneal response_weight_end must be between "
                f"0 and {maximum:g}"
            )
        return cls(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            response_weight_start=start,
            response_weight_end=end,
            schedule_type=schedule_type,
        )

    def weights(self, base_weights: KDWeights, epoch: int) -> KDWeights:
        """Return the effective weights for one zero-based epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        end = (
            base_weights.response_weight
            if self.response_weight_end is None
            else self.response_weight_end
        )
        if self.end_epoch == self.start_epoch:
            response = end
        elif epoch <= self.start_epoch:
            response = self.response_weight_start
        elif epoch >= self.end_epoch:
            response = end
        else:
            progress = (epoch - self.start_epoch) / (
                self.end_epoch - self.start_epoch
            )
            response = self.response_weight_start + progress * (
                end - self.response_weight_start
            )
        return replace(
            base_weights,
            response_weight=response,
            classification_weight=(
                1.0 - base_weights.feature_weight - response
            ),
        )

    def as_dict(self) -> dict:
        return {
            "type": self.schedule_type,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "response_weight_start": self.response_weight_start,
            "response_weight_end": self.response_weight_end,
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


@torch.no_grad()
def distillation_diagnostics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weights: KDWeights,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """Detached batch diagnostics, computed on-device without extra backward passes.

    Teacher probabilities describe the actual augmented training inputs at T=1.
    The gradient ratio and cosine compare the weighted response and CE gradients
    with respect to student *logits*, not model parameters or the feature/gate
    losses. The common batch-mean factor cancels in both statistics. Epoch logs
    report sample-weighted means of these batch statistics, not a global cosine.
    Zero gradient norms use a 1e-12 denominator floor to keep logging finite.
    """
    student = student_logits.detach().float()
    teacher = teacher_logits.detach().float()
    teacher_log_probabilities = F.log_softmax(teacher, dim=1)
    teacher_probabilities = teacher_log_probabilities.exp()
    targets = F.one_hot(labels, num_classes=student.shape[1]).to(student.dtype)
    targets = targets * (1.0 - label_smoothing) + label_smoothing / student.shape[1]
    classification_gradient = weights.classification_weight * (
        F.softmax(student, dim=1) - targets
    )
    response_gradient = weights.response_weight * weights.temperature * (
        F.softmax(student / weights.temperature, dim=1)
        - F.softmax(teacher / weights.temperature, dim=1)
    )
    classification_flat = classification_gradient.flatten()
    response_flat = response_gradient.flatten()
    return {
        "teacher_accuracy": (teacher.argmax(dim=1) == labels).float().mean(),
        "teacher_confidence": teacher_probabilities.max(dim=1).values.mean(),
        "teacher_true_class_probability": teacher_probabilities.gather(
            1, labels.unsqueeze(1)
        ).mean(),
        "teacher_entropy_nats": -(
            teacher_probabilities * teacher_log_probabilities
        ).sum(dim=1).mean(),
        "kd_logit_grad_norm_ratio": (
            torch.linalg.vector_norm(response_flat)
            / torch.linalg.vector_norm(classification_flat).clamp_min(1e-12)
        ),
        "kd_logit_grad_cosine": F.cosine_similarity(
            response_flat, classification_flat, dim=0, eps=1e-12
        ),
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
        # Loading a teacher must not perturb the student's seeded
        # initialization.  ``build_ds_cnn`` initializes parameters before the
        # checkpoint overwrites them, consuming the process-global CPU RNG.
        # Isolate that construction so a supervised run and its KD control
        # start from exactly the same student weights for the same seed.
        with torch.random.fork_rng(devices=[]):
            model = build_ds_cnn(
                checkpoint["model_cfg"],
                checkpoint_input_shape(checkpoint),
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
        self.input_shape = checkpoint_input_shape(checkpoint)
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

    state_format_version = 1

    def __init__(
        self,
        teacher: FrozenTeacher,
        weights: KDWeights,
        label_smoothing: float,
        device: torch.device,
        *,
        student_feature_dim: int | None = None,
        anneal: Mapping | None = None,
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
        self.base_weights = self.weights
        self.anneal = KDAnnealing.from_config(anneal, self.base_weights)
        self.current_epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Set the effective KD mix for a zero-based training epoch."""
        self.current_epoch = int(epoch)
        self.weights = (
            self.anneal.weights(self.base_weights, self.current_epoch)
            if self.anneal is not None
            else self.base_weights
        )

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

    def state_dict(self) -> dict:
        """Return the training-only KD state required for exact resume.

        ``DistillationCriterion`` intentionally is not an ``nn.Module`` because
        registering the frozen teacher would duplicate the complete teacher in
        every student checkpoint.  The only trainable criterion state is the
        optional feature adapter, so keep a small explicit schema for it.
        """
        return {
            "format_version": self.state_format_version,
            "weights": self.base_weights.as_dict(),
            "adapter_state_dict": (
                self.adapter.state_dict() if self.adapter is not None else None
            ),
            "anneal": self.anneal.as_dict() if self.anneal is not None else None,
        }

    def load_state_dict(self, state: Mapping, *, strict: bool = True):
        """Restore and validate the training-only KD state."""
        if not isinstance(state, Mapping):
            raise TypeError("KD state must be a mapping")

        required = {"format_version", "weights", "adapter_state_dict"}
        optional = {"anneal"}
        missing = required.difference(state)
        unexpected = set(state).difference(required | optional)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing keys: {sorted(missing)}")
            if unexpected:
                details.append(f"unexpected keys: {sorted(unexpected)}")
            raise RuntimeError("invalid KD state (" + "; ".join(details) + ")")

        version = state.get("format_version")
        if version != self.state_format_version:
            raise RuntimeError(
                f"unsupported KD state format {version!r}; "
                f"expected {self.state_format_version}"
            )
        if state.get("weights") != self.base_weights.as_dict():
            raise RuntimeError("KD state weights do not match the requested recipe")
        saved_anneal = state.get("anneal")
        expected_anneal = self.anneal.as_dict() if self.anneal is not None else None
        if saved_anneal != expected_anneal:
            raise RuntimeError("KD state annealing does not match the requested recipe")

        adapter_state = state.get("adapter_state_dict")
        if self.adapter is None:
            if adapter_state not in (None, {}):
                raise RuntimeError(
                    "checkpoint contains a feature adapter but this KD recipe does not"
                )
            return None
        if adapter_state is None:
            raise RuntimeError(
                "checkpoint is missing the feature adapter required by this KD recipe"
            )
        return self.adapter.load_state_dict(adapter_state, strict=strict)

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
        losses = distillation_losses(
            student_logits,
            teacher_logits,
            labels,
            weights=self.weights,
            label_smoothing=self.label_smoothing,
            student_features=projected,
            teacher_features=teacher_features if self.uses_features else None,
        )
        # Training loops log every scalar, but backpropagate only "total".
        # Keep distillation_losses' legacy four-key contract unchanged.
        losses.update(distillation_diagnostics(
            student_logits, teacher_logits, labels,
            weights=self.weights, label_smoothing=self.label_smoothing,
        ))
        losses["weighted_response_loss"] = (
            self.weights.response_weight * losses["response"].detach()
        )
        losses["weighted_classification_loss"] = (
            self.weights.classification_weight * losses["classification"].detach()
        )
        return losses

    def describe(self) -> dict:
        return {
            "teacher_checkpoint": self.teacher.checkpoint_path,
            "teacher_sha256": self.teacher.checkpoint_sha256,
            "teacher_model": self.teacher.model_cfg["name"],
            "teacher_val_acc": self.teacher.val_acc,
            "weights": self.base_weights.as_dict(),
            "anneal": self.anneal.as_dict() if self.anneal is not None else None,
            "effective_weights": self.weights.as_dict(),
            "feature_adapter": (
                "training_only_linear_checkpointed_not_deployed"
                if self.adapter is not None
                else None
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
            feature_model = cast(FeatureModel, model)
            features = feature_model.forward_features(
                torch.zeros(1, 1, *input_shape, device=device)
            )
        return features.dim() == 2
    except Exception:  # noqa: BLE001 - any failure means "not usable", not a crash
        return False
    finally:
        model.train(was_training)
