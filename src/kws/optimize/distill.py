"""IMC-oriented KWS knowledge distillation from Song et al., Interspeech 2022.

The paper combines feature-based distillation, response-based distillation, and
ground-truth classification. Teacher and student encoder widths differ in this
project, so a training-only linear adapter maps student features to the teacher
width. The adapter is intentionally absent from the saved deployment model.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.train import build_lr_scheduler, evaluate_loss_acc
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

logger = get_logger(__name__)


def imc_distillation_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    feature_weight: float,
    response_weight: float,
    classification_weight: float,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """Return the paper's weighted feature, response, and classification losses."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    weights = (feature_weight, response_weight, classification_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("distillation weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("distillation weights must sum to 1")
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "student and teacher features must match after projection: "
            f"{student_features.shape} != {teacher_features.shape}"
        )

    classification_loss = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing,
    )
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    response_loss = (
        F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        * (temperature ** 2)
    )
    feature_loss = F.mse_loss(student_features, teacher_features)
    total_loss = (
        feature_weight * feature_loss
        + response_weight * response_loss
        + classification_weight * classification_loss
    )
    return {
        "total": total_loss,
        "feature": feature_loss,
        "response": response_loss,
        "classification": classification_loss,
    }


def distill(
    teacher_checkpoint: str,
    student_model_cfg: dict,
    data_cfg: dict,
    train_cfg: dict,
    out_checkpoint: Path,
    student_checkpoint: str | None = None,
) -> float:
    set_seed(train_cfg["seed"])
    device = get_device()

    teacher_ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    input_shape = tuple(teacher_ckpt["input_shape"])
    num_classes = teacher_ckpt["num_classes"]

    # Build the fresh student first so constructing the frozen teacher cannot
    # affect its seeded initialization.
    student = build_ds_cnn(student_model_cfg, input_shape, num_classes).to(device)
    if student_checkpoint is not None:
        initial_student_ckpt = torch.load(student_checkpoint, map_location=device, weights_only=False)
        if initial_student_ckpt["model_cfg"] != student_model_cfg:
            raise ValueError("student checkpoint architecture does not match student model config")
        student.load_state_dict(initial_student_ckpt["model_state_dict"])

    teacher = build_ds_cnn(teacher_ckpt["model_cfg"], input_shape, num_classes)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Song et al. directly compare equal-width encoders. Our XS and L models
    # have different widths, so this adapter makes the representations
    # comparable and is discarded after training.
    feature_adapter = nn.Linear(
        student.fc.in_features, teacher.fc.in_features, bias=False,
    ).to(device)

    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    if len(label_map) != num_classes:
        raise ValueError(
            f"data config has {len(label_map)} classes but teacher has {num_classes}"
        )
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(feature_adapter.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, train_cfg["warmup_fraction"])
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    kd_cfg = train_cfg.get("distillation", {})
    temperature = float(kd_cfg.get("temperature", 1.0))
    feature_weight = float(kd_cfg.get("feature_weight", 0.3))
    response_weight = float(kd_cfg.get("response_weight", 0.1))
    classification_weight = float(kd_cfg.get("classification_weight", 0.6))

    best_val_acc = 0.0
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        student.train()
        feature_adapter.train()
        running_losses = {
            "total": 0.0,
            "feature": 0.0,
            "response": 0.0,
            "classification": 0.0,
        }
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            with torch.no_grad():
                teacher_features = teacher.forward_features(features)
                teacher_logits = teacher.classify_features(teacher_features)
            student_features = student.forward_features(features)
            student_logits = student.classify_features(student_features)
            projected_student_features = feature_adapter(student_features)
            losses = imc_distillation_losses(
                student_logits,
                teacher_logits,
                projected_student_features,
                teacher_features,
                labels,
                temperature=temperature,
                feature_weight=feature_weight,
                response_weight=response_weight,
                classification_weight=classification_weight,
                label_smoothing=train_cfg["label_smoothing"],
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
            for name, loss in losses.items():
                running_losses[name] += loss.item() * labels.size(0)

        train_losses = {
            name: value / len(datasets[TRAIN])
            for name, value in running_losses.items()
        }
        val_loss, val_acc = evaluate_loss_acc(student, val_loader, device, eval_criterion)
        logger.info(
            "epoch %d/%d train_loss=%.4f (cls=%.4f feat=%.4f resp=%.4f) "
            "val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            train_cfg["epochs"],
            train_losses["total"],
            train_losses["classification"],
            train_losses["feature"],
            train_losses["response"],
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": student.state_dict(),
                "model_cfg": student_model_cfg,
                "input_shape": input_shape,
                "num_classes": num_classes,
                "label_map": label_map,
                "num_keywords": teacher_ckpt["num_keywords"],
                "val_acc": val_acc,
                "distillation": {
                    "method": "song22c_imc_feature_response",
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "student_initialization": (
                        str(student_checkpoint) if student_checkpoint else "fresh"
                    ),
                    "temperature": temperature,
                    "feature_weight": feature_weight,
                    "response_weight": response_weight,
                    "classification_weight": classification_weight,
                    "feature_adapter": "training_only_linear_not_saved",
                },
            }, out_checkpoint)
            logger.info("Saved new best distilled checkpoint (val_acc=%.4f) -> %s", val_acc, out_checkpoint)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/distill_imc.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-model-config", default="configs/model/ds_cnn_xs.yaml")
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        help="Optional warm-start checkpoint; omit to train the student from scratch",
    )
    parser.add_argument("--out-checkpoint", required=True)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.student_model_config) as f:
        student_model_cfg = yaml.safe_load(f)

    distill(
        args.teacher_checkpoint,
        student_model_cfg,
        data_cfg,
        train_cfg,
        Path(args.out_checkpoint),
        student_checkpoint=args.student_checkpoint,
    )


if __name__ == "__main__":
    main()
