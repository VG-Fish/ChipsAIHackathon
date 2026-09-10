"""Optional knowledge distillation: soft targets from a larger teacher (e.g. DS-CNN-L)
into a smaller student (e.g. pruned DS-CNN-L or DS-CNN-XS). Only worth running if the
student shows a noticeable accuracy gap worth closing.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.train import build_lr_scheduler, evaluate_loss_acc
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

logger = get_logger(__name__)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                       temperature: float, alpha: float, label_smoothing: float) -> torch.Tensor:
    """alpha * hard-label CE + (1 - alpha) * soft-target KL, per Hinton et al."""
    hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def distill(teacher_checkpoint: str, student_checkpoint: str, data_cfg: dict, train_cfg: dict,
            out_checkpoint: Path, temperature: float = 4.0, alpha: float = 0.5) -> float:
    device = get_device()

    teacher_ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher = build_ds_cnn(teacher_ckpt["model_cfg"], tuple(teacher_ckpt["input_shape"]), teacher_ckpt["num_classes"])
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_ckpt = torch.load(student_checkpoint, map_location=device, weights_only=False)
    student = build_ds_cnn(student_ckpt["model_cfg"], tuple(student_ckpt["input_shape"]), student_ckpt["num_classes"])
    student.load_state_dict(student_ckpt["model_state_dict"])
    student.to(device)

    set_seed(train_cfg["seed"])
    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    train_loader = DataLoader(datasets[TRAIN], batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets[VAL], batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(student.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, train_cfg["warmup_fraction"])
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    best_val_acc = 0.0
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        student.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(features)
            student_logits = student(features)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha,
                                      train_cfg["label_smoothing"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(datasets[TRAIN])
        val_loss, val_acc = evaluate_loss_acc(student, val_loader, device, eval_criterion)
        logger.info("epoch %d/%d distill_train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, train_cfg["epochs"], train_loss, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": student.state_dict(),
                "model_cfg": student_ckpt["model_cfg"],
                "input_shape": student_ckpt["input_shape"],
                "num_classes": student_ckpt["num_classes"],
                "label_map": label_map,
                "num_keywords": student_ckpt["num_keywords"],
                "val_acc": val_acc,
            }, out_checkpoint)
            logger.info("Saved new best distilled checkpoint (val_acc=%.4f) -> %s", val_acc, out_checkpoint)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/full.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    distill(args.teacher_checkpoint, args.student_checkpoint, data_cfg, train_cfg,
            Path(args.out_checkpoint), args.temperature, args.alpha)


if __name__ == "__main__":
    main()
