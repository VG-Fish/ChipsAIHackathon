import argparse
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import FeatureModel, build_ds_cnn
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lr_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup_steps = max(int(total_steps * warmup_fraction), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # PAI phases can outlive the nominal epoch budget. Once the planned
        # schedule is exhausted, hold the minimum learning rate instead of
        # letting cosine continue oscillating back to a high rate.
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_loss_acc(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


@dataclass
class FinetuneResult:
    """Outcome of one fine-tuning phase, for the stage report."""

    best_val_acc: float
    final_val_acc: float
    epochs: int
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "best_val_acc": self.best_val_acc,
            "final_val_acc": self.final_val_acc,
            "epochs": self.epochs,
            "history": self.history,
        }


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm statistics fixed when all of its affine params are frozen."""
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        local_parameters = list(module.parameters(recurse=False))
        if not any(parameter.requires_grad for parameter in local_parameters):
            module.eval()


def set_train_mode_preserving_frozen_batchnorm(model: nn.Module) -> None:
    """Train active modules without changing frozen BatchNorm statistics.

    ``requires_grad=False`` freezes affine parameters, but ``model.train()``
    would still update a frozen BatchNorm's running mean and variance. That is
    especially easy to miss during PAI's dendrite phase, where the base graph
    must be fixed while candidate residual nodes learn. Any BatchNorm with no
    active affine parameters is therefore put back in eval mode after the
    caller switches the model to training mode.
    """
    model.train()
    keep_frozen_batchnorm_eval(model)


def run_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    train_cfg: Mapping,
    *,
    kd=None,
    parameters: Iterable[nn.Parameter] | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
    on_best: Callable[[nn.Module, float], None] | None = None,
    epochs: int | None = None,
    label: str = "finetune",
) -> FinetuneResult:
    """One fine-tuning phase, shared by every stage of the framework.

    The framework fine-tunes after pruning, after the dendrite phase, after
    weight clustering, and under fake quantization.  Those differ only in three
    respects, which are the three hooks here: whether a distillation criterion
    supplements the task loss (``kd``), which parameters are active
    (``parameters``, so a frozen base or a centroid-only update is expressible),
    and what has to be re-imposed after each optimizer step (``post_step``, for
    N:M masks and codebook projection).
    """
    epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    active = list(parameters) if parameters is not None else _trainable(model.parameters())
    if kd is not None:
        active = active + list(kd.extra_parameters())
    if not active:
        raise ValueError(f"{label}: no trainable parameters were selected")

    optimizer = torch.optim.AdamW(
        active, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_lr_scheduler(
        optimizer, max(epochs * len(train_loader), 1), train_cfg["warmup_fraction"],
    )

    best_val_acc, final_val_acc = 0.0, 0.0
    history: list[dict] = []

    for epoch in range(epochs):
        set_train_mode_preserving_frozen_batchnorm(model)
        if kd is not None:
            kd.train()
        running = {"total": 0.0}
        seen = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            if kd is None:
                logits = model(features)
                losses = {"total": criterion(logits, labels)}
            else:
                feature_model = cast(FeatureModel, model)
                student_features = (
                    feature_model.forward_features(features) if kd.uses_features else None
                )
                logits = (
                    feature_model.classify_features(student_features)
                    if student_features is not None
                    else model(features)
                )
                losses = kd(features, logits, labels, student_features)
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
            if post_step is not None:
                post_step(model)
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value.item() * labels.size(0)
            seen += labels.size(0)

        train_losses = {name: value / max(seen, 1) for name, value in running.items()}
        if kd is not None:
            kd.eval()
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
        final_val_acc = val_acc
        record = {
            "epoch": epoch + 1,
            "train_loss": train_losses["total"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            **{f"train_{name}": value for name, value in train_losses.items() if name != "total"},
        }
        history.append(record)
        logger.info(
            "%s epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            label, epoch + 1, epochs, train_losses["total"], val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if on_best is not None:
                on_best(model, val_acc)

    return FinetuneResult(
        best_val_acc=best_val_acc,
        final_val_acc=final_val_acc,
        epochs=epochs,
        history=history,
    )


def train_model(model, datasets, label_map: dict, model_cfg: dict, train_cfg: dict,
                 checkpoint_path: Path, device, num_keywords: int, *,
                 kd=None, post_step=None, parameters=None, extra_checkpoint_fields=None) -> float:
    """Train `model` in place, checkpointing the best val_acc.

    Used to train a fresh model, and to fine-tune one that has already been
    structurally modified (after channel pruning, for example). `kd` adds the
    fixed-teacher distillation loss to the task loss; `post_step` re-imposes a
    sparsity mask or codebook after each optimizer step.
    """
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)
    logger.info(
        "DataLoader workers=%d persistent=%s prefetch_factor=%s",
        train_loader.num_workers,
        train_loader.persistent_workers,
        train_loader.prefetch_factor,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_best(trained_model, val_acc):
        checkpoint = {
            "model_state_dict": trained_model.state_dict(),
            "model_cfg": model_cfg,
            "input_shape": trained_model.input_shape,
            "num_classes": trained_model.fc.out_features,
            "label_map": label_map,
            "num_keywords": num_keywords,
            "val_acc": val_acc,
            "seed": train_cfg["seed"],
        }
        if extra_checkpoint_fields:
            checkpoint.update(extra_checkpoint_fields)
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, checkpoint_path)

    result = run_finetune(
        model,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        parameters=parameters,
        post_step=post_step,
        on_best=save_best,
        label=model_cfg.get("name", "train"),
    )
    return result.best_val_acc


def train(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    checkpoint_path: Path,
    *,
    seed: int | None = None,
):
    train_cfg = with_seed(train_cfg, seed)
    set_seed(train_cfg["seed"])
    device = get_device()

    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_keywords = len(data_cfg["target_keywords"])

    sample_features, _ = datasets[TRAIN][0]
    input_shape = (sample_features.shape[-2], sample_features.shape[-1])
    num_classes = len(label_names)

    model = build_ds_cnn(model_cfg, input_shape, num_classes).to(device)
    logger.info("Model %s: %d params, input_shape=%s, num_classes=%d",
                model_cfg["name"], sum(p.numel() for p in model.parameters()), input_shape, num_classes)

    return train_model(model, datasets, label_map, model_cfg, train_cfg, checkpoint_path, device, num_keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training-config seed (must be non-negative)",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    train(data_cfg, model_cfg, train_cfg, Path(args.checkpoint), seed=args.seed)


if __name__ == "__main__":
    main()
