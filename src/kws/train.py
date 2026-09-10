import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

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


def train_model(model, datasets, label_map: dict, model_cfg: dict, train_cfg: dict,
                 checkpoint_path: Path, device, num_keywords: int) -> float:
    """Reusable epoch loop: trains `model` in place, checkpointing the best val_acc.

    Used both for training a fresh model from scratch and for fine-tuning a model
    that's already been structurally modified (e.g. after channel pruning).
    """
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)
    logger.info(
        "DataLoader workers=%d persistent=%s prefetch_factor=%s",
        train_loader.num_workers,
        train_loader.persistent_workers,
        train_loader.prefetch_factor,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, train_cfg["warmup_fraction"])

    best_val_acc = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(datasets[TRAIN])
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
        logger.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, train_cfg["epochs"], train_loss, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "input_shape": model.input_shape,
                "num_classes": model.fc.out_features,
                "label_map": label_map,
                "num_keywords": num_keywords,
                "val_acc": val_acc,
            }, checkpoint_path)
            logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, checkpoint_path)

    return best_val_acc


def train(data_cfg: dict, model_cfg: dict, train_cfg: dict, checkpoint_path: Path):
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
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    train(data_cfg, model_cfg, train_cfg, Path(args.checkpoint))


if __name__ == "__main__":
    main()
