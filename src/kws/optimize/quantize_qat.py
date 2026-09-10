"""Quantization-aware training (int8) -- a characterization spike only, same
caveat as quantize_ptq.py: not the artifact handed to Phase 3.
"""
import argparse
from pathlib import Path

import torch
import torch.ao.quantization as tq
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TEST, TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.train import build_lr_scheduler, evaluate_loss_acc
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

logger = get_logger(__name__)


class QATWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


def quantize_aware_train(checkpoint_path: str, data_cfg: dict, train_cfg: dict, out_checkpoint: Path) -> float:
    device = torch.device("cpu")  # QAT fake-quant + eventual int8 convert targets CPU inference
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_ds_cnn(ckpt["model_cfg"], tuple(ckpt["input_shape"]), ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])

    # fbgemm (the torch.ao.quantization default) targets x86; qnnpack is the
    # correct backend for ARM (Apple Silicon) CPUs, which is what this runs on.
    backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = backend

    wrapped = QATWrapper(model)
    wrapped.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(wrapped, inplace=True)

    set_seed(train_cfg["seed"])
    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, train_cfg["warmup_fraction"])

    best_val_acc = 0.0
    for epoch in range(train_cfg["epochs"]):
        wrapped.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = wrapped(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        _, val_acc = evaluate_loss_acc(wrapped, val_loader, device, criterion)
        logger.info("QAT epoch %d/%d val_acc=%.4f", epoch + 1, train_cfg["epochs"], val_acc)
        best_val_acc = max(best_val_acc, val_acc)

    wrapped.eval()
    quantized = tq.convert(wrapped, inplace=False)

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(quantized), str(out_checkpoint))
    logger.info("Saved QAT int8 TorchScript model -> %s (val_acc=%.4f)", out_checkpoint, best_val_acc)
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/qat.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    quantize_aware_train(args.checkpoint, data_cfg, train_cfg, Path(args.out_checkpoint))


if __name__ == "__main__":
    main()
