import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TEST
from kws.models.ds_cnn import build_ds_cnn
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.metrics import compute_metrics
from kws.utils.seed import set_seed

logger = get_logger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_ds_cnn(ckpt["model_cfg"], tuple(ckpt["input_shape"]), ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def run_inference(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", default=None, help="Optional path to write JSON metrics report")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for deterministic test-set construction (default: 0)")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    # In particular, this fixes the synthesized silence crops and gains for a
    # given evaluation seed. Training still seeds itself separately and keeps
    # drawing fresh silence samples throughout training.
    set_seed(args.seed)
    device = get_device()
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    datasets, label_map = build_datasets(data_cfg, augment=False, seed=args.seed)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    test_loader = DataLoader(datasets[TEST], batch_size=128, shuffle=False, num_workers=0)

    y_true, y_pred = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, ckpt["num_keywords"], label_names)
    metrics["num_params"] = sum(p.numel() for p in model.parameters())

    logger.info("Test accuracy: %.4f  FAR: %.4f  FRR: %.4f  params: %d",
                metrics["accuracy"], metrics["far"], metrics["frr"], metrics["num_params"])
    for name, f1 in metrics["f1_per_class"].items():
        logger.info("  F1[%s] = %.4f", name, f1)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote report to %s", args.report)


if __name__ == "__main__":
    main()
