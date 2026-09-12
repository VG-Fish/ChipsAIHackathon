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
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import record_input
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for deterministic test-set construction (default: 0)")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    layout = ArtifactLayout(args.output_dir) if args.output_dir else None
    report_path = None
    if layout is not None:
        layout.ensure_tree()
        report_path = layout.output_path(
            args.report,
            category="reports",
            default="evaluation.json",
        )

    def execute() -> None:
        if layout is not None:
            manifest = yaml.safe_load((layout.root / "manifest.yaml").read_text())
            record_input(manifest, args.checkpoint, role="checkpoint")
            record_input(manifest, args.data_config, role="data_config")
            layout.atomic_yaml(layout.root / "manifest.yaml", manifest)

        # In particular, this fixes the synthesized silence crops and gains for
        # a given evaluation seed. Training still seeds itself separately and
        # keeps drawing fresh silence samples throughout training.
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

        if args.report or report_path is not None:
            destination = report_path or Path(args.report)
            if layout is not None:
                layout.atomic_json(destination, metrics)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with temporary.open("w") as f:
                    json.dump(metrics, f, indent=2)
                temporary.replace(destination)
            logger.info("Wrote report to %s", destination)

    with run_session(
        args.output_dir,
        command="kws.evaluate",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=[(args.data_config, "data_config"), (args.checkpoint, "checkpoint")],
    ):
        execute()


if __name__ == "__main__":
    main()
