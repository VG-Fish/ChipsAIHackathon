"""Post-training int8 quantization -- a characterization spike only.

This proves the MCU-deployment path (ONNX -> int8) works and produces
size/accuracy numbers for the demo. It is NOT the artifact handed to Phase 3:
Perforated AI's dendrite library needs a real-valued, differentiable FP32
PyTorch graph, so the checkpoint that continues to Phase 3 must stay FP32.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TEST
from kws.export.to_onnx import export_to_onnx
from kws.utils.logging import get_logger
from kws.utils.metrics import compute_metrics

logger = get_logger(__name__)


def quantize_ptq(checkpoint_path: str, data_cfg: dict, onnx_fp32_path: Path, onnx_int8_path: Path) -> dict:
    export_to_onnx(checkpoint_path, str(onnx_fp32_path))

    quantize_dynamic(str(onnx_fp32_path), str(onnx_int8_path), weight_type=QuantType.QInt8)

    fp32_size = os.path.getsize(onnx_fp32_path)
    int8_size = os.path.getsize(onnx_int8_path)
    logger.info("FP32 ONNX size: %d bytes, int8 ONNX size: %d bytes (%.1f%% of fp32)",
                fp32_size, int8_size, 100 * int8_size / fp32_size)

    datasets, label_map = build_datasets(data_cfg, augment=False)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_keywords = sum(1 for n in label_names if n not in ("_unknown_", "_silence_"))
    test_loader = DataLoader(datasets[TEST], batch_size=128, shuffle=False, num_workers=0)

    session = ort.InferenceSession(str(onnx_int8_path))
    y_true, y_pred = [], []
    for features, labels in test_loader:
        logits = session.run(None, {"features": features.numpy()})[0]
        y_pred.extend(np.argmax(logits, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), num_keywords, label_names)
    metrics["fp32_size_bytes"] = fp32_size
    metrics["int8_size_bytes"] = int8_size
    logger.info("PTQ int8 test accuracy: %.4f  FAR: %.4f  FRR: %.4f", metrics["accuracy"], metrics["far"], metrics["frr"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-fp32-path", required=True)
    parser.add_argument("--onnx-int8-path", required=True)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    quantize_ptq(args.checkpoint, data_cfg, Path(args.onnx_fp32_path), Path(args.onnx_int8_path))


if __name__ == "__main__":
    main()
