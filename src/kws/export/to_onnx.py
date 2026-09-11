import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from kws.evaluate import load_model_from_checkpoint
from kws.utils.logging import get_logger

logger = get_logger(__name__)


def export_module_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    onnx_path: str,
    atol: float = 1e-4,
) -> float:
    """Export any live module and verify parity; returns the max abs difference.

    Taking a module rather than a checkpoint path is what lets the pipeline
    export a model whose architecture cannot be rebuilt from a config -- a
    PerforatedAI clean graph, or one carrying codebook parametrizations.
    """
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 1, *input_shape)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), onnx_path,
        input_names=["features"], output_names=["logits"],
        opset_version=18,
    )
    logger.info("Exported ONNX model to %s", onnx_path)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"features": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    logger.info("Max abs diff between PyTorch and ONNX Runtime outputs: %e", max_diff)
    if max_diff > atol:
        raise ValueError(f"ONNX parity check failed: max diff {max_diff} > tolerance {atol}")
    logger.info("ONNX parity check passed (tolerance=%e)", atol)
    return max_diff


def export_to_onnx(checkpoint_path: str, onnx_path: str, atol: float = 1e-4) -> None:
    model, ckpt = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    export_module_to_onnx(model, tuple(ckpt["input_shape"]), onnx_path, atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=True)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.onnx_path)


if __name__ == "__main__":
    main()
