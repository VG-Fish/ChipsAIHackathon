import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from kws.evaluate import load_model_from_checkpoint
from kws.utils.artifacts import ArtifactLayout
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
from kws.utils.seed import set_seed

logger = get_logger(__name__)


def export_module_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    onnx_path: str,
    atol: float = 1e-4,
    *,
    seed: int | None = None,
) -> float:
    """Export any live module and verify parity; returns the max abs difference.

    Taking a module rather than a checkpoint path is what lets the pipeline
    export a model whose architecture cannot be rebuilt from a config -- a
    PerforatedAI clean graph, or one carrying codebook parametrizations.
    """
    model = model.to("cpu").eval()
    if seed is not None:
        set_seed(seed)
    dummy = torch.randn(1, 1, *input_shape)

    destination = Path(onnx_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    torch.onnx.export(
        model, (dummy,), str(temporary_path),
        input_names=["features"], output_names=["logits"],
        opset_version=18,
    )
    temporary_path.replace(destination)
    logger.info("Exported ONNX model to %s", onnx_path)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(str(destination))
    onnx_out = session.run(None, {"features": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    logger.info("Max abs diff between PyTorch and ONNX Runtime outputs: %e", max_diff)
    if max_diff > atol:
        raise ValueError(f"ONNX parity check failed: max diff {max_diff} > tolerance {atol}")
    logger.info("ONNX parity check passed (tolerance=%e)", atol)
    return max_diff


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    atol: float = 1e-4,
    *,
    seed: int | None = None,
) -> float:
    model, ckpt = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    return export_module_to_onnx(
        model, tuple(ckpt["input_shape"]), onnx_path, atol, seed=seed
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic parity-check input (default: 0)",
    )
    args = parser.parse_args()
    if args.onnx_path is None and args.output_dir is None:
        parser.error("--onnx-path is required unless --output-dir is supplied")
    layout = ArtifactLayout(args.output_dir) if args.output_dir else None
    destination = (
        layout.output_path(
            args.onnx_path,
            category="models/exported",
            default="kws.onnx",
        )
        if layout is not None else Path(args.onnx_path)
    )
    with run_session(
        args.output_dir,
        command="kws.export.to_onnx",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=[(args.checkpoint, "checkpoint")],
    ):
        parity = export_to_onnx(args.checkpoint, str(destination), seed=args.seed)
        metadata = {
            "format_version": 1,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "onnx_path": str(destination),
            "onnx_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "parity_max_abs_diff": parity,
        }
        if layout is not None:
            layout.atomic_yaml(layout.root / "reports" / "export.yaml", metadata)
        else:
            metadata_path = destination.with_suffix(destination.suffix + ".yaml")
            temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            temporary_metadata.write_text(
                __import__("yaml").safe_dump(metadata), encoding="utf-8"
            )
            temporary_metadata.replace(metadata_path)


if __name__ == "__main__":
    main()
