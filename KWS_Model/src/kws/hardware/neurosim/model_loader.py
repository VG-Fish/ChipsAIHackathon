"""Adapters to the repository's existing checkpoint reconstruction paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    model_config_path: Path | None
    model_config_sha256: str | None
    run_id: str | None
    metadata: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in checkpoint.items():
        if key in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            result[key] = list(value)
    return result


def load_inference_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    model_config_path: str | Path | None = None,
) -> LoadedModel:
    """Load a normal KWS checkpoint through ``kws.evaluate`` machinery."""

    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if path.suffix.lower() in {".safetensors", ".safetensor"}:
        raise ValueError(
            "a clean PerforatedAI safetensors artifact requires explicit project "
            "rebuild metadata; use load_clean_inference_model"
        )
    from kws.evaluate import load_model_from_checkpoint

    target_device = torch.device(device)
    model, checkpoint = load_model_from_checkpoint(str(path), target_device)
    model.cpu().eval()
    config_path = Path(model_config_path).resolve() if model_config_path else None
    return LoadedModel(
        model=model,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        model_config_path=config_path,
        model_config_sha256=sha256_file(config_path) if config_path else None,
        run_id=checkpoint.get("run_id"),
        metadata=_metadata(checkpoint),
    )


load_model = load_inference_model


def load_clean_inference_model(
    checkpoint_path: str | Path,
    *,
    model_config_path: str | Path,
    input_shape: tuple[int, int] | list[int] | None = None,
    num_classes: int | None = None,
) -> LoadedModel:
    """Rebuild an explicit clean PerforatedAI safetensors artifact.

    Clean PAI state files do not contain enough metadata to reconstruct the
    project model independently.  Requiring the model config and dimensions
    prevents silently choosing a different architecture.
    """

    path = Path(checkpoint_path).resolve()
    config_path = Path(model_config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if not config_path.exists():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    if input_shape is None or len(input_shape) != 2 or num_classes is None:
        raise ValueError(
            "clean PerforatedAI loading requires explicit input_shape and num_classes"
        )
    import yaml
    from kws.optimize.grow_clean_rebuild import load_clean_state, rebuild_clean_model

    model_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    state, metadata = load_clean_state(path)
    model = rebuild_clean_model(model_cfg, input_shape, int(num_classes), state)
    return LoadedModel(
        model=model.cpu().eval(),
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        model_config_path=config_path,
        model_config_sha256=sha256_file(config_path),
        run_id=metadata.get("run_id"),
        metadata={
            "artifact_kind": "clean_perforatedai",
            "input_shape": list(input_shape),
            "num_classes": int(num_classes),
            "safetensors_metadata": dict(metadata),
        },
    )
