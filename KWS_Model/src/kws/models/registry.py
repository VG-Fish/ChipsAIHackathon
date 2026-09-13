"""Student model families, selected by ``family`` in the model config.

A model config without ``family`` is DS-CNN, so every existing config and
checkpoint keeps loading unchanged. Checkpoints written by the SparkNet port
record the family as a top-level ``model_family`` instead; both spellings are
accepted and must agree when both are present.
"""
from collections.abc import Callable, Mapping
from typing import Any

import torch.nn as nn

from kws.models.ds_cnn import build_ds_cnn
from kws.models.sparknet import build_sparknet

DEFAULT_FAMILY = "ds_cnn"

MODEL_BUILDERS: dict[str, Callable[[dict, tuple[int, int], int], nn.Module]] = {
    "ds_cnn": build_ds_cnn,
    "sparknet": build_sparknet,
}


def checkpoint_input_shape(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    """Return and validate the two-dimensional feature shape in a checkpoint."""
    raw = checkpoint.get("input_shape")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("checkpoint input_shape must contain exactly two dimensions")
    height, width = raw
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("checkpoint input_shape dimensions must be integers")
    return height, width


def model_family(model_cfg: Mapping) -> str:
    family = model_cfg.get("family", DEFAULT_FAMILY)
    if family not in MODEL_BUILDERS:
        raise ValueError(
            f"unknown model family {family!r}; expected one of {sorted(MODEL_BUILDERS)}"
        )
    return family


def build_model(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> nn.Module:
    return MODEL_BUILDERS[model_family(model_cfg)](model_cfg, input_shape, num_classes)


def checkpoint_model_family(checkpoint: Mapping) -> str:
    """Resolve the family of a saved checkpoint from either spelling."""
    from_cfg = model_family(checkpoint["model_cfg"])
    recorded = checkpoint.get("model_family")
    if recorded is None:
        return from_cfg
    if recorded not in MODEL_BUILDERS:
        raise ValueError(f"unknown model_family: {recorded!r}")
    if "family" in checkpoint["model_cfg"] and recorded != from_cfg:
        raise ValueError(
            f"checkpoint model_family {recorded!r} disagrees with model_cfg family {from_cfg!r}"
        )
    return recorded


def build_model_from_checkpoint(checkpoint: Mapping) -> nn.Module:
    """Build the architecture a checkpoint describes, without loading weights."""
    family = checkpoint_model_family(checkpoint)
    return MODEL_BUILDERS[family](
        checkpoint["model_cfg"], checkpoint_input_shape(checkpoint), checkpoint["num_classes"],
    )
