"""Dependency-light configuration and cost projection for dendritic placements.

This module deliberately does not import PerforatedAI.  Compression searches can
therefore validate placements and budgets (including in ``--dry-run`` mode) on
machines where the optional compiled PAI runtime or its license is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from kws.models.ds_cnn import DSCNN
from kws.models.layers import DSConvBlock
from kws.utils.profile import count_macs, deployed_parameter_count


CONVERSION_MODES = ("blocks_and_linear", "fc_only", "module_ids")
# ``-1`` is this pipeline's CLI/config sentinel. PerforatedAI documents a
# positive integer cap, so translate the sentinel at the runtime boundary
# instead of depending on how a particular PAI release compares negatives.
UNLIMITED_DENDRITES = -1
PAI_EFFECTIVELY_UNLIMITED_DENDRITES = 2_147_483_647


def validate_max_dendrites(value: int) -> int:
    """Validate a finite positive dendrite cap or this pipeline's sentinel."""
    # ``bool`` is an ``int`` subclass, but accepting YAML values such as
    # ``max_dendrites: true`` as one dendrite is a configuration error rather
    # than a useful shorthand.  Do not silently truncate floats either.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_dendrites must be a positive integer or -1 for unlimited")
    if value == UNLIMITED_DENDRITES or value >= 1:
        return value
    raise ValueError("max_dendrites must be a positive integer or -1 for unlimited")


def pai_runtime_dendrite_limit(max_dendrites: int) -> int:
    """Return the positive cap passed to PerforatedAI.

    PAI still owns the normal no-improvement/retry stop. The large positive
    value only removes the reachable numeric cap without relying on an
    undocumented negative-value convention in the third-party API.
    """
    validate_max_dendrites(max_dendrites)
    return (
        PAI_EFFECTIVELY_UNLIMITED_DENDRITES
        if max_dendrites == UNLIMITED_DENDRITES
        else max_dendrites
    )


def projected_dendrite_count(max_dendrites: int) -> int:
    """Return the finite count usable for pre-training cost admission.

    An unlimited search cannot have a finite upper-bound projection. Its
    one-dendrite projection is instead the minimum useful augmented model;
    the clean exported graph remains the final budget authority.
    """
    validate_max_dendrites(max_dendrites)
    return 1 if max_dendrites == UNLIMITED_DENDRITES else max_dendrites


def normalize_module_ids(values: object) -> tuple[str, ...]:
    """Return exact PAI module IDs with one leading dot and no overlaps."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("conversion=module_ids requires a non-empty module_ids list")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("perforation module IDs must be non-empty strings")
        name = value.strip().lstrip(".")
        if not name or any(part in {"", ".", ".."} for part in name.split(".")):
            raise ValueError(f"invalid perforation module ID: {value!r}")
        module_id = f".{name}"
        if module_id not in normalized:
            normalized.append(module_id)

    names = [value[1:] for value in normalized]
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if name.startswith(other + ".") or other.startswith(name + "."):
                raise ValueError(
                    "perforation module IDs may not overlap: "
                    f"{name!r} and {other!r}"
                )
    return tuple(normalized)


def placement_module_names(model: nn.Module, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve a placement recipe to exact, validated module names."""
    conversion = str(config.get("conversion", "blocks_and_linear"))
    if conversion not in CONVERSION_MODES:
        raise ValueError(
            f"unsupported dendrite conversion {conversion!r}; "
            f"expected one of {list(CONVERSION_MODES)}"
        )
    if conversion == "blocks_and_linear":
        names = [name for name, module in model.named_modules() if isinstance(module, DSConvBlock)]
        names.append("fc")
    elif conversion == "fc_only":
        names = ["fc"]
    else:
        raw_ids = config.get("module_ids_to_perforate", config.get("module_ids"))
        names = [value[1:] for value in normalize_module_ids(raw_ids)]

    for name in names:
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"perforation module {name!r} does not exist") from exc
        if not any(True for _ in module.parameters(recurse=True)):
            raise ValueError(f"perforation module {name!r} has no parameters to copy")
    return tuple(names)


def pai_module_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact module IDs for PAI when ``conversion=module_ids``."""
    if config.get("conversion") != "module_ids":
        return ()
    return normalize_module_ids(
        config.get("module_ids_to_perforate", config.get("module_ids"))
    )


@dataclass(frozen=True)
class DendriticCostProjection:
    """Conservative clean-graph cost before committing to a PAI run."""

    base_params: int
    base_macs: int
    projected_params: int
    projected_macs: int
    copied_params_per_dendrite: int
    copied_macs_per_dendrite: int
    residual_params_per_dendrite: int
    residual_macs_per_dendrite: int
    scale_connections: int
    max_dendrites: int
    module_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_dendritic_cost(
    model: nn.Module,
    input_shape: tuple[int, int],
    placement: Mapping[str, Any],
    *,
    max_dendrites: int,
) -> DendriticCostProjection:
    """Project params/MACs for copied modules plus residual scale operations.

    The final clean graph is still profiled after training and remains the
    budget authority.  This projection charges each copied branch, each
    branch-to-parent scale, and the triangular set of branch-to-branch scales
    introduced when more than one dendrite is accepted.
    """
    if max_dendrites < 1:
        raise ValueError("max_dendrites must be at least 1")
    names = placement_module_names(model, placement)
    selected = set(names)
    copied_params = sum(
        parameter.numel()
        for name in names
        for parameter in model.get_submodule(name).parameters()
    )
    copied_macs = 0
    residual_params = 0
    residual_macs = 0
    handles: list[Any] = []

    def is_selected(name: str) -> bool:
        return any(name == parent or name.startswith(parent + ".") for parent in selected)

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal copied_macs
        copied_macs += output.numel() * (
            module.in_channels // module.groups
            * module.kernel_size[0]
            * module.kernel_size[1]
        )

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal copied_macs
        copied_macs += output.numel() * module.in_features

    def selected_output_hook(_module: nn.Module, _inputs, output: object) -> None:
        nonlocal residual_params, residual_macs
        if not isinstance(output, torch.Tensor):
            raise ValueError("selected perforation modules must return a tensor")
        residual_params += int(output.shape[1]) if output.ndim > 1 else 1
        residual_macs += output.numel()

    for name, module in model.named_modules():
        if is_selected(name):
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
        if name in selected:
            handles.append(module.register_forward_hook(selected_output_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            model(torch.zeros(1, 1, *input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    base_params = deployed_parameter_count(model)
    base_macs = count_macs(model, input_shape)
    # D branches have D connections to the parent output plus one connection
    # for each earlier-to-later dendrite pair: D + D(D-1)/2 = D(D+1)/2.
    scale_connections = max_dendrites * (max_dendrites + 1) // 2
    return DendriticCostProjection(
        base_params=base_params,
        base_macs=base_macs,
        projected_params=base_params
        + max_dendrites * copied_params
        + scale_connections * residual_params,
        projected_macs=base_macs
        + max_dendrites * copied_macs
        + scale_connections * residual_macs,
        copied_params_per_dendrite=copied_params,
        copied_macs_per_dendrite=copied_macs,
        residual_params_per_dendrite=residual_params,
        residual_macs_per_dendrite=residual_macs,
        scale_connections=scale_connections,
        max_dendrites=max_dendrites,
        module_names=names,
    )
