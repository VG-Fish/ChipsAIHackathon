"""Grow one PerforatedAI dendrite during SparkNet's paper-recipe training.

The v2 dendrite study (``outputs/sparknet-dendritic-study-v2``) grew dendrites
on already-converged scratch networks, through a 40-epoch identity fine-tune
and a PAI phase that always ran AdamW with a restarted cosine schedule.  The
in-search dendrite delta was the same for every placement at every width, so
what v2 measured was the optimizer handoff, not the dendrites.  This driver
removes the handoff: the dendrite arrives *inside* the from-scratch run and the
base network never leaves the paper recipe.

Timeline in wall-clock epochs (S = switch_epoch, C = candidate_epochs,
E = epochs):

``1 .. S``  -- ``pre_switch``, neuron mode
    The unmodified recipe.  PAI wraps the model but adds nothing, and training
    is bit-identical to ``kws.train`` for the same seed and device, so the
    existing scratch run with that seed is this run's paired control.
``S+1 .. S+C`` -- ``candidate``, dendrite mode
    PAI strips the base out of the optimizer and base BatchNorm statistics are
    pinned; only the candidate learns, against Perforated Backpropagation's
    correlation objective.  These epochs consume no base-schedule steps.
``S+C+1 .. E+C`` -- ``post_switch``, neuron mode with one integrated dendrite
    The base resumes at schedule step ``S * steps_per_epoch`` with its
    optimizer state (SGD momentum) restored, so it sees the same E-epoch
    trajectory as scratch; dendrite parameters join in their own
    weight-decay group.

Three PAI behaviours are overridden, and each is verified while running:

- Switches are forced at exactly S and S + C (``DOING_NO_SWITCH`` plus
  ``add_validation_score(force_switch=True)``).  History switching would make
  the dendrite's arrival epoch, and therefore its training budget, data
  dependent.
- PAI reloads its best-validation checkpoint on every n->p switch.  Mid-run
  that would rewind the scratch trajectory to an earlier epoch and leak a
  validation selection into training.  The tracker's best score is reset just
  before the forced switch, so PAI saves the *current* weights as its best
  and reloads those.  Base tensors are compared across the switch and any
  difference aborts the run.
- ``retain_all_dendrites`` keeps the dendrite whatever its validation effect,
  so every run deploys exactly one dendrite and no run silently turns into a
  no-dendrite one.

Run variants (CLI flags; each is recorded under ``variant`` in the summary):

``--dendrite-weight-decay W``
    Overrides ``grow_dendrites.dendrite_weight_decay``, the weight decay of
    the post-switch SGD's dendrite parameter group.  The base group keeps the
    recipe's decay.
``--sham``
    The noise-floor arm.  Identical to a real run through the candidate phase
    and the p->n switch (same RNG consumption, same checks); immediately after
    the switch every dendrite parameter (skip weights included) is zeroed and
    frozen and the post-switch SGD holds the base group only.  The deployed
    function is then the base alone, so the paired delta of a sham run is what
    the C extra candidate epochs and the optimizer rebuild cost on their own.
    ``checks.sham_skip_weight_max_abs_final`` must be exactly 0.
``--dendrite-input-scale C``
    Overrides ``grow_dendrites.dendrite_input_scale`` (default 1).  PAI's
    forward function ``f`` becomes ``f(z / C)`` for the dendrite's
    pre-activation ``z``, in the candidate phase and after integration.  A
    placement whose input is unnormalized (SparkNet's pointwise convs see
    the raw depthwise output, std ~50-100) otherwise starts its candidates
    in tanh saturation and trains them deeper into it.  The base network is
    untouched, so pairing with scratch holds.  The exports fold ``1 / C``
    into the dendrite weights and run under plain ``f``, so deployment cost
    and every downstream reader are unchanged.
``--arm NAME``
    Label recorded as ``arm`` in the summary (``[A-Za-z0-9._-]+``).  Defaults
    to the placement, with ``-sham`` appended for a sham run.

Further checks in ``reports/grow_summary.yaml``:

``checks.integration_output_max_abs_diff``
    Eval-mode logits on a fixed validation probe at the end of base epoch S
    (before the n->p switch) against the same probe right after the p->n
    switch (after sham zeroing, before any post-switch step).  The probe
    restores every RNG stream and module mode, so it does not perturb the
    run.  A sham run must show exactly 0; a real run is only recorded, with a
    warning above 1e-6, since PAI starts the skip weights at zero.
``dendrite_input_std_at_switch``
    Standard deviation of each dendrite module's input on that probe at the
    end of base epoch S; the natural value of ``--dendrite-input-scale``.
``dendrite_diagnostics.{final,best}``
    :func:`kws.optimize.grow_diagnostics.dendrite_diagnostics` of each
    exported clean model on the training device (dendrite on/off validation
    accuracy, per-module linearity).  ``checks.diagnostics_val_acc_on_minus_*``
    compare its dendrite-on accuracy with the live graph's recorded accuracy
    (~0).  Diagnostics are advisory: a failure is recorded as
    ``{error: ...}`` and never fails the run.

The test split is never built.  Selection metrics are validation only.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.registry import build_model, model_family
from kws.optimize.dendritic import (
    GPA,
    UPA,
    _collect_dendrite_top_weights,
    _is_dendrite_parameter,
    _pai_cwd,
    base_params_in_optimizer,
    configure_perforatedai,
    current_pai_mode,
    export_final_pai_model,
    freeze_base_batchnorm_stats,
)
from kws.optimize.grow_diagnostics import dendrite_diagnostics
from kws.train import (
    _task_loss_scale,
    build_lr_scheduler,
    build_optimizer,
    collect_auxiliary_losses,
    evaluate_loss_acc,
    set_train_mode_preserving_frozen_batchnorm,
)
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import MetricsRecorder, capture_rng_state, restore_rng_state
from kws.utils.device import get_device
from kws.utils.logging import get_logger, run_session
from kws.utils.profile import profile_model
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)

STAGE = "grow"
SUMMARY_NAME = "grow_summary.yaml"
SUMMARY_FORMAT_VERSION = 1
SUMMARY_KIND = "sparknet_grow_dendrites"
EXIT_REFUSED = 2
EXIT_WALL_CLOCK = 3
# PAI wraps every tracked or perforated module and keeps the original under
# ``main_module``.  Those tensors -- weights and BatchNorm buffers -- are the
# base network.  (``dendrite_module.parent_module`` is a PAI-internal copy that
# PAI keeps out of the optimizer, so it is deliberately not matched.)
BASE_STATE_MARKER = ".main_module."
PARITY_TOLERANCE = 1e-4
PARITY_PROBE_EXAMPLES = 256
# A no-switch run never consults history, but the shared PAI configurator
# requires a patience value.  Large enough that it can never fire.
UNUSED_HISTORY_PATIENCE = 1_000_000
# Arm labels end up in directory names and report tables.
ARM_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
SHAM_SUFFIX = "-sham"
# A real run's logits may move this much across the integration before the
# driver warns; PAI starts the skip weights at zero, so ~0 is expected.
INTEGRATION_WARN_TOLERANCE = 1e-6
# Which recorded validation accuracy each exported model reproduces.
DIAGNOSTIC_REFERENCES = {"final": "final_val_acc", "best": "best_val_acc_post_switch"}


class WallClockExceeded(RuntimeError):
    """The run passed its ``max_wall_clock_minutes`` cap."""


class GrowInvariantError(RuntimeError):
    """A property the paired comparison depends on did not hold."""


@dataclass(frozen=True)
class GrowConfig:
    """The validated ``grow_dendrites`` block plus CLI overrides."""

    epochs: int
    switch_epoch: int
    candidate_epochs: int
    placement: str
    module_ids: tuple[str, ...]
    candidate_optimizer: dict[str, Any]
    dendrite_weight_decay: float
    carry_momentum: bool
    max_wall_clock_minutes: float | None
    # Zero and freeze the dendrite right after the p->n switch (noise floor).
    sham: bool = False
    # PAI's forward function runs as f(z / dendrite_input_scale).
    dendrite_input_scale: float = 1.0
    # Summary label; empty means ``default_arm(placement, sham)``.
    arm: str = ""

    def __post_init__(self) -> None:
        if not self.arm:
            object.__setattr__(self, "arm", default_arm(self.placement, self.sham))

    @property
    def total_epochs(self) -> int:
        return self.epochs + self.candidate_epochs

    def variant(self) -> dict[str, Any]:
        """What distinguishes this run from the other arms of the same placement."""
        return {
            "sham": self.sham,
            "dendrite_weight_decay": self.dendrite_weight_decay,
            "switch_epoch": self.switch_epoch,
            "candidate_epochs": self.candidate_epochs,
            "dendrite_input_scale": self.dendrite_input_scale,
        }

    def segment(self, epoch: int) -> str:
        """Segment of 1-based wall-clock ``epoch``."""
        if epoch <= self.switch_epoch:
            return "pre_switch"
        if epoch <= self.switch_epoch + self.candidate_epochs:
            return "candidate"
        return "post_switch"

    def base_epoch(self, epoch: int) -> int | None:
        """Base-schedule epoch of wall-clock ``epoch``; ``None`` while a candidate trains."""
        segment = self.segment(epoch)
        if segment == "pre_switch":
            return epoch
        if segment == "candidate":
            return None
        return epoch - self.candidate_epochs


def default_arm(placement: str, sham: bool) -> str:
    """The arm label a run gets without ``--arm``."""
    return f"{placement}{SHAM_SUFFIX}" if sham else placement


def resolve_grow_config(
    train_cfg: Mapping[str, Any],
    *,
    placement: str | None = None,
    switch_epoch: int | None = None,
    candidate_epochs: int | None = None,
    max_minutes: float | None = None,
    dendrite_weight_decay: float | None = None,
    sham: bool = False,
    arm: str | None = None,
    dendrite_input_scale: float | None = None,
) -> GrowConfig:
    """Validate the ``grow_dendrites`` block before any output is written.

    Keyword arguments are CLI overrides; ``None`` keeps the config's value.
    ``arm`` defaults to the placement, with ``-sham`` appended when ``sham``.
    """
    grow = train_cfg.get("grow_dendrites")
    if not isinstance(grow, Mapping):
        raise ValueError("train config has no grow_dendrites block")
    epochs = int(train_cfg["epochs"])
    switch = int(grow["switch_epoch"] if switch_epoch is None else switch_epoch)
    candidates = int(
        grow["candidate_epochs"] if candidate_epochs is None else candidate_epochs
    )
    if not 1 <= switch < epochs:
        raise ValueError(
            f"switch_epoch must be in [1, {epochs - 1}] so both neuron phases "
            f"train; got {switch}"
        )
    if candidates < 1:
        raise ValueError(f"candidate_epochs must be at least 1; got {candidates}")

    placements = grow.get("placements")
    if not isinstance(placements, Mapping) or not placements:
        raise ValueError("grow_dendrites.placements must map names to module ids")
    chosen = placement if placement is not None else grow.get("placement")
    if chosen not in placements:
        raise ValueError(
            f"unknown placement {chosen!r}; configured: {sorted(placements)}"
        )
    module_ids = tuple(str(module_id) for module_id in placements[chosen])
    if not module_ids:
        # The no-dendrite control for this design is the scratch run itself;
        # an empty placement would only re-train it under a PAI wrapper.
        raise ValueError(f"placement {chosen!r} names no modules")
    for module_id in module_ids:
        # PAI matches module ids exactly and requires the leading dot.
        if not module_id.startswith(".") or "[" in module_id:
            raise ValueError(f"PAI module ids must be dot-prefixed paths: {module_id!r}")

    candidate_optimizer = dict(
        grow.get("candidate_optimizer")
        or {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0}
    )
    candidate_optimizer["name"] = str(candidate_optimizer.get("name", "adamw")).lower()
    if candidate_optimizer["name"] not in {"adamw", "sgd"}:
        raise ValueError("candidate_optimizer.name must be 'adamw' or 'sgd'")
    candidate_optimizer["lr"] = float(candidate_optimizer.get("lr", 1e-3))
    candidate_optimizer["weight_decay"] = float(candidate_optimizer.get("weight_decay", 0.0))
    if candidate_optimizer["lr"] <= 0 or candidate_optimizer["weight_decay"] < 0:
        raise ValueError("candidate_optimizer needs lr > 0 and weight_decay >= 0")
    if candidate_optimizer["name"] == "sgd":
        candidate_optimizer["momentum"] = float(candidate_optimizer.get("momentum", 0.9))

    dendrite_weight_decay = float(
        grow.get("dendrite_weight_decay", 0.0)
        if dendrite_weight_decay is None else dendrite_weight_decay
    )
    if not math.isfinite(dendrite_weight_decay) or dendrite_weight_decay < 0:
        raise ValueError(
            f"dendrite_weight_decay must be non-negative and finite; got {dendrite_weight_decay}"
        )
    dendrite_input_scale = float(
        grow.get("dendrite_input_scale", 1.0)
        if dendrite_input_scale is None else dendrite_input_scale
    )
    if not math.isfinite(dendrite_input_scale) or dendrite_input_scale <= 0:
        raise ValueError(
            f"dendrite_input_scale must be positive and finite; got {dendrite_input_scale}"
        )

    cap =max_minutes if max_minutes is not None else grow.get("max_wall_clock_minutes")
    if cap is not None:
        cap = float(cap)
        if cap <= 0:
            raise ValueError("max_wall_clock_minutes must be positive")

    label = default_arm(str(chosen), bool(sham)) if arm is None else arm
    if not isinstance(label, str) or not ARM_PATTERN.fullmatch(label):
        raise ValueError(
            f"arm must be a non-empty name of letters, digits, '.', '_' or '-'; got {label!r}"
        )

    return GrowConfig(
        epochs=epochs,
        switch_epoch=switch,
        candidate_epochs=candidates,
        placement=str(chosen),
        module_ids=module_ids,
        candidate_optimizer=candidate_optimizer,
        dendrite_weight_decay=dendrite_weight_decay,
        carry_momentum=bool(grow.get("carry_momentum", True)),
        max_wall_clock_minutes=cap,
        sham=bool(sham),
        dendrite_input_scale=dendrite_input_scale,
        arm=label,
    )


class ScaledForwardFunction:
    """``base(z / scale)``: PAI's forward function on a rescaled pre-activation.

    PAI applies its forward function to every dendrite pre-activation, in the
    candidate phase and after integration, so dividing there is the same as
    feeding the dendrite ``x / scale`` while the base module still sees ``x``.
    """

    def __init__(self, base: Any, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be positive and finite; got {scale}")
        self.base = base
        self.scale = float(scale)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.base(z / self.scale)

    def __repr__(self) -> str:
        name = getattr(self.base, "__name__", repr(self.base))
        return f"{name}(z / {self.scale:g})"


def plain_forward_function() -> Any:
    """PAI's current forward function without any :class:`ScaledForwardFunction`."""
    function = GPA.pc.get_pai_forward_function()
    return function.base if isinstance(function, ScaledForwardFunction) else function


@contextlib.contextmanager
def pai_forward_function(function: Any):
    """Run the block with PAI's global forward function set to ``function``."""
    previous = GPA.pc.get_pai_forward_function()
    GPA.pc.set_pai_forward_function(function)
    try:
        yield
    finally:
        GPA.pc.set_pai_forward_function(previous)


def configure_grow_pai(
    pai_cfg: Mapping[str, Any],
    module_ids: Sequence[str],
    device: torch.device,
    *,
    dendrite_input_scale: float = 1.0,
) -> None:
    """Configure PAI for exactly one dendrite at externally forced switches."""
    if bool(pai_cfg.get("testing_dendrite_capacity", False)):
        # PAI's capacity test overrides the switch mode and dendrite limits;
        # a run with it on is not the experiment.
        raise ValueError("perforatedai.testing_dendrite_capacity must be false")
    configure_perforatedai(
        {
            "testing_dendrite_capacity": False,
            "max_dendrites": 1,
            "n_epochs_to_switch": UNUSED_HISTORY_PATIENCE,
            # Acceptance is not PAI's decision here; retain_all_dendrites
            # below keeps the dendrite and the paired analysis judges it.
            "improvement_threshold": 0.0,
            "candidate_weight_initialization_multiplier": float(
                pai_cfg.get("candidate_weight_initialization_multiplier", 0.01)
            ),
            "initial_correlation_batches": int(
                pai_cfg.get("initial_correlation_batches", 40)
            ),
            "max_dendrite_tries": 1,
            "forward_function": str(pai_cfg.get("forward_function", "tanh")),
            "conversion": "module_ids",
            "module_ids": list(module_ids),
        },
        device,
    )
    if dendrite_input_scale != 1.0:
        GPA.pc.set_pai_forward_function(
            ScaledForwardFunction(GPA.pc.get_pai_forward_function(), dendrite_input_scale)
        )
    GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
    GPA.pc.set_retain_all_dendrites(True)
    output_dimensions = pai_cfg.get("output_dimensions")
    if output_dimensions is not None:
        GPA.pc.set_output_dimensions(list(output_dimensions))
    # Base decay is part of the scratch recipe being reproduced, and dendrite
    # parameters get their own group; PAI's generic warning does not apply.
    if hasattr(GPA.pc, "set_weight_decay_accepted"):
        GPA.pc.set_weight_decay_accepted(True)


def build_base_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Mapping[str, Any],
    total_steps: int,
    *,
    start_step: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """The recipe's per-step schedule, entered at ``start_step``.

    Step ``k`` of the returned scheduler applies the multiplier the recipe's
    single uninterrupted schedule applies at step ``start_step + k``, so a base
    optimizer rebuilt after the candidate phase continues where the old one
    stopped instead of re-warming.
    """
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    scheduler = build_lr_scheduler(
        optimizer,
        max(total_steps, 1),
        train_cfg["warmup_fraction"],
        name=str(train_cfg.get("scheduler", "cosine")),
        hold_fraction=float(train_cfg.get("hold_fraction", 0.0)),
        min_lr=float(train_cfg.get("min_lr", 0.0)),
        power=float(train_cfg.get("polynomial_power", 2.0)),
    )
    if start_step == 0:
        return scheduler
    schedule = scheduler.lr_lambdas[0]
    # ``initial_lr`` is already recorded on the groups, so the new scheduler
    # keeps the recipe's peak LR as its base and only shifts the step index.
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step + start_step)
    )


def split_base_and_dendrite_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    base: list[nn.Parameter] = []
    dendrite: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (dendrite if _is_dendrite_parameter(name, model) else base).append(parameter)
    return base, dendrite


def build_post_switch_optimizer(
    model: nn.Module, train_cfg: Mapping[str, Any], grow: GrowConfig
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], list[nn.Parameter]]:
    """The recipe's optimizer for the integrated network.

    Base parameters keep the recipe's settings (its weight decay included);
    dendrite parameters get their own group with ``grow.dendrite_weight_decay``.
    A sham run's dendrite is frozen at zero, so its optimizer holds the base
    group only.  Returns the optimizer and the base and dendrite parameters.
    """
    base_parameters, dendrite_parameters = split_base_and_dendrite_parameters(model)
    groups: list[dict[str, Any]] = [{"params": base_parameters}]
    if not grow.sham:
        groups.append(
            {"params": dendrite_parameters, "weight_decay": grow.dendrite_weight_decay}
        )
    return build_optimizer(groups, train_cfg), base_parameters, dendrite_parameters


def zero_and_freeze_dendrite(model: nn.Module) -> tuple[int, int]:
    """Zero every dendrite-side parameter and stop its gradient.

    "Dendrite-side" is what :func:`split_base_and_dendrite_parameters` says,
    skip (``dendrites_to_top``) weights included, so the integrated network
    computes exactly the base.  Returns ``(tensors, elements)`` touched.
    """
    _base, dendrite = split_base_and_dendrite_parameters(model)
    if not dendrite:
        raise GrowInvariantError("sham run found no dendrite parameters to zero")
    with torch.no_grad():
        for parameter in dendrite:
            parameter.zero_()
            parameter.requires_grad_(False)
    return len(dendrite), sum(parameter.numel() for parameter in dendrite)


def dendrite_skip_weight_max_abs(model: nn.Module) -> float:
    """Largest ``|dendrites_to_top|`` entry across the network."""
    weights = [
        weight
        for module_weights in _collect_dendrite_top_weights(model).values()
        for weight in module_weights
        if weight.numel()
    ]
    if not weights:
        raise GrowInvariantError("no dendrite skip weights found; the sham check would be vacuous")
    return max(float(weight.abs().max()) for weight in weights)


def capture_optimizer_state_by_name(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, Any]]:
    """Per-parameter optimizer state keyed by parameter name.

    PAI's n->p switch reloads the network, so parameter objects are not
    stable across it; the names are.
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    captured: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            name = names.get(id(parameter))
            if not state or name is None:
                continue
            captured[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in state.items()
            }
    return captured


def restore_optimizer_state_by_name(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    captured: Mapping[str, Mapping[str, Any]],
) -> int:
    """Load captured state into ``optimizer`` for same-named parameters."""
    in_optimizer = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    restored = 0
    for name, parameter in model.named_parameters():
        state = captured.get(name)
        if state is None or id(parameter) not in in_optimizer:
            continue
        if any(
            torch.is_tensor(value) and value.dim() > 0 and value.shape != parameter.shape
            for value in state.values()
        ):
            continue
        optimizer.state[parameter] = {
            key: (
                value.detach().clone().to(parameter.device)
                if torch.is_tensor(value) else copy.deepcopy(value)
            )
            for key, value in state.items()
        }
        restored += 1
    return restored


def base_state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the base network's floating-point weights and BN buffers."""
    snapshot = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if BASE_STATE_MARKER in name and tensor.is_floating_point()
    }
    if not snapshot:
        raise GrowInvariantError(
            f"no base tensors matched {BASE_STATE_MARKER!r}; PAI's wrapper "
            "naming changed and the rewind/freeze checks would be vacuous"
        )
    return snapshot


def max_abs_change(before: Mapping[str, torch.Tensor], model: nn.Module) -> float:
    """Largest elementwise change of the ``before`` tensors in ``model``."""
    current = model.state_dict()
    missing = [name for name in before if name not in current]
    if missing:
        raise GrowInvariantError(f"base tensors disappeared from the model: {missing[:5]}")
    change = 0.0
    for name, tensor in before.items():
        after = current[name].detach().cpu()
        if after.shape != tensor.shape:
            raise GrowInvariantError(f"base tensor {name} changed shape")
        if tensor.numel():
            change = max(change, float((after - tensor).abs().max()))
    return change


def optimizer_numel(optimizer: torch.optim.Optimizer, parameters: Sequence[nn.Parameter]) -> int:
    wanted = {id(parameter) for parameter in parameters}
    seen: set[int] = set()
    total = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in wanted and id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def load_weights_for_export(model: nn.Module, state: Mapping[str, torch.Tensor]) -> list[str]:
    """Load a saved epoch's weights into the live PAI graph for export.

    PAI keeps its serialized tracker (``tracker_string``) in the state dict,
    and its length changes every epoch, so a strict load of an earlier
    epoch's state always fails.  Every tensor whose shape still matches is
    loaded; the skipped keys are returned for the run record, and a skipped
    *parameter* is an error because the export would then mix two epochs.
    """
    current = model.state_dict()
    loadable = {
        name: tensor for name, tensor in state.items()
        if name in current and current[name].shape == tensor.shape
    }
    unloaded = sorted(
        name for name, _parameter in model.named_parameters() if name not in loadable
    )
    if unloaded:
        raise GrowInvariantError(f"cannot reload saved parameters for export: {unloaded[:5]}")
    model.load_state_dict(loadable, strict=False)
    return sorted(set(state) - set(loadable))


def summarize_history(history: Sequence[Mapping[str, Any]], grow: GrowConfig) -> dict[str, Any]:
    """Validation results from per-epoch records; ``None`` where not reached."""
    neuron = [record for record in history if record.get("base_epoch") is not None]
    by_base_epoch = {int(record["base_epoch"]): float(record["val_acc"]) for record in neuron}
    pre = {epoch: acc for epoch, acc in by_base_epoch.items() if epoch <= grow.switch_epoch}
    post = {epoch: acc for epoch, acc in by_base_epoch.items() if epoch > grow.switch_epoch}
    candidate = [float(record["val_acc"]) for record in history if record.get("segment") == "candidate"]

    def window_mean(size: int) -> float | None:
        epochs = range(grow.epochs - size + 1, grow.epochs + 1)
        if not all(epoch in by_base_epoch for epoch in epochs):
            return None
        return sum(by_base_epoch[epoch] for epoch in epochs) / size

    best_post_epoch = max(post, key=lambda epoch: (post[epoch], -epoch)) if post else None
    return {
        "val_acc_at_switch": pre.get(grow.switch_epoch),
        "best_val_acc_pre_switch": max(pre.values()) if pre else None,
        "best_val_acc_post_switch": post[best_post_epoch] if best_post_epoch is not None else None,
        "best_base_epoch_post_switch": best_post_epoch,
        "final_val_acc": by_base_epoch.get(grow.epochs),
        "last5_mean_val_acc": window_mean(5),
        "last10_mean_val_acc": window_mean(10),
        "best_val_acc_overall": max(by_base_epoch.values()) if by_base_epoch else None,
        "candidate_val_acc_span": (max(candidate) - min(candidate)) if candidate else None,
    }


def existing_run_reason(output_dir: Path) -> str | None:
    """Why ``output_dir`` cannot host a fresh run, or ``None`` if it can.

    Runs are ~12 minutes and PAI's tracker is process-global, so this driver
    does not resume; a partial run is moved aside and retrained.
    """
    summary = output_dir / "reports" / SUMMARY_NAME
    if summary.exists():
        return f"{summary} already exists"
    for directory in (output_dir / "pai" / "candidates", output_dir / "metrics" / STAGE):
        if directory.is_dir() and any(directory.iterdir()):
            return f"{directory} holds output from an earlier attempt"
    return None


def _portable(value: Any) -> Any:
    """YAML-safe plain types."""
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _capture_rng(device: torch.device) -> dict[str, Any]:
    state = capture_rng_state()
    if device.type == "mps":
        state["torch_mps_rng_state"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    restore_rng_state(dict(state))
    if state.get("torch_mps_rng_state") is not None:
        torch.mps.set_rng_state(state["torch_mps_rng_state"])


def integration_probe(val_loader, device: torch.device) -> torch.Tensor:
    """The export parity probe, drawn mid-run without moving any RNG stream.

    Starting a loader iterator draws a worker seed from the loader's own
    generator (or the global one), so both are put back afterwards.
    """
    rng = _capture_rng(device)
    generator = getattr(val_loader, "generator", None)
    generator_state = generator.get_state().clone() if generator is not None else None
    try:
        return _parity_probe(val_loader, device)
    finally:
        if generator_state is not None:
            generator.set_state(generator_state)
        _restore_rng(rng)


def probe_logits(
    model: nn.Module,
    probe: torch.Tensor,
    device: torch.device,
    *,
    input_std: tuple[Sequence[str], dict[str, float]] | None = None,
) -> torch.Tensor:
    """Eval-mode logits on ``probe`` (on CPU), leaving no trace on the run.

    Every module's train/eval flag is restored exactly (base BatchNorm layers
    pinned in eval stay pinned), every RNG stream is restored, and no autograd
    graph is built, so PAI's backward hooks never see the probe.

    ``input_std=(module_ids, out)`` also fills ``out`` with the standard
    deviation of each listed module's input over the probe.
    """
    rng = _capture_rng(device)
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    if input_std is not None:
        module_ids, out = input_std
        modules = dict(model.named_modules())
        for module_id in module_ids:
            name = module_id.lstrip(".")
            if name not in modules:
                raise GrowInvariantError(f"no module {name!r} to measure the input of")
            handles.append(modules[name].register_forward_hook(
                lambda _m, args, _out, name=name: out.__setitem__(
                    name, float(args[0].detach().cpu().double().std())
                )
            ))
    try:
        model.eval()
        with torch.no_grad():
            return model(probe).detach().cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
        _restore_rng(rng)


def _train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    task_loss_scale: float,
    device: torch.device,
    *,
    pin_base_batchnorm: bool,
) -> tuple[dict[str, float], float]:
    """One epoch with ``kws.train.run_finetune``'s exact step sequence."""
    set_train_mode_preserving_frozen_batchnorm(model)
    if pin_base_batchnorm:
        # BatchNorm statistics move in forward(), which no optimizer filter
        # covers.  eval() pins them without touching requires_grad, which
        # PAI's neuron-error hook needs.
        freeze_base_batchnorm_stats(model)
    running: dict[str, torch.Tensor] = {
        "total": torch.zeros((), dtype=torch.float32, device=device)
    }
    correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        losses = {"total": task_loss_scale * criterion(logits, labels)}
        for name, (value, weight) in collect_auxiliary_losses(model).items():
            if name in losses:
                raise ValueError(f"auxiliary loss {name!r} collides with a task loss")
            losses[name] = value
            if weight:
                losses["total"] = losses["total"] + weight * value
        losses["total"].backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for name, value in losses.items():
            contribution = value.detach() * labels.size(0)
            running[name] = running.get(name, torch.zeros_like(contribution)) + contribution
        correct = correct + (logits.detach().argmax(dim=1) == labels).sum()
        seen += labels.size(0)
    return (
        {name: value.item() / max(seen, 1) for name, value in running.items()},
        correct.item() / max(seen, 1),
    )


def _skip_weight_mean_abs(model: nn.Module) -> dict[str, float]:
    return {
        name: float(weights[0].abs().mean())
        for name, weights in _collect_dendrite_top_weights(model).items()
        if weights
    }


def _pb_scores() -> Any:
    """PAI's current candidate correlation scores, for diagnostics only."""
    try:
        return _portable(GPA.pai_tracker.get_current_pb_scores())
    except Exception as exc:  # noqa: BLE001 - diagnostics must not end a run
        return f"unavailable: {type(exc).__name__}"


def run_grow(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    grow: GrowConfig,
    *,
    output_dir: str | Path,
    seed: int | None = None,
    config_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Train one grow-during-training run and write its summary.

    Returns the summary.  On a wall-clock stop or an invariant failure an
    incomplete summary is written before the exception propagates.
    """
    started = monotonic()
    train_cfg = with_seed(train_cfg, seed)
    seed = int(train_cfg["seed"])
    if model_family(model_cfg) != "sparknet":
        raise ValueError("the grow driver supports SparkNet model configs only")
    layout = ArtifactLayout(output_dir).ensure_tree()
    run_id = layout.manifest_run_id
    phase = str(model_cfg.get("name", "sparknet"))
    candidate_name = f"{phase}_{grow.placement}"
    run_dir = layout.pai_candidate_path(candidate_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = MetricsRecorder(
        layout.metrics_path(STAGE, phase), layout=layout, stage=STAGE, phase=phase
    )
    if recorder.count:
        raise FileExistsError(f"{recorder.path} already has records; move the run aside")
    summary_path = layout.report_path(SUMMARY_NAME)
    history: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    schedule: dict[str, Any] = {
        "base_epochs": grow.epochs,
        "switch_epoch": grow.switch_epoch,
        "candidate_epochs": grow.candidate_epochs,
        "total_epochs": grow.total_epochs,
        "candidate_optimizer": dict(grow.candidate_optimizer),
        "dendrite_weight_decay": grow.dendrite_weight_decay,
        "dendrite_input_scale": grow.dendrite_input_scale,
        "carry_momentum": grow.carry_momentum,
    }
    device = get_device()

    def write_summary(status: str, reason: str | None = None) -> dict[str, Any]:
        summary = {
            "format_version": SUMMARY_FORMAT_VERSION,
            "kind": SUMMARY_KIND,
            "status": status,
            "incomplete_reason": reason,
            "run_id": run_id,
            "width": int(model_cfg["channels"]),
            "seed": seed,
            "placement": grow.placement,
            "arm": grow.arm,
            "variant": grow.variant(),
            "module_ids": list(grow.module_ids),
            "model_name": phase,
            "device": str(device),
            **{key: value for key, value in (config_paths or {}).items()},
            "schedule": schedule,
            "results": summarize_history(history, grow),
            "checks": checks,
            **extra,
            "artifacts": {
                "metrics": layout.relative(recorder.path),
                "pai_dir": layout.relative(run_dir),
                **extra.get("artifact_paths", {}),
            },
            "elapsed_seconds": monotonic() - started,
            "test_split_used": False,
            "selection_split": "validation",
        }
        summary.pop("artifact_paths", None)
        summary = _portable(summary)
        layout.atomic_yaml(summary_path, summary)
        return summary

    try:
        # Mirror kws.train.train's seeding exactly: the paired scratch run's
        # initial weights and data order are what make it the control.
        set_seed(seed)
        datasets, label_map = build_datasets(
            data_cfg,
            augment=train_cfg["augment"],
            seed=seed,
            cache_features=bool(train_cfg.get("cache_features", True)),
            cache_train_features=bool(train_cfg.get("cache_train_features", False)),
            augmentation=train_cfg.get("augmentation"),
            splits=(TRAIN, VAL),
        )
        sample_features, _ = datasets[TRAIN][0]
        input_shape = (int(sample_features.shape[-2]), int(sample_features.shape[-1]))
        num_classes = len(label_map)
        configure_grow_pai(
            train_cfg.get("perforatedai", {}), grow.module_ids, device,
            dendrite_input_scale=grow.dendrite_input_scale,
        )

        set_seed(seed)
        base = build_model(model_cfg, input_shape, num_classes).to(device)
        base_params = sum(parameter.numel() for parameter in base.parameters())
        # Wrapping must not consume the stream the scratch run's first epoch
        # draws its gate noise from.
        rng_after_init = _capture_rng(device)
        with _pai_cwd(run_dir):
            model = UPA.perforate_model(
                base,
                doing_pai=True,
                save_name=run_dir.name,
                making_graphs=bool(train_cfg.get("perforatedai", {}).get("making_graphs", False)),
                maximizing_score=True,
            ).to(device)
        _restore_rng(rng_after_init)
        if UPA.count_params(model) != base_params:
            raise GrowInvariantError(
                f"PAI wrapping changed the trainable size {base_params} -> "
                f"{UPA.count_params(model)} before any dendrite exists"
            )
        logger.info(
            "Grow run: %s arm=%s placement=%s modules=%s seed=%d, %d params, switch after "
            "base epoch %d, %d candidate epochs, %d total epochs, dendrite weight decay %g, "
            "forward function %r",
            phase, grow.arm, grow.placement, list(grow.module_ids), seed, base_params,
            grow.switch_epoch, grow.candidate_epochs, grow.total_epochs,
            grow.dendrite_weight_decay, GPA.pc.get_pai_forward_function(),
        )
        if grow.sham:
            logger.warning(
                "SHAM RUN (arm %s): the dendrite is grown exactly as in a real run, then "
                "zeroed and frozen at the p->n switch.  This run measures the design's "
                "noise floor; its deployed function is the base network alone.",
                grow.arm,
            )

        train_loader = build_data_loader(
            datasets[TRAIN], train_cfg, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = build_data_loader(
            datasets[VAL], train_cfg, shuffle=False,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        steps_per_epoch = len(train_loader)
        total_base_steps = grow.epochs * steps_per_epoch
        schedule["steps_per_epoch"] = steps_per_epoch
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
        task_loss_scale = _task_loss_scale(train_cfg)
        pin_bn = bool(
            train_cfg.get("perforatedai", {}).get("freeze_base_batchnorm_in_dendrite_phase", True)
        )

        optimizer = build_optimizer(model.parameters(), train_cfg)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
        scheduler = build_base_scheduler(optimizer, train_cfg, total_base_steps)

        captured_state: dict[str, dict[str, Any]] = {}
        frozen_base: dict[str, torch.Tensor] = {}
        best_post_state: dict[str, torch.Tensor] | None = None
        best_post_acc = float("-inf")
        pb_history: list[dict[str, Any]] = []
        extra["pb_scores_by_candidate_epoch"] = pb_history
        global_step = 0
        base_step = 0
        # Logits on a fixed validation probe at the end of base epoch S, to be
        # compared with the integrated network before it trains.
        probe_at_switch: torch.Tensor | None = None
        logits_at_switch: torch.Tensor | None = None

        for epoch in range(1, grow.total_epochs + 1):
            epoch_started = monotonic()
            segment = grow.segment(epoch)
            mode_before = current_pai_mode()
            expected_mode = "p" if segment == "candidate" else "n"
            if mode_before != expected_mode:
                raise GrowInvariantError(
                    f"epoch {epoch} ({segment}) expected PAI mode {expected_mode!r}, "
                    f"found {mode_before!r}"
                )
            train_losses, train_acc = _train_epoch(
                model, train_loader, optimizer,
                scheduler if segment != "candidate" else None,
                criterion, task_loss_scale, device,
                pin_base_batchnorm=segment == "candidate" and pin_bn,
            )
            global_step += steps_per_epoch
            if segment != "candidate":
                base_step += steps_per_epoch
            val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
            record = {
                "schema_version": 1,
                "stage": STAGE,
                "phase": phase,
                "epoch": epoch,
                "segment": segment,
                "pai_mode": mode_before,
                "base_epoch": grow.base_epoch(epoch),
                "global_step": global_step,
                "base_step": base_step,
                "elapsed_seconds": monotonic() - epoch_started,
                "train_loss": train_losses["total"],
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_accuracy": val_acc,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "learning_rate": [group["lr"] for group in optimizer.param_groups],
                "parameter_count": int(UPA.count_params(model)),
                "seed": seed,
                **{f"train_{name}": value for name, value in train_losses.items() if name != "total"},
            }
            history.append(record)
            recorder.append(record)
            logger.info(
                "epoch %d/%d %s base_epoch=%s lr=%.6g train_loss=%.4f val_acc=%.4f params=%d",
                epoch, grow.total_epochs, segment, record["base_epoch"],
                record["learning_rate"][0], train_losses["total"], val_acc,
                record["parameter_count"],
            )

            if segment == "post_switch" and val_acc > best_post_acc:
                best_post_acc = val_acc
                best_post_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

            GPA.pai_tracker.add_extra_score(train_acc, "Train")
            force_to_candidate = epoch == grow.switch_epoch
            force_to_neuron = epoch == grow.switch_epoch + grow.candidate_epochs
            if force_to_candidate:
                if grow.carry_momentum:
                    captured_state = capture_optimizer_state_by_name(model, optimizer)
                schedule["base_lr_at_switch"] = [group["lr"] for group in optimizer.param_groups][0]
                before_switch = base_state_snapshot(model)
                probe_at_switch = integration_probe(val_loader, device)
                input_std: dict[str, float] = {}
                logits_at_switch = probe_logits(
                    model, probe_at_switch, device,
                    input_std=(grow.module_ids, input_std),
                )
                extra["dendrite_input_std_at_switch"] = input_std
                logger.info(
                    "Dendrite module input std at the switch: %s (input scale %g)",
                    {name: round(value, 4) for name, value in input_std.items()},
                    grow.dendrite_input_scale,
                )
                # PAI reloads ``best_model`` on this switch.  With the best
                # score cleared, check_new_best saves the current weights as
                # that file first, so the reload is a no-op.
                GPA.pai_tracker.member_vars["current_best_validation_score"] = 0
                GPA.pai_tracker.member_vars["global_best_validation_score"] = 0
            if force_to_neuron:
                # Read before the switch resets the candidate bookkeeping.
                extra["pb_scores_at_integration"] = _pb_scores()
            with _pai_cwd(run_dir):
                model, restructured, _complete = GPA.pai_tracker.add_validation_score(
                    val_acc, model, force_switch=force_to_candidate or force_to_neuron
                )
            model = model.to(device)
            if segment == "candidate" and current_pai_mode() == "p":
                # PAI refreshes PB correlation scores inside add_validation_score
                # and resets them at the p->n switch, so they are read here.
                pb_history.append({"epoch": epoch, "scores": _pb_scores()})

            if force_to_candidate:
                if not restructured or current_pai_mode() != "p":
                    raise GrowInvariantError(
                        f"forced n->p switch did not enter dendrite mode "
                        f"(restructured={restructured}, mode={current_pai_mode()!r})"
                    )
                rewind = max_abs_change(before_switch, model)
                checks["n_to_p_base_max_abs_change"] = rewind
                if rewind > 0:
                    raise GrowInvariantError(
                        f"PAI rewound the base at the n->p switch (max change {rewind:.3g})"
                    )
                optimizer = _build_candidate_optimizer(model, grow.candidate_optimizer)
                GPA.pai_tracker.set_optimizer_instance(optimizer)
                live_base = base_params_in_optimizer(model, optimizer)
                checks["base_params_in_optimizer_candidate_phase"] = live_base
                if live_base > 0:
                    raise GrowInvariantError(
                        f"{live_base} base parameters remain in the candidate optimizer"
                    )
                frozen_base = base_state_snapshot(model)
                logger.info("Candidate phase: base frozen, %s", _candidate_description(model))
            elif force_to_neuron:
                if not restructured or current_pai_mode() != "n":
                    raise GrowInvariantError(
                        f"forced p->n switch did not return to neuron mode "
                        f"(restructured={restructured}, mode={current_pai_mode()!r})"
                    )
                added = int(GPA.pai_tracker.member_vars["num_dendrites_added"])
                if added != 1:
                    raise GrowInvariantError(f"expected one dendrite, PAI reports {added}")
                drift = max_abs_change(frozen_base, model)
                checks["candidate_phase_base_max_abs_drift"] = drift
                if drift > 0:
                    raise GrowInvariantError(
                        f"the base moved during the candidate phase (max drift {drift:.3g})"
                    )
                if grow.sham:
                    zeroed_tensors, zeroed_values = zero_and_freeze_dendrite(model)
                    logger.warning(
                        "SHAM: zeroed and froze %d dendrite tensors (%d values, skip weights "
                        "included); post-switch training updates the base only",
                        zeroed_tensors, zeroed_values,
                    )
                optimizer, _base_parameters, dendrite_parameters = build_post_switch_optimizer(
                    model, train_cfg, grow
                )
                GPA.pai_tracker.set_optimizer_instance(optimizer)
                restored = (
                    restore_optimizer_state_by_name(model, optimizer, captured_state)
                    if grow.carry_momentum else 0
                )
                checks["momentum_buffers_restored"] = restored
                checks["momentum_buffers_expected"] = len(captured_state)
                if restored != len(captured_state):
                    logger.error(
                        "restored optimizer state for %d of %d base parameters",
                        restored, len(captured_state),
                    )
                checks["base_params_in_optimizer_post_switch"] = base_params_in_optimizer(
                    model, optimizer
                )
                dendrite_live = optimizer_numel(optimizer, dendrite_parameters)
                checks["dendrite_params_in_optimizer_post_switch"] = dendrite_live
                if grow.sham:
                    if dendrite_live != 0:
                        raise GrowInvariantError(
                            f"{dendrite_live} dendrite parameters are in a sham run's "
                            "post-switch optimizer"
                        )
                elif dendrite_live == 0:
                    raise GrowInvariantError("no dendrite parameters are in the post-switch optimizer")
                scheduler = build_base_scheduler(
                    optimizer, train_cfg, total_base_steps, start_step=base_step
                )
                if probe_at_switch is None or logits_at_switch is None:
                    raise GrowInvariantError("no integration probe was taken at the n->p switch")
                integration_diff = float(
                    (probe_logits(model, probe_at_switch, device) - logits_at_switch).abs().max()
                )
                checks["integration_output_max_abs_diff"] = integration_diff
                if grow.sham:
                    if integration_diff != 0.0:
                        raise GrowInvariantError(
                            "the zeroed sham dendrite changed the network's output "
                            f"(max logit change {integration_diff:.3g})"
                        )
                elif integration_diff > INTEGRATION_WARN_TOLERANCE:
                    logger.warning(
                        "integrating the dendrite moved the probe logits by up to %.3g "
                        "(> %.0e) before any post-switch step",
                        integration_diff, INTEGRATION_WARN_TOLERANCE,
                    )
                logger.info(
                    "Dendrite integrated at base step %d: %d params, LR resumes at %.6g, "
                    "probe logits moved by %.3g",
                    base_step, UPA.count_params(model), optimizer.param_groups[0]["lr"],
                    integration_diff,
                )
            elif restructured:
                raise GrowInvariantError(f"PAI restructured the network unprompted at epoch {epoch}")

            if (
                grow.max_wall_clock_minutes is not None
                and monotonic() - started > grow.max_wall_clock_minutes * 60
                and epoch < grow.total_epochs
            ):
                raise WallClockExceeded(
                    f"stopped after epoch {epoch}: over the "
                    f"{grow.max_wall_clock_minutes:g}-minute cap"
                )

        if grow.sham:
            sham_skip = dendrite_skip_weight_max_abs(model)
            checks["sham_skip_weight_max_abs_final"] = sham_skip
            if sham_skip != 0.0:
                raise GrowInvariantError(
                    f"a sham run's dendrite skip weights moved off zero (max |w| {sham_skip:.3g})"
                )
        candidate_values = [r["val_acc"] for r in history if r["segment"] == "candidate"]
        checks["candidate_phase_val_acc_span"] = max(candidate_values) - min(candidate_values)
        extra["dendrite"] = {
            "num_dendrites_added": int(GPA.pai_tracker.member_vars["num_dendrites_added"]),
            "skip_weight_mean_abs": _skip_weight_mean_abs(model),
        }

        # Export the last-epoch graph, then the best post-switch epoch.
        probe = _parity_probe(val_loader, device)
        final_clean, checks["clean_parity_max_abs_diff_final"] = _export_with_parity(
            model, run_dir, run_dir, probe, run_id, device, grow.dendrite_input_scale
        )
        artifact_paths = {"final_clean": layout.relative(run_dir / "final_clean_pai.pt")}
        clean_models = {"final": final_clean}
        if best_post_state is not None:
            checks["best_reload_skipped_keys"] = load_weights_for_export(model, best_post_state)
            extra["dendrite"]["skip_weight_mean_abs_best"] = _skip_weight_mean_abs(model)
            best_dir = run_dir / "best_dendritic"
            clean_models["best"], checks["clean_parity_max_abs_diff_best"] = _export_with_parity(
                model, run_dir, best_dir, probe, run_id, device, grow.dendrite_input_scale
            )
            artifact_paths["best_clean"] = layout.relative(best_dir / "final_clean_pai.pt")
        extra["artifact_paths"] = artifact_paths
        for key in ("clean_parity_max_abs_diff_final", "clean_parity_max_abs_diff_best"):
            if checks.get(key, 0.0) > PARITY_TOLERANCE:
                logger.error("%s = %.3g exceeds %.0e", key, checks[key], PARITY_TOLERANCE)
        # The exports hold the input scale folded into their weights, so from
        # here on every forward is a clean graph under the plain function.
        GPA.pc.set_pai_forward_function(plain_forward_function())
        extra["dendrite_diagnostics"] = diagnose_exports(
            clean_models, val_loader, device, summarize_history(history, grow), checks
        )

        deployment = train_cfg.get("deployment", {})
        cpu = torch.device("cpu")
        profile_args = {
            "device": cpu,
            "bits_per_weight": int(deployment.get("bits_per_weight", 32)),
            "latency_iterations": int(deployment.get("latency_iterations", 50)),
        }
        extra["cost"] = {
            "base": profile_model(
                build_model(model_cfg, input_shape, num_classes), input_shape, **profile_args
            ).as_dict(),
            "deployed": profile_model(
                copy.deepcopy(final_clean).to(cpu), input_shape, **profile_args
            ).as_dict(),
        }
        extra["input_shape"] = list(input_shape)
        extra["num_classes"] = num_classes
        summary = write_summary("complete")
        logger.info(
            "Grow run complete: best post-switch val %.4f, final %.4f, %d -> %d params, "
            "%d -> %d MACs",
            summary["results"]["best_val_acc_post_switch"] or float("nan"),
            summary["results"]["final_val_acc"] or float("nan"),
            summary["cost"]["base"]["params"], summary["cost"]["deployed"]["params"],
            summary["cost"]["base"]["macs"], summary["cost"]["deployed"]["macs"],
        )
        return summary
    except WallClockExceeded as exc:
        write_summary("incomplete", f"wall_clock_cap_exceeded: {exc}")
        raise
    except Exception as exc:
        write_summary("incomplete", f"error: {type(exc).__name__}: {exc}")
        raise


def _build_candidate_optimizer(
    model: nn.Module, spec: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Optimizer over every parameter; PAI's filter keeps only the candidate."""
    parameters = list(model.parameters())
    if spec["name"] == "adamw":
        return torch.optim.AdamW(parameters, lr=spec["lr"], weight_decay=spec["weight_decay"])
    return torch.optim.SGD(
        parameters, lr=spec["lr"], momentum=spec.get("momentum", 0.9),
        weight_decay=spec["weight_decay"],
    )


def _candidate_description(model: nn.Module) -> str:
    groups = {"base": 0, "dendrite": 0}
    for name, parameter in model.named_parameters():
        groups["dendrite" if _is_dendrite_parameter(name, model) else "base"] += parameter.numel()
    return f"{groups['base']} base / {groups['dendrite']} dendrite-side parameters in the graph"


def _parity_probe(val_loader, device: torch.device) -> torch.Tensor:
    batches = []
    count = 0
    for features, _labels in val_loader:
        batches.append(features)
        count += features.shape[0]
        if count >= PARITY_PROBE_EXAMPLES:
            break
    return torch.cat(batches)[:PARITY_PROBE_EXAMPLES].to(device)


def _export_with_parity(
    model: nn.Module,
    run_dir: Path,
    save_dir: Path,
    probe: torch.Tensor,
    run_id: str,
    device: torch.device,
    dendrite_input_scale: float = 1.0,
) -> tuple[nn.Module, float]:
    """Export a clean inference graph and measure its output parity.

    The trained graph runs under the run's (possibly scaled) forward function;
    the export has the scale folded in and runs under the plain one.
    """
    model.eval()
    with torch.no_grad():
        reference = model(probe).detach().cpu()
    save_dir.mkdir(parents=True, exist_ok=True)
    with _pai_cwd(run_dir):
        clean = export_final_pai_model(
            model, str(save_dir), run_id=run_id, dendrite_input_scale=dendrite_input_scale
        )
    clean = clean.to(device).eval()
    with torch.no_grad(), pai_forward_function(plain_forward_function()):
        exported = clean(probe).detach().cpu()
    return clean, float((reference - exported).abs().max())


def diagnose_exports(
    clean_models: Mapping[str, nn.Module],
    val_loader,
    device: torch.device,
    results: Mapping[str, Any],
    checks: dict[str, Any],
) -> dict[str, Any]:
    """``dendrite_diagnostics`` of each exported clean model; advisory only.

    Each entry gains ``device``.  ``checks`` gains
    ``diagnostics_val_acc_on_minus_<label>``: the clean model's dendrite-on
    accuracy minus the accuracy the live graph recorded for the same weights,
    ~0 because clean parity is exact on one device.  A failure anywhere is
    returned as ``{"error": message}``, adds no checks, and never ends the run.
    """
    try:
        diagnostics: dict[str, Any] = {}
        agreement: dict[str, float] = {}
        for label, clean in clean_models.items():
            report = dict(dendrite_diagnostics(clean, val_loader, device))
            report["device"] = str(device)
            diagnostics[label] = report
            logger.info(
                "Diagnostics (%s): val acc dendrite on %.4f / off %.4f over %d samples",
                label, report["val_acc_dendrite_on"], report["val_acc_dendrite_off"],
                report["n_samples"],
            )
            reference = results.get(DIAGNOSTIC_REFERENCES.get(label, ""))
            if reference is None:
                continue
            difference = float(report["val_acc_dendrite_on"]) - float(reference)
            agreement[f"diagnostics_val_acc_on_minus_{label}"] = difference
            tolerance = 1.0 / max(int(report["n_samples"]), 1)
            if abs(difference) > tolerance:
                logger.warning(
                    "clean %s model's dendrite-on val acc differs from the recorded "
                    "%s by %.4g (> 1/n_samples = %.3g)",
                    label, DIAGNOSTIC_REFERENCES[label], difference, tolerance,
                )
    except Exception as exc:  # noqa: BLE001 - diagnostics are advisory, not a gate
        logger.warning("dendrite diagnostics failed; the run continues", exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}
    checks.update(agreement)
    return diagnostics


def _load_yaml(path: str) -> dict:
    with open(path) as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SparkNet from scratch and grow one PAI dendrite mid-run."
    )
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2_mfcc32_paper.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", default="configs/train/sparknet_grow_dendrites_paper.yaml")
    parser.add_argument("--placement", default=None, help="key of grow_dendrites.placements")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--switch-epoch", type=int, default=None)
    parser.add_argument("--candidate-epochs", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument(
        "--dendrite-weight-decay", type=float, default=None,
        help="override grow_dendrites.dendrite_weight_decay, the weight decay of the "
        "post-switch optimizer's dendrite group",
    )
    parser.add_argument(
        "--sham", action="store_true",
        help="noise-floor arm: grow the dendrite as usual, then zero and freeze it at "
        "the p->n switch so post-switch training updates the base only",
    )
    parser.add_argument(
        "--dendrite-input-scale", type=float, default=None,
        help="override grow_dendrites.dendrite_input_scale: PAI's forward function runs "
        "as f(z / C) on the dendrite pre-activation (folded into the exported weights)",
    )
    parser.add_argument(
        "--arm", default=None,
        help="label recorded in the summary ([A-Za-z0-9._-]+); default: the placement, "
        f"with '{SHAM_SUFFIX}' appended for --sham",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")

    data_cfg = _load_yaml(args.data_config)
    model_cfg = _load_yaml(args.model_config)
    train_cfg = _load_yaml(args.train_config)
    try:
        grow = resolve_grow_config(
            train_cfg,
            placement=args.placement,
            switch_epoch=args.switch_epoch,
            candidate_epochs=args.candidate_epochs,
            max_minutes=args.max_minutes,
            dendrite_weight_decay=args.dendrite_weight_decay,
            sham=args.sham,
            arm=args.arm,
            dendrite_input_scale=args.dendrite_input_scale,
        )
    except ValueError as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir)
    reason = existing_run_reason(output_dir)
    if reason is not None:
        print(
            f"refusing to start: {reason}. Runs do not resume; move the directory "
            "aside to retrain it.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    with run_session(
        output_dir,
        command="kws.optimize.sparknet_grow_dendrites",
        argv=list(sys.argv if argv is None else ["sparknet_grow_dendrites", *argv]),
        seed=args.seed,
        inputs=[
            (args.data_config, "data_config"),
            (args.model_config, "model_config"),
            (args.train_config, "train_config"),
        ],
    ):
        try:
            run_grow(
                data_cfg,
                model_cfg,
                train_cfg,
                grow,
                output_dir=output_dir,
                seed=args.seed,
                config_paths={
                    "data_config": args.data_config,
                    "model_config": args.model_config,
                    "train_config": args.train_config,
                },
            )
        except WallClockExceeded as exc:
            logger.error("%s", exc)
            return EXIT_WALL_CLOCK
    return 0


if __name__ == "__main__":
    sys.exit(main())
