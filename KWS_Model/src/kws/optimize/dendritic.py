"""Framework steps 3c-3e: the perforation/dendrite phase and its KD resume.

One candidate enters here already pruned and KD-fine-tuned (steps 3a-3b) and
leaves with a recorded accuracy and deployment cost. In between:

- **3c, perforation.** PerforatedAI freezes the base weights, trains candidate
  dendritic residual nodes against the residual correlation, then selects the
  useful ones and freezes them into the network. The phase trail this module
  records makes that freeze/select/freeze cycle auditable after the fact rather
  than taking the library's word for it.
- **3d, resume.** The dendrites are fixed; the base student weights are not yet
  adapted to them. So the clean deployment graph resumes KD fine-tuning on its
  active parameters, with the same fixed teacher used everywhere else.
- **3e, record.** Validation accuracy plus latency, memory, and compute, so the
  step-3f Pareto rule has all four axes to compare candidates on.

The test split is never loaded or consulted by this module.
"""

import argparse
import csv
import hashlib
import importlib
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, cast

import torch
import torch.nn as nn
import yaml

# PerforatedAI is distributed as compiled extension modules without typing
# stubs. Keep its dynamic API behind one narrow, explicit boundary.
GPA: Any = importlib.import_module("perforatedai.globals_perforatedai")
UPA: Any = importlib.import_module("perforatedai.utils_perforatedai")

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.models.ds_cnn import FeatureModel
from kws.models.ds_cnn import DSCNN
from kws.models.layers import DSConvBlock
from kws.optimize.kd import (
    DistillationCriterion,
    FrozenTeacher,
    KDWeights,
    file_sha256,
    supports_pooled_features,
)
from kws.optimize.prune import prune_ds_cnn
from kws.train import (
    build_lr_scheduler,
    evaluate_loss_acc,
    keep_frozen_batchnorm_eval,
    resolve_manifest_run_id,
    run_finetune,
    set_train_mode_preserving_frozen_batchnorm,
    train_model,
    validate_checkpoint_run_id,
)
from kws.utils.device import get_device
from kws.utils.artifacts import ArtifactLayout, sha256_path
from kws.utils.checkpointing import (
    MetricsRecorder,
    atomic_torch_save,
    capture_rng_state,
    move_optimizer_state_to_device,
    restore_rng_state,
)
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
from kws.utils.profile import profile_model
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)

FRAMEWORK_CYCLE_VERSION = 3

PAI_SIDECAR_KEYS = frozenset(
    {
        "format_version",
        "kind",
        "stage",
        "phase",
        "run_id",
        "completed_epoch",
        "next_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "kd_state_dict",
        "adapter_optimizer_state_dict",
        "adapter_scheduler_state_dict",
        "phase_trail",
        "global_step",
        "history_length",
        "last_metric_digest",
        "pai_run_dir",
        "native_pai_latest",
        "native_pai_latest_sha256",
        "teacher_checkpoint",
        "recipe_fingerprint",
        "rng_state",
        "loader_generator_states",
    }
)


def cycle_fingerprint(
    checkpoint_path: str,
    teacher_checkpoint: str | None,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
) -> str:
    """Stable identity for every input that can change one candidate run."""
    encoded = yaml.safe_dump(
        {
            "framework_cycle_version": FRAMEWORK_CYCLE_VERSION,
            # The source checkpoint is identified by content so moving a
            # complete run root does not change the resume recipe.
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_sha256": (
                file_sha256(teacher_checkpoint) if teacher_checkpoint else None
            ),
            "data": data_cfg,
            "model": model_cfg,
            "train": train_cfg,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_pai_resume_sidecar(
    run_dir: Path,
    canonical_sidecar: Path,
    *,
    resume: bool | None,
    resume_from: str | Path | None,
) -> Path | None:
    """Resolve explicit/automatic PAI recovery without overwriting artifacts.

    Standalone callers pass ``True`` or ``False`` and therefore resume only
    when the user asked. Internal pipeline/search callers retain ``None`` as
    the compatibility behavior: a canonical sidecar is resumed automatically.
    In every mode, a nonempty candidate without selected recovery state is an
    error rather than permission to let PAI overwrite its native files.
    """
    if resume is True and resume_from is not None:
        raise ValueError("use only one of resume=True or resume_from")

    selected: Path | None = None
    if resume_from is not None:
        selected = Path(resume_from).expanduser().resolve()
    elif resume is True:
        selected = canonical_sidecar.resolve()
    elif resume is None and canonical_sidecar.is_file():
        selected = canonical_sidecar.resolve()

    if selected is not None:
        if not selected.is_file():
            raise FileNotFoundError(f"PAI resume sidecar not found: {selected}")
        return selected

    has_native_artifacts = run_dir.is_dir() and any(run_dir.iterdir())
    if has_native_artifacts or canonical_sidecar.exists():
        raise FileExistsError(
            f"PAI candidate {run_dir} is nonempty but no resume checkpoint was "
            "selected; use --resume for its canonical KWS sidecar or "
            "--resume-from PATH. If the sidecar is missing, preserve or move "
            "the native candidate and start with a new --save-name."
        )
    return None


def _load_pai_sidecar(
    sidecar_path: Path,
    run_dir: Path,
    recipe_fingerprint: str,
    device: torch.device,
    expected_run_id: str | None = None,
) -> dict:
    """Load and validate one complete KWS/native PAI recovery pair."""
    state = torch.load(sidecar_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"PAI sidecar {sidecar_path} is not a mapping")

    missing = PAI_SIDECAR_KEYS.difference(state)
    unexpected = set(state).difference(PAI_SIDECAR_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise ValueError(
            f"PAI sidecar {sidecar_path} has an incompatible schema "
            f"({'; '.join(details)})"
        )
    if state["format_version"] != FRAMEWORK_CYCLE_VERSION:
        raise ValueError(
            f"PAI sidecar {sidecar_path} uses format "
            f"{state['format_version']!r}; expected {FRAMEWORK_CYCLE_VERSION}"
        )
    if (state["kind"], state["stage"], state["phase"]) != (
        "kws_pai_training_state",
        "sparsity",
        "pai",
    ):
        raise ValueError(f"PAI sidecar {sidecar_path} has incompatible identity fields")
    validate_checkpoint_run_id(state, expected_run_id, source=sidecar_path)

    expected_run_dir = run_dir.resolve()
    recorded_run_raw = Path(state["pai_run_dir"])
    if recorded_run_raw.is_absolute():
        recorded_run_dir = recorded_run_raw.expanduser().resolve()
    else:
        recorded_run_dir = Path(state["pai_run_dir"])
        expected_relative = Path("pai") / "candidates" / expected_run_dir.name
        if recorded_run_dir != expected_relative:
            raise ValueError(
                f"PAI sidecar {sidecar_path} belongs to {recorded_run_dir}, "
                f"not {expected_relative}"
            )
        recorded_run_dir = expected_run_dir
    if recorded_run_dir != expected_run_dir:
        raise ValueError(
            f"PAI sidecar {sidecar_path} belongs to {recorded_run_dir}, "
            f"not {expected_run_dir}"
        )
    if state["recipe_fingerprint"] != recipe_fingerprint:
        raise ValueError(
            f"PAI sidecar {sidecar_path} does not match the requested cycle recipe"
        )

    completed_epoch = int(state["completed_epoch"])
    if completed_epoch < 1 or int(state["next_epoch"]) != completed_epoch + 1:
        raise ValueError(f"PAI sidecar {sidecar_path} has invalid epoch metadata")
    if int(state["global_step"]) < 0 or int(state["history_length"]) < 0:
        raise ValueError(f"PAI sidecar {sidecar_path} has invalid progress metadata")
    if not isinstance(state["phase_trail"], list) or not state["phase_trail"]:
        raise ValueError(f"PAI sidecar {sidecar_path} has no phase trail")
    if not isinstance(state["loader_generator_states"], list) or len(
        state["loader_generator_states"]
    ) != 2:
        raise ValueError(
            f"PAI sidecar {sidecar_path} must contain two loader generator states"
        )

    expected_native = (run_dir / "latest.pt").resolve()
    recorded_native_raw = Path(state["native_pai_latest"])
    if recorded_native_raw.is_absolute():
        recorded_native = recorded_native_raw.expanduser().resolve()
    else:
        run_root = expected_run_dir.parent.parent.parent
        recorded_native = (run_root / recorded_native_raw).resolve()
    if recorded_native != expected_native:
        raise ValueError(
            f"PAI sidecar {sidecar_path} references native state "
            f"{recorded_native}, expected {expected_native}"
        )
    native_digest = state["native_pai_latest_sha256"]
    if not isinstance(native_digest, str) or len(native_digest) != 64:
        raise ValueError(f"PAI sidecar {sidecar_path} has no valid native digest")
    if not expected_native.is_file() or expected_native.is_symlink():
        raise ValueError(
            f"PAI native latest checkpoint is missing or unsafe: {expected_native}"
        )
    if sha256_path(expected_native) != native_digest:
        raise ValueError(
            f"PAI native latest state does not match sidecar {sidecar_path}; "
            "refusing to resume an unpaired candidate"
        )
    native_state = torch.load(expected_native, map_location="cpu", weights_only=False)
    validate_checkpoint_run_id(native_state, state["run_id"], source=expected_native)
    return state


def _restore_pai_training_state(
    state: dict,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    kd: DistillationCriterion | None,
    adapter_optimizer: torch.optim.Optimizer | None,
    adapter_scheduler: Any,
    device: torch.device,
) -> None:
    """Restore graph and KD state before either optimizer consumes its state."""
    model.load_state_dict(state["model_state_dict"], strict=True)

    kd_state = state["kd_state_dict"]
    if kd is None:
        if kd_state is not None:
            raise ValueError("PAI sidecar contains KD state but this run has no teacher")
    else:
        if kd_state is None:
            raise ValueError("PAI sidecar is missing required KD adapter state")
        kd.load_state_dict(kd_state, strict=True)

    optimizer.load_state_dict(state["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(state["scheduler_state_dict"])

    adapter_optimizer_state = state["adapter_optimizer_state_dict"]
    adapter_scheduler_state = state["adapter_scheduler_state_dict"]
    if adapter_optimizer is None:
        if adapter_optimizer_state is not None or adapter_scheduler_state is not None:
            raise ValueError(
                "PAI sidecar contains adapter optimizer state but this KD recipe "
                "has no trainable adapter"
            )
        return
    if adapter_optimizer_state is None or adapter_scheduler_state is None:
        raise ValueError("PAI sidecar is missing adapter optimizer/scheduler state")
    adapter_optimizer.load_state_dict(adapter_optimizer_state)
    move_optimizer_state_to_device(adapter_optimizer, device)
    if adapter_scheduler is None:
        raise ValueError("PAI run created an adapter optimizer without its scheduler")
    adapter_scheduler.load_state_dict(adapter_scheduler_state)


def _pai_epoch_record(
    *,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    train_losses: dict[str, float],
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    optimizer: torch.optim.Optimizer,
    adapter_optimizer: torch.optim.Optimizer | None,
    mode: str,
    restructured: bool,
    model: nn.Module,
    seed: int,
) -> dict:
    """Build the canonical PAI epoch record, including all KD components."""
    model_lrs = [group["lr"] for group in optimizer.param_groups]
    adapter_lrs = (
        [group["lr"] for group in adapter_optimizer.param_groups]
        if adapter_optimizer is not None
        else []
    )
    return {
        "schema_version": 1,
        "stage": "sparsity",
        "phase": "pai",
        "epoch": epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
        "train_loss": train_losses["total"],
        **{
            f"train_{name}": value
            for name, value in train_losses.items()
            if name != "total"
        },
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_acc": val_accuracy,
        "val_accuracy": val_accuracy,
        "learning_rates": model_lrs,
        "learning_rate": model_lrs,
        "adapter_learning_rates": adapter_lrs,
        "optimizer_learning_rates": {
            "model": model_lrs,
            "kd_adapter": adapter_lrs,
        },
        "seed": seed,
        "pai_mode": mode,
        "restructured": restructured,
        "parameter_count": UPA.count_params(model),
        **describe_learning_phase(model),
    }


def _save_pai_restart_pair(
    *,
    model: nn.Module,
    run_dir: Path,
    pai_run_name: str,
    sidecar_path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    kd: DistillationCriterion | None,
    adapter_optimizer: torch.optim.Optimizer | None,
    adapter_scheduler: Any,
    phase_trail: list[dict],
    completed_epoch: int,
    global_step: int,
    history_length: int,
    last_metric_digest: str | None,
    teacher_checkpoint: str | None,
    recipe_fingerprint: str,
    loaders: tuple[Any, Any],
    pai_run_dir_ref: str | None = None,
    native_pai_latest_ref: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Commit native PAI state first, then atomically attest it in the sidecar."""
    run_id = run_id or uuid.uuid4().hex
    save_system = getattr(UPA, "save_system", None)
    if not callable(save_system):
        raise RuntimeError("installed PerforatedAI has no supported save_system API")
    save_system(model, str(run_dir.parent), pai_run_name)

    native_latest = (run_dir / "latest.pt").resolve()
    if not native_latest.is_file() or native_latest.is_symlink():
        raise RuntimeError(
            f"PerforatedAI did not create the required native checkpoint: {native_latest}"
        )
    native_state = torch.load(native_latest, map_location="cpu", weights_only=False)
    if not isinstance(native_state, dict):
        raise RuntimeError(
            f"PerforatedAI native checkpoint is not a mapping: {native_latest}"
        )
    if "run_id" in native_state and native_state["run_id"] != run_id:
        raise ValueError(
            f"native PAI checkpoint {native_latest} belongs to "
            f"run_id={native_state.get('run_id')!r}, expected {run_id!r}"
        )
    native_state["run_id"] = run_id
    atomic_torch_save(native_latest, native_state)
    state = {
        "format_version": FRAMEWORK_CYCLE_VERSION,
        "kind": "kws_pai_training_state",
        "stage": "sparsity",
        "phase": "pai",
        "run_id": run_id,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "kd_state_dict": kd.state_dict() if kd is not None else None,
        "adapter_optimizer_state_dict": (
            adapter_optimizer.state_dict() if adapter_optimizer is not None else None
        ),
        "adapter_scheduler_state_dict": (
            adapter_scheduler.state_dict() if adapter_scheduler is not None else None
        ),
        "phase_trail": phase_trail,
        "global_step": global_step,
        "history_length": history_length,
        "last_metric_digest": last_metric_digest,
        "pai_run_dir": pai_run_dir_ref or str(run_dir.resolve()),
        "native_pai_latest": native_pai_latest_ref or str(native_latest),
        "native_pai_latest_sha256": sha256_path(native_latest),
        "teacher_checkpoint": teacher_checkpoint,
        "recipe_fingerprint": recipe_fingerprint,
        "rng_state": capture_rng_state(),
        "loader_generator_states": [
            loader.generator.get_state().clone()
            if getattr(loader, "generator", None) is not None
            else None
            for loader in loaders
        ],
    }
    atomic_torch_save(sidecar_path, state)
    return state


# Parameters PerforatedAI adds are named for the structure they belong to.
# Classifying by name is what lets the phase trail below separate "base student
# weights" from "dendritic residual nodes" without depending on PAI internals.
DENDRITE_PARAMETER_MARKERS = ("dendrite", "candidate", "to_top", "pb_")


@dataclass(frozen=True)
class DendriticCycleResult:
    """Deployment-relevant result from one completed PAI cycle."""

    save_name: str
    block_channels: list[int]
    base_params: int
    deployed_params: int
    best_val_acc: float
    epochs: int
    elapsed_seconds: float
    pai_deployed_params: int | None = None
    cost: dict | None = None
    distillation: dict | None = None
    resume: dict | None = None
    prune_finetune: dict | None = None
    phase_trail: list[dict] = field(default_factory=list)
    run_id: str | None = None


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cycle_base(
    checkpoint_path: str,
    keep_ratio: float,
    target_model_cfg: dict | None = None,
    expected_run_id: str | None = None,
):
    """Load a trained DS-CNN and structurally prune its block channels."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_run_id(checkpoint, expected_run_id, source=checkpoint_path)
    model = build_ds_cnn(
        checkpoint["model_cfg"],
        tuple(checkpoint["input_shape"]),
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    pruned = prune_ds_cnn(model, keep_ratio)

    generated_channels = [
        block.pointwise.out_channels for block in pruned.blocks
    ]
    model_cfg = dict(checkpoint["model_cfg"])
    model_cfg["name"] = f'{model_cfg["name"]}_dendritic_cycle1_base'
    model_cfg["block_channels"] = generated_channels

    if target_model_cfg is not None:
        stem_conv = cast(nn.Conv2d, pruned.stem[0])
        if target_model_cfg["initial_channels"] != stem_conv.out_channels:
            raise ValueError("XXS initial_channels does not match the pruned model")
        if target_model_cfg["block_channels"] != generated_channels:
            raise ValueError(
                "XXS block_channels does not match the pruned model: "
                f'{target_model_cfg["block_channels"]} != {generated_channels}'
            )
        model_cfg = dict(target_model_cfg)
    return pruned, checkpoint, model_cfg


def estimate_one_dendrite_params(model: DSCNN) -> int:
    """Estimate deployed parameters after one dendrite on blocks and head.

    Each selected module is copied once. PAI folds its branch connections
    during deployment cleanup, so temporary candidates and connection
    scaffolding are excluded from the deployed parameter count.
    """
    selected_modules = [*model.blocks, model.fc]
    dendrite_params = sum(
        parameter.numel()
        for module in selected_modules
        for parameter in module.parameters()
    )
    return (
        sum(parameter.numel() for parameter in model.parameters())
        + dendrite_params
    )


def read_pai_architecture_results(save_name: str) -> tuple[float, int]:
    """Return the best validation score and its deployed parameter count.

    PAI's architecture summary contains neuron-mode maxima only, so it avoids
    treating the temporary candidate-correlation phase as a deployable model.
    """
    path = Path(save_name) / f"{Path(save_name).name}_best_arch_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"PAI architecture results not found: {path}")

    rows: list[tuple[float, int]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append((float(row["Max Valid Scores"]), int(row["Param Counts"])))
    if not rows:
        raise ValueError(f"PAI architecture results are empty: {path}")
    best_val_acc, deployed_params = max(rows, key=lambda item: item[0])
    return best_val_acc, deployed_params


def _collect_dendrite_top_weights(model: nn.Module) -> dict[str, list[torch.Tensor]]:
    """Copy PAI's learned dendrite-to-output coefficients before cleanup.

    ``prepare_final_model`` deep-copies the network and, in the installed PAI
    release, drops the only ``skip_weights`` entry when a module has exactly
    one integrated dendrite.  Those coefficients are required for inference,
    so collect them while the full PAI modules are still available.
    """
    top_weights: dict[str, list[torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "dendrites_to_top"):
            continue
        weights = getattr(module, "dendrites_to_top")
        if len(weights):
            top_weights[name] = [weight.detach().cpu().clone() for weight in weights]
    return top_weights


def _restore_single_dendrite_skip_weights(
    clean_model: nn.Module,
    top_weights: dict[str, list[torch.Tensor]],
) -> int:
    """Restore one-dendrite skip coefficients omitted by PAI cleanup.

    PAI's clean wrapper stores the dendrite branch first and the neuron branch
    second.  With one dendrite, its clean-wrapper constructor intentionally
    omits ``skip_weights``; adding the saved coefficient makes the forward
    pass compute ``neuron + coefficient * dendrite`` again.
    """
    restored = 0
    for name, weights in top_weights.items():
        if len(weights) != 1:
            # Multi-dendrite cleanup has its own skip-weight handling.  The
            # current experiments deliberately deploy one dendrite only.
            continue
        try:
            clean_module = clean_model.get_submodule(name)
        except AttributeError as exc:
            raise RuntimeError(
                f"PAI cleanup removed expected module {name!r}; refusing to "
                "write an unverifiable final checkpoint"
            ) from exc
        layer_array = getattr(clean_module, "layer_array", None)
        if not isinstance(layer_array, nn.ModuleList) or len(layer_array) < 2:
            raise RuntimeError(
                f"PAI cleanup module {name!r} has no two-branch layer_array"
            )
        if hasattr(clean_module, "skip_weights"):
            continue
        clean_module.skip_weights = nn.ParameterList(
            [nn.Parameter(weights[0].clone(), requires_grad=False)]
        )
        restored += 1
    return restored


def export_final_pai_model(
    model: nn.Module, save_name: str, *, run_id: str | None = None
) -> nn.Module:
    """Write a clean, inference-ready PAI checkpoint with branch parity.

    PerforatedAI writes ``final_clean_pai.pt`` automatically when a cycle
    ends.  That release omits the single-dendrite skip coefficients, so this
    wrapper repeats the cleanup, restores those coefficients, and atomically
    replaces the library-generated file.  The returned model is the exact
    model represented by the saved tensors and is useful for parity checks.
    """
    if run_id is None:
        candidate = Path(save_name).expanduser().resolve()
        for parent in (candidate, *candidate.parents):
            if (parent / "manifest.yaml").is_file():
                run_id = ArtifactLayout(parent).manifest_run_id
                break
    top_weights = _collect_dendrite_top_weights(model)
    clean_model = UPA.prepare_final_model(model)
    restored = _restore_single_dendrite_skip_weights(clean_model, top_weights)
    clean_model.eval()

    path = Path(save_name) / "final_clean_pai.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in clean_model.state_dict().items()
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        UPA.save_file(
            tensors,
            temporary,
            metadata={
                "format": "perforatedai_clean_inference",
                "single_dendrite_skip_weights_restored": str(restored),
                "run_id": run_id,
            },
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(
        "Exported parity-checked PAI model to %s (restored %d one-dendrite "
        "skip coefficient sets)",
        path,
        restored,
    )
    return clean_model


def ensure_clean_dendrite_skip_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Add PAI's optional clean-graph skip coefficients before loading a state.

    PAI omits ``skip_weights`` when cleaning a module with one dendrite, while
    :func:`export_final_pai_model` restores those coefficients because they are
    part of the deployed residual computation.  A later process reconstructs
    the graph through PAI's loader, so it must recreate the optional
    ``ParameterList`` before loading either the clean artifact or the resumed
    KD checkpoint.  The helper is deliberately driven by the checkpoint keys,
    which keeps old PAI artifacts without skip coefficients loadable too.
    """
    device = next(model.parameters()).device
    grouped: dict[str, list[tuple[int, torch.Tensor]]] = {}
    marker = ".skip_weights."
    for key, tensor in state_dict.items():
        if marker not in key:
            continue
        module_name, index_text = key.rsplit(marker, 1)
        if not index_text.isdigit():
            continue
        grouped.setdefault(module_name, []).append((int(index_text), tensor))

    for module_name, entries in grouped.items():
        try:
            module = model.get_submodule(module_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"clean checkpoint references missing PAI module {module_name!r}"
            ) from exc
        skip_weights = getattr(module, "skip_weights", None)
        if not isinstance(skip_weights, nn.ParameterList):
            skip_weights = nn.ParameterList()
            module.skip_weights = skip_weights
        for index, tensor in sorted(entries):
            while len(skip_weights) <= index:
                skip_weights.append(
                    nn.Parameter(
                        torch.zeros_like(tensor, device=device),
                        requires_grad=False,
                    )
                )
            if skip_weights[index].shape != tensor.shape:
                raise RuntimeError(
                    f"skip coefficient shape mismatch for {module_name!r}: "
                    f"{skip_weights[index].shape} != {tensor.shape}"
                )
            skip_weights[index].requires_grad_(False)


def _is_dendrite_parameter(name: str, model: nn.Module | None = None) -> bool:
    """Recognize PAI residual parameters in both training and clean graphs.

    PAI uses different names before and after cleanup. During training a
    residual lives below ``dendrite_module`` (while its
    ``dendrite_module.parent_module`` is still the base). In the clean graph,
    the selected dendrites are ``layer_array[:-1]`` and the original base
    module is ``layer_array[-1]``. The module is needed to distinguish the
    last branch from a dendrite when more than one branch is present.
    """
    parts = name.lower().split(".")
    if "dendrites_to_top" in parts or "skip_weights" in parts:
        return True

    for index, part in enumerate(parts):
        if part == "layer_array" and index + 1 < len(parts):
            branch = parts[index + 1]
            if branch.isdigit():
                branch_index = int(branch)
                if model is not None:
                    array_name = ".".join(parts[: index + 1])
                    try:
                        layer_array = model.get_submodule(array_name)
                    except AttributeError:
                        layer_array = None
                    if isinstance(layer_array, nn.ModuleList):
                        return branch_index < len(layer_array) - 1
                # The only safe name-only assumption is the one-dendrite
                # deployment used by this cycle: branch zero is residual.
                return branch_index == 0
        if part == "dendrite_module":
            # PAI stores the original neuron under this wrapper as
            # ``dendrite_module.parent_module``. That subtree remains base.
            return "parent_module" not in parts[index + 1:]

    lowered = name.lower()
    return any(marker in lowered for marker in DENDRITE_PARAMETER_MARKERS)


def current_pai_mode(default: str = "?") -> str:
    """PerforatedAI's current phase: ``n`` = neuron training, ``p`` = dendrite.

    Read defensively: the tracker's internals are not a public API, and the
    phase trail is diagnostic, so an unreadable mode must not end a run.
    """
    try:
        return str(GPA.pai_tracker.member_vars["mode"])
    except Exception:  # noqa: BLE001 - diagnostics must never break training
        return default


def describe_learning_phase(model: nn.Module) -> dict:
    """Which parameters are frozen right now, split into base vs dendritic.

    Step 3c requires the base weights to be frozen while dendrites train. PAI
    does that freezing itself; this reports what actually happened so the claim
    is checkable in the run record instead of assumed.
    """
    groups = {
        "base": {"trainable": 0, "frozen": 0},
        "dendrite": {"trainable": 0, "frozen": 0},
    }
    for name, parameter in model.named_parameters():
        group = "dendrite" if _is_dendrite_parameter(name, model) else "base"
        key = "trainable" if parameter.requires_grad else "frozen"
        groups[group][key] += parameter.numel()
    return {
        "mode": current_pai_mode(),
        "total_params": UPA.count_params(model),
        **groups,
    }


def enforce_base_weight_freeze(model: nn.Module) -> int:
    """Freeze every non-dendritic parameter; return how many were still live.

    Belt and braces over PAI's own freezing: if a future release leaves a base
    tensor trainable during the dendrite phase, the dendrites would learn
    against a moving target and the correlation scores that select them would
    be measuring the wrong thing.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if not _is_dendrite_parameter(name, model) and parameter.requires_grad:
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def restore_base_weight_training(model: nn.Module) -> int:
    """Re-enable the base weights; the counterpart to the dendrite-phase freeze.

    Enforcing the freeze without this would leave the base permanently frozen
    after the first dendrite phase, and neuron mode would train nothing.
    """
    restored = 0
    for name, parameter in model.named_parameters():
        if not _is_dendrite_parameter(name, model) and not parameter.requires_grad:
            parameter.requires_grad = True
            restored += parameter.numel()
    return restored


def apply_phase_freezing(model: nn.Module) -> str:
    """Set requires_grad to match PerforatedAI's current phase.

    Neuron mode trains the base; dendrite mode holds it still while candidates
    are scored. PAI sets this itself -- re-asserting it each epoch is what makes
    step 3c's "freeze the relevant base weights" a property of this pipeline
    rather than an assumption about the library. An unreadable mode changes
    nothing, so a PAI internals change degrades to the library's own behaviour.
    """
    mode = current_pai_mode()
    if mode == "p":
        enforce_base_weight_freeze(model)
    elif mode == "n":
        restore_base_weight_training(model)
    return mode


def freeze_selected_dendrites(model: nn.Module) -> int:
    """Freeze the dendrites PAI selected, leaving the base student active.

    This is the boundary between steps 3c and 3d: the dendritic residual nodes
    are settled, and the KD resume that follows adapts only the base student
    parameters to them.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if _is_dendrite_parameter(name, model) and parameter.requires_grad:
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def resume_kd_finetune(
    clean_model: nn.Module,
    datasets,
    train_cfg: dict,
    device: torch.device,
    teacher: FrozenTeacher,
    input_shape: tuple[int, int],
    save_name: str,
    baseline_val_acc: float = 0.0,
    output_dir: str | Path | None = None,
    data_cfg: dict | None = None,
    cycle_recipe_fingerprint: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Framework step 3d: resume KD fine-tuning of the active student parameters.

    The dendrites were selected and frozen against a base network that has not
    seen them. Re-running the task + KD objective over the clean deployment
    graph lets the base weights settle around the capacity the dendrites added,
    and it runs on the exact graph that gets exported -- not on the PAI training
    wrapper, whose extra scaffolding is gone by deployment.
    """
    epochs = int(train_cfg.get("resume_epochs", 0))
    if epochs < 1:
        return {"status": "skipped", "reason": "resume_epochs is not set"}

    frozen_dendrites = freeze_selected_dendrites(clean_model)
    active = [
        parameter for parameter in clean_model.parameters() if parameter.requires_grad
    ]
    if not active:
        return {
            "status": "skipped",
            "reason": "no active parameters remain after dendrite selection",
        }

    clean_model.to(device)
    kd = DistillationCriterion(
        teacher,
        KDWeights.from_config(train_cfg.get("distillation")),
        train_cfg["label_smoothing"],
        device,
        student_feature_dim=(
            _clean_feature_dim(clean_model, input_shape)
            if supports_pooled_features(clean_model, input_shape)
            else None
        ),
    )
    train_generator = torch.Generator().manual_seed(int(train_cfg["seed"]))
    val_generator = torch.Generator().manual_seed(int(train_cfg["seed"]) + 1)
    train_loader = build_data_loader(
        datasets[TRAIN], train_cfg, shuffle=True, generator=train_generator
    )
    val_loader = build_data_loader(
        datasets[VAL], train_cfg, shuffle=False, generator=val_generator
    )
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)
    recorder = (
        MetricsRecorder(
            layout.metrics_path("sparsity", "resume_kd", candidate=Path(save_name).name),
            stage="sparsity",
            phase="resume_kd",
        )
        if layout is not None else None
    )
    latest_path = (
        layout.checkpoint_path(
            "sparsity", "resume_kd", "latest", candidate=Path(save_name).name
        )
        if layout is not None else None
    )
    resume_state = (
        torch.load(latest_path, map_location=device, weights_only=False)
        if latest_path is not None and latest_path.exists() else None
    )

    logger.info(
        "Step 3d: resuming KD fine-tuning for %d epochs on %d active parameters "
        "(%d dendrite parameters frozen)",
        epochs,
        sum(parameter.numel() for parameter in active),
        frozen_dendrites,
    )
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clean_model.state_dict().items()
    }
    best_state: dict[str, torch.Tensor] = {}

    def keep_best(model, _val_acc):
        best_state.clear()
        best_state.update(
            {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        )

    result = run_finetune(
        clean_model,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        parameters=active,
        on_best=keep_best,
        recorder=recorder,
        resume_state=resume_state,
        latest_path=latest_path,
        stage="sparsity",
        phase="resume_kd",
        run_id=run_id,
        recipe={
            "train": train_cfg,
            "data": data_cfg,
            "teacher_checkpoint": teacher.checkpoint_path,
            "teacher_sha256": getattr(teacher, "checkpoint_sha256", None),
            "cycle_recipe_fingerprint": cycle_recipe_fingerprint,
            "save_name": Path(save_name).name,
        },
        epochs=epochs,
        label="resume-kd",
    )
    if not best_state and resume_state is not None:
        best_state.update(resume_state.get("best_model_state_dict") or {})
    improved = result.best_val_acc > baseline_val_acc
    if best_state and improved:
        clean_model.load_state_dict(best_state)
        selected_val_acc = result.best_val_acc
    else:
        clean_model.load_state_dict(initial_state)
        selected_val_acc = baseline_val_acc

    path = (
        layout.checkpoint_path(
            "sparsity", "resume_kd", "best", candidate=Path(save_name).name
        )
        if layout is not None
        else Path(save_name) / "resumed_clean_pai.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "model_state_dict": clean_model.state_dict(),
            "stage": "dendrite_kd_resume",
            "run_id": run_id,
            "val_acc": selected_val_acc,
            "resume_best_val_acc": result.best_val_acc,
            "input_shape": list(input_shape),
            "num_classes": teacher.num_classes,
            "num_keywords": teacher.num_keywords,
            "teacher_checkpoint": teacher.checkpoint_path,
            "distillation": kd.describe(),
        },
    )

    if not improved:
        logger.info(
            "Step 3d did not improve %.4f baseline (best %.4f); keeping the "
            "pre-resume clean graph in %s",
            baseline_val_acc,
            result.best_val_acc,
            path,
        )
        return {
            "status": "no_improvement",
            "epochs": epochs,
            "best_val_acc": baseline_val_acc,
            "resume_best_val_acc": result.best_val_acc,
            "frozen_dendrite_params": frozen_dendrites,
            "active_params": sum(parameter.numel() for parameter in active),
            "distillation": kd.describe(),
            "checkpoint": str(path),
            "history": result.history,
            "run_id": run_id,
        }

    logger.info(
        "Step 3d complete: val_acc %.4f -> %s", result.best_val_acc, path,
    )
    return {
        "status": "complete",
        "epochs": epochs,
        "best_val_acc": result.best_val_acc,
        "frozen_dendrite_params": frozen_dendrites,
        "active_params": sum(parameter.numel() for parameter in active),
        "distillation": kd.describe(),
        "checkpoint": str(path),
        "history": result.history,
        "run_id": run_id,
    }


def _clean_feature_dim(model: nn.Module, input_shape: tuple[int, int]) -> int:
    device = next(model.parameters()).device
    with torch.no_grad():
        feature_model = cast(FeatureModel, model)
        return feature_model.forward_features(
            torch.zeros(1, 1, *input_shape, device=device)
        ).shape[1]


def configure_perforatedai(config: dict, device: torch.device) -> None:
    """Apply the paper's size-focused, correlation-based PAI settings."""
    testing = config["testing_dendrite_capacity"]
    conversion = config["conversion"]
    if conversion != "blocks_and_linear":
        raise ValueError("Cycle 1 supports only conversion=blocks_and_linear")

    # PerforatedAI defaults to CUDA-or-CPU and does not auto-detect Apple MPS.
    GPA.pc.set_device(device)
    GPA.pc.set_use_cuda(device.type == "cuda")
    GPA.pc.set_testing_dendrite_capacity(testing)
    # PAI's capacity check expects to exercise three dendrites.  The real run
    # returns to the configured one-dendrite deployment budget.
    GPA.pc.set_max_dendrites(3 if testing else config["max_dendrites"])
    GPA.pc.set_n_epochs_to_switch(config["n_epochs_to_switch"])
    GPA.pc.set_improvement_threshold(config["improvement_threshold"])
    GPA.pc.set_candidate_weight_initialization_multiplier(
        config["candidate_weight_initialization_multiplier"]
    )
    GPA.pc.set_initial_correlation_batches(config["initial_correlation_batches"])
    GPA.pc.set_max_dendrite_tries(config["max_dendrite_tries"])
    GPA.pc.set_pai_forward_function(getattr(torch, config["forward_function"]))
    # Keep each block's convolution, normalization, and nonlinearity together
    # in its dendritic copy. The stem is tracked but deliberately not copied.
    GPA.pc.set_modules_to_perforate([DSConvBlock, nn.Linear])
    GPA.pc.set_modules_to_track([nn.Conv2d, nn.BatchNorm2d])
    GPA.pc.set_perforated_backpropagation(True)
    GPA.pc.set_dendrite_update_mode(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_configuration_confirmed(True)
    GPA.pc.set_verbose(False)
    GPA.pc.set_silent(False)


def _make_optimizer_and_scheduler(model, train_cfg: dict, loader_length: int, *, kd=None):
    # PerforatedAI must own an optimizer containing only parameters from its
    # wrapped/tracked model. A KD feature adapter is a separate, training-only
    # module, so putting it in this optimizer makes PAI's parameter filter enter
    # its interactive debugger because the adapter has no PAI parameter_type.
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_lr_scheduler(
        optimizer,
        int(train_cfg.get("dendritic_schedule_epochs", train_cfg["epochs"]))
        * loader_length,
        train_cfg["warmup_fraction"],
    )
    GPA.pai_tracker.set_optimizer_instance(optimizer)

    adapter_optimizer = None
    adapter_scheduler = None
    if kd is not None:
        adapter_parameters = list(kd.extra_parameters())
        if adapter_parameters:
            adapter_optimizer = torch.optim.AdamW(
                adapter_parameters,
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )
            adapter_scheduler = build_lr_scheduler(
                adapter_optimizer,
                int(train_cfg.get("dendritic_schedule_epochs", train_cfg["epochs"]))
                * loader_length,
                train_cfg["warmup_fraction"],
            )

    return optimizer, scheduler, adapter_optimizer, adapter_scheduler


def run_cycle(
    checkpoint_path: str,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    save_name: str,
    *,
    teacher_checkpoint: str | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool | None = None,
    resume_from: str | Path | None = None,
    run_id: str | None = None,
) -> DendriticCycleResult:
    """Run steps 3c-3e for one candidate: perforate, resume with KD, measure.

    ``teacher_checkpoint`` is the same fixed teacher every other stage uses. If
    it is omitted the cycle falls back to plain task loss, which is the older
    behaviour and still useful for isolating the dendrites' own contribution.
    """
    train_cfg = with_seed(train_cfg, seed)
    layout = None
    if output_dir is not None:
        layout = ArtifactLayout(output_dir)
        run_id = resolve_manifest_run_id(layout, run_id)
        requested_save_name = Path(save_name).expanduser()
        if requested_save_name.is_absolute():
            resolved_save_name = layout._descendant(requested_save_name)
            expected_parent = layout.root / "pai" / "candidates"
            if resolved_save_name.parent != expected_parent:
                raise ValueError(
                    f"PAI candidate path must be directly below {expected_parent}: "
                    f"{save_name!s}"
                )
            candidate_name = resolved_save_name.name
        else:
            if (
                requested_save_name.name != str(requested_save_name)
                or ".." in requested_save_name.parts
                or requested_save_name.name in {"", ".", ".."}
            ):
                raise ValueError(
                    f"PAI candidate name must be a single logical name: {save_name!s}"
                )
            candidate_name = requested_save_name.name
        save_name = str(layout.pai_candidate_path(candidate_name))
    started_at = monotonic()
    set_seed(train_cfg["seed"])
    pruning_kind = train_cfg.get("pruning", {}).get("kind", "structured")
    if pruning_kind != "structured":
        raise ValueError(
            "The dendritic width sweep supports structured pruning only; "
            f"use kws.optimize.prune for {pruning_kind!r}"
        )
    device = get_device()
    base, checkpoint, model_cfg = build_cycle_base(
        checkpoint_path,
        train_cfg["pruning"]["keep_ratio"],
        target_model_cfg=model_cfg,
        expected_run_id=(
            run_id
            if run_id is not None
            and (
                layout is None
                or Path(checkpoint_path).expanduser().resolve().is_relative_to(layout.root)
            )
            else None
        ),
    )
    # ``build_cycle_base`` deliberately constructs on CPU because checkpoints
    # are device-neutral.  The pre-dendrite KD phase moves its batches to the
    # selected accelerator, so the model must follow before that first forward.
    base = base.to(device)
    fingerprint = cycle_fingerprint(
        checkpoint_path, teacher_checkpoint, data_cfg, model_cfg, train_cfg
    )
    base_params = sum(parameter.numel() for parameter in base.parameters())
    input_shape = tuple(checkpoint["input_shape"])
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)
    run_dir = (
        layout.pai_candidate_path(Path(save_name).name)
        if layout is not None
        else Path(save_name).expanduser().resolve()
    )
    save_name = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = (
        layout.checkpoint_path(
            "sparsity", "pai", "latest", candidate=run_dir.name
        )
        if layout is not None
        else run_dir / "kws_latest.pt"
    )
    resume_sidecar = _select_pai_resume_sidecar(
        run_dir,
        sidecar_path,
        resume=resume,
        resume_from=resume_from,
    )
    pai_state = (
        _load_pai_sidecar(
            resume_sidecar,
            run_dir,
            fingerprint,
            device,
            expected_run_id=run_id,
        )
        if resume_sidecar is not None
        else None
    )
    if run_id is None:
        run_id = (pai_state or {}).get("run_id") or uuid.uuid4().hex
    pai_recorder = (
        MetricsRecorder(
            layout.metrics_path(
                "sparsity", "pai", candidate=run_dir.name
            ),
            stage="sparsity",
            phase="pai",
        )
        if layout is not None else None
    )
    resume_candidate = pai_state is not None
    logger.info(
        "Dendrite-cycle base: %d params, estimated one-dendrite deployment: %d "
        "params, block_channels=%s",
        base_params,
        estimate_one_dendrite_params(base),
        model_cfg["block_channels"],
    )

    datasets, label_map = build_datasets(
        data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"]
    )

    if teacher_checkpoint and run_id is not None and (
        layout is None
        or Path(teacher_checkpoint).expanduser().resolve().is_relative_to(layout.root)
    ):
        teacher_state = torch.load(
            teacher_checkpoint, map_location="cpu", weights_only=False
        )
        validate_checkpoint_run_id(teacher_state, run_id, source=teacher_checkpoint)
    teacher = FrozenTeacher(teacher_checkpoint, device) if teacher_checkpoint else None

    # Framework steps 3a-3b are intentionally a separate phase from PAI. The
    # student first has to adapt to the structurally smaller graph with task +
    # KD loss; otherwise the first PAI neuron phase is doing double duty and a
    # run cannot prove that pruning itself was fine-tuned before perforation.
    prune_finetune = {"status": "skipped", "reason": "no teacher supplied"}
    if teacher is not None:
        pre_checkpoint = (
            ArtifactLayout(output_dir).checkpoint_path(
                "sparsity", "prune_kd", "best", candidate=run_dir.name
            )
            if output_dir is not None
            else run_dir / "pruned_kd.pt"
        )
        pre_latest = (
            layout.checkpoint_path(
                "sparsity", "prune_kd", "latest", candidate=run_dir.name
            )
            if layout is not None
            else pre_checkpoint.with_name("pruned_kd_latest.pt")
        )
        pre_kd = DistillationCriterion(
            teacher,
            KDWeights.from_config(train_cfg.get("distillation")),
            train_cfg["label_smoothing"],
            device,
            student_feature_dim=base.fc.in_features,
        )
        pre_val_acc = train_model(
            base,
            datasets,
            label_map,
            model_cfg,
            train_cfg,
            pre_checkpoint,
            device,
            checkpoint["num_keywords"],
            kd=pre_kd,
            extra_checkpoint_fields={
                "stage": "pruned_kd_finetune",
                "sparsity": train_cfg.get("pruning"),
                "distillation": pre_kd.describe(),
            },
            output_dir=output_dir,
            stage="sparsity",
            phase="prune_kd",
                artifact_candidate=run_dir.name if output_dir is not None else None,
                resume_from=pre_latest if pre_latest.exists() else None,
                run_id=run_id,
                recipe={
                "source_checkpoint": checkpoint_path,
                "teacher_checkpoint": teacher_checkpoint,
                "train": train_cfg,
                "pruning": train_cfg.get("pruning"),
            },
        )
        best_pruned = torch.load(pre_checkpoint, map_location=device, weights_only=False)
        base.load_state_dict(best_pruned["model_state_dict"])
        prune_finetune = {
            "status": "complete",
            "best_val_acc": pre_val_acc.best_val_acc,
            "checkpoint": str(pre_checkpoint),
            "distillation": pre_kd.describe(),
        }
        logger.info(
            "Steps 3a-3b complete: pruned student KD val_acc=%.4f -> %s",
            pre_val_acc.best_val_acc,
            pre_checkpoint,
        )

    train_generator = torch.Generator().manual_seed(int(train_cfg["seed"]))
    val_generator = torch.Generator().manual_seed(int(train_cfg["seed"]) + 1)
    train_loader = build_data_loader(
        datasets[TRAIN], train_cfg, shuffle=True, generator=train_generator
    )
    val_loader = build_data_loader(
        datasets[VAL], train_cfg, shuffle=False, generator=val_generator
    )

    pai_cfg = train_cfg["perforatedai"]
    configure_perforatedai(pai_cfg, device)
    # PAI treats save_name as a filename prefix and explicitly rejects path
    # separators.  The run layout owns the directory; PAI receives only the
    # candidate leaf while its process-wide working directory is scoped to the
    # candidate parent and restored even on failure.
    pai_run_name = run_dir.name
    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir.parent)
        model = UPA.perforate_model(
            base,
            doing_pai=True,
            save_name=pai_run_name,
            making_graphs=True,
            maximizing_score=True,
        ).to(device)
        if resume_candidate:
            load_system = getattr(UPA, "load_system", None)
            if not callable(load_system):
                raise RuntimeError(
                    "installed PerforatedAI has no supported load_system API"
                )
            restored = load_system(
                model,
                str(run_dir.parent),
                pai_run_name,
                load_from_restart=True,
            )
            if restored is not None:
                model = restored.to(device)
            logger.info("Loaded paired PAI restart state for %s", run_dir)
    finally:
        os.chdir(previous_cwd)
    logger.info("PAI-wrapped initial parameter count: %d", UPA.count_params(model))

    kd = None
    if teacher is not None:
        # PAI rewrites the module tree, so whether the pooled features survive
        # is a property of the wrapped object and has to be probed, not assumed.
        kd = DistillationCriterion(
            teacher,
            KDWeights.from_config(train_cfg.get("distillation")),
            train_cfg["label_smoothing"],
            device,
            student_feature_dim=(
                _clean_feature_dim(model, input_shape)
                if supports_pooled_features(model, input_shape)
                else None
            ),
        )
        logger.info(
            "Dendrite cycle distilling from %s (feature loss %s)",
            teacher_checkpoint,
            "on" if kd.uses_features else "off",
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    (
        optimizer,
        scheduler,
        adapter_optimizer,
        adapter_scheduler,
    ) = _make_optimizer_and_scheduler(
        model, train_cfg, len(train_loader), kd=kd
    )

    if resume_candidate:
        _restore_pai_training_state(
            pai_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            kd=kd,
            adapter_optimizer=adapter_optimizer,
            adapter_scheduler=adapter_scheduler,
            device=device,
        )
        phase_trail = list(pai_state.get("phase_trail") or [])
        if pai_state.get("rng_state") is not None:
            restore_rng_state(pai_state["rng_state"])
        for loader, loader_state in zip(
            (train_loader, val_loader), pai_state.get("loader_generator_states", [])
        ):
            if getattr(loader, "generator", None) is not None and loader_state is not None:
                loader.generator.set_state(loader_state)
        if pai_recorder is not None:
            pai_recorder.reconcile(
                int(pai_state.get("history_length", pai_recorder.count)),
                pai_state.get("last_metric_digest"),
            )
        logger.info(
            "Resuming PAI candidate %s after epoch %d",
            run_dir,
            int(pai_state.get("completed_epoch", 0)),
        )

    freeze_base = bool(pai_cfg.get("enforce_base_weight_freeze", True))
    phase_trail: list[dict] = (
        phase_trail if pai_state is not None else [{"epoch": 0, **describe_learning_phase(model)}]
    )
    last_mode = phase_trail[-1]["mode"]

    epoch = int(pai_state.get("completed_epoch", 0)) - 1 if pai_state else -1
    global_step = int(pai_state.get("global_step", 0)) if pai_state else 0
    while True:
        epoch += 1
        epoch_started = monotonic()
        set_train_mode_preserving_frozen_batchnorm(model)
        if freeze_base:
            # Step 3c: the base weights stay fixed while candidate dendrites
            # are scored, so the correlation that selects them is measured
            # against a network that is not simultaneously moving -- and they
            # are handed back the moment neuron training resumes.
            apply_phase_freezing(model)
            # PAI may change requires_grad during the call above. Re-apply the
            # BatchNorm rule after that phase transition as well.
            keep_frozen_batchnorm_eval(model)
        if kd is not None:
            kd.train()
        running_losses = {"total": 0.0}
        train_correct = 0
        train_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            if adapter_optimizer is not None:
                adapter_optimizer.zero_grad()
            if kd is None:
                logits = model(features)
                losses = {"total": criterion(logits, labels)}
            else:
                student_features = (
                    model.forward_features(features) if kd.uses_features else None
                )
                logits = (
                    model.classify_features(student_features)
                    if student_features is not None
                    else model(features)
                )
                losses = kd(features, logits, labels, student_features)
            loss = losses["total"]
            loss.backward()
            optimizer.step()
            if adapter_optimizer is not None:
                adapter_optimizer.step()
            scheduler.step()
            if adapter_scheduler is not None:
                adapter_scheduler.step()
            for name, value in losses.items():
                running_losses[name] = (
                    running_losses.get(name, 0.0) + value.item() * labels.size(0)
                )
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)

        train_losses = {
            name: value / train_count for name, value in running_losses.items()
        }
        train_loss = train_losses["total"]
        train_acc = train_correct / train_count
        global_step += len(train_loader)
        if kd is not None:
            kd.eval()
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
        GPA.pai_tracker.add_extra_score(train_acc, "Train")
        model, restructured, training_complete = (
            GPA.pai_tracker.add_validation_score(val_acc, model)
        )
        model = model.to(device)
        mode = current_pai_mode()
        phase_changed = mode != last_mode or restructured
        if phase_changed:
            phase: dict[str, Any] = {
                "epoch": epoch + 1,
                "restructured": restructured,
                **describe_learning_phase(model),
            }
            phase_trail.append(phase)
            last_mode = mode

        # The metric describes the optimizer that executed this epoch. If PAI
        # restructured the graph at validation, the replacement optimizer is
        # created only after this record and before the paired checkpoint.
        metric_digest = None
        if pai_recorder is not None:
            metric_digest = pai_recorder.append(
                _pai_epoch_record(
                    epoch=epoch + 1,
                    global_step=global_step,
                    elapsed_seconds=monotonic() - epoch_started,
                    train_losses=train_losses,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    optimizer=optimizer,
                    adapter_optimizer=adapter_optimizer,
                    mode=mode,
                    restructured=restructured,
                    model=model,
                    seed=int(train_cfg["seed"]),
                )
            )

        if restructured:
            logger.info("PAI restructured the network; resetting optimizer phase")
            (
                optimizer,
                scheduler,
                adapter_optimizer,
                adapter_scheduler,
            ) = _make_optimizer_and_scheduler(
                model, train_cfg, len(train_loader), kd=kd
            )

        # Keep the vendor restart and the KWS sidecar at the same validation
        # boundary. This is deliberately after any graph restructure and
        # optimizer reset, so the first durable pair is internally loadable.
        _save_pai_restart_pair(
            model=model,
            run_dir=run_dir,
            pai_run_name=pai_run_name,
            sidecar_path=sidecar_path,
            optimizer=optimizer,
            scheduler=scheduler,
            kd=kd,
            adapter_optimizer=adapter_optimizer,
            adapter_scheduler=adapter_scheduler,
            phase_trail=phase_trail,
            completed_epoch=epoch + 1,
            global_step=global_step,
            history_length=(
                pai_recorder.count if pai_recorder is not None else epoch + 1
            ),
            last_metric_digest=metric_digest,
            teacher_checkpoint=teacher_checkpoint,
            recipe_fingerprint=fingerprint,
            loaders=(train_loader, val_loader),
            pai_run_dir_ref=(layout.relative(run_dir) if layout is not None else None),
            native_pai_latest_ref=(
                layout.relative(run_dir / "latest.pt") if layout is not None else None
            ),
            run_id=run_id,
        )

        logger.info(
            "epoch %d mode=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f "
            "val_acc=%.4f params=%d",
            epoch + 1,
            current_pai_mode(),
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            UPA.count_params(model),
        )

        if phase_changed:
            logger.info(
                "PAI phase -> %s: base %d trainable / %d frozen, dendrite %d "
                "trainable / %d frozen",
                phase_trail[-1]["mode"],
                phase_trail[-1]["base"]["trainable"],
                phase_trail[-1]["base"]["frozen"],
                phase_trail[-1]["dendrite"]["trainable"],
                phase_trail[-1]["dendrite"]["frozen"],
            )

        if training_complete:
            clean_model = export_final_pai_model(model, save_name)
            logger.info("PAI cycle complete; results saved under %s", save_name)
            break

    best_val_acc, deployed_params = read_pai_architecture_results(save_name)

    # Step 3d: resume KD fine-tuning of the active student parameters.
    resume = {"status": "skipped", "reason": "no teacher supplied"}
    if teacher is not None:
        resume = resume_kd_finetune(
            clean_model,
            datasets,
            train_cfg,
            device,
            teacher,
            input_shape,
            save_name,
            baseline_val_acc=best_val_acc,
            output_dir=output_dir,
            data_cfg=data_cfg,
            cycle_recipe_fingerprint=fingerprint,
            run_id=run_id,
        )
        if resume.get("status") == "complete":
            logger.info(
                "KD resume improved validation accuracy %.4f -> %.4f",
                best_val_acc,
                resume["best_val_acc"],
            )
            best_val_acc = resume["best_val_acc"]

    # Step 3e: record latency, memory, and compute for the exported graph.
    profile_device = torch.device(
        train_cfg.get("deployment", {}).get("profile_device", "cpu")
    )
    cost = profile_model(
        clean_model.to(profile_device),
        input_shape,
        device=profile_device,
        bits_per_weight=int(train_cfg.get("deployment", {}).get("bits_per_weight", 32)),
        latency_iterations=int(train_cfg.get("deployment", {}).get("latency_iterations", 50)),
    ).as_dict()
    logger.info(
        "Step 3e: %d deployed params, %d MACs, %d weight bytes, %.3f ms p50",
        deployed_params,
        cost["macs"],
        cost["weight_bytes"],
        cost["latency_ms_p50"],
    )

    # PAI's CSV excludes clean-graph parameters such as restored skip
    # coefficients. The profiled clean graph is the deployment authority.
    pai_deployed_params = deployed_params
    deployed_params = int(cost["params"])
    result = DendriticCycleResult(
        save_name=save_name,
        block_channels=list(model_cfg["block_channels"]),
        base_params=base_params,
        deployed_params=deployed_params,
        best_val_acc=best_val_acc,
        epochs=epoch + 1,
        elapsed_seconds=monotonic() - started_at,
        pai_deployed_params=pai_deployed_params,
        cost=cost,
        distillation=kd.describe() if kd is not None else None,
        resume=resume,
        prune_finetune=prune_finetune,
        phase_trail=phase_trail,
        run_id=run_id,
    )

    # PAI owns its architecture checkpoints. Preserve the source metadata next
    # to them so the exporter, evaluator, and pruning search can reconstruct it.
    metadata_path = Path(save_name) / "cycle_metadata.yaml"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    clean_artifact_path = Path(save_name) / "final_clean_pai.pt"

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute() and layout is not None:
            try:
                return layout.relative(value)
            except ValueError:
                return value
        return value

    metadata = {
        "framework_cycle_version": FRAMEWORK_CYCLE_VERSION,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "status": "complete",
        "result": portable(asdict(result)),
        "source_checkpoint": portable(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_val_acc": checkpoint.get("val_acc"),
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha256": (
            file_sha256(teacher_checkpoint) if teacher_checkpoint else None
        ),
        "clean_artifact_sha256": file_sha256(clean_artifact_path),
        "base_model_cfg": model_cfg,
        "base_params": base_params,
        "selection_split": "validation",
        "test_split_used": False,
        "objective": train_cfg["objective"],
        "perforatedai": pai_cfg,
    }
    if output_dir is not None:
        ArtifactLayout(output_dir).atomic_yaml(metadata_path, metadata)
    else:
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        temporary.replace(metadata_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config", default="configs/data/speech_commands_v2.yaml"
    )
    parser.add_argument(
        "--train-config", default="configs/train/dendritic_cycle1.yaml"
    )
    parser.add_argument(
        "--model-config", default="configs/model/ds_cnn_xxs.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/ds_cnn_xs_distilled_warm_12class.pt",
    )
    parser.add_argument("--save-name", default="dendritic_xxs_cycle1")
    parser.add_argument("--output-dir", default=None)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume this candidate from its canonical paired PAI/KWS latest state",
    )
    resume_group.add_argument(
        "--resume-from",
        default=None,
        help="Resume from an explicit KWS PAI training-state sidecar",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="models/checkpoints/ds_cnn_l_12class.pt",
        help="Fixed teacher for the cycle's KD loss and the step-3d resume",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training-config seed (must be non-negative)",
    )
    args = parser.parse_args()

    inputs = [
        (args.data_config, "data_config"),
        (args.model_config, "model_config"),
        (args.train_config, "train_config"),
        (args.checkpoint, "source_checkpoint"),
    ]
    if args.teacher_checkpoint:
        inputs.append((args.teacher_checkpoint, "teacher_checkpoint"))
    if args.resume_from:
        inputs.append((args.resume_from, "resume_checkpoint"))
    with run_session(
        args.output_dir,
        command="kws.optimize.dendritic",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=inputs,
    ):
        run_cycle(
            args.checkpoint,
            load_yaml(args.data_config),
            load_yaml(args.model_config),
            load_yaml(args.train_config),
            args.save_name,
            teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed,
            output_dir=args.output_dir,
            resume=args.resume,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
