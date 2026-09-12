"""Structured (channel) and N:M pruning for DS-CNN.

Step 3a of the framework asks for "structured or N:M pruning", and the two
answer different deployment stories:

- **Structured channel pruning** removes output channels from each block's
  pointwise conv (ranked by L1 norm) and threads the surviving input-channel
  selection through the next block's depthwise conv and BN, or the final FC
  layer. It produces a genuinely smaller dense model, which is the only form of
  sparsity a TFLite Micro / ESP-DL runtime turns into real savings.
- **N:M pruning** keeps N of every M contiguous weights along the input
  dimension. It leaves the tensor shapes alone, so it costs nothing until the
  target has an N:M sparse kernel -- but the mask is exactly what a sparse
  accelerator (or an MRAM macro that skips zero columns) consumes, so the
  framework can measure the accuracy cost ahead of that hardware.

Both are expressed as a :class:`SparsitySpec`; the standalone pruning entry
point supports either, while the dendritic width sweep deliberately uses
structured pruning because its search variable is channel width.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.models.ds_cnn import DSCNN
from kws.optimize.kd import DistillationCriterion, FrozenTeacher, KDWeights, file_sha256
from kws.train import (
    resolve_manifest_run_id,
    train_model,
    validate_checkpoint_run_id,
)
from kws.utils.device import get_device
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import (
    recipe_fingerprint,
    require_training_state,
    validate_resume_recipe,
)
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)


def _l1_topk_indices(weight: torch.Tensor, k: int) -> torch.Tensor:
    scores = weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(scores, k).indices.sort().values


def _copy_bn_subset(old_bn: nn.BatchNorm2d, new_bn: nn.BatchNorm2d, indices: torch.Tensor) -> None:
    old_mean, new_mean = old_bn.running_mean, new_bn.running_mean
    old_var, new_var = old_bn.running_var, new_bn.running_var
    if old_mean is None or new_mean is None or old_var is None or new_var is None:
        raise ValueError("channel pruning requires BatchNorm running statistics")
    with torch.no_grad():
        new_bn.weight.copy_(old_bn.weight[indices])
        new_bn.bias.copy_(old_bn.bias[indices])
        new_mean.copy_(old_mean[indices])
        new_var.copy_(old_var[indices])


def prune_ds_cnn(model: DSCNN, keep_ratio: float) -> DSCNN:
    old_block_channels = [block.pointwise.out_channels for block in model.blocks]
    new_block_channels = [max(1, int(round(c * keep_ratio))) for c in old_block_channels]
    stem_conv = cast(nn.Conv2d, model.stem[0])

    new_model = DSCNN(
        input_shape=model.input_shape,
        num_classes=model.fc.out_features,
        initial_channels=stem_conv.out_channels,
        initial_kernel=stem_conv.kernel_size[0],
        initial_stride=stem_conv.stride[0],
        block_channels=new_block_channels,
        dropout=model.dropout.p,
    )
    new_model.stem.load_state_dict(model.stem.state_dict())

    keep_in: torch.Tensor | None = None  # surviving input channels; None = all
    for old_block, new_block, n_keep in zip(model.blocks, new_model.blocks, new_block_channels):
        if keep_in is None:
            new_block.depthwise.weight.data = old_block.depthwise.weight.data.clone()
            new_block.bn1.load_state_dict(old_block.bn1.state_dict())
        else:
            new_block.depthwise.weight.data = old_block.depthwise.weight.data[keep_in].clone()
            _copy_bn_subset(old_block.bn1, new_block.bn1, keep_in)

        pw_weight = old_block.pointwise.weight.data
        if keep_in is not None:
            pw_weight = pw_weight[:, keep_in]
        # Rank output channels using only the input channels that survived
        # the preceding block. Ranking the pre-pruned tensor can select a
        # channel whose apparent importance lives entirely in deleted inputs.
        keep_out = _l1_topk_indices(pw_weight, n_keep)
        new_block.pointwise.weight.data = pw_weight[keep_out].clone()
        _copy_bn_subset(old_block.bn2, new_block.bn2, keep_out)

        keep_in = keep_out

    new_model.fc.weight.data = model.fc.weight.data[:, keep_in].clone()
    new_model.fc.bias.data = model.fc.bias.data.clone()
    return new_model


class SparsityMasks:
    """N:M masks, re-imposed after every optimizer step.

    A mask applied once is undone by the next gradient update, so the training
    loop calls this after each step. Holding the masks by module name (rather
    than by object) keeps them valid across the module swaps that
    ``prepare_qat`` performs later in the pipeline.
    """

    def __init__(self, masks: dict[str, torch.Tensor], n: int, m: int):
        self.masks = masks
        self.n = n
        self.m = m

    def __len__(self) -> int:
        return len(self.masks)

    def apply(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, mask in self.masks.items():
                module = cast(nn.Conv2d | nn.Linear, model.get_submodule(name))
                module.weight.mul_(mask.to(module.weight.device, module.weight.dtype))

    def __call__(self, model: nn.Module) -> None:
        self.apply(model)

    def sparsity(self, model: nn.Module) -> float:
        """Fraction of all prunable weights currently zeroed."""
        zeros = total = 0
        for name in self.masks:
            weight = cast(nn.Conv2d | nn.Linear, model.get_submodule(name)).weight
            zeros += int((weight == 0).sum().item())
            total += weight.numel()
        return zeros / max(total, 1)

    def describe(self) -> dict:
        return {
            "pattern": f"{self.n}:{self.m}",
            "masked_modules": sorted(self.masks),
            "masked_weights": sum(mask.numel() for mask in self.masks.values()),
        }


def nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Keep the ``n`` largest-magnitude weights in each contiguous block of ``m``.

    Blocks run along the flattened input dimension -- for a conv that is
    ``in_channels * kh * kw`` -- which is the dimension an N:M sparse MAC array
    reduces over, so the pattern matches what the hardware can actually skip.
    """
    if not 0 < n < m:
        raise ValueError(f"N:M pattern requires 0 < n < m, got {n}:{m}")
    flat = weight.detach().reshape(weight.shape[0], -1)
    in_features = flat.shape[1]
    if in_features % m:
        raise ValueError(
            f"input dimension {in_features} is not divisible by m={m}; "
            "pad the layer or exclude it from N:M pruning"
        )
    blocks = flat.abs().reshape(-1, m)
    keep = torch.topk(blocks, n, dim=1).indices
    mask = torch.zeros_like(blocks)
    mask.scatter_(1, keep, 1.0)
    return mask.reshape(flat.shape).reshape(weight.shape)


def apply_nm_sparsity(
    model: nn.Module,
    n: int,
    m: int,
    *,
    module_types: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
) -> SparsityMasks:
    """Mask every eligible weight tensor in ``model`` to an N:M pattern.

    Layers whose input dimension is not a multiple of ``m`` are skipped rather
    than padded: a DS-CNN's depthwise convs have an input dimension of
    ``1 * kh * kw`` and carry a small share of the weights, so forcing a pattern
    on them costs accuracy for almost no memory.
    """
    masks: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, module_types) or not hasattr(module, "weight"):
            continue
        try:
            typed_module = cast(nn.Conv2d | nn.Linear, module)
            masks[name] = nm_mask(typed_module.weight, n, m)
        except ValueError:
            skipped.append(name)
    if not masks:
        raise ValueError(
            f"no layer in this model admits a {n}:{m} pattern (skipped: {skipped})"
        )
    if skipped:
        logger.info("N:M %d:%d skipped %d indivisible layers: %s", n, m, len(skipped), skipped)
    sparsity_masks = SparsityMasks(masks, n, m)
    sparsity_masks.apply(model)
    logger.info(
        "Applied %d:%d sparsity to %d layers (%.1f%% of maskable weights zeroed)",
        n, m, len(masks), 100 * sparsity_masks.sparsity(model),
    )
    return sparsity_masks


@dataclass(frozen=True)
class SparsitySpec:
    """One sparsity target in the step-3 sweep."""

    kind: str  # "structured" | "nm" | "dense"
    keep_ratio: float | None = None
    n: int | None = None
    m: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "structured":
            if not self.keep_ratio or not 0 < self.keep_ratio <= 1:
                raise ValueError("structured pruning needs 0 < keep_ratio <= 1")
        elif self.kind == "nm":
            if self.n is None or self.m is None:
                raise ValueError("N:M pruning needs both n and m")
            if not 0 < self.n < self.m:
                raise ValueError(f"N:M pruning needs 0 < n < m, got {self.n}:{self.m}")
        elif self.kind != "dense":
            raise ValueError(f"unknown sparsity kind {self.kind!r}")

    @classmethod
    def from_config(cls, config: dict) -> "SparsitySpec":
        return cls(
            kind=config.get("kind", "structured"),
            keep_ratio=(
                float(config["keep_ratio"]) if config.get("keep_ratio") is not None else None
            ),
            n=int(config["n"]) if config.get("n") is not None else None,
            m=int(config["m"]) if config.get("m") is not None else None,
        )

    @property
    def label(self) -> str:
        if self.kind == "structured":
            return f"structured_keep{self.keep_ratio:.2f}"
        if self.kind == "nm":
            return f"nm{self.n}of{self.m}"
        return "dense"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "keep_ratio": self.keep_ratio,
            "n": self.n,
            "m": self.m,
            "label": self.label,
        }


def apply_sparsity(model: DSCNN, spec: SparsitySpec) -> tuple[DSCNN, SparsityMasks | None]:
    """Return the sparsified model and, for N:M, the masks to re-impose."""
    if spec.kind == "dense":
        return model, None
    if spec.kind == "structured":
        if spec.keep_ratio is None:
            raise ValueError("structured sparsity requires keep_ratio")
        return prune_ds_cnn(model, spec.keep_ratio), None
    if spec.n is None or spec.m is None:
        raise ValueError("N:M sparsity requires both n and m")
    return model, apply_nm_sparsity(model, spec.n, spec.m)


def _resolve_prune_resume_from(
    *,
    resume: bool,
    resume_from: str | Path | None,
    output_dir: str | Path | None,
    out_checkpoint: str | Path,
    spec: SparsitySpec,
) -> Path | None:
    """Resolve pruning's explicit or phase-standard training-state input."""
    if resume_from is not None:
        return Path(resume_from)
    if not resume:
        return None

    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    if layout is not None:
        latest_path = layout.checkpoint_path(
            "sparsity", "prune_kd", "latest", candidate=spec.label
        )
        metrics_path = layout.metrics_path(
            "sparsity", "prune_kd", candidate=spec.label
        )
    else:
        latest_path = Path(out_checkpoint).with_name("latest.pt")
        metrics_path = None

    if latest_path.exists():
        return latest_path
    if (
        metrics_path is not None
        and metrics_path.exists()
        and metrics_path.stat().st_size
    ):
        raise FileNotFoundError(
            f"--resume requested for sparsity/{spec.label}/prune_kd, but the "
            f"phase checkpoint does not exist at {latest_path} while metrics exist "
            f"at {metrics_path}; refusing to restart at epoch 1 and append duplicate metrics"
        )
    return None


def _load_prune_resume_state(
    path: str | Path, device, *, expected_run_id: str | None = None
) -> dict:
    state = require_training_state(
        torch.load(path, map_location=device, weights_only=False), source=path
    )
    validate_checkpoint_run_id(state, expected_run_id, source=path)
    if state.get("stage") != "sparsity" or state.get("phase") != "prune_kd":
        raise ValueError(
            f"resume checkpoint belongs to {state.get('stage')}/{state.get('phase')}, "
            "not sparsity/prune_kd"
        )
    return state


def _validate_prune_resume_state(
    state: dict, *, source: str | Path, recipe: dict
) -> None:
    validate_resume_recipe(state, recipe_fingerprint(recipe), source=source)


def prune_and_fine_tune(
    checkpoint_path: str,
    data_cfg: dict,
    train_cfg: dict,
    spec: SparsitySpec,
    out_checkpoint: Path,
    *,
    teacher_checkpoint: str | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    resume_from: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Framework steps 3a-3b: sparsify the student, then fine-tune it with KD.

    Fine-tuning after pruning uses task loss *and* the fixed teacher's KD loss.
    The teacher still has the capacity that pruning just removed, so its soft
    targets carry more information about the classes the pruned student is now
    confusing than the one-hot labels do.
    """
    train_cfg = with_seed(train_cfg, seed)
    device = get_device()
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)
    if layout is not None:
        layout.ensure_tree()
        out_checkpoint = layout.checkpoint_path(
            "sparsity", "prune_kd", "best", candidate=spec.label
        )
    resume_path = _resolve_prune_resume_from(
        resume=resume,
        resume_from=resume_from,
        output_dir=output_dir,
        out_checkpoint=out_checkpoint,
        spec=spec,
    )
    resume_state = (
        _load_prune_resume_state(resume_path, device, expected_run_id=run_id)
        if resume_path is not None else None
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if run_id is not None and (
        layout is None
        or Path(checkpoint_path).expanduser().resolve().is_relative_to(layout.root)
    ):
        validate_checkpoint_run_id(checkpoint, run_id, source=checkpoint_path)

    from kws.models.ds_cnn import build_ds_cnn
    model = build_ds_cnn(checkpoint["model_cfg"], tuple(checkpoint["input_shape"]), checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])

    old_params = sum(parameter.numel() for parameter in model.parameters())
    sparsified, masks = apply_sparsity(model, spec)
    sparsified = sparsified.to(device)
    new_params = sum(parameter.numel() for parameter in sparsified.parameters())
    logger.info(
        "Sparsity %s: %d -> %d params%s",
        spec.label,
        old_params,
        new_params,
        f" ({100 * masks.sparsity(sparsified):.1f}% zeroed)" if masks else "",
    )

    set_seed(train_cfg["seed"])
    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])

    kd = None
    if teacher_checkpoint is not None:
        if run_id is not None and (
            layout is None
            or Path(teacher_checkpoint).expanduser().resolve().is_relative_to(layout.root)
        ):
            teacher_state = torch.load(
                teacher_checkpoint, map_location="cpu", weights_only=False
            )
            validate_checkpoint_run_id(teacher_state, run_id, source=teacher_checkpoint)
        teacher = FrozenTeacher(teacher_checkpoint, device)
        kd = DistillationCriterion(
            teacher,
            KDWeights.from_config(train_cfg.get("distillation")),
            train_cfg["label_smoothing"],
            device,
            student_feature_dim=sparsified.fc.in_features,
        )
        logger.info("Fine-tuning with task loss + KD from %s", teacher_checkpoint)

    pruned_model_cfg = dict(checkpoint["model_cfg"])
    pruned_model_cfg["name"] = f'{checkpoint["model_cfg"]["name"]}_{spec.label}'
    pruned_model_cfg["block_channels"] = [
        block.pointwise.out_channels for block in sparsified.blocks
    ]

    recipe = {
        "source_sha256": file_sha256(checkpoint_path),
        "teacher_sha256": (
            file_sha256(teacher_checkpoint) if teacher_checkpoint is not None else None
        ),
        "data": data_cfg,
        "model": pruned_model_cfg,
        "spec": spec.as_dict(),
        "train": train_cfg,
        "distillation": kd.describe() if kd else None,
    }
    if resume_state is not None and resume_path is not None:
        _validate_prune_resume_state(
            resume_state, source=resume_path, recipe=recipe
        )
        saved_masks = (resume_state.get("stage_specific_state") or {}).get(
            "sparsity_masks"
        )
        if masks is not None:
            if (
                not isinstance(saved_masks, dict)
                or set(saved_masks) != set(masks.masks)
            ):
                raise ValueError(
                    "resume checkpoint is missing the required N:M sparsity masks "
                    "or contains masks for a different model"
                )
            for name, expected_mask in masks.masks.items():
                saved_mask = saved_masks[name]
                if (
                    not isinstance(saved_mask, torch.Tensor)
                    or saved_mask.shape != expected_mask.shape
                ):
                    raise ValueError(
                        f"resume checkpoint has an incompatible N:M mask for {name}"
                    )
            masks = SparsityMasks(saved_masks, masks.n, masks.m)
        elif saved_masks is not None:
            raise ValueError(
                "resume checkpoint contains N:M sparsity masks for a non-N:M pruning recipe"
            )

    finetune_result = train_model(
        sparsified,
        datasets,
        label_map,
        pruned_model_cfg,
        train_cfg,
        out_checkpoint,
        device,
        checkpoint["num_keywords"],
        kd=kd,
        post_step=masks,
        extra_checkpoint_fields={
            "sparsity": spec.as_dict(),
            "sparsity_masks": masks.describe() if masks else None,
            "distillation": kd.describe() if kd else None,
        },
        output_dir=output_dir,
        stage="sparsity",
        phase="prune_kd",
        resume_state=resume_state,
        recipe=recipe,
        stage_specific_state={
            "sparsity_masks": masks.masks if masks is not None else None,
        },
        artifact_candidate=spec.label,
        run_id=run_id,
    )
    return {
        "spec": spec.as_dict(),
        "base_params": old_params,
        "pruned_params": new_params,
        "best_val_acc": finetune_result.best_val_acc,
        "final_val_acc": finetune_result.final_val_acc,
        "history": finetune_result.history,
        "checkpoint": str(out_checkpoint),
        "masks": masks.describe() if masks else None,
        "distillation": kd.describe() if kd else None,
        "run_id": finetune_result.run_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/full.yaml")
    parser.add_argument("--checkpoint", required=True, help="Trained DS-CNN checkpoint to prune")
    parser.add_argument("--kind", choices=["structured", "nm", "dense"], default="structured")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=None, help="N of the N:M pattern")
    parser.add_argument("--m", type=int, default=None, help="M of the N:M pattern")
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Fixed teacher for KD fine-tuning; omit to fine-tune on task loss only",
    )
    parser.add_argument("--out-checkpoint", required=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training-config seed (must be non-negative)",
    )
    args = parser.parse_args()
    if args.out_checkpoint is None and args.output_dir is None:
        parser.error("--out-checkpoint is required unless --output-dir is supplied")

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    spec = SparsitySpec(
        kind=args.kind,
        keep_ratio=args.keep_ratio if args.kind == "structured" else None,
        n=args.n,
        m=args.m,
    )
    layout = ArtifactLayout(args.output_dir) if args.output_dir else None
    destination = (
        layout.checkpoint_path("sparsity", "prune_kd", "best", candidate=spec.label)
        if layout is not None else Path(args.out_checkpoint)
    )
    resume_from = _resolve_prune_resume_from(
        resume=args.resume,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        out_checkpoint=destination,
        spec=spec,
    )
    inputs = [
        (args.data_config, "data_config"),
        (args.train_config, "train_config"),
        (args.checkpoint, "source_checkpoint"),
    ]
    if args.teacher_checkpoint:
        inputs.append((args.teacher_checkpoint, "teacher_checkpoint"))
    with run_session(
        args.output_dir,
        command="kws.optimize.prune",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=inputs,
    ):
        result = prune_and_fine_tune(
            args.checkpoint,
            data_cfg,
            train_cfg,
            spec,
            destination,
            teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed,
            output_dir=args.output_dir,
            resume=args.resume,
            resume_from=resume_from,
        )
        if layout is not None:
            layout.atomic_yaml(layout.report_path("prune.yaml"), result)


if __name__ == "__main__":
    main()
