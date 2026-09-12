import argparse
import copy
import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import FeatureModel, build_ds_cnn
from kws.utils.device import get_device
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import (
    MetricsRecorder,
    atomic_torch_save,
    build_training_state,
    move_optimizer_state_to_device,
    recipe_fingerprint,
    require_training_state,
    restore_rng_state,
    write_phase_summary,
)
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)


def validate_checkpoint_run_id(
    checkpoint: Mapping,
    expected_run_id: str | None,
    *,
    source: str | Path,
) -> None:
    """Reject a checkpoint from a different manifest run when one is supplied."""
    if expected_run_id is None:
        return
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint {source} is not a mapping")
    if checkpoint.get("run_id") != expected_run_id:
        raise ValueError(
            f"checkpoint {source} belongs to run_id={checkpoint.get('run_id')!r}, "
            f"expected {expected_run_id!r}"
        )


def resolve_manifest_run_id(
    output_dir: str | Path | ArtifactLayout | None,
    run_id: str | None,
) -> str | None:
    """Bind a checkpoint writer to the manifest for its active output root."""
    if output_dir is None:
        return run_id
    layout = output_dir if isinstance(output_dir, ArtifactLayout) else ArtifactLayout(output_dir)
    manifest_run_id = layout.manifest_run_id
    if run_id is not None and run_id != manifest_run_id:
        raise ValueError(
            f"checkpoint run_id {run_id!r} does not match manifest run_id "
            f"{manifest_run_id!r}"
        )
    return manifest_run_id


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lr_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup_steps = max(int(total_steps * warmup_fraction), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # PAI phases can outlive the nominal epoch budget. Once the planned
        # schedule is exhausted, hold the minimum learning rate instead of
        # letting cosine continue oscillating back to a high rate.
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_loss_acc(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


@dataclass
class FinetuneResult:
    """Outcome of one fine-tuning phase, for the stage report."""

    best_val_acc: float
    final_val_acc: float
    epochs: int
    history: list[dict] = field(default_factory=list)
    global_step: int = 0
    completed_epoch: int = 0
    run_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "best_val_acc": self.best_val_acc,
            "final_val_acc": self.final_val_acc,
            "epochs": self.epochs,
            "history": self.history,
            "global_step": self.global_step,
            "completed_epoch": self.completed_epoch,
            "run_id": self.run_id,
        }

    def __float__(self) -> float:
        return float(self.best_val_acc)


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm statistics fixed when all of its affine params are frozen."""
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        local_parameters = list(module.parameters(recurse=False))
        if not any(parameter.requires_grad for parameter in local_parameters):
            module.eval()


def set_train_mode_preserving_frozen_batchnorm(model: nn.Module) -> None:
    """Train active modules without changing frozen BatchNorm statistics.

    ``requires_grad=False`` freezes affine parameters, but ``model.train()``
    would still update a frozen BatchNorm's running mean and variance. That is
    especially easy to miss during PAI's dendrite phase, where the base graph
    must be fixed while candidate residual nodes learn. Any BatchNorm with no
    active affine parameters is therefore put back in eval mode after the
    caller switches the model to training mode.
    """
    model.train()
    keep_frozen_batchnorm_eval(model)


def run_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    train_cfg: Mapping,
    *,
    kd=None,
    parameters: Iterable[nn.Parameter] | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
    on_best: Callable[[nn.Module, float], None] | None = None,
    on_epoch_end: Callable[[dict], None] | None = None,
    epochs: int | None = None,
    label: str = "finetune",
    stage: str = "train",
    phase: str | None = None,
    recorder: MetricsRecorder | None = None,
    resume_state: dict | None = None,
    latest_path: Path | None = None,
    best_path: Path | None = None,
    run_id: str | None = None,
    recipe: dict | None = None,
    upstream: dict | None = None,
    stage_specific_state: dict | None = None,
    extra_epoch_fields: Callable[[], Mapping] | None = None,
    reset_metrics: bool = False,
) -> FinetuneResult:
    """One fine-tuning phase, shared by every stage of the framework.

    The framework fine-tunes after pruning, after the dendrite phase, after
    weight clustering, and under fake quantization.  Those differ only in three
    respects, which are the three hooks here: whether a distillation criterion
    supplements the task loss (``kd``), which parameters are active
    (``parameters``, so a frozen base or a centroid-only update is expressible),
    and what has to be re-imposed after each optimizer step (``post_step``, for
    N:M masks and codebook projection).
    """
    target_epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    if target_epochs < 0:
        raise ValueError("epochs must be non-negative")
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    active = list(parameters) if parameters is not None else _trainable(model.parameters())
    if kd is not None:
        active = active + list(kd.extra_parameters())
    if not active:
        raise ValueError(f"{label}: no trainable parameters were selected")

    optimizer = torch.optim.AdamW(
        active, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_lr_scheduler(
        optimizer, max(target_epochs * len(train_loader), 1), train_cfg["warmup_fraction"],
    )

    phase = phase or label
    if resume_state is None and recorder is not None and recorder.count:
        if not reset_metrics:
            raise ValueError(
                f"metrics already contain {recorder.count} records for {stage}/{phase}; "
                "resume from the compatible latest checkpoint or explicitly reset "
                "the phase before starting fresh"
            )
        recorder.reset_for_fresh_run()
    state = None
    if resume_state is not None:
        state = require_training_state(resume_state)
        validate_checkpoint_run_id(state, run_id, source="resume checkpoint")
        if state.get("stage") != stage or state.get("phase") != phase:
            raise ValueError(
                f"resume checkpoint belongs to {state.get('stage')}/{state.get('phase')}, "
                f"not {stage}/{phase}"
            )
        if int(state.get("target_epochs", target_epochs)) != target_epochs:
            raise ValueError(
                f"resume target epochs differ: checkpoint has {state.get('target_epochs')}, "
                f"requested {target_epochs}"
            )
        expected_recipe = recipe_fingerprint(recipe or dict(train_cfg))
        if state.get("recipe_fingerprint") != expected_recipe:
            raise ValueError(
                f"resume recipe mismatch for {stage}/{phase}; checkpoint was created "
                "with a different architecture, data/training recipe, seed, or stage configuration"
            )
        model.load_state_dict(state["model_state_dict"], strict=True)
        checkpoint_stage_state = state.get("stage_specific_state", {})
        if kd is not None:
            if checkpoint_stage_state.get("kd_state_dict") is None:
                raise ValueError(
                    f"resume checkpoint for {stage}/{phase} is missing required KD state"
                )
            kd.load_state_dict(checkpoint_stage_state["kd_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        if state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state)
        start_epoch = int(state.get("completed_epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_val_acc = float(state.get("best_metric_value", float("-inf")))
        best_epoch = int(state.get("best_epoch", 0))
        best_state = copy.deepcopy(state.get("best_model_state_dict") or state["model_state_dict"])
        history = list(state.get("history", []))
        run_id = state.get("run_id") or run_id or uuid.uuid4().hex
        if recorder is not None:
            recorder.reconcile(
                int(state.get("history_length", recorder.count)),
                state.get("last_metric_digest"),
            )
            history = list(recorder.records)
        # The explicit generator is the reproducibility boundary for shuffle.
        for loader, loader_state in zip(
            (train_loader, val_loader),
            checkpoint_stage_state.get("loader_generator_states", []),
        ):
            if getattr(loader, "generator", None) is not None and loader_state is not None:
                loader.generator.set_state(loader_state)
    else:
        start_epoch = 0
        global_step = 0
        best_val_acc = float("-inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] = {}
        history = list(recorder.records) if recorder is not None else []
        run_id = run_id or uuid.uuid4().hex

    final_val_acc = float(history[-1].get("val_acc", 0.0)) if history else 0.0
    phase_started = time.monotonic()

    def save_latest(completed_epoch: int) -> None:
        if latest_path is None:
            return
        state_payload = build_training_state(
            run_id=run_id,
            stage=stage,
            phase=phase,
            completed_epoch=completed_epoch,
            target_epochs=target_epochs,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric_name="validation_accuracy",
            best_metric_value=best_val_acc,
            best_epoch=best_epoch,
            best_model_state_dict=best_state or model.state_dict(),
            history=history,
            metrics=recorder,
            recipe=recipe or dict(train_cfg),
            upstream=upstream,
            stage_specific_state={
                **(stage_specific_state or {}),
                "kd_state_dict": (
                    copy.deepcopy(kd.state_dict()) if kd is not None else None
                ),
                "loader_generator_states": [
                    loader.generator.get_state().clone()
                    if getattr(loader, "generator", None) is not None else None
                    for loader in (train_loader, val_loader)
                ],
            },
        )
        # Keep the full history in the latest state so resuming still works if
        # a user moves the run directory and the JSONL sidecar is unavailable.
        state_payload["history"] = copy.deepcopy(history)
        atomic_torch_save(latest_path, state_payload)

    for epoch in range(start_epoch, target_epochs):
        epoch_started = time.monotonic()
        set_train_mode_preserving_frozen_batchnorm(model)
        if kd is not None:
            kd.train()
        running = {"total": 0.0}
        seen = 0
        correct = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            if kd is None:
                logits = model(features)
                losses = {"total": criterion(logits, labels)}
            else:
                feature_model = cast(FeatureModel, model)
                student_features = (
                    feature_model.forward_features(features) if kd.uses_features else None
                )
                logits = (
                    feature_model.classify_features(student_features)
                    if student_features is not None
                    else model(features)
                )
                losses = kd(features, logits, labels, student_features)
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
            if post_step is not None:
                post_step(model)
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value.item() * labels.size(0)
            correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)

        train_losses = {name: value / max(seen, 1) for name, value in running.items()}
        if kd is not None:
            kd.eval()
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
        final_val_acc = val_acc
        record = {
            "schema_version": 1,
            "stage": stage,
            "phase": phase,
            "epoch": epoch + 1,
            "global_step": global_step + len(train_loader),
            "elapsed_seconds": time.monotonic() - epoch_started,
            "train_loss": train_losses["total"],
            "train_accuracy": correct / max(seen, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_accuracy": val_acc,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "learning_rate": [group["lr"] for group in optimizer.param_groups],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "seed": train_cfg.get("seed"),
            **{f"train_{name}": value for name, value in train_losses.items() if name != "total"},
        }
        if extra_epoch_fields is not None:
            record.update(dict(extra_epoch_fields()))
        global_step += len(train_loader)
        history.append(record)
        metric_digest = recorder.append(record) if recorder is not None else None
        logger.info(
            "%s epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            label, epoch + 1, target_epochs, train_losses["total"], val_loss, val_acc,
        )

        if not best_state or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            if on_best is not None:
                on_best(model, val_acc)
            elif best_path is not None:
                atomic_torch_save(
                    best_path,
                    {
                        "model_state_dict": copy.deepcopy(best_state),
                        "stage": stage,
                        "phase": phase,
                        "val_acc": val_acc,
                        "run_id": run_id,
                    },
                )
        save_latest(epoch + 1)
        if on_epoch_end is not None:
            on_epoch_end({**record, "metric_digest": metric_digest})

    return FinetuneResult(
        best_val_acc=0.0 if best_val_acc == float("-inf") else best_val_acc,
        final_val_acc=final_val_acc,
        epochs=target_epochs,
        history=history,
        global_step=global_step,
        completed_epoch=target_epochs if target_epochs >= start_epoch else start_epoch,
        run_id=run_id,
    )


def train_model(model, datasets, label_map: dict, model_cfg: dict, train_cfg: dict,
                 checkpoint_path: Path, device, num_keywords: int, *,
                 kd=None, post_step=None, parameters=None, extra_checkpoint_fields=None,
                 output_dir: str | Path | None = None, stage: str = "train",
                 phase: str | None = None, resume_state: dict | None = None,
                 resume_from: str | Path | None = None, recorder: MetricsRecorder | None = None,
                 recipe: dict | None = None, upstream: dict | None = None,
                 run_id: str | None = None,
                 stage_specific_state: dict | None = None,
                 artifact_candidate: str | None = None,
                 reset_metrics: bool = False) -> FinetuneResult:
    """Train ``model`` in place and persist best and latest state separately.

    Used to train a fresh model, and to fine-tune one that has already been
    structurally modified (after channel pruning, for example). `kd` adds the
    fixed-teacher distillation loss to the task loss; `post_step` re-imposes a
    sparsity mask or codebook after each optimizer step.
    """
    phase = phase or model_cfg.get("name", "train")
    output_layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(output_layout, run_id)
    if output_layout is not None:
        output_layout.ensure_tree()
        best_path = output_layout._descendant(checkpoint_path)
        latest_path = output_layout.checkpoint_path(
            stage, phase, "latest", candidate=artifact_candidate
        )
        recorder = recorder or MetricsRecorder(
            output_layout.metrics_path(stage, phase, candidate=artifact_candidate),
            layout=output_layout,
            stage=stage,
            phase=phase,
        )
    else:
        best_path = Path(checkpoint_path)
        latest_path = best_path.with_name("latest.pt")
    train_generator = torch.Generator()
    train_generator.manual_seed(int(train_cfg.get("seed", 0)))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(train_cfg.get("seed", 0)) + 1)
    train_loader = build_data_loader(
        datasets[TRAIN], train_cfg, shuffle=True, generator=train_generator
    )
    val_loader = build_data_loader(
        datasets[VAL], train_cfg, shuffle=False, generator=val_generator
    )
    logger.info(
        "DataLoader workers=%d persistent=%s prefetch_factor=%s",
        train_loader.num_workers,
        train_loader.persistent_workers,
        train_loader.prefetch_factor,
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is not None:
        resume_state = torch.load(resume_from, map_location=device, weights_only=False)
        validate_checkpoint_run_id(resume_state, run_id, source=resume_from)
    run_id = run_id or (
        resume_state.get("run_id") if resume_state is not None else None
    ) or uuid.uuid4().hex

    def save_best(trained_model, val_acc):
        checkpoint = {
            "model_state_dict": trained_model.state_dict(),
            "model_cfg": model_cfg,
            "input_shape": trained_model.input_shape,
            "num_classes": trained_model.fc.out_features,
            "label_map": label_map,
            "num_keywords": num_keywords,
            "val_acc": val_acc,
            "seed": train_cfg["seed"],
            "run_id": run_id,
        }
        if extra_checkpoint_fields:
            checkpoint.update(extra_checkpoint_fields)
        checkpoint["run_id"] = run_id
        atomic_torch_save(best_path, checkpoint)
        logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, best_path)

    result = run_finetune(
        model,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        parameters=parameters,
        post_step=post_step,
        on_best=save_best,
        recorder=recorder,
        resume_state=resume_state,
        latest_path=latest_path,
        best_path=best_path,
        stage=stage,
        phase=phase,
        run_id=run_id,
        recipe=recipe or {
            "model": model_cfg,
            "train": dict(train_cfg),
            "stage": stage,
            "phase": phase,
        },
        upstream=upstream,
        stage_specific_state=stage_specific_state,
        reset_metrics=reset_metrics,
        label=model_cfg.get("name", "train"),
    )
    # A split run can have a latest checkpoint but no best artifact if it was
    # interrupted between validation and the first best write.  Re-materialize
    # the best deployment state from the latest checkpoint in that case.
    best_is_stale = False
    if best_path.exists():
        try:
            existing_best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_is_stale = abs(
                float(existing_best.get("val_acc", float("-inf"))) - result.best_val_acc
            ) > 1e-12
        except (OSError, ValueError, RuntimeError):
            best_is_stale = True
    if (not best_path.exists() or best_is_stale) and latest_path.exists():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        checkpoint = {
            "model_state_dict": latest["best_model_state_dict"],
            "model_cfg": model_cfg,
            "input_shape": model.input_shape,
            "num_classes": model.fc.out_features,
            "label_map": label_map,
            "num_keywords": num_keywords,
            "val_acc": latest.get("best_metric_value", 0.0),
            "seed": train_cfg.get("seed"),
            "run_id": run_id,
        }
        if extra_checkpoint_fields:
            checkpoint.update(extra_checkpoint_fields)
        checkpoint["run_id"] = run_id
        atomic_torch_save(best_path, checkpoint)
    if output_layout is not None:
        write_phase_summary(
            output_layout,
            stage=stage,
            phase=phase,
            result=result,
            artifacts={"best_checkpoint": best_path, "latest_checkpoint": latest_path},
            candidate=artifact_candidate,
        )
    return result


def train(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    checkpoint_path: Path,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    resume_from: str | Path | None = None,
    reset_metrics: bool = False,
    run_id: str | None = None,
):
    train_cfg = with_seed(train_cfg, seed)
    set_seed(train_cfg["seed"])
    device = get_device()

    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_keywords = len(data_cfg["target_keywords"])

    sample_features, _ = datasets[TRAIN][0]
    input_shape = (sample_features.shape[-2], sample_features.shape[-1])
    num_classes = len(label_names)

    model = build_ds_cnn(model_cfg, input_shape, num_classes).to(device)
    logger.info("Model %s: %d params, input_shape=%s, num_classes=%d",
                model_cfg["name"], sum(p.numel() for p in model.parameters()), input_shape, num_classes)

    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    phase = model_cfg.get("name", "train")
    if layout is not None:
        layout.ensure_tree()
        requested_checkpoint = Path(checkpoint_path)
        checkpoint_path = layout.legacy_output_path(
            requested_checkpoint,
            category="models/checkpoints/teacher",
            default="best.pt",
        )
        latest_path = layout.checkpoint_path("teacher", phase, "latest")
        if resume and resume_from is None and latest_path.exists():
            resume_from = latest_path
    elif resume and resume_from is None:
        latest_path = Path(checkpoint_path).with_name("latest.pt")
        if latest_path.exists():
            resume_from = latest_path

    return train_model(
        model,
        datasets,
        label_map,
        model_cfg,
        train_cfg,
        Path(checkpoint_path),
        device,
        num_keywords,
        output_dir=output_dir,
        stage="teacher",
        phase=phase,
        resume_from=resume_from,
        recipe={
            "data": data_cfg,
            "model": model_cfg,
            "train": train_cfg,
            "stage": "teacher",
            "phase": phase,
        },
        run_id=run_id,
        reset_metrics=reset_metrics,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=False)
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

    if args.checkpoint is None and args.output_dir is None:
        parser.error("--checkpoint is required unless --output-dir is supplied")

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path("models/checkpoints/best.pt")
    )
    with __import__("kws.utils.logging", fromlist=["run_session"]).run_session(
        args.output_dir,
        command="kws.train",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=[
            (args.data_config, "data_config"),
            (args.model_config, "model_config"),
            (args.train_config, "train_config"),
        ],
    ):
        train(
            data_cfg,
            model_cfg,
            train_cfg,
            checkpoint,
            seed=args.seed,
            output_dir=args.output_dir,
            resume=args.resume,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
