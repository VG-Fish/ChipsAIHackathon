"""Framework step 2: distill the fixed teacher into the deployment student.

The three losses -- feature, response, and ground-truth classification -- come
from Song et al., "Knowledge Distillation for In-Memory Keyword Spotting
Model" (Interspeech 2022). They now live in :mod:`kws.optimize.kd` because
every later stage distills from the same fixed teacher with the same objective;
this module is the stage that produces the strong student baseline the sparsity
sweep starts from.

Teacher and student encoder widths differ here, so a training-only linear
adapter maps student features to the teacher width. The adapter is
intentionally absent from the saved deployment model.
"""
import argparse
import hashlib
import uuid
from pathlib import Path

import torch
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.optimize.kd import (
    DistillationCriterion,
    FrozenTeacher,
    KDWeights,
    distillation_losses,
    file_sha256,
)
from kws.train import (
    resolve_manifest_run_id,
    run_finetune,
    validate_checkpoint_run_id,
)
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import MetricsRecorder, atomic_torch_save, write_phase_summary
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
from kws.utils.seed import set_seed, with_seed

logger = get_logger(__name__)


def distillation_fingerprint(
    teacher_checkpoint: str,
    student_model_cfg: dict,
    data_cfg: dict,
    train_cfg: dict,
    student_checkpoint: str | None = None,
) -> str:
    """Stable identity for the complete stage-2 training recipe.

    A checkpoint is reusable only when its architecture, teacher contents,
    data recipe, optimization/KD settings, and optional warm start all match.
    Paths alone are not sufficient because a file can be replaced in place.
    """
    payload = {
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha256": file_sha256(teacher_checkpoint),
        "student_model": student_model_cfg,
        "data": data_cfg,
        "train": train_cfg,
        "student_checkpoint": student_checkpoint,
        "student_checkpoint_sha256": (
            file_sha256(student_checkpoint) if student_checkpoint else None
        ),
    }
    encoded = yaml.safe_dump(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def imc_distillation_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    feature_weight: float,
    response_weight: float,
    classification_weight: float,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """The paper's weighted feature, response, and classification losses.

    Kept as the paper-named entry point onto the shared implementation.
    """
    weights = KDWeights(
        temperature=temperature,
        feature_weight=feature_weight,
        response_weight=response_weight,
        classification_weight=classification_weight,
    )
    return distillation_losses(
        student_logits,
        teacher_logits,
        labels,
        weights=weights,
        label_smoothing=label_smoothing,
        student_features=student_features,
        teacher_features=teacher_features,
    )


def distill(
    teacher_checkpoint: str,
    student_model_cfg: dict,
    data_cfg: dict,
    train_cfg: dict,
    out_checkpoint: Path,
    student_checkpoint: str | None = None,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    resume_from: str | Path | None = None,
    reset_metrics: bool = False,
    run_id: str | None = None,
) -> object:
    """Train the student against the fixed teacher and return full history."""
    train_cfg = with_seed(train_cfg, seed)
    set_seed(train_cfg["seed"])
    device = get_device()
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)

    if run_id is not None and (
        layout is None
        or Path(teacher_checkpoint).expanduser().resolve().is_relative_to(layout.root)
    ):
        teacher_checkpoint_state = torch.load(
            teacher_checkpoint, map_location="cpu", weights_only=False
        )
        validate_checkpoint_run_id(
            teacher_checkpoint_state, run_id, source=teacher_checkpoint
        )
    teacher = FrozenTeacher(teacher_checkpoint, device)
    input_shape = teacher.input_shape
    num_classes = teacher.num_classes
    recipe_fingerprint = distillation_fingerprint(
        teacher_checkpoint,
        student_model_cfg,
        data_cfg,
        train_cfg,
        student_checkpoint,
    )

    student = build_ds_cnn(student_model_cfg, input_shape, num_classes).to(device)
    if student_checkpoint is not None:
        initial = torch.load(student_checkpoint, map_location=device, weights_only=False)
        if run_id is not None and (
            layout is None
            or Path(student_checkpoint).expanduser().resolve().is_relative_to(layout.root)
        ):
            validate_checkpoint_run_id(initial, run_id, source=student_checkpoint)
        if initial["model_cfg"] != student_model_cfg:
            raise ValueError("student checkpoint architecture does not match student model config")
        student.load_state_dict(initial["model_state_dict"])

    datasets, label_map = build_datasets(
        data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"],
    )
    if len(label_map) != num_classes:
        raise ValueError(
            f"data config has {len(label_map)} classes but teacher has {num_classes}"
        )
    train_generator = torch.Generator().manual_seed(int(train_cfg["seed"]))
    val_generator = torch.Generator().manual_seed(int(train_cfg["seed"]) + 1)
    train_loader = build_data_loader(
        datasets[TRAIN], train_cfg, shuffle=True, generator=train_generator
    )
    val_loader = build_data_loader(
        datasets[VAL], train_cfg, shuffle=False, generator=val_generator
    )

    kd = DistillationCriterion(
        teacher,
        KDWeights.from_config(train_cfg.get("distillation")),
        train_cfg["label_smoothing"],
        device,
        student_feature_dim=student.fc.in_features,
    )
    if layout is not None:
        layout.ensure_tree()
        requested_checkpoint = Path(out_checkpoint)
        out_checkpoint = layout.legacy_output_path(
            requested_checkpoint,
            category="models/checkpoints/student",
            default="best.pt",
        )
        latest_path = layout.checkpoint_path("student", "distill", "latest")
        recorder = MetricsRecorder(
            layout.metrics_path("student", "distill"),
            layout=layout,
            stage="student",
            phase="distill",
        )
        if resume_from is None and resume and latest_path.exists():
            resume_from = latest_path
    else:
        latest_path = out_checkpoint.with_name("latest.pt")
        recorder = None
    resume_state = None
    if resume_from is not None:
        resume_state = torch.load(resume_from, map_location=device, weights_only=False)
        validate_checkpoint_run_id(resume_state, run_id, source=resume_from)
    run_id = run_id or (
        resume_state.get("run_id") if resume_state is not None else None
    ) or uuid.uuid4().hex
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save_best(model, val_acc):
        atomic_torch_save(
            out_checkpoint,
            {
                "model_state_dict": model.state_dict(),
                "model_cfg": student_model_cfg,
                "input_shape": input_shape,
                "num_classes": num_classes,
                "label_map": label_map,
                "num_keywords": teacher.num_keywords,
                "val_acc": val_acc,
                "run_id": run_id,
                "distillation": {
                    "method": "song22c_imc_feature_response",
                    "student_initialization": (
                        str(student_checkpoint) if student_checkpoint else "fresh"
                    ),
                    "recipe_fingerprint": recipe_fingerprint,
                    **kd.describe(),
                },
            },
        )
        logger.info("Saved new best distilled checkpoint (val_acc=%.4f) -> %s", val_acc, out_checkpoint)

    result = run_finetune(
        student,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        on_best=save_best,
        recorder=recorder,
        resume_state=resume_state,
        latest_path=latest_path,
        stage="student",
        phase="distill",
        recipe={
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_sha256": file_sha256(teacher_checkpoint),
            "student_model": student_model_cfg,
            "data": data_cfg,
            "train": train_cfg,
            "distillation": kd.describe(),
            "student_checkpoint": student_checkpoint,
            "student_checkpoint_sha256": (
                file_sha256(student_checkpoint) if student_checkpoint else None
            ),
        },
        upstream={
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_sha256": file_sha256(teacher_checkpoint),
        },
        run_id=run_id,
        reset_metrics=reset_metrics,
        label=f'distill-{student_model_cfg["name"]}',
    )
    if latest_path.exists():
        regenerate_best = not out_checkpoint.exists()
        if out_checkpoint.exists():
            try:
                current_best = torch.load(
                    out_checkpoint, map_location="cpu", weights_only=False
                )
                validate_checkpoint_run_id(
                    current_best, run_id, source=out_checkpoint
                )
                regenerate_best = abs(
                    float(current_best.get("val_acc", float("-inf")))
                    - result.best_val_acc
                ) > 1e-12
            except (OSError, ValueError, RuntimeError):
                regenerate_best = True
        if regenerate_best:
            latest = torch.load(latest_path, map_location=device, weights_only=False)
            validate_checkpoint_run_id(latest, run_id, source=latest_path)
            student.load_state_dict(latest["best_model_state_dict"], strict=True)
            save_best(student, result.best_val_acc)
    if layout is not None:
        write_phase_summary(
            layout,
            stage="student",
            phase="distill",
            result=result,
            artifacts={"best_checkpoint": out_checkpoint, "latest_checkpoint": latest_path},
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/distill_imc.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-model-config", default="configs/model/ds_cnn_xs.yaml")
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        help="Optional warm-start checkpoint; omit to train the student from scratch",
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
    with open(args.student_model_config) as f:
        student_model_cfg = yaml.safe_load(f)

    inputs = [
        (args.data_config, "data_config"),
        (args.train_config, "train_config"),
        (args.student_model_config, "student_model_config"),
        (args.teacher_checkpoint, "teacher_checkpoint"),
    ]
    if args.student_checkpoint:
        inputs.append((args.student_checkpoint, "student_warm_start"))
    with run_session(
        args.output_dir,
        command="kws.optimize.distill",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=inputs,
    ):
        result = distill(
            args.teacher_checkpoint,
            student_model_cfg,
            data_cfg,
            train_cfg,
            Path(args.out_checkpoint or "models/checkpoints/student/best.pt"),
            student_checkpoint=args.student_checkpoint,
            seed=args.seed,
            output_dir=args.output_dir,
            resume=args.resume,
            resume_from=args.resume_from,
        )
        if args.output_dir is not None:
            ArtifactLayout(args.output_dir).atomic_yaml(
                ArtifactLayout(args.output_dir).report_path("distill.yaml"),
                result.as_dict(),
            )


if __name__ == "__main__":
    main()
