"""The compression framework, stage by stage.

    1. Train a strong full-precision teacher.
    2. Choose a deployment-shaped student; distill teacher -> student.
    3. For each sparsity target: prune, KD fine-tune, run the perforation /
       dendrite phase, resume KD fine-tuning, record accuracy and deployment
       cost, and stop when the Pareto frontier stops improving.
    4. Apply layer-wise weight clustering / codebook learning.
    5. Run quantization-aware distillation fine-tuning.
    6. Export and benchmark the exact inference graph on the intended target.

Each stage is independently runnable and reuses whatever earlier stages already
produced, because stage 3 alone takes hours and re-running it to reach stage 5
would be the difference between using this and not. `--stages` selects which to
run; a stage whose output already exists is reused unless `--force` is given.

One invariant holds throughout: the teacher from stage 1 is loaded once and
never updated, so "KD loss from the fixed teacher" means the same teacher in
stages 2, 3b, 3d, 4, and 5.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.export.benchmark import benchmark_module, benchmark_scripted_model
from kws.models.ds_cnn import FeatureModel, build_ds_cnn
from kws.optimize.cluster import (
    ClusterSpec,
    CodebookProjector,
    apply_weight_clustering,
    bake_codebooks,
    codebook_finetune,
    load_clustered_model,
    save_clustered_model,
)
from kws.optimize.dendritic import FRAMEWORK_CYCLE_VERSION, load_yaml
from kws.optimize.dendritic_prune_loop import run_pruning_search, search_fingerprint
from kws.optimize.distill import distill, distillation_fingerprint
from kws.optimize.kd import (
    DistillationCriterion,
    FrozenTeacher,
    KDWeights,
    file_sha256,
    supports_pooled_features,
)
from kws.optimize.quantize_qat import quantize_aware_distill_model
from kws.train import train as train_from_scratch
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.profile import profile_model
from kws.utils.seed import resolve_seed, set_seed, with_seed

logger = get_logger(__name__)

STAGES = ("teacher", "student", "sparsity", "cluster", "quantize", "benchmark")


def _fingerprint(payload: dict) -> str:
    encoded = yaml.safe_dump(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_artifact_id(model: torch.nn.Module, label: str) -> str:
    """Content identity for a live float graph, independent of its file name."""
    digest = hashlib.sha256(label.encode())
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return f"{label}:{digest.hexdigest()}"


def _cluster_recipe_id(config: dict, *, seed: int | None = None) -> str:
    cluster_cfg = config["cluster"]
    train_cfg = with_seed(load_yaml(cluster_cfg["train_config"]), seed)
    return _fingerprint(
        {
            "cluster": cluster_cfg,
            "train": train_cfg,
            "data": load_yaml(config["data_config"]),
            "teacher_checkpoint": config["teacher"]["checkpoint"],
            "teacher_sha256": file_sha256(config["teacher"]["checkpoint"]),
        }
    )


def _quantize_recipe_id(config: dict, *, seed: int | None = None) -> str:
    quantize_cfg = config["quantize"]
    train_cfg = with_seed(load_yaml(quantize_cfg["train_config"]), seed)
    return _fingerprint(
        {
            "quantize": quantize_cfg,
            "train": train_cfg,
            "data": load_yaml(config["data_config"]),
            "teacher_checkpoint": config["teacher"]["checkpoint"],
            "teacher_sha256": file_sha256(config["teacher"]["checkpoint"]),
        }
    )


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        yaml.safe_dump(payload, file, sort_keys=False, default_flow_style=False)
    temporary.replace(path)


def _json_safe(value):
    """YAML-safe report values; tensors and Paths appear in stage results."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.nn.Module):
        return f"<{type(value).__name__}>"
    return value


def stage_teacher(config: dict, *, force: bool, seed: int | None = None) -> dict:
    """Step 1: a strong full-precision teacher, trained or reused."""
    teacher_cfg = config["teacher"]
    checkpoint = Path(teacher_cfg["checkpoint"])
    teacher_train_cfg = with_seed(load_yaml(teacher_cfg["train_config"]), seed)
    if checkpoint.exists() and not force:
        existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_model = load_yaml(teacher_cfg["model_config"])
        if existing.get("model_cfg") != expected_model:
            raise ValueError(
                f"teacher checkpoint {checkpoint} does not match "
                f"{teacher_cfg['model_config']}"
            )
        if seed is not None and existing.get("seed") != teacher_train_cfg["seed"]:
            raise ValueError(
                f"teacher checkpoint {checkpoint} was trained with seed "
                f"{existing.get('seed')!r}, not {teacher_train_cfg['seed']}; "
                "retrain it with --force"
            )
        logger.info(
            "Stage 1: reusing teacher %s (val_acc=%s)",
            checkpoint,
            existing.get("val_acc"),
        )
        model = build_ds_cnn(
            existing["model_cfg"],
            tuple(existing["input_shape"]),
            existing["num_classes"],
        )
        return {
            "status": "reused",
            "checkpoint": str(checkpoint),
            "model": existing["model_cfg"]["name"],
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "val_acc": existing.get("val_acc"),
        }

    logger.info("Stage 1: training teacher %s", teacher_cfg["model_config"])
    val_acc = train_from_scratch(
        load_yaml(config["data_config"]),
        load_yaml(teacher_cfg["model_config"]),
        teacher_train_cfg,
        checkpoint,
        seed=seed,
    )
    return {"status": "trained", "checkpoint": str(checkpoint), "val_acc": val_acc}


def _resolve_optional_checkpoint(path: str | None) -> str | None:
    """Use a configured warm start only when its artifact is available."""
    if not path:
        return None
    if Path(path).is_file():
        return path
    logger.warning(
        "Configured optional student warm-start checkpoint %s is missing; "
        "training the student from scratch",
        path,
    )
    return None


def stage_student(config: dict, *, force: bool, seed: int | None = None) -> dict:
    """Step 2: distill the fixed teacher into the deployment student."""
    student_cfg = config["student"]
    checkpoint = Path(student_cfg["checkpoint"])
    warm_start_checkpoint = _resolve_optional_checkpoint(
        student_cfg.get("warm_start_checkpoint")
    )
    student_model_cfg = load_yaml(student_cfg["model_config"])
    student_data_cfg = load_yaml(config["data_config"])
    student_train_cfg = with_seed(load_yaml(student_cfg["train_config"]), seed)
    expected_recipe = distillation_fingerprint(
        config["teacher"]["checkpoint"],
        student_model_cfg,
        student_data_cfg,
        student_train_cfg,
        warm_start_checkpoint,
    )
    if checkpoint.exists() and not force:
        existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if existing.get("model_cfg") != student_model_cfg:
            raise ValueError(
                f"student checkpoint {checkpoint} does not match "
                f"{student_cfg['model_config']}"
            )
        recorded_teacher = (existing.get("distillation") or {}).get(
            "teacher_checkpoint"
        )
        if recorded_teacher != config["teacher"]["checkpoint"]:
            raise ValueError(
                f"student checkpoint {checkpoint} used teacher "
                f"{recorded_teacher!r}, not {config['teacher']['checkpoint']!r}"
            )
        recorded_teacher_sha = (existing.get("distillation") or {}).get(
            "teacher_sha256"
        )
        expected_teacher_sha = file_sha256(config["teacher"]["checkpoint"])
        if recorded_teacher_sha != expected_teacher_sha:
            raise ValueError(
                f"student checkpoint {checkpoint} has teacher digest "
                f"{recorded_teacher_sha!r}, expected {expected_teacher_sha!r}; "
                "re-distill it with --force"
            )
        recorded_recipe = (existing.get("distillation") or {}).get("recipe_fingerprint")
        if recorded_recipe != expected_recipe:
            raise ValueError(
                f"student checkpoint {checkpoint} has recipe fingerprint "
                f"{recorded_recipe!r}, expected {expected_recipe!r}; "
                "re-distill it with --force"
            )
        logger.info(
            "Stage 2: reusing distilled student %s (val_acc=%s)",
            checkpoint,
            existing.get("val_acc"),
        )
        return {
            "status": "reused",
            "checkpoint": str(checkpoint),
            "model": existing["model_cfg"]["name"],
            "val_acc": existing.get("val_acc"),
            "distillation": existing.get("distillation"),
            "recipe_fingerprint": expected_recipe,
        }

    logger.info(
        "Stage 2: distilling %s -> %s",
        config["teacher"]["checkpoint"],
        student_cfg["model_config"],
    )
    val_acc = distill(
        config["teacher"]["checkpoint"],
        student_model_cfg,
        student_data_cfg,
        student_train_cfg,
        checkpoint,
        student_checkpoint=warm_start_checkpoint,
        seed=seed,
    )
    return {
        "status": "distilled",
        "checkpoint": str(checkpoint),
        "val_acc": val_acc,
        "recipe_fingerprint": expected_recipe,
    }


def stage_sparsity(config: dict, *, force: bool, seed: int | None = None) -> dict:
    """Step 3: the sparsity sweep, stopping on the Pareto frontier."""
    search_cfg = load_yaml(config["sparsity"]["search_config"])
    if force:
        # PAI run directories contain many interdependent checkpoints. Never
        # partially overwrite one under the broad --force flag; disabling
        # reuse makes the search either start cleanly or tell the caller to
        # choose a new save_prefix.
        search_cfg = dict(search_cfg)
        search_cfg["reuse_completed_start_run"] = None
        search_cfg["reuse_completed_candidates"] = False
    data_cfg = load_yaml(config["data_config"])
    train_cfg = with_seed(load_yaml(config["sparsity"]["train_config"]), seed)
    teacher_checkpoint = config["teacher"]["checkpoint"]
    expected_fingerprint = search_fingerprint(
        config["student"]["checkpoint"],
        teacher_checkpoint,
        data_cfg,
        train_cfg,
        search_cfg,
    )
    summary_path = Path(search_cfg["summary_path"])
    if summary_path.exists() and not force:
        with summary_path.open() as file:
            summary = yaml.safe_load(file)
        if (
            summary.get("status") == "complete"
            and summary.get("stopping_rule")
            in {"pareto_frontier_stalled", "minimum_channels_reached"}
            and summary.get("pareto")
            and summary.get("fingerprint") == expected_fingerprint
        ):
            logger.info("Stage 3: reusing completed sweep from %s", summary_path)
            return summary

    return run_pruning_search(
        config["student"]["checkpoint"],
        data_cfg,
        train_cfg,
        search_cfg,
        teacher_checkpoint=teacher_checkpoint,
        seed=seed,
    )


def validate_sparsity_report(
    sweep: dict, config: dict, *, seed: int | None = None
) -> None:
    """Reject a step-3 report whose inputs no longer match this pipeline.

    Downstream-only invocations are intentionally supported, but they must not
    silently deploy a candidate selected under a different teacher, student,
    data recipe, or sparsity configuration.
    """
    if not isinstance(sweep, dict):
        raise ValueError("stage 3 report is not a mapping")
    search_cfg = load_yaml(config["sparsity"]["search_config"])
    expected = search_fingerprint(
        config["student"]["checkpoint"],
        config["teacher"]["checkpoint"],
        load_yaml(config["data_config"]),
        with_seed(load_yaml(config["sparsity"]["train_config"]), seed),
        search_cfg,
    )
    if sweep.get("fingerprint") != expected:
        raise ValueError(
            "stage 3 report fingerprint does not match the current pipeline "
            "inputs; rerun the sparsity stage before using stages 4-6"
        )
    if sweep.get("status") != "complete" or not sweep.get("pareto"):
        raise ValueError(
            "stage 3 report is not a completed sparsity sweep; rerun the "
            "sparsity stage before using stages 4-6"
        )


def select_deployment_candidate(sweep: dict, config: dict) -> dict:
    """Pick the frontier point stages 4-6 continue from.

    The frontier is the deliverable of step 3; choosing one point off it is a
    product decision, so it is a config knob (`select_by`) rather than a
    hard-coded "smallest" or "most accurate".
    """
    pareto = sweep.get("pareto") or {}
    frontier = pareto.get("frontier") or []
    if not frontier:
        raise ValueError(
            "The sparsity sweep produced no Pareto frontier; nothing to deploy"
        )
    select_by = config["sparsity"].get("select_by", "deployed_params")
    if select_by == "accuracy":
        chosen = max(frontier, key=lambda point: point["accuracy"])
    else:
        unavailable = [
            point.get("label", "?")
            for point in frontier
            if select_by not in (point.get("costs") or {})
        ]
        if unavailable:
            raise ValueError(
                f"cannot select by unrecorded cost {select_by!r}; missing on "
                f"frontier points {unavailable}"
            )
        chosen = min(frontier, key=lambda point: point["costs"][select_by])
    logger.info(
        "Selected %s off the frontier by %s: accuracy %.4f, costs %s",
        chosen["label"],
        select_by,
        chosen["accuracy"],
        chosen["costs"],
    )
    return chosen


def load_candidate_model(
    candidate: dict, config: dict
) -> tuple[torch.nn.Module, tuple[int, int], int]:
    """Rebuild the selected candidate's deployment graph.

    The dendritic graph is PerforatedAI's clean export, whose architecture is
    not reconstructible from a model config -- so the base is rebuilt from the
    run's recorded metadata, re-perforated to the same shape, and loaded from
    the run's own clean checkpoint.
    """
    UPA: Any = importlib.import_module("perforatedai.utils_perforatedai")

    from kws.optimize.dendritic import (
        build_cycle_base,
        configure_perforatedai,
        ensure_clean_dendrite_skip_weights,
    )

    save_name = candidate["save_name"]
    metadata_path = Path(save_name) / "cycle_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is missing; stage 4 cannot rebuild {save_name!r}"
        )
    with metadata_path.open() as file:
        metadata = yaml.safe_load(file)
    if metadata.get("framework_cycle_version") != FRAMEWORK_CYCLE_VERSION:
        raise ValueError(
            f"candidate {save_name!r} predates framework cycle version "
            f"{FRAMEWORK_CYCLE_VERSION}; rerun the sparsity stage"
        )

    source_checkpoint = metadata["source_checkpoint"]
    if source_checkpoint != config["student"]["checkpoint"]:
        raise ValueError(
            f"candidate {save_name!r} was built from {source_checkpoint!r}, "
            f"not the configured student {config['student']['checkpoint']!r}"
        )
    source_digest = metadata.get("source_checkpoint_sha256")
    expected_source_digest = file_sha256(source_checkpoint)
    if source_digest != expected_source_digest:
        raise ValueError(
            f"candidate {save_name!r} has stale or missing source checkpoint "
            "content provenance; rerun the sparsity stage"
        )
    recorded_teacher = metadata.get("teacher_checkpoint")
    if (
        recorded_teacher is not None
        and recorded_teacher != config["teacher"]["checkpoint"]
    ):
        raise ValueError(
            f"candidate {save_name!r} used teacher {recorded_teacher!r}, "
            f"not the configured teacher {config['teacher']['checkpoint']!r}"
        )
    teacher_digest = metadata.get("teacher_sha256")
    expected_teacher_digest = file_sha256(config["teacher"]["checkpoint"])
    if teacher_digest != expected_teacher_digest:
        raise ValueError(
            f"candidate {save_name!r} has stale or missing teacher content "
            "provenance; rerun the sparsity stage"
        )
    candidate_width = candidate.get("width")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_width = source["model_cfg"]["block_channels"][0]
    width = metadata["base_model_cfg"]["block_channels"][0]
    if candidate_width is not None and int(candidate_width) != int(width):
        raise ValueError(
            f"candidate metadata width {candidate_width} does not match "
            f"recorded graph width {width}"
        )

    device = get_device()
    base, checkpoint, _ = build_cycle_base(source_checkpoint, width / source_width)
    configure_perforatedai(metadata["perforatedai"], device)
    model = UPA.perforate_model(
        base,
        doing_pai=False,
        save_name=f"{save_name}_reload",
        making_graphs=False,
        maximizing_score=True,
    ).to(device)

    # ``final_clean_pai.pt`` is a safetensors-style state file, not a PyTorch
    # state_dict that can create the missing dendrite branches by itself. Use
    # PAI's supported pretrained loader to restore the learned structure and
    # clean it for inference, then load the exact clean state that was exported.
    best_model = Path(save_name) / "best_model.pt"
    if not best_model.exists():
        raise FileNotFoundError(
            f"{best_model} is missing; cannot restore the selected PAI graph"
        )
    clean_path = Path(save_name) / "final_clean_pai.pt"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} is missing; cannot restore the selected clean graph"
        )
    if metadata.get("clean_artifact_sha256") != file_sha256(clean_path):
        raise ValueError(
            f"candidate {save_name!r} clean artifact digest does not match its "
            "metadata; rerun the sparsity stage"
        )
    model = UPA.load_pretrained_model(
        model,
        save_name,
        "best_model",
        remove_dendrite_scaffolding=True,
    ).to(device)
    clean_state = UPA.load_file(str(clean_path))
    ensure_clean_dendrite_skip_weights(model, clean_state)
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    # PAI's runtime-only tracker string is present in the reconstructed wrapper
    # but is intentionally absent from the exported clean tensor file. Every
    # inference tensor must still match exactly; only that non-inference field
    # is allowed to be missing.
    if unexpected or set(missing) - {"tracker_string"}:
        raise RuntimeError(
            "PAI clean checkpoint did not match the reconstructed graph: "
            f"missing={missing}, unexpected={unexpected}"
        )
    resume_record = (metadata.get("result") or {}).get("resume") or {}
    resumed_value = resume_record.get("checkpoint")
    resumed = Path(resumed_value) if resumed_value else None
    if resume_record.get("status") == "complete" and (
        resumed is None or not resumed.exists()
    ):
        raise FileNotFoundError(
            f"candidate {save_name!r} records a completed KD resume but its "
            f"checkpoint is missing: {resumed_value!r}"
        )
    if resume_record.get("status") == "complete":
        if resumed is None:
            raise FileNotFoundError(
                f"candidate {save_name!r} records a completed KD resume without "
                "a checkpoint path"
            )
        state = torch.load(resumed, map_location="cpu", weights_only=False)
        ensure_clean_dendrite_skip_weights(model, state["model_state_dict"])
        model.load_state_dict(state["model_state_dict"], strict=True)
        logger.info("Loaded the step-3d resumed weights from %s", resumed)
    return model, tuple(checkpoint["input_shape"]), checkpoint["num_keywords"]


def stage_cluster(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    config: dict,
    teacher: FrozenTeacher,
    device: torch.device,
    candidate_id: str,
    recipe_id: str,
    *,
    seed: int | None = None,
) -> tuple[dict, CodebookProjector]:
    """Step 4: layer-wise weight clustering, then learn the codebook."""
    cluster_cfg = config["cluster"]
    spec = ClusterSpec.from_config(cluster_cfg)
    train_cfg = with_seed(load_yaml(cluster_cfg["train_config"]), seed)

    set_seed(train_cfg["seed"])
    datasets, _ = build_datasets(
        load_yaml(config["data_config"]),
        augment=train_cfg["augment"],
        seed=train_cfg["seed"],
    )
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)

    model = model.to(device)
    report = apply_weight_clustering(model, spec)

    kd = DistillationCriterion(
        teacher,
        KDWeights.from_config(train_cfg.get("distillation")),
        train_cfg["label_smoothing"],
        device,
        student_feature_dim=(
            _feature_dim(model, input_shape)
            if supports_pooled_features(model, input_shape)
            else None
        ),
    )
    learned = codebook_finetune(
        model, train_loader, val_loader, device, train_cfg, kd=kd
    )
    projector = bake_codebooks(model, bits=spec.bits)
    artifact_path = Path(
        cluster_cfg.get("checkpoint", "models/exported/kws_clustered.pt")
    )
    save_clustered_model(
        model,
        projector,
        artifact_path,
        input_shape=input_shape,
        candidate_id=candidate_id,
        recipe_id=recipe_id,
    )
    return (
        {
            "clustering": report.as_dict(),
            "codebook_learning": learned,
            "distillation": kd.describe(),
            "checkpoint": str(artifact_path),
            "candidate_id": candidate_id,
            "recipe_id": recipe_id,
        },
        projector,
    )


def _feature_dim(model: torch.nn.Module, input_shape: tuple[int, int]) -> int:
    device = next(model.parameters()).device
    with torch.no_grad():
        feature_model = cast(FeatureModel, model)
        return feature_model.forward_features(
            torch.zeros(1, 1, *input_shape, device=device)
        ).shape[1]


def stage_quantize(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    num_keywords: int,
    config: dict,
    projector,
    candidate_id: str,
    source_artifact_id: str,
    recipe_id: str,
    *,
    seed: int | None = None,
) -> dict:
    """Step 5: quantization-aware distillation over the fake-quantized graph."""
    quantize_cfg = config["quantize"]
    return quantize_aware_distill_model(
        model,
        input_shape,
        load_yaml(config["data_config"]),
        with_seed(load_yaml(quantize_cfg["train_config"]), seed),
        Path(quantize_cfg["checkpoint"]),
        teacher_checkpoint=config["teacher"]["checkpoint"],
        codebook_projector=projector,
        num_keywords=num_keywords,
        candidate_id=candidate_id,
        source_artifact_id=source_artifact_id,
        recipe_id=recipe_id,
        seed=seed,
    )


def stage_benchmark(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    num_keywords: int,
    config: dict,
    torch_cost: dict | None = None,
    *,
    seed: int | None = None,
) -> dict:
    """Step 6: export the exact inference graph and benchmark it."""
    benchmark_cfg = config["benchmark"]
    benchmark_seed = int(benchmark_cfg.get("seed", 0) if seed is None else seed)
    if isinstance(model, torch.jit.ScriptModule):
        if torch_cost is None:
            raise ValueError(
                "the scripted inference artifact has no recorded eager cost; "
                "rerun the quantize stage before benchmarking"
            )
        result = benchmark_scripted_model(
            model,
            config["quantize"]["checkpoint"],
            input_shape,
            load_yaml(config["data_config"]),
            num_keywords,
            target=benchmark_cfg.get("torchscript_target", "torchscript-cpu"),
            split=benchmark_cfg.get("split", "testing"),
            latency_iterations=int(benchmark_cfg.get("latency_iterations", 200)),
            report_path=benchmark_cfg.get("report_path"),
            seed=benchmark_seed,
        )
        return {
            "inference_graph": result.as_dict(),
            "torch_cost": {
                **(torch_cost or {}),
                "graph_format": "torchscript",
                "graph_bytes": result.onnx_bytes,
            },
        }
    result = benchmark_module(
        model,
        input_shape,
        benchmark_cfg["onnx_path"],
        load_yaml(config["data_config"]),
        num_keywords,
        target=benchmark_cfg.get("target", "onnxruntime-cpu"),
        split=benchmark_cfg.get("split", "testing"),
        latency_iterations=int(benchmark_cfg.get("latency_iterations", 200)),
        report_path=benchmark_cfg.get("report_path"),
        seed=benchmark_seed,
    )
    cost = profile_model(
        model,
        input_shape,
        device=torch.device("cpu"),
        bits_per_weight=int(benchmark_cfg.get("bits_per_weight", 8)),
    )
    return {"onnx": result.as_dict(), "torch_cost": cost.as_dict()}


def run_pipeline(
    config: dict,
    stages: tuple[str, ...],
    *,
    force: bool,
    seed: int | None = None,
) -> dict:
    """Run the requested stages in order, threading artifacts between them."""
    if seed is not None:
        seed = resolve_seed({}, seed)
    report_path = Path(config["report_path"])
    report: dict = {"stages": {}, "config": config}
    if report_path.exists():
        with report_path.open() as file:
            report = yaml.safe_load(file) or report
        report.setdefault("stages", {})
        report["config"] = config
    if seed is not None:
        report["seed"] = seed

    def checkpoint_report() -> None:
        _write(report_path, _json_safe(report))

    if "teacher" in stages:
        teacher_report = stage_teacher(config, force=force, seed=seed)
        report["stages"]["1_teacher"] = teacher_report
        if force or teacher_report.get("status") != "reused":
            for stage in (
                "2_student",
                "3_sparsity",
                "4_cluster",
                "5_quantize",
                "6_benchmark",
            ):
                report["stages"].pop(stage, None)
        checkpoint_report()
    if "student" in stages:
        student_report = stage_student(config, force=force, seed=seed)
        report["stages"]["2_student"] = student_report
        if force or student_report.get("status") != "reused":
            for stage in ("3_sparsity", "4_cluster", "5_quantize", "6_benchmark"):
                report["stages"].pop(stage, None)
        checkpoint_report()
    if "sparsity" in stages:
        old_sweep = report["stages"].get("3_sparsity")
        new_sweep = stage_sparsity(config, force=force, seed=seed)
        report["stages"]["3_sparsity"] = new_sweep
        if (
            force
            or not old_sweep
            or old_sweep.get("fingerprint") != new_sweep.get("fingerprint")
        ):
            for stage in ("4_cluster", "5_quantize", "6_benchmark"):
                report["stages"].pop(stage, None)
        checkpoint_report()

    later = [stage for stage in ("cluster", "quantize", "benchmark") if stage in stages]
    if not later:
        logger.info("Pipeline report written to %s", report_path)
        return report

    sweep = report["stages"].get("3_sparsity")
    if not sweep:
        raise ValueError(
            "Stages 4-6 continue from the step-3 frontier; run the sparsity "
            "stage first (or keep its report)"
        )
    validate_sparsity_report(sweep, config, seed=seed)
    candidate = dict(select_deployment_candidate(sweep, config))
    report["selected_candidate"] = candidate
    checkpoint_report()

    device = get_device()
    teacher = FrozenTeacher(config["teacher"]["checkpoint"], device)
    model, input_shape, num_keywords = load_candidate_model(candidate, config)
    candidate_id = _model_artifact_id(model, candidate["save_name"])
    candidate["artifact_id"] = candidate_id
    cluster_recipe = _cluster_recipe_id(config, seed=seed)
    quantize_recipe = _quantize_recipe_id(config, seed=seed)
    checkpoint_report()

    projector: CodebookProjector | None = None
    cluster_changed = False
    if "cluster" in stages:
        cluster_cfg = config["cluster"]
        cluster_checkpoint = Path(
            cluster_cfg.get("checkpoint", "models/exported/kws_clustered.pt")
        )
        existing_cluster = report["stages"].get("4_cluster")
        if existing_cluster and cluster_checkpoint.exists() and not force:
            projector = load_clustered_model(
                model,
                cluster_checkpoint,
                expected_candidate=candidate_id,
                expected_recipe=cluster_recipe,
            )
            cluster_report = dict(existing_cluster)
            cluster_report["status"] = "reused"
        else:
            cluster_changed = True
            cluster_report, projector = stage_cluster(
                model,
                input_shape,
                config,
                teacher,
                device,
                candidate_id,
                cluster_recipe,
                seed=seed,
            )
        report["stages"]["4_cluster"] = cluster_report
        if cluster_changed:
            # A changed stage-4 artifact invalidates both downstream stages,
            # even when this invocation will immediately recompute one of them.
            # Otherwise a later benchmark-only invocation can mistake an old
            # stage-6 report for the output of this cluster run.
            report["stages"].pop("5_quantize", None)
            report["stages"].pop("6_benchmark", None)
        checkpoint_report()
    else:
        # A later invocation may request only quantize/benchmark. If step 4 has
        # already completed, restore its exact baked graph and projector rather
        # than silently continuing from the raw PAI candidate.
        cluster_report = report["stages"].get("4_cluster")
        if cluster_report:
            cluster_checkpoint_value = cluster_report.get("checkpoint")
            cluster_checkpoint = (
                Path(cluster_checkpoint_value) if cluster_checkpoint_value else None
            )
            if cluster_checkpoint is not None and cluster_checkpoint.exists():
                projector = load_clustered_model(
                    model,
                    cluster_checkpoint,
                    expected_candidate=candidate_id,
                    expected_recipe=cluster_recipe,
                )
            elif cluster_report:
                raise FileNotFoundError(
                    "stage 4 is recorded but its clustered checkpoint is missing: "
                    f"{cluster_checkpoint_value!r}"
                )

    if projector is None and ("quantize" in stages or "benchmark" in stages):
        raise ValueError(
            "Stages 5-6 require the exact stage-4 clustered artifact; run the "
            "cluster stage first (or keep its report and checkpoint)"
        )

    source_artifact_id = _model_artifact_id(
        model, f"clustered:{candidate_id}:{cluster_recipe}"
    )

    def validate_quantized_artifact(metadata: dict, checkpoint: Path) -> None:
        expected = {
            "candidate_id": candidate_id,
            "source_artifact_id": source_artifact_id,
            "recipe_id": quantize_recipe,
            "teacher_checkpoint": config["teacher"]["checkpoint"],
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"quantized artifact {checkpoint} provenance mismatch: {mismatches}"
            )
        if projector is not None and len(projector):
            codebook_info = metadata.get("codebook_artifact") or {}
            codebook_path_value = codebook_info.get("path")
            if not codebook_path_value:
                raise ValueError(
                    f"quantized artifact {checkpoint} has no packed codebook "
                    "artifact; rerun the quantize stage"
                )
            codebook_path = Path(codebook_path_value)
            if not codebook_path.exists():
                raise FileNotFoundError(
                    f"packed codebook artifact is missing: {codebook_path}"
                )
            codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)
            if codebook.get("format") != "kws-codebook-v1":
                raise ValueError(
                    f"unsupported packed codebook artifact format at {codebook_path}"
                )
            if codebook.get("graph_path") != str(checkpoint):
                raise ValueError(
                    f"packed codebook artifact {codebook_path} is linked to "
                    f"{codebook.get('graph_path')!r}, not {str(checkpoint)!r}"
                )
            layers = codebook.get("layers") or {}
            if set(layers) != set(projector.assignments):
                raise ValueError(
                    f"packed codebook artifact {codebook_path} does not contain "
                    "the same clustered layers as stage 4"
                )
            for name, indices in projector.assignments.items():
                if int(layers[name].get("index_count", -1)) != indices.numel():
                    raise ValueError(
                        f"packed codebook layer {name!r} has an index-count mismatch"
                    )

    quantize_changed = False
    if "quantize" in stages:
        quantize_checkpoint = Path(config["quantize"]["checkpoint"])
        existing_quantize = report["stages"].get("5_quantize")
        quantize_metadata = quantize_checkpoint.with_suffix(
            quantize_checkpoint.suffix + ".yaml"
        )
        reusable_quantize = (
            existing_quantize
            and quantize_checkpoint.exists()
            and quantize_metadata.exists()
            and not force
        )
        if reusable_quantize:
            with quantize_metadata.open() as file:
                artifact_meta = yaml.safe_load(file) or {}
            validate_quantized_artifact(artifact_meta, quantize_checkpoint)
            # Stage 6 must consume the converted inference graph, not the
            # fake-quantized training wrapper. It is saved as TorchScript so it
            # can be reloaded in a later process without rebuilding QAT state.
            model = torch.jit.load(str(quantize_checkpoint), map_location="cpu").eval()
            quantize_report = dict(existing_quantize)
            quantize_report["torch_cost"] = artifact_meta.get("torch_cost")
            quantize_report["status"] = "reused"
        else:
            quantize_changed = True
            quantize_report = stage_quantize(
                model,
                input_shape,
                num_keywords,
                config,
                projector,
                candidate_id,
                source_artifact_id,
                quantize_recipe,
                seed=seed,
            )
            # Benchmark the converted inference graph produced by QAT. The
            # fake-quantized wrapper is training-only and is deliberately not
            # what step 6 ships or measures.
            model = quantize_report.pop("scripted_model")
            quantize_report.pop("fake_quantized_model", None)
            quantize_report.pop("int8_model", None)
        report["stages"]["5_quantize"] = quantize_report
        if quantize_changed:
            report["stages"].pop("6_benchmark", None)
        checkpoint_report()
    elif "benchmark" in stages:
        # A benchmark-only invocation still consumes the existing final QAT
        # artifact. Benchmarking the float candidate or clustered intermediate
        # would not be framework step 6.
        existing_quantize = report["stages"].get("5_quantize")
        if not existing_quantize or not existing_quantize.get("checkpoint"):
            raise ValueError("stage 6 requires a completed stage-5 QAT artifact")
        quantize_checkpoint = Path(existing_quantize["checkpoint"])
        if not quantize_checkpoint.exists():
            raise FileNotFoundError(
                f"quantized artifact is missing: {quantize_checkpoint}"
            )
        quantize_metadata = quantize_checkpoint.with_suffix(
            quantize_checkpoint.suffix + ".yaml"
        )
        if not quantize_metadata.exists():
            raise FileNotFoundError(
                f"quantized artifact metadata is missing: {quantize_metadata}"
            )
        with quantize_metadata.open() as file:
            artifact_meta = yaml.safe_load(file) or {}
        validate_quantized_artifact(artifact_meta, quantize_checkpoint)
        model = torch.jit.load(str(quantize_checkpoint), map_location="cpu").eval()
        existing_quantize["torch_cost"] = artifact_meta.get("torch_cost")

    if "benchmark" in stages:
        if not isinstance(model, torch.jit.ScriptModule):
            raise TypeError("stage 6 expected the exact scripted int8 inference graph")
        report["stages"]["6_benchmark"] = stage_benchmark(
            model,
            input_shape,
            num_keywords,
            config,
            torch_cost=(report["stages"].get("5_quantize") or {}).get("torch_cost"),
            seed=seed,
        )
        checkpoint_report()

    logger.info("Pipeline report written to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/pipeline.yaml")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a stage even when its output already exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in every stochastic stage (must be non-negative)",
    )
    args = parser.parse_args()

    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")

    run_pipeline(load_yaml(args.config), stages, force=args.force, seed=args.seed)


if __name__ == "__main__":
    main()
