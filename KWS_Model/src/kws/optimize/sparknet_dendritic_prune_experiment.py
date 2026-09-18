"""No-KD SparkNet channel-pruning versus dendritic-capacity experiment.

Every candidate is pruned from the same single source checkpoint. The
conventional arm is the best supervised prune-fine-tune checkpoint created by
``run_cycle`` immediately before PAI wraps the model, so the dendritic arm is
guaranteed to start from those exact weights. The test split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import numbers
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from kws.models.registry import (
    build_model_from_checkpoint,
    checkpoint_input_shape,
    checkpoint_model_family,
    model_family,
)
from kws.models.sparknet import SparkNet
from kws.optimize.dendritic_config import (
    normalize_module_ids,
    placement_module_names,
    project_dendritic_cost,
    validate_max_dendrites,
)
from kws.optimize.prune import prune_sparknet
from kws.utils.artifacts import ArtifactLayout, resolve_resume_dir
from kws.utils.logging import run_session
from kws.utils.profile import count_macs, deployed_parameter_count
from kws.utils.seed import resolve_seed, with_seed

# SparkNet is four TCSBlocks; the placement surface is built from the indices
# rather than spelled out so it cannot drift from the architecture.
SPARKNET_BLOCK_INDICES = range(4)
# The backbone is where the pruned capacity was removed, so it is where a
# dendrite most plausibly buys it back; the earlier selection could only test
# the last block as a whole plus the two heads.  Inside a TCSBlock the two
# convolutions add different things: a ``pointwise`` dendrite adds
# cross-channel mixing capacity, while a ``depthwise`` dendrite adds temporal
# receptive-field capacity.  They are also cheap.  At C12 a pointwise dendrite
# copies a 12x12 1x1 convolution for 144 parameters against ``.fc``'s 396 --
# 2.75x cheaper, so three pointwise dendrites cost less than one classifier
# dendrite and a fixed parameter budget buys far more placements in the
# backbone than at the head.
SUPPORTED_MODULE_IDS = frozenset(
    (".gate_conv", ".fc")
    + tuple(
        f".blocks.{index}{suffix}"
        for index in SPARKNET_BLOCK_INDICES
        for suffix in ("", ".pointwise", ".depthwise")
    )
)
REPORT_NAME = "sparknet_dendritic_prune_experiment.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml(path, label="experiment config")


def _load_yaml(path: str | Path, *, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate invariants that make this a matched, teacher-free study."""
    cfg = copy.deepcopy(dict(config))
    if cfg.get("teacher_checkpoint") is not None:
        raise ValueError("SparkNet experiment is no-KD; teacher_checkpoint must be null")
    if cfg.get("budget") is not None:
        raise ValueError("this run records costs but must not configure a budget gate")
    for key in (
        "source_checkpoint", "data_config", "model_config", "train_config", "output_dir"
    ):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a non-empty path")

    # The study is defined by pruning every candidate from one source width, not
    # by that width being 12; the C16 paper-replication checkpoints are a valid
    # source too.  ``source_channels`` is still cross-checked against the
    # checkpoint's own model_cfg below, so a wrong value here cannot slip past.
    source_width = cfg.get("source_channels")
    if isinstance(source_width, bool) or not isinstance(source_width, int) or source_width < 2:
        raise ValueError("source_channels must be an integer width of at least 2")
    widths = cfg.get("widths")
    pruning = cfg.get("pruning")
    pruning_method = pruning.get("method") if isinstance(pruning, Mapping) else None
    if (
        not isinstance(widths, list)
        or not widths
        or any(isinstance(width, bool) or not isinstance(width, int) for width in widths)
        or widths != sorted(set(widths), reverse=True)
        or any(not 0 < width <= source_width for width in widths)
    ):
        raise ValueError(
            "widths must be unique descending integers between 1 and "
            f"{source_width}"
        )
    if pruning_method == "identity":
        if widths != [source_width]:
            raise ValueError(
                "pruning.method=identity requires widths to contain only source_channels"
            )
    elif pruning_method == "l1_filter":
        if any(width >= source_width for width in widths):
            raise ValueError(
                "pruning.method=l1_filter requires every width to be narrower "
                "than source_channels"
            )
    else:
        raise ValueError("pruning.method must be identity or l1_filter")

    objective = cfg.get("objective")
    if not isinstance(objective, Mapping) or objective.get("metric") != "validation_accuracy":
        raise ValueError("objective.metric must be validation_accuracy")
    if objective.get("use_test") is not False:
        raise ValueError("objective.use_test must be false")
    pai = cfg.get("perforatedai")
    if not isinstance(pai, Mapping) or pai.get("conversion") != "module_ids":
        raise ValueError("perforatedai.conversion must be module_ids")
    # ``module_ids: []`` is a deliberate selection, not a missing one: it
    # selects the standard fine-tuning control and is dispatched before PAI
    # configuration or mode switching. An empty tuple trivially satisfies the
    # support check below.
    module_ids = normalize_module_ids(pai.get("module_ids"), allow_empty=True)
    unsupported = set(module_ids).difference(SUPPORTED_MODULE_IDS)
    if unsupported:
        raise ValueError(
            "PAI module_ids may only select SparkNet blocks (whole, .pointwise, "
            "or .depthwise), .gate_conv, and .fc; "
            f"unsupported: {sorted(unsupported)}"
        )
    # A finite cap is useful for focused follow-up runs; ``-1`` remains the
    # explicit unlimited sentinel used by the original experiment.
    validate_max_dendrites(pai.get("max_dendrites"))
    if pai.get("testing_dendrite_capacity") is not False:
        raise ValueError("testing_dendrite_capacity must be false for the full run")
    if pai.get("max_dendrite_tries") != 3:
        raise ValueError("perforatedai.max_dendrite_tries must be 3")
    if pai.get("switch_mode") != "history" or pai.get("history_lookback") != 8:
        raise ValueError("PAI must use history switching with an 8-epoch lookback")
    return cfg


def dendritic_delta(pai: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, int]:
    """Compute dendrite-only deployment growth against a matched baseline."""
    return {
        "parameter_delta": int(pai["deployed_params"]) - int(baseline["deployed_params"]),
        "mac_delta": int(pai["macs"]) - int(baseline["macs"]),
    }


def report_metrics(
    pai: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    zero_dendrite_accuracy: float | None = None,
    pai_search_accuracy: float | None = None,
) -> dict[str, Any]:
    """Compute accuracy/cost deltas without claiming an unmatched control is causal.

    ``pai`` is the final exported model, after the optional base-only resume.
    Its headline delta is therefore an end-to-end delta from the prune
    fine-tune baseline, not a dendrite-only effect.  The PAI architecture-row
    comparisons are retained separately because the smallest-parameter row in
    ``best_arch_scores.csv`` is a search observation, not an independently
    trained, epoch-matched no-dendrite counterfactual.
    """
    result: dict[str, Any] = dendritic_delta(pai, baseline)
    final_accuracy = float(pai["validation_accuracy"])
    baseline_accuracy = float(baseline["validation_accuracy"])
    result["validation_accuracy_gain_vs_prune_finetune"] = (
        final_accuracy - baseline_accuracy
    )
    result["validation_accuracy_gain"] = result[
        "validation_accuracy_gain_vs_prune_finetune"
    ]
    result["validation_accuracy_gain_basis"] = "prune_finetune_baseline"
    if zero_dendrite_accuracy is not None:
        result["final_validation_accuracy_gain_vs_pai_zero_architecture"] = (
            final_accuracy - float(zero_dendrite_accuracy)
        )
        result["pai_zero_architecture_comparison_basis"] = (
            "minimum_parameter_row_in_best_arch_scores"
        )
    if pai_search_accuracy is not None:
        search_accuracy = float(pai_search_accuracy)
        result["pai_search_accuracy_gain_vs_prune_finetune"] = (
            search_accuracy - baseline_accuracy
        )
        if zero_dendrite_accuracy is not None:
            result["pai_search_accuracy_gain"] = (
                search_accuracy - float(zero_dendrite_accuracy)
            )
            result["pai_search_accuracy_gain_vs_pai_zero_architecture"] = result[
                "pai_search_accuracy_gain"
            ]
            result["pai_search_accuracy_gain_basis"] = (
                "best_architecture_row_minus_minimum_parameter_row"
            )
        else:
            result["pai_search_accuracy_gain"] = result[
                "pai_search_accuracy_gain_vs_prune_finetune"
            ]
            result["pai_search_accuracy_gain_basis"] = "prune_finetune_baseline"
    # Cost deltas stay measured against the pruned baseline: they describe what
    # the deployed graph grew by, which is a cost question, not an accuracy
    # control question.
    result["parameter_growth_fraction"] = (
        result["parameter_delta"] / int(baseline["deployed_params"])
    )
    result["mac_growth_fraction"] = result["mac_delta"] / int(baseline["macs"])
    return result


def interpolate_pruning_curve(
    parameter_count: int, conventional_points: Sequence[Mapping[str, Any]]
) -> float | None:
    """Interpolate validation accuracy at an in-range parameter count only."""
    points = sorted(
        (
            int(point["deployed_params"]),
            float(point["validation_accuracy"]),
        )
        for point in conventional_points
        if point.get("validation_accuracy") is not None
    )
    for params, accuracy in points:
        if params == parameter_count:
            return accuracy
    for (low_params, low_acc), (high_params, high_acc) in zip(points, points[1:]):
        if low_params < parameter_count < high_params:
            fraction = (parameter_count - low_params) / (high_params - low_params)
            return low_acc + fraction * (high_acc - low_acc)
    return None


def _model_cost(model: SparkNet, input_shape: tuple[int, int]) -> dict[str, int]:
    model.eval()
    return {
        "deployed_params": int(deployed_parameter_count(model)),
        "macs": int(count_macs(model, input_shape)),
    }


def _write_report(layout: ArtifactLayout, report: Mapping[str, Any]) -> None:
    layout.atomic_yaml(layout.report_path(REPORT_NAME), dict(report))


def _load_resume_report(layout: ArtifactLayout, report: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load and check a prior aggregate report before it can be replaced."""
    report_path = layout.report_path(REPORT_NAME)
    if not report_path.exists():
        return None
    try:
        saved = _load_yaml(report_path, label="existing aggregate report")
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load existing aggregate report: {report_path}") from exc

    if saved.get("seed") != report.get("seed"):
        raise ValueError(
            "existing aggregate report has a different or unrecorded seed; "
            "start a fresh output directory rather than mixing stochastic runs"
        )
    expected_widths = report["widths"]
    if saved.get("widths") != expected_widths:
        raise ValueError("existing aggregate report has different candidate widths")
    saved_source = saved.get("source")
    expected_source = report["source"]
    if not isinstance(saved_source, Mapping) or any(
        saved_source.get(key) != expected_source[key] for key in ("checkpoint", "channels")
    ):
        raise ValueError("existing aggregate report belongs to a different source checkpoint")
    if saved.get("perforatedai") != report["perforatedai"]:
        raise ValueError("existing aggregate report has different PerforatedAI settings")
    candidates = saved.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("existing aggregate report candidates must be a list")
    expected = set(expected_widths)
    observed: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or isinstance(candidate.get("width"), bool):
            raise ValueError("existing aggregate report contains an invalid candidate")
        width = candidate.get("width")
        if not isinstance(width, int) or width not in expected or width in observed:
            raise ValueError("existing aggregate report has invalid candidate widths")
        observed.add(width)
        if candidate.get("status") == "complete":
            if not all(isinstance(candidate.get(key), Mapping) for key in ("baseline", "dendritic", "comparison")):
                raise ValueError("completed candidate in existing aggregate report is incomplete")
            for section, keys in (
                ("baseline", ("validation_accuracy", "deployed_params", "macs")),
                ("dendritic", ("validation_accuracy", "deployed_params", "macs")),
            ):
                if any(candidate[section].get(key) is None for key in keys):
                    raise ValueError("completed candidate in existing aggregate report is incomplete")
            comparison = candidate["comparison"]
            # These values are emitted by report_metrics and are part of the
            # durable meaning of a completed candidate.  In particular, do
            # not let an empty/corrupt comparison survive merge and later be
            # treated as a valid completed result.
            required_comparison = (
                "validation_accuracy_gain",
                "validation_accuracy_gain_vs_prune_finetune",
                "parameter_delta",
                "mac_delta",
                "parameter_growth_fraction",
                "mac_growth_fraction",
                "pai_search_accuracy_gain",
            )
            if any(
                key not in comparison
                or isinstance(comparison[key], bool)
                or not isinstance(comparison[key], numbers.Real)
                for key in required_comparison
            ):
                raise ValueError("completed candidate in existing aggregate report has an invalid comparison")
            # The basis is a label, not a measurement, so it is checked apart
            # from the numeric tuple above: adding it there would fail the
            # ``numbers.Real`` test on every valid report.
            if comparison.get("validation_accuracy_gain_basis") not in {
                "zero_dendrite",
                "prune_finetune_baseline",
            }:
                raise ValueError("completed candidate in existing aggregate report has an invalid comparison")
    if observed != expected:
        raise ValueError("existing aggregate report is missing expected candidates")
    return saved


def _merge_completed_candidates(report: dict[str, Any], saved: Mapping[str, Any] | None) -> None:
    """Carry durable completed candidate records into a freshly built report."""
    if saved is None:
        return
    completed = {
        candidate["width"]: copy.deepcopy(dict(candidate))
        for candidate in saved["candidates"]
        if candidate.get("status") == "complete"
    }
    report["candidates"] = [
        completed.get(candidate["width"], candidate) for candidate in report["candidates"]
    ]


def _load_inputs(
    cfg: Mapping[str, Any], seed: int | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], SparkNet]:
    data_cfg = _load_yaml(cfg["data_config"], label="data config")
    configured_model = _load_yaml(cfg["model_config"], label="model config")
    train_cfg = with_seed(_load_yaml(cfg["train_config"], label="train config"), seed)
    checkpoint = torch.load(cfg["source_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint_model_family(checkpoint) != "sparknet":
        raise ValueError("source_checkpoint must contain a SparkNet model")
    source_model = build_model_from_checkpoint(checkpoint)
    if not isinstance(source_model, SparkNet):
        raise TypeError("source checkpoint did not build a SparkNet")
    source_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    input_shape = checkpoint_input_shape(checkpoint)
    features = data_cfg.get("features", {})
    if (
        input_shape[0] != 32
        or features.get("type") != "mfcc"
        or int(features.get("n_mels", -1)) != 32
    ):
        raise ValueError("targeted run requires the MFCC-32 source and data frontend")
    source_model_cfg = checkpoint["model_cfg"]
    if (
        model_family(configured_model) != "sparknet"
        or int(source_model_cfg.get("channels", -1)) != int(cfg["source_channels"])
        or int(source_model_cfg.get("gate_channels", -1)) != 32
        or int(configured_model.get("gate_channels", -1)) != 32
    ):
        raise ValueError(
            "targeted run requires a SparkNet source with gate width 32 whose "
            f"checkpoint width matches source_channels={cfg['source_channels']}"
        )
    if train_cfg.get("perforatedai") != cfg.get("perforatedai"):
        raise ValueError("train and experiment PAI settings must match exactly")
    if train_cfg.get("objective") != cfg.get("objective"):
        raise ValueError("train and experiment objectives must match exactly")
    return data_cfg, configured_model, train_cfg, checkpoint, source_model


def run_experiment(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    cycle_runner: Any | None = None,
    control_runner: Any | None = None,
) -> dict[str, Any]:
    """Plan or execute the no-KD prune-versus-dendrite experiment."""
    cfg = validate_config(config)
    module_ids = normalize_module_ids(
        cfg["perforatedai"].get("module_ids"), allow_empty=True
    )
    is_standard_control = not module_ids
    selected_output = output_dir or cfg["output_dir"]
    layout = ArtifactLayout(selected_output).ensure_tree()
    report_path = layout.report_path(REPORT_NAME)
    if report_path.exists() and not resume:
        raise FileExistsError(
            f"aggregate report already exists: {report_path}; pass resume=True to continue it"
        )
    if "seed" in cfg:
        resolve_seed(cfg)
    configured_seed = seed
    if configured_seed is None and "seed" in cfg:
        configured_seed = resolve_seed(cfg)
    data_cfg, configured_model, train_cfg, checkpoint, source_model = _load_inputs(
        cfg, configured_seed
    )
    # Lightweight callers may omit a train-config seed; the shared seed
    # resolver supplies the documented deterministic default in that case.
    effective_seed = resolve_seed(train_cfg)
    source_path = str(cfg["source_checkpoint"])
    input_shape = checkpoint_input_shape(checkpoint)
    source_cost = _model_cost(source_model, input_shape)
    source_accuracy = checkpoint.get("val_acc", cfg.get("source_validation_accuracy"))
    if source_accuracy is None:
        raise ValueError("source checkpoint/config must record validation accuracy")

    report: dict[str, Any] = {
        "status": "planned" if dry_run else "running",
        "dry_run": dry_run,
        "resume": resume,
        "seed": effective_seed,
        "selection_split": "validation",
        "test_split_used": False,
        "knowledge_distillation": {"enabled": False, "teacher_checkpoint": None},
        "budget": None,
        "budget_enforced": False,
        "source": {
            "checkpoint": source_path,
            "channels": int(cfg["source_channels"]),
            "gate_channels": 32,
            "validation_accuracy": float(source_accuracy),
            **source_cost,
        },
        "widths": list(cfg["widths"]),
        "pruning": dict(cfg["pruning"]),
        "perforatedai": dict(cfg["perforatedai"]),
        "candidates": [],
    }

    plans: list[dict[str, Any]] = []
    for width in cfg["widths"]:
        identity_width = int(width) == int(cfg["source_channels"])
        pruned = (
            copy.deepcopy(source_model)
            if identity_width
            else prune_sparknet(source_model, int(width))
        ).eval()
        baseline_cost = _model_cost(pruned, input_shape)
        target_model_cfg = dict(configured_model)
        target_model_cfg["name"] = f"sparknet_c{width}"
        target_model_cfg["channels"] = int(width)
        cycle_train_cfg = copy.deepcopy(train_cfg)
        cycle_train_cfg["pruning"] = {
            "kind": "structured",
            "method": "identity" if identity_width else "l1_filter",
            "keep_ratio": float(width) / int(cfg["source_channels"]),
            "target_channels": int(width),
        }
        placement_module_names(pruned, cycle_train_cfg["perforatedai"])
        projection = project_dendritic_cost(
            pruned,
            input_shape,
            cycle_train_cfg["perforatedai"],
            max_dendrites=1,
        ).as_dict()
        candidate_name = f"sparknet_c{width}_multilayer"
        baseline_name = f"sparknet_c{width}"
        record = {
            "width": int(width),
            "prune_fraction": 1.0 - float(width) / int(cfg["source_channels"]),
            "status": "planned",
            "baseline": {
                "training": (
                    "supervised_no_kd_identity_finetune"
                    if identity_width
                    else "supervised_no_kd"
                ),
                "validation_accuracy": None,
                **baseline_cost,
                "checkpoint": layout.relative(
                    layout.checkpoint_path(
                        "sparsity",
                        "prune_supervised",
                        "best",
                        candidate=baseline_name,
                    )
                ),
            },
            "dendritic": {
                "training": (
                    "standard_finetune_control"
                    if is_standard_control
                    else "pai_no_kd"
                ),
                "validation_accuracy": None,
                "pai_search_validation_accuracy": None,
                # PAI's minimum-parameter architecture row, filled in from the
                # cycle result for an explicitly labeled within-search delta.
                "zero_dendrite_validation_accuracy": None,
                "zero_dendrite_params": None,
                "deployed_params": None,
                "macs": None,
                "one_dendrite_cost_projection": projection,
                "checkpoint": (
                    layout.relative(
                        layout.checkpoint_path(
                            "sparsity",
                            "standard_control",
                            "best",
                            candidate=candidate_name,
                        )
                    )
                    if is_standard_control
                    else layout.relative(
                        layout.pai_candidate_path(candidate_name)
                        / "final_clean_pai.pt"
                    )
                ),
            },
            "comparison": None,
        }
        report["candidates"].append(record)
        plans.append(
            {
                "record": record,
                "candidate_name": candidate_name,
                "baseline_name": baseline_name,
                "model_cfg": target_model_cfg,
                "train_cfg": cycle_train_cfg,
            }
        )

    saved_report = _load_resume_report(layout, report) if resume else None
    _merge_completed_candidates(report, saved_report)
    if dry_run:
        return report

    # This is deliberately after recovery/merge: a failure while loading a
    # resume report must never destroy the only record of completed widths.
    _write_report(layout, report)

    # Importing this module initializes the optional compiled PAI runtime, so
    # it remains below the dry-run return. The empty-placement arm has its own
    # conventional lifecycle and must never enter PerforatedAI mode switching.
    if is_standard_control:
        if control_runner is None:
            from kws.optimize.dendritic import run_standard_control

            control_runner = run_standard_control
        selected_runner = control_runner
    else:
        if cycle_runner is None:
            from kws.optimize.dendritic import run_cycle

            cycle_runner = run_cycle
        selected_runner = cycle_runner

    run_id = layout.manifest_run_id
    completed_widths = {
        candidate["width"]
        for candidate in report["candidates"]
        if candidate.get("status") == "complete"
    }
    for plan in plans:
        if plan["record"]["width"] in completed_widths:
            continue
        cycle = selected_runner(
            source_path,
            data_cfg,
            plan["model_cfg"],
            plan["train_cfg"],
            plan["candidate_name"],
            teacher_checkpoint=None,
            seed=effective_seed,
            output_dir=layout.root,
            # Aggregate recovery is intentionally tri-state: ``None`` lets
            # run_cycle reuse a canonical PAI sidecar when one exists, while
            # an empty candidate directory starts PAI cleanly.  Standalone
            # run_cycle callers retain strict True/False semantics.
            resume=None if resume else False,
            prune_finetune_candidate=plan["baseline_name"],
            run_id=run_id,
        )
        result = asdict(cycle) if is_dataclass(cycle) else dict(cycle)
        record = plan["record"]
        prune_finetune = result.get("prune_finetune") or {}
        baseline_accuracy = prune_finetune.get("best_val_acc")
        if baseline_accuracy is None:
            raise RuntimeError("PAI cycle did not report its exact prune-finetune baseline")
        record["baseline"].update(
            {
                "validation_accuracy": float(baseline_accuracy),
                "checkpoint": str(prune_finetune.get("checkpoint")),
                "reused": bool(prune_finetune.get("reused", False)),
                "distillation": prune_finetune.get("distillation"),
            }
        )
        cost = result.get("cost") or {}
        pai = {
            "validation_accuracy": float(result["best_val_acc"]),
            "deployed_params": int(result["deployed_params"]),
            "macs": int(cost["macs"]),
        }
        selected_checkpoint = result.get("checkpoint") or record["dendritic"][
            "checkpoint"
        ]
        if (result.get("resume") or {}).get("status") == "complete":
            selected_checkpoint = result["resume"]["checkpoint"]
        # A cycle that predates the control (or a stubbed runner) reports
        # ``None`` here, which keeps the older prune-fine-tune basis below.
        zero_dendrite_accuracy = result.get("zero_dendrite_val_acc")
        zero_dendrite_params = result.get("zero_dendrite_params")
        record["dendritic"].update(
            {
                **pai,
                "pai_search_validation_accuracy": (
                    None
                    if is_standard_control
                    else float(
                        result.get("pai_best_val_acc") or result["best_val_acc"]
                    )
                ),
                "zero_dendrite_validation_accuracy": (
                    float(zero_dendrite_accuracy)
                    if zero_dendrite_accuracy is not None
                    else None
                ),
                "zero_dendrite_params": (
                    int(zero_dendrite_params)
                    if zero_dendrite_params is not None
                    else None
                ),
                "checkpoint": selected_checkpoint,
                "resume": result.get("resume"),
                "pai_deployed_params": result.get("pai_deployed_params"),
                "full_cost": cost,
            }
        )
        baseline = record["baseline"]
        control_accuracy = record["dendritic"]["zero_dendrite_validation_accuracy"]
        search_accuracy = record["dendritic"]["pai_search_validation_accuracy"]
        comparison = report_metrics(
            pai,
            baseline,
            zero_dendrite_accuracy=control_accuracy,
            pai_search_accuracy=search_accuracy,
        )
        record["comparison"] = comparison
        record["status"] = "complete"
        _write_report(layout, report)

    conventional_points = [report["source"]] + [
        record["baseline"] for record in report["candidates"]
    ]
    for record in report["candidates"]:
        dendritic = record["dendritic"]
        curve_accuracy = interpolate_pruning_curve(
            int(dendritic["deployed_params"]), conventional_points
        )
        comparison = record["comparison"]
        comparison["pruning_curve_validation_accuracy_at_parameter_count"] = curve_accuracy
        comparison["validation_accuracy_above_pruning_curve"] = (
            float(dendritic["validation_accuracy"]) - curve_accuracy
            if curve_accuracy is not None else None
        )
        comparison["pruning_curve_comparison_status"] = (
            "interpolated" if curve_accuracy is not None else "outside_measured_range"
        )

    report["status"] = "complete"
    _write_report(layout, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/sparknet_c12_dendritic_prune_no_kd.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each width from its canonical PAI/KWS checkpoint pair",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-KD", "--no-kd", dest="no_kd", action="store_true",
        help="Force teacher_checkpoint=null (this experiment is no-KD by design)",
    )
    args = parser.parse_args()
    try:
        args.output_dir = resolve_resume_dir(args.output_dir, args.resume_dir)
    except ValueError as error:
        parser.error(str(error))

    config = load_config(args.config)
    if args.no_kd:
        config["teacher_checkpoint"] = None
    selected_output = args.output_dir or config.get("output_dir")
    if selected_output is None:
        parser.error("output_dir is required in config or with --output-dir")
    inputs = [
        (args.config, "sparknet_dendritic_prune_config"),
        (config["data_config"], "data_config"),
        (config["model_config"], "model_config"),
        (config["train_config"], "train_config"),
        (config["source_checkpoint"], "source_checkpoint"),
    ]
    session_seed = args.seed
    if session_seed is None and "seed" in config:
        session_seed = resolve_seed(config)
    if session_seed is None:
        session_seed = resolve_seed(
            _load_yaml(config["train_config"], label="train config")
        )
    with run_session(
        selected_output,
        command="kws.optimize.sparknet_dendritic_prune_experiment",
        argv=sys.argv,
        seed=session_seed,
        inputs=inputs,
    ):
        report = run_experiment(
            config,
            dry_run=args.dry_run,
            resume=args.resume or args.resume_dir is not None,
            output_dir=selected_output,
            seed=args.seed,
        )
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
