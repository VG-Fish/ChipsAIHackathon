"""Budget-matched conventional versus PerforatedAI compression experiment.

The runner trains structured DS-CNN backbones once, reuses those exact
checkpoints for selective dendrite placements, and reports only validation
accuracy and measured deployment costs.  Importing this module never imports
PerforatedAI; ``--dry-run`` is consequently useful on any development machine.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from kws.models.ds_cnn import DSCNN, build_ds_cnn
from kws.models.registry import (
    build_model_from_checkpoint,
    checkpoint_input_shape,
    checkpoint_model_family,
)
from kws.optimize.dendritic_config import (
    UNLIMITED_DENDRITES,
    pai_module_ids,
    placement_module_names,
    projected_dendrite_count,
    project_dendritic_cost,
    validate_max_dendrites,
)
from kws.optimize.pareto import ParetoPoint, pareto_front
from kws.optimize.prune import SparsitySpec, prune_and_fine_tune, prune_ds_cnn
from kws.utils.artifacts import ArtifactLayout, resolve_resume_dir
from kws.utils.logging import get_logger, run_session
from kws.utils.profile import profile_model
from kws.utils.seed import with_seed

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompressionBudget:
    """Hard deployment ceilings shared by both experiment arms."""

    max_deployed_params: int | None
    max_macs: int | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CompressionBudget":
        params = config.get("max_deployed_params")
        macs = config.get("max_macs")
        if params is None and macs is None:
            raise ValueError(
                "budget must configure max_deployed_params, max_macs, or both"
            )
        if params is not None and int(params) < 1:
            raise ValueError("budget.max_deployed_params must be positive")
        if macs is not None and int(macs) < 1:
            raise ValueError("budget.max_macs must be positive")
        return cls(
            max_deployed_params=int(params) if params is not None else None,
            max_macs=int(macs) if macs is not None else None,
        )

    def violations(self, *, params: int, macs: int) -> list[str]:
        failures: list[str] = []
        if self.max_deployed_params is not None and params > self.max_deployed_params:
            failures.append(
                f"params {params} exceed budget {self.max_deployed_params}"
            )
        if self.max_macs is not None and macs > self.max_macs:
            failures.append(f"MACs {macs} exceed budget {self.max_macs}")
        return failures

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _safe_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if (
        Path(value).name != value
        or value in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
    ):
        raise ValueError(f"{label} must be a single path-safe name: {value!r}")
    return value


def resolve_keep_ratios(
    source_channels: list[int], config: Mapping[str, Any]
) -> list[float]:
    """Resolve explicit widths, width multipliers, or a descending sweep."""
    modes = [
        key
        for key in ("widths", "width_multipliers", "sweep")
        if config.get(key) is not None
    ]
    if len(modes) != 1:
        raise ValueError(
            "backbones must configure exactly one of widths, width_multipliers, or sweep"
        )

    mode = modes[0]
    if mode == "width_multipliers":
        raw_ratios = config[mode]
        if not isinstance(raw_ratios, list) or not raw_ratios:
            raise ValueError("backbones.width_multipliers must be a non-empty list")
        ratios = [float(value) for value in raw_ratios]
    else:
        if len(set(source_channels)) != 1:
            raise ValueError(
                f"absolute width candidates require equal source widths; got {source_channels}"
            )
        source_width = source_channels[0]
        if mode == "widths":
            widths = config[mode]
            if not isinstance(widths, list) or not widths:
                raise ValueError("backbones.widths must be a non-empty list")
            resolved_widths = [int(value) for value in widths]
        else:
            sweep = config[mode]
            if not isinstance(sweep, Mapping):
                raise ValueError("backbones.sweep must be a mapping")
            start = int(sweep.get("start", source_width))
            minimum = int(sweep["minimum"])
            step = int(sweep.get("step", 1))
            if start < minimum or minimum < 1 or step < 1:
                raise ValueError(
                    "backbone sweep requires start >= minimum >= 1 and step >= 1"
                )
            resolved_widths = list(range(start, minimum - 1, -step))
            if resolved_widths[-1] != minimum:
                resolved_widths.append(minimum)
        if any(width < 1 or width > source_width for width in resolved_widths):
            raise ValueError(
                f"candidate widths must be in [1, {source_width}]: {resolved_widths}"
            )
        ratios = [width / source_width for width in resolved_widths]

    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("all backbone width multipliers must be in (0, 1]")
    unique: list[float] = []
    seen_shapes: set[tuple[int, ...]] = set()
    for ratio in ratios:
        shape = tuple(max(1, int(round(width * ratio))) for width in source_channels)
        if shape not in seen_shapes:
            unique.append(ratio)
            seen_shapes.add(shape)
    return unique


def _candidate_label(block_channels: list[int], index: int) -> str:
    if len(set(block_channels)) == 1:
        return f"w{block_channels[0]}"
    return f"candidate_{index:02d}_" + "x".join(str(value) for value in block_channels)


def _cost_dict(cost: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(cost)
    if "params" in result:
        result["deployed_params"] = int(result["params"])
    return result


def _profile_checkpoint(
    checkpoint_path: str | Path,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_ds_cnn(
        checkpoint["model_cfg"],
        checkpoint_input_shape(checkpoint),
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    profile_device = torch.device(str(deployment.get("profile_device", "cpu")))
    cost = profile_model(
        model.to(profile_device).eval(),
        checkpoint_input_shape(checkpoint),
        device=profile_device,
        bits_per_weight=int(deployment.get("bits_per_weight", 32)),
        bytes_per_activation=int(deployment.get("bytes_per_activation", 1)),
        latency_iterations=int(deployment.get("latency_iterations", 10)),
        latency_warmup=int(deployment.get("latency_warmup", 3)),
    )
    return _cost_dict(cost.as_dict())


def _placement_config(
    placement: Mapping[str, Any], train_cfg: Mapping[str, Any], max_dendrites: int
) -> dict[str, Any]:
    pai_cfg = dict(train_cfg.get("perforatedai") or {})
    conversion = str(placement.get("conversion", "fc_only"))
    pai_cfg["conversion"] = conversion
    pai_cfg["max_dendrites"] = max_dendrites
    if conversion == "module_ids":
        pai_cfg["module_ids_to_perforate"] = list(
            pai_module_ids(
                {
                    "conversion": conversion,
                    "module_ids": placement.get("module_ids"),
                }
            )
        )
    else:
        pai_cfg.pop("module_ids_to_perforate", None)
        pai_cfg.pop("module_ids", None)
    return pai_cfg


def _write_report(layout: ArtifactLayout, report: dict[str, Any]) -> None:
    layout.atomic_yaml(layout.report_path("compression_experiment.yaml"), report)


def _completed_pai_result(
    run_dir: Path, expected_fingerprint: str
) -> dict[str, Any] | None:
    metadata_path = run_dir / "cycle_metadata.yaml"
    clean_path = run_dir / "final_clean_pai.pt"
    if not metadata_path.is_file() or not clean_path.is_file():
        return None
    metadata = _load_yaml(metadata_path)
    if metadata.get("status") != "complete":
        return None
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"completed PAI candidate {run_dir.name!r} has a different recipe; "
            "choose a new output directory"
        )
    result = metadata.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"completed PAI candidate has no result: {metadata_path}")
    return result


def _finalize_comparisons(
    records: list[dict[str, Any]], cost_keys: tuple[str, ...]
) -> dict[str, Any]:
    completed = [
        record
        for record in records
        if record.get("status") == "complete"
        and record.get("budget_admitted")
        and record.get("accuracy") is not None
    ]
    if not completed:
        return {"status": "not_computed", "reason": "no completed candidates"}

    points = [
        ParetoPoint(
            label=record["label"],
            accuracy=float(record["accuracy"]),
            costs={key: float(record["costs"][key]) for key in cost_keys},
            detail={"arm": record["arm"]},
        )
        for record in completed
    ]
    frontier = [point.as_dict() for point in pareto_front(points, cost_keys)]
    conventional = [record for record in completed if record["arm"] == "conventional"]
    dendritic = [record for record in completed if record["arm"] == "dendritic"]

    def best(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        return max(values, key=lambda item: float(item["accuracy"])) if values else None

    matched: list[dict[str, Any]] = []
    for candidate in dendritic:
        eligible = [
            baseline
            for baseline in conventional
            if all(
                float(baseline["costs"][key]) <= float(candidate["costs"][key])
                for key in cost_keys
            )
        ]
        baseline = best(eligible)
        matched.append(
            {
                "dendritic": candidate["label"],
                "conventional": baseline["label"] if baseline else None,
                "accuracy_delta": (
                    float(candidate["accuracy"]) - float(baseline["accuracy"])
                    if baseline
                    else None
                ),
                "rule": "conventional costs do not exceed dendritic final costs",
            }
        )

    best_conventional = best(conventional)
    best_dendritic = best(dendritic)
    return {
        "status": "complete",
        "cost_keys": list(cost_keys),
        "frontier": frontier,
        "best_conventional_under_fixed_budget": (
            best_conventional["label"] if best_conventional else None
        ),
        "best_dendritic_under_fixed_budget": (
            best_dendritic["label"] if best_dendritic else None
        ),
        "best_accuracy_delta": (
            float(best_dendritic["accuracy"])
            - float(best_conventional["accuracy"])
            if best_conventional and best_dendritic
            else None
        ),
        "matched_budget_comparisons": matched,
    }


def run_compression_experiment(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Plan or execute the two-arm compression experiment."""
    source_path = str(config["source_checkpoint"])
    teacher_path = config.get("teacher_checkpoint")
    data_cfg = _load_yaml(config["data_config"])
    train_cfg = with_seed(_load_yaml(config["train_config"]), seed)
    selected_output = output_dir or config.get("output_dir")
    if selected_output is None:
        raise ValueError("output_dir is required in config or as a CLI override")
    layout = ArtifactLayout(selected_output).ensure_tree()
    run_id = layout.manifest_run_id
    budget = CompressionBudget.from_config(config["budget"])
    deployment = dict(config.get("deployment") or train_cfg.get("deployment") or {})
    pai_experiment = dict(config.get("perforatedai") or {})
    pai_enabled = bool(pai_experiment.get("enabled", True))
    on_unavailable = str(pai_experiment.get("on_unavailable", "error"))
    if on_unavailable not in {"error", "skip"}:
        raise ValueError("perforatedai.on_unavailable must be 'error' or 'skip'")
    max_dendrites = validate_max_dendrites(
        pai_experiment.get(
            "max_dendrites",
            (train_cfg.get("perforatedai") or {}).get("max_dendrites", 1),
        )
    )
    projection_dendrites = projected_dendrite_count(max_dendrites)
    unlimited_dendrites = max_dendrites == UNLIMITED_DENDRITES

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if checkpoint_model_family(source) != "ds_cnn":
        raise ValueError("compression_experiment currently supports DS-CNN checkpoints")
    source_model = build_model_from_checkpoint(source)
    if not isinstance(source_model, DSCNN):
        raise TypeError("source checkpoint did not build a DS-CNN")
    source_model.load_state_dict(source["model_state_dict"])
    source_channels = [block.pointwise.out_channels for block in source_model.blocks]
    ratios = resolve_keep_ratios(source_channels, config["backbones"])
    placements = config.get("placements") or []
    if pai_enabled and not isinstance(placements, list):
        raise ValueError("placements must be a list")
    if pai_enabled and not placements:
        raise ValueError("at least one placement is required when PAI is enabled")
    placement_names: set[str] = set()
    for placement in placements if pai_enabled else []:
        if not isinstance(placement, Mapping):
            raise ValueError("each placement must be a mapping")
        placement_name = _safe_name(placement.get("name"), label="placement.name")
        if placement_name in placement_names:
            raise ValueError(f"duplicate placement name: {placement_name!r}")
        placement_names.add(placement_name)
    cost_keys = tuple((config.get("pareto") or {}).get(
        "cost_keys", ["deployed_params", "macs"]
    ))
    if not cost_keys or any(
        key not in {"deployed_params", "macs"} for key in cost_keys
    ):
        raise ValueError("pareto.cost_keys may contain deployed_params and/or macs")

    report: dict[str, Any] = {
        "status": "planned" if dry_run else "running",
        "dry_run": dry_run,
        "selection_split": "validation",
        "test_split_used": False,
        "source_checkpoint": source_path,
        "teacher_checkpoint": teacher_path,
        "budget": budget.as_dict(),
        "max_dendrites": max_dendrites,
        "candidates": [],
        "pareto": {"status": "not_computed", "reason": "dry run"},
    }

    plans: list[dict[str, Any]] = []
    input_shape = checkpoint_input_shape(source)
    for index, ratio in enumerate(ratios, start=1):
        base = prune_ds_cnn(source_model, ratio).eval()
        block_channels = [block.pointwise.out_channels for block in base.blocks]
        label = _candidate_label(block_channels, index)
        spec = SparsitySpec("structured", keep_ratio=ratio)
        model_cfg = dict(source["model_cfg"])
        model_cfg["name"] = f'{source["model_cfg"]["name"]}_{spec.label}'
        model_cfg["block_channels"] = block_channels
        base_projection = {
            "deployed_params": sum(parameter.numel() for parameter in base.parameters()),
            "macs": project_dendritic_cost(
                base, input_shape, {"conversion": "fc_only"}, max_dendrites=1
            ).base_macs,
        }
        base_violations = budget.violations(
            params=base_projection["deployed_params"], macs=base_projection["macs"]
        )
        conventional = {
            "label": f"conventional/{label}",
            "arm": "conventional",
            "backbone": label,
            "keep_ratio": ratio,
            "block_channels": block_channels,
            "status": "planned" if not base_violations else "skipped_budget",
            "budget_admitted": not base_violations,
            "budget_violations": base_violations,
            "accuracy": None,
            "costs": base_projection,
            "checkpoint": layout.relative(
                layout.checkpoint_path(
                    "sparsity", "prune_kd", "best", candidate=label
                )
            ),
        }
        report["candidates"].append(conventional)
        dendritic_records: list[dict[str, Any]] = []
        for placement_value in placements if pai_enabled else []:
            assert isinstance(placement_value, Mapping)
            placement = dict(placement_value)
            placement_name = _safe_name(placement.get("name"), label="placement.name")
            pai_cfg = _placement_config(placement, train_cfg, max_dendrites)
            # Resolve against every width because a path can exist in one
            # topology and not another.
            placement_module_names(base, pai_cfg)
            projection = project_dendritic_cost(
                base, input_shape, pai_cfg, max_dendrites=projection_dendrites
            )
            projection_details = projection.as_dict()
            projection_details["basis"] = (
                "one_dendrite_lower_bound"
                if unlimited_dendrites
                else "configured_maximum"
            )
            projection_details["configured_max_dendrites"] = max_dendrites
            violations = budget.violations(
                params=projection.projected_params, macs=projection.projected_macs
            )
            candidate_name = f"{label}_{placement_name}"
            record = {
                "label": f"dendritic/{candidate_name}",
                "arm": "dendritic",
                "backbone": label,
                "placement": placement_name,
                "perforatedai": pai_cfg,
                "status": "planned" if not violations else "skipped_projected_budget",
                "budget_admitted": not violations,
                "budget_violations": violations,
                "budget_admission_basis": projection_details["basis"],
                "accuracy": None,
                "cost_projection": projection_details,
                "costs_source": projection_details["basis"],
                "costs": {
                    "deployed_params": projection.projected_params,
                    "macs": projection.projected_macs,
                },
                "cycle_checkpoint": layout.relative(
                    layout.checkpoint_path(
                        "sparsity", "pai", "latest", candidate=candidate_name
                    )
                ),
                "cycle_checkpoints": [],
                "final_checkpoint": layout.relative(
                    layout.pai_candidate_path(candidate_name) / "final_clean_pai.pt"
                ),
            }
            report["candidates"].append(record)
            dendritic_records.append(record)
        plans.append(
            {
                "label": label,
                "ratio": ratio,
                "spec": spec,
                "model_cfg": model_cfg,
                "conventional": conventional,
                "dendritic": dendritic_records,
            }
        )

    _write_report(layout, report)
    if dry_run:
        return report

    if pai_enabled and any(plan["dendritic"] for plan in plans) and not teacher_path:
        raise ValueError(
            "teacher_checkpoint is required when PAI is enabled so every arm "
            "reuses the same trained conventional base"
        )

    for plan in plans:
        conventional = plan["conventional"]
        if conventional["budget_admitted"]:
            train_candidate_cfg = dict(train_cfg)
            train_candidate_cfg["pruning"] = {
                "kind": "structured",
                "keep_ratio": plan["ratio"],
            }
            latest = layout.checkpoint_path(
                "sparsity", "prune_kd", "latest", candidate=plan["label"]
            )
            outcome = prune_and_fine_tune(
                source_path,
                data_cfg,
                train_candidate_cfg,
                plan["spec"],
                layout.checkpoint_path(
                    "sparsity", "prune_kd", "best", candidate=plan["label"]
                ),
                teacher_checkpoint=(
                    str(teacher_path) if teacher_path is not None else None
                ),
                seed=seed,
                output_dir=layout.root,
                resume=latest.is_file(),
                artifact_candidate=plan["label"],
                run_id=run_id,
            )
            conventional["status"] = "complete"
            conventional["accuracy"] = float(outcome["best_val_acc"])
            conventional["costs"] = _profile_checkpoint(
                outcome["checkpoint"], deployment
            )
            conventional["checkpoint"] = layout.relative(outcome["checkpoint"])
            actual_violations = budget.violations(
                params=int(conventional["costs"]["deployed_params"]),
                macs=int(conventional["costs"]["macs"]),
            )
            conventional["budget_violations"] = actual_violations
            conventional["budget_admitted"] = not actual_violations
            if actual_violations:
                conventional["status"] = "rejected_actual_budget"
            _write_report(layout, report)

    eligible_dendritic = [
        record
        for plan in plans
        for record in plan["dendritic"]
        if record["budget_admitted"]
    ]
    dendritic_module = None
    pai_error = None
    if eligible_dendritic:
        try:
            dendritic_module = importlib.import_module("kws.optimize.dendritic")
        except (ImportError, RuntimeError) as exc:
            pai_error = f"{type(exc).__name__}: {exc}"
            if on_unavailable == "error":
                raise RuntimeError(
                    "PerforatedAI is unavailable; rerun with "
                    "perforatedai.on_unavailable=skip or use --dry-run"
                ) from exc
            for record in eligible_dendritic:
                record["status"] = "skipped_pai_unavailable"
                record["budget_admitted"] = False
                record["skip_reason"] = pai_error
            report["perforatedai_unavailable"] = pai_error
            _write_report(layout, report)

    if dendritic_module is not None:
        for plan in plans:
            if plan["conventional"]["status"] != "complete":
                for record in plan["dendritic"]:
                    if record["budget_admitted"]:
                        record["status"] = "skipped_no_conventional_base"
                        record["budget_admitted"] = False
                continue
            for record in plan["dendritic"]:
                if not record["budget_admitted"]:
                    continue
                candidate_name = record["label"].split("/", 1)[1]
                cycle_train_cfg = dict(train_cfg)
                cycle_train_cfg["pruning"] = {
                    "kind": "structured",
                    "keep_ratio": plan["ratio"],
                }
                cycle_train_cfg["perforatedai"] = dict(record["perforatedai"])
                fingerprint = dendritic_module.cycle_fingerprint(
                    source_path,
                    str(teacher_path),
                    data_cfg,
                    plan["model_cfg"],
                    cycle_train_cfg,
                )
                run_dir = layout.pai_candidate_path(candidate_name)
                result = _completed_pai_result(run_dir, fingerprint)
                if result is None:
                    cycle = dendritic_module.run_cycle(
                        source_path,
                        data_cfg,
                        plan["model_cfg"],
                        cycle_train_cfg,
                        candidate_name,
                        teacher_checkpoint=str(teacher_path),
                        seed=seed,
                        output_dir=layout.root,
                        prune_finetune_candidate=plan["label"],
                        run_id=run_id,
                    )
                    result = asdict(cycle)
                record["accuracy"] = float(result["best_val_acc"])
                record["costs"] = _cost_dict(result.get("cost") or {})
                record["costs_source"] = "measured_clean_export"
                record["cycle_checkpoints"] = []
                for checkpoint_value in result.get("cycle_checkpoints") or []:
                    checkpoint_path = Path(checkpoint_value)
                    if checkpoint_path.is_absolute():
                        try:
                            checkpoint_value = layout.relative(checkpoint_path)
                        except ValueError:
                            checkpoint_value = str(checkpoint_path)
                    record["cycle_checkpoints"].append(str(checkpoint_value))
                actual_violations = budget.violations(
                    params=int(record["costs"]["deployed_params"]),
                    macs=int(record["costs"]["macs"]),
                )
                record["budget_violations"] = actual_violations
                record["budget_admitted"] = not actual_violations
                record["budget_admission_basis"] = "measured_clean_export"
                record["status"] = (
                    "complete" if not actual_violations else "rejected_actual_budget"
                )
                _write_report(layout, report)

    report["pareto"] = _finalize_comparisons(report["candidates"], cost_keys)
    report["status"] = "complete"
    _write_report(layout, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/train/compression_experiment.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        args.output_dir = resolve_resume_dir(args.output_dir, args.resume_dir)
    except ValueError as error:
        parser.error(str(error))
    config = _load_yaml(args.config)
    selected_output = args.output_dir or config.get("output_dir")
    if selected_output is None:
        parser.error("output_dir is required in config or with --output-dir")
    inputs = [
        (args.config, "compression_experiment_config"),
        (config["data_config"], "data_config"),
        (config["train_config"], "train_config"),
        (config["source_checkpoint"], "source_checkpoint"),
    ]
    if config.get("teacher_checkpoint"):
        inputs.append((config["teacher_checkpoint"], "teacher_checkpoint"))
    with run_session(
        selected_output,
        command="kws.optimize.compression_experiment",
        argv=sys.argv,
        seed=args.seed,
        inputs=inputs,
    ):
        run_compression_experiment(
            config,
            dry_run=args.dry_run,
            output_dir=selected_output,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
