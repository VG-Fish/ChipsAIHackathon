"""Framework step 3: sweep sparsity targets, stopping on the Pareto frontier.

PerforatedAI adds capacity; it does not structurally prune a learned dendritic
network. So each sparsity target gets its own base network, pruned from the
same KD-trained student checkpoint, and its own complete pass through steps
3a-3e: prune, KD fine-tune, perforate, resume with KD, and measure.

Step 3f is what distinguishes this from the earlier search. The old rule --
stop at the first candidate below the accuracy floor -- ends the sweep at a
point that may still have had cheaper, still-deployable models behind it. The
Pareto rule keeps going while candidates continue to extend the
accuracy-versus-cost frontier, and only stops once ``pareto_patience``
consecutive candidates have failed to extend it. The accuracy floor stays on as
an admission constraint: a model below it is not deployable, whatever it costs.

Only validation accuracy controls acceptance. The test split is never loaded.
"""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from kws.optimize.dendritic import (
    DendriticCycleResult,
    FRAMEWORK_CYCLE_VERSION,
    cycle_fingerprint,
    load_yaml,
    read_pai_architecture_results,
    run_cycle,
)
from kws.optimize.kd import file_sha256
from kws.optimize.pareto import ParetoFrontier, ParetoPoint
from kws.train import resolve_manifest_run_id, validate_checkpoint_run_id
from kws.utils import graphs
from kws.utils.logging import get_logger
from kws.utils.logging import run_session
from kws.utils.artifacts import ArtifactLayout
from kws.utils.seed import with_seed

logger = get_logger(__name__)

# Costs the frontier can trade accuracy against, if the candidate recorded them.
# `deployed_params` is always present; the rest come from step 3e's profiling,
# so a run completed before that instrumentation existed contributes only the
# parameter count and the sweep compares on that axis alone.
CANDIDATE_COST_KEYS = (
    "deployed_params",
    "macs",
    "latency_ms_p50",
    "weight_bytes",
    "activation_peak_bytes",
)


def search_fingerprint(
    checkpoint_path: str,
    teacher_checkpoint: str,
    data_cfg: dict,
    train_cfg: dict,
    search_cfg: dict,
) -> str:
    """Stable identity for every input that can change a sweep result."""
    operational_keys = {
        "reuse_completed_start_run",
        "reuse_completed_candidates",
        "save_prefix",
        "summary_path",
    }
    search_recipe = {
        key: value for key, value in search_cfg.items() if key not in operational_keys
    }
    payload = {
        "framework_cycle_version": FRAMEWORK_CYCLE_VERSION,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha256": file_sha256(teacher_checkpoint),
        "data": data_cfg,
        "train": train_cfg,
        "search": search_recipe,
    }
    encoded = yaml.safe_dump(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SearchDecision:
    """Whether a completed candidate is admissible at all."""

    accepted: bool
    reason: str
    cause: str = "admitted"


def candidate_widths(start: int, minimum: int, step: int) -> list[int]:
    """Generate descending widths, including ``start`` and ``minimum``."""
    if start < 1 or minimum < 1:
        raise ValueError("channel widths must be positive")
    if step < 1:
        raise ValueError("channel_step must be positive")
    if start < minimum:
        raise ValueError("start_channels must be >= minimum_channels")
    widths = list(range(start, minimum - 1, -step))
    if widths[-1] != minimum:
        widths.append(minimum)
    return widths


def judge_candidate(
    val_acc: float,
    minimum_val_accuracy: float,
    previous_accepted_accuracy: float | None,
    maximum_accuracy_drop: float | None,
) -> SearchDecision:
    """Admission rule: is this candidate deployable at all?

    This no longer ends the search -- step 3f's Pareto rule does that. A
    candidate rejected here is simply not a deployable model, so it never joins
    the frontier, but narrower candidates behind it still get their turn.
    """
    if val_acc < minimum_val_accuracy:
        return SearchDecision(
            False,
            f"validation accuracy {val_acc:.4f} fell below "
            f"the {minimum_val_accuracy:.4f} floor",
            "accuracy_floor",
        )
    if (
        maximum_accuracy_drop is not None
        and previous_accepted_accuracy is not None
        and previous_accepted_accuracy - val_acc > maximum_accuracy_drop
    ):
        return SearchDecision(
            False,
            f"validation accuracy dropped by "
            f"{previous_accepted_accuracy - val_acc:.4f}, exceeding "
            f"the {maximum_accuracy_drop:.4f} limit",
            "accuracy_drop",
        )
    return SearchDecision(True, "validation objective satisfied")


def candidate_costs(result: DendriticCycleResult) -> dict[str, float]:
    """The deployment costs this candidate actually recorded."""
    costs: dict[str, float] = {"deployed_params": float(result.deployed_params)}
    for key in CANDIDATE_COST_KEYS:
        if key != "deployed_params" and result.cost and key in result.cost:
            costs[key] = float(result.cost[key])
    return costs


class ParetoSearch:
    """Replays every admitted candidate through a frontier over shared cost axes.

    Candidates can differ in which costs they recorded -- a run reused from
    before step 3e's profiling has only its parameter count. Comparing on an
    axis one candidate is missing would silently treat it as free, so the
    comparison uses the intersection of axes and is replayed from scratch
    whenever that intersection shrinks.
    """

    def __init__(
        self,
        *,
        minimum_accuracy: float,
        patience: int,
        accuracy_tolerance: float = 0.0,
        relative_cost_tolerance: float = 0.0,
    ):
        self.minimum_accuracy = minimum_accuracy
        self.patience = patience
        self.accuracy_tolerance = accuracy_tolerance
        self.relative_cost_tolerance = relative_cost_tolerance
        self.entries: list[ParetoPoint] = []
        self.frontier = ParetoFrontier(
            cost_keys=("deployed_params",),
            minimum_accuracy=minimum_accuracy,
            patience=patience,
            accuracy_tolerance=accuracy_tolerance,
            relative_cost_tolerance=relative_cost_tolerance,
        )
        # Keep this outside ParetoFrontier: the frontier object is rebuilt when
        # a legacy candidate removes an axis, and rebuilding must not erase
        # candidates already spent against the patience budget.
        self.stagnant_streak = 0
        self.stall_causes: list[str] = []

    @property
    def cost_keys(self) -> tuple[str, ...]:
        if not self.entries:
            return ("deployed_params",)
        shared = set(self.entries[0].costs)
        for point in self.entries[1:]:
            shared &= set(point.costs)
        return tuple(key for key in CANDIDATE_COST_KEYS if key in shared)

    def add(self, point: ParetoPoint):
        previous_keys = self.cost_keys
        self.entries.append(point)
        keys = self.cost_keys
        if keys != previous_keys:
            if set(keys) < set(previous_keys):
                logger.info(
                    "Candidate %s did not record %s; comparing the frontier on %s only",
                    point.label,
                    sorted(set(previous_keys) - set(keys)),
                    list(keys),
                )
            self.frontier = ParetoFrontier(
                cost_keys=keys,
                minimum_accuracy=self.minimum_accuracy,
                patience=self.patience,
                accuracy_tolerance=self.accuracy_tolerance,
                relative_cost_tolerance=self.relative_cost_tolerance,
            )
            update = None
            for entry in self.entries:
                update = self.frontier.add(entry)
            assert update is not None
        else:
            update = self.frontier.add(point)

        self.stagnant_streak = (
            0 if update.extended_frontier else self.stagnant_streak + 1
        )
        if update.extended_frontier:
            self.stall_causes.clear()
        else:
            self.stall_causes.append(
                "duplicate_or_dominated"
                if update.admitted
                else "accuracy_inadmissible"
            )
        self.frontier.stagnant_streak = self.stagnant_streak
        return update

    def record_inadmissible(
        self,
        reason: str = "candidate failed admission",
        cause: str = "accuracy_inadmissible",
    ) -> None:
        """Count a rejected candidate without coupling to a replaceable frontier."""
        self.stagnant_streak += 1
        self.stall_causes.append(cause)
        self.frontier.stagnant_streak = self.stagnant_streak
        logger.info(
            "Pareto: inadmissible candidate (%s), stagnant streak %d/%d",
            reason,
            self.stagnant_streak,
            self.patience,
        )

    def should_stop(self) -> bool:
        return self.stagnant_streak >= self.patience

    @property
    def stop_cause(self) -> str:
        """Explain whether patience was spent on admissibility or frontier progress."""
        if not self.stall_causes:
            return "unknown"
        recent = self.stall_causes[-self.patience :]
        if recent and all(cause == "accuracy_inadmissible" for cause in recent):
            return "accuracy_inadmissible"
        return "pareto_stagnation"


def _source_block_width(checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    widths = list(checkpoint["model_cfg"]["block_channels"])
    if not widths or len(set(widths)) != 1:
        raise ValueError(
            "The pruning search currently requires equal source block widths; "
            f"got {widths}"
        )
    return widths[0]


def _target_model_cfg(checkpoint_path: str, width: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_cfg"])
    config["name"] = f"ds_cnn_dendritic_w{width}"
    config["block_channels"] = [width] * len(config["block_channels"])
    return config


def _load_completed_result(
    save_name: str,
    width: int,
    checkpoint_path: str,
    teacher_checkpoint: str,
    data_cfg: dict,
    train_cfg: dict,
    expected_run_id: str | None = None,
) -> DendriticCycleResult:
    """Read a completed run only when it contains every framework substage."""
    run_dir = Path(save_name)
    if not (run_dir / "final_clean_pai.pt").exists():
        raise ValueError(
            f"Cannot reuse incomplete run {save_name!r}: final_clean_pai.pt is missing"
        )
    best_val_acc, deployed_params = read_pai_architecture_results(save_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_width = _source_block_width(checkpoint_path)
    keep_ratio = width / source_width

    from kws.optimize.dendritic import build_cycle_base

    base, _, _ = build_cycle_base(checkpoint_path, keep_ratio)
    scores_path = run_dir / f"{run_dir.name}Scores.csv"
    epochs = max(sum(1 for _ in scores_path.open()) - 1, 0)

    metadata_path = run_dir / "cycle_metadata.yaml"
    if not metadata_path.exists():
        raise ValueError(
            f"Cannot reuse legacy run {save_name!r}: {metadata_path} is missing"
        )
    with metadata_path.open() as f:
        metadata = yaml.safe_load(f) or {}
    if expected_run_id is not None and metadata.get("run_id") != expected_run_id:
        raise ValueError(
            f"completed run {save_name!r} belongs to run_id={metadata.get('run_id')!r}, "
            f"expected {expected_run_id!r}"
        )
    if metadata.get("framework_cycle_version") != FRAMEWORK_CYCLE_VERSION:
        raise ValueError(
            f"Cannot reuse legacy run {save_name!r}: framework cycle version "
            f"{metadata.get('framework_cycle_version')!r} != "
            f"{FRAMEWORK_CYCLE_VERSION}"
        )

    candidate_train_cfg = dict(train_cfg)
    candidate_train_cfg["pruning"] = {
        "kind": "structured",
        "keep_ratio": width / source_width,
    }
    expected_fingerprint = cycle_fingerprint(
        checkpoint_path,
        teacher_checkpoint,
        data_cfg,
        _target_model_cfg(checkpoint_path, width),
        candidate_train_cfg,
    )
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"completed run {save_name!r} was produced by a different "
            "data/model/training recipe"
        )

    recorded = metadata.get("result") or {}
    recorded_cfg = metadata.get("base_model_cfg") or {}
    recorded_widths = list(recorded_cfg.get("block_channels") or [])
    expected_widths = [width] * len(checkpoint["model_cfg"]["block_channels"])
    if recorded_widths != expected_widths:
        raise ValueError(
            f"completed run {save_name!r} records block_channels="
            f"{recorded_widths}, requested {expected_widths}"
        )
    recorded_source = metadata.get("source_checkpoint")
    if recorded_source != checkpoint_path:
        raise ValueError(
            f"completed run {save_name!r} came from {recorded_source!r}, "
            f"not the requested checkpoint {checkpoint_path!r}"
        )
    if metadata.get("teacher_checkpoint") != teacher_checkpoint:
        raise ValueError(
            f"completed run {save_name!r} used teacher "
            f"{metadata.get('teacher_checkpoint')!r}, not {teacher_checkpoint!r}"
        )

    prune_finetune = recorded.get("prune_finetune") or {}
    resume = recorded.get("resume") or {}
    cost = recorded.get("cost") or {}
    missing_costs = [
        key for key in CANDIDATE_COST_KEYS if key != "deployed_params" and key not in cost
    ]
    if (
        prune_finetune.get("status") != "complete"
        or resume.get("status") not in {"complete", "no_improvement"}
        or missing_costs
        or not recorded.get("phase_trail")
    ):
        raise ValueError(
            f"Cannot reuse incomplete framework run {save_name!r}: "
            f"prune_finetune={prune_finetune.get('status')!r}, "
            f"resume={resume.get('status')!r}, missing_costs={missing_costs}, "
            f"phase_trail={bool(recorded.get('phase_trail'))}"
        )

    return DendriticCycleResult(
        save_name=save_name,
        block_channels=(recorded_widths or [width] * len(checkpoint["model_cfg"]["block_channels"])),
        base_params=sum(parameter.numel() for parameter in base.parameters()),
        deployed_params=int((recorded.get("cost") or {}).get("params", deployed_params)),
        best_val_acc=float(recorded.get("best_val_acc", best_val_acc)),
        epochs=epochs,
        elapsed_seconds=0.0,
        pai_deployed_params=int(recorded.get("pai_deployed_params", deployed_params)),
        cost=cost,
        distillation=recorded.get("distillation"),
        resume=recorded.get("resume"),
        prune_finetune=recorded.get("prune_finetune"),
        phase_trail=recorded.get("phase_trail") or [],
        run_id=expected_run_id,
    )


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    temporary.replace(path)


def run_pruning_search(
    checkpoint_path: str,
    data_cfg: dict,
    train_cfg: dict,
    search_cfg: dict,
    *,
    teacher_checkpoint: str | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Run steps 3a-3f over descending base widths until the frontier stalls."""
    train_cfg = with_seed(train_cfg, seed)
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)
    if layout is not None:
        layout.ensure_tree()
        search_cfg = dict(search_cfg)
        search_cfg["summary_path"] = str(layout.report_path("sparsity.yaml"))
        search_cfg["save_prefix"] = str(layout.root / "pai" / "candidates" / "candidate")
    source_width = _source_block_width(checkpoint_path)
    start_width = search_cfg["start_channels"]
    widths = candidate_widths(
        start_width,
        search_cfg["minimum_channels"],
        search_cfg["channel_step"],
    )
    if start_width > source_width:
        raise ValueError(
            f"start_channels={start_width} exceeds source width {source_width}"
        )

    minimum_accuracy = float(search_cfg["minimum_validation_accuracy"])
    configured_objective = train_cfg.get("objective", {}).get("minimum")
    if configured_objective is not None and float(configured_objective) != minimum_accuracy:
        raise ValueError(
            "train objective.minimum and search minimum_validation_accuracy "
            "must agree"
        )
    maximum_drop = search_cfg.get("maximum_accuracy_drop")
    if maximum_drop is not None:
        maximum_drop = float(maximum_drop)
    accuracy_tolerance = float(search_cfg.get("pareto_accuracy_tolerance", 0.0))
    relative_cost_tolerance = float(
        search_cfg.get("pareto_relative_cost_tolerance", 0.0)
    )
    save_prefix = search_cfg["save_prefix"]
    summary_path = Path(search_cfg["summary_path"])
    reuse_run = search_cfg.get("reuse_completed_start_run")
    reuse_completed_candidates = search_cfg.get("reuse_completed_candidates", True)
    teacher_checkpoint = teacher_checkpoint or search_cfg.get("teacher_checkpoint")
    if not teacher_checkpoint:
        raise ValueError("the framework sparsity sweep requires a fixed teacher")
    source_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if run_id is not None and (
        layout is None
        or Path(checkpoint_path).expanduser().resolve().is_relative_to(layout.root)
    ):
        validate_checkpoint_run_id(source_checkpoint, run_id, source=checkpoint_path)
    source_distillation = source_checkpoint.get("distillation") or {}
    expected_teacher_sha = file_sha256(teacher_checkpoint)
    if (
        source_distillation.get("teacher_checkpoint") != teacher_checkpoint
        or source_distillation.get("teacher_sha256") != expected_teacher_sha
    ):
        raise ValueError(
            "the sparsity source is not provenance-linked to the configured "
            "fixed teacher; rerun student distillation"
        )
    fingerprint = search_fingerprint(
        checkpoint_path, teacher_checkpoint, data_cfg, train_cfg, search_cfg
    )

    search = ParetoSearch(
        minimum_accuracy=minimum_accuracy,
        patience=int(search_cfg.get("pareto_patience", 2)),
        accuracy_tolerance=accuracy_tolerance,
        relative_cost_tolerance=relative_cost_tolerance,
    )
    summary = {
        "status": "running",
        "framework_cycle_version": FRAMEWORK_CYCLE_VERSION,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "selection_split": "validation",
        "test_split_used": False,
        "stopping_rule": None,
        "source_checkpoint": checkpoint_path,
        "teacher_checkpoint": teacher_checkpoint,
        "source_block_width": source_width,
        "minimum_validation_accuracy": minimum_accuracy,
        "maximum_accuracy_drop": maximum_drop,
        "pareto_patience": search.patience,
        "pareto_accuracy_tolerance": accuracy_tolerance,
        "pareto_relative_cost_tolerance": relative_cost_tolerance,
        "candidates": [],
        "last_accepted": None,
        "pareto": None,
        "stopped_on": None,
    }
    previous_accuracy = None

    for index, width in enumerate(widths):
        save_name = f"{save_prefix}_w{width}"
        completed_run = None
        if index == 0 and reuse_run:
            completed_run = reuse_run
        elif reuse_completed_candidates and (
            Path(save_name) / "final_clean_pai.pt"
        ).exists():
            completed_run = save_name

        if completed_run:
            result = _load_completed_result(
                completed_run,
                width,
                checkpoint_path,
                teacher_checkpoint,
                data_cfg,
                train_cfg,
                expected_run_id=run_id,
            )
            logger.info(
                "Reusing completed width-%d run from %s", width, completed_run
            )
        else:
            run_dir = Path(save_name)
            if run_dir.exists() and any(run_dir.iterdir()):
                sidecar = (
                    layout.checkpoint_path("sparsity", "pai", "latest", candidate=Path(save_name).name)
                    if layout is not None else run_dir / "kws_latest.pt"
                )
                if not sidecar.exists():
                    raise ValueError(
                        f"Refusing to overwrite partial run {save_name!r}: paired PAI/KWS "
                        "latest state is missing or incompatible; choose a new output root"
                    )
                logger.info("Resuming compatible partial candidate %s", save_name)
            candidate_train_cfg = dict(train_cfg)
            candidate_train_cfg["pruning"] = {
                "kind": "structured",
                "keep_ratio": width / source_width,
            }
            result = run_cycle(
                checkpoint_path,
                data_cfg,
                _target_model_cfg(checkpoint_path, width),
                candidate_train_cfg,
                save_name,
                teacher_checkpoint=teacher_checkpoint,
                output_dir=output_dir,
                run_id=run_id,
            )

        decision = judge_candidate(
            result.best_val_acc,
            minimum_accuracy,
            previous_accuracy,
            maximum_drop,
        )
        record = asdict(result)
        record.update(asdict(decision))

        update = None
        if decision.accepted:
            update = search.add(
                ParetoPoint(
                    label=f"w{width}",
                    accuracy=result.best_val_acc,
                    costs=candidate_costs(result),
                    detail={"save_name": result.save_name, "width": width},
                )
            )
            summary["last_accepted"] = record
            previous_accuracy = result.best_val_acc
        else:
            # An inadmissible candidate still counts against the patience
            # budget: it is evidence the sweep has gone past the useful range.
            search.record_inadmissible(decision.reason, decision.cause)
            logger.info(
                "Width %d not admissible (%s); patience %d/%d",
                width,
                decision.reason,
                search.frontier.stagnant_streak,
                search.patience,
            )

        record["pareto"] = update.as_dict() if update else None
        summary["candidates"].append(record)
        summary["pareto"] = search.frontier.as_dict()
        _write_summary(summary_path, summary)

        if search.should_stop():
            summary["status"] = "complete"
            summary["stopping_rule"] = "pareto_frontier_stalled"
            summary["stopped_on"] = {
                "candidate": record,
                "stop_cause": search.stop_cause,
                "stall_causes": list(search.stall_causes[-search.patience:]),
                "reason": (
                    f"the Pareto frontier did not improve for "
                    f"{search.patience} consecutive candidates"
                ),
            }
            _write_summary(summary_path, summary)
            logger.info(
                "Stopping the sweep at width %d: %s",
                width,
                summary["stopped_on"]["reason"],
            )
            return summary

    summary["status"] = "complete"
    summary["stopping_rule"] = "minimum_channels_reached"
    summary["stopped_on"] = {
        "stop_cause": "minimum_channels_reached",
        "reason": "minimum channel width reached while the frontier was still improving"
    }
    _write_summary(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config", default="configs/data/speech_commands_v2.yaml"
    )
    parser.add_argument(
        "--train-config", default="configs/train/dendritic_cycle1.yaml"
    )
    parser.add_argument(
        "--search-config", default="configs/train/dendritic_prune_loop.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/ds_cnn_xs_distilled_warm_12class.pt",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Fixed teacher for KD; defaults to the search config's value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training-config seed (must be non-negative)",
    )
    parser.add_argument("--output-dir", default=None)
    graphs.add_cli_flag(parser)
    args = parser.parse_args()
    if args.graphs:
        if args.output_dir is None:
            parser.error("--graphs requires --output-dir")
        graphs.enable()

    inputs = [
        (args.data_config, "data_config"),
        (args.train_config, "train_config"),
        (args.search_config, "search_config"),
        (args.checkpoint, "source_checkpoint"),
    ]
    if args.teacher_checkpoint:
        inputs.append((args.teacher_checkpoint, "teacher_checkpoint"))
    with run_session(
        args.output_dir,
        command="kws.optimize.dendritic_prune_loop",
        argv=__import__("sys").argv,
        seed=args.seed,
        inputs=inputs,
    ):
        run_pruning_search(
            args.checkpoint,
            load_yaml(args.data_config),
            load_yaml(args.train_config),
            load_yaml(args.search_config),
            teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
