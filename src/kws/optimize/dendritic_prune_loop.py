"""Search for the narrowest DS-CNN that dendrites keep above an accuracy floor.

PerforatedAI adds capacity; it does not structurally prune a learned dendritic
network.  Following its recommended size-optimization workflow, this module
prunes progressively narrower *base* networks from the same trained source
checkpoint and gives every candidate one complete dynamic dendrite cycle.
Only validation accuracy controls acceptance.  The test split is never loaded.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from kws.optimize.dendritic import (
    DendriticCycleResult,
    load_yaml,
    read_pai_architecture_results,
    run_cycle,
)
from kws.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchDecision:
    """Acceptance decision for a completed width candidate."""

    accepted: bool
    reason: str


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
    """Apply the configured validation-only degradation rule."""
    if val_acc < minimum_val_accuracy:
        return SearchDecision(
            False,
            f"validation accuracy {val_acc:.4f} fell below "
            f"the {minimum_val_accuracy:.4f} floor",
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
        )
    return SearchDecision(True, "validation objective satisfied")


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
) -> DendriticCycleResult:
    """Read a completed PAI run, including the cycle-1 run made before this loop."""
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
    return DendriticCycleResult(
        save_name=save_name,
        block_channels=[width] * len(checkpoint["model_cfg"]["block_channels"]),
        base_params=sum(parameter.numel() for parameter in base.parameters()),
        deployed_params=deployed_params,
        best_val_acc=best_val_acc,
        epochs=epochs,
        elapsed_seconds=0.0,
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
) -> dict:
    """Run successively narrower dendritic models until validation degrades."""
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
    maximum_drop = search_cfg.get("maximum_accuracy_drop")
    if maximum_drop is not None:
        maximum_drop = float(maximum_drop)
    save_prefix = search_cfg["save_prefix"]
    summary_path = Path(search_cfg["summary_path"])
    reuse_run = search_cfg.get("reuse_completed_start_run")
    reuse_completed_candidates = search_cfg.get("reuse_completed_candidates", True)

    summary = {
        "status": "running",
        "selection_split": "validation",
        "test_split_used": False,
        "source_checkpoint": checkpoint_path,
        "source_block_width": source_width,
        "minimum_validation_accuracy": minimum_accuracy,
        "maximum_accuracy_drop": maximum_drop,
        "candidates": [],
        "last_accepted": None,
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
            result = _load_completed_result(completed_run, width, checkpoint_path)
            logger.info(
                "Reusing completed width-%d run from %s", width, completed_run
            )
        else:
            run_dir = Path(save_name)
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(
                    f"Refusing to overwrite partial run {save_name!r}. Resume it "
                    "with PerforatedAI or choose a different save_prefix."
                )
            candidate_train_cfg = dict(train_cfg)
            candidate_train_cfg["pruning"] = {"keep_ratio": width / source_width}
            result = run_cycle(
                checkpoint_path,
                data_cfg,
                _target_model_cfg(checkpoint_path, width),
                candidate_train_cfg,
                save_name,
            )

        decision = judge_candidate(
            result.best_val_acc,
            minimum_accuracy,
            previous_accuracy,
            maximum_drop,
        )
        record = asdict(result)
        record.update(asdict(decision))
        summary["candidates"].append(record)

        if not decision.accepted:
            summary["status"] = "complete"
            summary["stopped_on"] = record
            _write_summary(summary_path, summary)
            logger.info("Stopping pruning search at width %d: %s", width, decision.reason)
            return summary

        summary["last_accepted"] = record
        previous_accuracy = result.best_val_acc
        _write_summary(summary_path, summary)
        logger.info(
            "Accepted width %d: val_acc=%.4f deployed_params=%d",
            width,
            result.best_val_acc,
            result.deployed_params,
        )

    summary["status"] = "complete"
    summary["stopped_on"] = {
        "reason": "minimum channel width reached without degradation"
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
        default="models/checkpoints/ds_cnn_xs_distilled_warm.pt",
    )
    args = parser.parse_args()

    run_pruning_search(
        args.checkpoint,
        load_yaml(args.data_config),
        load_yaml(args.train_config),
        load_yaml(args.search_config),
    )


if __name__ == "__main__":
    main()
