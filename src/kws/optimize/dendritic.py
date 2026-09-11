"""Run one size-focused PerforatedAI dendrite cycle on a pruned DS-CNN.

The first cycle starts from the trained XS checkpoint, removes channels to
form a 1,716-parameter XXS base model, and perforates its two complete
depthwise-separable blocks plus its classifier. PAI's validation tracker
controls neuron/dendrite phase changes; the test split is never loaded or
consulted by this module.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import torch
import torch.nn as nn
import yaml
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.models.layers import DSConvBlock
from kws.optimize.prune import prune_ds_cnn
from kws.train import build_lr_scheduler, evaluate_loss_acc
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

logger = get_logger(__name__)


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


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cycle_base(
    checkpoint_path: str,
    keep_ratio: float,
    target_model_cfg: dict | None = None,
):
    """Load a trained DS-CNN and structurally prune its block channels."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        if target_model_cfg["initial_channels"] != pruned.stem[0].out_channels:
            raise ValueError("XXS initial_channels does not match the pruned model")
        if target_model_cfg["block_channels"] != generated_channels:
            raise ValueError(
                "XXS block_channels does not match the pruned model: "
                f'{target_model_cfg["block_channels"]} != {generated_channels}'
            )
        model_cfg = dict(target_model_cfg)
    return pruned, checkpoint, model_cfg


def estimate_one_dendrite_params(model: nn.Module) -> int:
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


def _make_optimizer_and_scheduler(model, train_cfg: dict, loader_length: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_lr_scheduler(
        optimizer,
        train_cfg["epochs"] * loader_length,
        train_cfg["warmup_fraction"],
    )
    GPA.pai_tracker.set_optimizer_instance(optimizer)
    return optimizer, scheduler


def run_cycle(
    checkpoint_path: str,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    save_name: str,
) -> DendriticCycleResult:
    """Run PAI's alternating neuron/dendrite phases using validation accuracy."""
    started_at = monotonic()
    set_seed(train_cfg["seed"])
    device = get_device()
    base, checkpoint, model_cfg = build_cycle_base(
        checkpoint_path,
        train_cfg["pruning"]["keep_ratio"],
        target_model_cfg=model_cfg,
    )
    base_params = sum(parameter.numel() for parameter in base.parameters())
    logger.info(
        "Cycle-1 XXS base: %d params, estimated one-dendrite deployment: %d "
        "params, block_channels=%s",
        base_params,
        estimate_one_dendrite_params(base),
        model_cfg["block_channels"],
    )

    datasets, _ = build_datasets(
        data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"]
    )
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)

    configure_perforatedai(train_cfg["perforatedai"], device)
    model = UPA.perforate_model(
        base,
        doing_pai=True,
        save_name=save_name,
        making_graphs=True,
        maximizing_score=True,
    ).to(device)
    logger.info("PAI-wrapped initial parameter count: %d", UPA.count_params(model))

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, train_cfg, len(train_loader)
    )

    epoch = -1
    while True:
        epoch += 1
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)

        train_loss = running_loss / train_count
        train_acc = train_correct / train_count
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device, criterion)
        GPA.pai_tracker.add_extra_score(train_acc, "Train")
        model, restructured, training_complete = (
            GPA.pai_tracker.add_validation_score(val_acc, model)
        )
        model = model.to(device)

        logger.info(
            "epoch %d train_loss=%.4f train_acc=%.4f val_loss=%.4f "
            "val_acc=%.4f params=%d",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            UPA.count_params(model),
        )

        if training_complete:
            logger.info("PAI cycle complete; results saved under %s", save_name)
            break
        if restructured:
            logger.info("PAI restructured the network; resetting optimizer phase")
            optimizer, scheduler = _make_optimizer_and_scheduler(
                model, train_cfg, len(train_loader)
            )

    best_val_acc, deployed_params = read_pai_architecture_results(save_name)
    result = DendriticCycleResult(
        save_name=save_name,
        block_channels=list(model_cfg["block_channels"]),
        base_params=base_params,
        deployed_params=deployed_params,
        best_val_acc=best_val_acc,
        epochs=epoch + 1,
        elapsed_seconds=monotonic() - started_at,
    )

    # PAI owns its architecture checkpoints. Preserve the source metadata next
    # to them so the exporter, evaluator, and pruning search can reconstruct it.
    metadata_path = Path(save_name) / "cycle_metadata.yaml"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        yaml.safe_dump(
            {
                "status": "complete",
                "result": asdict(result),
                "source_checkpoint": checkpoint_path,
                "source_val_acc": checkpoint.get("val_acc"),
                "base_model_cfg": model_cfg,
                "base_params": base_params,
                "selection_split": "validation",
                "test_split_used": False,
                "objective": train_cfg["objective"],
                "perforatedai": train_cfg["perforatedai"],
            },
            f,
            sort_keys=False,
        )
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
        default="models/checkpoints/ds_cnn_xs_distilled_warm.pt",
    )
    parser.add_argument("--save-name", default="dendritic_xxs_cycle1")
    args = parser.parse_args()

    run_cycle(
        args.checkpoint,
        load_yaml(args.data_config),
        load_yaml(args.model_config),
        load_yaml(args.train_config),
        args.save_name,
    )


if __name__ == "__main__":
    main()
