"""Structured (channel) pruning for DS-CNN.

Only channel pruning is implemented -- unstructured/weight-level sparsity buys
nothing for TFLite Micro / ESP-DL deployment (no sparse-kernel support), so it
isn't worth the complexity here. Pruning actually removes output channels from
each block's pointwise conv (ranked by L1 norm) and threads the corresponding
input-channel selection through the next block's depthwise conv and BN, or the
final FC layer for the last block -- producing a genuinely smaller model, not
just a masked one.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.models.ds_cnn import DSCNN
from kws.train import train_model
from kws.utils.device import get_device
from kws.utils.logging import get_logger
from kws.utils.seed import set_seed

logger = get_logger(__name__)


def _l1_topk_indices(weight: torch.Tensor, k: int) -> torch.Tensor:
    scores = weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(scores, k).indices.sort().values


def _copy_bn_subset(old_bn: nn.BatchNorm2d, new_bn: nn.BatchNorm2d, indices: torch.Tensor) -> None:
    new_bn.weight.data = old_bn.weight.data[indices].clone()
    new_bn.bias.data = old_bn.bias.data[indices].clone()
    new_bn.running_mean.data = old_bn.running_mean.data[indices].clone()
    new_bn.running_var.data = old_bn.running_var.data[indices].clone()


def prune_ds_cnn(model: DSCNN, keep_ratio: float) -> DSCNN:
    old_block_channels = [block.pointwise.out_channels for block in model.blocks]
    new_block_channels = [max(1, int(round(c * keep_ratio))) for c in old_block_channels]

    new_model = DSCNN(
        input_shape=model.input_shape,
        num_classes=model.fc.out_features,
        initial_channels=model.stem[0].out_channels,
        initial_kernel=model.stem[0].kernel_size[0],
        initial_stride=model.stem[0].stride[0],
        block_channels=new_block_channels,
        dropout=model.dropout.p,
    )
    new_model.stem.load_state_dict(model.stem.state_dict())

    keep_in = None  # surviving input-channel indices for the current block; None = keep all (from stem)
    for old_block, new_block, n_keep in zip(model.blocks, new_model.blocks, new_block_channels):
        if keep_in is None:
            new_block.depthwise.weight.data = old_block.depthwise.weight.data.clone()
            new_block.bn1.load_state_dict(old_block.bn1.state_dict())
        else:
            new_block.depthwise.weight.data = old_block.depthwise.weight.data[keep_in].clone()
            _copy_bn_subset(old_block.bn1, new_block.bn1, keep_in)

        keep_out = _l1_topk_indices(old_block.pointwise.weight, n_keep)
        pw_weight = old_block.pointwise.weight.data
        if keep_in is not None:
            pw_weight = pw_weight[:, keep_in]
        new_block.pointwise.weight.data = pw_weight[keep_out].clone()
        _copy_bn_subset(old_block.bn2, new_block.bn2, keep_out)

        keep_in = keep_out

    new_model.fc.weight.data = model.fc.weight.data[:, keep_in].clone()
    new_model.fc.bias.data = model.fc.bias.data.clone()
    return new_model


def prune_and_fine_tune(checkpoint_path: str, data_cfg: dict, train_cfg: dict,
                         keep_ratio: float, out_checkpoint: Path) -> float:
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    from kws.models.ds_cnn import build_ds_cnn
    model = build_ds_cnn(ckpt["model_cfg"], tuple(ckpt["input_shape"]), ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])

    pruned = prune_ds_cnn(model, keep_ratio).to(device)
    old_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in pruned.parameters())
    logger.info("Pruned %d -> %d params (keep_ratio=%.2f)", old_params, new_params, keep_ratio)

    set_seed(train_cfg["seed"])
    datasets, label_map = build_datasets(data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"])
    pruned_model_cfg = dict(ckpt["model_cfg"])
    pruned_model_cfg["name"] = ckpt["model_cfg"]["name"] + f"_pruned{keep_ratio:.2f}"

    return train_model(pruned, datasets, label_map, pruned_model_cfg, train_cfg,
                        out_checkpoint, device, ckpt["num_keywords"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/full.yaml")
    parser.add_argument("--checkpoint", required=True, help="Trained DS-CNN-L checkpoint to prune")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--out-checkpoint", required=True)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    prune_and_fine_tune(args.checkpoint, data_cfg, train_cfg, args.keep_ratio, Path(args.out_checkpoint))


if __name__ == "__main__":
    main()
