#!/usr/bin/env python
"""Empirically enumerate which SparkNet modules each placement arm wraps.

Read-only audit helper for notes/dendrite-study-v2/INTEGRATION_AUDIT.md.
Builds every SparkNet width used by the study, applies each arm's placement
filter through the *same* code paths the training loop uses
(``placement_module_names`` -> ``configure_perforatedai`` ->
``UPA.perforate_model``) and prints exactly which modules PAI wraps plus the
parameters each dendrite would add.

Run from BASE:
    uv run --env-file .env python notes/dendrite-study-v2/verify_placements.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kws.models.sparknet import build_sparknet  # noqa: E402
from kws.optimize.dendritic_config import (  # noqa: E402
    placement_module_names,
    project_dendritic_cost,
)

WIDTHS = (16, 12, 10, 8, 6, 4, 2)
ARMS = {
    "fc": "configs/train/sparknet_c16_dendritic_prune_no_kd.yaml",
    "pointwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_pointwise.yaml",
    "depthwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_depthwise.yaml",
    "gate_conv": "configs/train/sparknet_c16_dendritic_prune_no_kd_gate_conv.yaml",
    "control": "configs/train/sparknet_c16_dendritic_prune_no_kd_control.yaml",
}
INPUT_SHAPE = (32, 101)
NUM_CLASSES = 12


def build(width: int) -> nn.Module:
    cfg = yaml.safe_load((ROOT / f"configs/model/sparknet_c{width}_paper.yaml").read_text())
    return build_sparknet(cfg, INPUT_SHAPE, NUM_CLASSES)


def pai_wrapped_report(width: int, arm: str, pai_cfg: dict) -> dict:
    """Actually run PAI conversion and report the wrapped module inventory."""
    from kws.optimize.dendritic import GPA, UPA, configure_perforatedai

    model = build(width)
    before = sum(p.numel() for p in model.parameters())
    configure_perforatedai(pai_cfg, torch.device("cpu"))
    GPA.pc.set_testing_dendrite_capacity(False)
    with tempfile.TemporaryDirectory() as scratch:
        prev = Path.cwd()
        try:
            os.chdir(scratch)
            wrapped_model = UPA.perforate_model(
                model,
                doing_pai=True,
                save_name=f"probe_{arm}_c{width}",
                making_graphs=False,
                maximizing_score=True,
            )
        finally:
            os.chdir(prev)
    neuron_modules = []
    for name, module in wrapped_model.named_modules():
        cls = type(module).__name__
        if cls in {"PAINeuronModule", "PAINeuronLayer"} or hasattr(module, "dendrite_module"):
            if hasattr(module, "dendrite_module"):
                neuron_modules.append(name)
    after = sum(p.numel() for p in wrapped_model.parameters())
    # Report the underlying wrapped module type for each neuron module.
    detail = []
    for name in neuron_modules:
        mod = wrapped_model.get_submodule(name)
        inner = getattr(mod, "main_module", None)
        if inner is None:
            dm = getattr(mod, "dendrite_module", None)
            inner = getattr(dm, "parent_module", None) if dm is not None else None
        detail.append(
            {
                "neuron_module": name,
                "inner_type": type(inner).__name__ if inner is not None else "?",
                "inner_params": (
                    sum(p.numel() for p in inner.parameters()) if inner is not None else None
                ),
            }
        )
    return {
        "params_before_wrap": before,
        "params_after_wrap": after,
        "neuron_modules": detail,
        "n_wrapped": len(neuron_modules),
    }


def main() -> int:
    out: dict = {}
    for arm, cfg_path in ARMS.items():
        train_cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        pai_cfg = train_cfg["perforatedai"]
        out[arm] = {"module_ids": pai_cfg.get("module_ids"), "widths": {}}
        for width in WIDTHS:
            model = build(width)
            names = placement_module_names(model, pai_cfg)
            per_module = {}
            for name in names:
                sub = model.get_submodule(name)
                per_module[name] = {
                    "type": type(sub).__name__,
                    "params": sum(p.numel() for p in sub.parameters()),
                    "shape": (
                        list(sub.weight.shape) if hasattr(sub, "weight") else None
                    ),
                    "groups": getattr(sub, "groups", None),
                }
            proj = project_dendritic_cost(
                model, INPUT_SHAPE, pai_cfg, max_dendrites=1
            ).as_dict()
            pai = pai_wrapped_report(width, arm, dict(pai_cfg))
            out[arm]["widths"][width] = {
                "resolved_module_names": list(names),
                "per_module": per_module,
                "base_params": proj["base_params"],
                "base_macs": proj["base_macs"],
                "copied_params_per_dendrite": proj["copied_params_per_dendrite"],
                "copied_macs_per_dendrite": proj["copied_macs_per_dendrite"],
                "residual_params_per_dendrite": proj["residual_params_per_dendrite"],
                "residual_macs_per_dendrite": proj["residual_macs_per_dendrite"],
                "projected_params_1d": proj["projected_params"],
                "projected_macs_1d": proj["projected_macs"],
                "pai": pai,
            }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("placements.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"WROTE {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
