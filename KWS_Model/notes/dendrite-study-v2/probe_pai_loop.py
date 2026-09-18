#!/usr/bin/env python
"""Drive the real PAI switch loop on synthetic data to observe its dynamics.

Read-only audit helper for notes/dendrite-study-v2/INTEGRATION_AUDIT.md.  It
reuses the production helpers from ``kws.optimize.dendritic`` (same
``configure_perforatedai``, same ``_make_optimizer_and_scheduler``, same
``_restructure_lr_multiplier``) but replaces the dataset with 128 random
samples so a full n -> p -> n search finishes in seconds.  Nothing here writes
into the repository; PAI's artifacts land in a temporary directory.

Usage (from BASE):
    uv run --env-file .env python notes/dendrite-study-v2/probe_pai_loop.py ARM [WIDTH]
"""
from __future__ import annotations

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
from kws.optimize import dendritic as D  # noqa: E402

ARMS = {
    "fc": "configs/train/sparknet_c16_dendritic_prune_no_kd.yaml",
    "pointwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_pointwise.yaml",
    "depthwise": "configs/train/sparknet_c16_dendritic_prune_no_kd_backbone_depthwise.yaml",
    "gate_conv": "configs/train/sparknet_c16_dendritic_prune_no_kd_gate_conv.yaml",
    "control": "configs/train/sparknet_c16_dendritic_prune_no_kd_control.yaml",
}
INPUT_SHAPE = (32, 101)
NUM_CLASSES = 12
MAX_EPOCHS = 45


def main() -> int:
    arm = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    torch.manual_seed(0)

    train_cfg = yaml.safe_load((ROOT / ARMS[arm]).read_text())
    pai_cfg = dict(train_cfg["perforatedai"])
    # Shrink the schedule so the full search terminates in this probe.
    pai_cfg["n_epochs_to_switch"] = 3
    pai_cfg["history_lookback"] = 2
    train_cfg["lr"] = 1e-3
    train_cfg["dendritic_schedule_epochs"] = 10

    model_cfg = yaml.safe_load(
        (ROOT / f"configs/model/sparknet_c{width}_paper.yaml").read_text()
    )
    base = build_sparknet(model_cfg, INPUT_SHAPE, NUM_CLASSES)

    device = torch.device("cpu")
    N, BS = 480, 8  # 60 batches/epoch > initial_correlation_batches=40
    x = torch.randn(N, 1, *INPUT_SHAPE)
    y = torch.randint(0, NUM_CLASSES, (N,))
    batches = [(x[i : i + BS], y[i : i + BS]) for i in range(0, N, BS)]

    D.configure_perforatedai(pai_cfg, device)
    post_mult = float(pai_cfg["post_integration_lr_multiplier"])

    scratch = tempfile.mkdtemp()
    run_dir = Path(scratch) / f"probe_{arm}_c{width}"
    run_dir.mkdir(parents=True)
    prev = Path.cwd()
    os.chdir(run_dir.parent)
    try:
        model = D.UPA.perforate_model(
            base, doing_pai=True, save_name=run_dir.name,
            making_graphs=False, maximizing_score=True,
        ).to(device)
        optimizer, scheduler, _, _ = D._make_optimizer_and_scheduler(
            model, train_cfg, len(batches)
        )
        criterion = nn.CrossEntropyLoss()
        last_mode = D.current_pai_mode()
        print(
            f"{'ep':>3} {'mode':>4} {'->':>4} {'restr':>5} {'done':>5} "
            f"{'added':>5} {'integ':>5} {'grps':>4} {'lr':>10} "
            f"{'baseInOpt':>9} {'params':>7} {'mult':>5} {'bnPinned':>8}"
        )
        for epoch in range(MAX_EPOCHS):
            D.set_train_mode_preserving_frozen_batchnorm(model)
            phase_mode = D.current_pai_mode()
            base_live_pre = D.base_params_in_optimizer(model, optimizer)
            pinned = 0
            if phase_mode == "p":
                pinned = D.freeze_base_batchnorm_stats(model)
            for fx, fy in batches:
                optimizer.zero_grad()
                losses = {"total": 100.0 * criterion(model(fx), fy)}
                D._add_auxiliary_losses(losses, model)
                losses["total"].backward()
                optimizer.step()
                scheduler.step()
            model.eval()
            with torch.no_grad():
                val_acc = float((model(x).argmax(1) == y).float().mean())
            # Nudge the score downward over time so history switching fires.
            score = val_acc
            lr_before = optimizer.param_groups[0]["lr"]
            ngroups = len(optimizer.param_groups)
            integ_before = D.current_pai_integration_count()
            model, restructured, complete = D.GPA.pai_tracker.add_validation_score(
                score, model
            )
            model = model.to(device)
            mode = D.current_pai_mode()
            integ_after = D.current_pai_integration_count()
            dendrite_integrated = (
                integ_before is not None
                and integ_after is not None
                and integ_after > integ_before
            )
            mult = 1.0
            if restructured:
                mult = D._restructure_lr_multiplier(
                    phase_mode, mode, post_mult,
                    dendrite_integrated=dendrite_integrated,
                )
                optimizer, scheduler, _, _ = D._make_optimizer_and_scheduler(
                    model, train_cfg, len(batches), lr_multiplier=mult
                )
            added = D.GPA.pai_tracker.member_vars.get("num_dendrites_added")
            print(
                f"{epoch:>3} {phase_mode:>4} {mode:>4} {str(restructured):>5} "
                f"{str(complete):>5} {str(added):>5} {str(integ_after):>5} "
                f"{ngroups:>4} {lr_before:>10.3e} {base_live_pre:>9} "
                f"{D.UPA.count_params(model):>7} {mult:>5.2f} {pinned:>8}",
                flush=True,
            )
            last_mode = mode
            if complete:
                print("TRAINING COMPLETE at epoch", epoch)
                break
        else:
            print(f"did not complete within {MAX_EPOCHS} epochs (last mode {last_mode})")
        mv = D.GPA.pai_tracker.member_vars
        print(
            "FINAL: switch_epochs=", mv.get("switch_epochs"),
            " num_cycles=", mv.get("num_cycles"),
            " num_dendrites_added=", mv.get("num_dendrites_added"),
            " num_dendrites_integrated=", mv.get("num_dendrites_integrated"),
            " num_dendrite_tries=", mv.get("num_dendrite_tries"),
        )
        arch = run_dir / f"{run_dir.name}_best_arch_scores.csv"
        print("arch csv exists:", arch.exists())
        if arch.exists():
            print(arch.read_text())
    finally:
        os.chdir(prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
