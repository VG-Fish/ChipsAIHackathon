#!/usr/bin/env python
"""Faithful replication of PerforatedAI's own KWS recipe on GSC-12.

PerforatedAI's keyword-spotting result (arXiv 2605.15647) came from the
"perforated-impulse-nn-block" Edge Impulse block
(``PerforatedAI/examples/submitted_projects/perforated-impulse-nn-block``).
This script mirrors that block's ``train.py`` loop (lines ~776-990) as closely
as possible, swapping in this project's Google Speech Commands v2 12-class data:

* Model ``EICNN(c1, c2)``: the block's audio CNN (``AudioClassifier`` in
  ``train_audio_dendritic.py``; the same layers the generalized ``train.py``
  builds from the EI layer list): conv3x3 c1 -> ReLU -> maxpool2 -> dropout .25
  -> conv3x3 c2 -> ReLU -> maxpool2 -> dropout .25 -> flatten -> Linear.
  ``--c3 N`` (default 0) appends a third conv3x3 N -> ReLU -> maxpool2 -> dropout
  block, the equal-parameter depth control (not part of the block's recipe);
  ``--c4 N`` (needs ``--c3``) appends a fourth such block the same way.
  MaxNorm(1.0) on the two convs after every optimizer step (``train.py``'s
  ``enforce_max_norm``), Xavier-uniform init with zero biases (``train.py``),
  Gaussian input noise in training only. Our input is MFCC (B, 1, 32, 101); the
  model keeps the first 13 coefficients (Edge Impulse uses 13) and applies
  per-utterance CMVN (``kws.models.ei_conv1d.InputNorm``) first, so the noise
  std is in normalized units (EI's features are normalized too).
* Loop: ``GPA.pai_tracker.set_optimizer(Adam)`` +
  ``set_scheduler(ReduceLROnPlateau)`` with ``mode=max`` and
  ``patience=int(0.75 * n_epochs_to_switch)``, Adam lr 0.005, betas
  (0.9, 0.999), eps 1e-7, batch 128 (the block's defaults), CrossEntropy.
  Validation accuracy (a fraction) drives ``add_validation_score``; on
  ``restructured`` the optimizer and scheduler are re-created with
  ``setup_optimizer``; on ``training_complete`` the loop stops with PAI's
  ``best_model`` already loaded.
* PAI config as in the block: improvement threshold "medium"
  [0.001, 0.0001, 0], candidate init multiplier 0.01, dendrite_update_mode,
  initial_correlation_batches 40, max_dendrite_tries 2, conversion
  "All Layers" (Conv2d + Linear perforated) or "Linear Only" (Conv2d tracked).

Arms (``--dendrites``): ``pb`` (perforated backpropagation / CC dendrites),
``gd`` (``set_perforated_backpropagation(False)``), ``none`` (same loop,
tracker, optimizer, scheduler and best-model bookkeeping, but
``doing_pai=False``, ``max_dendrites 0`` and switch mode ``DOING_NO_SWITCH`` so
PAI never restructures or early-stops it; it runs to ``--epochs-cap``).
``--restart-every K`` (``none`` only) re-creates optimizer + scheduler every K
epochs, the LR-restart control for PAI's per-restructure resets.
``--stop-lr X`` (``none`` only) ends the run once the scheduler drops the LR
below X. In b20-b23 (57 runs) no run set a new best validation accuracy once
LR < 5e-6, and 65% of no-dendrite compute came after LR hit its 5e-9 floor, so
with best-validation selection a 1e-6 stop gives the same model.

Deviations from the block, all deliberate:
* ``--phase0-optimizer tracked`` (default). The block calls ``setup_optimizer``
  and then immediately replaces the returned optimizer with a second, plain
  ``torch.optim.Adam(**optimArgs)`` (train.py:860-862), so until the first
  restructure it trains with an optimizer the tracker's scheduler never
  touches: constant lr 0.005 for the whole first neuron phase, while the
  tracker steps a scheduler attached to an unused optimizer. ``detached``
  reproduces that exactly; ``tracked`` uses the optimizer ``setup_optimizer``
  returns, so ReduceLROnPlateau governs every phase of every arm.
* Validation is our VAL split (the block validates on its test split when
  ``split-test`` is false); test is evaluated once, after training, on the
  best-validation checkpoint, through the same dataset/inference path as
  ``scripts/report_test_accuracy.py`` (fixed eval seed 0).
* Augmentation: none except the model's Gaussian noise (the block's audio
  model's only augmentation; its SpecAugment options were off by default).
  Features are precomputed once through ``kws.data.dataset.build_datasets``
  (augment=False) and kept on the device, like the block's on-device
  ``TensorDataset``.
* PAI graphs are off by default (``--graphs`` turns them on); they do not
  affect training.
* The model is on MPS: PAI does not auto-detect it, so ``GPA.pc.set_device`` is
  set explicitly (as ``kws.optimize.dendritic`` does).

Budgets: with n_epochs_to_switch 25, PAI's first switch came at epoch 77-137 in
pilots and a max-1 run completed at epoch 266, so a 200-epoch cap truncates
dendritic runs. ``--budget-snapshots`` copies PAI's best_model at the listed
epochs and test-evaluates each at the end, so one long run also gives the
result of every shorter cap (same trajectory, same selection rule).

Writes ``<out>/result.json`` (arm config, params, dendrite counts, epochs,
val/test accuracy, budget snapshots), ``<out>/epochs.jsonl`` (one row per loop
epoch) and PAI's own files under ``<out>/pai/``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]

from kws.data.dataset import build_datasets  # noqa: E402
from kws.data.splits import TEST, TRAIN, VAL  # noqa: E402
from kws.evaluate import run_inference  # noqa: E402
from kws.models.ei_conv1d import InputNorm  # noqa: E402
from kws.optimize.pai_import import import_pai_module  # noqa: E402
from kws.utils.device import get_device  # noqa: E402
from kws.utils.metrics import compute_metrics  # noqa: E402
from kws.utils.seed import set_seed  # noqa: E402

GPA: Any = import_pai_module("perforatedai.globals_perforatedai")
UPA: Any = import_pai_module("perforatedai.utils_perforatedai")

PAI_SAVE_NAME = "pai"  # PAI rejects path separators; it resolves against cwd
EVAL_SEED = 0  # scripts/report_test_accuracy.py --eval-seed default
N_MFCC_KEEP = 13
IMPROVEMENT_THRESHOLDS = [0.001, 0.0001, 0]  # block "medium"
FORWARD_FUNCTIONS = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class EICNN(nn.Module):
    """The PAI block's 2-conv audio CNN on (B, 1, 32, 101) MFCC input."""

    def __init__(
        self,
        c1: int = 8,
        c2: int = 16,
        num_classes: int = 12,
        c3: int = 0,
        c4: int = 0,
        n_keep: int = N_MFCC_KEEP,
        frames: int = 101,
        noise_std: float = 0.2,
        dropout: float = 0.25,
        pool: str = "same",
    ):
        super().__init__()
        if pool not in ("same", "floor"):
            raise ValueError("pool must be 'same' or 'floor'")
        if c4 and not c3:
            raise ValueError("c4 needs c3")
        self.n_keep, self.noise_std = n_keep, noise_std
        ceil = pool == "same"
        self.input_norm = InputNorm("cmvn", n_keep)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)
        self.drop2 = nn.Dropout(dropout)
        half = (lambda n: math.ceil(n / 2)) if ceil else (lambda n: n // 2)
        self.out_hw = (half(half(n_keep)), half(half(frames)))
        self.depth3 = bool(c3)
        if c3:
            self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)
            self.drop3 = nn.Dropout(dropout)
            self.out_hw = (half(self.out_hw[0]), half(self.out_hw[1]))
        self.depth4 = bool(c4)
        if c4:
            self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)
            self.drop4 = nn.Dropout(dropout)
            self.out_hw = (half(self.out_hw[0]), half(self.out_hw[1]))
        self.fc = nn.Linear((c4 or c3 or c2) * self.out_hw[0] * self.out_hw[1], num_classes)
        # train.py: Xavier/Glorot uniform, zero bias (Keras defaults).
        convs = (self.conv1, self.conv2, self.conv3) if c3 else (self.conv1, self.conv2)
        if c4:
            convs = (*convs, self.conv4)
        for module in (*convs, self.fc):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x[:, 0]
        x = self.input_norm(x[:, : self.n_keep])  # (B, 13, T), per-utterance CMVN
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.relu2(self.conv2(x))))
        if self.depth3:
            x = self.drop3(self.pool3(self.relu3(self.conv3(x))))
        if self.depth4:
            x = self.drop4(self.pool4(self.relu4(self.conv4(x))))
        return self.fc(x.reshape(x.size(0), -1))

    def enforce_max_norm(self, max_value: float = 1.0) -> None:
        """train.py's MaxNorm on the neuron convs (not on dendrite copies)."""
        with torch.no_grad():
            for conv in (self.conv1, self.conv2, getattr(self, "conv3", None),
                         getattr(self, "conv4", None)):
                if conv is None:
                    continue
                weight = getattr(conv, "main_module", conv).weight
                norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
                weight *= torch.clamp(norm, max=max_value) / (norm + 1e-8)


def base_param_count(c1: int, c2: int, pool: str = "same", num_classes: int = 12,
                     c3: int = 0, c4: int = 0) -> int:
    model = EICNN(c1, c2, num_classes=num_classes, pool=pool, c3=c3, c4=c4)
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def materialize(dataset, workers: int, n_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a split's features once (in parallel) and keep n_keep MFCCs."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers)
    xs, ys = [], []
    for features, labels in loader:
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        xs.append(features[:, :n_keep].contiguous())
        ys.append(labels)
    return torch.cat(xs), torch.cat(ys).long()


def load_train_val(data_cfg: dict, seed: int, workers: int, device: torch.device):
    set_seed(seed)
    datasets, label_map = build_datasets(
        data_cfg, augment=False, seed=seed, cache_features=False, splits=(TRAIN, VAL)
    )
    tensors = {}
    for split in (TRAIN, VAL):
        x, y = materialize(datasets[split], workers, N_MFCC_KEEP)
        tensors[split] = (x.to(device), y.to(device))
    return tensors, label_map


def build_test_loader(data_cfg: dict) -> tuple[DataLoader, list[str]]:
    """scripts/report_test_accuracy.py's evaluate(): same seed, build and loader.

    ``cache_features=True`` only keeps the (deterministic) test features in
    memory so budget snapshots do not recompute MFCCs; the tensors are the
    same ones the report script computes on the fly.
    """
    set_seed(EVAL_SEED)
    datasets, label_map = build_datasets(
        data_cfg, augment=False, seed=EVAL_SEED, cache_features=True, splits={TEST}
    )
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    loader = DataLoader(datasets[TEST], batch_size=128, shuffle=False, num_workers=0)
    return loader, label_names


def evaluate_test(model: nn.Module, loader: DataLoader, label_names: list[str],
                  num_keywords: int, device: torch.device) -> dict:
    """Test metrics exactly as scripts/report_test_accuracy.py computes them."""
    model.eval()
    y_true, y_pred = run_inference(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, num_keywords, label_names)
    return {
        "test_accuracy": float(metrics["accuracy"]),
        "test_far": float(metrics["far"]),
        "test_frr": float(metrics["frr"]),
        "test_samples": int(len(y_true)),
        "eval_seed": EVAL_SEED,
    }


# ----------------------------------------------------------------------------
# PAI helpers
# ----------------------------------------------------------------------------


def disarm_pai_debugger() -> None:
    """PAI calls pdb.set_trace() on internal errors; headless, raise instead."""
    import pdb

    def _raise(*_args, **_kwargs):
        raise RuntimeError("PerforatedAI entered pdb.set_trace(); see its message above")

    pdb.set_trace = _raise


def configure_pai(args, device: torch.device) -> None:
    """The block's PAI configuration (train.py main(), lines 779-836)."""
    disarm_pai_debugger()
    GPA.pc.set_device(device)
    GPA.pc.set_use_cuda(device.type == "cuda")
    GPA.pc.set_improvement_threshold(list(IMPROVEMENT_THRESHOLDS))
    GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
    GPA.pc.set_pai_forward_function(FORWARD_FUNCTIONS[args.forward])
    if args.conversion == "all":
        GPA.pc.set_modules_to_perforate([nn.Conv2d, nn.Linear])
        GPA.pc.set_modules_to_track([])
    else:
        GPA.pc.set_modules_to_perforate([nn.Linear])
        GPA.pc.set_modules_to_track([nn.Conv2d])
    dendritic = args.dendrites != "none"
    GPA.pc.set_max_dendrites(args.max_dendrites if dendritic else 0)
    GPA.pc.set_perforated_backpropagation(args.dendrites == "pb")
    GPA.pc.set_dendrite_update_mode(True)
    GPA.pc.set_initial_correlation_batches(40)
    GPA.pc.set_max_dendrite_tries(2)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_configuration_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_n_epochs_to_switch(args.switch_epochs)
    if not dendritic:
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
    GPA.pc.set_verbose(False)
    GPA.pc.set_silent(not args.pai_debug)


def pai_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if type(module).__name__ == "PAINeuronModule"
    }


def dendrite_structure(model: nn.Module) -> dict[str, dict[str, int]]:
    """Per perforated module: dendrites actually present in the graph."""
    out = {}
    for name, module in pai_modules(model).items():
        dendrite_module = getattr(module, "dendrite_module", None)
        row = {
            "dendrite_modules_added": int(getattr(module, "dendrite_modules_added", -1)),
            "dendrites_to_top": len(getattr(module, "dendrites_to_top", [])),
        }
        if dendrite_module is not None:
            row["num_dendrites"] = int(getattr(dendrite_module, "num_dendrites", -1))
            row["dendrite_layers"] = len(getattr(dendrite_module, "layers", []))
        out[name or "<root>"] = row
    return out


def numel_unique(model: nn.Module) -> int:
    """Sum of numel over unique parameters (PAI's GD count_params rule)."""
    unique = {
        p.data_ptr(): p for name, p in model.named_parameters() if "parent_module" not in name
    }
    return int(sum(p.numel() for p in unique.values()))


def clean_param_count(model: nn.Module) -> tuple[int | None, str | None]:
    """Params after the block's export path: blockwise_network + refresh_net."""
    try:
        BPA = import_pai_module("perforatedai.blockwise_perforatedai")
        CPA = import_pai_module("perforatedai.clean_perforatedai")
        clean = copy.deepcopy(model).cpu().eval()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            clean = BPA.blockwise_network(clean)
            clean = CPA.refresh_net(clean)
        return int(sum(p.numel() for p in clean.parameters())), None
    except Exception as exc:  # recorded, not fatal: the count is informational
        return None, f"{type(exc).__name__}: {exc}"


def tracker_vars() -> dict:
    mv = GPA.pai_tracker.member_vars
    keys = (
        "mode", "num_dendrites_added", "num_dendrites_integrated", "num_epochs_run",
        "total_epochs_run", "switch_epochs", "global_best_validation_score",
        "num_dendrite_tries",
    )
    out = {}
    for key in keys:
        try:
            value = mv[key]
        except (KeyError, TypeError):
            continue
        out[key] = list(value) if isinstance(value, (list, tuple)) else value
    return out


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# ----------------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------------


def train_epoch(model, x, y, batch_size, optimizer, criterion, generator):
    """train.py train_epoch(): shuffle, drop_last=False, MaxNorm after step."""
    model.train()
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator).to(x.device)
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device, dtype=torch.long)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        inputs, labels = x[idx], y[idx]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        model.enforce_max_norm(max_value=1.0)
        loss_sum += loss.detach() * labels.shape[0]
        correct += (outputs.detach().argmax(1) == labels).sum()
    return float(loss_sum) / n, float(correct) / n


@torch.no_grad()
def evaluate(model, x, y, criterion, batch_size=1024):
    model.eval()
    n = x.shape[0]
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device, dtype=torch.long)
    for start in range(0, n, batch_size):
        inputs, labels = x[start : start + batch_size], y[start : start + batch_size]
        outputs = model(inputs)
        loss_sum += criterion(outputs, labels) * labels.shape[0]
        correct += (outputs.argmax(1) == labels).sum()
    return float(loss_sum) / n, float(correct) / n


def arm_name(args) -> str:
    width3 = (f"x{args.c3}" if args.c3 else "") + (f"x{args.c4}" if args.c4 else "")
    if args.dendrites == "none":
        restart = f"-r{args.restart_every}" if args.restart_every else ""
        return f"none{restart}-c{args.c1}x{args.c2}{width3}"
    return (
        f"{args.dendrites}-{args.conversion}-max{args.max_dendrites}-{args.forward}"
        f"-sw{args.switch_epochs}-c{args.c1}x{args.c2}{width3}"
    )


def run(args) -> dict:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and not args.overwrite:
        raise SystemExit(f"{out}/result.json exists; pass --overwrite to replace it")
    data_cfg = yaml.safe_load((REPO / args.data_config).read_text())
    device = torch.device(args.device) if args.device != "auto" else get_device()

    started = time.monotonic()
    tensors, label_map = load_train_val(data_cfg, args.seed, args.workers, device)
    (x_train, y_train), (x_val, y_val) = tensors[TRAIN], tensors[VAL]
    data_seconds = time.monotonic() - started
    num_classes = len(label_map)
    print(f"data: train {tuple(x_train.shape)} val {tuple(x_val.shape)} "
          f"classes {num_classes} on {device} in {data_seconds:.1f}s", flush=True)

    set_seed(args.seed)
    model = EICNN(args.c1, args.c2, num_classes=num_classes, c3=args.c3, c4=args.c4,
                  noise_std=args.noise_std, pool=args.pool).to(device)
    base_params = sum(p.numel() for p in model.parameters())
    configure_pai(args, device)

    dendritic = args.dendrites != "none"
    epochs_path = out / "epochs.jsonl"
    epochs_path.write_text("")
    history: list[dict] = []
    restructure_epochs: list[int] = []
    restart_epochs: list[int] = []
    snapshot_epochs = sorted(e for e in args.budget_snapshots if 0 < e < args.epochs_cap)
    snapshots: list[dict] = []
    training_complete = False
    lr_stopped = False
    os.chdir(out)  # PAI writes <cwd>/pai/...; all data is already in memory
    try:
        model = UPA.perforate_model(
            model, doing_pai=dendritic, save_name=PAI_SAVE_NAME, making_graphs=args.graphs
        )
        model.to(device)
        initial_pai_params = int(UPA.count_params(model))
        initial_numel = numel_unique(model)

        GPA.pai_tracker.set_optimizer(torch.optim.Adam)
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        patience = int(GPA.pc.get_n_epochs_to_switch() * 0.75)

        def optim_args():
            return {"params": model.parameters(), "lr": args.lr,
                    "betas": (0.9, 0.999), "eps": 1e-7}

        sched_args = {"mode": "max", "patience": patience}
        opt_args = optim_args()
        optimizer, _ = GPA.pai_tracker.setup_optimizer(model, opt_args, sched_args)
        if args.phase0_optimizer == "detached":
            # train.py:862 -- a second Adam the tracker's scheduler never steps.
            optimizer = torch.optim.Adam(**opt_args)
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(args.seed)

        epoch = 0
        while epoch < args.epochs_cap:
            t0 = time.monotonic()
            train_loss, train_acc = train_epoch(
                model, x_train, y_train, args.batch_size, optimizer, criterion, generator
            )
            val_loss, val_acc = evaluate(model, x_val, y_val, criterion)
            lr_used = current_lr(optimizer)
            mode_before = tracker_vars().get("mode")
            epoch += 1
            GPA.pai_tracker.add_extra_score(train_acc, "Train")
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                val_acc, model
            )
            model.to(device)
            event = None
            if training_complete:
                event = "training_complete"
            elif restructured:
                event = "restructured"
                restructure_epochs.append(epoch)
                optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optim_args(), sched_args)
            elif args.restart_every and epoch % args.restart_every == 0:
                # Fresh optimizer + scheduler, as after a PAI restructure. PAI's
                # setup_optimizer fast-forwards a new scheduler by the epochs
                # since the last switch; a restructure resets that count, so
                # mark this restart as the "last switch" to get the same reset.
                event = "lr_restart"
                restart_epochs.append(epoch)
                mv = GPA.pai_tracker.member_vars
                mv["last_switch"] = mv["num_epochs_run"]
                optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optim_args(), sched_args)
            elif args.stop_lr and current_lr(optimizer) < args.stop_lr:
                event = "lr_stop"
                lr_stopped = True
            tv = tracker_vars()
            row = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr_used,
                "mode": mode_before, "event": event,
                "dendrites_added": tv.get("num_dendrites_added"),
                "pai_params": int(UPA.count_params(model)),
                "seconds": time.monotonic() - t0,
            }
            history.append(row)
            with epochs_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"epoch {epoch:3d} {mode_before} loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val {val_acc:.4f} | lr {lr_used:.2e} | dendrites "
                f"{row['dendrites_added']} | params {row['pai_params']} | "
                f"{row['seconds']:.1f}s{' | ' + event if event else ''}",
                flush=True,
            )
            if epoch in snapshot_epochs and not training_complete:
                # PAI's best_model.pt is the best-validation model so far, i.e.
                # exactly what this run would have selected with cap = epoch.
                name = f"best_at_{epoch:04d}"
                shutil.copyfile(Path(PAI_SAVE_NAME) / "best_model.pt",
                                Path(PAI_SAVE_NAME) / f"{name}.pt")
                snapshots.append({"budget_epochs": epoch, "pai_name": name,
                                  "best_val_acc_pai": tv.get("global_best_validation_score")})
            if training_complete or lr_stopped:
                break

        stop_reason = ("pai_training_complete" if training_complete
                       else "lr_stop" if lr_stopped else "epoch_cap")
        if not training_complete:
            # train_audio_dendritic.py: UPA.load_system(model, 'PAI', 'best_model', True)
            model = UPA.load_system(model, PAI_SAVE_NAME, "best_model", True)
            model.to(device)
        final_tracker = tracker_vars()
        structure = dendrite_structure(model)
        pai_params = int(UPA.count_params(model))
        numel_best = numel_unique(model)
        clean_params, clean_error = clean_param_count(model)
        _, val_acc_selected = evaluate(model, x_val, y_val, criterion)
    finally:
        os.chdir(REPO)

    num_keywords = len(data_cfg["target_keywords"])
    test_loader, label_names = build_test_loader(data_cfg)
    test = evaluate_test(model, test_loader, label_names, num_keywords, device)

    # Budget snapshots are evaluated after the final model, so they cannot
    # affect anything above. Loading restores PAI's tracker from each file.
    for snap in snapshots:
        try:
            os.chdir(out)
            try:
                model = UPA.load_system(model, PAI_SAVE_NAME, snap["pai_name"], True)
            finally:
                os.chdir(REPO)
            model.to(device)
            _, snap["val_acc_selected"] = evaluate(model, x_val, y_val, criterion)
            snap["test_accuracy"] = evaluate_test(
                model, test_loader, label_names, num_keywords, device
            )["test_accuracy"]
            snap["dendrites_structural_max"] = max(
                (row.get("num_dendrites", 0) for row in dendrite_structure(model).values()),
                default=0,
            )
            snap["pai_count"] = int(UPA.count_params(model))
            snap["clean_params"], _ = clean_param_count(model)
        except Exception as exc:  # recorded; the main result is already computed
            snap["error"] = f"{type(exc).__name__}: {exc}"
    best_epoch_row = max(history, key=lambda r: r["val_acc"])
    train_seconds = [r["seconds"] for r in history]
    structural = [row.get("num_dendrites", 0) for row in structure.values()]
    result = {
        "arm": arm_name(args),
        "config": {**vars(args), "out": str(out)},
        "pai_config": {
            "n_epochs_to_switch": args.switch_epochs,
            "scheduler": "ReduceLROnPlateau", "scheduler_mode": "max",
            "scheduler_patience": patience, "optimizer": "Adam", "adam_eps": 1e-7,
            "improvement_threshold": IMPROVEMENT_THRESHOLDS,
            "candidate_weight_initialization_multiplier": 0.01,
            "initial_correlation_batches": 40, "max_dendrite_tries": 2,
            "perforated_backpropagation": args.dendrites == "pb",
            "max_dendrites": args.max_dendrites if dendritic else 0,
            "switch_mode": "DOING_HISTORY" if dendritic else "DOING_NO_SWITCH",
            "doing_pai": dendritic,
        },
        "device": str(device),
        "host": platform.node(),
        "torch": torch.__version__,
        "stop_reason": stop_reason,
        "epochs_run": len(history),
        "epochs_cap": args.epochs_cap,
        "restructure_epochs": restructure_epochs,
        "lr_restart_epochs": restart_epochs,
        "tracker_final": final_tracker,
        "params": {
            "base_model": base_params,
            "pai_count_initial": initial_pai_params,
            "numel_initial": initial_numel,
            "pai_count_best": pai_params,
            "numel_best": numel_best,
            "clean_best": clean_params,
            "clean_error": clean_error,
            "growth_vs_base": (clean_params if clean_params is not None else pai_params)
            - base_params,
        },
        "dendrites": {
            "tracker_num_dendrites_added_best": final_tracker.get("num_dendrites_added"),
            "tracker_num_dendrites_integrated": final_tracker.get("num_dendrites_integrated"),
            "max_num_dendrites_added_during_run": max(
                (r["dendrites_added"] or 0) for r in history
            ),
            "structural_max_per_module": max(structural, default=0),
            "per_module": structure,
        },
        "best_val_acc_pai": final_tracker.get("global_best_validation_score"),
        "val_acc_selected": val_acc_selected,
        "max_val_acc_any_epoch": best_epoch_row["val_acc"],
        "max_val_acc_epoch": best_epoch_row["epoch"],
        **test,
        # Best-validation model at shorter budgets, from the same trajectory
        # (a run capped at E follows it exactly up to E). Budgets at or past
        # epochs_run are the final model above.
        "budget_snapshots": snapshots,
        "seconds_per_epoch_mean": sum(train_seconds) / len(train_seconds),
        "seconds_total": time.monotonic() - started,
        "data_seconds": data_seconds,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in (
        "arm", "stop_reason", "epochs_run", "val_acc_selected", "test_accuracy",
        "seconds_per_epoch_mean")}, indent=None))
    print("params", json.dumps(result["params"]))
    print("dendrites", json.dumps({k: v for k, v in result["dendrites"].items()
                                   if k != "per_module"}))
    return result


def param_table(pool: str) -> None:
    """Print baseline params (c1 = c2/2) for sizing the width ladder."""
    for c2 in (8, 12, 16, 20, 24, 32, 40, 48, 56, 64):
        print(f"c1={c2 // 2:3d} c2={c2:3d} params={base_param_count(c2 // 2, c2, pool)}")
    print(f"c1=  8 c2= 16 params={base_param_count(8, 16, pool)} (PAI default)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dendrites", choices=("none", "pb", "gd"), default="pb")
    p.add_argument("--conversion", choices=("all", "linear"), default="all")
    p.add_argument("--max-dendrites", type=int, default=1)
    p.add_argument("--forward", choices=tuple(FORWARD_FUNCTIONS), default="tanh")
    p.add_argument("--switch-epochs", type=int, default=25, help="n_epochs_to_switch")
    p.add_argument("--c1", type=int, default=8)
    p.add_argument("--c2", type=int, default=16)
    p.add_argument("--c3", type=int, default=0, help="third conv block width (0 = the block's 2 convs)")
    p.add_argument("--c4", type=int, default=0, help="fourth conv block width (needs --c3)")
    p.add_argument("--epochs-cap", type=int, default=200)
    p.add_argument("--restart-every", type=int, default=0,
                   help="none arm only: re-create optimizer+scheduler every K epochs")
    p.add_argument("--stop-lr", type=float, default=0.0,
                   help="none arm only: stop once the scheduler's LR falls below this (0 = off)")
    p.add_argument("--phase0-optimizer", choices=("tracked", "detached"), default="tracked",
                   help="detached = the block's train.py:862 constant-LR first phase")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=False)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--pool", choices=("same", "floor"), default="same",
                   help="same = Keras padding='same' pooling (the block's train.py); "
                        "floor = train_audio_dendritic.py's MaxPool2d")
    p.add_argument("--data-config", default="configs/data/speech_commands_v2_mfcc32_paper.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--budget-snapshots", type=lambda s: [int(x) for x in s.split(",") if x],
                   default=[100, 200, 300, 400, 500, 600, 700],
                   help="loop epochs at which to snapshot the best-so-far model; each is "
                        "test-evaluated at the end (equal-budget comparisons)")
    p.add_argument("--graphs", action="store_true", help="let PAI draw its PNG graphs")
    p.add_argument("--pai-debug", action="store_true", help="do not silence PAI")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--param-table", action="store_true",
                   help="print baseline parameter counts and exit")
    args = p.parse_args(argv)
    if args.param_table:
        return args
    if not args.out:
        p.error("--out is required")
    if args.restart_every and args.dendrites != "none":
        p.error("--restart-every is the none-arm LR-restart control")
    if args.stop_lr and (args.dendrites != "none" or args.restart_every):
        # PB/GD runs can pass LR < 1e-6 before PAI adds a dendrite (which resets LR).
        p.error("--stop-lr is for the none arm without --restart-every")
    if args.c4 and not args.c3:
        p.error("--c4 needs --c3")
    if args.dendrites != "none" and args.max_dendrites < 1:
        p.error("dendritic arms need --max-dendrites >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.param_table:
        param_table(args.pool)
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
