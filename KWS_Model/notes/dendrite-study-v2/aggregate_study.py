#!/usr/bin/env python3
"""Aggregate ``outputs/sparknet-dendritic-study-v2`` into one JSON snapshot.

Run from the repository root:

    uv run python notes/dendrite-study-v2/aggregate_study.py \
        --study-root outputs/sparknet-dendritic-study-v2 \
        --json notes/dendrite-study-v2/aggregate_study.json

This file is read-only with respect to ``outputs/``.  It exists because
``scripts/select_sparknet_arms.py`` -- the study's own aggregator -- refuses to
run until every arm at every width has all five seeds, and only the
``pointwise`` arm has been trained so far.  This script therefore reports the
partial matrix instead of refusing it, and it reports it *paired*: an arm run
and its from-scratch baseline share a seed and a starting checkpoint, so the
per-seed difference removes the between-seed variance that dominates the
raw means.

Everything below reads only committed run artifacts:

* scratch cell  -> ``scratch/c{W}-seed{S}/metrics/summaries.yaml`` (accuracy)
                   ``scratch/c{W}-seed{S}/manifest.yaml``         (wall clock)
* arm cell      -> ``arms/{arm}/c{W}-seed{S}/reports/sparknet_dendritic_prune_experiment.yaml``
                   ``arms/{arm}/c{W}-seed{S}/manifest.yaml``      (wall clock)
* PAI internals -> ``arms/{arm}/c{W}-seed{S}/pai/candidates/*/`` CSVs and
                   ``cycle_metadata.yaml``

Accuracy vocabulary, kept distinct on purpose (all of it is VALIDATION; no run
in this study has touched the test split, and this script never loads data):

``scratch_best``      best validation accuracy of the 200-epoch paper-recipe
                      from-scratch run.  This is the checkpoint every arm at
                      that (width, seed) starts from, so it is the paired
                      reference.
``scratch_final``     validation accuracy at epoch 200, for drift context.
``arm_finetune``      ``candidates[0].baseline.validation_accuracy`` -- the
                      40-epoch identity "prune" fine-tune, i.e. the arm recipe
                      applied with no dendrites yet.  Isolates the cost of
                      changing optimizer/schedule before PAI runs.
``arm_final``         ``candidates[0].dendritic.validation_accuracy`` -- what
                      ``select_sparknet_arms.py`` selects on.
``arm_pai_search``    best validation accuracy inside the PAI search itself.
``arm_zero_dendrite`` PAI's own minimum-parameter row from
                      ``*_best_arch_scores.csv``.  SPARKNET_DENDRITE_FIXES.md
                      is explicit that this is a *within-search* row and not a
                      causal no-dendrite control; it is carried here with that
                      label and never used as the headline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

WIDTHS = (12, 10, 8, 6, 4, 2)
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("pointwise", "fc", "gate_conv", "depthwise", "control")
REPORT_NAME = "reports/sparknet_dendritic_prune_experiment.yaml"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _wall_clock(manifest_path: Path) -> dict[str, Any]:
    """Wall-clock span of a run directory, from its manifest."""
    if not manifest_path.exists():
        return {"started_at": None, "ended_at": None, "seconds": None}
    manifest = _load_yaml(manifest_path) or {}
    start = _parse_ts(manifest.get("started_at"))
    end = _parse_ts(manifest.get("ended_at"))
    seconds = (end - start).total_seconds() if start and end else None
    return {
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "seconds": seconds,
        "status": manifest.get("status"),
        "manifest_seed": manifest.get("seed"),
    }


def _mean_sd(values: list[float]) -> tuple[float | None, float | None, int]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else None
    return mean, sd, len(clean)


def paired_t_test(deltas: list[float]) -> dict[str, Any]:
    """One-sample t-test on paired differences (H0: mean delta == 0).

    scipy is a project dependency, but the fallback keeps this script usable
    without it; both paths compute the same statistic.
    """
    clean = [float(d) for d in deltas if d is not None]
    n = len(clean)
    if n < 2:
        return {"n": n, "t": None, "p_two_sided": None, "method": None}
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    if sd == 0:
        return {"n": n, "t": None, "p_two_sided": None, "method": "zero_variance"}
    t_stat = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        p = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)
        method = "scipy.stats.t"
    except Exception:  # pragma: no cover - only when scipy is unavailable
        p = None
        method = "t_only_no_scipy"
    return {"n": n, "t": float(t_stat), "p_two_sided": p, "method": method}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# scratch baselines
# --------------------------------------------------------------------------
@dataclass
class ScratchCell:
    width: int
    seed: int
    exists: bool
    complete: bool
    best_val_acc: float | None = None
    final_val_acc: float | None = None
    epochs: int | None = None
    phase: str | None = None
    wall_clock: dict[str, Any] = field(default_factory=dict)
    path: str = ""


def collect_scratch(study_root: Path) -> list[ScratchCell]:
    cells: list[ScratchCell] = []
    for width in WIDTHS:
        for seed in SEEDS:
            run = study_root / "scratch" / f"c{width}-seed{seed}"
            summaries = run / "metrics/summaries.yaml"
            cell = ScratchCell(
                width=width,
                seed=seed,
                exists=run.exists(),
                complete=False,
                path=str(run),
                wall_clock=_wall_clock(run / "manifest.yaml"),
            )
            if summaries.exists():
                payload = _load_yaml(summaries) or {}
                phases = payload.get("phases") or {}
                # The scratch sweep writes exactly one phase per run.
                if len(phases) == 1:
                    name, phase = next(iter(phases.items()))
                    cell.phase = name
                    cell.best_val_acc = phase.get("best_val_acc")
                    cell.final_val_acc = phase.get("final_val_acc")
                    cell.epochs = phase.get("completed_epoch")
                    cell.complete = (
                        cell.best_val_acc is not None
                        and (
                            run / "models/checkpoints/paper_replication/best.pt"
                        ).exists()
                        and cell.wall_clock.get("status") == "completed"
                    )
            cells.append(cell)
    return cells


# --------------------------------------------------------------------------
# arm runs
# --------------------------------------------------------------------------
@dataclass
class ArmCell:
    arm: str
    width: int
    seed: int
    exists: bool
    status: str | None
    report_seed: int | None = None
    source_checkpoint: str | None = None
    source_val_acc: float | None = None
    source_params: int | None = None
    source_macs: int | None = None
    finetune_val_acc: float | None = None
    final_val_acc: float | None = None
    pai_search_val_acc: float | None = None
    zero_dendrite_val_acc: float | None = None
    zero_dendrite_params: int | None = None
    deployed_params: int | None = None
    pai_deployed_params: int | None = None
    macs: int | None = None
    param_delta: int | None = None
    mac_delta: int | None = None
    resume_status: str | None = None
    resume_best_val_acc: float | None = None
    max_dendrites: int | None = None
    module_ids: list[str] = field(default_factory=list)
    pai: dict[str, Any] = field(default_factory=dict)
    wall_clock: dict[str, Any] = field(default_factory=dict)
    path: str = ""


def _pai_candidate_dir(run: Path) -> Path | None:
    root = run / "pai/candidates"
    if not root.exists():
        return None
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    return subdirs[0] if len(subdirs) == 1 else (subdirs[0] if subdirs else None)


def collect_pai_internals(run: Path) -> dict[str, Any]:
    """Read the PAI sidecar CSVs and cycle metadata for one arm run.

    The CSVs are named ``<save_name><suffix>.csv``; the ``before_final`` and
    ``_beforeSwitch_N`` variants are mid-search snapshots of the same series,
    so only the unprefixed final copies are read for the headline numbers and
    the snapshots are counted to recover how many switch boundaries the run
    actually crossed.
    """
    candidate = _pai_candidate_dir(run)
    if candidate is None:
        return {"available": False}
    stem = candidate.name
    out: dict[str, Any] = {"available": True, "candidate_dir": str(candidate)}

    # -- switch trajectory ------------------------------------------------
    switches = _read_csv_rows(candidate / f"{stem}switch_epochs.csv")
    out["switch_epochs"] = [
        {"switch": int(r["Switch Number"]), "epoch": int(r["Switch Epoch"])}
        for r in switches
        if r.get("Switch Epoch")
    ]
    out["num_switches"] = len(out["switch_epochs"])

    params = _read_csv_rows(candidate / f"{stem}param_counts.csv")
    out["param_counts"] = [
        {"switch": int(r["Switch Number"]), "params": int(r["Param Count"])}
        for r in params
        if r.get("Param Count")
    ]

    # -- architecture scores: did a larger architecture ever win? ---------
    arch = _read_csv_rows(candidate / f"{stem}_best_arch_scores.csv")
    rows = [
        {
            "params": int(float(r["Param Counts"])),
            "val": float(r["Max Valid Scores"]),
            "train": float(r["Train"]) if r.get("Train") else None,
        }
        for r in arch
        if r.get("Param Counts")
    ]
    out["best_arch_scores"] = rows
    if rows:
        base = min(rows, key=lambda r: r["params"])
        best = max(rows, key=lambda r: r["val"])
        out["arch_min_param_row"] = base
        out["arch_best_row"] = best
        # "Accepted" here means: the architecture that carries the dendrite is
        # the one holding PAI's best validation score.  It is a within-search
        # statement about PAI's own bookkeeping, not a controlled A/B.
        out["dendrite_architecture_won_search"] = bool(best["params"] > base["params"])
        out["arch_score_gain"] = best["val"] - base["val"]
        out["num_architectures_scored"] = len(rows)

    # -- validation trace: NaNs, collapse after a switch ------------------
    scores = _read_csv_rows(candidate / f"{stem}Scores.csv")
    trace: list[tuple[int, float]] = []
    nan_epochs: list[int] = []
    for r in scores:
        raw = (r.get("Validation Scores") or "").strip()
        if not raw:
            continue
        epoch = int(float(r["Epochs"]))
        value = float(raw)
        if math.isnan(value):
            nan_epochs.append(epoch)
            continue
        trace.append((epoch, value))
    out["val_trace_points"] = len(trace)
    out["val_nan_epochs"] = nan_epochs
    if trace:
        out["val_first"] = trace[0][1]
        out["val_last"] = trace[-1][1]
        out["val_max"] = max(v for _, v in trace)
        out["val_min"] = min(v for _, v in trace)
        # Worst dip within 5 epochs after each recorded switch, relative to
        # the last value before it.  A large negative number is the signature
        # of a switch that destabilised the network.
        dips = []
        for entry in out["switch_epochs"]:
            e = entry["epoch"]
            before = [v for ep, v in trace if ep <= e]
            after = [v for ep, v in trace if e < ep <= e + 5]
            if before and after:
                dips.append(
                    {
                        "switch_epoch": e,
                        "val_before": before[-1],
                        "val_min_after_5": min(after),
                        "drop": min(after) - before[-1],
                    }
                )
        out["post_switch_dips"] = dips
        out["worst_post_switch_drop"] = min((d["drop"] for d in dips), default=None)

    # -- how many snapshot generations exist ------------------------------
    out["before_switch_snapshots"] = len(
        sorted(candidate.glob(f"{stem}_beforeSwitch_*Scores.csv"))
    )

    # -- cycle metadata ---------------------------------------------------
    meta_path = candidate / "cycle_metadata.yaml"
    if meta_path.exists():
        meta = _load_yaml(meta_path) or {}
        result = meta.get("result") or {}
        out["cycle_status"] = meta.get("status")
        out["pai_epochs"] = result.get("epochs")
        out["pai_elapsed_seconds"] = result.get("elapsed_seconds")
        trail = result.get("phase_trail") or []
        out["phase_trail"] = [
            {
                "epoch": t.get("epoch"),
                "mode": t.get("mode"),
                "total_params": t.get("total_params"),
                "dendrites_integrated_after": t.get("dendrites_integrated_after"),
                "restructured": t.get("restructured"),
            }
            for t in trail
        ]
        out["mode_sequence"] = "".join(str(t.get("mode") or "?") for t in trail)
        integrated = [
            t.get("dendrites_integrated_after")
            for t in trail
            if t.get("dendrites_integrated_after") is not None
        ]
        out["dendrites_integrated"] = max(integrated) if integrated else 0
        out["num_cycle_checkpoints"] = len(result.get("cycle_checkpoints") or [])
    return out


def collect_arms(study_root: Path, *, with_pai: bool = True) -> list[ArmCell]:
    cells: list[ArmCell] = []
    for arm in ARMS:
        for width in WIDTHS:
            for seed in SEEDS:
                run = study_root / "arms" / arm / f"c{width}-seed{seed}"
                report_path = run / REPORT_NAME
                cell = ArmCell(
                    arm=arm,
                    width=width,
                    seed=seed,
                    exists=run.exists(),
                    status=None,
                    path=str(run),
                    wall_clock=_wall_clock(run / "manifest.yaml"),
                )
                if not report_path.exists():
                    cell.status = "not_started" if not run.exists() else "no_report"
                    cells.append(cell)
                    continue
                report = _load_yaml(report_path) or {}
                cell.status = report.get("status")
                cell.report_seed = report.get("seed")
                source = report.get("source") or {}
                cell.source_checkpoint = source.get("checkpoint")
                cell.source_val_acc = source.get("validation_accuracy")
                cell.source_params = source.get("deployed_params")
                cell.source_macs = source.get("macs")
                pai_cfg = report.get("perforatedai") or {}
                cell.max_dendrites = pai_cfg.get("max_dendrites")
                cell.module_ids = list(pai_cfg.get("module_ids") or [])
                candidates = report.get("candidates") or []
                if candidates:
                    candidate = candidates[0]
                    baseline = candidate.get("baseline") or {}
                    cell.finetune_val_acc = baseline.get("validation_accuracy")
                    dend = candidate.get("dendritic") or {}
                    cell.final_val_acc = dend.get("validation_accuracy")
                    cell.pai_search_val_acc = dend.get("pai_search_validation_accuracy")
                    cell.zero_dendrite_val_acc = dend.get(
                        "zero_dendrite_validation_accuracy"
                    )
                    cell.zero_dendrite_params = dend.get("zero_dendrite_params")
                    cell.deployed_params = dend.get("deployed_params")
                    cell.pai_deployed_params = dend.get("pai_deployed_params")
                    cell.macs = dend.get("macs")
                    resume = dend.get("resume") or {}
                    cell.resume_status = resume.get("status")
                    cell.resume_best_val_acc = resume.get("resume_best_val_acc")
                    comparison = candidate.get("comparison") or {}
                    cell.param_delta = comparison.get("parameter_delta")
                    cell.mac_delta = comparison.get("mac_delta")
                if with_pai and cell.status == "complete":
                    cell.pai = collect_pai_internals(run)
                cells.append(cell)
    return cells


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def summarize(scratch: list[ScratchCell], arms: list[ArmCell]) -> dict[str, Any]:
    scratch_by = {(c.width, c.seed): c for c in scratch}

    scratch_rows = []
    for width in WIDTHS:
        cells = [
            scratch_by[(width, s)] for s in SEEDS if scratch_by[(width, s)].complete
        ]
        best_mean, best_sd, n = _mean_sd([c.best_val_acc for c in cells])
        fin_mean, fin_sd, _ = _mean_sd([c.final_val_acc for c in cells])
        secs = [c.wall_clock.get("seconds") for c in cells]
        sec_mean, _, _ = _mean_sd([s for s in secs if s is not None])
        scratch_rows.append(
            {
                "width": width,
                "n_complete": n,
                "best_val_mean": best_mean,
                "best_val_sd": best_sd,
                "final_val_mean": fin_mean,
                "final_val_sd": fin_sd,
                "per_seed_best": {c.seed: c.best_val_acc for c in cells},
                "wall_clock_mean_seconds": sec_mean,
            }
        )

    # Parameter / MAC counts for a scratch model at each width are recorded by
    # the arm reports (source.deployed_params / source.macs) because the arm
    # profiles the exact checkpoint it loaded.  They are cross-checked for
    # agreement across every arm cell that saw that width.
    scratch_cost: dict[int, dict[str, Any]] = {}
    for width in WIDTHS:
        params = {
            c.source_params
            for c in arms
            if c.width == width and c.source_params is not None
        }
        macs = {
            c.source_macs
            for c in arms
            if c.width == width and c.source_macs is not None
        }
        scratch_cost[width] = {
            "deployed_params": sorted(params)[0]
            if len(params) == 1
            else sorted(params),
            "macs": sorted(macs)[0] if len(macs) == 1 else sorted(macs),
            "consistent": len(params) <= 1 and len(macs) <= 1,
            "n_sources": len([c for c in arms if c.width == width and c.source_params]),
        }

    arm_rows = []
    for arm in ARMS:
        for width in WIDTHS:
            cells = [
                c
                for c in arms
                if c.arm == arm and c.width == width and c.status == "complete"
            ]
            if not cells:
                continue
            fin_mean, fin_sd, n = _mean_sd([c.final_val_acc for c in cells])
            ft_mean, ft_sd, _ = _mean_sd([c.finetune_val_acc for c in cells])
            zd_mean, _, _ = _mean_sd([c.zero_dendrite_val_acc for c in cells])
            par_mean, _, _ = _mean_sd([float(c.deployed_params) for c in cells])
            mac_mean, _, _ = _mean_sd([float(c.macs) for c in cells if c.macs])
            sec_mean, _, _ = _mean_sd(
                [
                    c.wall_clock.get("seconds")
                    for c in cells
                    if c.wall_clock.get("seconds")
                ]
            )

            paired_final: list[float] = []
            paired_finetune: list[float] = []
            paired_dendrite_only: list[float] = []
            per_seed: dict[int, dict[str, Any]] = {}
            for c in cells:
                base = scratch_by.get((c.width, c.seed))
                if base is None or base.best_val_acc is None:
                    continue
                d_final = c.final_val_acc - base.best_val_acc
                d_ft = (
                    c.finetune_val_acc - base.best_val_acc
                    if c.finetune_val_acc is not None
                    else None
                )
                d_dend = (
                    c.final_val_acc - c.zero_dendrite_val_acc
                    if c.zero_dendrite_val_acc is not None
                    else None
                )
                paired_final.append(d_final)
                if d_ft is not None:
                    paired_finetune.append(d_ft)
                if d_dend is not None:
                    paired_dendrite_only.append(d_dend)
                per_seed[c.seed] = {
                    "scratch_best": base.best_val_acc,
                    "arm_finetune": c.finetune_val_acc,
                    "arm_final": c.final_val_acc,
                    "arm_zero_dendrite_row": c.zero_dendrite_val_acc,
                    "delta_final_vs_scratch": d_final,
                    "delta_finetune_vs_scratch": d_ft,
                    "delta_final_vs_zero_dendrite_row": d_dend,
                    "deployed_params": c.deployed_params,
                    "macs": c.macs,
                }

            pf_mean, pf_sd, _ = _mean_sd(paired_final)
            pft_mean, pft_sd, _ = _mean_sd(paired_finetune)
            pdo_mean, pdo_sd, _ = _mean_sd(paired_dendrite_only)
            arm_rows.append(
                {
                    "arm": arm,
                    "width": width,
                    "n_complete": n,
                    "final_val_mean": fin_mean,
                    "final_val_sd": fin_sd,
                    "finetune_val_mean": ft_mean,
                    "finetune_val_sd": ft_sd,
                    "zero_dendrite_row_mean": zd_mean,
                    "mean_deployed_params": par_mean,
                    "mean_macs": mac_mean,
                    "wall_clock_mean_seconds": sec_mean,
                    "paired_delta_final_vs_scratch_mean": pf_mean,
                    "paired_delta_final_vs_scratch_sd": pf_sd,
                    "paired_delta_final_vs_scratch_positive": sum(
                        1 for d in paired_final if d > 0
                    ),
                    "paired_delta_final_vs_scratch_n": len(paired_final),
                    "paired_t_test_final_vs_scratch": paired_t_test(paired_final),
                    "paired_delta_finetune_vs_scratch_mean": pft_mean,
                    "paired_delta_finetune_vs_scratch_sd": pft_sd,
                    "paired_delta_final_vs_zero_dendrite_row_mean": pdo_mean,
                    "paired_delta_final_vs_zero_dendrite_row_sd": pdo_sd,
                    "paired_delta_final_vs_zero_dendrite_row_positive": sum(
                        1 for d in paired_dendrite_only if d > 0
                    ),
                    "paired_t_test_final_vs_zero_dendrite_row": paired_t_test(
                        paired_dendrite_only
                    ),
                    "per_seed": per_seed,
                }
            )

    # "Is a dendrite cheaper than making the network wider?"  Compare the
    # dendrite arm at C{W} against the *scratch* baseline at C{W+2}: two ways
    # of spending parameters for accuracy, unpaired because they are different
    # networks (the seeds are shared but the architectures are not).
    ladder = []
    scratch_row_by_width = {r["width"]: r for r in scratch_rows}
    for row in arm_rows:
        wider = row["width"] + 2
        wider_row = scratch_row_by_width.get(wider)
        if wider_row is None or wider_row["n_complete"] == 0:
            continue
        wider_cost = scratch_cost.get(wider, {})
        base_cost = scratch_cost.get(row["width"], {})
        ladder.append(
            {
                "arm": row["arm"],
                "width": row["width"],
                "dendrite_val_mean": row["final_val_mean"],
                "dendrite_params": row["mean_deployed_params"],
                "dendrite_macs": row["mean_macs"],
                "scratch_same_width_val_mean": scratch_row_by_width[row["width"]][
                    "best_val_mean"
                ],
                "scratch_same_width_params": base_cost.get("deployed_params"),
                "wider_width": wider,
                "wider_scratch_val_mean": wider_row["best_val_mean"],
                "wider_scratch_val_sd": wider_row["best_val_sd"],
                "wider_scratch_params": wider_cost.get("deployed_params"),
                "wider_scratch_macs": wider_cost.get("macs"),
                "dendrite_minus_wider_val": (
                    row["final_val_mean"] - wider_row["best_val_mean"]
                ),
                "dendrite_minus_wider_params": (
                    row["mean_deployed_params"] - wider_cost["deployed_params"]
                    if isinstance(wider_cost.get("deployed_params"), int)
                    else None
                ),
            }
        )

    # The same question asked at *matched parameters* rather than at a whole
    # width step.  A dendrite adds a fraction of a width step, so the fair
    # alternative is not "C{W+2}" but "the point on the scratch width curve
    # that costs the same as C{W} + dendrite".  Linear interpolation between
    # the two bracketing scratch widths is the cheapest defensible way to get
    # that point; it is an interpolation of measured means, labelled as such,
    # and it is only computed when the arm's cost actually falls inside the
    # bracket.
    param_matched = []
    for row in arm_rows:
        lower_w, upper_w = row["width"], row["width"] + 2
        lower = scratch_row_by_width.get(lower_w)
        upper = scratch_row_by_width.get(upper_w)
        p_lo = scratch_cost.get(lower_w, {}).get("deployed_params")
        p_hi = scratch_cost.get(upper_w, {}).get("deployed_params")
        if not (lower and upper and isinstance(p_lo, int) and isinstance(p_hi, int)):
            continue
        if upper["best_val_mean"] is None or lower["best_val_mean"] is None:
            continue
        p = row["mean_deployed_params"]
        inside = p_lo <= p <= p_hi
        frac = (p - p_lo) / (p_hi - p_lo) if p_hi != p_lo else 0.0
        interp = lower["best_val_mean"] + frac * (
            upper["best_val_mean"] - lower["best_val_mean"]
        )
        param_matched.append(
            {
                "arm": row["arm"],
                "width": row["width"],
                "arm_params": p,
                "arm_val_mean": row["final_val_mean"],
                "bracket": [lower_w, upper_w],
                "bracket_params": [p_lo, p_hi],
                "bracket_val_mean": [lower["best_val_mean"], upper["best_val_mean"]],
                "interpolated_scratch_val_at_arm_params": interp,
                "arm_minus_interpolated_scratch": row["final_val_mean"] - interp,
                "within_bracket": inside,
                "scratch_pp_per_100_params_in_bracket": (
                    100.0
                    * (upper["best_val_mean"] - lower["best_val_mean"])
                    / (p_hi - p_lo)
                ),
            }
        )

    return {
        "scratch": scratch_rows,
        "scratch_cost": scratch_cost,
        "arms": arm_rows,
        "width_ladder": ladder,
        "param_matched_width_curve": param_matched,
    }


# --------------------------------------------------------------------------
# historical (pre-study) run families
# --------------------------------------------------------------------------
HISTORICAL_FAMILIES = (
    "outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3",
    "outputs/sparknet-c16-dendritic-prune-no-kd-gate-conv-d3",
    "outputs/sparknet-c12-dendritic-prune-no-kd-fc-only-d3",
    "outputs/sparknet-c12-dendritic-prune-no-kd-unlimited",
)


def collect_historical(
    families: tuple[str, ...] = HISTORICAL_FAMILIES,
) -> list[dict[str, Any]]:
    """Summarise the older prune-from-a-trained-checkpoint families.

    These are NOT comparable to the scratch-start study: different source
    checkpoint, different dendrite cap, different thresholds, different epoch
    budget, and -- per SPARKNET_DENDRITE_FIXES.md -- no recorded downstream
    seed, so their five directories are not five independently seeded runs.
    The per-directory rows are kept so that spread can be shown without being
    called a seed spread.
    """
    out: list[dict[str, Any]] = []
    for family in families:
        root = Path(family)
        if not root.exists():
            out.append({"family": family, "present": False})
            continue
        run_dirs = sorted(root.glob("seed*")) or [root]
        rows: dict[int, list[dict[str, Any]]] = {}
        meta: dict[str, Any] = {}
        statuses: list[str] = []
        recorded_seeds: set[Any] = set()
        for run in run_dirs:
            report_path = run / REPORT_NAME
            if not report_path.exists():
                continue
            report = _load_yaml(report_path) or {}
            statuses.append(str(report.get("status")))
            recorded_seeds.add(report.get("seed"))
            if not meta:
                meta = {
                    "source": report.get("source"),
                    "perforatedai": report.get("perforatedai"),
                }
            for candidate in report.get("candidates") or []:
                dend = candidate.get("dendritic") or {}
                base = candidate.get("baseline") or {}
                cmp_ = candidate.get("comparison") or {}
                rows.setdefault(int(candidate["width"]), []).append(
                    {
                        "dir": run.name,
                        "candidate_status": candidate.get("status"),
                        "final_val": dend.get("validation_accuracy"),
                        "prune_finetune_val": base.get("validation_accuracy"),
                        "deployed_params": dend.get("deployed_params"),
                        "pai_deployed_params": dend.get("pai_deployed_params"),
                        "macs": dend.get("macs"),
                        "gain_vs_prune_finetune": cmp_.get(
                            "validation_accuracy_gain_vs_prune_finetune"
                        ),
                        "gain_vs_pai_zero_row": cmp_.get(
                            "final_validation_accuracy_gain_vs_pai_zero_architecture"
                        ),
                        "above_pruning_curve": cmp_.get(
                            "validation_accuracy_above_pruning_curve"
                        ),
                    }
                )
        widths = {}
        for width, members in sorted(rows.items(), reverse=True):
            done = [m for m in members if m["final_val"] is not None]
            fin_mean, fin_sd, n = _mean_sd([m["final_val"] for m in done])
            ft_mean, _, _ = _mean_sd([m["prune_finetune_val"] for m in done])
            g_mean, _, _ = _mean_sd([m["gain_vs_prune_finetune"] for m in done])
            z_mean, _, _ = _mean_sd([m["gain_vs_pai_zero_row"] for m in done])
            c_mean, _, _ = _mean_sd(
                [
                    m["above_pruning_curve"]
                    for m in done
                    if m["above_pruning_curve"] is not None
                ]
            )
            p_mean, _, _ = _mean_sd(
                [float(m["deployed_params"]) for m in done if m["deployed_params"]]
            )
            widths[width] = {
                "n_directories": len(members),
                "n_with_result": n,
                "final_val_mean": fin_mean,
                "final_val_sd_over_directories": fin_sd,
                "prune_finetune_val_mean": ft_mean,
                "gain_vs_prune_finetune_mean": g_mean,
                "gain_vs_pai_zero_row_mean": z_mean,
                "above_pruning_curve_mean": c_mean,
                "mean_deployed_params": p_mean,
                "per_directory": members,
            }
        out.append(
            {
                "family": family,
                "present": True,
                "run_dirs": [d.name for d in run_dirs],
                "report_statuses": statuses,
                "recorded_seeds": sorted(str(s) for s in recorded_seeds),
                "meta": meta,
                "widths": widths,
            }
        )
    return out


def inventory(scratch: list[ScratchCell], arms: list[ArmCell]) -> dict[str, Any]:
    scratch_missing = [f"c{c.width}-seed{c.seed}" for c in scratch if not c.complete]
    by_status: dict[str, list[str]] = {}
    for c in arms:
        key = c.status or "not_started"
        by_status.setdefault(key, []).append(f"{c.arm}/c{c.width}-seed{c.seed}")
    return {
        "scratch_total": len(scratch),
        "scratch_complete": sum(1 for c in scratch if c.complete),
        "scratch_missing": scratch_missing,
        "arm_total_planned": len(arms),
        "arm_by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "arm_status_counts": {k: len(v) for k, v in sorted(by_status.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=Path("outputs/sparknet-dendritic-study-v2")
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-pai", action="store_true", help="skip the PAI CSV pass")
    parser.add_argument(
        "--no-historical", action="store_true", help="skip the pre-study run families"
    )
    args = parser.parse_args()

    root = args.study_root
    scratch = collect_scratch(root)
    arms = collect_arms(root, with_pai=not args.no_pai)
    payload = {
        "snapshot_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "study_root": str(root.resolve()),
        "inventory": inventory(scratch, arms),
        "summary": summarize(scratch, arms),
        "historical": [] if args.no_historical else collect_historical(),
        "scratch_cells": [asdict(c) for c in scratch],
        "arm_cells": [asdict(c) for c in arms if c.exists],
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.json}")

    inv = payload["inventory"]
    print(f"\nsnapshot {payload['snapshot_utc']}")
    print(f"scratch: {inv['scratch_complete']}/{inv['scratch_total']} complete")
    print(f"arms:    {inv['arm_status_counts']}")

    print("\n== scratch baselines (validation) ==")
    print(
        f"{'width':>6} {'n':>3} {'best mean':>10} {'sd':>8} {'final mean':>11} {'params':>8} {'MACs':>9} {'sec':>7}"
    )
    cost = payload["summary"]["scratch_cost"]
    for row in payload["summary"]["scratch"]:
        c = cost.get(row["width"], {})
        print(
            f"C{row['width']:<5d} {row['n_complete']:>3d} "
            f"{100 * row['best_val_mean']:>9.2f}% {100 * (row['best_val_sd'] or 0):>7.2f}% "
            f"{100 * row['final_val_mean']:>10.2f}% "
            f"{str(c.get('deployed_params')):>8} {str(c.get('macs')):>9} "
            f"{row['wall_clock_mean_seconds'] or 0:>7.0f}"
        )

    for arm in ARMS:
        rows = [r for r in payload["summary"]["arms"] if r["arm"] == arm]
        if not rows:
            continue
        print(f"\n== arm: {arm} (validation) ==")
        print(
            f"{'width':>6} {'n':>3} {'final mean':>11} {'sd':>8} "
            f"{'paired d':>10} {'sd':>8} {'+/n':>6} {'t':>7} {'p':>8} {'params':>8}"
        )
        for row in rows:
            tt = row["paired_t_test_final_vs_scratch"]
            print(
                f"C{row['width']:<5d} {row['n_complete']:>3d} "
                f"{100 * row['final_val_mean']:>10.2f}% {100 * (row['final_val_sd'] or 0):>7.2f}% "
                f"{100 * row['paired_delta_final_vs_scratch_mean']:>+9.2f}pp "
                f"{100 * (row['paired_delta_final_vs_scratch_sd'] or 0):>7.2f}pp "
                f"{row['paired_delta_final_vs_scratch_positive']:>2d}/{row['paired_delta_final_vs_scratch_n']:<3d} "
                f"{(tt['t'] if tt['t'] is not None else float('nan')):>7.2f} "
                f"{(tt['p_two_sided'] if tt['p_two_sided'] is not None else float('nan')):>8.4f} "
                f"{row['mean_deployed_params']:>8.0f}"
            )

    print("\n== dendrite vs. one width step up (unpaired, different architectures) ==")
    print(
        f"{'arm':>10} {'C W+dend':>10} {'params':>8} {'val':>8} | "
        f"{'C W+2 scratch':>14} {'params':>8} {'val':>8} | {'d val':>9} {'d params':>9}"
    )
    for row in payload["summary"]["width_ladder"]:
        print(
            f"{row['arm']:>10} C{row['width']:<9d} {row['dendrite_params']:>8.0f} "
            f"{100 * row['dendrite_val_mean']:>7.2f}% | "
            f"C{row['wider_width']:<13d} {str(row['wider_scratch_params']):>8} "
            f"{100 * row['wider_scratch_val_mean']:>7.2f}% | "
            f"{100 * row['dendrite_minus_wider_val']:>+8.2f}pp "
            f"{(row['dendrite_minus_wider_params'] if row['dendrite_minus_wider_params'] is not None else 0):>+9.0f}"
        )

    print("\n== dendrite vs. the scratch width curve at MATCHED parameters ==")
    print(
        f"{'arm':>10} {'width':>6} {'params':>8} {'arm val':>8} "
        f"{'interp scratch':>15} {'delta':>9} {'bracket':>10} {'in?':>4}"
    )
    for row in payload["summary"]["param_matched_width_curve"]:
        print(
            f"{row['arm']:>10} C{row['width']:<5d} {row['arm_params']:>8.0f} "
            f"{100 * row['arm_val_mean']:>7.2f}% "
            f"{100 * row['interpolated_scratch_val_at_arm_params']:>14.2f}% "
            f"{100 * row['arm_minus_interpolated_scratch']:>+8.2f}pp "
            f"{'C' + str(row['bracket'][0]) + '-C' + str(row['bracket'][1]):>10} "
            f"{str(row['within_bracket']):>4}"
        )

    print("\n== PAI internals (complete arm runs) ==")
    print(
        f"{'run':>26} {'switch':>7} {'epochs':>7} {'modes':>8} {'integ':>6} "
        f"{'archwon':>8} {'archgain':>9} {'worstdip':>9} {'nan':>4}"
    )
    for c in arms:
        if c.status != "complete" or not c.pai.get("available"):
            continue
        p = c.pai
        print(
            f"{c.arm + '/c' + str(c.width) + '-seed' + str(c.seed):>26} "
            f"{p.get('num_switches', 0):>7d} {str(p.get('pai_epochs')):>7} "
            f"{str(p.get('mode_sequence')):>8} {str(p.get('dendrites_integrated')):>6} "
            f"{str(p.get('dendrite_architecture_won_search')):>8} "
            f"{100 * (p.get('arch_score_gain') or 0):>+8.3f}pp "
            f"{100 * (p.get('worst_post_switch_drop') or 0):>+8.2f}pp "
            f"{len(p.get('val_nan_epochs') or []):>4d}"
        )

    if payload["historical"]:
        print("\n== historical (pre-study) families -- NOT comparable to the above ==")
        for fam in payload["historical"]:
            if not fam.get("present"):
                print(f"  {fam['family']}: ABSENT")
                continue
            print(
                f"\n  {fam['family']}\n"
                f"    report statuses={fam['report_statuses']} "
                f"recorded seeds={fam['recorded_seeds']}"
            )
            pai = (fam["meta"] or {}).get("perforatedai") or {}
            src = (fam["meta"] or {}).get("source") or {}
            print(
                f"    placement={pai.get('module_ids')} max_dendrites={pai.get('max_dendrites')} "
                f"thresholds={pai.get('improvement_threshold')}"
            )
            print(
                f"    source={src.get('checkpoint')} C{src.get('channels')} "
                f"val={src.get('validation_accuracy')} params={src.get('deployed_params')}"
            )
            for width, row in fam["widths"].items():
                if not row["n_with_result"]:
                    print(
                        f"    C{width}: no dendritic result ({row['n_directories']} dirs)"
                    )
                    continue

                def pct(value: float | None, digits: int = 2) -> str:
                    return "n/a" if value is None else f"{100 * value:+.{digits}f}pp"

                print(
                    f"    C{width}: n={row['n_with_result']}/{row['n_directories']} "
                    f"final={100 * row['final_val_mean']:.2f}% "
                    f"+/-{100 * (row['final_val_sd_over_directories'] or 0):.2f} (over directories) "
                    f"prune-ft={100 * row['prune_finetune_val_mean']:.2f}% "
                    f"gain_vs_prune-ft={pct(row['gain_vs_prune_finetune_mean'])} "
                    f"gain_vs_pai_zero_row={pct(row['gain_vs_pai_zero_row_mean'], 3)} "
                    f"params={row['mean_deployed_params']:.0f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
