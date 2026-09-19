#!/usr/bin/env python3
"""Does growing one dendrite mid-training move SparkNet's accuracy-vs-cost frontier?

``kws.optimize.sparknet_grow_dendrites`` trains SparkNet from scratch on the
paper recipe and grows one PerforatedAI dendrite partway through: base epochs
1..S train normally (``pre_switch``), then C candidate epochs train only the
dendrite with the base frozen, then base epochs S+1..200 continue
(``post_switch``) with the dendrite integrated, on the same LR schedule.

Before the switch that run is bit-identical to a from-scratch run with the same
seed on the same device, so the existing scratch runs are *paired* controls:
every grow run is compared with the scratch run of its own width and seed, and
the seed-to-seed noise that dominates a 0.2-0.5 pp effect cancels out of the
difference.  This script only reads finished artifacts; it trains nothing and
never reads the test split -- every number here is validation accuracy.

Runs are grouped into **arms**: ``<grow-root>/<arm>/c<W>-seed<S>/``.  An arm is
one configuration -- a placement plus a variant (sham or not, dendrite weight
decay, switch epoch, candidate epochs, dendrite input scale) -- e.g. ``fc``,
``fc-wd1e-3``, ``fc-switch170``, ``fc-sham``.  Every (arm, width) is its own
cell.  A run whose summary names a different arm than its directory, or whose
variant disagrees with the majority of its arm, is INVALID: it is not the
configuration the cell claims to measure.

A **sham** arm integrates the dendrite and then zeroes and freezes it: the
same schedule, candidate phase and optimizer rebuild, with no dendrite
capacity.  Its paired delta is the noise floor of the procedure itself.  Sham
cells are reported (section 3) but never placed against the frontier or gated.

Two questions, answered separately because they can disagree:

* **Paired gain.**  grow - scratch at the same width and seed, in percentage
  points.  ``best`` compares the grow run's best over base epochs S+1..200
  against scratch's best over all 200 epochs -- deliberately conservative,
  since scratch gets to pick from more epochs.  A max over noisy epochs is
  biased upward and ``final`` / ``last5`` / ``last10`` are still only a few
  noisy epochs, so ``last<N>`` (``--window``, default 40) is the mean *paired
  per-epoch* delta over base epochs 201-N..200: grow's post-switch val_acc at
  each epoch minus scratch's at the same epoch.  If the post-switch segment is
  shorter than N (a late switch), the window is all of S+1..200 and is marked
  truncated (``[last30*]``).
* **Frontier.**  A dendrite is not free: it adds parameters and MACs.  The
  honest comparison is not "is C8+dendrite better than C8" but "is it better
  than a *wider* scratch SparkNet of the same cost".  The scratch frontier is
  built from every available scratch seed per width, linearly interpolated at
  the grow cell's deployed cost on each axis (params and MACs are different
  budgets on a microcontroller, so a cell can win on one and lose on the other).
  ``needed_gain_pp`` is what a dendrite must add over its own base width just
  to break even with widening; ``margin_pp`` is the mean paired delta minus
  the all-seed needed gain -- how far the dendrite clears the frontier with
  seed luck cancelled on both sides (``margin_unpaired_pp`` = grow mean minus
  frontier, kept for reference).

Gate, per non-sham cell and axis, on the ``best`` and ``last5`` metrics.  A
mean that clears a small needed gain can be pure noise (with three seeds and
0.3 pp seed noise, a zero-effect dendrite clears a 0.01 pp bar about a quarter
of the time), so the gate uses a one-sided Student-t lower confidence bound on
the mean paired delta: ``lcb = mean - t(confidence, n-1) * sd / sqrt(n)``
(``--confidence``, default 0.90) and ``lcb_margin = lcb - needed (all seeds)``.

* ``INSUFFICIENT SEEDS``: n < ``--min-seeds`` (default 3); margins still shown.
* ``PASS``: lcb_margin > 0 for both best and last5, AND at least ceil(2n/3)
  seeds show a positive paired best-metric delta.
* ``INCONCLUSIVE``: both mean margins > 0 but the PASS conditions fail -- the
  point estimate beats widening, the evidence does not yet.
* ``FAIL``: otherwise.

A run whose own ``checks`` block says the base moved when it should not have,
the optimizer saw base parameters in the candidate phase, momentum was not
fully restored, the clean export does not match the trained model, or (sham
arms) the sham dendrite is not exactly zero, is INVALID and excluded -- its
numbers do not measure what the study claims to measure.

Per-run dendrite diagnostics (``dendrite_diagnostics`` in the summary, else
``reports/grow_diagnostics.yaml``) are shown beside the deltas: ``off-drop`` =
validation accuracy lost when the dendrite's skip weights are zeroed (how much
the base now leans on it), ``lin R²`` = how well an affine map of the
dendrite's own pre-activation explains its contribution (near 1: it is a
second linear layer the base could already express), ``corr`` = correlation
of the contribution with the base module's output.

Usage:

    uv run --env-file .env python scripts/report_sparknet_grow.py \\
        --per-run --json-out outputs/sparknet-grow-dendrites-v3/report.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml
from scipy.stats import t as student_t

from kws.models.registry import build_model
from kws.utils.profile import count_macs, deployed_parameter_count

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "model"
DEFAULT_GROW_ROOT = Path("outputs/sparknet-grow-dendrites-v3")
DEFAULT_SCRATCH_ROOT = Path("outputs/sparknet-dendritic-study-v2/scratch")
DEFAULT_C16_SCRATCH_ROOT = Path("outputs/sparknet-paper-replication")
# Widths trained by the v2 scratch sweep; C16 is the earlier paper replication.
NARROW_WIDTHS = (2, 4, 6, 8, 10, 12)
C16_WIDTH = 16
DEFAULT_SEEDS = tuple(range(5))
RUN_DIR_TEMPLATE = "c{width}-seed{seed}"
RUN_DIR_PATTERN = re.compile(r"^c(?P<width>\d+)-seed(?P<seed>\d+)$")
GROW_SUMMARY = Path("reports/grow_summary.yaml")
GROW_DIAGNOSTICS = Path("reports/grow_diagnostics.yaml")
SCRATCH_PHASE = "paper_replication/sparknet_c{width}_paper"
INPUT_SHAPE = (32, 101)
NUM_CLASSES = 12

METRICS = ("best", "final", "last5", "last10")
WINDOWS = {"last5": 5, "last10": 10}
# Key of the paired last-N per-epoch window delta, beside METRICS in every
# ``deltas_pp`` / ``delta_pp`` block.  N is ``--window``; the key is fixed so
# JSON consumers need not know it.
WINDOW = "window"
CELL_METRICS = METRICS + (WINDOW,)
DEFAULT_WINDOW = 40
FRONTIER_METRICS = ("best", "last5")
AXES = ("params", "macs")
# Where each paired metric lives in grow_summary.yaml's ``results`` block.
GROW_RESULT_KEYS = {
    "best": "best_val_acc_post_switch",
    "final": "final_val_acc",
    "last5": "last5_mean_val_acc",
    "last10": "last10_mean_val_acc",
}
# What makes two runs the same arm.  Summaries without a ``variant`` block
# (written before arms existed) fall back to their schedule and are not shams;
# summaries from before ``dendrite_input_scale`` existed ran with scale 1.
VARIANT_KEYS = (
    "sham", "dendrite_weight_decay", "switch_epoch", "candidate_epochs", "dendrite_input_scale",
)

# One validation clip in ~5,000 is 0.0002; 0.002 is ten clips.  Anything past
# that before the switch means the grow run did not replay the scratch run
# (different device, worker count or a nondeterministic kernel).  The pairing
# is still sound -- same seed, same init -- but the "bit-identical until S"
# premise is not, so it is flagged, not failed.
REPLICATION_TOLERANCE = 0.002
# The clean (deployable) export must reproduce the trained model's outputs.
PARITY_TOLERANCE = 1e-4
# Summary numbers recomputed from the per-epoch JSONL should agree to float
# round-off; a larger gap means the summary and the metrics came from
# different runs or a different epoch window.
CROSS_CHECK_TOLERANCE = 1e-6
# checks.integration_output_max_abs_diff above this is shown in the per-run
# flags (informational: it does not invalidate the run).
INTEGRATION_TOLERANCE = 1e-6
MIN_SEEDS = 3
DEFAULT_CONFIDENCE = 0.90

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INCONCLUSIVE = "INCONCLUSIVE"
GATE_INSUFFICIENT = "INSUFFICIENT SEEDS"
GATE_NO_FRONTIER = "NO FRONTIER"
GATE_INCONSISTENT_COST = "INCONSISTENT COST"
GATE_SHAM = "NOT GATED (sham)"

C16_SCRATCH_NOTE = (
    "C16's scratch control is the earlier paper-replication run, which used a "
    "different dataloader worker count from the grow driver.  Worker count "
    "changes the augmentation RNG stream, so C16 grow runs will NOT replay the "
    "scratch pre-switch trajectory (expect replication flags).  The paired "
    "comparison is still valid: same seed, same initialization, same recipe."
)


class RunUnavailable(Exception):
    """A scratch control that is missing or did not finish."""


# --------------------------------------------------------------------------
# Per-epoch metrics
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON record per non-blank line."""
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: record is not an object")
        records.append(record)
    return records


def _record_val_acc(record: Mapping[str, Any]) -> float | None:
    value = record.get("val_acc", record.get("val_accuracy"))
    return None if value is None else float(value)


def val_acc_by_epoch(
    records: Iterable[Mapping[str, Any]],
    *,
    epoch_key: str = "epoch",
    segments: Sequence[str] | None = None,
) -> dict[int, float]:
    """Map epoch -> validation accuracy.

    A resumed run re-logs the epochs it replays; the later record is the one
    the final weights descend from, so it wins.  ``segments`` restricts a grow
    run's log to the base-training segments -- candidate epochs have no base
    epoch and do not advance the base schedule.
    """
    by_epoch: dict[int, float] = {}
    for record in records:
        if segments is not None and record.get("segment") not in segments:
            continue
        epoch = record.get(epoch_key)
        value = _record_val_acc(record)
        if epoch is None or value is None:
            continue
        by_epoch[int(epoch)] = value
    return by_epoch


def window_mean(val_acc: Mapping[int, float], last_epoch: int, size: int) -> float:
    """Mean validation accuracy over epochs ``last_epoch - size + 1 .. last_epoch``."""
    epochs = range(last_epoch - size + 1, last_epoch + 1)
    missing = [epoch for epoch in epochs if epoch not in val_acc]
    if missing:
        raise KeyError(f"epochs {missing} are missing from the metrics log")
    return statistics.fmean(val_acc[epoch] for epoch in epochs)


def window_delta(
    grow_post: Mapping[int, float],
    scratch: Mapping[int, float],
    switch_epoch: int,
    last_epoch: int,
    size: int,
) -> dict[str, Any] | None:
    """Mean paired per-epoch grow - scratch over the last ``size`` base epochs.

    ``grow_post`` holds only the grow run's post-switch rows (keyed by base
    epoch): pre-switch epochs are the scratch run replayed and would dilute
    the delta with zeros.  When the post-switch segment is shorter than
    ``size`` the window is all of it, and ``truncated`` says so.  Returns
    None when there is no post-switch segment; raises ValueError when an
    epoch in the window is missing from either log.
    """
    post_epochs = last_epoch - switch_epoch
    if post_epochs < 1:
        return None
    used = min(size, post_epochs)
    epochs = range(last_epoch - used + 1, last_epoch + 1)
    missing = [epoch for epoch in epochs if epoch not in grow_post or epoch not in scratch]
    if missing:
        raise ValueError(
            f"base epochs {missing[:5]}{'...' if len(missing) > 5 else ''} are missing "
            "from the grow post-switch or scratch log"
        )
    return {
        "requested": size,
        "epochs": used,
        "first_epoch": epochs[0],
        "last_epoch": last_epoch,
        "truncated": used < size,
        "grow_mean": statistics.fmean(grow_post[epoch] for epoch in epochs),
        "scratch_mean": statistics.fmean(scratch[epoch] for epoch in epochs),
        "delta_pp": statistics.fmean(
            (grow_post[epoch] - scratch[epoch]) * 100.0 for epoch in epochs
        ),
    }


def trajectory_max_abs_diff(
    grow: Mapping[int, float], scratch: Mapping[int, float], switch_epoch: int
) -> tuple[float | None, int]:
    """Largest |grow - scratch| over shared base epochs <= the switch epoch."""
    shared = [
        epoch for epoch in grow if epoch <= switch_epoch and epoch in scratch
    ]
    if not shared:
        return None, 0
    return max(abs(grow[epoch] - scratch[epoch]) for epoch in shared), len(shared)


# --------------------------------------------------------------------------
# Scratch controls
# --------------------------------------------------------------------------


@dataclass
class ScratchRun:
    width: int
    seed: int
    run_dir: str
    epochs: int
    # Fractions, keyed by METRICS.  ``best`` is summaries.yaml's best_val_acc.
    values: dict[str, float]
    best_from_jsonl: float
    warnings: list[str] = field(default_factory=list)
    val_acc: dict[int, float] = field(default_factory=dict, repr=False)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("val_acc")
        return payload


def load_scratch_run(run_dir: Path, width: int, seed: int) -> ScratchRun:
    """Read one finished from-scratch run; raise RunUnavailable if it is not one."""
    run_dir = Path(run_dir)
    summaries_path = run_dir / "metrics" / "summaries.yaml"
    if not summaries_path.exists():
        raise RunUnavailable(f"no {summaries_path}")
    summaries = yaml.safe_load(summaries_path.read_text()) or {}
    phase_key = SCRATCH_PHASE.format(width=width)
    phase = (summaries.get("phases") or {}).get(phase_key)
    if not isinstance(phase, dict):
        raise RunUnavailable(f"{summaries_path} has no phase {phase_key!r}")
    metrics_path = run_dir / str(phase.get("metrics") or f"metrics/{phase_key}.jsonl")
    if not metrics_path.exists():
        raise RunUnavailable(f"no metrics log {metrics_path}")
    val_acc = val_acc_by_epoch(read_jsonl(metrics_path))
    if not val_acc:
        raise RunUnavailable(f"{metrics_path} has no validation records")

    epochs = int(phase.get("epochs") or max(val_acc))
    completed = int(phase.get("completed_epoch") or max(val_acc))
    if completed != epochs or max(val_acc) != epochs:
        raise RunUnavailable(
            f"incomplete: completed {completed} / logged {max(val_acc)} of {epochs} epochs"
        )
    missing = sorted(set(range(1, epochs + 1)) - set(val_acc))
    if missing:
        raise RunUnavailable(f"{metrics_path} is missing epochs {missing[:5]}...")
    if phase.get("best_val_acc") is None:
        raise RunUnavailable(f"{summaries_path} records no best_val_acc")

    best = float(phase["best_val_acc"])
    best_from_jsonl = max(val_acc.values())
    values = {
        "best": best,
        "final": val_acc[epochs],
        **{name: window_mean(val_acc, epochs, size) for name, size in WINDOWS.items()},
    }
    run = ScratchRun(width, seed, str(run_dir), epochs, values, best_from_jsonl, val_acc=val_acc)
    # summaries.yaml and the JSONL are written by the same loop; if they
    # disagree the best checkpoint is not the one the log describes.
    if abs(best - best_from_jsonl) > CROSS_CHECK_TOLERANCE:
        run.warnings.append(
            f"summaries.yaml best_val_acc {best:.6f} != JSONL max {best_from_jsonl:.6f}"
        )
    final = phase.get("final_val_acc")
    if final is not None and abs(float(final) - values["final"]) > CROSS_CHECK_TOLERANCE:
        run.warnings.append(
            f"summaries.yaml final_val_acc {float(final):.6f} != JSONL epoch "
            f"{epochs} {values['final']:.6f}"
        )
    return run


def default_scratch_templates(
    scratch_root: Path, c16_scratch_root: Path
) -> dict[int, str]:
    """Run-directory templates (``{width}``, ``{seed}``) for every scratch width."""
    templates = {
        width: str(Path(scratch_root) / RUN_DIR_TEMPLATE) for width in NARROW_WIDTHS
    }
    templates[C16_WIDTH] = str(Path(c16_scratch_root) / RUN_DIR_TEMPLATE)
    return templates


def collect_scratch(
    templates: Mapping[int, str], seeds: Sequence[int]
) -> tuple[dict[tuple[int, int], ScratchRun], list[dict[str, Any]]]:
    """Load every (width, seed) scratch control; report the ones that are not there."""
    runs: dict[tuple[int, int], ScratchRun] = {}
    unavailable: list[dict[str, Any]] = []
    for width in sorted(templates):
        for seed in seeds:
            run_dir = Path(templates[width].format(width=width, seed=seed))
            try:
                runs[(width, seed)] = load_scratch_run(run_dir, width, seed)
            except (RunUnavailable, OSError, ValueError, yaml.YAMLError) as reason:
                unavailable.append(
                    {"width": width, "seed": seed, "run_dir": str(run_dir), "reason": str(reason)}
                )
    return runs, unavailable


@functools.lru_cache(maxsize=None)
def scratch_cost(width: int) -> dict[str, int]:
    """Deployed parameters and MACs of scratch SparkNet C``width``.

    Measured by building the model from its paper config rather than copied
    from a comment, so a config change cannot silently leave the frontier
    priced at the old architecture.
    """
    config_path = MODEL_CONFIG_DIR / f"sparknet_c{width}_paper.yaml"
    if not config_path.exists():
        raise ValueError(f"no model config for scratch width C{width}: {config_path}")
    model_cfg = yaml.safe_load(config_path.read_text())
    model = build_model(model_cfg, INPUT_SHAPE, NUM_CLASSES)
    return {
        "params": int(deployed_parameter_count(model)),
        "macs": int(count_macs(model, INPUT_SHAPE)),
    }


# --------------------------------------------------------------------------
# Grow runs
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def check_failures(checks: Mapping[str, Any] | None, *, sham: bool = False) -> list[str]:
    """Reasons a grow run's own integrity checks disqualify it (empty = clean).

    Each check guards a premise the paired comparison depends on: the base
    must not move at the neuron->candidate switch or during the candidate
    phase, must not be in the optimizer while it is supposed to be frozen,
    must resume with its full SGD momentum, and the exported clean model must
    be the model that was measured.  A sham run's dendrite must be exactly
    zero at the end, or it is not a noise floor.  A check that was not
    recorded cannot be assumed to have passed.
    """
    if not isinstance(checks, Mapping):
        return ["summary has no checks block"]
    failures: list[str] = []

    def recorded(key: str) -> float | None:
        value = _number(checks.get(key))
        if value is None:
            failures.append(f"checks.{key} not recorded")
        return value

    for key, meaning in (
        ("n_to_p_base_max_abs_change", "base weights changed at the switch"),
        ("candidate_phase_base_max_abs_drift", "base weights drifted while frozen"),
        (
            "base_params_in_optimizer_candidate_phase",
            "base parameters were in the candidate-phase optimizer",
        ),
    ):
        value = recorded(key)
        if value is not None and value > 0:
            failures.append(f"{key}={value:g} > 0 ({meaning})")
    restored = recorded("momentum_buffers_restored")
    expected = recorded("momentum_buffers_expected")
    if restored is not None and expected is not None and restored != expected:
        failures.append(
            f"momentum_buffers_restored={restored:g} != expected {expected:g} "
            "(post-switch SGD did not resume where it left off)"
        )
    for key in ("clean_parity_max_abs_diff_final", "clean_parity_max_abs_diff_best"):
        value = recorded(key)
        if value is not None and value > PARITY_TOLERANCE:
            failures.append(
                f"{key}={value:g} > {PARITY_TOLERANCE:g} (clean export != trained model)"
            )
    if sham:
        value = recorded("sham_skip_weight_max_abs_final")
        if value is not None and value > 0:
            failures.append(
                f"sham_skip_weight_max_abs_final={value:g} > 0 "
                "(the sham dendrite is not zero, so this is not a noise floor)"
            )
    return failures


def resolve_variant(summary: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The run's variant, filled from its schedule where the summary predates it.

    Returns ``(variant, problems)``; a problem means the run is not
    demonstrably the variant it claims (e.g. ``variant.switch_epoch`` says
    170 but the schedule it ran switched at 120) and makes it INVALID.
    """
    schedule = summary.get("schedule") or {}
    fallback = {
        "sham": False,
        "dendrite_weight_decay": _number(schedule.get("dendrite_weight_decay")),
        "switch_epoch": _integer(schedule.get("switch_epoch")),
        "candidate_epochs": _integer(schedule.get("candidate_epochs")),
        "dendrite_input_scale": _number(schedule.get("dendrite_input_scale", 1.0)),
    }
    declared = summary.get("variant")
    problems: list[str] = []
    if declared is not None and not isinstance(declared, Mapping):
        problems.append(f"summary variant is not a mapping: {declared!r}")
        declared = None
    declared = dict(declared or {})

    variant: dict[str, Any] = {}
    sham = declared.get("sham")
    if sham is None:
        sham = fallback["sham"]
    elif not isinstance(sham, bool):
        problems.append(f"variant.sham={sham!r} is not a boolean")
        sham = bool(sham)
    variant["sham"] = sham
    for key, parse in (
        ("dendrite_weight_decay", _number),
        ("switch_epoch", _integer),
        ("candidate_epochs", _integer),
        ("dendrite_input_scale", _number),
    ):
        if declared.get(key) is None:
            variant[key] = fallback[key]
            continue
        value = parse(declared[key])
        if value is None:
            problems.append(f"variant.{key}={declared[key]!r} is not a number")
        elif fallback[key] is not None and not math.isclose(
            value, fallback[key], rel_tol=1e-9, abs_tol=1e-12
        ):
            problems.append(
                f"variant.{key}={value:g} but schedule.{key}={fallback[key]:g} "
                "(the run is not the variant it claims)"
            )
        variant[key] = value
    return variant, problems


def load_diagnostics(
    summary: Mapping[str, Any], run_dir: Path
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """``(block, source, warnings)`` for the run's dendrite diagnostics.

    The summary's ``dendrite_diagnostics`` wins; runs whose diagnostics were
    computed afterwards keep them in ``reports/grow_diagnostics.yaml``.
    Diagnostics are informational, so every failure is a warning.
    """
    block = summary.get("dendrite_diagnostics")
    source = "summary"
    if block is None:
        path = Path(run_dir) / GROW_DIAGNOSTICS
        if not path.exists():
            return None, None, []
        source = str(GROW_DIAGNOSTICS)
        try:
            block = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as error:
            return None, source, [f"unreadable {GROW_DIAGNOSTICS}: {error}"]
    if not isinstance(block, Mapping):
        return None, source, [f"dendrite diagnostics ({source}) is not a mapping"]
    block = dict(block)
    if block.get("error"):
        return block, source, [f"dendrite diagnostics failed ({source}): {block['error']}"]
    return block, source, []


def _module_values(modules: Any, key: str) -> dict[str, float]:
    if not isinstance(modules, Mapping):
        return {}
    values = {}
    for name, stats in modules.items():
        value = _number(stats.get(key)) if isinstance(stats, Mapping) else None
        if value is not None:
            values[str(name)] = value
    return values


def summarize_diagnostics(block: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per export (final / best): on/off accuracy, off-drop pp and per-module stats."""
    if not block or block.get("error"):
        return {}
    summary: dict[str, Any] = {}
    for export in ("final", "best"):
        entry = block.get(export)
        if not isinstance(entry, Mapping):
            continue
        if entry.get("error"):
            summary[export] = {"error": str(entry["error"])}
            continue
        on = _number(entry.get("val_acc_dendrite_on"))
        off = _number(entry.get("val_acc_dendrite_off"))
        modules = entry.get("modules")
        summary[export] = {
            "val_acc_dendrite_on": on,
            "val_acc_dendrite_off": off,
            "off_drop_pp": None if on is None or off is None else (on - off) * 100.0,
            "n_samples": _integer(entry.get("n_samples")),
            "linear_r2": _module_values(modules, "linear_r2_vs_preactivation"),
            "corr": _module_values(modules, "corr_with_base_output"),
            "std_ratio": _module_values(modules, "dendrite_to_base_std_ratio"),
        }
    return summary


@dataclass
class GrowRun:
    arm: str
    width: int
    seed: int
    run_dir: str
    # valid | invalid | incomplete | missing
    status: str = "valid"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    placement: str | None = None
    # The arm the summary names (``arm``, else ``placement`` for old runs).
    arm_declared: str | None = None
    variant: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    switch_epoch: int | None = None
    candidate_epochs: int | None = None
    base_epochs: int | None = None
    # Fractions, keyed by METRICS.
    values: dict[str, float] = field(default_factory=dict)
    best_overall: float | None = None
    deployed: dict[str, int] = field(default_factory=dict)
    base_cost: dict[str, int] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    scratch_values: dict[str, float] = field(default_factory=dict)
    # Keyed by CELL_METRICS once paired; ``window`` is None if not computable.
    deltas_pp: dict[str, float | None] = field(default_factory=dict)
    window: dict[str, Any] | None = None
    replication_max_abs_diff: float | None = None
    replication_epochs: int = 0
    diagnostics: dict[str, Any] | None = None
    diagnostics_source: str | None = None
    diagnostics_summary: dict[str, Any] = field(default_factory=dict)
    val_acc: dict[int, float] = field(default_factory=dict, repr=False)
    post_val_acc: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return f"{self.arm} C{self.width} seed{self.seed}"

    @property
    def sham(self) -> bool:
        return bool(self.variant.get("sham"))

    def invalidate(self, reason: str) -> None:
        self.status = "invalid"
        self.reasons.append(reason)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("val_acc")
        payload.pop("post_val_acc")
        return payload


def load_grow_run(run_dir: Path, arm: str, width: int, seed: int) -> GrowRun:
    """Parse one grow run's summary and per-epoch log; decide its status."""
    run_dir = Path(run_dir)
    run = GrowRun(arm, width, seed, str(run_dir))
    summary_path = run_dir / GROW_SUMMARY
    if not summary_path.exists():
        run.status = "missing"
        run.reasons.append(f"no {GROW_SUMMARY} (still running, or died before writing one)")
        return run
    try:
        summary = yaml.safe_load(summary_path.read_text())
    except yaml.YAMLError as error:
        run.invalidate(f"unreadable {GROW_SUMMARY}: {error}")
        return run
    if not isinstance(summary, dict):
        run.invalidate(f"{GROW_SUMMARY} is not a mapping")
        return run

    # Identity is read before the status check so an arm still in flight can
    # be described in the arms table.
    placement = summary.get("placement")
    run.placement = placement if isinstance(placement, str) and placement else None
    declared_arm = summary.get("arm")
    run.arm_declared = run.placement if declared_arm is None else str(declared_arm)
    run.variant, variant_problems = resolve_variant(summary)

    if summary.get("status") != "complete":
        run.status = "incomplete"
        run.reasons.append(
            str(summary.get("incomplete_reason") or f"status={summary.get('status')!r}")
        )
        return run

    # The directory name is what pairs a run with its control and its cell; a
    # summary that disagrees with it would silently land in the wrong place.
    if run.arm_declared != arm:
        source = "arm" if declared_arm is not None else "arm (fallback: placement)"
        run.invalidate(
            f"summary {source}={run.arm_declared!r} but the arm directory is {arm!r}"
        )
    if run.placement is None:
        run.invalidate(f"summary placement={placement!r} is missing or not a name")
    for problem in variant_problems:
        run.invalidate(problem)
    for key, expected in (("width", width), ("seed", seed)):
        if summary.get(key) != expected:
            run.invalidate(
                f"summary {key}={summary.get(key)!r} but the run directory says {expected!r}"
            )
    if summary.get("selection_split") != "validation" or summary.get(
        "test_split_used"
    ) is not False:
        run.invalidate("summary is not validation-only (selection_split / test_split_used)")

    run.device = summary.get("device")
    schedule = summary.get("schedule") or {}
    run.switch_epoch = _integer(schedule.get("switch_epoch"))
    run.candidate_epochs = _integer(schedule.get("candidate_epochs"))
    run.base_epochs = _integer(schedule.get("base_epochs"))
    for key in ("switch_epoch", "base_epochs"):
        if getattr(run, key) is None:
            run.invalidate(f"schedule.{key} missing")

    results = summary.get("results") or {}
    for metric, key in GROW_RESULT_KEYS.items():
        value = _number(results.get(key))
        if value is None:
            run.invalidate(f"results.{key} missing")
        else:
            run.values[metric] = value
    run.best_overall = _number(results.get("best_val_acc_overall"))

    cost = summary.get("cost") or {}
    for axis in AXES:
        deployed = _integer((cost.get("deployed") or {}).get(axis))
        if deployed is None:
            run.invalidate(f"cost.deployed.{axis} missing")
        else:
            run.deployed[axis] = deployed
        base = _integer((cost.get("base") or {}).get(axis))
        if base is not None:
            run.base_cost[axis] = base

    checks = summary.get("checks")
    run.checks = dict(checks) if isinstance(checks, dict) else {}
    for failure in check_failures(checks, sham=run.sham):
        run.invalidate(failure)
    span = _number(run.checks.get("candidate_phase_val_acc_span"))
    if span is not None and span > 0:
        # The base is frozen and PAI keeps the candidate off the output path,
        # so validation accuracy should be flat across the candidate epochs.
        run.warnings.append(f"validation accuracy moved {span:g} during the candidate phase")

    metrics_rel = (summary.get("artifacts") or {}).get("metrics")
    metrics_path = run_dir / str(metrics_rel) if metrics_rel else None
    if metrics_path is None or not metrics_path.exists():
        run.warnings.append(
            "no per-epoch metrics log; trajectory replication, summary cross-checks "
            "and the last-N window skipped"
        )
    else:
        try:
            records = read_jsonl(metrics_path)
            run.val_acc = val_acc_by_epoch(
                records, epoch_key="base_epoch", segments=("pre_switch", "post_switch")
            )
            run.post_val_acc = val_acc_by_epoch(
                records, epoch_key="base_epoch", segments=("post_switch",)
            )
        except (OSError, ValueError) as error:
            # The per-epoch log only feeds informational checks; a corrupt one
            # must not take the summary's (already validated) results with it.
            run.warnings.append(f"unreadable metrics log {metrics_path}: {error}")

    run.diagnostics, run.diagnostics_source, diagnostic_warnings = load_diagnostics(
        summary, run_dir
    )
    run.warnings.extend(diagnostic_warnings)
    run.diagnostics_summary = summarize_diagnostics(run.diagnostics)
    for export, entry in run.diagnostics_summary.items():
        if "error" in entry:
            run.warnings.append(f"dendrite diagnostics ({export}) failed: {entry['error']}")
    return run


def paired_deltas_pp(
    grow: Mapping[str, float], scratch: Mapping[str, float]
) -> dict[str, float]:
    """grow - scratch per metric, in percentage points."""
    return {metric: (grow[metric] - scratch[metric]) * 100.0 for metric in METRICS}


def _cross_check_grow_summary(run: GrowRun) -> None:
    """Recompute the summary's results from the JSONL and warn on disagreement."""
    last, switch = run.base_epochs, run.switch_epoch
    if not run.val_acc or last is None or switch is None:
        return
    recomputed: dict[str, float] = {}
    post = [value for epoch, value in run.val_acc.items() if switch < epoch <= last]
    if post:
        recomputed["best"] = max(post)
    if last in run.val_acc:
        recomputed["final"] = run.val_acc[last]
    for name, size in WINDOWS.items():
        try:
            recomputed[name] = window_mean(run.val_acc, last, size)
        except KeyError:
            continue
    for metric, value in recomputed.items():
        if metric in run.values and abs(value - run.values[metric]) > CROSS_CHECK_TOLERANCE:
            run.warnings.append(
                f"summary {GROW_RESULT_KEYS[metric]}={run.values[metric]:.6f} but the "
                f"metrics log gives {value:.6f}"
            )


def pair_grow_run(
    run: GrowRun,
    scratch: ScratchRun | None,
    base_cost: Mapping[str, int] | None,
    *,
    window: int = DEFAULT_WINDOW,
) -> None:
    """Attach the paired scratch control; compute deltas, window and replication."""
    if run.status not in ("valid", "invalid"):
        return
    if scratch is None:
        run.invalidate(f"no complete scratch control for C{run.width} seed{run.seed}")
        return
    if run.base_epochs is not None and run.base_epochs != scratch.epochs:
        run.invalidate(
            f"base_epochs {run.base_epochs} != scratch epochs {scratch.epochs}; "
            "final/last-N windows would not line up"
        )
        return
    if any(metric not in run.values for metric in METRICS):
        return
    run.scratch_values = dict(scratch.values)
    run.deltas_pp = dict(paired_deltas_pp(run.values, scratch.values))
    run.window = None
    if run.post_val_acc and run.switch_epoch is not None and run.base_epochs is not None:
        try:
            run.window = window_delta(
                run.post_val_acc, scratch.val_acc, run.switch_epoch, run.base_epochs, window
            )
        except ValueError as error:
            run.warnings.append(f"last{window} window not computed: {error}")
    elif run.val_acc:
        run.warnings.append(
            f"metrics log has no post_switch rows; last{window} window not computed"
        )
    run.deltas_pp[WINDOW] = None if run.window is None else run.window["delta_pp"]
    if run.val_acc and run.switch_epoch is not None:
        diff, shared = trajectory_max_abs_diff(run.val_acc, scratch.val_acc, run.switch_epoch)
        run.replication_max_abs_diff, run.replication_epochs = diff, shared
        if diff is not None and diff > REPLICATION_TOLERANCE:
            run.warnings.append(
                f"pre-switch trajectory differs from scratch by up to {diff * 100:.2f} pp "
                f"(> {REPLICATION_TOLERANCE * 100:.1f} pp) over {shared} epochs -- "
                "nondeterminism or device/worker mismatch; pairing still by seed"
            )
    _cross_check_grow_summary(run)
    if base_cost and run.base_cost and dict(run.base_cost) != dict(base_cost):
        run.warnings.append(
            f"summary cost.base {run.base_cost} != scratch C{run.width} {dict(base_cost)}"
        )


def discover_grow_runs(grow_root: Path) -> list[tuple[str, int, int, Path]]:
    """Every ``<grow-root>/<arm>/c<width>-seed<seed>/`` directory."""
    grow_root = Path(grow_root)
    if not grow_root.is_dir():
        return []
    found = []
    for arm_dir in sorted(
        p for p in grow_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for run_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            match = RUN_DIR_PATTERN.match(run_dir.name)
            if match:
                found.append((arm_dir.name, int(match["width"]), int(match["seed"]), run_dir))
    return found


def _variant_signature(run: GrowRun) -> tuple[Any, ...]:
    return (run.placement, *(run.variant.get(key) for key in VARIANT_KEYS))


def _describe_signature(signature: Sequence[Any]) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in zip(("placement", *VARIANT_KEYS), signature)
    )


def reconcile_arms(runs: Sequence[GrowRun]) -> dict[str, dict[str, Any]]:
    """Describe each arm by the variant its runs agree on; invalidate dissenters.

    Complete runs that name their own arm vote with (placement, variant).  A
    run outvoted by its arm is not the configuration its cell claims and is
    INVALID.  With no strict majority every voter is INVALID: there is no
    way to tell which configuration the arm was meant to be.  An arm with no
    complete run yet is described by any run that has a summary.
    """
    by_arm: dict[str, list[GrowRun]] = defaultdict(list)
    for run in runs:
        by_arm[run.arm].append(run)
    arms: dict[str, dict[str, Any]] = {}
    for arm, members in sorted(by_arm.items()):
        voters = [
            run
            for run in members
            if run.status in ("valid", "invalid") and run.arm_declared == arm and run.variant
        ]
        votes = Counter(_variant_signature(run) for run in voters)
        ranked = votes.most_common()
        majority = None
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            majority = ranked[0][0]
        notes: list[str] = []
        if len(ranked) > 1:
            seen = "; ".join(
                f"{count} run(s) with {_describe_signature(signature)}"
                for signature, count in ranked
            )
            for run in voters:
                signature = _variant_signature(run)
                if majority is None:
                    run.invalidate(f"arm {arm!r} mixes variants with no majority ({seen})")
                elif signature != majority:
                    run.invalidate(
                        f"variant ({_describe_signature(signature)}) disagrees with the "
                        f"majority of arm {arm!r} ({_describe_signature(majority)}; "
                        f"{votes[majority]} of {len(voters)} runs)"
                    )
            if majority is None:
                notes.append("MIXED VARIANTS, no majority: every complete run INVALID")
            else:
                outvoted = len(voters) - votes[majority]
                notes.append(f"MIXED VARIANTS: {outvoted} outvoted run(s) INVALID")
        if majority is None and not ranked:
            described = next(
                (run for run in members if run.variant and run.arm_declared == arm), None
            )
            if described is not None:
                majority = _variant_signature(described)
                notes.append("no complete run yet; variant from an incomplete run")
        mismatched = [
            run
            for run in members
            if run.status in ("valid", "invalid")
            and run.arm_declared is not None
            and run.arm_declared != arm
        ]
        if mismatched:
            notes.append(f"{len(mismatched)} run(s) name a different arm (INVALID)")
        arms[arm] = {
            "arm": arm,
            "placement": None if majority is None else majority[0],
            "variant": None if majority is None else dict(zip(VARIANT_KEYS, majority[1:])),
            "mixed_variants": len(ranked) > 1,
            "variants_seen": [
                {
                    **dict(zip(("placement", *VARIANT_KEYS), signature)),
                    "runs": [run.label for run in voters if _variant_signature(run) == signature],
                }
                for signature, _count in ranked
            ],
            "runs_found": len(members),
            "notes": notes,
        }
    return arms


# --------------------------------------------------------------------------
# Cells, frontier and gate
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def t_quantile(confidence: float, df: int) -> float:
    """One-sided Student-t quantile t(confidence, df)."""
    return float(student_t.ppf(confidence, df))


def lower_confidence_bound(values: Sequence[float], confidence: float) -> float | None:
    """One-sided Student-t lower confidence bound on the mean of ``values``.

    ``mean - t(confidence, n-1) * sd / sqrt(n)``; None for fewer than two
    values, where there is no spread to bound it with.
    """
    n = len(values)
    if n < 2:
        return None
    return statistics.fmean(values) - t_quantile(confidence, n - 1) * statistics.stdev(
        values
    ) / math.sqrt(n)


def summarize_cell(
    arm: str,
    width: int,
    runs: Sequence[GrowRun],
    *,
    placement: str | None = None,
    variant: Mapping[str, Any] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Aggregate the valid, paired runs of one (arm, width) cell."""
    cell: dict[str, Any] = {
        "arm": arm,
        "placement": placement,
        "variant": None if variant is None else dict(variant),
        "sham": bool(variant and variant.get("sham")),
        "width": width,
        "n": len(runs),
        "seeds": sorted(run.seed for run in runs),
        "delta_pp": {},
        "grow_mean": {},
        "scratch_mean_paired": {},
    }
    for metric in CELL_METRICS:
        members = [run for run in runs if run.deltas_pp.get(metric) is not None]
        deltas = [run.deltas_pp[metric] for run in members]
        cell["delta_pp"][metric] = {
            "n": len(deltas),
            "mean": _mean(deltas),
            "sd": _sd(deltas),
            "lcb": lower_confidence_bound(deltas, confidence),
            "n_positive": sum(1 for delta in deltas if delta > 0),
            "by_seed": {run.seed: run.deltas_pp[metric] for run in members},
        }
    for metric in METRICS:
        cell["grow_mean"][metric] = _mean([run.values[metric] for run in runs])
        cell["scratch_mean_paired"][metric] = _mean(
            [run.scratch_values[metric] for run in runs]
        )
    windows = sorted({run.window["epochs"] for run in runs if run.window})
    cell["window"] = {
        "requested": window,
        "epochs": windows,
        "truncated": any(epochs < window for epochs in windows),
    }
    off_drops = [
        run.diagnostics_summary["final"]["off_drop_pp"]
        for run in runs
        if (run.diagnostics_summary.get("final") or {}).get("off_drop_pp") is not None
    ]
    cell["diagnostics"] = {
        "off_drop_pp": {"n": len(off_drops), "mean": _mean(off_drops), "sd": _sd(off_drops)}
    }
    overall = [run.best_overall for run in runs if run.best_overall is not None]
    cell["grow_best_overall_mean"] = _mean(overall)
    costs = {axis: sorted({run.deployed[axis] for run in runs}) for axis in AXES}
    # Every seed of a cell grows the same dendrite on the same architecture;
    # a different deployed size means some run is not the configuration the
    # cell claims to be.
    cell["cost_consistent"] = all(len(values) <= 1 for values in costs.values())
    cell["deployed"] = (
        {axis: costs[axis][0] for axis in AXES}
        if runs and cell["cost_consistent"]
        else None
    )
    cell["deployed_values_seen"] = costs
    return cell


def build_frontier(
    scratch_runs: Mapping[tuple[int, int], ScratchRun],
    costs: Mapping[int, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """One row per scratch width: cost and all-seed mean accuracy per metric."""
    by_width: dict[int, list[ScratchRun]] = defaultdict(list)
    for (width, _seed), run in scratch_runs.items():
        by_width[width].append(run)
    rows = []
    for width in sorted(by_width):
        members = sorted(by_width[width], key=lambda run: run.seed)
        row: dict[str, Any] = {
            "width": width,
            "n_seeds": len(members),
            "seeds": [run.seed for run in members],
            "params": int(costs[width]["params"]),
            "macs": int(costs[width]["macs"]),
        }
        for metric in METRICS:
            values = [run.values[metric] for run in members]
            row[metric] = {"mean": _mean(values), "sd": _sd(values)}
        rows.append(row)
    # Marginal value of widening, for reading break-even before any grow run
    # exists: what the next width buys per unit of added cost.
    for row, wider in zip(rows, rows[1:]):
        gain_pp = (wider["best"]["mean"] - row["best"]["mean"]) * 100.0
        row["slope_to_next"] = {
            "to_width": wider["width"],
            "best_pp_per_100_params": gain_pp / ((wider["params"] - row["params"]) / 100.0),
            "best_pp_per_10k_macs": gain_pp / ((wider["macs"] - row["macs"]) / 10_000.0),
        }
    return rows


def interpolate(points: Sequence[tuple[float, float]], x: float) -> tuple[float, bool]:
    """Piecewise-linear y at ``x``; returns ``(y, extrapolated)``.

    Outside the range the nearest end segment is extended.  A grown dendrite
    always costs more than its own base width, so extrapolation happens only
    past the widest scratch width -- where there is no wider network to
    compare with, and the number is a straight-line guess, hence the flag.
    """
    ordered = sorted((float(px), float(py)) for px, py in points)
    if len(ordered) < 2:
        raise ValueError("a frontier needs at least two widths")
    xs = [px for px, _ in ordered]
    if len(set(xs)) != len(xs):
        raise ValueError(f"frontier has duplicate costs: {xs}")
    if x < xs[0]:
        (x0, y0), (x1, y1), extrapolated = ordered[0], ordered[1], True
    elif x > xs[-1]:
        (x0, y0), (x1, y1), extrapolated = ordered[-2], ordered[-1], True
    else:
        index = max(i for i in range(len(xs) - 1) if xs[i] <= x)
        (x0, y0), (x1, y1), extrapolated = ordered[index], ordered[index + 1], False
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0), extrapolated


def required_positive(n: int) -> int:
    """ceil(2n/3) without floating point: seeds that must show a positive delta."""
    return -(-2 * n // 3)


def gate_verdict(
    n: int,
    n_positive_best: int,
    margin_best_pp: float | None,
    margin_last5_pp: float | None,
    lcb_margin_best_pp: float | None,
    lcb_margin_last5_pp: float | None,
    *,
    min_seeds: int = MIN_SEEDS,
) -> str:
    """INSUFFICIENT SEEDS / NO FRONTIER / PASS / INCONCLUSIVE / FAIL.

    PASS needs n >= min_seeds, both LCB margins > 0 and ceil(2n/3) seeds
    improving on best.  INCONCLUSIVE: both mean margins > 0, PASS not met.
    """
    if n < min_seeds:
        return GATE_INSUFFICIENT
    if margin_best_pp is None or margin_last5_pp is None:
        return GATE_NO_FRONTIER
    if (
        lcb_margin_best_pp is not None
        and lcb_margin_last5_pp is not None
        and lcb_margin_best_pp > 0
        and lcb_margin_last5_pp > 0
        and n_positive_best >= required_positive(n)
    ):
        return GATE_PASS
    if margin_best_pp > 0 and margin_last5_pp > 0:
        return GATE_INCONCLUSIVE
    return GATE_FAIL


def break_even(
    cell: dict[str, Any],
    frontier: Sequence[dict[str, Any]],
    *,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Frontier accuracy at the cell's deployed cost, needed gain and margins, per axis."""
    own = next((row for row in frontier if row["width"] == cell["width"]), None)
    result: dict[str, Any] = {}
    for axis in AXES:
        cost = cell["deployed"][axis] if cell["deployed"] else None
        entry: dict[str, Any] = {"cost": cost}
        for metric in FRONTIER_METRICS:
            frontier_acc = extrapolated = None
            if cost is not None:
                try:
                    frontier_acc, extrapolated = interpolate(
                        [(row[axis], row[metric]["mean"]) for row in frontier], cost
                    )
                except ValueError:
                    pass
            paired = cell["scratch_mean_paired"][metric]
            all_seeds = own[metric]["mean"] if own else None
            grow = cell["grow_mean"][metric]

            def pp(a: float | None, b: float | None) -> float | None:
                return None if a is None or b is None else (a - b) * 100.0

            needed_all = pp(frontier_acc, all_seeds)
            delta_mean = cell["delta_pp"][metric]["mean"]
            lcb = cell["delta_pp"][metric]["lcb"]
            entry[metric] = {
                "frontier_acc": frontier_acc,
                "extrapolated": extrapolated,
                "scratch_own_width_paired": paired,
                "scratch_own_width_all": all_seeds,
                "needed_gain_pp_paired": pp(frontier_acc, paired),
                "needed_gain_pp_all": needed_all,
                "grow_mean": grow,
                # Gate margin: the paired gain the dendrite bought, minus the
                # gain the frontier demands at this cost.  Both halves are
                # seed-luck free -- the delta is paired by seed, and the
                # required gain comes from all-seed scratch means -- whereas
                # grow_mean - frontier_acc would credit a lucky seed subset.
                "margin_pp": (
                    None if delta_mean is None or needed_all is None
                    else delta_mean - needed_all
                ),
                # The same margin from the one-sided lower confidence bound
                # on the mean paired delta: what the gate actually requires.
                "lcb_pp": lcb,
                "lcb_margin_pp": (
                    None if lcb is None or needed_all is None else lcb - needed_all
                ),
                "margin_unpaired_pp": pp(grow, frontier_acc),
            }
        if not cell["cost_consistent"]:
            entry["verdict"] = GATE_INCONSISTENT_COST
        else:
            entry["verdict"] = gate_verdict(
                cell["n"],
                cell["delta_pp"]["best"]["n_positive"],
                entry["best"]["margin_pp"],
                entry["last5"]["margin_pp"],
                entry["best"]["lcb_margin_pp"],
                entry["last5"]["lcb_margin_pp"],
                min_seeds=min_seeds,
            )
        result[axis] = entry
    return result


def noise_floor(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Paired-delta spread of every sham cell: what doing nothing looks like."""
    rows = []
    for cell in cells:
        if not cell["sham"]:
            continue
        rows.append(
            {
                "arm": cell["arm"],
                "placement": cell["placement"],
                "width": cell["width"],
                "n": cell["n"],
                "seeds": cell["seeds"],
                "window": cell["window"],
                "delta_pp": {
                    metric: {
                        key: cell["delta_pp"][metric][key]
                        for key in ("n", "mean", "sd", "n_positive")
                    }
                    | {
                        "max_abs": max(
                            (abs(v) for v in cell["delta_pp"][metric]["by_seed"].values()),
                            default=None,
                        )
                    }
                    for metric in CELL_METRICS
                },
                "off_drop_pp": cell["diagnostics"]["off_drop_pp"],
            }
        )
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(
    grow_root: Path,
    scratch_templates: Mapping[int, str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    grow_seeds: Sequence[int] | None = None,
    scratch_notes: Mapping[int, str] | None = None,
    cost_fn: Callable[[int], Mapping[str, int]] = scratch_cost,
    window: int = DEFAULT_WINDOW,
    confidence: float = DEFAULT_CONFIDENCE,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Everything the markdown and the JSON output are rendered from."""
    seeds = tuple(int(seed) for seed in seeds)
    # The frontier uses every scratch seed; the grow sweep may run fewer, so
    # "missing" is judged against the seeds the sweep was asked to produce.
    expected_grow_seeds = seeds if grow_seeds is None else tuple(int(s) for s in grow_seeds)
    scratch_runs, scratch_unavailable = collect_scratch(scratch_templates, seeds)
    widths = sorted({width for width, _ in scratch_runs})
    costs = {width: dict(cost_fn(width)) for width in widths}
    frontier = build_frontier(scratch_runs, costs)

    runs: list[GrowRun] = []
    seen: dict[tuple[str, int], set[int]] = defaultdict(set)
    for arm, width, seed, run_dir in discover_grow_runs(grow_root):
        runs.append(load_grow_run(run_dir, arm, width, seed))
        seen[(arm, width)].add(seed)
    arms = reconcile_arms(runs)
    for run in runs:
        pair_grow_run(run, scratch_runs.get((run.width, run.seed)), costs.get(run.width), window=window)
    # Only a cell with some seeds on disk is expected to have them all: a
    # one-seed pilot at one width is not a sweep missing every other cell.
    for (arm, width), present in sorted(seen.items()):
        for seed in expected_grow_seeds:
            if seed not in present:
                runs.append(
                    GrowRun(
                        arm,
                        width,
                        seed,
                        str(Path(grow_root) / arm / RUN_DIR_TEMPLATE.format(width=width, seed=seed)),
                        status="missing",
                        reasons=["no run directory"],
                        placement=arms[arm]["placement"],
                    )
                )
    runs.sort(key=lambda run: (run.arm, run.width, run.seed))
    for arm, info in arms.items():
        members = [run for run in runs if run.arm == arm]
        info["valid"] = sum(1 for run in members if run.status == "valid")
        info["widths"] = sorted({run.width for run in members})

    cells = []
    for arm, width in sorted(seen):
        valid = [
            run
            for run in runs
            if run.arm == arm and run.width == width and run.status == "valid"
        ]
        cell = summarize_cell(
            arm,
            width,
            valid,
            placement=arms[arm]["placement"],
            variant=arms[arm]["variant"],
            confidence=confidence,
            window=window,
        )
        if cell["sham"]:
            # A sham is the noise floor, not a candidate: it never faces the
            # frontier or the gate.
            cell["break_even"] = None
            cell["verdicts"] = {axis: GATE_SHAM for axis in AXES}
        else:
            cell["break_even"] = break_even(cell, frontier, min_seeds=min_seeds)
            cell["verdicts"] = {axis: cell["break_even"][axis]["verdict"] for axis in AXES}
        cell["passes_on"] = [axis for axis in AXES if cell["verdicts"][axis] == GATE_PASS]
        cells.append(cell)

    inventory = {
        status: [run.label for run in runs if run.status == status]
        for status in ("valid", "incomplete", "missing", "invalid")
    }
    return {
        "kind": "sparknet_grow_frontier_report",
        "format_version": 2,
        "selection_split": "validation",
        "test_split_used": False,
        "inputs": {
            "grow_root": str(grow_root),
            "grow_root_exists": Path(grow_root).is_dir(),
            "scratch_templates": {str(w): t for w, t in sorted(scratch_templates.items())},
            "seeds": list(seeds),
            "grow_seeds": list(expected_grow_seeds),
            "replication_tolerance": REPLICATION_TOLERANCE,
            "parity_tolerance": PARITY_TOLERANCE,
            "integration_tolerance": INTEGRATION_TOLERANCE,
            "min_seeds": min_seeds,
            "confidence": confidence,
            "window": window,
        },
        "inventory": inventory,
        "arms": [arms[arm] for arm in sorted(arms)],
        "scratch": {
            "runs": [scratch_runs[key].to_json() for key in sorted(scratch_runs)],
            "unavailable": scratch_unavailable,
            "notes": {str(w): note for w, note in sorted((scratch_notes or {}).items())},
        },
        "frontier": frontier,
        "cells": cells,
        "noise_floor": noise_floor(cells),
        "runs": [run.to_json() for run in runs],
    }


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}"


def _pp(value: float | None) -> str:
    return "--" if value is None else f"{value:+.2f}"


def _pct_sd(stat: Mapping[str, float | None]) -> str:
    if stat["mean"] is None:
        return "--"
    sd = "n/a" if stat["sd"] is None else f"{stat['sd'] * 100:.2f}"
    return f"{stat['mean'] * 100:.2f} ± {sd}"


def _delta(stat: Mapping[str, Any]) -> str:
    if stat["mean"] is None:
        return "--"
    sd = "n/a" if stat["sd"] is None else f"{stat['sd']:.2f}"
    return f"{stat['mean']:+.2f} ± {sd} ({stat['n_positive']}/{stat['n']}+)"


def _count(value: int | None) -> str:
    return "--" if value is None else f"{value:,}"


def _general(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _window_mark(epochs: Sequence[int], requested: int) -> str:
    """``[last30*]`` when the post-switch segment is shorter than the window."""
    short = sorted(e for e in epochs if e < requested)
    return f" [last{'/'.join(map(str, short))}*]" if short else ""


def _run_window(run: Mapping[str, Any], requested: int) -> str:
    window = run.get("window")
    if not window:
        return "--"
    return _pp(window["delta_pp"]) + _window_mark([window["epochs"]], requested)


def _cell_window(cell: Mapping[str, Any]) -> str:
    stat = cell["delta_pp"][WINDOW]
    if stat["mean"] is None:
        return "--"
    return _delta(stat) + _window_mark(cell["window"]["epochs"], cell["window"]["requested"])


def _modules(values: Mapping[str, float]) -> str:
    """``fc:0.86`` per module, or the minimum when there are many."""
    if not values:
        return "--"
    if len(values) > 2:
        name = min(values, key=values.get)
        return f"min {values[name]:.2f} ({name.lstrip('.')}; {len(values)} modules)"
    return ", ".join(f"{name.lstrip('.')}:{value:.2f}" for name, value in sorted(values.items()))


def _diag(run: Mapping[str, Any]) -> tuple[str, str, str]:
    final = (run.get("diagnostics_summary") or {}).get("final") or {}
    if not final or "error" in final:
        return "--", "--", "--"
    return _pp(final.get("off_drop_pp")), _modules(final["linear_r2"]), _modules(final["corr"])


def _cell_off_drop(cell: Mapping[str, Any]) -> str:
    stat = cell["diagnostics"]["off_drop_pp"]
    if stat["mean"] is None:
        return "--"
    partial = f" ({stat['n']}/{cell['n']})" if stat["n"] < cell["n"] else ""
    return f"{stat['mean']:+.2f}{partial}"


def _flags(run: Mapping[str, Any]) -> str:
    parts = []
    if run["reasons"]:
        parts.append(f"{len(run['reasons'])} invalid reason(s)")
    if run["warnings"]:
        parts.append(f"{len(run['warnings'])} warning(s)")
    text = "; ".join(parts) + " (see section 1)" if parts else ""
    diff = _number((run.get("checks") or {}).get("integration_output_max_abs_diff"))
    if diff is not None and diff > INTEGRATION_TOLERANCE:
        text = "; ".join(filter(None, (text, f"integration output max abs diff {diff:.2g}")))
    return text or "--"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _arm_notes(arm: Mapping[str, Any]) -> str:
    text = "; ".join(arm["notes"])
    if not text:
        return "--"
    return f"**{text}**" if arm["mixed_variants"] else text


def _label(run: Mapping[str, Any]) -> str:
    return f"{run['arm']} C{run['width']} seed{run['seed']}"


def _axis_name(axis: str) -> str:
    return "MACs" if axis == "macs" else axis


def render_markdown(report: Mapping[str, Any], *, per_run: bool = False) -> str:
    runs = report["runs"]
    by_label = {_label(r): r for r in runs}
    inventory = report["inventory"]
    inputs = report["inputs"]
    window = inputs["window"]
    confidence = inputs["confidence"]
    min_seeds = inputs["min_seeds"]
    window_name = f"Δlast{window}"
    window_header = f"{window_name} pp"
    out: list[str] = [
        "# SparkNet grow-dendrites: accuracy-vs-cost frontier",
        "",
        "Validation split only -- the test split is never read.  Accuracies are "
        "validation %; deltas are percentage points, grow - scratch, paired by "
        "width and seed.",
        "",
        "## 1. Run inventory",
        "",
        f"Grow root: `{inputs['grow_root']}`"
        + ("" if inputs["grow_root_exists"] else " (does not exist -- no grow runs yet)"),
        "",
        f"- complete and valid: {len(inventory['valid'])}",
        f"- incomplete: {len(inventory['incomplete'])}",
        f"- missing (expected seeds {','.join(map(str, inputs['grow_seeds']))}, only in "
        f"(arm, width) cells with at least one run directory): {len(inventory['missing'])}",
        f"- **INVALID (excluded from the gate): {len(inventory['invalid'])}**",
        "",
        "### Arms",
        "",
    ]
    if not report["arms"]:
        out += ["No arm directories with run directories yet.", ""]
    else:
        out += _table(
            (
                "arm", "placement", "sham", "switch_epoch", "candidate_epochs",
                "dendrite_weight_decay", "dendrite_input_scale", "widths", "runs found",
                "valid", "notes",
            ),
            (
                (
                    f"`{arm['arm']}`",
                    _general(arm["placement"]),
                    *(
                        _general((arm["variant"] or {}).get(key))
                        for key in (
                            "sham", "switch_epoch", "candidate_epochs",
                            "dendrite_weight_decay", "dendrite_input_scale",
                        )
                    ),
                    ",".join(f"C{w}" for w in arm["widths"]) or "--",
                    str(arm["runs_found"]),
                    str(arm["valid"]),
                    _arm_notes(arm),
                )
                for arm in report["arms"]
            ),
        )
        out.append("")
    for status, title in (
        ("invalid", "INVALID runs -- excluded from every cell and the gate"),
        ("incomplete", "Incomplete runs"),
        ("missing", "Missing runs"),
    ):
        if inventory[status]:
            out += [f"### {title}", ""]
            out += _table(
                ("run", "reason(s)"),
                ((label, "; ".join(by_label[label]["reasons"])) for label in inventory[status]),
            )
            out.append("")

    scratch = report["scratch"]
    found: dict[int, list[int]] = defaultdict(list)
    for run in scratch["runs"]:
        found[run["width"]].append(run["seed"])
    out += ["### Scratch controls", ""]
    out += _table(
        ("width", "run template", "seeds found", "unavailable"),
        (
            (
                f"C{width}",
                f"`{template}`",
                ",".join(str(s) for s in sorted(found[int(width)])) or "none",
                "; ".join(
                    f"seed{u['seed']}: {u['reason']}"
                    for u in scratch["unavailable"]
                    if u["width"] == int(width)
                )
                or "--",
            )
            for width, template in inputs["scratch_templates"].items()
        ),
    )
    out.append("")
    for width, note in scratch["notes"].items():
        out += [f"> **C{width} note:** {note}", ""]

    warnings = [f"{_label(r)}: {w}" for r in runs for w in r["warnings"]]
    warnings += [
        f"scratch C{r['width']} seed{r['seed']}: {w}" for r in scratch["runs"] for w in r["warnings"]
    ]
    if warnings:
        out += ["### Warnings (informational; runs still counted)", ""]
        out += [f"- {warning}" for warning in warnings]
        out.append("")

    cells = report["cells"]
    truncation_note = (
        f"`[lastK*]`: the arm switches late, leaving only K < {window} post-switch "
        f"base epochs, so {window_name} is the mean over all of them (S+1..200), "
        f"not the last {window}."
    )
    out += ["## 2. Per-cell paired deltas (valid runs only)", ""]
    if not cells:
        out += ["No grow runs found.", ""]
    else:
        out += [
            "Each delta is mean ± sample sd over seeds, then (seeds with a positive "
            "delta / n).  `best` = grow's best over base epochs S+1..200 vs "
            f"scratch's best over all 200.  `{window_name}` = mean over base "
            f"epochs {201 - window}..200 of the paired per-epoch delta (grow "
            "post-switch val_acc - scratch val_acc at the same epoch).  `off-drop` = "
            "mean validation pp lost when the final model's dendrite is switched off.  "
            "`(sham)` rows are noise-floor arms (section 3), never gated.",
            "",
        ]
        out += _table(
            (
                "arm", "width", "n", "seeds", "Δbest pp", "Δfinal pp",
                "Δlast5 pp", "Δlast10 pp", window_header, "grow best %", "grow last5 %",
                "grow best-overall %", "off-drop pp", "deployed params", "deployed MACs",
            ),
            (
                (
                    cell["arm"] + (" (sham)" if cell["sham"] else ""),
                    f"C{cell['width']}",
                    str(cell["n"]),
                    ",".join(str(s) for s in cell["seeds"]) or "--",
                    *(_delta(cell["delta_pp"][m]) for m in METRICS),
                    _cell_window(cell),
                    _pct(cell["grow_mean"]["best"]),
                    _pct(cell["grow_mean"]["last5"]),
                    _pct(cell["grow_best_overall_mean"]),
                    _cell_off_drop(cell),
                    _count(cell["deployed"]["params"]) if cell["deployed"] else "--",
                    _count(cell["deployed"]["macs"]) if cell["deployed"] else "--",
                )
                for cell in cells
            ),
        )
        out.append("")
        if any(cell["window"]["truncated"] for cell in cells):
            out += [truncation_note, ""]
        for cell in cells:
            if not cell["cost_consistent"]:
                out += [
                    f"- **{cell['arm']} C{cell['width']}: deployed cost differs "
                    f"across seeds {cell['deployed_values_seen']} -- cell not gated.**"
                ]
        out.append("")

    floor = report["noise_floor"]
    out += ["## 3. Noise floor (sham arms)", ""]
    if not floor:
        out += [
            "No sham arms yet.  A sham arm runs the identical grow schedule but zeroes "
            "and freezes the dendrite after integration; without one there is no "
            "measured noise floor for the procedure itself, only the gate's "
            "seed-to-seed confidence bound.",
            "",
        ]
    else:
        out += [
            "A sham arm runs the same schedule as its real counterpart -- candidate "
            "phase, integration, optimizer rebuild, momentum carry -- but its dendrite "
            "is zeroed and frozen, so it adds no capacity.  Its paired delta is what "
            "the procedure does on its own (different post-switch batch order, "
            "optimizer-state side effects).  How to read it: a real arm's mean delta at "
            "the same width should sit clearly outside this spread (beyond the sham's "
            "mean ± sd, and larger than its max abs Δ) before it is read as a dendrite "
            "effect.  A sham mean far from 0 means the procedure itself shifts "
            "accuracy, and real arms should be judged against the sham, not against 0.",
            "",
        ]
        out += _table(
            (
                "sham arm", "placement", "width", "n", "Δbest pp", "Δfinal pp",
                "Δlast5 pp", window_header, "max abs Δ pp (any metric)", "off-drop pp",
            ),
            (
                (
                    row["arm"],
                    _general(row["placement"]),
                    f"C{row['width']}",
                    str(row["n"]),
                    *(
                        _delta(row["delta_pp"][m])
                        for m in ("best", "final", "last5")
                    ),
                    (
                        "--"
                        if row["delta_pp"][WINDOW]["mean"] is None
                        else _delta(row["delta_pp"][WINDOW])
                        + _window_mark(row["window"]["epochs"], row["window"]["requested"])
                    ),
                    (
                        "--"
                        if all(row["delta_pp"][m]["max_abs"] is None for m in CELL_METRICS)
                        else f"{max(row['delta_pp'][m]['max_abs'] or 0.0 for m in CELL_METRICS):.2f}"
                    ),
                    (
                        "--"
                        if row["off_drop_pp"]["mean"] is None
                        else f"{row['off_drop_pp']['mean']:+.2f}"
                    ),
                )
                for row in floor
            ),
        )
        out.append("")
        if any(row["window"]["truncated"] for row in floor):
            out += [truncation_note, ""]

    frontier = report["frontier"]
    out += [
        "## 4. Frontier and break-even",
        "",
        "### Scratch frontier (all available scratch seeds per width)",
        "",
    ]
    out += _table(
        (
            "width", "seeds", "params", "MACs", "best val % (mean ± sd)",
            "last5 val % (mean ± sd)", "final val %", "best: pp per +100 params",
            "best: pp per +10k MACs",
        ),
        (
            (
                f"C{row['width']}",
                str(row["n_seeds"]),
                _count(row["params"]),
                _count(row["macs"]),
                _pct_sd(row["best"]),
                _pct_sd(row["last5"]),
                _pct(row["final"]["mean"]),
                (
                    f"{row['slope_to_next']['best_pp_per_100_params']:+.3f} (to C{row['slope_to_next']['to_width']})"
                    if "slope_to_next" in row
                    else "--"
                ),
                (
                    f"{row['slope_to_next']['best_pp_per_10k_macs']:+.3f}"
                    if "slope_to_next" in row
                    else "--"
                ),
            )
            for row in frontier
        ),
    )
    out.append("")

    gated = [cell for cell in cells if not cell["sham"]]
    if gated:
        conf = f"{confidence:g}"
        out += [
            "### Break-even against the scratch frontier",
            "",
            "`frontier` = scratch accuracy linearly interpolated at the cell's "
            "deployed cost on that axis (`*` = extrapolated past the widest scratch "
            "width).  `needed` = frontier - scratch at the cell's own width, over "
            "the cell's paired seeds / over all scratch seeds.  `margin` = mean "
            "paired delta - needed (all seeds).  `LCB margin` = LCB - needed (all "
            f"seeds), where LCB = mean - t({conf}, n-1)·sd/√n is the one-sided "
            f"{confidence * 100:g}% Student-t lower confidence bound on the mean "
            "paired delta (`--` for n < 2).",
            "",
            f"Gate: **PASS** needs n >= {min_seeds}, LCB margin > 0 on best AND "
            "last5, and >= ceil(2n/3) seeds with a positive Δbest.  "
            "**INCONCLUSIVE**: both mean margins > 0 but PASS is not met -- the "
            "point estimate beats widening, the evidence does not yet.  **FAIL**: "
            f"otherwise.  **INSUFFICIENT SEEDS**: n < {min_seeds} (margins still "
            "shown).  Sham arms are not gated.",
            "",
        ]

        def frontier_cell(entry: Mapping[str, Any]) -> str:
            star = "*" if entry["extrapolated"] else ""
            return f"{_pct(entry['frontier_acc'])}{star}"

        def needed(entry: Mapping[str, Any]) -> str:
            return f"{_pp(entry['needed_gain_pp_paired'])} / {_pp(entry['needed_gain_pp_all'])}"

        rows = []
        for cell in gated:
            for axis in AXES:
                entry = cell["break_even"][axis]
                rows.append(
                    (
                        f"{cell['arm']} C{cell['width']}",
                        str(cell["n"]),
                        axis,
                        _count(entry["cost"]),
                        frontier_cell(entry["best"]),
                        needed(entry["best"]),
                        _pct(entry["best"]["grow_mean"]),
                        _pp(entry["best"]["margin_pp"]),
                        _pp(entry["best"]["lcb_margin_pp"]),
                        frontier_cell(entry["last5"]),
                        needed(entry["last5"]),
                        _pct(entry["last5"]["grow_mean"]),
                        _pp(entry["last5"]["margin_pp"]),
                        _pp(entry["last5"]["lcb_margin_pp"]),
                        f"{cell['delta_pp']['best']['n_positive']}/{required_positive(cell['n'])}",
                        f"**{entry['verdict']}**",
                    )
                )
        out += _table(
            (
                "cell", "n", "axis", "deployed cost", "frontier best %",
                "needed best pp (paired / all)", "grow best %", "margin best pp",
                "LCB margin best pp", "frontier last5 %",
                "needed last5 pp (paired / all)", "grow last5 %", "margin last5 pp",
                "LCB margin last5 pp", "+seeds / required", "verdict",
            ),
            rows,
        )
        out.append("")
    elif not cells:
        out += [
            "No grow cells to place against the frontier yet.  Read the slope "
            "columns as break-even rates: a dendrite on width C must add at least "
            "(slope × its added cost) over C's own accuracy to beat widening.",
            "",
        ]
    else:
        out += ["Only sham cells so far; nothing to place against the frontier.", ""]

    if cells:
        out += ["### Gate verdicts", ""]
        for cell in cells:
            if cell["sham"]:
                out.append(
                    f"- {cell['arm']} C{cell['width']} (n={cell['n']}): not gated "
                    "(sham arm -- noise floor, section 3)"
                )
                continue
            verdicts = cell["verdicts"]
            if cell["passes_on"]:
                summary = "PASS on " + " and ".join(_axis_name(axis) for axis in cell["passes_on"])
                failed = [axis for axis in AXES if axis not in cell["passes_on"]]
                if failed:
                    summary += "; " + "; ".join(
                        f"{verdicts[axis]} on {_axis_name(axis)}" for axis in failed
                    )
            elif len(set(verdicts.values())) == 1:
                summary = next(iter(verdicts.values()))
            else:
                summary = "; ".join(f"{verdicts[axis]} on {_axis_name(axis)}" for axis in AXES)
            out.append(f"- **{cell['arm']} C{cell['width']}** (n={cell['n']}): {summary}")
        out.append("")

    if per_run:
        out += ["## 5. Per-run detail", ""]
        detail = [r for r in runs if r["deltas_pp"]]
        if not detail:
            out += ["No paired runs.", ""]
        else:
            out += [
                "`off-drop` = 100·(val acc with dendrite - with its skip weights "
                "zeroed), final model.  `lin R²` = affine fit of the dendrite's "
                "contribution on its own pre-activation (near 1: a second linear "
                "layer).  `corr` = correlation of the contribution with the base "
                "module's output.  Per module; the minimum when there are more than two.",
                "",
            ]
            out += _table(
                (
                    "run", "status", "device", "S", "Δbest pp", "Δfinal pp",
                    "Δlast5 pp", "Δlast10 pp", window_header, "grow best-overall %",
                    "pre-switch max abs diff pp", "deployed params / MACs",
                    "off-drop pp", "lin R²", "corr", "flags",
                ),
                (
                    (
                        _label(r) + (" (sham)" if (r.get("variant") or {}).get("sham") else ""),
                        r["status"].upper() if r["status"] == "invalid" else r["status"],
                        str(r["device"] or "--"),
                        str(r["switch_epoch"]),
                        *(_pp(r["deltas_pp"][m]) for m in METRICS),
                        _run_window(r, window),
                        _pct(r["best_overall"]),
                        (
                            "--"
                            if r["replication_max_abs_diff"] is None
                            else f"{r['replication_max_abs_diff'] * 100:.2f}"
                            + (" (!)" if r["replication_max_abs_diff"] > REPLICATION_TOLERANCE else "")
                        ),
                        f"{_count(r['deployed'].get('params'))} / {_count(r['deployed'].get('macs'))}",
                        *_diag(r),
                        _flags(r),
                    )
                    for r in detail
                ),
            )
            out.append("")
            if any((r.get("window") or {}).get("truncated") for r in detail):
                out += [truncation_note, ""]
            out += ["INVALID rows are shown for diagnosis only; they are in no cell mean.", ""]
    return "\n".join(out)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write atomically so a reader never sees half a report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_scratch_override(text: str) -> tuple[int, str]:
    """``WIDTH=PATH_TEMPLATE`` -> (width, template); the template needs ``{seed}``."""
    width, separator, template = text.partition("=")
    if not separator or not width.strip().isdigit() or not template:
        raise ValueError(f"--scratch-run expects WIDTH=PATH_TEMPLATE, got {text!r}")
    if "{seed}" not in template:
        raise ValueError(f"--scratch-run template must contain {{seed}}: {template!r}")
    return int(width), template


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--grow-root",
        type=Path,
        default=DEFAULT_GROW_ROOT,
        help=f"Grow runs, laid out <root>/<arm>/c<W>-seed<S>/ (default: {DEFAULT_GROW_ROOT}).",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help=f"Scratch controls for C{','.join(map(str, NARROW_WIDTHS))} (default: {DEFAULT_SCRATCH_ROOT}).",
    )
    parser.add_argument(
        "--c16-scratch-root",
        type=Path,
        default=DEFAULT_C16_SCRATCH_ROOT,
        help=f"Scratch controls for C16 (default: {DEFAULT_C16_SCRATCH_ROOT}).",
    )
    parser.add_argument(
        "--scratch-run",
        action="append",
        default=[],
        metavar="WIDTH=PATH_TEMPLATE",
        help=(
            "Override (or add) one width's scratch run directory; the template "
            "may use {width} and must use {seed}. Repeatable."
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Scratch seeds used for the frontier and the paired controls (default 0-4).",
    )
    parser.add_argument(
        "--grow-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help=(
            "Seeds each grow (arm, width) cell is expected to have; absent ones are "
            "listed as missing, but only in cells with at least one run directory "
            "(default 0 1 2 3 4)."
        ),
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=(
            "N for the paired last-N per-epoch window delta (default "
            f"{DEFAULT_WINDOW}); truncated to the post-switch epochs when shorter."
        ),
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=(
            "One-sided confidence of the Student-t lower bound the gate uses "
            f"(default {DEFAULT_CONFIDENCE:g})."
        ),
    )
    parser.add_argument(
        "--min-seeds", type=int, default=MIN_SEEDS,
        help=f"Valid seeds a cell needs before it is gated (default {MIN_SEEDS}).",
    )
    parser.add_argument("--per-run", action="store_true", help="Also print the per-run table.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the full result as JSON.")
    args = parser.parse_args(argv)

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if args.window < 1:
        parser.error("--window must be >= 1")
    # Below 0.5 the one-sided t quantile is negative and the "lower" bound
    # sits above the mean.
    if not 0.5 <= args.confidence < 1.0:
        parser.error("--confidence must be in [0.5, 1)")
    if args.min_seeds < 1:
        parser.error("--min-seeds must be >= 1")
    templates = default_scratch_templates(args.scratch_root, args.c16_scratch_root)
    notes = {C16_WIDTH: C16_SCRATCH_NOTE}
    for text in args.scratch_run:
        try:
            width, template = parse_scratch_override(text)
        except ValueError as error:
            parser.error(str(error))
        templates[width] = template
        # An overridden C16 control is not the paper-replication run the note
        # describes; whoever supplied it knows how it was trained.
        notes.pop(width, None)

    try:
        report = build_report(
            args.grow_root, templates, args.seeds,
            grow_seeds=args.grow_seeds, scratch_notes=notes,
            window=args.window, confidence=args.confidence, min_seeds=args.min_seeds,
        )
    except (ValueError, OSError, yaml.YAMLError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not report["frontier"]:
        parser.error(
            "no complete scratch controls found; check --scratch-root / --c16-scratch-root"
        )
    print(render_markdown(report, per_run=args.per_run))
    if args.json_out is not None:
        _write_json(args.json_out, report)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
