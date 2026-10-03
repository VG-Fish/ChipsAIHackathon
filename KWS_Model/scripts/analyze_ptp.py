"""Offline "Pruning Then Perforating" analysis of SparkNet PTP runs.

Reads the aggregate report and the per-epoch PAI logs written by
``kws.optimize.sparknet_dendritic_prune_experiment`` with
``configs/train/sparknet_ptp_paper.yaml`` (``pai_eval_splits: [test, train]``)
and computes, per run (seed x prune rate):

* dendrite count per epoch from parameter-count deltas (never PAI's
  integration counter): ``round((n(e) - n(first epoch)) / step)``, where
  ``n(e)`` is the parameter count of the architecture that produced epoch
  ``e``'s scores and ``step`` is one classifier copy (inferred from the
  smallest positive delta, or ``--dendrite-step``);
* budget-N (N = 0..3): the epoch with the best validation accuracy among
  epochs with at most N dendrites (earliest on ties), its test accuracy,
  clean train accuracy and true parameter count
  (``pruned_params + n(e) - n(first epoch)``);
* the zero-dendrite reference Z: per prune rate, the seed-mean budget-0 test
  accuracy placed at that rate's pruned parameter count, linearly
  interpolated in log10(params) and extended flat beyond the end points;
* parameter-matched gain ``m = A - Zhat(n_final)``, the raw gain
  ``g = A_final - A_pre``, the parameter cost
  ``c = Zhat(n_final) - Zhat(n_start)`` and the pre-dendrite term
  ``m_pre = A_pre - Zhat(n_start)`` (so ``m = m_pre + g - c``);
* per rate: mean, 95% t-interval and two-sided one-sample t-test p of the
  gain, plus mean raw gain, cost, parameters added and budget-0 train acc.

``--scratch-frontier <json>`` (a list of ``{"params": n, "test_acc": a}``)
additionally reports the gain against a scratch-trained width curve,
interpolated the same way.

Usage::

    uv run python scripts/analyze_ptp.py outputs/sparknet-c16-ptp/seed* \\
        [--budget 3] [--json out.json] [--scratch-frontier scratch.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPORT_RELATIVE = Path("reports") / "sparknet_dendritic_prune_experiment.yaml"
MAX_BUDGET = 3


# --------------------------------------------------------------------------
# Per-epoch records -> dendrite counts and budgets
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Epoch records, the last record winning when a resume rewrote an epoch."""
    by_epoch: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_epoch[int(record["epoch"])] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def evaluated_parameter_counts(records: Sequence[Mapping[str, Any]]) -> list[int]:
    """Parameter count of the architecture that produced each epoch's scores.

    ``evaluated_parameter_count`` is logged when ``pai_eval_splits`` is on.
    Older logs only carry ``parameter_count``, read *after* PAI's validation
    step, i.e. the next epoch's architecture; the previous epoch's value is
    then the evaluated one.
    """
    counts: list[int] = []
    for index, record in enumerate(records):
        if record.get("evaluated_parameter_count") is not None:
            counts.append(int(record["evaluated_parameter_count"]))
        elif index == 0:
            counts.append(int(record["parameter_count"]))
        else:
            counts.append(int(records[index - 1]["parameter_count"]))
    return counts


def infer_dendrite_step(counts: Sequence[int]) -> int | None:
    """Smallest positive parameter increase over the run's starting count."""
    if not counts:
        return None
    start = counts[0]
    deltas = sorted({c - start for c in counts if c > start})
    return deltas[0] if deltas else None


def dendrite_counts(
    counts: Sequence[int], step: int | None, *, tolerance: float = 1e-6
) -> list[int]:
    if not counts:
        return []
    start = counts[0]
    if step is None:
        if any(c != start for c in counts):
            raise ValueError("parameter count changes but no dendrite step is known")
        return [0] * len(counts)
    result: list[int] = []
    for count in counts:
        ratio = (count - start) / step
        nearest = round(ratio)
        if abs(ratio - nearest) > tolerance or nearest < 0:
            raise ValueError(
                f"parameter delta {count - start} is not a non-negative multiple "
                f"of the dendrite step {step}"
            )
        result.append(int(nearest))
    return result


@dataclass
class BudgetPoint:
    budget: int
    epoch: int
    dendrites: int
    params: int
    val_acc: float
    test_acc: float | None
    train_acc: float | None
    running_train_acc: float | None


def select_budgets(
    records: Sequence[Mapping[str, Any]],
    dendrites: Sequence[int],
    params: Sequence[int],
    *,
    max_budget: int = MAX_BUDGET,
) -> dict[int, BudgetPoint]:
    """Best-validation epoch among epochs with at most N dendrites, N=0..max."""
    out: dict[int, BudgetPoint] = {}
    for budget in range(max_budget + 1):
        best: int | None = None
        for index, record in enumerate(records):
            if dendrites[index] > budget:
                continue
            if best is None or float(record["val_acc"]) > float(records[best]["val_acc"]):
                best = index
        if best is None:
            continue
        record = records[best]
        out[budget] = BudgetPoint(
            budget=budget,
            epoch=int(record["epoch"]),
            dendrites=int(dendrites[best]),
            params=int(params[best]),
            val_acc=float(record["val_acc"]),
            test_acc=_optional_float(record.get("test_accuracy")),
            train_acc=_optional_float(record.get("train_eval_accuracy")),
            running_train_acc=_optional_float(record.get("train_accuracy")),
        )
    return out


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


# --------------------------------------------------------------------------
# Reference curves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LogCurve:
    """Accuracy vs log10(params), linear between points, flat beyond the ends."""

    points: tuple[tuple[float, float], ...]  # (params, accuracy), params ascending

    @classmethod
    def from_points(cls, points: Iterable[tuple[float, float]]) -> "LogCurve":
        grouped: dict[float, list[float]] = defaultdict(list)
        for params, accuracy in points:
            if params <= 0:
                raise ValueError("reference curve parameter counts must be positive")
            grouped[float(params)].append(float(accuracy))
        if not grouped:
            raise ValueError("reference curve needs at least one point")
        return cls(tuple(
            (params, sum(values) / len(values)) for params, values in sorted(grouped.items())
        ))

    def __call__(self, params: float) -> float:
        xs = [math.log10(p) for p, _ in self.points]
        ys = [a for _, a in self.points]
        x = math.log10(float(params))
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            if x0 <= x <= x1:
                return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@dataclass
class Run:
    run_dir: str
    seed: int | None
    width: int
    prune_rate: float | None
    pruned_params: int
    step: int | None
    epochs: int
    finetune_epochs: int | None
    budgets: dict[int, BudgetPoint]
    gains: dict[str, Any] = field(default_factory=dict)


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    checkpoint = str((candidate.get("dendritic") or {}).get("checkpoint") or "")
    parts = Path(checkpoint).parts
    if "candidates" in parts:
        index = parts.index("candidates")
        if index + 1 < len(parts):
            return parts[index + 1]
    return f"sparknet_c{int(candidate['width'])}_multilayer"


def load_runs(
    run_dirs: Iterable[str | Path],
    *,
    dendrite_step: int | None = None,
    max_budget: int = MAX_BUDGET,
) -> list[Run]:
    runs: list[Run] = []
    for run_dir in map(Path, run_dirs):
        report_path = run_dir / REPORT_RELATIVE
        if not report_path.exists():
            print(f"skip {run_dir}: no {REPORT_RELATIVE}", file=sys.stderr)
            continue
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        for candidate in report.get("candidates") or []:
            width = int(candidate["width"])
            name = _candidate_name(candidate)
            log = run_dir / "metrics" / "sparsity" / name / "pai.jsonl"
            if not log.exists():
                print(f"skip {run_dir} C{width}: no PAI log {log}", file=sys.stderr)
                continue
            records = load_jsonl(log)
            if not records:
                continue
            group = candidate.get("group_prune") or {}
            pruned_params = int(
                group.get("pruned_params")
                or candidate.get("baseline", {}).get("deployed_params")
            )
            counts = evaluated_parameter_counts(records)
            step = dendrite_step or infer_dendrite_step(counts)
            dendrites = dendrite_counts(counts, step)
            params = [pruned_params + c - counts[0] for c in counts]
            finetune_log = (
                run_dir / "metrics" / "sparsity" / f"sparknet_c{width}" / "prune_supervised.jsonl"
            )
            runs.append(
                Run(
                    run_dir=str(run_dir),
                    seed=report.get("seed"),
                    width=width,
                    prune_rate=(
                        float(group["prune_rate"]) if group.get("prune_rate") is not None else None
                    ),
                    pruned_params=pruned_params,
                    step=step,
                    epochs=len(records),
                    finetune_epochs=(
                        len(load_jsonl(finetune_log)) if finetune_log.exists() else None
                    ),
                    budgets=select_budgets(records, dendrites, params, max_budget=max_budget),
                )
            )
    return runs


def _rate_key(run: Run) -> float:
    return run.prune_rate if run.prune_rate is not None else float(-run.width)


def zero_dendrite_reference(runs: Sequence[Run]) -> LogCurve:
    """Seed-mean budget-0 test accuracy per rate, at that rate's pruned count."""
    by_rate: dict[float, list[Run]] = defaultdict(list)
    for run in runs:
        if 0 in run.budgets and run.budgets[0].test_acc is not None:
            by_rate[_rate_key(run)].append(run)
    points = []
    for group in by_rate.values():
        params = sum(r.pruned_params for r in group) / len(group)
        accuracy = sum(r.budgets[0].test_acc for r in group) / len(group)  # type: ignore[misc]
        points.append((params, accuracy))
    return LogCurve.from_points(points)


def decompose(run: Run, z: LogCurve, *, budget: int) -> dict[str, float] | None:
    """Gain against Z at ``budget`` and its raw-gain / parameter-cost split."""
    start = run.budgets.get(0)
    final = run.budgets.get(budget)
    if start is None or final is None or start.test_acc is None or final.test_acc is None:
        return None
    z_start, z_final = z(start.params), z(final.params)
    return {
        "gain": final.test_acc - z_final,
        "raw_gain": final.test_acc - start.test_acc,
        "param_cost": z_final - z_start,
        "pre_gain": start.test_acc - z_start,
        "a_pre": start.test_acc,
        "a_final": final.test_acc,
        "z_final": z_final,
        "n_start": start.params,
        "n_final": final.params,
        "params_added_fraction": (final.params - start.params) / start.params,
        "dendrites": final.dendrites,
    }


def t_summary(values: Sequence[float], confidence: float = 0.95) -> dict[str, float | int | None]:
    """Mean, t-interval and two-sided one-sample t-test p against zero."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "p": None, "sd": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "ci_low": None, "ci_high": None, "p": None, "sd": None}
    from scipy import stats

    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0:
        return {"n": n, "mean": mean, "ci_low": mean, "ci_high": mean,
                "p": 0.0 if mean != 0 else 1.0, "sd": 0.0}
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * se
    p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    return {"n": n, "mean": mean, "ci_low": mean - half, "ci_high": mean + half, "p": p, "sd": sd}


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def analyze(
    runs: Sequence[Run],
    *,
    budget: int = MAX_BUDGET,
    scratch: LogCurve | None = None,
) -> dict[str, Any]:
    z = zero_dendrite_reference(runs)
    per_run: list[dict[str, Any]] = []
    by_rate: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        row: dict[str, Any] = {
            "run_dir": run.run_dir,
            "seed": run.seed,
            "width": run.width,
            "prune_rate": run.prune_rate,
            "pruned_params": run.pruned_params,
            "dendrite_step": run.step,
            "pai_epochs": run.epochs,
            "finetune_epochs": run.finetune_epochs,
            "budgets": {n: asdict(point) for n, point in run.budgets.items()},
            "by_budget": {
                n: decompose(run, z, budget=n) for n in sorted(run.budgets) if n > 0
            },
        }
        decomposition = decompose(run, z, budget=budget)
        row["decomposition"] = decomposition
        start = run.budgets.get(0)
        row["train_acc_budget0"] = start.train_acc if start else None
        row["running_train_acc_budget0"] = start.running_train_acc if start else None
        if scratch is not None and decomposition is not None:
            row["gain_vs_scratch"] = decomposition["a_final"] - scratch(decomposition["n_final"])
            row["pre_gain_vs_scratch"] = decomposition["a_pre"] - scratch(decomposition["n_start"])
        per_run.append(row)
        by_rate[_rate_key(run)].append(row)

    per_rate: list[dict[str, Any]] = []
    for key in sorted(by_rate):
        rows = by_rate[key]
        decs = [r["decomposition"] for r in rows if r["decomposition"] is not None]
        summary: dict[str, Any] = {
            "prune_rate": rows[0]["prune_rate"],
            "width": rows[0]["width"],
            "pruned_params": _mean([r["pruned_params"] for r in rows]),
            "runs": len(rows),
            "z_at_start": z(_mean([r["pruned_params"] for r in rows]) or 1),
            "a_pre": _mean([d["a_pre"] for d in decs]),
            "a_final": _mean([d["a_final"] for d in decs]),
            "gain": t_summary([d["gain"] for d in decs]),
            "raw_gain": _mean([d["raw_gain"] for d in decs]),
            "param_cost": _mean([d["param_cost"] for d in decs]),
            "params_added_fraction": _mean([d["params_added_fraction"] for d in decs]),
            "dendrites": _mean([d["dendrites"] for d in decs]),
            "train_acc_budget0": _mean([r["train_acc_budget0"] for r in rows]),
            "running_train_acc_budget0": _mean([r["running_train_acc_budget0"] for r in rows]),
            "finetune_epochs": _mean([r["finetune_epochs"] for r in rows]),
            "pai_epochs": _mean([r["pai_epochs"] for r in rows]),
        }
        if scratch is not None:
            summary["gain_vs_scratch"] = t_summary(
                [r["gain_vs_scratch"] for r in rows if "gain_vs_scratch" in r]
            )
        per_rate.append(summary)

    pooled = t_summary(
        [r["decomposition"]["gain"] for r in per_run if r["decomposition"] is not None]
    )
    return {
        "budget": budget,
        "zero_dendrite_reference": [
            {"params": p, "test_acc": a} for p, a in z.points
        ],
        "per_rate": per_rate,
        "pooled_gain": pooled,
        "runs": per_run,
    }


def load_scratch_frontier(path: str | Path) -> LogCurve:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        data = data.get("points", data.get("frontier"))
    if not isinstance(data, list):
        raise ValueError("scratch frontier must be a list of {params, test_acc}")
    return LogCurve.from_points((float(p["params"]), float(p["test_acc"])) for p in data)


def _pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}"


def _pts(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{100 * value:+.{digits}f}"


def format_table(result: Mapping[str, Any]) -> str:
    budget = result["budget"]
    header = (
        f"Budget {budget}: gain vs zero-dendrite reference Z (points; A and Z in %)\n"
        "rate  width params runs   A_pre     Z  A_final   gain  [95% CI]           p    "
        "raw    cost  +params  dend  train@b0  ft_ep  pai_ep"
    )
    lines = [header]
    for row in result["per_rate"]:
        gain = row["gain"]
        ci = (
            f"[{_pts(gain['ci_low'])}, {_pts(gain['ci_high'])}]"
            if gain.get("ci_low") is not None else "[-]"
        )
        p = "-" if gain.get("p") is None else f"{gain['p']:.3f}"
        rate = "-" if row["prune_rate"] is None else f"{100 * row['prune_rate']:.0f}%"
        lines.append(
            f"{rate:>5} {row['width']:>5} {row['pruned_params']:>6.0f} {row['runs']:>4} "
            f"{_pct(row['a_pre']):>7} {_pct(row['z_at_start']):>6} {_pct(row['a_final']):>7} "
            f"{_pts(gain['mean']):>7} {ci:<18} {p:>6} "
            f"{_pts(row['raw_gain']):>6} {_pts(row['param_cost'], 3):>7} "
            f"{_pct(row['params_added_fraction'], 1):>6}% "
            f"{row['dendrites'] if row['dendrites'] is None else round(row['dendrites'], 2):>5} "
            f"{_pct(row['train_acc_budget0']):>8} "
            f"{'-' if row['finetune_epochs'] is None else round(row['finetune_epochs'], 1):>6} "
            f"{'-' if row['pai_epochs'] is None else round(row['pai_epochs'], 1):>6}"
        )
        if "gain_vs_scratch" in row:
            scratch = row["gain_vs_scratch"]
            scratch_p = "-" if scratch.get("p") is None else f"{scratch['p']:.3f}"
            lines.append(
                f"{'':>5}   vs scratch frontier: {_pts(scratch['mean'])} (p={scratch_p})"
            )
    pooled = result["pooled_gain"]
    lines.append(
        f"pooled: n={pooled['n']} gain={_pts(pooled['mean'])} "
        f"CI=[{_pts(pooled['ci_low'])}, {_pts(pooled['ci_high'])}] "
        f"p={'-' if pooled.get('p') is None else format(pooled['p'], '.4f')}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="experiment output dirs (one per seed)")
    parser.add_argument("--budget", type=int, default=MAX_BUDGET)
    parser.add_argument("--dendrite-step", type=int, default=None,
                        help="parameters per dendrite (default: inferred per run)")
    parser.add_argument("--scratch-frontier", default=None,
                        help="JSON list of {params, test_acc} for a scratch width curve")
    parser.add_argument("--json", default=None, help="write the full result as JSON")
    args = parser.parse_args(argv)
    if not 0 <= args.budget <= MAX_BUDGET:
        parser.error(f"--budget must be in 0..{MAX_BUDGET}")

    runs = load_runs(args.run_dirs, dendrite_step=args.dendrite_step)
    if not runs:
        parser.error("no PAI logs found under the given run directories")
    scratch = load_scratch_frontier(args.scratch_frontier) if args.scratch_frontier else None
    result = analyze(runs, budget=args.budget, scratch=scratch)
    print(format_table(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
