import json
import statistics

import pytest
import yaml

from scripts import report_sparknet_grow as grow_report

BASE_EPOCHS = 200
SWITCH = 120
CLEAN_CHECKS = {
    "n_to_p_base_max_abs_change": 0.0,
    "candidate_phase_base_max_abs_drift": 0.0,
    "candidate_phase_val_acc_span": 0.0,
    "base_params_in_optimizer_candidate_phase": 0,
    "momentum_buffers_restored": 30,
    "momentum_buffers_expected": 30,
    "clean_parity_max_abs_diff_final": 0.0,
    "clean_parity_max_abs_diff_best": 0.0,
}
# Stub costs keep the synthetic frontier readable; the real ones are measured
# in test_scratch_cost_is_measured_from_the_paper_configs.
STUB_COSTS = {
    4: {"params": 1000, "macs": 50_000},
    8: {"params": 2000, "macs": 100_000},
    12: {"params": 3000, "macs": 200_000},
}


def _stub_cost(width):
    return STUB_COSTS[width]


def _write_scratch(root, width, seed, curve):
    """A finished scratch run in the sweep's layout; curve maps epoch -> val_acc."""
    run = root / f"c{width}-seed{seed}"
    relative = f"metrics/paper_replication/sparknet_c{width}_paper.jsonl"
    metrics = run / relative
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "".join(
            json.dumps({"epoch": epoch, "val_acc": value, "seed": seed}) + "\n"
            for epoch, value in sorted(curve.items())
        )
    )
    (run / "metrics" / "summaries.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "phases": {
                    f"paper_replication/sparknet_c{width}_paper": {
                        "best_val_acc": max(curve.values()),
                        "final_val_acc": curve[BASE_EPOCHS],
                        "epochs": BASE_EPOCHS,
                        "completed_epoch": BASE_EPOCHS,
                        "metrics": relative,
                    }
                },
            }
        )
    )


def _write_grow(
    root,
    arm,
    width,
    seed,
    curve,
    *,
    placement=None,
    summary_arm=None,
    variant=None,
    switch=SWITCH,
    candidate_epochs=15,
    dendrite_weight_decay=0.0,
    dendrite_input_scale=None,
    checks=None,
    deployed=(2500, 150_000),
    status="complete",
    diagnostics=None,
):
    """A grow run whose summary results are computed from its own per-epoch log.

    ``summary_arm`` / ``variant`` / ``diagnostics`` are written only when
    given, so the default is a summary from before arms existed; its
    placement defaults to the arm directory name, as the first sweep laid out.
    """
    run = root / arm / f"c{width}-seed{seed}"
    relative = f"metrics/grow/sparknet_c{width}_paper.jsonl"
    metrics = run / relative
    metrics.parent.mkdir(parents=True)
    records = [
        {"epoch": e, "segment": "pre_switch", "base_epoch": e, "val_acc": curve[e]}
        for e in range(1, switch + 1)
    ]
    records += [
        {"epoch": switch + k, "segment": "candidate", "base_epoch": None, "val_acc": curve[switch]}
        for k in range(1, candidate_epochs + 1)
    ]
    records += [
        {
            "epoch": e + candidate_epochs,
            "segment": "post_switch",
            "base_epoch": e,
            "val_acc": curve[e],
        }
        for e in range(switch + 1, BASE_EPOCHS + 1)
    ]
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
    post = [curve[e] for e in range(switch + 1, BASE_EPOCHS + 1)]
    summary = {
        "format_version": 1,
        "kind": "sparknet_grow_dendrites",
        "status": status,
        "incomplete_reason": None if status == "complete" else "wall clock cap",
        "width": width,
        "seed": seed,
        "placement": arm if placement is None else placement,
        "device": "mps",
        "schedule": {
            "base_epochs": BASE_EPOCHS,
            "switch_epoch": switch,
            "candidate_epochs": candidate_epochs,
            "total_epochs": BASE_EPOCHS + candidate_epochs,
            "dendrite_weight_decay": dendrite_weight_decay,
            # Summaries from before the input scale existed have no such key.
            **(
                {} if dendrite_input_scale is None
                else {"dendrite_input_scale": dendrite_input_scale}
            ),
        },
        "results": {
            "val_acc_at_switch": curve[switch],
            "best_val_acc_pre_switch": max(curve[e] for e in range(1, switch + 1)),
            "best_val_acc_post_switch": max(post),
            "final_val_acc": curve[BASE_EPOCHS],
            "last5_mean_val_acc": statistics.fmean(post[-5:]),
            "last10_mean_val_acc": statistics.fmean(post[-10:]),
            "best_val_acc_overall": max(curve.values()),
        },
        "checks": dict(CLEAN_CHECKS if checks is None else checks),
        "cost": {
            "base": dict(STUB_COSTS[width]),
            "deployed": {"params": deployed[0], "macs": deployed[1]},
        },
        "artifacts": {"metrics": relative},
        "test_split_used": False,
        "selection_split": "validation",
    }
    if summary_arm is not None:
        summary["arm"] = summary_arm
    if variant is not None:
        summary["variant"] = variant
    if diagnostics is not None:
        summary["dendrite_diagnostics"] = diagnostics
    (run / "reports").mkdir()
    (run / "reports" / "grow_summary.yaml").write_text(yaml.safe_dump(summary))
    return run


def _flat_scratch(scratch_root, seeds, c8=0.90, c12=0.92):
    for seed in seeds:
        _write_scratch(scratch_root, 8, seed, {e: c8 for e in range(1, BASE_EPOCHS + 1)})
        _write_scratch(scratch_root, 12, seed, {e: c12 for e in range(1, BASE_EPOCHS + 1)})


def _templates(scratch_root):
    return {
        width: str(scratch_root / "c{width}-seed{seed}") for width in STUB_COSTS
    }


def _run(report, label):
    return next(
        run
        for run in report["runs"]
        if f"{run['arm']} C{run['width']} seed{run['seed']}" == label
    )


def _ramp(epoch):
    return 0.80 + 0.0005 * epoch


def test_noncanonical_model_name_runs_are_discovered_and_retained_for_real_and_sham_arms(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _write_scratch(scratch_root, 4, 0, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
    _write_scratch(scratch_root, 12, 0, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 4, 0, curve)
    _write_grow(
        grow_root, "fc-sham", 4, 0, curve, variant={"sham": True},
        checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
    )
    # The exported sweep names these models c4g16/c10g16, not c4/c10.
    for arm, sham in (("fc", False), ("fc-sham", True)):
        source = grow_root / arm / "c4-seed0"
        target = grow_root / arm / "c4g16-seed0"
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        summary = yaml.safe_load((target / "reports" / "grow_summary.yaml").read_text())
        summary["model_name"] = "c4g16"
        (target / "reports" / "grow_summary.yaml").write_text(yaml.safe_dump(summary))

    report = grow_report.build_report(
        grow_root, {4: str(scratch_root / "c{width}-seed{seed}"), 12: str(scratch_root / "c{width}-seed{seed}")},
        seeds=(0,), cost_fn=_stub_cost,
    )
    assert {(run["arm"], run["model_name"], run["seed"]) for run in report["runs"] if run["status"] == "valid"} == {
        ("fc", "c4g16", 0), ("fc-sham", "c4g16", 0)
    }
    assert {cell["arm"] for cell in report["cells"]} == {"fc", "fc-sham"}


def test_standard_and_g16_model_names_are_isolated_within_one_arm(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    for seed in (0, 1):
        _write_scratch(scratch_root, 4, seed, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 4, 0, curve)
    _write_grow(grow_root, "fc", 4, 1, curve)
    source = grow_root / "fc" / "c4-seed1"
    target = grow_root / "fc" / "c4g16-seed1"
    source.rename(target)
    summary = yaml.safe_load((target / "reports" / "grow_summary.yaml").read_text())
    summary["model_name"] = "c4g16"
    (target / "reports" / "grow_summary.yaml").write_text(yaml.safe_dump(summary))

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1), cost_fn=_stub_cost
    )

    assert [(cell["model_name"], cell["seeds"]) for cell in report["cells"]] == [
        ("c4", [0]), ("c4g16", [1])
    ]
    assert len(report["inventory"]["missing"]) == 2
    assert {(run["model_name"], run["seed"]) for run in report["runs"] if run["status"] == "missing"} == {
        ("c4", 1), ("c4g16", 0)
    }
    markdown = grow_report.render_markdown(report)
    assert "C4g16" in markdown
    assert "may not be architecture-matched" in markdown


def test_noise_floor_keeps_standard_and_g16_model_names_distinct(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _write_scratch(scratch_root, 4, 0, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(
        grow_root, "fc-sham", 4, 0, curve, placement="fc",
        summary_arm="fc-sham", variant={**_variant(sham=True)},
        checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
    )
    source = grow_root / "fc-sham" / "c4-seed0"
    target = grow_root / "fc-sham" / "c4g16-seed0"
    source.rename(target)
    summary_path = target / "reports" / "grow_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text())
    summary["model_name"] = "c4g16"
    summary_path.write_text(yaml.safe_dump(summary))
    _write_grow(
        grow_root, "fc-sham", 4, 0, curve, placement="fc",
        summary_arm="fc-sham", variant={**_variant(sham=True)},
        checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
    )

    report = grow_report.build_report(
        grow_root, {4: str(scratch_root / "c{width}-seed{seed}")},
        seeds=(0,), cost_fn=_stub_cost,
    )
    assert [row["model_name"] for row in report["noise_floor"]] == ["c4", "c4g16"]
    markdown = grow_report.render_markdown(report)
    assert markdown.count("C4g16") >= 2


# --------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------


def test_interpolate_inside_range_is_piecewise_linear_between_adjacent_points():
    points = [(400, 0.93), (100, 0.90), (200, 0.92)]  # unsorted on purpose

    assert grow_report.interpolate(points, 150) == (pytest.approx(0.91), False)
    assert grow_report.interpolate(points, 300) == (pytest.approx(0.925), False)
    assert grow_report.interpolate(points, 200) == (pytest.approx(0.92), False)
    assert grow_report.interpolate(points, 100) == (pytest.approx(0.90), False)


def test_interpolate_past_the_widest_point_extends_the_last_segment():
    points = [(100, 0.90), (200, 0.92), (400, 0.93)]

    value, extrapolated = grow_report.interpolate(points, 600)

    # Last segment slope is 0.01 per 200; 200 past the end adds 0.01.
    assert value == pytest.approx(0.94)
    assert extrapolated is True


def test_interpolate_refuses_a_single_point_frontier():
    with pytest.raises(ValueError):
        grow_report.interpolate([(100, 0.9)], 150)


# --------------------------------------------------------------------------
# paired deltas from on-disk artifacts
# --------------------------------------------------------------------------


def test_paired_deltas_come_from_summary_and_scratch_jsonl(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    scratch_curve = {e: _ramp(e) for e in range(1, BASE_EPOCHS + 1)}
    _write_scratch(scratch_root, 8, 0, scratch_curve)
    _write_scratch(scratch_root, 12, 0, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
    # Identical before the switch, +0.4 pp after it, except the last epoch
    # gains only +0.1 pp -- so the four metrics all differ.
    grow_curve = {
        e: _ramp(e) + (0 if e <= SWITCH else 0.004 if e < BASE_EPOCHS else 0.001)
        for e in range(1, BASE_EPOCHS + 1)
    }
    _write_grow(grow_root, "fc", 8, 0, grow_curve)

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )
    run = _run(report, "fc C8 seed0")

    assert run["status"] == "valid"
    # best: grow best over 121..200 is epoch 199 (0.8995 + 0.004) vs scratch's 0.90.
    assert run["deltas_pp"]["best"] == pytest.approx(0.35)
    assert run["deltas_pp"]["final"] == pytest.approx(0.10)
    assert run["deltas_pp"]["last5"] == pytest.approx((4 * 0.4 + 0.1) / 5)
    assert run["deltas_pp"]["last10"] == pytest.approx((9 * 0.4 + 0.1) / 10)
    assert run["scratch_values"]["last5"] == pytest.approx(_ramp(198))
    assert run["replication_max_abs_diff"] == pytest.approx(0.0)
    assert run["replication_epochs"] == SWITCH
    assert run["warnings"] == []


def test_a_diverged_pre_switch_trajectory_is_flagged_but_still_counted(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _write_scratch(scratch_root, 8, 0, {e: _ramp(e) for e in range(1, BASE_EPOCHS + 1)})
    _write_scratch(scratch_root, 12, 0, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
    grow_curve = {e: _ramp(e) for e in range(1, BASE_EPOCHS + 1)}
    grow_curve[50] += 0.003
    _write_grow(grow_root, "fc", 8, 0, grow_curve)

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )
    run = _run(report, "fc C8 seed0")

    assert run["status"] == "valid"
    assert run["replication_max_abs_diff"] == pytest.approx(0.003)
    assert any("pre-switch trajectory" in warning for warning in run["warnings"])


# --------------------------------------------------------------------------
# invalid runs
# --------------------------------------------------------------------------


def test_check_failures_names_each_broken_premise():
    assert grow_report.check_failures(CLEAN_CHECKS) == []

    broken = {
        "n_to_p_base_max_abs_change": {"n_to_p_base_max_abs_change": 1e-6},
        "candidate_phase_base_max_abs_drift": {"candidate_phase_base_max_abs_drift": 0.01},
        "base_params_in_optimizer_candidate_phase": {
            "base_params_in_optimizer_candidate_phase": 4
        },
        "momentum_buffers_restored": {"momentum_buffers_restored": 29},
        "clean_parity_max_abs_diff_final": {"clean_parity_max_abs_diff_final": 2e-4},
        "clean_parity_max_abs_diff_best": {"clean_parity_max_abs_diff_best": 1e-3},
    }
    for key, override in broken.items():
        failures = grow_report.check_failures({**CLEAN_CHECKS, **override})
        assert len(failures) == 1 and key in failures[0], (key, failures)

    # Parity at the tolerance is fine; an unrecorded check is not.
    assert grow_report.check_failures(
        {**CLEAN_CHECKS, "clean_parity_max_abs_diff_best": 1e-4}
    ) == []
    missing = {k: v for k, v in CLEAN_CHECKS.items() if k != "momentum_buffers_expected"}
    assert grow_report.check_failures(missing) == [
        "checks.momentum_buffers_expected not recorded"
    ]
    assert grow_report.check_failures(None) == ["summary has no checks block"]


def test_runs_failing_their_checks_are_excluded_from_the_cell(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    seeds = (0, 1, 2, 3)
    for seed in seeds:
        _write_scratch(scratch_root, 8, seed, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
        _write_scratch(scratch_root, 12, seed, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
    curve = {e: 0.90 if e <= SWITCH else 0.95 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 8, 0, curve)
    _write_grow(grow_root, "fc", 8, 1, curve)
    _write_grow(
        grow_root, "fc", 8, 2, curve,
        checks={**CLEAN_CHECKS, "momentum_buffers_restored": 29},
    )
    _write_grow(
        grow_root, "fc", 8, 3, curve,
        checks={**CLEAN_CHECKS, "clean_parity_max_abs_diff_final": 1e-3},
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=seeds, cost_fn=_stub_cost
    )

    assert report["inventory"]["valid"] == ["fc C8 seed0", "fc C8 seed1"]
    assert report["inventory"]["invalid"] == ["fc C8 seed2", "fc C8 seed3"]
    cell = report["cells"][0]
    assert cell["n"] == 2 and cell["seeds"] == [0, 1]
    # Two valid seeds is below the minimum, however good they look.
    assert cell["break_even"]["params"]["verdict"] == grow_report.GATE_INSUFFICIENT
    markdown = grow_report.render_markdown(report, per_run=True)
    assert "INVALID runs" in markdown and "momentum_buffers_restored=29" in markdown


def test_incomplete_and_missing_runs_are_inventoried_not_counted(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    for seed in (0, 1, 2):
        _write_scratch(scratch_root, 8, seed, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
        _write_scratch(scratch_root, 12, seed, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
    curve = {e: 0.91 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 8, 0, curve)
    _write_grow(grow_root, "fc", 8, 1, curve, status="incomplete")
    (grow_root / "fc" / "c8-seed2").mkdir()  # started, no summary yet

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1, 2), cost_fn=_stub_cost
    )

    assert report["inventory"]["valid"] == ["fc C8 seed0"]
    assert report["inventory"]["incomplete"] == ["fc C8 seed1"]
    assert report["inventory"]["missing"] == ["fc C8 seed2"]
    assert report["cells"][0]["n"] == 1


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def test_required_positive_is_ceil_two_thirds():
    assert [grow_report.required_positive(n) for n in (3, 4, 5, 6)] == [2, 3, 4, 4]


def test_lower_confidence_bound_uses_the_one_sided_student_t_quantile():
    # One-sided 90% Student-t quantiles from a t table: df=2 -> 1.886, df=4 -> 1.533.
    assert grow_report.t_quantile(0.90, 2) == pytest.approx(1.886, abs=5e-4)
    assert grow_report.t_quantile(0.90, 4) == pytest.approx(1.533, abs=5e-4)

    # mean 0.3, sd 0.2, n 3: 0.3 - 1.8856 * 0.2 / sqrt(3) = 0.0823.
    lcb = grow_report.lower_confidence_bound([0.5, 0.3, 0.1], 0.90)
    assert lcb == pytest.approx(0.3 - 1.885618 * 0.2 / 3**0.5, abs=1e-6)
    assert lcb == pytest.approx(0.0823, abs=1e-4)
    # Higher confidence, lower bound; no spread, bound = mean; one value, no bound.
    assert grow_report.lower_confidence_bound([0.5, 0.3, 0.1], 0.95) < lcb
    assert grow_report.lower_confidence_bound([0.2, 0.2, 0.2], 0.90) == pytest.approx(0.2)
    assert grow_report.lower_confidence_bound([0.4], 0.90) is None


def test_gate_verdict_matrix():
    verdict = grow_report.gate_verdict
    # PASS: both LCB margins > 0 and ceil(2n/3) positive seeds.
    assert verdict(5, 4, 0.10, 0.05, 0.02, 0.01) == grow_report.GATE_PASS
    assert verdict(3, 2, 0.10, 0.05, 0.02, 0.01) == grow_report.GATE_PASS
    # INCONCLUSIVE: the means clear the frontier, the lower bound does not ...
    assert verdict(3, 3, 0.10, 0.05, -0.20, 0.01) == grow_report.GATE_INCONCLUSIVE
    assert verdict(3, 3, 0.10, 0.05, 0.02, -0.01) == grow_report.GATE_INCONCLUSIVE
    # ... or too few seeds improve (3 of 5 is not ceil(10/3) = 4).
    assert verdict(5, 3, 0.10, 0.05, 0.02, 0.01) == grow_report.GATE_INCONCLUSIVE
    # FAIL: a mean margin is not positive.
    assert verdict(5, 5, 0.10, -0.01, 0.05, -0.05) == grow_report.GATE_FAIL
    assert verdict(5, 5, 0.0, 0.10, -0.1, 0.05) == grow_report.GATE_FAIL
    # INSUFFICIENT SEEDS before anything else, however good the numbers.
    assert verdict(2, 2, 1.0, 1.0, 0.5, 0.5) == grow_report.GATE_INSUFFICIENT
    assert verdict(0, 0, None, None, None, None) == grow_report.GATE_INSUFFICIENT
    assert verdict(2, 2, 1.0, 1.0, 0.5, 0.5, min_seeds=2) == grow_report.GATE_PASS
    assert verdict(3, 3, None, 0.1, None, 0.1) == grow_report.GATE_NO_FRONTIER


def test_noisy_zero_effect_clears_the_mean_but_not_the_lower_bound(tmp_path):
    """The failure the LCB fixes: fc on MACs needs ~0 pp, noise alone clears it."""
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    seeds = (0, 1, 2)
    _flat_scratch(scratch_root, seeds)
    # Paired deltas +0.6, -0.3, +0.3 pp: mean +0.2, sd 0.46.
    for seed, offset in zip(seeds, (0.006, -0.003, 0.003)):
        _write_grow(
            grow_root, "fc", 8, seed,
            {e: 0.90 if e <= SWITCH else 0.90 + offset for e in range(1, BASE_EPOCHS + 1)},
            # 100 MACs over C8: the MACs frontier needs 0.002 pp.
            deployed=(2500, 100_100),
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=seeds, cost_fn=_stub_cost
    )
    cell = report["cells"][0]
    macs = cell["break_even"]["macs"]
    lcb = 0.2 - 1.885618 * statistics.stdev([0.6, -0.3, 0.3]) / 3**0.5

    assert cell["delta_pp"]["best"]["lcb"] == pytest.approx(lcb, abs=1e-5)
    assert macs["best"]["needed_gain_pp_all"] == pytest.approx(0.002)
    assert macs["best"]["margin_pp"] == pytest.approx(0.198)
    assert macs["best"]["lcb_margin_pp"] == pytest.approx(lcb - 0.002, abs=1e-5)
    assert macs["best"]["lcb_margin_pp"] < 0 < macs["last5"]["margin_pp"]
    # The old mean-based gate would have said PASS here.
    assert macs["verdict"] == grow_report.GATE_INCONCLUSIVE
    assert cell["break_even"]["params"]["verdict"] == grow_report.GATE_FAIL
    assert cell["passes_on"] == []
    markdown = grow_report.render_markdown(report)
    assert "LCB margin best pp" in markdown and "**INCONCLUSIVE**" in markdown


def test_break_even_passes_on_params_and_fails_on_macs(tmp_path):
    """C8 at 90% and C12 at 92% (stub costs); fc C8 reaches 91.5%."""
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    seeds = (0, 1, 2)
    for seed in seeds:
        _write_scratch(scratch_root, 8, seed, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
        _write_scratch(scratch_root, 12, seed, {e: 0.92 for e in range(1, BASE_EPOCHS + 1)})
        _write_grow(
            grow_root, "fc", 8, seed,
            {e: 0.90 if e <= SWITCH else 0.915 for e in range(1, BASE_EPOCHS + 1)},
            # Halfway to C12 in params (frontier 91.0%), 90% of the way in MACs
            # (frontier 91.8%).
            deployed=(2500, 190_000),
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=seeds, cost_fn=_stub_cost
    )
    cell = report["cells"][0]
    params, macs = cell["break_even"]["params"], cell["break_even"]["macs"]

    assert cell["delta_pp"]["best"]["mean"] == pytest.approx(1.5)
    assert cell["delta_pp"]["best"]["n_positive"] == 3
    assert params["best"]["frontier_acc"] == pytest.approx(0.91)
    assert params["best"]["extrapolated"] is False
    assert params["best"]["needed_gain_pp_paired"] == pytest.approx(1.0)
    assert params["best"]["needed_gain_pp_all"] == pytest.approx(1.0)
    assert params["best"]["margin_pp"] == pytest.approx(0.5)
    assert params["last5"]["margin_pp"] == pytest.approx(0.5)
    # Every seed gained exactly 1.5 pp: no spread, so the bound is the mean.
    assert params["best"]["lcb_margin_pp"] == pytest.approx(0.5)
    assert params["last5"]["lcb_margin_pp"] == pytest.approx(0.5)
    assert params["verdict"] == grow_report.GATE_PASS
    assert macs["best"]["frontier_acc"] == pytest.approx(0.918)
    assert macs["best"]["margin_pp"] == pytest.approx(-0.3)
    assert macs["verdict"] == grow_report.GATE_FAIL
    assert cell["passes_on"] == ["params"]


def test_missing_grow_root_still_reports_the_scratch_frontier(tmp_path):
    scratch_root = tmp_path / "scratch"
    for width, accuracy in ((8, 0.90), (12, 0.92)):
        _write_scratch(scratch_root, width, 0, {e: accuracy for e in range(1, BASE_EPOCHS + 1)})

    report = grow_report.build_report(
        tmp_path / "does-not-exist", _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )

    assert report["cells"] == [] and report["runs"] == []
    assert [row["width"] for row in report["frontier"]] == [8, 12]
    assert report["frontier"][0]["slope_to_next"]["best_pp_per_100_params"] == pytest.approx(0.2)
    assert "Scratch frontier" in grow_report.render_markdown(report)


# --------------------------------------------------------------------------
# last-N window
# --------------------------------------------------------------------------


def test_window_is_the_mean_paired_per_epoch_delta_over_the_last_n_epochs(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _write_scratch(scratch_root, 8, 0, {e: _ramp(e) for e in range(1, BASE_EPOCHS + 1)})
    _write_scratch(scratch_root, 12, 0, {e: 0.95 for e in range(1, BASE_EPOCHS + 1)})
    # +0.2 pp over 121..160, +0.4 pp over 161..190, -0.2 pp over 191..200.
    offsets = {e: 0.002 for e in range(SWITCH + 1, 161)}
    offsets.update({e: 0.004 for e in range(161, 191)})
    offsets.update({e: -0.002 for e in range(191, BASE_EPOCHS + 1)})
    _write_grow(
        grow_root, "fc", 8, 0,
        {e: _ramp(e) + offsets.get(e, 0.0) for e in range(1, BASE_EPOCHS + 1)},
    )

    def run_for(window):
        report = grow_report.build_report(
            grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost, window=window
        )
        return report, _run(report, "fc C8 seed0")

    report, run = run_for(40)
    assert run["window"]["epochs"] == 40 and run["window"]["first_epoch"] == 161
    assert run["window"]["truncated"] is False
    assert run["deltas_pp"]["window"] == pytest.approx((30 * 0.4 + 10 * -0.2) / 40)
    assert run["deltas_pp"]["last5"] == pytest.approx(-0.2)
    cell = report["cells"][0]
    assert cell["delta_pp"]["window"]["mean"] == pytest.approx(0.25)
    assert cell["window"] == {"requested": 40, "epochs": [40], "truncated": False}
    markdown = grow_report.render_markdown(report, per_run=True)
    assert "Δlast40 pp" in markdown and "+0.25" in markdown and "[last" not in markdown

    _, run = run_for(80)
    assert run["deltas_pp"]["window"] == pytest.approx((40 * 0.2 + 30 * 0.4 - 10 * 0.2) / 80)
    # Longer than the post-switch segment: all 80 post-switch epochs, never
    # the pre-switch replay (whose zero deltas would dilute the mean).
    _, run = run_for(100)
    assert run["window"]["epochs"] == 80 and run["window"]["first_epoch"] == SWITCH + 1
    assert run["window"]["truncated"] is True
    assert run["deltas_pp"]["window"] == pytest.approx(0.225)


def test_a_late_switch_truncates_the_window_and_marks_it(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    _write_grow(
        grow_root, "fc-switch170", 8, 0,
        {e: 0.90 if e <= 170 else 0.903 for e in range(1, BASE_EPOCHS + 1)},
        placement="fc",
        summary_arm="fc-switch170",
        variant={
            "sham": False, "dendrite_weight_decay": 0.0,
            "switch_epoch": 170, "candidate_epochs": 15,
        },
        switch=170,
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )
    run = _run(report, "fc-switch170 C8 seed0")

    assert run["status"] == "valid", run["reasons"]
    assert run["window"]["epochs"] == 30 and run["window"]["truncated"] is True
    assert run["window"]["first_epoch"] == 171
    assert run["deltas_pp"]["window"] == pytest.approx(0.3)
    assert report["cells"][0]["window"]["truncated"] is True
    markdown = grow_report.render_markdown(report, per_run=True)
    assert "[last30*]" in markdown
    assert "`[lastK*]`" in markdown


def test_window_is_skipped_with_a_warning_when_the_log_lacks_post_switch_epochs(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    run_dir = _write_grow(grow_root, "fc", 8, 0, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)})
    metrics = run_dir / "metrics/grow/sparknet_c8_paper.jsonl"
    kept = [
        line for line in metrics.read_text().splitlines()
        if json.loads(line).get("base_epoch") != 195
    ]
    metrics.write_text("\n".join(kept) + "\n")

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )
    run = _run(report, "fc C8 seed0")

    assert run["status"] == "valid"
    assert run["window"] is None and run["deltas_pp"]["window"] is None
    assert any("last40 window not computed" in warning for warning in run["warnings"])
    assert report["cells"][0]["delta_pp"]["window"]["n"] == 0


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def _variant(
    sham=False, dendrite_weight_decay=0.0, switch_epoch=SWITCH, candidate_epochs=15,
    dendrite_input_scale=1.0,
):
    return {
        "sham": sham,
        "dendrite_weight_decay": dendrite_weight_decay,
        "switch_epoch": switch_epoch,
        "candidate_epochs": candidate_epochs,
        "dendrite_input_scale": dendrite_input_scale,
    }


def test_runs_are_grouped_by_arm_and_width(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    curve = {e: 0.90 if e <= SWITCH else 0.91 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 8, 0, curve)  # an old summary: no arm, no variant
    for width in (8, 12):
        _write_grow(
            grow_root, "fc-wd1e-3", width, 0, curve,
            placement="fc", summary_arm="fc-wd1e-3",
            variant=_variant(dendrite_weight_decay=1e-3), dendrite_weight_decay=1e-3,
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )

    assert report["inventory"]["valid"] == ["fc C8 seed0", "fc-wd1e-3 C8 seed0", "fc-wd1e-3 C12 seed0"]
    assert [(cell["arm"], cell["width"]) for cell in report["cells"]] == [
        ("fc", 8), ("fc-wd1e-3", 8), ("fc-wd1e-3", 12)
    ]
    arms = {arm["arm"]: arm for arm in report["arms"]}
    assert arms["fc"]["placement"] == "fc"
    assert arms["fc"]["variant"] == _variant()
    assert arms["fc-wd1e-3"]["placement"] == "fc"
    assert arms["fc-wd1e-3"]["variant"]["dendrite_weight_decay"] == pytest.approx(1e-3)
    assert arms["fc-wd1e-3"]["runs_found"] == 2 and arms["fc-wd1e-3"]["valid"] == 2
    assert arms["fc-wd1e-3"]["widths"] == [8, 12]
    assert all(not arm["mixed_variants"] for arm in arms.values())
    markdown = grow_report.render_markdown(report)
    assert "### Arms" in markdown
    assert "| `fc-wd1e-3` | fc | no | 120 | 15 | 0.001 | 1 | C8,C12 | 2 | 2 | -- |" in markdown


def test_a_summary_naming_another_arm_is_invalid(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0, 1))
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    # New summary that names a different arm than its directory.
    _write_grow(
        grow_root, "fc-switch170", 8, 0, curve,
        placement="fc", summary_arm="fc", variant=_variant(),
    )
    # Old summary (no arm key) in an arm directory: its arm falls back to
    # its placement, which is not the directory name either.
    _write_grow(grow_root, "fc-switch170", 8, 1, curve, placement="fc")

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1), cost_fn=_stub_cost
    )

    assert report["inventory"]["invalid"] == ["fc-switch170 C8 seed0", "fc-switch170 C8 seed1"]
    assert "summary arm='fc' but the arm directory is 'fc-switch170'" in _run(
        report, "fc-switch170 C8 seed0"
    )["reasons"]
    assert "summary arm (fallback: placement)='fc' but the arm directory is 'fc-switch170'" in _run(
        report, "fc-switch170 C8 seed1"
    )["reasons"]
    assert report["cells"][0]["n"] == 0
    assert "2 run(s) name a different arm (INVALID)" in report["arms"][0]["notes"]


def test_mixed_variants_within_an_arm_invalidate_the_minority(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0, 1, 2))
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    for seed, decay in ((0, 1e-3), (1, 1e-3), (2, 0.0)):
        _write_grow(
            grow_root, "fc-wd1e-3", 8, seed, curve,
            placement="fc", summary_arm="fc-wd1e-3",
            variant=_variant(dendrite_weight_decay=decay), dendrite_weight_decay=decay,
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1, 2), cost_fn=_stub_cost
    )

    assert report["inventory"]["valid"] == ["fc-wd1e-3 C8 seed0", "fc-wd1e-3 C8 seed1"]
    assert report["inventory"]["invalid"] == ["fc-wd1e-3 C8 seed2"]
    assert "disagrees with the majority of arm 'fc-wd1e-3'" in _run(
        report, "fc-wd1e-3 C8 seed2"
    )["reasons"][0]
    arm = report["arms"][0]
    assert arm["mixed_variants"] is True
    assert arm["variant"]["dendrite_weight_decay"] == pytest.approx(1e-3)
    assert "MIXED VARIANTS" in grow_report.render_markdown(report)


def test_mixed_variants_with_no_majority_invalidate_every_run(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0, 1))
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    for seed, sham in ((0, True), (1, False)):
        _write_grow(
            grow_root, "fc-x", 8, seed, curve,
            placement="fc", summary_arm="fc-x", variant=_variant(sham=sham),
            checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1), cost_fn=_stub_cost
    )

    assert report["inventory"]["invalid"] == ["fc-x C8 seed0", "fc-x C8 seed1"]
    assert report["arms"][0]["variant"] is None
    assert "no majority" in report["arms"][0]["notes"][0]


def test_a_variant_that_contradicts_its_own_schedule_is_invalid(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    _write_grow(
        grow_root, "fc-switch170", 8, 0, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)},
        placement="fc", summary_arm="fc-switch170", variant=_variant(switch_epoch=170),
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )

    reasons = _run(report, "fc-switch170 C8 seed0")["reasons"]
    assert reasons == [
        "variant.switch_epoch=170 but schedule.switch_epoch=120 "
        "(the run is not the variant it claims)"
    ]


def test_an_input_scale_arm_is_its_own_arm_and_old_summaries_ran_at_scale_one(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    curve = {e: 0.90 if e <= SWITCH else 0.91 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "pointwise", 8, 0, curve)  # predates the input scale
    _write_grow(
        grow_root, "pointwise-in75", 8, 0, curve,
        placement="pointwise", summary_arm="pointwise-in75",
        variant=_variant(dendrite_input_scale=75.0), dendrite_input_scale=75.0,
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )

    assert report["inventory"]["valid"] == ["pointwise C8 seed0", "pointwise-in75 C8 seed0"]
    arms = {arm["arm"]: arm for arm in report["arms"]}
    assert arms["pointwise"]["variant"]["dendrite_input_scale"] == 1.0
    assert arms["pointwise-in75"]["variant"]["dendrite_input_scale"] == 75.0
    markdown = grow_report.render_markdown(report)
    assert "| dendrite_input_scale |" in markdown
    assert "| `pointwise-in75` | pointwise | no | 120 | 15 | 0 | 75 |" in markdown


def test_an_input_scale_the_schedule_did_not_run_is_invalid(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    _write_grow(
        grow_root, "pointwise-in75", 8, 0, {e: 0.90 for e in range(1, BASE_EPOCHS + 1)},
        placement="pointwise", summary_arm="pointwise-in75",
        variant=_variant(dendrite_input_scale=75.0),  # schedule has no scale: it ran at 1
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )

    assert _run(report, "pointwise-in75 C8 seed0")["reasons"] == [
        "variant.dendrite_input_scale=75 but schedule.dendrite_input_scale=1 "
        "(the run is not the variant it claims)"
    ]


def test_missing_seeds_are_counted_only_in_populated_cells(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, range(5))
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(grow_root, "fc", 8, 0, curve)  # a one-seed pilot at one width
    for seed in (0, 1):
        _write_grow(
            grow_root, "fc-sham", 12, seed, curve,
            placement="fc", summary_arm="fc-sham", variant=_variant(sham=True),
            checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=range(5), grow_seeds=range(5),
        cost_fn=_stub_cost,
    )

    assert report["inventory"]["missing"] == [
        "fc C8 seed1", "fc C8 seed2", "fc C8 seed3", "fc C8 seed4",
        "fc-sham C12 seed2", "fc-sham C12 seed3", "fc-sham C12 seed4",
    ]
    assert _run(report, "fc-sham C12 seed3")["placement"] == "fc"
    assert [(cell["arm"], cell["width"]) for cell in report["cells"]] == [
        ("fc", 8), ("fc-sham", 12)
    ]


# --------------------------------------------------------------------------
# sham arms
# --------------------------------------------------------------------------


def test_sham_arms_are_the_noise_floor_and_never_gated(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    seeds = (0, 1, 2)
    _flat_scratch(scratch_root, seeds)
    sham_checks = {**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0}
    for seed, offset in zip(seeds, (0.014, 0.015, 0.016)):
        curve = {e: 0.90 if e <= SWITCH else 0.90 + offset for e in range(1, BASE_EPOCHS + 1)}
        # Big enough that it would PASS on params if it were gated.
        _write_grow(
            grow_root, "fc-sham", 8, seed, curve, deployed=(2500, 190_000),
            placement="fc", summary_arm="fc-sham", variant=_variant(sham=True),
            checks=sham_checks,
        )
        _write_grow(grow_root, "fc", 8, seed, curve, deployed=(2500, 190_000))

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=seeds, cost_fn=_stub_cost
    )
    cells = {cell["arm"]: cell for cell in report["cells"]}

    assert cells["fc"]["sham"] is False
    assert cells["fc"]["passes_on"] == ["params"]
    sham = cells["fc-sham"]
    assert sham["sham"] is True and sham["n"] == 3
    assert sham["break_even"] is None
    assert sham["verdicts"] == {"params": grow_report.GATE_SHAM, "macs": grow_report.GATE_SHAM}
    assert sham["passes_on"] == []

    [floor] = report["noise_floor"]
    assert (floor["arm"], floor["placement"], floor["width"], floor["n"]) == ("fc-sham", "fc", 8, 3)
    assert floor["delta_pp"]["best"]["mean"] == pytest.approx(1.5)
    assert floor["delta_pp"]["best"]["sd"] == pytest.approx(0.1)
    assert floor["delta_pp"]["best"]["max_abs"] == pytest.approx(1.6)
    assert floor["delta_pp"]["window"]["mean"] == pytest.approx(1.5)

    markdown = grow_report.render_markdown(report, per_run=True)
    assert "## 3. Noise floor (sham arms)" in markdown
    assert "| fc-sham (sham) | C8 | 3 |" in markdown
    assert "| fc-sham | fc | C8 | 3 | +1.50 ± 0.10 (3/3+) |" in markdown
    assert "| fc C8 | 3 | params |" in markdown
    assert "| fc-sham C8 |" not in markdown  # not in the break-even table
    assert "fc-sham C8 (n=3): not gated" in markdown


def test_a_sham_whose_dendrite_is_not_zero_is_invalid(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0, 1))
    curve = {e: 0.90 for e in range(1, BASE_EPOCHS + 1)}
    for seed, checks in (
        (0, {**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 1e-3}),
        (1, CLEAN_CHECKS),  # the sham check was never recorded
    ):
        _write_grow(
            grow_root, "fc-sham", 8, seed, curve,
            placement="fc", summary_arm="fc-sham", variant=_variant(sham=True), checks=checks,
        )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1), cost_fn=_stub_cost
    )

    assert report["inventory"]["invalid"] == ["fc-sham C8 seed0", "fc-sham C8 seed1"]
    assert "sham_skip_weight_max_abs_final=0.001 > 0" in _run(report, "fc-sham C8 seed0")["reasons"][0]
    assert _run(report, "fc-sham C8 seed1")["reasons"] == [
        "checks.sham_skip_weight_max_abs_final not recorded"
    ]
    # A non-sham run does not need the sham check.
    assert grow_report.check_failures(CLEAN_CHECKS) == []


# --------------------------------------------------------------------------
# dendrite diagnostics
# --------------------------------------------------------------------------


def _diagnostics(on, off, modules):
    export = {
        "val_acc_dendrite_on": on,
        "val_acc_dendrite_off": off,
        "n_samples": 4445,
        "device": "cpu",
        "modules": {
            name: {
                "n_dendrites": 1,
                "skip_weight_mean_abs": 1.7,
                "corr_with_base_output": corr,
                "dendrite_to_base_std_ratio": 0.2,
                "linear_r2_vs_preactivation": r2,
                "tanh_saturated_fraction": 0.1,
                "tanh_linear_fraction": 0.5,
            }
            for name, (r2, corr) in modules.items()
        },
    }
    return {"final": export, "best": dict(export)}


def test_diagnostics_come_from_the_summary_or_the_fallback_file(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0, 1, 2))
    curve = {e: 0.90 if e <= SWITCH else 0.91 for e in range(1, BASE_EPOCHS + 1)}
    _write_grow(
        grow_root, "fc", 8, 0, curve,
        diagnostics=_diagnostics(0.915, 0.900, {"fc": (0.86, 0.63)}),
        checks={**CLEAN_CHECKS, "integration_output_max_abs_diff": 1e-3},
    )
    run_dir = _write_grow(
        grow_root, "fc", 8, 1, curve,
        checks={**CLEAN_CHECKS, "integration_output_max_abs_diff": 1e-7},
    )
    (run_dir / "reports" / "grow_diagnostics.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "source": "final_clean_pai.pt",
                **_diagnostics(0.910, 0.905, {"a": (0.9, 0.1), "b": (0.5, 0.2), "c": (0.7, 0.3)}),
            }
        )
    )
    _write_grow(grow_root, "fc", 8, 2, curve, diagnostics={"error": "no clean model"})

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0, 1, 2), cost_fn=_stub_cost
    )
    from_summary = _run(report, "fc C8 seed0")
    from_file = _run(report, "fc C8 seed1")
    failed = _run(report, "fc C8 seed2")

    assert from_summary["diagnostics_source"] == "summary"
    final = from_summary["diagnostics_summary"]["final"]
    assert final["off_drop_pp"] == pytest.approx(1.5)
    assert final["linear_r2"] == {"fc": 0.86} and final["corr"] == {"fc": 0.63}
    assert from_summary["diagnostics_summary"]["best"]["off_drop_pp"] == pytest.approx(1.5)
    assert from_file["diagnostics_source"] == "reports/grow_diagnostics.yaml"
    assert from_file["diagnostics_summary"]["final"]["off_drop_pp"] == pytest.approx(0.5)
    assert failed["status"] == "valid" and failed["diagnostics_summary"] == {}
    assert any("no clean model" in warning for warning in failed["warnings"])
    off_drop = report["cells"][0]["diagnostics"]["off_drop_pp"]
    assert off_drop["n"] == 2 and off_drop["mean"] == pytest.approx(1.0)

    markdown = grow_report.render_markdown(report, per_run=True)
    row = next(line for line in markdown.splitlines() if line.startswith("| fc C8 seed0 |"))
    assert "| +1.50 | fc:0.86 | fc:0.63 |" in row
    assert "integration output max abs diff 0.001" in row
    row = next(line for line in markdown.splitlines() if line.startswith("| fc C8 seed1 |"))
    assert "| +0.50 | min 0.50 (b; 3 modules) | min 0.10 (a; 3 modules) | -- |" in row
    row = next(line for line in markdown.splitlines() if line.startswith("| fc C8 seed2 |"))
    assert "| -- | -- | -- | 1 warning(s) (see section 1) |" in row
    # Mean off-drop over the two runs that have it, out of three.
    assert "| +1.00 (2/3) |" in markdown


def test_an_old_summary_without_any_new_key_still_reports(tmp_path):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, (0,))
    _write_grow(
        grow_root, "fc", 8, 0,
        {e: 0.90 if e <= SWITCH else 0.91 for e in range(1, BASE_EPOCHS + 1)},
    )

    report = grow_report.build_report(
        grow_root, _templates(scratch_root), seeds=(0,), cost_fn=_stub_cost
    )
    run = _run(report, "fc C8 seed0")

    assert run["status"] == "valid" and run["warnings"] == []
    assert (run["arm"], run["arm_declared"], run["placement"]) == ("fc", "fc", "fc")
    assert run["variant"] == _variant()
    assert run["diagnostics"] is None and run["diagnostics_summary"] == {}
    assert run["deltas_pp"]["window"] == pytest.approx(1.0)
    cell = report["cells"][0]
    assert cell["sham"] is False
    assert cell["verdicts"] == {
        "params": grow_report.GATE_INSUFFICIENT, "macs": grow_report.GATE_INSUFFICIENT
    }
    assert "| -- | -- | -- | -- |" in grow_report.render_markdown(report, per_run=True)


# --------------------------------------------------------------------------
# command line and JSON
# --------------------------------------------------------------------------


def test_main_writes_every_new_field_to_json(tmp_path, capsys):
    scratch_root = tmp_path / "scratch"
    grow_root = tmp_path / "grow"
    _flat_scratch(scratch_root, range(5))
    curve = {e: 0.90 if e <= SWITCH else 0.91 for e in range(1, BASE_EPOCHS + 1)}
    for seed in (0, 1, 2):
        _write_grow(
            grow_root, "fc", 8, seed, curve, deployed=(2764, 177_732),
            diagnostics=_diagnostics(0.915, 0.900, {"fc": (0.86, 0.63)}),
        )
        _write_grow(
            grow_root, "fc-sham", 8, seed, curve, deployed=(2764, 177_732),
            placement="fc", summary_arm="fc-sham", variant=_variant(sham=True),
            checks={**CLEAN_CHECKS, "sham_skip_weight_max_abs_final": 0.0},
        )
    json_out = tmp_path / "report.json"

    assert grow_report.main(
        [
            "--grow-root", str(grow_root),
            "--scratch-root", str(scratch_root),
            "--c16-scratch-root", str(tmp_path / "no-c16"),
            "--window", "30",
            "--confidence", "0.95",
            "--json-out", str(json_out),
        ]
    ) == 0
    printed = capsys.readouterr().out
    report = json.loads(json_out.read_text())

    assert "Δlast30 pp" in printed and "t(0.95, n-1)" in printed
    inputs = report["inputs"]
    assert inputs["grow_seeds"] == [0, 1, 2, 3, 4]  # the new default
    assert (inputs["window"], inputs["confidence"], inputs["min_seeds"]) == (30, 0.95, 3)
    assert report["inventory"]["missing"] == [
        f"{arm} C8 seed{seed}" for arm in ("fc", "fc-sham") for seed in (3, 4)
    ]
    assert [arm["arm"] for arm in report["arms"]] == ["fc", "fc-sham"]
    run = next(r for r in report["runs"] if r["arm"] == "fc" and r["seed"] == 0)
    assert run["variant"] == _variant()
    assert run["window"]["requested"] == 30 and run["deltas_pp"]["window"] == pytest.approx(1.0)
    assert run["diagnostics_summary"]["final"]["off_drop_pp"] == pytest.approx(1.5)
    real, sham = report["cells"]
    assert real["arm"] == "fc" and sham["arm"] == "fc-sham"
    macs = real["break_even"]["macs"]
    assert macs["best"]["lcb_pp"] == pytest.approx(1.0)
    assert macs["best"]["lcb_margin_pp"] == pytest.approx(macs["best"]["margin_pp"])
    assert real["verdicts"]["macs"] == macs["verdict"]
    assert real["diagnostics"]["off_drop_pp"]["mean"] == pytest.approx(1.5)
    assert sham["break_even"] is None and sham["verdicts"]["params"] == grow_report.GATE_SHAM
    assert report["noise_floor"][0]["arm"] == "fc-sham"


@pytest.mark.parametrize(
    "flags", [["--window", "0"], ["--confidence", "0.4"], ["--confidence", "1"], ["--min-seeds", "0"]]
)
def test_main_rejects_meaningless_gate_settings(tmp_path, flags):
    with pytest.raises(SystemExit):
        grow_report.main(["--grow-root", str(tmp_path), *flags])


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_scratch_cost_is_measured_from_the_paper_configs():
    assert grow_report.scratch_cost(8) == {"params": 2356, "macs": 177_336}
    assert grow_report.scratch_cost(12) == {"params": 3400, "macs": 277_124}
