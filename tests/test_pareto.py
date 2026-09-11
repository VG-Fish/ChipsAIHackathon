import pytest

from kws.optimize.pareto import ParetoFrontier, ParetoPoint, dominates, pareto_front


def point(label, accuracy, params, latency):
    return ParetoPoint(
        label=label,
        accuracy=accuracy,
        costs={"deployed_params": params, "latency_ms_p50": latency},
    )


KEYS = ("deployed_params", "latency_ms_p50")


def test_dominance_requires_no_regression_and_one_strict_gain():
    better = point("a", 0.92, 2000, 1.0)
    worse = point("b", 0.90, 2500, 1.2)
    assert dominates(better, worse, KEYS)
    assert not dominates(worse, better, KEYS)

    # Equal on every axis: neither dominates, so both stay on the frontier.
    twin = point("c", 0.92, 2000, 1.0)
    assert not dominates(better, twin, KEYS)


def test_a_cheaper_but_less_accurate_model_is_not_dominated():
    accurate = point("big", 0.95, 3000, 2.0)
    cheap = point("small", 0.91, 1500, 0.8)
    assert not dominates(accurate, cheap, KEYS)
    assert {p.label for p in pareto_front([accurate, cheap], KEYS)} == {"big", "small"}


def test_frontier_keeps_searching_past_an_accuracy_drop():
    """The rule that separates this from "stop when accuracy stops rising"."""
    frontier = ParetoFrontier(cost_keys=KEYS, minimum_accuracy=0.90, patience=2)

    assert frontier.add(point("w18", 0.918, 2946, 2.0)).extended_frontier
    # Less accurate, but cheaper on both cost axes: still extends the frontier.
    update = frontier.add(point("w16", 0.905, 2654, 1.7))
    assert update.extended_frontier
    assert not frontier.should_stop()


def test_frontier_stops_after_patience_consecutive_stalls():
    frontier = ParetoFrontier(cost_keys=KEYS, minimum_accuracy=0.90, patience=2)
    frontier.add(point("w18", 0.918, 2946, 2.0))
    frontier.add(point("w16", 0.905, 2654, 1.7))

    # Dominated: worse accuracy at higher cost than w16.
    assert not frontier.add(point("bad1", 0.901, 2900, 1.9)).extended_frontier
    assert not frontier.should_stop()
    assert not frontier.add(point("bad2", 0.902, 2800, 1.8)).extended_frontier
    assert frontier.should_stop()


def test_below_floor_candidates_are_refused_admission_but_counted():
    frontier = ParetoFrontier(cost_keys=KEYS, minimum_accuracy=0.90, patience=3)
    frontier.add(point("ok", 0.91, 2000, 1.0))

    update = frontier.add(point("low", 0.88, 500, 0.2))
    assert not update.admitted
    assert "floor" in update.reason
    assert [p.label for p in frontier.front] == ["ok"]
    assert frontier.stagnant_streak == 1


def test_exact_duplicate_does_not_reset_patience_or_extend_frontier():
    frontier = ParetoFrontier(cost_keys=KEYS, minimum_accuracy=0.90, patience=2)
    frontier.add(point("original", 0.91, 2000, 1.0))

    update = frontier.add(point("duplicate", 0.91, 2000, 1.0))

    assert not update.extended_frontier
    assert frontier.stagnant_streak == 1
    assert [candidate.label for candidate in frontier.front] == ["original"]


def test_configured_tolerance_requires_material_cost_progress():
    frontier = ParetoFrontier(
        cost_keys=KEYS,
        minimum_accuracy=0.90,
        patience=2,
        relative_cost_tolerance=0.05,
    )
    frontier.add(point("large", 0.92, 2000, 1.0))

    update = frontier.add(point("nearly_same", 0.91, 1950, 0.98))

    assert not update.extended_frontier
    assert frontier.stagnant_streak == 1


def test_best_by_selects_the_cheapest_frontier_point():
    frontier = ParetoFrontier(cost_keys=KEYS, minimum_accuracy=0.0, patience=2)
    frontier.add(point("big", 0.95, 3000, 2.0))
    frontier.add(point("small", 0.91, 1500, 0.8))
    best = frontier.best_by("deployed_params")
    assert best is not None
    assert best.label == "small"


def test_patience_must_be_positive():
    with pytest.raises(ValueError):
        ParetoFrontier(patience=0)
