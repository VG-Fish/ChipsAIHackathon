"""Pareto frontier tracking and the framework's stopping rule.

Step 3f stops the sparsity sweep when the *frontier* stops improving, not when
accuracy stops increasing.  Those are different stopping points: a candidate
that gives up a little accuracy but halves latency still extends the frontier
and the search must continue past it, while a candidate that is worse on every
axis than something already found adds nothing even if its accuracy went up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kws.utils.logging import get_logger

logger = get_logger(__name__)

# Deployment costs the frontier trades accuracy against. All are minimized.
DEFAULT_COST_KEYS = (
    "deployed_params",
    "macs",
    "latency_ms_p50",
    "weight_bytes",
    "activation_peak_bytes",
)


@dataclass(frozen=True)
class ParetoPoint:
    """One evaluated candidate: accuracy to maximize, costs to minimize."""

    label: str
    accuracy: float
    costs: dict[str, float]
    detail: dict = field(default_factory=dict)

    def cost(self, key: str) -> float:
        if key not in self.costs:
            raise KeyError(f"candidate {self.label!r} has no recorded cost {key!r}")
        return float(self.costs[key])

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "accuracy": self.accuracy,
            "costs": dict(self.costs),
            **self.detail,
        }


def dominates(a: ParetoPoint, b: ParetoPoint, cost_keys: tuple[str, ...]) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better somewhere."""
    if a.accuracy < b.accuracy:
        return False
    if any(a.cost(key) > b.cost(key) for key in cost_keys):
        return False
    return a.accuracy > b.accuracy or any(
        a.cost(key) < b.cost(key) for key in cost_keys
    )


def materially_dominates(
    a: ParetoPoint,
    b: ParetoPoint,
    cost_keys: tuple[str, ...],
    *,
    accuracy_tolerance: float = 0.0,
    relative_cost_tolerance: float = 0.0,
) -> bool:
    """Return whether ``a`` beats ``b`` by more than configured noise margins.

    A sweep can produce rounded or noisy measurements.  Treating a 0.1% cost
    change as progress makes patience depend on measurement noise rather than
    on a useful deployment improvement.  The exact ``dominates`` function is
    intentionally kept for mathematical frontier queries; this tolerant form
    is used by the incremental stopping rule.
    """
    if accuracy_tolerance < 0 or relative_cost_tolerance < 0:
        raise ValueError("Pareto tolerances must be non-negative")
    if a.accuracy + accuracy_tolerance < b.accuracy:
        return False
    if any(
        a.cost(key) > b.cost(key) * (1.0 + relative_cost_tolerance)
        for key in cost_keys
    ):
        return False
    return a.accuracy > b.accuracy + accuracy_tolerance or any(
        a.cost(key) < b.cost(key) * (1.0 - relative_cost_tolerance)
        for key in cost_keys
    )


def materially_equivalent(
    a: ParetoPoint,
    b: ParetoPoint,
    cost_keys: tuple[str, ...],
    *,
    accuracy_tolerance: float = 0.0,
    relative_cost_tolerance: float = 0.0,
) -> bool:
    """Whether two points are indistinguishable for stopping purposes."""
    if abs(a.accuracy - b.accuracy) > accuracy_tolerance:
        return False
    return all(
        abs(a.cost(key) - b.cost(key))
        <= abs(b.cost(key)) * relative_cost_tolerance
        for key in cost_keys
    )


def pareto_front(
    points: list[ParetoPoint], cost_keys: tuple[str, ...] = DEFAULT_COST_KEYS,
) -> list[ParetoPoint]:
    """The non-dominated subset, ordered by descending accuracy."""
    front = [
        point
        for point in points
        if not any(dominates(other, point, cost_keys) for other in points if other is not point)
    ]
    return sorted(front, key=lambda point: -point.accuracy)


@dataclass(frozen=True)
class FrontierUpdate:
    """What one candidate did to the frontier."""

    point: ParetoPoint
    extended_frontier: bool
    dominated_by: list[str]
    displaced: list[str]
    admitted: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "extended_frontier": self.extended_frontier,
            "dominated_by": list(self.dominated_by),
            "displaced": list(self.displaced),
            "admitted": self.admitted,
            "reason": self.reason,
        }


class ParetoFrontier:
    """Incremental frontier with a patience-based "stopped improving" rule.

    ``minimum_accuracy`` is a hard admission constraint, not a stopping rule --
    a candidate below the floor is simply not deployable, so it never joins the
    frontier, but the search only ends once ``patience`` consecutive candidates
    in a row have failed to extend it.
    """

    def __init__(
        self,
        *,
        cost_keys: tuple[str, ...] = DEFAULT_COST_KEYS,
        minimum_accuracy: float = 0.0,
        patience: int = 2,
        accuracy_tolerance: float = 0.0,
        relative_cost_tolerance: float = 0.0,
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.cost_keys = tuple(cost_keys)
        self.minimum_accuracy = float(minimum_accuracy)
        self.patience = int(patience)
        if accuracy_tolerance < 0 or relative_cost_tolerance < 0:
            raise ValueError("Pareto tolerances must be non-negative")
        self.accuracy_tolerance = float(accuracy_tolerance)
        self.relative_cost_tolerance = float(relative_cost_tolerance)
        self.points: list[ParetoPoint] = []
        self._frontier: list[ParetoPoint] = []
        self.history: list[FrontierUpdate] = []
        self.stagnant_streak = 0

    @property
    def front(self) -> list[ParetoPoint]:
        return list(self._frontier)

    def add(self, point: ParetoPoint) -> FrontierUpdate:
        if point.accuracy < self.minimum_accuracy:
            update = FrontierUpdate(
                point=point,
                extended_frontier=False,
                dominated_by=[],
                displaced=[],
                admitted=False,
                reason=(
                    f"accuracy {point.accuracy:.4f} is below the "
                    f"{self.minimum_accuracy:.4f} deployment floor"
                ),
            )
        else:
            current = self.front
            dominated_by = [
                existing.label
                for existing in current
                if materially_dominates(
                    existing,
                    point,
                    self.cost_keys,
                    accuracy_tolerance=self.accuracy_tolerance,
                    relative_cost_tolerance=self.relative_cost_tolerance,
                )
                or materially_equivalent(
                    existing,
                    point,
                    self.cost_keys,
                    accuracy_tolerance=self.accuracy_tolerance,
                    relative_cost_tolerance=self.relative_cost_tolerance,
                )
            ]
            displaced = [
                existing.label
                for existing in current
                if materially_dominates(
                    point,
                    existing,
                    self.cost_keys,
                    accuracy_tolerance=self.accuracy_tolerance,
                    relative_cost_tolerance=self.relative_cost_tolerance,
                )
            ]
            extended = not dominated_by
            self.points.append(point)
            if extended:
                self._frontier = [
                    existing
                    for existing in current
                    if existing.label not in displaced
                ] + [point]
            update = FrontierUpdate(
                point=point,
                extended_frontier=extended,
                dominated_by=dominated_by,
                displaced=displaced,
                admitted=True,
                reason=(
                    "extends the Pareto frontier"
                    if extended
                    else "dominated by an existing frontier point"
                ),
            )

        self.stagnant_streak = 0 if update.extended_frontier else self.stagnant_streak + 1
        self.history.append(update)
        logger.info(
            "Pareto: %s acc=%.4f %s (stagnant streak %d/%d)",
            point.label,
            point.accuracy,
            update.reason,
            self.stagnant_streak,
            self.patience,
        )
        return update

    def should_stop(self) -> bool:
        return self.stagnant_streak >= self.patience

    def best_by(self, cost_key: str) -> ParetoPoint | None:
        """The cheapest frontier point on one axis -- the shipping candidate."""
        front = self.front
        if not front:
            return None
        return min(front, key=lambda point: point.cost(cost_key))

    def as_dict(self) -> dict:
        return {
            "cost_keys": list(self.cost_keys),
            "minimum_accuracy": self.minimum_accuracy,
            "patience": self.patience,
            "accuracy_tolerance": self.accuracy_tolerance,
            "relative_cost_tolerance": self.relative_cost_tolerance,
            "stopped": self.should_stop(),
            "stagnant_streak": self.stagnant_streak,
            "evaluated": [point.as_dict() for point in self.points],
            "frontier": [point.as_dict() for point in self.front],
        }
