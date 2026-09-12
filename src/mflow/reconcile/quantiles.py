"""Quantile reconciliation and crossing repair (M5.3).

A reconciler acts on point forecasts, but the evaluation is probabilistic, so the whole
quantile fan has to be carried through. Two strategies are implemented and compared in
the E3 ablation:

``shift``
    Reconcile the median, then shift every quantile of a series by the same per-series,
    per-step correction. The predictive shape is preserved exactly and the median fan is
    coherent. Because the shift is common to all levels, the reconciled quantiles remain
    a valid, ordered set for each series, and the *median* path satisfies conservation.

``per_quantile``
    Reconcile each quantile level independently. Each level is then individually
    coherent, which is what a user reading, say, the 90th percentile of every room at
    once would want. This does **not** give a probabilistically coherent joint
    distribution: the reconciled marginals no longer correspond to any single joint law
    over the constrained space, and quantile crossings introduced by the projection have
    to be repaired by sorting afterwards. The paper states this limitation explicitly
    rather than presenting per-quantile reconciliation as distributionally valid.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from mflow.reconcile.constraints import ConstraintSystem
from mflow.reconcile.projection import (
    Reconciler,
    ReconciliationResult,
    sigma_from_quantiles,
)

QuantileStrategy = Literal["shift", "per_quantile"]


def enforce_monotone(quantile_forecast: np.ndarray) -> np.ndarray:
    """Repair quantile crossings by sorting along the quantile axis.

    Sorting is the minimal repair in the sense that it is the projection of the crossed
    vector onto the monotone cone under the Euclidean norm, and it leaves the set of
    predicted values unchanged.

    Args:
        quantile_forecast: ``(..., n_quantiles)`` with ascending levels.

    Returns:
        An array of the same shape with a non-decreasing last axis.
    """
    return np.sort(np.asarray(quantile_forecast), axis=-1)


class QuantileReconciler:
    """Lift a point :class:`~mflow.reconcile.projection.Reconciler` to a quantile fan.

    Args:
        reconciler: the point reconciler to apply.
        strategy: ``shift`` or ``per_quantile``.
        median_level: the level treated as the point forecast, normally 0.5.
    """

    def __init__(
        self,
        reconciler: Reconciler,
        *,
        strategy: QuantileStrategy = "shift",
        median_level: float = 0.5,
    ) -> None:
        if strategy not in ("shift", "per_quantile"):
            raise ValueError(f"unknown quantile strategy {strategy!r}")
        self.reconciler = reconciler
        self.strategy: QuantileStrategy = strategy
        self.median_level = median_level

    @property
    def name(self) -> str:
        """Composite name used in results tables, e.g. ``proposed+shift``."""
        return f"{self.reconciler.name}+{self.strategy}"

    def reconcile(
        self,
        quantile_forecast: np.ndarray,
        quantile_levels: list[float] | tuple[float, ...],
        system: ConstraintSystem,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, ReconciliationResult]:
        """Reconcile a full quantile forecast.

        Args:
            quantile_forecast: ``(n_series, H, n_quantiles)``, ascending levels.
            quantile_levels: the levels, matching the last axis.
            system: the constraint system.
            rhs: right-hand side for this origin.

        Returns:
            ``(reconciled_quantiles, diagnostics)`` where the diagnostics describe the
            median path, which is the path the conservation claims are made about.
        """
        levels = [round(float(q), 6) for q in quantile_levels]
        if quantile_forecast.shape[-1] != len(levels):
            raise ValueError(
                f"forecast has {quantile_forecast.shape[-1]} quantile columns but "
                f"{len(levels)} levels were given"
            )
        try:
            median_index = levels.index(round(self.median_level, 6))
        except ValueError as exc:
            raise ValueError(
                f"the median level {self.median_level} is not among {levels}"
            ) from exc

        sigma = sigma_from_quantiles(quantile_forecast, levels)
        median = np.asarray(quantile_forecast[..., median_index], dtype=np.float64)

        if self.strategy == "shift":
            result = self.reconciler.reconcile(median, system, rhs, sigma=sigma)
            correction = (result.reconciled - median)[..., None]
            shifted = np.asarray(quantile_forecast, dtype=np.float64) + correction
            return enforce_monotone(shifted), result

        reconciled_levels = []
        median_result: ReconciliationResult | None = None
        for index in range(len(levels)):
            level_forecast = np.asarray(quantile_forecast[..., index], dtype=np.float64)
            level_result = self.reconciler.reconcile(level_forecast, system, rhs, sigma=sigma)
            reconciled_levels.append(level_result.reconciled)
            if index == median_index:
                median_result = level_result
        assert median_result is not None
        stacked = np.stack(reconciled_levels, axis=-1)
        return enforce_monotone(stacked), median_result


def crossing_rate(quantile_forecast: np.ndarray) -> float:
    """Fraction of adjacent quantile pairs that are out of order.

    Reported for the ``per_quantile`` strategy so that the cost of independent
    reconciliation is visible rather than hidden by the repair.
    """
    diffs = np.diff(np.asarray(quantile_forecast), axis=-1)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs < 0.0))
