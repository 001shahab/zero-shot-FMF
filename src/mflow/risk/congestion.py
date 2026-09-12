"""Congestion alerting from a probabilistic forecast (M6).

A forecast is only worth deploying if it changes a decision. The decision here is whether
to send a steward to a room before it fills, so the metric is not error but whether the
alert fires, whether it fires in time, and how often it fires for nothing.

An alert is raised at an origin when the forecast says the probability of exceeding the
node's threshold within the horizon is at least ``alert_probability``. Because the
forecast is a quantile fan rather than a distribution, the exceedance probability is read
off the fan: the threshold is located between two quantile levels and the probability is
interpolated between them. That is a coarse estimate and it is deliberately coarse -- nine
levels is what the models give, and fitting a parametric distribution to them would
invent tail behaviour that no model produced.

The comparison is against a reactive detector, which alerts only once the threshold is
already exceeded. The reactive detector has perfect precision by construction and zero
lead time, which is exactly the trade-off the forecast is supposed to improve on. Quoting
the forecast's recall without that baseline would be meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class CongestionError(ValueError):
    """Raised when an alerting evaluation cannot be carried out as specified."""


@dataclass(frozen=True)
class AlertOutcome:
    """Alerting performance of one detector.

    Attributes:
        precision: share of alerts followed by a real breach within the horizon.
        recall: share of real breaches that were alerted.
        median_lead_time_steps: median number of steps between the alert and the breach
            it anticipated, over the breaches that were caught. NaN when none were.
        false_alarms_per_node_per_day: nuisance rate, which is what determines whether
            staff keep listening to the system.
        n_alerts: alerts raised.
        n_breaches: real breaches in the evaluation window.
        n_caught: breaches anticipated by an alert.
    """

    precision: float
    recall: float
    median_lead_time_steps: float
    false_alarms_per_node_per_day: float
    n_alerts: int
    n_breaches: int
    n_caught: int

    def lead_time_minutes(self, interval_seconds: int) -> float:
        """Median lead time in minutes, for a site sampled at ``interval_seconds``."""
        return self.median_lead_time_steps * interval_seconds / 60.0

    def as_dict(self, interval_seconds: int) -> dict[str, float]:
        """Flat mapping for a results table."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "median_lead_time_minutes": self.lead_time_minutes(interval_seconds),
            "false_alarms_per_node_per_day": self.false_alarms_per_node_per_day,
            "n_alerts": float(self.n_alerts),
            "n_breaches": float(self.n_breaches),
            "n_caught": float(self.n_caught),
        }


def exceedance_probability(
    quantile_forecast: np.ndarray,
    quantile_levels: Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Probability that the value exceeds ``threshold``, read off the quantile fan.

    Args:
        quantile_forecast: ``(..., n_quantiles)`` with ascending levels.
        quantile_levels: the levels.
        threshold: the value to exceed.

    Returns:
        An array shaped like ``quantile_forecast`` without its last axis.

    The fan is treated as a piecewise linear quantile function. Below the lowest
    quantile the probability is capped at ``1 - min(level)`` and above the highest at
    ``1 - max(level)``, rather than extrapolated: with a lowest level of 0.1 the model has
    said nothing about the bottom decile, and inventing a value there would put confident
    numbers on the tail that drives every alerting decision.
    """
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if quantile_forecast.shape[-1] != levels.size:
        raise CongestionError(
            f"forecast has {quantile_forecast.shape[-1]} quantile columns but "
            f"{levels.size} levels were given"
        )
    if levels.size < 2:
        raise CongestionError("need at least two quantile levels to interpolate")
    if np.any(np.diff(levels) <= 0):
        raise CongestionError(f"quantile levels must be strictly ascending, got {levels}")

    values = np.asarray(quantile_forecast, dtype=np.float64)
    flat = values.reshape(-1, levels.size)
    # np.interp needs an ascending x, which the quantile values are by construction once
    # the forecast has been made monotone.
    cdf = np.empty(flat.shape[0], dtype=np.float64)
    for index in range(flat.shape[0]):
        cdf[index] = np.interp(threshold, flat[index], levels)
    return (1.0 - cdf).reshape(values.shape[:-1])


def forecast_alerts(
    quantile_forecast: np.ndarray,
    quantile_levels: Sequence[float],
    threshold: float,
    alert_probability: float,
) -> np.ndarray:
    """Whether a breach is predicted within the horizon, per origin.

    Args:
        quantile_forecast: ``(n_origins, horizon, n_quantiles)`` for one node.
        quantile_levels: the levels.
        threshold: the occupancy at which the node counts as congested.
        alert_probability: the exceedance probability at which the alert fires.

    Returns:
        A boolean array of length ``n_origins``.
    """
    if quantile_forecast.ndim != 3:
        raise CongestionError(
            f"expected (n_origins, horizon, n_quantiles), got {quantile_forecast.shape}"
        )
    if not 0.0 < alert_probability < 1.0:
        raise CongestionError(
            f"alert_probability must lie strictly inside (0, 1), got {alert_probability}"
        )
    probability = exceedance_probability(quantile_forecast, quantile_levels, threshold)
    return (probability >= alert_probability).any(axis=1)


def first_breach_step(actual: np.ndarray, threshold: float) -> np.ndarray:
    """Index of the first step in each horizon where the truth exceeds ``threshold``.

    Args:
        actual: ``(n_origins, horizon)`` of actuals.
        threshold: the congestion threshold.

    Returns:
        Integer array of length ``n_origins``; ``-1`` where no breach occurred. Missing
        actuals are not breaches, since an unobserved room cannot be confirmed congested.
    """
    breached = np.isfinite(actual) & (actual > threshold)
    any_breach = breached.any(axis=1)
    first = np.argmax(breached, axis=1)
    return np.where(any_breach, first, -1)


def score_alerts(
    alerts: np.ndarray,
    breach_step: np.ndarray,
    *,
    steps_per_day: int,
    n_nodes: int = 1,
    n_origins_per_step: float = 1.0,
) -> AlertOutcome:
    """Precision, recall, lead time and nuisance rate for one detector.

    Args:
        alerts: whether the detector fired at each origin.
        breach_step: from :func:`first_breach_step`.
        steps_per_day: sampling steps in a day, used to turn a count of false alarms into
            a rate an operations manager can reason about.
        n_nodes: how many nodes these origins cover, for the per-node rate.
        n_origins_per_step: how many origins fall in one sampling step, which is the
            reciprocal of the stride. An alerting system evaluated every 5 steps produces
            a fifth as many nuisance alerts per day as one evaluated every step.

    Raises:
        CongestionError: if the arrays are not aligned by origin.
    """
    if alerts.shape != breach_step.shape:
        raise CongestionError(
            f"alerts {alerts.shape} and breaches {breach_step.shape} must be aligned"
        )
    if alerts.ndim != 1:
        raise CongestionError(f"expected one value per origin, got {alerts.shape}")

    will_breach = breach_step >= 0
    caught = alerts & will_breach
    n_alerts = int(alerts.sum())
    n_breaches = int(will_breach.sum())
    n_caught = int(caught.sum())
    false_alarms = n_alerts - n_caught

    precision = n_caught / n_alerts if n_alerts else float("nan")
    recall = n_caught / n_breaches if n_breaches else float("nan")
    lead = breach_step[caught]
    median_lead = float(np.median(lead)) if lead.size else float("nan")

    n_origins = alerts.shape[0]
    days = n_origins / (steps_per_day * n_origins_per_step)
    rate = false_alarms / (days * n_nodes) if days > 0 else float("nan")

    return AlertOutcome(
        precision=precision,
        recall=recall,
        median_lead_time_steps=median_lead,
        false_alarms_per_node_per_day=rate,
        n_alerts=n_alerts,
        n_breaches=n_breaches,
        n_caught=n_caught,
    )


def reactive_alerts(context_last: np.ndarray, threshold: float) -> np.ndarray:
    """The baseline detector: alert when the threshold is already exceeded.

    Args:
        context_last: the observed value at each origin, i.e. the last value the detector
            could have seen.
        threshold: the congestion threshold.

    This is what a building already does without a forecast. Its lead time is zero or
    negative by construction, so any lead time the forecast buys is the whole benefit.
    """
    return np.isfinite(context_last) & (context_last > threshold)


def node_thresholds(
    capacity: dict[str, float], *, occupancy_fraction: float
) -> dict[str, float]:
    """Congestion threshold per node as a fraction of its capacity.

    Args:
        capacity: persons per node.
        occupancy_fraction: the share of capacity at which a room counts as congested.
            Expressing the threshold relative to capacity rather than as an absolute
            headcount is what lets one configuration apply to a study and a great hall.

    Raises:
        CongestionError: if the fraction is not in ``(0, 1]``.
    """
    if not 0.0 < occupancy_fraction <= 1.0:
        raise CongestionError(
            f"occupancy_fraction must lie in (0, 1], got {occupancy_fraction}"
        )
    return {node: value * occupancy_fraction for node, value in capacity.items()}
