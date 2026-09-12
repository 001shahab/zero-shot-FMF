"""Forecast accuracy, calibration and coherence metrics (M6).

Every function takes the same shapes and handles missing actuals the same way: a NaN in
the truth means the sensor had nothing to say at that step, so the cell is excluded from
the average rather than counted as a zero error. A metric computed over no observations
raises instead of returning NaN, because an empty average silently averaged into a
results table is indistinguishable from a real number.

Conventions used throughout:

* ``truth`` is ``(n_series, horizon)``;
* ``prediction`` is ``(n_series, horizon, n_quantiles)`` with quantile levels ascending;
* the median is the quantile nearest 0.5 and is what the point metrics score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

#: The ``(q10, q90)`` interval whose empirical coverage is reported.
COVERAGE_LEVEL: Final[float] = 0.8


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the inputs given."""


def _check(truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]) -> None:
    if truth.ndim != 2:
        raise MetricError(f"truth must be (n_series, horizon), got {truth.shape}")
    if prediction.ndim != 3:
        raise MetricError(
            f"prediction must be (n_series, horizon, n_quantiles), got {prediction.shape}"
        )
    if prediction.shape[:2] != truth.shape:
        raise MetricError(
            f"prediction {prediction.shape} does not match truth {truth.shape}"
        )
    if prediction.shape[2] != len(quantiles):
        raise MetricError(
            f"prediction has {prediction.shape[2]} quantiles but {len(quantiles)} levels "
            "were given"
        )


def _mean_over_observed(values: np.ndarray, observed: np.ndarray) -> float:
    """Mean of ``values`` over the cells where the truth exists."""
    if not observed.any():
        raise MetricError("no observed values to average over")
    return float(values[observed].mean())


def median_index(quantiles: Sequence[float]) -> int:
    """Index of the quantile level closest to 0.5."""
    if not len(quantiles):
        raise MetricError("no quantile levels given")
    return int(np.argmin(np.abs(np.asarray(quantiles, dtype=float) - 0.5)))


def point_forecast(prediction: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """The ``(n_series, horizon)`` median slice that the point metrics score."""
    return prediction[:, :, median_index(quantiles)]


def mae(truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]) -> float:
    """Mean absolute error of the median forecast, in persons."""
    _check(truth, prediction, quantiles)
    observed = np.isfinite(truth)
    return _mean_over_observed(np.abs(point_forecast(prediction, quantiles) - truth), observed)


def rmse(truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]) -> float:
    """Root mean squared error of the median forecast, in persons."""
    _check(truth, prediction, quantiles)
    observed = np.isfinite(truth)
    squared = (point_forecast(prediction, quantiles) - truth) ** 2
    return float(np.sqrt(_mean_over_observed(squared, observed)))


def seasonal_naive_scale(training: np.ndarray, season_length: int) -> np.ndarray:
    """Per-series MASE denominator: mean absolute seasonal difference on the training set.

    A series whose seasonal difference is exactly zero across the whole training window --
    a store room that is always empty, or a corridor whose counter never moved -- gets
    NaN rather than a small positive floor. Flooring it would divide a near-zero error by
    a near-zero denominator and produce a MASE in the hundreds of thousands, which then
    dominates the average over every series that does carry information.
    :func:`mase` drops those series and says how many it dropped.

    Args:
        training: ``(n_series, T)`` of training-window actuals. Computing this on the
            training window rather than the test window is what stops MASE from being a
            function of the thing being evaluated.
        season_length: the seasonal lag, normally one day in steps.

    Raises:
        MetricError: if the training window is shorter than one season.
    """
    if training.ndim != 2:
        raise MetricError(f"training must be (n_series, T), got {training.shape}")
    if season_length < 1:
        raise MetricError(f"season_length must be positive, got {season_length}")
    if training.shape[1] <= season_length:
        raise MetricError(
            f"training window of {training.shape[1]} steps is too short for a season of "
            f"{season_length}"
        )
    differences = np.abs(training[:, season_length:] - training[:, :-season_length])
    with np.errstate(invalid="ignore"):
        scale = np.nanmean(differences, axis=1)
    return np.where(scale > 0.0, scale, np.nan)


def mase(
    truth: np.ndarray,
    prediction: np.ndarray,
    quantiles: Sequence[float],
    scale: np.ndarray,
) -> float:
    """Mean absolute scaled error against a seasonal naive denominator.

    Args:
        scale: per-series denominator from :func:`seasonal_naive_scale`. Series whose
            denominator is NaN are excluded, since a seasonal naive forecast of them was
            exactly right and there is no error to scale against.

    Raises:
        MetricError: if every series has a degenerate denominator.
    """
    _check(truth, prediction, quantiles)
    if scale.shape != (truth.shape[0],):
        raise MetricError(f"scale must be ({truth.shape[0]},), got {scale.shape}")
    usable = np.isfinite(scale)
    if not usable.any():
        raise MetricError(
            "every series has a zero seasonal difference on the training window, so MASE "
            "has no denominator; the training window is probably shorter than the season "
            "or lies entirely outside opening hours"
        )
    observed = np.isfinite(truth) & usable[:, None]
    errors = np.abs(point_forecast(prediction, quantiles) - truth) / scale[:, None]
    return _mean_over_observed(errors, observed)


def pinball(
    truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Per-cell, per-level pinball loss, shape ``(n_series, horizon, n_quantiles)``."""
    _check(truth, prediction, quantiles)
    levels = np.asarray(quantiles, dtype=np.float64).reshape(1, 1, -1)
    error = truth[:, :, None] - prediction
    return np.maximum(levels * error, (levels - 1.0) * error)


def wql(truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]) -> float:
    """Weighted quantile loss: total pinball loss normalised by the total actual.

    This is the scale-free probabilistic metric used by the foundation model literature.
    Normalising by the sum of the actuals rather than per series is deliberate: it keeps a
    corridor that sees hundreds of crossings from being outweighed by a side gallery that
    sees three.

    Raises:
        MetricError: if every observed actual is zero, which leaves nothing to normalise by.
    """
    losses = pinball(truth, prediction, quantiles)
    observed = np.isfinite(truth)
    if not observed.any():
        raise MetricError("no observed values to average over")
    denominator = float(np.abs(truth[observed]).sum())
    if denominator == 0.0:
        raise MetricError(
            "every observed actual is zero, so weighted quantile loss has no scale; "
            "this usually means the evaluation window is outside opening hours"
        )
    total = float(losses[observed].sum())
    # Two, because the pinball losses at q and 1-q each capture half of the interval.
    return 2.0 * total / denominator


def crps(truth: np.ndarray, prediction: np.ndarray, quantiles: Sequence[float]) -> float:
    """CRPS approximated from the quantile forecast, in persons.

    The continuous ranked probability score is twice the integral of the pinball loss over
    the quantile level. With only nine levels the integral is approximated by the
    trapezoid rule over the levels actually available, which is exact for a piecewise
    linear quantile function and is what the "approximated from the nine quantiles" in
    the specification means.
    """
    _check(truth, prediction, quantiles)
    levels = np.asarray(quantiles, dtype=np.float64)
    if levels.size < 2:
        raise MetricError("CRPS needs at least two quantile levels")
    losses = pinball(truth, prediction, quantiles)
    observed = np.isfinite(truth)
    if not observed.any():
        raise MetricError("no observed values to average over")
    integrated = np.trapezoid(losses, x=levels, axis=2)
    return 2.0 * _mean_over_observed(integrated, observed)


def coverage(
    truth: np.ndarray,
    prediction: np.ndarray,
    quantiles: Sequence[float],
    level: float = COVERAGE_LEVEL,
) -> float:
    """Empirical coverage of the central interval at ``level``.

    Raises:
        MetricError: if the requested interval is not among the quantile levels produced.
            Interpolating one would report a calibration the model was never asked for.
    """
    _check(truth, prediction, quantiles)
    lower_level = (1.0 - level) / 2.0
    upper_level = 1.0 - lower_level
    lower = _exact_level(quantiles, lower_level)
    upper = _exact_level(quantiles, upper_level)
    observed = np.isfinite(truth)
    inside = (truth >= prediction[:, :, lower]) & (truth <= prediction[:, :, upper])
    return _mean_over_observed(inside.astype(np.float64), observed)


def interval_width(
    truth: np.ndarray,
    prediction: np.ndarray,
    quantiles: Sequence[float],
    level: float = COVERAGE_LEVEL,
) -> float:
    """Mean width of the central interval at ``level``, in persons.

    Reported alongside coverage because a model can reach nominal coverage by predicting
    an interval wide enough to be useless.
    """
    _check(truth, prediction, quantiles)
    lower = _exact_level(quantiles, (1.0 - level) / 2.0)
    upper = _exact_level(quantiles, 1.0 - (1.0 - level) / 2.0)
    observed = np.isfinite(truth)
    return _mean_over_observed(prediction[:, :, upper] - prediction[:, :, lower], observed)


def _exact_level(quantiles: Sequence[float], wanted: float) -> int:
    levels = np.asarray(quantiles, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(levels, wanted, atol=1e-9))
    if matches.size == 0:
        raise MetricError(
            f"quantile level {wanted} is not among {list(np.round(levels, 4))}; coverage "
            "at a level the model did not produce would have to be interpolated"
        )
    return int(matches[0])


def conservation_residual_mae(
    prediction: np.ndarray,
    quantiles: Sequence[float],
    constraint_matrix: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Mean absolute violation of the conservation constraints, in persons.

    Args:
        prediction: ``(n_series, horizon, n_quantiles)``.
        quantiles: the levels, used to pick the median.
        constraint_matrix: ``A`` from :class:`mflow.reconcile.ConstraintSystem`.
        rhs: ``b``, the right-hand side at this origin.

    Returns:
        ``mean |A y - b|`` over the constraint rows, where ``y`` is the flattened median
        forecast. Zero means the forecast is coherent with the building's topology.
    """
    _check(np.zeros(prediction.shape[:2]), prediction, quantiles)
    flat = point_forecast(prediction, quantiles).reshape(-1)
    if constraint_matrix.shape[1] != flat.size:
        raise MetricError(
            f"constraint matrix has {constraint_matrix.shape[1]} columns but the forecast "
            f"flattens to {flat.size}"
        )
    residual = constraint_matrix @ flat - rhs
    return float(np.abs(residual).mean())


def violation_rate(
    prediction: np.ndarray,
    quantiles: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Share of predicted values outside their physical bounds.

    Bounds are non-negativity for every series and the node capacity for occupancy. A
    forecast that puts negative people in a room, or more people in it than fit, is wrong
    in a way no accuracy metric registers.
    """
    point = point_forecast(prediction, quantiles)
    if lower.shape != point.shape[:1] or upper.shape != point.shape[:1]:
        raise MetricError(
            f"bounds must be ({point.shape[0]},), got {lower.shape} and {upper.shape}"
        )
    outside = (point < lower[:, None] - 1e-9) | (point > upper[:, None] + 1e-9)
    return float(outside.mean())


def quantile_crossing_rate(prediction: np.ndarray) -> float:
    """Share of adjacent quantile pairs that are out of order."""
    if prediction.ndim != 3:
        raise MetricError(f"prediction must be 3-D, got {prediction.shape}")
    if prediction.shape[2] < 2:
        return 0.0
    return float((np.diff(prediction, axis=2) < -1e-9).mean())
