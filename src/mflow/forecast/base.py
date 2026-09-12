"""The forecaster interface (M4).

Every method in the study -- a seasonal naive, a gradient booster, a graph neural network
and a 330M-parameter foundation model -- is used through this one interface, so the
evaluation harness cannot accidentally treat them differently.

The output contract is deliberately strict: ``(n_series, horizon, n_quantiles)``, float32,
no NaNs, non-decreasing along the quantile axis. Anything looser would push per-model
special cases into the metrics code, which is where silent bugs become published numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from mflow.profiling import ResourceUsage
from mflow.schema import Panel

#: The nine levels used throughout the study, matching the TimesFM quantile head.
DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ForecastError(RuntimeError):
    """Raised when a forecaster cannot produce a valid prediction.

    Never caught to substitute a fallback forecast: a method that fails on a window must
    be recorded as failing, not quietly replaced by a naive prediction.
    """


class LeakageError(AssertionError):
    """Raised when a zero-shot model is handed data it must not see.

    Ground rule 4 is that foundation models never see evaluation data in any form,
    including for normalisation statistics. ``requires_training = False`` models assert
    this in :meth:`Forecaster.fit`.
    """


class Forecaster(ABC):
    """Common interface for every forecasting method.

    Attributes:
        name: identifier used in configs, result tables and figures.
        requires_training: whether :meth:`fit` does anything. Zero-shot models set this
            to False and must refuse training data.
        supports_multivariate: whether the model conditions each series on the others.
        supports_future_covariates: whether the model can use known-future covariates.
    """

    name: str = "unnamed"
    requires_training: bool = False
    supports_multivariate: bool = False
    supports_future_covariates: bool = False

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._last_usage: ResourceUsage | None = None

    @abstractmethod
    def fit(self, panel: Panel) -> None:
        """Fit on a training panel.

        Zero-shot models implement this as a no-op that asserts it was given nothing it
        should not see.
        """

    @abstractmethod
    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        """Forecast ``horizon`` steps beyond the end of ``panel``.

        Args:
            panel: the context window. Its last timestamp is the forecast origin.
            horizon: number of steps to predict.
            quantiles: ascending predictive levels.

        Returns:
            ``(n_series, horizon, n_quantiles)`` float32, finite, non-decreasing along
            the quantile axis.
        """

    # -- shared machinery ----------------------------------------------------- #

    @property
    def last_usage(self) -> ResourceUsage | None:
        """Latency and peak memory of the most recent :meth:`predict` call."""
        return self._last_usage

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in the run manifest."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "requires_training": self.requires_training,
            "supports_multivariate": self.supports_multivariate,
            "supports_future_covariates": self.supports_future_covariates,
            "seed": self.seed,
        }

    def _assert_zero_shot(self, panel: Panel) -> None:
        """Guard for ``requires_training = False`` models."""
        if self.requires_training:
            return
        raise LeakageError(
            f"{self.name} is zero-shot and was handed a training panel of shape "
            f"{panel.series.shape}; zero-shot models must derive everything, including "
            "normalisation statistics, from the context window at prediction time"
        )

    @staticmethod
    def _validate_output(
        forecast: np.ndarray, panel: Panel, horizon: int, quantiles: Sequence[float], name: str
    ) -> np.ndarray:
        """Enforce the output contract, converting to float32 and repairing ordering.

        Quantile crossings are repaired by sorting, which is a documented, minimal fix.
        NaNs are not repaired: a NaN means the model failed and that must surface.
        """
        expected = (panel.n_series, horizon, len(quantiles))
        if forecast.shape != expected:
            raise ForecastError(
                f"{name} returned shape {forecast.shape}, expected {expected}"
            )
        if not np.isfinite(forecast).all():
            bad = int(np.sum(~np.isfinite(forecast)))
            raise ForecastError(
                f"{name} returned {bad} non-finite values; the output contract forbids "
                "NaN and inf, and imputing them here would hide a broken model"
            )
        return np.sort(forecast, axis=-1).astype(np.float32, copy=False)


def validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Check that ``quantiles`` is a strictly ascending sequence inside ``(0, 1)``."""
    levels = tuple(float(q) for q in quantiles)
    if not levels:
        raise ValueError("at least one quantile level is required")
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError(f"quantile levels must lie strictly inside (0, 1), got {levels}")
    if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
        raise ValueError(f"quantile levels must be strictly ascending, got {levels}")
    return levels


def season_length(interval_seconds: int) -> int:
    """Number of steps in one day, the dominant season for visitor flow.

    Raises:
        ValueError: if the interval does not divide a day, which would make a daily
            seasonal index ill-defined.
    """
    day = 24 * 60 * 60
    if day % interval_seconds != 0:
        raise ValueError(
            f"an interval of {interval_seconds}s does not divide a day evenly, so the "
            "daily seasonal index is not well defined"
        )
    return day // interval_seconds


def fill_context(series: np.ndarray) -> np.ndarray:
    """Fill NaNs in a context window so that arithmetic baselines can run.

    Interior gaps are linearly interpolated and leading gaps are back-filled with the
    first observation; a series that is entirely missing becomes zero. Callers that must
    distinguish "imputed" from "observed" should consult the original array: this
    function deliberately returns only the filled values, and the sensor fault model
    records where it removed data.
    """
    filled = np.array(series, dtype=np.float64, copy=True)
    for row in range(filled.shape[0]):
        values = filled[row]
        missing = np.isnan(values)
        if not missing.any():
            continue
        if missing.all():
            values[:] = 0.0
            continue
        index = np.arange(values.size)
        values[missing] = np.interp(index[missing], index[~missing], values[~missing])
    return filled
