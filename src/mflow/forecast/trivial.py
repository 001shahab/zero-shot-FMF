"""Trivial baselines: the floor every other method has to clear.

These are not filler. The point of the paper is whether a zero-shot foundation model is
useful in a museum, and the honest test of that is whether it beats a seasonal naive that
costs nothing to run. Ground rule: if a foundation model does not clearly beat these,
that is the result, and it gets reported rather than tuned away.

All three produce genuine predictive intervals from the empirical distribution of their
own historical errors, so the probabilistic metrics compare like with like instead of
pitting a calibrated foundation model against a point forecast dressed up with a
constant band.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    ForecastError,
    Forecaster,
    fill_context,
    season_length,
    validate_quantiles,
)
from mflow.schema import Panel


class _EmpiricalResidualMixin:
    """Turn a point forecast into a fan using the empirical residual distribution.

    The residuals come from applying the same rule inside the context window, so nothing
    beyond the origin is used. When the window is too short to produce residuals the fan
    collapses to the point forecast, which is the honest representation of "this baseline
    has no information about its own uncertainty here".
    """

    @staticmethod
    def _fan(
        point: np.ndarray, residuals: np.ndarray, quantiles: Sequence[float]
    ) -> np.ndarray:
        n_series, horizon = point.shape
        fan = np.empty((n_series, horizon, len(quantiles)), dtype=np.float64)
        for i in range(n_series):
            usable = residuals[i][np.isfinite(residuals[i])]
            offsets = (
                np.quantile(usable, quantiles)
                if usable.size >= 5
                else np.zeros(len(quantiles))
            )
            fan[i] = point[i][:, None] + offsets[None, :]
        return np.maximum(fan, 0.0)


class LastValue(_EmpiricalResidualMixin, Forecaster):
    """Repeat the last observation across the horizon (the random-walk forecast)."""

    name = "last_value"
    requires_training = False

    def fit(self, panel: Panel) -> None:
        """No-op; the last value is read from the context at prediction time."""
        self._assert_zero_shot(panel)

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        context = fill_context(panel.series)
        point = np.repeat(context[:, -1:], horizon, axis=1)
        residuals = np.diff(context, axis=1)
        return self._validate_output(
            self._fan(point, residuals, levels), panel, horizon, levels, self.name
        )


class SeasonalNaive(_EmpiricalResidualMixin, Forecaster):
    """Repeat the value observed one daily season ago.

    Args:
        season: season length in steps. Defaults to one day, derived from the panel's
            sampling interval.
    """

    name = "seasonal_naive"
    requires_training = False

    def __init__(self, *, season: int | None = None, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.season = season

    def _season_for(self, panel: Panel) -> int:
        return self.season if self.season is not None else season_length(panel.interval_seconds)

    def fit(self, panel: Panel) -> None:
        """No-op; the seasonal lag is read from the context at prediction time."""
        self._assert_zero_shot(panel)

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        season = self._season_for(panel)
        context = fill_context(panel.series)
        n_time = context.shape[1]
        if n_time < season:
            raise ForecastError(
                f"{self.name} needs at least one full season of context ({season} steps) "
                f"but was given {n_time}. Shorten the season or lengthen the context "
                "rather than silently degrading to a last-value forecast."
            )

        # Step h reuses the observation season steps earlier, wrapping for horizons that
        # exceed one season.
        offsets = [(-season + (h % season)) for h in range(horizon)]
        point = np.stack([context[:, n_time + o] for o in offsets], axis=1)
        residuals = context[:, season:] - context[:, :-season]
        return self._validate_output(
            self._fan(point, residuals, levels), panel, horizon, levels, self.name
        )


class HistoricalAverage(_EmpiricalResidualMixin, Forecaster):
    """Mean by (time of day, day type), the standard operational baseline.

    A museum's own planning spreadsheet is essentially this model, so beating it is the
    minimum bar for a forecasting system to be worth deploying.

    Args:
        day_type: ``weekday_weekend`` splits Monday-Friday from Saturday-Sunday;
            ``dow`` keeps all seven days separate, which needs a much longer context.
    """

    name = "historical_average"
    requires_training = False

    def __init__(self, *, day_type: str = "weekday_weekend", seed: int = 0) -> None:
        super().__init__(seed=seed)
        if day_type not in ("weekday_weekend", "dow"):
            raise ValueError(f"unknown day_type {day_type!r}")
        self.day_type = day_type

    def _keys(self, panel: Panel, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Bucket key for every context step and every forecast step."""
        step = pd.Timedelta(seconds=panel.interval_seconds)
        future = pd.date_range(
            panel.timestamps[-1] + step, periods=horizon, freq=step, tz=panel.timestamps.tz
        )
        stamps = panel.timestamps.append(future)

        seconds_of_day = (
            stamps.hour.to_numpy() * 3600
            + stamps.minute.to_numpy() * 60
            + stamps.second.to_numpy()
        )
        slot = seconds_of_day // panel.interval_seconds
        dow = stamps.dayofweek.to_numpy()
        group = dow if self.day_type == "dow" else (dow >= 5).astype(np.int64)
        keys = group * 100_000 + slot
        return keys[: len(panel.timestamps)], keys[len(panel.timestamps) :]

    def fit(self, panel: Panel) -> None:
        """No-op; the averages are computed from the context at prediction time."""
        self._assert_zero_shot(panel)

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        context = fill_context(panel.series)
        context_keys, future_keys = self._keys(panel, horizon)

        n_series = context.shape[0]
        point = np.empty((n_series, horizon), dtype=np.float64)
        residuals = np.full_like(context, np.nan)
        overall = context.mean(axis=1)

        for i in range(n_series):
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            for key, value in zip(context_keys, context[i], strict=True):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
            means = {k: sums[k] / counts[k] for k in sums}
            # Buckets never seen in the context fall back to the series mean; that is a
            # visible, documented choice rather than an arbitrary zero.
            point[i] = [means.get(int(k), float(overall[i])) for k in future_keys]
            residuals[i] = context[i] - np.array(
                [means.get(int(k), float(overall[i])) for k in context_keys]
            )

        return self._validate_output(
            self._fan(point, residuals, levels), panel, horizon, levels, self.name
        )
