"""Classical and gradient-boosted baselines.

``GlobalLightGBM`` is the strong tabular baseline: one model per quantile level, trained
across every series of a site, with the forecast step as a feature so that a single model
covers the whole horizon (direct multi-step, no error accumulation from recursion).

``SARIMA`` and ``ETS`` are the textbook univariate baselines, fitted per series through
``statsmodels``. They are slow on long minute-resolution windows, so both carry a time
budget: a series that blows the budget is recorded as a failure for that origin rather
than silently replaced by a simpler model.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastError,
    fill_context,
    season_length,
    validate_quantiles,
)
from mflow.schema import Panel, parse_series_id

#: Lags offered to the booster, in steps. Clipped to what the context can supply.
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60, 120)

#: Rolling window lengths for the moving-average features.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 15, 60)


class GlobalLightGBM(Forecaster):
    """Gradient boosting trained across all series of a site, one model per quantile.

    Args:
        lags: lagged target values offered as features.
        windows: rolling-mean window lengths.
        num_leaves, learning_rate, n_estimators, min_data_in_leaf: LightGBM parameters.
        max_horizon: largest forecast step the model is trained for.
        use_covariates: include past covariates at the origin and known-future
            covariates at the target step.
    """

    name = "global_lightgbm"
    requires_training = True
    supports_multivariate = True  # one model sees every series of the site
    supports_future_covariates = True

    def __init__(
        self,
        *,
        lags: Sequence[int] = DEFAULT_LAGS,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        min_data_in_leaf: int = 20,
        max_horizon: int = 60,
        use_covariates: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.lags = tuple(int(lag) for lag in lags)
        self.windows = tuple(int(w) for w in windows)
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.max_horizon = max_horizon
        self.use_covariates = use_covariates
        self._models: dict[float, Any] = {}
        self._feature_names: list[str] = []
        self._trained_series: list[str] = []

    # -- features -------------------------------------------------------------- #

    def _feature_row(
        self,
        panel: Panel,
        context: np.ndarray,
        series_index: int,
        origin: int,
        step: int,
    ) -> tuple[list[float], list[str]]:
        """Features describing series ``series_index`` at ``origin`` for step ``step``."""
        values: list[float] = []
        names: list[str] = []
        history = context[series_index, : origin + 1]

        for lag in self.lags:
            values.append(float(history[-lag]) if history.size >= lag else np.nan)
            names.append(f"lag_{lag}")
        for window in self.windows:
            window_values = history[-window:]
            values.append(float(window_values.mean()) if window_values.size else np.nan)
            names.append(f"roll_mean_{window}")
            values.append(float(window_values.std()) if window_values.size > 1 else 0.0)
            names.append(f"roll_std_{window}")

        target_time = panel.timestamps[origin] + step * np.timedelta64(
            panel.interval_seconds, "s"
        )
        seconds = target_time.hour * 3600 + target_time.minute * 60 + target_time.second
        fraction = seconds / 86_400.0
        values += [
            float(step),
            float(np.sin(2 * np.pi * fraction)),
            float(np.cos(2 * np.pi * fraction)),
            float(target_time.dayofweek),
            float(target_time.dayofweek >= 5),
        ]
        names += ["horizon_step", "tod_sin", "tod_cos", "dow", "is_weekend"]

        kind, _ = parse_series_id(panel.series_ids[series_index])
        values += [float(series_index), float(kind == "flow")]
        names += ["series_index", "is_flow"]

        if self.use_covariates:
            if panel.past_covariates is not None:
                values += [float(v) for v in panel.past_covariates[:, origin]]
                names += [f"pastcov_{c}" for c in panel.past_covariate_ids]
            if panel.future_covariates is not None:
                column = origin + step
                if column < panel.future_covariates.shape[1]:
                    values += [float(v) for v in panel.future_covariates[:, column]]
                else:
                    values += [np.nan] * len(panel.future_covariate_ids)
                names += [f"futcov_{c}" for c in panel.future_covariate_ids]
        return values, names

    def _design(
        self, panel: Panel, origins: Sequence[int], steps: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build the training design matrix and targets."""
        context = fill_context(panel.series)
        rows: list[list[float]] = []
        targets: list[float] = []
        names: list[str] = []
        for origin in origins:
            for step in steps:
                target_index = origin + step
                if target_index >= context.shape[1]:
                    continue
                for series_index in range(context.shape[0]):
                    features, names = self._feature_row(
                        panel, context, series_index, origin, step
                    )
                    rows.append(features)
                    targets.append(float(context[series_index, target_index]))
        if not rows:
            raise ForecastError(
                f"{self.name}: the training panel is too short to build a single "
                f"(origin, step) example"
            )
        return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), names

    # -- interface -------------------------------------------------------------- #

    def fit(self, panel: Panel) -> None:
        """Train one booster per quantile level on the whole site."""
        import lightgbm as lgb

        min_origin = max([*self.lags, *self.windows])
        usable = panel.n_timesteps - self.max_horizon
        if usable <= min_origin:
            raise ForecastError(
                f"{self.name} needs more than {min_origin + self.max_horizon} steps of "
                f"training data, got {panel.n_timesteps}"
            )
        # Subsample origins so that a 90-day minute-resolution site stays trainable; the
        # stride is deterministic, so the design matrix is a function of the panel alone.
        stride = max(1, usable // 2000)
        origins = list(range(min_origin, usable, stride))
        steps = sorted({1, 5, 15, 30, min(60, self.max_horizon), self.max_horizon})
        steps = [s for s in steps if 1 <= s <= self.max_horizon]

        features, targets, names = self._design(panel, origins, steps)
        self._feature_names = names
        self._trained_series = list(panel.series_ids)
        self._models = {}
        for level in DEFAULT_QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=level,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                min_child_samples=self.min_data_in_leaf,
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(features, targets)
            self._models[level] = model

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        if not self._models:
            raise ForecastError(f"{self.name}.predict called before fit")
        if list(panel.series_ids) != self._trained_series:
            raise ForecastError(
                f"{self.name} was trained on a different set of series than it is being "
                "asked to predict; retrain per site"
            )
        missing = [q for q in levels if q not in self._models]
        if missing:
            raise ForecastError(f"{self.name} has no model for quantile levels {missing}")

        context = fill_context(panel.series)
        origin = panel.n_timesteps - 1
        rows = [
            self._feature_row(panel, context, series_index, origin, step)[0]
            for step in range(1, horizon + 1)
            for series_index in range(panel.n_series)
        ]
        design = np.asarray(rows, dtype=np.float64)

        out = np.empty((panel.n_series, horizon, len(levels)), dtype=np.float64)
        for column, level in enumerate(levels):
            flat = self._models[level].predict(design)
            out[..., column] = np.asarray(flat).reshape(horizon, panel.n_series).T
        return self._validate_output(
            np.maximum(out, 0.0), panel, horizon, levels, self.name
        )


class _StatsmodelsForecaster(Forecaster):
    """Shared plumbing for the per-series statsmodels baselines."""

    requires_training = True

    def __init__(self, *, time_budget_s: float = 20.0, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.time_budget_s = time_budget_s

    def fit(self, panel: Panel) -> None:
        """Nothing is retained between origins.

        Both models are refitted on the context of each origin, which is how they are
        normally used for rolling-origin evaluation and keeps them comparable with the
        zero-shot models: neither sees anything beyond the origin.
        """
        del panel

    def _fit_one(self, values: np.ndarray, horizon: int, quantiles: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        context = fill_context(panel.series)
        out = np.empty((panel.n_series, horizon, len(levels)), dtype=np.float64)
        started = time.perf_counter()
        with warnings.catch_warnings():
            # statsmodels is voluble about convergence on short count series; the
            # information that matters is whether the fit succeeded, which is handled
            # by the explicit budget check below.
            warnings.simplefilter("ignore")
            for i in range(panel.n_series):
                out[i] = self._fit_one(context[i], horizon, levels)
                elapsed = time.perf_counter() - started
                if elapsed > self.time_budget_s:
                    raise ForecastError(
                        f"{self.name} exceeded its {self.time_budget_s:g}s budget after "
                        f"{i + 1}/{panel.n_series} series; raise the budget or drop the "
                        "method for this site rather than reporting a partial forecast"
                    )
        return self._validate_output(np.maximum(out, 0.0), panel, horizon, levels, self.name)


class SARIMA(_StatsmodelsForecaster):
    """Seasonal ARIMA per series.

    Args:
        order: non-seasonal ``(p, d, q)``.
        seasonal_order: ``(P, D, Q)``; the period is the daily season of the panel.
        seasonal_period: override the daily season, in steps. Minute-resolution data has
            a 1440-step day, which is intractable for a seasonal ARIMA, so experiments
            normally set a shorter period or run this method on aggregated data only.
    """

    name = "sarima"

    def __init__(
        self,
        *,
        order: tuple[int, int, int] = (2, 0, 1),
        seasonal_order: tuple[int, int, int] = (1, 0, 0),
        seasonal_period: int | None = None,
        time_budget_s: float = 60.0,
        seed: int = 0,
    ) -> None:
        super().__init__(time_budget_s=time_budget_s, seed=seed)
        self.order = order
        self.seasonal_order = seasonal_order
        self.seasonal_period = seasonal_period
        self._period: int | None = None

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        self._period = self.seasonal_period or min(
            season_length(panel.interval_seconds), 96
        )
        return super().predict(panel, horizon, quantiles)

    def _fit_one(self, values: np.ndarray, horizon: int, quantiles: Sequence[float]) -> np.ndarray:
        from scipy.stats import norm
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        period = self._period or 1
        seasonal = (*self.seasonal_order, period) if period > 1 else (0, 0, 0, 0)
        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend=None,
        )
        result = model.fit(disp=False, maxiter=100)
        forecast = result.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=np.float64)
        sigma = np.sqrt(np.asarray(forecast.var_pred_mean, dtype=np.float64))
        return mean[:, None] + sigma[:, None] * norm.ppf(quantiles)[None, :]


class ETS(_StatsmodelsForecaster):
    """Exponential smoothing (Holt-Winters) per series.

    The predictive fan comes from the empirical distribution of in-sample residuals
    scaled by the square root of the horizon step, which is the standard random-walk
    widening and avoids claiming a Gaussian likelihood the model does not have.
    """

    name = "ets"

    def __init__(
        self,
        *,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_period: int | None = None,
        time_budget_s: float = 60.0,
        seed: int = 0,
    ) -> None:
        super().__init__(time_budget_s=time_budget_s, seed=seed)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period

    def _fit_one(self, values: np.ndarray, horizon: int, quantiles: Sequence[float]) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        seasonal_periods = self.seasonal_period if self.seasonal else None
        model = ExponentialSmoothing(
            values,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        result = model.fit(optimized=True)
        mean = np.asarray(result.forecast(horizon), dtype=np.float64)
        residuals = np.asarray(result.resid, dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        offsets = (
            np.quantile(residuals, quantiles)
            if residuals.size >= 5
            else np.zeros(len(quantiles))
        )
        widening = np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))[:, None]
        return mean[:, None] + widening * offsets[None, :]
