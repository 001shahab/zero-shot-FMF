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

#: Hyperparameter grid searched when ``tune=True``. Deliberately small: the point of the
#: search is to establish that the gradient-boosted comparison is not lost on an
#: untuned configuration, not to find the global optimum, and every additional cell is
#: paid for nine times over because there is one booster per quantile level.
DEFAULT_TUNE_GRID: tuple[dict[str, float | int], ...] = (
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300, "min_data_in_leaf": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "min_data_in_leaf": 20},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300, "min_data_in_leaf": 20},
    {"num_leaves": 31, "learning_rate": 0.02, "n_estimators": 800, "min_data_in_leaf": 20},
    {"num_leaves": 31, "learning_rate": 0.10, "n_estimators": 150, "min_data_in_leaf": 20},
    {"num_leaves": 63, "learning_rate": 0.02, "n_estimators": 800, "min_data_in_leaf": 50},
)


def seasonal_lags(interval_seconds: int, *, weekly: bool = True) -> tuple[int, ...]:
    """Lags that include the site's own daily and weekly periods.

    :data:`DEFAULT_LAGS` is stated in steps and therefore reaches different wall-clock
    distances on different sites. On a site sampled every fifteen minutes its longest lag
    is thirty hours and it contains no daily lag at all, so the booster is asked to learn
    a daily cycle from calendar features alone while the seasonal baseline it is being
    compared against is handed that lag directly. A referee was right to ask whether the
    gradient-boosted arm was handicapped; this is the lag set that removes the handicap.

    Args:
        interval_seconds: the site's sampling interval.
        weekly: include the weekly lag and its neighbours.
    """
    day = season_length(interval_seconds)
    lags = {1, 2, 3, 5, 10, 15, 30, 60, 120}
    lags |= {day - 1, day, day + 1, 2 * day}
    if weekly:
        lags |= {7 * day - 1, 7 * day, 7 * day + 1}
    return tuple(sorted(lag for lag in lags if lag >= 1))


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
        train_steps: Sequence[int] | None = None,
        use_covariates: bool = True,
        tune: bool = False,
        tune_grid: Sequence[dict[str, float | int]] = DEFAULT_TUNE_GRID,
        tune_fraction: float = 0.2,
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
        self.train_steps = None if train_steps is None else tuple(int(s) for s in train_steps)
        self.use_covariates = use_covariates
        self.tune = tune
        self.tune_grid = tuple(dict(cell) for cell in tune_grid)
        self.tune_fraction = tune_fraction
        self._models: dict[float, Any] = {}
        self._feature_names: list[str] = []
        self._trained_series: list[str] = []
        self._tuning: dict[str, Any] = {}
        self._training_steps = 0

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """The configuration actually used, after any search."""
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_data_in_leaf": self.min_data_in_leaf,
        }

    def describe(self) -> dict[str, Any]:
        """Record the lag set and the search outcome, so the table can be audited."""
        out = super().describe()
        out.update(
            {
                "lags": list(self.lags),
                "windows": list(self.windows),
                "max_horizon": self.max_horizon,
                "train_steps": None if self.train_steps is None else list(self.train_steps),
                "use_covariates": self.use_covariates,
                "forecast_strategy": "direct multi-horizon",
                "tuned": self.tune,
                "hyperparameters": self.hyperparameters,
            }
        )
        if self._tuning:
            out["tuning"] = self._tuning
        return out

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
        self,
        panel: Panel,
        origins: Sequence[int],
        steps: Sequence[int],
        series_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build the training design matrix and targets.

        Args:
            series_indices: restrict the rows to these series. ``None`` uses every
                series, which is what makes the model global.
        """
        context = fill_context(panel.series)
        wanted = (
            range(context.shape[0]) if series_indices is None else list(series_indices)
        )
        rows: list[list[float]] = []
        targets: list[float] = []
        names: list[str] = []
        for origin in origins:
            for step in steps:
                target_index = origin + step
                if target_index >= context.shape[1]:
                    continue
                for series_index in wanted:
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

    def _booster(self, level: float, **override: Any) -> Any:
        """One quantile regressor at the current, or overridden, hyperparameters."""
        import lightgbm as lgb

        settings = {**self.hyperparameters, **override}
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=level,
            num_leaves=int(settings["num_leaves"]),
            learning_rate=float(settings["learning_rate"]),
            n_estimators=int(settings["n_estimators"]),
            min_child_samples=int(settings["min_data_in_leaf"]),
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            n_jobs=1,
        )

    def _training_rows(
        self, panel: Panel, series_indices: Sequence[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Design matrix for the whole training panel."""
        # Origins start after the first genuine observation, not at step zero. The
        # training-budget mechanism marks the steps outside the budget as missing, and
        # ``fill_context`` back-fills a leading gap with the first observed value, so
        # origins before that point would contribute rows whose history is a synthetic
        # constant and whose target is that same constant. Training on them makes a small
        # budget look larger than it is and teaches the booster to predict a flat line,
        # which is a property of the imputation rather than of the data budget. A referee
        # asked whether the data-scarcity explanation had been isolated from the
        # treatment of missing observations; this is the part of the answer that belongs
        # in the model rather than in the prose.
        observed = np.isfinite(panel.series).any(axis=0)
        if not observed.any():
            raise ForecastError(f"{self.name}: the training panel is entirely missing")
        first_observed = int(np.argmax(observed))
        min_origin = max([*self.lags, *self.windows]) + first_observed
        usable = panel.n_timesteps - self.max_horizon
        if usable <= min_origin:
            raise ForecastError(
                f"{self.name} needs more than {min_origin + self.max_horizon} steps of "
                f"training data, got {panel.n_timesteps} with the first observation at "
                f"step {first_observed}"
            )
        # Subsample origins so that a 90-day minute-resolution site stays trainable; the
        # stride is deterministic, so the design matrix is a function of the panel alone.
        stride = max(1, usable // 2000)
        origins = list(range(min_origin, usable, stride))
        # The default ladder is a coarse grid chosen for a sixty-step horizon. It is kept
        # as the default so the submitted result stays reproducible, but it is the wrong
        # grid for a short horizon: asked for eight steps it trains on 1, 5 and 8 and
        # leaves the model extrapolating in its own horizon feature at steps 2, 3, 4, 6
        # and 7. ``train_steps`` lets an experiment state the grid it actually needs, and
        # the horizon-matched arm of E10 does.
        if self.train_steps is not None:
            steps = sorted(set(self.train_steps))
        else:
            steps = sorted({1, 5, 15, 30, min(60, self.max_horizon), self.max_horizon})
        steps = [s for s in steps if 1 <= s <= self.max_horizon]
        if not steps:
            raise ForecastError(
                f"{self.name}: no training step survives max_horizon={self.max_horizon}"
            )
        return self._design(panel, origins, steps, series_indices)

    @property
    def training_window_steps(self) -> int:
        """Length of the training window the last :meth:`fit` saw, in steps."""
        return self._training_steps

    def _search(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Pick hyperparameters on a held-out tail of the training window.

        The split is by position in the design matrix, whose rows are generated in origin
        order, so the held-out block is strictly later in time than the block fitted on.
        Selection is on the median booster alone: it is the quantile the reported point
        metric reads, and searching all nine would cost nine times as much to choose the
        same cell. The winner is then refitted on the whole training window.
        """
        cut = round((1.0 - self.tune_fraction) * features.shape[0])
        if cut < 2 or cut >= features.shape[0]:
            raise ForecastError(
                f"{self.name}: a tune_fraction of {self.tune_fraction} leaves no "
                f"validation block in {features.shape[0]} training rows"
            )
        fit_x, fit_y = features[:cut], targets[:cut]
        val_x, val_y = features[cut:], targets[cut:]

        scores: list[dict[str, Any]] = []
        for cell in self.tune_grid:
            model = self._booster(0.5, **cell)
            model.fit(fit_x, fit_y)
            error = val_y - np.asarray(model.predict(val_x))
            loss = float(np.maximum(0.5 * error, -0.5 * error).mean())
            scores.append({**cell, "validation_pinball_50": loss})
        best = min(scores, key=lambda row: float(row["validation_pinball_50"]))

        self.num_leaves = int(best["num_leaves"])
        self.learning_rate = float(best["learning_rate"])
        self.n_estimators = int(best["n_estimators"])
        self.min_data_in_leaf = int(best["min_data_in_leaf"])
        self._tuning = {
            "n_cells": len(scores),
            "n_fit_rows": int(cut),
            "n_validation_rows": int(features.shape[0] - cut),
            "selected": self.hyperparameters,
            "grid": scores,
        }

    def fit(self, panel: Panel) -> None:
        """Train one booster per quantile level on the whole site."""
        features, targets, names = self._training_rows(panel)
        if self.tune:
            self._search(features, targets)
        self._feature_names = names
        self._trained_series = list(panel.series_ids)
        self._training_steps = panel.n_timesteps
        self._models = {
            level: self._booster(level).fit(features, targets)
            for level in DEFAULT_QUANTILES
        }

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


class PerSeriesLightGBM(GlobalLightGBM):
    """One independent booster set per series, rather than one model for the site.

    A referee observed that a single globally trained configuration is not enough to
    establish that a zero-shot model beats a well-specified local one, because a global
    model is a particular modelling choice and not obviously the favourable one: it
    shares strength across series, which helps a short record and hurts a series whose
    dynamics differ from the panel average. This arm removes the sharing entirely. It is
    the costlier arm, since it fits ``n_series`` times nine boosters instead of nine, and
    it is the one a practitioner with a single room would actually build.

    The feature set is deliberately identical to the global arm's, including the series
    index and the flow indicator, which are constant within a series and therefore inert
    here. Keeping them means the two arms differ in exactly one thing.
    """

    name = "per_series_lightgbm"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._per_series: dict[int, dict[float, Any]] = {}

    def fit(self, panel: Panel) -> None:
        """Train a separate booster set for every series of the site."""
        self._trained_series = list(panel.series_ids)
        self._training_steps = panel.n_timesteps
        self._per_series = {}
        names: list[str] = []
        for series_index in range(panel.n_series):
            features, targets, names = self._training_rows(panel, [series_index])
            if self.tune and series_index == 0:
                # Search once, on the first series, and reuse the winner. Tuning each
                # series separately on its own tail would select on 240 independent
                # validation blocks and report the maximum of 240 noisy searches.
                self._search(features, targets)
            self._per_series[series_index] = {
                level: self._booster(level).fit(features, targets)
                for level in DEFAULT_QUANTILES
            }
        self._feature_names = names
        self._models = {}

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        if not self._per_series:
            raise ForecastError(f"{self.name}.predict called before fit")
        if list(panel.series_ids) != self._trained_series:
            raise ForecastError(
                f"{self.name} was trained on a different set of series than it is being "
                "asked to predict; retrain per site"
            )
        context = fill_context(panel.series)
        origin = panel.n_timesteps - 1
        out = np.empty((panel.n_series, horizon, len(levels)), dtype=np.float64)
        for series_index in range(panel.n_series):
            models = self._per_series[series_index]
            missing = [q for q in levels if q not in models]
            if missing:
                raise ForecastError(
                    f"{self.name} has no model for quantile levels {missing} on series "
                    f"{panel.series_ids[series_index]}"
                )
            design = np.asarray(
                [
                    self._feature_row(panel, context, series_index, origin, step)[0]
                    for step in range(1, horizon + 1)
                ],
                dtype=np.float64,
            )
            for column, level in enumerate(levels):
                out[series_index, :, column] = np.asarray(models[level].predict(design))
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
