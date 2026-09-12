"""Forecasting methods (M4).

Everything is reached through :func:`build_forecaster`, so an experiment config names a
method as a string and the harness never imports a model class directly. The heavy
dependencies (torch models, foundation model weights) are imported lazily inside the
factories: listing the available methods must not download a 330M-parameter checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    ForecastError,
    Forecaster,
    LeakageError,
)

if TYPE_CHECKING:
    from mflow.graph import BuildingGraph

#: Methods that need the building graph handed to their constructor.
GRAPH_METHODS: frozenset[str] = frozenset({"dcrnn", "stgcn", "graph_wavenet"})

#: Methods that run without any local training.
ZERO_SHOT_METHODS: frozenset[str] = frozenset(
    {
        "last_value",
        "seasonal_naive",
        "historical_average",
        "sarima",
        "ets",
        "timesfm3_univariate",
        "timesfm3_multivariate",
        "timesfm3_multivariate_covariates",
        "timesfm25_univariate",
        "chronos2_univariate",
        "chronos2_multivariate",
        "chronos2_multivariate_covariates",
        "toto2",
    }
)


def _factories() -> dict[str, Callable[..., Forecaster]]:
    """Lazily built registry of method name to constructor."""
    from mflow.forecast import classical, trivial

    def timesfm3_univariate(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import TimesFM3Univariate

        return TimesFM3Univariate(**kw)

    def timesfm3_multivariate(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import TimesFM3Multivariate

        return TimesFM3Multivariate(**kw)

    def timesfm3_multivariate_covariates(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import TimesFM3MultivariateCovariates

        return TimesFM3MultivariateCovariates(**kw)

    def timesfm25_univariate(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import TimesFM25Univariate

        return TimesFM25Univariate(**kw)

    def chronos2_univariate(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import Chronos2

        return Chronos2(multivariate=False, **kw)

    def chronos2_multivariate(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import Chronos2

        return Chronos2(multivariate=True, **kw)

    def chronos2_multivariate_covariates(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import Chronos2

        return Chronos2(multivariate=True, use_future_covariates=True, **kw)

    def toto2(**kw: Any) -> Forecaster:
        from mflow.forecast.foundation import Toto2

        return Toto2(**kw)

    def dcrnn(**kw: Any) -> Forecaster:
        from mflow.forecast.graphnn import DCRNN

        return DCRNN(**kw)

    def stgcn(**kw: Any) -> Forecaster:
        from mflow.forecast.graphnn import STGCN

        return STGCN(**kw)

    def graph_wavenet(**kw: Any) -> Forecaster:
        from mflow.forecast.graphnn import GraphWaveNet

        return GraphWaveNet(**kw)

    return {
        "last_value": trivial.LastValue,
        "seasonal_naive": trivial.SeasonalNaive,
        "historical_average": trivial.HistoricalAverage,
        "global_lightgbm": classical.GlobalLightGBM,
        "sarima": classical.SARIMA,
        "ets": classical.ETS,
        "timesfm3_univariate": timesfm3_univariate,
        "timesfm3_multivariate": timesfm3_multivariate,
        "timesfm3_multivariate_covariates": timesfm3_multivariate_covariates,
        "timesfm25_univariate": timesfm25_univariate,
        "chronos2_univariate": chronos2_univariate,
        "chronos2_multivariate": chronos2_multivariate,
        "chronos2_multivariate_covariates": chronos2_multivariate_covariates,
        "toto2": toto2,
        "dcrnn": dcrnn,
        "stgcn": stgcn,
        "graph_wavenet": graph_wavenet,
    }


def available_forecasters() -> list[str]:
    """Names accepted by :func:`build_forecaster`."""
    return sorted(_factories())


def build_forecaster(
    name: str,
    *,
    graph: BuildingGraph | None = None,
    seed: int = 0,
    **params: Any,
) -> Forecaster:
    """Instantiate a forecaster by name.

    Args:
        name: one of :func:`available_forecasters`.
        graph: required by the graph neural network methods.
        seed: run seed, forwarded to the constructor.
        **params: method-specific hyperparameters from ``configs/models/``.

    Raises:
        KeyError: for an unknown name.
        ValueError: if a graph method is requested without a graph.
    """
    factories = _factories()
    if name not in factories:
        raise KeyError(f"unknown forecaster {name!r}; known: {available_forecasters()}")
    if name in GRAPH_METHODS:
        if graph is None:
            raise ValueError(f"forecaster {name!r} needs the building graph")
        return factories[name](graph=graph, seed=seed, **params)
    return factories[name](seed=seed, **params)


__all__ = [
    "DEFAULT_QUANTILES",
    "GRAPH_METHODS",
    "ZERO_SHOT_METHODS",
    "ForecastError",
    "Forecaster",
    "LeakageError",
    "available_forecasters",
    "build_forecaster",
]
