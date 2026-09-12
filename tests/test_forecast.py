"""M4 acceptance: the shared forecaster contract.

Every forecaster available without downloading weights is put through the same test on
the toy site: correct shape and dtype, no NaNs, non-decreasing quantiles, and identical
output across two runs with the same seed. Methods that need foundation model weights are
covered by the same parametrised contract, marked ``requires_weights`` so that a normal
test run does not pull gigabytes from the Hub.
"""

from __future__ import annotations

import numpy as np
import pytest

from mflow.forecast import available_forecasters, build_forecaster
from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    ForecastError,
    LeakageError,
    fill_context,
    season_length,
    validate_quantiles,
)
from mflow.forecast.foundation import map_quantiles
from mflow.graph import BuildingGraph
from mflow.schema import Panel, SiteData

#: Methods that run from source without any downloaded checkpoint.
LOCAL_METHODS = [
    "last_value",
    "seasonal_naive",
    "historical_average",
    "global_lightgbm",
    "sarima",
    "ets",
    "dcrnn",
    "stgcn",
    "graph_wavenet",
]

WEIGHT_METHODS = [
    "timesfm3_univariate",
    "timesfm3_multivariate",
    "timesfm3_multivariate_covariates",
    "timesfm25_univariate",
    "chronos2_univariate",
    "chronos2_multivariate",
    "toto2",
]

HORIZON = 10


@pytest.fixture(scope="module")
def panel(toy_site: SiteData) -> Panel:
    return toy_site.to_panel(horizon=60)


def _fast_params(name: str) -> dict[str, object]:
    """Shrink the heavier methods so the contract test stays quick."""
    if name in {"dcrnn", "stgcn", "graph_wavenet"}:
        return {
            "context_length": 32,
            "horizon": HORIZON,
            "hidden_dim": 8,
            "epochs": 2,
            "batch_size": 16,
            "patience": 2,
        }
    if name == "global_lightgbm":
        return {"n_estimators": 20, "max_horizon": HORIZON, "lags": (1, 2, 5), "windows": (5,)}
    if name == "sarima":
        return {"order": (1, 0, 0), "seasonal_order": (0, 0, 0), "seasonal_period": 1}
    if name == "ets":
        return {"trend": None, "seasonal": None}
    if name == "seasonal_naive":
        # The toy site is four hours long, so a full 1440-step day does not fit.
        return {"season": 60}
    return {}


def _build(name: str, toy_graph: BuildingGraph):
    return build_forecaster(name, graph=toy_graph, seed=17, **_fast_params(name))


@pytest.mark.parametrize("name", LOCAL_METHODS)
def test_forecaster_contract(name: str, panel: Panel, toy_graph: BuildingGraph) -> None:
    model = _build(name, toy_graph)
    if model.requires_training:
        model.fit(panel)

    forecast = model.predict(panel, HORIZON, DEFAULT_QUANTILES)

    assert forecast.shape == (panel.n_series, HORIZON, len(DEFAULT_QUANTILES))
    assert forecast.dtype == np.float32
    assert np.isfinite(forecast).all()
    assert np.all(np.diff(forecast, axis=-1) >= 0.0)
    assert np.all(forecast >= 0.0)


@pytest.mark.parametrize("name", LOCAL_METHODS)
def test_forecaster_is_deterministic(name: str, panel: Panel, toy_graph: BuildingGraph) -> None:
    first_model = _build(name, toy_graph)
    second_model = _build(name, toy_graph)
    if first_model.requires_training:
        first_model.fit(panel)
        second_model.fit(panel)

    first = first_model.predict(panel, HORIZON)
    second = second_model.predict(panel, HORIZON)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name", WEIGHT_METHODS)
@pytest.mark.requires_weights
def test_foundation_forecaster_contract(
    name: str, panel: Panel, toy_graph: BuildingGraph
) -> None:
    model = build_forecaster(name, graph=toy_graph, seed=17, max_context=240)
    forecast = model.predict(panel, HORIZON, DEFAULT_QUANTILES)
    assert forecast.shape == (panel.n_series, HORIZON, len(DEFAULT_QUANTILES))
    assert forecast.dtype == np.float32
    assert np.isfinite(forecast).all()
    assert np.all(np.diff(forecast, axis=-1) >= 0.0)


# --------------------------------------------------------------------------- #
# Registry and guards
# --------------------------------------------------------------------------- #


def test_registry_lists_every_method() -> None:
    names = available_forecasters()
    assert set(LOCAL_METHODS) | set(WEIGHT_METHODS) <= set(names)


def test_graph_methods_need_a_graph() -> None:
    with pytest.raises(ValueError, match="needs the building graph"):
        build_forecaster("dcrnn")


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown forecaster"):
        build_forecaster("transformer_xl")


def test_zero_shot_models_refuse_training_data(panel: Panel) -> None:
    model = build_forecaster("seasonal_naive", season=60)
    with pytest.raises(LeakageError, match="zero-shot"):
        model.fit(panel)


def test_seasonal_naive_needs_a_full_season(panel: Panel) -> None:
    model = build_forecaster("seasonal_naive", season=10_000)
    with pytest.raises(ForecastError, match="at least one full season"):
        model.predict(panel, HORIZON)


def test_graph_model_refuses_a_longer_horizon_than_it_trained_for(
    panel: Panel, toy_graph: BuildingGraph
) -> None:
    model = _build("stgcn", toy_graph)
    model.fit(panel)
    with pytest.raises(ForecastError, match="trained for a horizon"):
        model.predict(panel, HORIZON + 5)


def test_lightgbm_refuses_to_predict_before_fit(panel: Panel) -> None:
    model = build_forecaster("global_lightgbm")
    with pytest.raises(ForecastError, match="called before fit"):
        model.predict(panel, HORIZON)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_validate_quantiles() -> None:
    assert validate_quantiles([0.1, 0.5, 0.9]) == (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="ascending"):
        validate_quantiles([0.5, 0.1])
    with pytest.raises(ValueError, match="strictly inside"):
        validate_quantiles([0.0, 0.5])


def test_season_length() -> None:
    assert season_length(60) == 1440
    assert season_length(900) == 96
    with pytest.raises(ValueError, match="does not divide a day"):
        season_length(7)


def test_fill_context_interpolates_and_zero_fills() -> None:
    series = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    filled = fill_context(series)
    np.testing.assert_allclose(filled[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(filled[1], [0.0, 0.0, 0.0])


def test_map_quantiles_interpolates_within_range() -> None:
    fan = np.array([[[1.0, 2.0, 3.0]]])
    out = map_quantiles(fan, (0.1, 0.5, 0.9), (0.3, 0.5))
    np.testing.assert_allclose(out[0, 0], [1.5, 2.0])


def test_map_quantiles_refuses_to_extrapolate() -> None:
    with pytest.raises(ForecastError, match="fabricate"):
        map_quantiles(np.zeros((1, 1, 3)), (0.1, 0.5, 0.9), (0.05, 0.5))


def test_historical_average_uses_time_of_day_buckets(panel: Panel) -> None:
    model = build_forecaster("historical_average")
    forecast = model.predict(panel, HORIZON)
    # The toy site has a smooth midday peak, so the bucket means must vary across the
    # horizon rather than collapsing to a single constant.
    assert float(np.std(forecast[:, :, 4])) > 0.0
