"""M5 acceptance: topology-constrained reconciliation.

Three things are asserted, in increasing order of importance:

1. the constraint system encodes exactly the balance equation written in the spec;
2. the projection returns a forecast whose conservation residual is at solver tolerance
   and whose bound violations are zero;
3. on a synthetic problem where the truth is known and noise is injected into the
   forecast, the projection *provably* reduces total squared error. This is the property
   the paper claims, so it is tested directly rather than inferred from a benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from mflow.graph import BuildingGraph
from mflow.reconcile.constraints import (
    ConstraintError,
    build_constraints,
    build_constraints_cached,
    flatten,
    unflatten,
)
from mflow.reconcile.mint import MinTReconciler, _shrunk_covariance
from mflow.reconcile.projection import (
    IdentityReconciler,
    ProjectionReconciler,
    ReconciliationError,
    sigma_from_quantiles,
)
from mflow.reconcile.quantiles import QuantileReconciler, crossing_rate, enforce_monotone
from mflow.schema import SiteData

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@pytest.fixture
def system(toy_graph: BuildingGraph, toy_site: SiteData):
    return build_constraints(toy_graph, toy_site.series_ids, horizon=6, interval_seconds=60)


@pytest.fixture
def origin(toy_site: SiteData) -> dict[str, float]:
    """Observed occupancy at the last timestamp of the toy site."""
    last = toy_site.occupancy["timestamp"].max()
    snapshot = toy_site.occupancy[toy_site.occupancy["timestamp"] == last]
    return dict(zip(snapshot["node_id"], snapshot["count"].astype(float), strict=True))


# --------------------------------------------------------------------------- #
# Constraint system
# --------------------------------------------------------------------------- #


def test_system_shape(system, toy_site: SiteData) -> None:
    assert system.n_vars == len(toy_site.series_ids) * 6
    assert system.n_constraints == 3 * 6  # three interior nodes, six steps


def test_true_trajectory_satisfies_the_constraints(
    toy_site: SiteData, toy_graph: BuildingGraph
) -> None:
    # Take six real steps out of the fixture and check that the ground truth is a
    # feasible point of the system built around them. If it is not, the constraint is
    # written down wrong and every downstream number is meaningless.
    horizon = 6
    wide = toy_site.wide_series()
    start = 100
    window = wide.iloc[start : start + horizon]
    previous = wide.iloc[start - 1]

    system = build_constraints(toy_graph, toy_site.series_ids, horizon, 60)
    last_occ = {
        node: float(previous[f"occ:{node}"]) for node in toy_graph.interior_nodes
    }
    rhs = system.rhs(last_occ)
    y = flatten(window.to_numpy().T)
    np.testing.assert_allclose(system.residual(y, rhs), 0.0, atol=1e-9)
    assert float(np.sum(system.violation(y))) == 0.0


def test_system_rejects_incomplete_panel(toy_graph: BuildingGraph, toy_site: SiteData) -> None:
    with pytest.raises(ConstraintError, match="missing"):
        build_constraints(toy_graph, toy_site.series_ids[:-1], horizon=3, interval_seconds=60)


def test_rhs_requires_an_observed_origin(system) -> None:
    with pytest.raises(ConstraintError, match="no observed occupancy"):
        system.rhs({"foyer": 3.0})


def test_bounds_come_from_capacity(system, toy_graph: BuildingGraph) -> None:
    # gallery_b holds 30 people; e_ab passes 25 per minute, so 25 per 60s interval.
    occ_b = system.series_ids.index("occ:gallery_b") * system.horizon
    flow_ab = system.series_ids.index("flow:e_ab") * system.horizon
    assert system.ub[occ_b] == toy_graph.node_capacity["gallery_b"] == 30.0
    assert system.ub[flow_ab] == 25.0
    assert float(system.lb.max()) == 0.0


def test_constraint_cache_returns_the_same_object(
    toy_graph: BuildingGraph, toy_site: SiteData
) -> None:
    first = build_constraints_cached(toy_graph, toy_site.series_ids, 6, 60)
    second = build_constraints_cached(toy_graph, toy_site.series_ids, 6, 60)
    assert first is second


def test_flatten_round_trip() -> None:
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.testing.assert_array_equal(unflatten(flatten(array), 3, 4), array)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def _incoherent_forecast(
    system, rhs: np.ndarray, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """A deliberately incoherent forecast plus a plausible spread."""
    generator = np.random.default_rng(seed)
    n_series = len(system.series_ids)
    forecast = generator.uniform(2.0, 10.0, size=(n_series, system.horizon))
    sigma = generator.uniform(0.5, 3.0, size=(n_series, system.horizon))
    del rhs
    return forecast, sigma


def test_projection_makes_the_forecast_coherent_and_feasible(system, origin) -> None:
    rhs = system.rhs(origin)
    forecast, sigma = _incoherent_forecast(system, rhs)
    reconciler = ProjectionReconciler()

    result = reconciler.reconcile(forecast, system, rhs, sigma=sigma)

    assert result.residual_before > 1.0
    assert result.residual_after < 1e-6
    assert result.violation_after == 0.0
    assert result.reconciled.shape == forecast.shape


def test_identity_reconciler_changes_nothing_but_measures(system, origin) -> None:
    rhs = system.rhs(origin)
    forecast, _ = _incoherent_forecast(system, rhs)
    result = IdentityReconciler().reconcile(forecast, system, rhs)
    np.testing.assert_array_equal(result.reconciled, forecast)
    assert result.residual_after == result.residual_before > 0.0


def test_projection_reduces_squared_error_when_noise_is_injected(
    toy_site: SiteData, toy_graph: BuildingGraph
) -> None:
    """The core claim: projecting a noisy forecast onto the true feasible set helps.

    The truth is a real window of the fixture, so it satisfies ``A y = b`` exactly.
    Gaussian noise is added to obtain a forecast that does not. Projection onto a convex
    set containing the truth is non-expansive, so the reconciled forecast cannot be
    further from the truth than the raw one; averaged over draws the improvement is
    strict. Both statements are asserted.
    """
    horizon = 8
    wide = toy_site.wide_series()
    start = 120
    truth = wide.iloc[start : start + horizon].to_numpy().T
    previous = wide.iloc[start - 1]

    system = build_constraints(toy_graph, toy_site.series_ids, horizon, 60)
    rhs = system.rhs({n: float(previous[f"occ:{n}"]) for n in toy_graph.interior_nodes})
    np.testing.assert_allclose(system.residual(flatten(truth), rhs), 0.0, atol=1e-9)

    reconciler = ProjectionReconciler()
    generator = np.random.default_rng(20260912)
    improvements = []
    for _ in range(20):
        noise = generator.normal(0.0, 1.5, size=truth.shape)
        forecast = np.clip(truth + noise, 0.0, None)
        sigma = np.full_like(forecast, 1.5)
        reconciled = reconciler.reconcile(forecast, system, rhs, sigma=sigma).reconciled

        error_before = float(np.sum((forecast - truth) ** 2))
        error_after = float(np.sum((reconciled - truth) ** 2))
        # Non-expansiveness of the projection, up to solver tolerance.
        assert error_after <= error_before + 1e-6
        improvements.append(error_before - error_after)

    assert float(np.mean(improvements)) > 0.0
    assert sum(i > 0 for i in improvements) == len(improvements)


def test_uniform_weight_differs_from_uncertainty_weighting(system, origin) -> None:
    rhs = system.rhs(origin)
    forecast, sigma = _incoherent_forecast(system, rhs)
    # Make the spreads genuinely heterogeneous so the weighting has something to do.
    sigma[0, :] = 0.05
    sigma[-1, :] = 8.0

    weighted = ProjectionReconciler(weighting="uncertainty").reconcile(
        forecast, system, rhs, sigma=sigma
    )
    uniform = ProjectionReconciler(weighting="uniform").reconcile(
        forecast, system, rhs, sigma=sigma
    )

    assert weighted.residual_after < 1e-6
    assert uniform.residual_after < 1e-6
    # The confident series is moved far less by the weighted projection.
    moved_weighted = abs(weighted.reconciled[0] - forecast[0]).sum()
    moved_uniform = abs(uniform.reconciled[0] - forecast[0]).sum()
    assert moved_weighted < moved_uniform


def test_uncertainty_weighting_requires_sigma(system, origin) -> None:
    rhs = system.rhs(origin)
    forecast, _ = _incoherent_forecast(system, rhs)
    with pytest.raises(ReconciliationError, match="needs a predictive spread"):
        ProjectionReconciler().reconcile(forecast, system, rhs)


def test_projection_is_deterministic(system, origin) -> None:
    rhs = system.rhs(origin)
    forecast, sigma = _incoherent_forecast(system, rhs)
    reconciler = ProjectionReconciler()
    first = reconciler.reconcile(forecast, system, rhs, sigma=sigma).reconciled
    second = reconciler.reconcile(forecast, system, rhs, sigma=sigma).reconciled
    np.testing.assert_allclose(first, second, atol=1e-9)


def test_sigma_from_quantiles_uses_the_normal_equivalent_spread() -> None:
    fan = np.zeros((1, 1, 9))
    fan[0, 0, :] = [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816]
    sigma = sigma_from_quantiles(fan, QUANTILE_LEVELS)
    # A standard normal must come back with sigma ~= 1.
    assert abs(float(sigma[0, 0]) - 1.0) < 1e-3


def test_sigma_needs_the_outer_quantiles() -> None:
    with pytest.raises(ValueError, match=r"0\.1 and 0\.9"):
        sigma_from_quantiles(np.zeros((1, 1, 3)), (0.25, 0.5, 0.75))


# --------------------------------------------------------------------------- #
# MinT baseline
# --------------------------------------------------------------------------- #


def test_mint_enforces_equalities_but_not_bounds(system, origin) -> None:
    rhs = system.rhs(origin)
    generator = np.random.default_rng(3)
    # A forecast pushed hard against the bounds: MinT has no way to respect them.
    forecast = generator.normal(-5.0, 2.0, size=(len(system.series_ids), system.horizon))
    result = MinTReconciler().reconcile(forecast, system, rhs)
    assert result.residual_after < 1e-6
    assert result.violation_after > 0.0


def test_mint_shrinkage_is_estimated_and_bounded() -> None:
    generator = np.random.default_rng(11)
    residuals = generator.normal(size=(5, 300))
    reconciler = MinTReconciler()
    reconciler.fit(residuals, [f"occ:n{i}" for i in range(5)])
    assert reconciler.shrinkage_intensity is not None
    assert 0.0 <= reconciler.shrinkage_intensity <= 1.0


def test_shrunk_covariance_is_positive_definite() -> None:
    generator = np.random.default_rng(5)
    observations = generator.normal(size=(30, 8))
    covariance, intensity = _shrunk_covariance(observations, intensity=None)
    assert 0.0 <= intensity <= 1.0
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_mint_needs_enough_residuals() -> None:
    with pytest.raises(ValueError, match="at least two complete"):
        MinTReconciler().fit(np.zeros((3, 1)), ["a", "b", "c"])


# --------------------------------------------------------------------------- #
# Quantiles
# --------------------------------------------------------------------------- #


def _quantile_fan(system, seed: int = 13) -> np.ndarray:
    generator = np.random.default_rng(seed)
    n_series = len(system.series_ids)
    median = generator.uniform(3.0, 9.0, size=(n_series, system.horizon, 1))
    offsets = np.array([-1.6, -1.1, -0.7, -0.3, 0.0, 0.3, 0.7, 1.1, 1.6])
    spread = generator.uniform(0.5, 2.0, size=(n_series, system.horizon, 1))
    return median + spread * offsets


def test_shift_strategy_keeps_the_fan_ordered_and_the_median_coherent(system, origin) -> None:
    rhs = system.rhs(origin)
    fan = _quantile_fan(system)
    reconciler = QuantileReconciler(ProjectionReconciler(), strategy="shift")

    reconciled, diagnostics = reconciler.reconcile(fan, QUANTILE_LEVELS, system, rhs)

    assert reconciled.shape == fan.shape
    assert np.all(np.diff(reconciled, axis=-1) >= -1e-9)
    assert diagnostics.residual_after < 1e-6
    np.testing.assert_allclose(reconciled[..., 4], diagnostics.reconciled, atol=1e-6)


def test_per_quantile_strategy_makes_every_level_coherent(system, origin) -> None:
    rhs = system.rhs(origin)
    fan = _quantile_fan(system)
    reconciler = QuantileReconciler(ProjectionReconciler(), strategy="per_quantile")

    reconciled, diagnostics = reconciler.reconcile(fan, QUANTILE_LEVELS, system, rhs)

    assert np.all(np.diff(reconciled, axis=-1) >= -1e-9)
    assert diagnostics.residual_after < 1e-6

    # Each level is projected to exact coherence, but the monotonicity repair that
    # follows moves values again, so the delivered fan is only approximately coherent
    # away from the median. That loss is the cost this ablation exists to measure, and
    # it is asserted as a large reduction rather than as exactness.
    for index in range(reconciled.shape[-1]):
        before = float(np.max(np.abs(system.residual(flatten(fan[..., index]), rhs))))
        after = float(np.max(np.abs(system.residual(flatten(reconciled[..., index]), rhs))))
        assert after < 0.2 * before


def test_quantile_reconciler_names_itself(system) -> None:
    assert QuantileReconciler(ProjectionReconciler(), strategy="shift").name == "proposed+shift"
    assert (
        QuantileReconciler(IdentityReconciler(), strategy="per_quantile").name
        == "none+per_quantile"
    )


def test_enforce_monotone_and_crossing_rate() -> None:
    crossed = np.array([[[3.0, 1.0, 2.0]]])
    assert crossing_rate(crossed) > 0.0
    repaired = enforce_monotone(crossed)
    np.testing.assert_array_equal(repaired, np.array([[[1.0, 2.0, 3.0]]]))
    assert crossing_rate(repaired) == 0.0
