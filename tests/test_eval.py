"""M6: the evaluation protocol, metrics, significance tests and harness.

The most important tests here are the leakage ones. Everything else measures how good a
forecast is; :func:`test_a_context_cannot_reach_past_its_origin` and its neighbours check
that the number means what the paper says it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mflow.eval import (
    MethodSpec,
    ProtocolError,
    SignificanceError,
    build_plan,
    context_panel,
    diebold_mariano,
    holm_bonferroni,
    limit_days,
    make_split,
    run_evaluation,
    training_panel,
    truth_for,
)
from mflow.eval import metrics as M
from mflow.eval.harness import paired_losses
from mflow.eval.significance import newey_west_variance
from mflow.forecast import build_forecaster
from mflow.reconcile import IdentityReconciler, MinTReconciler, ProjectionReconciler
from mflow.schema import Panel

QUANTILES = tuple(round(0.1 * i, 1) for i in range(1, 10))


@pytest.fixture
def plan(toy_site):
    panel = toy_site.to_panel()
    return build_plan(
        panel,
        context_length=48,
        horizons=[5, 10],
        stride=6,
        quantiles=QUANTILES,
    )


# --------------------------------------------------------------------------- #
# Splits and plans
# --------------------------------------------------------------------------- #


def test_splits_are_ordered_and_separated() -> None:
    stamps = pd.date_range("2026-01-01", periods=1000, freq="1min", tz="UTC")
    split = make_split(1000, stamps, horizon=12)
    assert split.train_end < split.validation_start < split.validation_end < split.test_start
    assert split.validation_start - split.train_end == 12
    assert split.test_start - split.validation_end == 12
    assert split.test_end == 1000


def test_a_training_target_cannot_overlap_a_test_context() -> None:
    # The gap is what guarantees this: the last training step plus one horizon still
    # falls before the test window begins.
    stamps = pd.date_range("2026-01-01", periods=1000, freq="1min", tz="UTC")
    horizon = 30
    split = make_split(1000, stamps, horizon=horizon)
    assert split.train_end + horizon <= split.validation_start
    assert split.validation_end + horizon <= split.test_start


def test_a_record_too_short_for_three_windows_is_rejected() -> None:
    stamps = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    with pytest.raises(ProtocolError, match="cannot hold three windows"):
        make_split(20, stamps, horizon=12)


def test_split_fractions_must_be_a_partition() -> None:
    stamps = pd.date_range("2026-01-01", periods=1000, freq="1min", tz="UTC")
    with pytest.raises(ProtocolError, match="sum to 1"):
        make_split(1000, stamps, horizon=5, fractions=(0.5, 0.2, 0.2))


def test_every_origin_lies_inside_the_test_window(plan, toy_site) -> None:
    for task in plan:
        assert task.origin >= plan.split.test_start
        assert task.origin + task.horizon <= plan.split.test_end
        assert task.context_start >= 0
        assert task.origin - task.context_start == plan.context_length
    _ = toy_site


def test_a_plan_with_no_room_for_an_origin_fails(toy_site) -> None:
    panel = toy_site.to_panel()
    with pytest.raises(ProtocolError, match="no valid origin"):
        build_plan(
            panel,
            context_length=panel.n_timesteps,
            horizons=[10],
            stride=1,
            quantiles=QUANTILES,
        )


def test_max_origins_thins_without_moving_the_window(toy_site) -> None:
    panel = toy_site.to_panel()
    full = build_plan(panel, context_length=48, horizons=[5], stride=1, quantiles=QUANTILES)
    thin = build_plan(
        panel, context_length=48, horizons=[5], stride=1, quantiles=QUANTILES, max_origins=4
    )
    assert len(thin) <= 4
    assert set(thin.origins()) <= set(full.origins())
    assert thin.origins()[0] == full.origins()[0]


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


def test_a_context_cannot_reach_past_its_origin(plan, toy_site) -> None:
    panel = toy_site.to_panel()
    for task in plan:
        context = context_panel(panel, task)
        assert context.n_timesteps == plan.context_length
        assert context.timestamps[-1] < panel.timestamps[task.origin]
        # The values are the ones before the origin, exactly.
        np.testing.assert_array_equal(
            context.series, panel.series[:, task.context_start : task.origin]
        )


def test_future_covariates_reach_the_horizon_but_carry_no_targets(plan, toy_site) -> None:
    panel = toy_site.to_panel()
    task = plan.tasks[0]
    context = context_panel(panel, task)
    assert context.horizon_covered == task.horizon
    # Known-future covariates are allowed past the origin because they are known at it;
    # the series themselves are not.
    assert context.series.shape[1] == plan.context_length


def test_the_training_panel_stops_before_the_gap(plan, toy_site) -> None:
    panel = toy_site.to_panel()
    train = training_panel(panel, plan.split, include_validation=False)
    assert train.n_timesteps == plan.split.train_end
    assert train.timestamps[-1] < panel.timestamps[plan.split.test_start]


def test_limit_days_refuses_a_budget_the_record_cannot_meet(plan, toy_site) -> None:
    panel = toy_site.to_panel()
    with pytest.raises(ProtocolError, match="but the training window is only"):
        limit_days(toy_site, panel, plan.split, days=30)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@pytest.fixture
def perfect():
    truth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    prediction = np.repeat(truth[:, :, None], len(QUANTILES), axis=2)
    return truth, prediction


def test_a_perfect_forecast_scores_zero(perfect) -> None:
    truth, prediction = perfect
    assert M.mae(truth, prediction, QUANTILES) == 0.0
    assert M.rmse(truth, prediction, QUANTILES) == 0.0
    assert M.wql(truth, prediction, QUANTILES) == 0.0
    assert M.crps(truth, prediction, QUANTILES) == 0.0
    assert M.coverage(truth, prediction, QUANTILES) == 1.0
    assert M.interval_width(truth, prediction, QUANTILES) == 0.0


def test_mae_is_the_mean_absolute_error(perfect) -> None:
    truth, prediction = perfect
    assert M.mae(truth, prediction + 2.0, QUANTILES) == pytest.approx(2.0)
    assert M.rmse(truth, prediction + 2.0, QUANTILES) == pytest.approx(2.0)


def test_missing_actuals_are_excluded_not_counted_as_zero(perfect) -> None:
    truth, prediction = perfect
    holed = truth.copy()
    holed[0, 0] = np.nan
    shifted = prediction + 1.0
    # Every observed cell is off by one, so the MAE is one whether or not a cell is
    # missing. Treating the gap as a zero actual would give a different answer.
    assert M.mae(holed, shifted, QUANTILES) == pytest.approx(1.0)


def test_a_window_with_no_observations_raises(perfect) -> None:
    truth, prediction = perfect
    with pytest.raises(M.MetricError, match="no observed values"):
        M.mae(np.full_like(truth, np.nan), prediction, QUANTILES)


def test_pinball_is_asymmetric() -> None:
    truth = np.array([[10.0]])
    low = np.full((1, 1, len(QUANTILES)), 8.0)
    high = np.full((1, 1, len(QUANTILES)), 12.0)
    # Under-forecasting is penalised more at the upper quantiles and less at the lower
    # ones, which is the whole point of a quantile loss.
    under = M.pinball(truth, low, QUANTILES)[0, 0]
    over = M.pinball(truth, high, QUANTILES)[0, 0]
    assert under[-1] > under[0]
    assert over[0] > over[-1]


def test_coverage_counts_the_interval() -> None:
    truth = np.array([[0.0, 5.0, 20.0]])
    prediction = np.zeros((1, 3, len(QUANTILES)))
    for index in range(len(QUANTILES)):
        prediction[0, :, index] = index  # q10=0 .. q90=8
    # 0 and 5 are inside [0, 8]; 20 is not.
    assert M.coverage(truth, prediction, QUANTILES) == pytest.approx(2 / 3)
    assert M.interval_width(truth, prediction, QUANTILES) == pytest.approx(8.0)


def test_coverage_at_an_unavailable_level_raises(perfect) -> None:
    truth, prediction = perfect
    with pytest.raises(M.MetricError, match="is not among"):
        M.coverage(truth, prediction, QUANTILES, level=0.95)


def test_mase_denominator_comes_from_the_training_window() -> None:
    training = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    scale = M.seasonal_naive_scale(training, season_length=2)
    # A perfectly two-periodic series has a zero seasonal difference, which is degenerate.
    assert np.isnan(scale[0])

    varying = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    assert M.seasonal_naive_scale(varying, season_length=2)[0] == pytest.approx(2.0)


def test_mase_excludes_degenerate_series_rather_than_flooring_them() -> None:
    truth = np.array([[1.0, 1.0], [5.0, 5.0]])
    prediction = np.repeat((truth + 1.0)[:, :, None], len(QUANTILES), axis=2)
    scale = np.array([np.nan, 2.0])
    # Only the second series is usable, and its error of 1 over a denominator of 2 is
    # 0.5. Flooring the first at 1e-6 would give a MASE in the hundreds of thousands.
    assert M.mase(truth, prediction, QUANTILES, scale) == pytest.approx(0.5)


def test_mase_with_no_usable_series_raises() -> None:
    truth = np.array([[1.0, 1.0]])
    prediction = np.repeat(truth[:, :, None], len(QUANTILES), axis=2)
    with pytest.raises(M.MetricError, match="no denominator"):
        M.mase(truth, prediction, QUANTILES, np.array([np.nan]))


def test_violation_rate_catches_impossible_forecasts() -> None:
    prediction = np.zeros((2, 2, len(QUANTILES)))
    prediction[0] = -1.0  # negative people
    prediction[1] = 100.0  # more people than the room holds
    lower = np.zeros(2)
    upper = np.array([10.0, 10.0])
    assert M.violation_rate(prediction, QUANTILES, lower, upper) == 1.0
    prediction[:] = 5.0
    assert M.violation_rate(prediction, QUANTILES, lower, upper) == 0.0


def test_quantile_crossing_is_detected() -> None:
    ordered = np.arange(len(QUANTILES), dtype=float).reshape(1, 1, -1)
    assert M.quantile_crossing_rate(ordered) == 0.0
    assert M.quantile_crossing_rate(ordered[:, :, ::-1]) == 1.0


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #


def test_a_clearly_better_method_is_significant() -> None:
    generator = np.random.default_rng(0)
    better = generator.normal(1.0, 0.2, size=200)
    worse = generator.normal(2.0, 0.2, size=200)
    result = diebold_mariano(better, worse, horizon=1)
    assert result.favours == "a"
    assert result.p_value < 1e-6
    assert result.marker() == "*"


def test_two_equally_good_methods_are_not_significant() -> None:
    generator = np.random.default_rng(1)
    a = generator.normal(1.0, 0.5, size=200)
    b = generator.normal(1.0, 0.5, size=200)
    assert diebold_mariano(a, b, horizon=1).p_value > 0.05


def test_the_truncation_lag_follows_the_horizon() -> None:
    generator = np.random.default_rng(2)
    a = generator.normal(1.0, 0.3, size=400)
    b = a + generator.normal(0.05, 0.3, size=400)
    assert diebold_mariano(a, b, horizon=1).lag == 0
    assert diebold_mariano(a, b, horizon=24).lag == 23


def test_autocorrelated_differentials_inflate_the_long_run_variance() -> None:
    # Overlapping multi-step forecasts produce serially correlated loss differentials,
    # and the whole reason for the Newey-West window is that the naive variance then
    # understates the uncertainty. On an i.i.d. series there is nothing to correct, so
    # the claim has to be tested on a series that actually has the dependence.
    generator = np.random.default_rng(4)
    noise = generator.normal(0.0, 1.0, size=2000)
    persistent = np.convolve(noise, np.ones(24), mode="valid")
    assert newey_west_variance(persistent, lag=23) > 3.0 * newey_west_variance(
        persistent, lag=0
    )
    assert newey_west_variance(noise, lag=0) == pytest.approx(np.var(noise), rel=0.05)


def test_too_few_origins_for_the_horizon_is_refused() -> None:
    generator = np.random.default_rng(3)
    a = generator.normal(1.0, 0.3, size=20)
    b = generator.normal(1.1, 0.3, size=20)
    with pytest.raises(SignificanceError, match="cannot support"):
        diebold_mariano(a, b, horizon=60)


def test_identical_methods_cannot_be_compared() -> None:
    a = np.arange(50.0)
    with pytest.raises(SignificanceError, match="not positive"):
        diebold_mariano(a, a.copy(), horizon=1)


def test_holm_is_stricter_than_no_correction_and_looser_than_bonferroni() -> None:
    p_values = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    verdict = holm_bonferroni(p_values, alpha=0.05)
    assert verdict["a"] is True
    # 0.02 <= 0.05/3 is false, so b fails and everything after it fails too.
    assert verdict["b"] is False
    assert verdict["c"] is False
    assert verdict["d"] is False


def test_holm_on_an_empty_family_is_empty() -> None:
    assert holm_bonferroni({}) == {}


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_result(toy_site, plan):
    specs = [
        MethodSpec(build_forecaster("last_value", seed=0)),
        MethodSpec(build_forecaster("historical_average", seed=0)),
        MethodSpec(
            build_forecaster("last_value", seed=0),
            ProjectionReconciler(weighting="uncertainty"),
            label="last_value+proposed",
        ),
    ]
    return run_evaluation(toy_site, plan, specs, mase_season=60)


def test_every_method_sees_the_same_origins(toy_result, plan) -> None:
    per_method = toy_result.metrics.groupby("method")["origin"].apply(set)
    assert len(set(map(frozenset, per_method))) == 1
    assert set(per_method.iloc[0]) == set(plan.origins())


def test_predictions_carry_every_quantile_and_the_actual(toy_result, plan, toy_site) -> None:
    columns = set(toy_result.predictions.columns)
    for level in plan.quantiles:
        assert f"q{round(level * 100):02d}" in columns
    assert "actual" in columns
    expected = len(plan) * len(toy_site.series_ids) * plan.horizon * 3
    assert len(toy_result.predictions) == expected


def test_the_recorded_actual_is_the_one_the_metric_used(toy_result, plan, toy_site) -> None:
    panel = toy_site.to_panel()
    task = plan.tasks[0]
    frame = toy_result.predictions
    rows = frame[(frame["origin"] == task.origin) & (frame["method"] == "last_value+none")]
    recorded = rows.pivot(index="series_id", columns="step", values="actual")
    recorded = recorded.reindex(toy_site.series_ids)
    np.testing.assert_allclose(recorded.to_numpy(), truth_for(panel, task), equal_nan=True)


def test_latency_and_memory_are_recorded_for_every_call(toy_result, plan) -> None:
    assert len(toy_result.resources) == 3 * len(plan)
    assert (toy_result.resources["latency_ms"] > 0).all()
    assert toy_result.resources["peak_memory_mb"].notna().all()


def test_reconciliation_removes_the_conservation_residual(toy_result) -> None:
    frame = toy_result.reconciliation
    assert not frame.empty
    assert (frame["residual_before"] > 0).any()
    assert (frame["residual_after"] < 1e-6).all()


def test_the_reconciled_method_is_coherent_and_the_raw_one_is_not(toy_result) -> None:
    summary = toy_result.summary().query("series_group == 'all'").set_index(["method", "horizon"])
    horizon = summary.index.get_level_values("horizon").max()
    raw = summary.loc[("last_value+none", horizon), "conservation_residual_mae"]
    reconciled = summary.loc[("last_value+proposed", horizon), "conservation_residual_mae"]
    assert raw > 1e-6
    assert reconciled < 1e-6


def test_an_identity_reconciler_changes_nothing(toy_site, plan) -> None:
    raw = run_evaluation(
        toy_site, plan, [MethodSpec(build_forecaster("last_value", seed=0))], mase_season=60
    )
    identity = run_evaluation(
        toy_site,
        plan,
        [
            MethodSpec(
                build_forecaster("last_value", seed=0),
                IdentityReconciler(),
                label="last_value+none",
            )
        ],
        mase_season=60,
    )
    pd.testing.assert_frame_equal(
        raw.predictions.drop(columns=["method"]),
        identity.predictions.drop(columns=["method"]),
    )


def test_the_harness_is_deterministic(toy_site, plan) -> None:
    def once():
        return run_evaluation(
            toy_site,
            plan,
            [MethodSpec(build_forecaster("historical_average", seed=0))],
            mase_season=60,
        ).predictions

    pd.testing.assert_frame_equal(once(), once())


def test_mint_and_the_proposal_both_reconcile_but_differ(toy_site, plan) -> None:
    result = run_evaluation(
        toy_site,
        plan,
        [
            MethodSpec(
                build_forecaster("last_value", seed=0), MinTReconciler(), label="mint"
            ),
            MethodSpec(
                build_forecaster("last_value", seed=0),
                ProjectionReconciler(weighting="uncertainty"),
                label="proposed",
            ),
        ],
        mase_season=60,
    )
    per_method = result.reconciliation.groupby("method")["residual_after"].max()
    assert (per_method < 1e-6).all()
    # The two projections use different metrics, so they land in different places.
    mint = result.predictions.query("method == 'mint'")["q50"].to_numpy()
    proposed = result.predictions.query("method == 'proposed'")["q50"].to_numpy()
    assert not np.allclose(mint, proposed)


# --------------------------------------------------------------------------- #
# Origin eligibility over a record with holes
# --------------------------------------------------------------------------- #


def _holed(panel: Panel, hole: slice) -> Panel:
    """The same panel with every target unobserved over ``hole``."""
    series = panel.series.copy()
    series[:, hole] = np.nan
    return Panel(
        series=series,
        series_ids=panel.series_ids,
        timestamps=panel.timestamps,
        past_covariates=panel.past_covariates,
        past_covariate_ids=panel.past_covariate_ids,
        future_covariates=panel.future_covariates,
        future_covariate_ids=panel.future_covariate_ids,
        interval_seconds=panel.interval_seconds,
        site_id=panel.site_id,
    )


def _plan_over(panel: Panel, **kwargs):
    return build_plan(
        panel, context_length=20, horizons=[4], stride=1, quantiles=QUANTILES, **kwargs
    )


def test_by_default_no_origin_is_dropped(toy_site) -> None:
    # Simulated sites are complete, and there dropping nothing is the stricter guarantee:
    # a dropped origin would mean a bug, not a hole in the record.
    plan = _plan_over(_holed(toy_site.to_panel(), slice(205, 232)))
    assert plan.dropped == {}
    assert plan.describe()["n_origins_dropped"] == 0


def test_an_origin_inside_a_hole_is_dropped_and_the_reason_recorded(toy_site) -> None:
    panel = toy_site.to_panel()
    full = _plan_over(panel)
    holed = _plan_over(_holed(panel, slice(205, 232)), require_observed=True)

    assert len(holed) < len(full)
    assert holed.dropped, "a forty-step hole must cost some origins"
    assert set(holed.dropped.values()) <= {
        "a target series has no observation anywhere in its context",
        "the target window is entirely unobserved",
    }
    # The count reaches the manifest, so a run over a gappy record cannot quietly
    # evaluate on a third of the origins the stride implies.
    described = holed.describe()
    assert described["n_origins_dropped"] == len(holed.dropped)
    assert sum(described["dropped_reasons"].values()) == len(holed.dropped)


def test_a_surviving_origin_really_has_context_and_truth(toy_site) -> None:
    panel = _holed(toy_site.to_panel(), slice(205, 232))
    plan = _plan_over(panel, require_observed=True)
    for task in plan:
        context = panel.series[:, task.context_start : task.origin]
        assert np.isfinite(context).any(axis=1).all()
        assert np.isfinite(panel.series[:, task.origin : task.origin + task.horizon]).any()


def test_a_channel_that_is_never_measured_does_not_veto_every_origin(toy_site) -> None:
    # ROBOD measures no doorway flow at all. That is a property of the dataset, not of any
    # origin, and judging origins against it would reject the entire test window.
    panel = toy_site.to_panel()
    series = panel.series.copy()
    series[3:, :] = np.nan
    dead = Panel(
        series=series,
        series_ids=panel.series_ids,
        timestamps=panel.timestamps,
        past_covariates=None,
        past_covariate_ids=[],
        future_covariates=None,
        future_covariate_ids=[],
        interval_seconds=panel.interval_seconds,
        site_id=panel.site_id,
    )
    plan = _plan_over(dead, require_observed=True)
    assert len(plan) == len(_plan_over(dead))
    assert plan.dropped == {}


def test_a_record_with_nothing_observed_refuses_rather_than_returning_nothing(
    toy_site,
) -> None:
    panel = _holed(toy_site.to_panel(), slice(None))
    with pytest.raises(ProtocolError, match="no window with both a full context"):
        _plan_over(panel, require_observed=True)


def test_capping_keeps_usable_origins_not_candidates(toy_site) -> None:
    # Thinning has to come after the eligibility filter, or a capped run over a gappy
    # record would spend most of its budget on holes.
    panel = _holed(toy_site.to_panel(), slice(205, 232))
    plan = _plan_over(panel, require_observed=True, max_origins=5)
    assert len(plan) == 5
    for task in plan:
        assert np.isfinite(panel.series[:, task.origin : task.origin + task.horizon]).any()


def test_paired_losses_refuse_an_unknown_method(toy_result) -> None:
    with pytest.raises(Exception, match="not in the metrics"):
        paired_losses(toy_result.metrics, "last_value+none", "nonexistent")


def test_a_mismatched_truth_panel_is_rejected(toy_site, plan) -> None:
    panel = toy_site.to_panel()
    shorter = Panel(
        series=panel.series[:, :10],
        series_ids=panel.series_ids,
        timestamps=panel.timestamps[:10],
        past_covariates=None,
        past_covariate_ids=[],
        future_covariates=None,
        future_covariate_ids=[],
        interval_seconds=panel.interval_seconds,
        site_id=panel.site_id,
    )
    with pytest.raises(Exception, match="must describe the same series and grid"):
        run_evaluation(
            toy_site,
            plan,
            [MethodSpec(build_forecaster("last_value", seed=0))],
            truth_panel=shorter,
            mase_season=60,
        )
