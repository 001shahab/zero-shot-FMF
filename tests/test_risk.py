"""M6 risk heads: congestion alerting, anomaly detection and exposure accounting.

The exposure tests check the formulation and its guard rails only. There is no validation
claim to test, because there is no dataset linking visitor load to a conservation outcome,
and the module says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mflow.risk import (
    AnomalyError,
    CongestionError,
    ExposureBudget,
    ExposureError,
    anomaly_score,
    calibrate_threshold,
    exceedance_probability,
    first_breach_step,
    forecast_alerts,
    inject_anomalies,
    node_thresholds,
    person_minutes,
    project_exposure,
    reactive_alerts,
    score_alerts,
    score_detections,
    worked_example,
)

QUANTILES = tuple(round(0.1 * i, 1) for i in range(1, 10))


def fan(median: float, spread: float, shape: tuple[int, ...] = ()) -> np.ndarray:
    """A symmetric quantile fan centred on ``median``, linear in the level."""
    offsets = (np.asarray(QUANTILES) - 0.5) * 2.0 * spread
    return np.broadcast_to(median + offsets, (*shape, len(QUANTILES))).copy()


# --------------------------------------------------------------------------- #
# Congestion
# --------------------------------------------------------------------------- #


def test_the_median_sits_at_probability_one_half() -> None:
    forecast = fan(20.0, 5.0)
    assert exceedance_probability(forecast, QUANTILES, 20.0) == pytest.approx(0.5)


def test_a_threshold_at_the_upper_decile_gives_one_tenth() -> None:
    forecast = fan(20.0, 5.0)
    assert exceedance_probability(forecast, QUANTILES, forecast[-1]) == pytest.approx(0.1)
    assert exceedance_probability(forecast, QUANTILES, forecast[0]) == pytest.approx(0.9)


def test_the_tails_are_capped_not_extrapolated() -> None:
    forecast = fan(20.0, 5.0)
    # Far above q90 the answer is 0.1, not something smaller: the model said nothing
    # about the top decile and the code must not invent it.
    assert exceedance_probability(forecast, QUANTILES, 1e6) == pytest.approx(0.1)
    assert exceedance_probability(forecast, QUANTILES, -1e6) == pytest.approx(0.9)


def test_exceedance_keeps_the_leading_shape() -> None:
    forecast = fan(20.0, 5.0, shape=(4, 7))
    assert exceedance_probability(forecast, QUANTILES, 20.0).shape == (4, 7)


def test_mismatched_levels_are_refused() -> None:
    with pytest.raises(CongestionError, match="quantile columns"):
        exceedance_probability(fan(20.0, 5.0), QUANTILES[:-1], 20.0)
    with pytest.raises(CongestionError, match="strictly ascending"):
        exceedance_probability(fan(20.0, 5.0), QUANTILES[::-1], 20.0)


def test_an_alert_fires_when_any_step_of_the_horizon_is_at_risk() -> None:
    forecast = np.stack(
        [
            np.stack([fan(5.0, 2.0), fan(5.0, 2.0)]),  # calm throughout
            np.stack([fan(5.0, 2.0), fan(30.0, 2.0)]),  # fills at the second step
        ]
    )
    alerts = forecast_alerts(forecast, QUANTILES, threshold=20.0, alert_probability=0.3)
    np.testing.assert_array_equal(alerts, [False, True])


def test_a_higher_alert_probability_raises_fewer_alerts() -> None:
    forecast = np.stack([np.stack([fan(19.0, 4.0)]), np.stack([fan(24.0, 4.0)])])
    lenient = forecast_alerts(forecast, QUANTILES, threshold=20.0, alert_probability=0.2)
    strict = forecast_alerts(forecast, QUANTILES, threshold=20.0, alert_probability=0.8)
    assert lenient.sum() >= strict.sum()
    assert not strict[0]


def test_an_impossible_alert_probability_is_refused() -> None:
    forecast = np.stack([np.stack([fan(19.0, 4.0)])])
    with pytest.raises(CongestionError, match="strictly inside"):
        forecast_alerts(forecast, QUANTILES, threshold=20.0, alert_probability=1.0)


def test_first_breach_reports_the_earliest_step_and_minus_one_otherwise() -> None:
    actual = np.array([[1.0, 2.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(first_breach_step(actual, 20.0), [2, -1])


def test_an_unobserved_room_is_not_a_breach() -> None:
    actual = np.array([[np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(first_breach_step(actual, 20.0), [-1])


def test_a_perfect_detector_scores_one_on_both() -> None:
    alerts = np.array([True, False, True, False])
    breaches = np.array([3, -1, 1, -1])
    outcome = score_alerts(alerts, breaches, steps_per_day=1440)
    assert outcome.precision == 1.0
    assert outcome.recall == 1.0
    assert outcome.median_lead_time_steps == pytest.approx(2.0)
    assert outcome.false_alarms_per_node_per_day == 0.0


def test_a_detector_that_always_alerts_has_full_recall_and_poor_precision() -> None:
    alerts = np.ones(100, dtype=bool)
    breaches = np.where(np.arange(100) < 10, 5, -1)
    outcome = score_alerts(alerts, breaches, steps_per_day=100)
    assert outcome.recall == 1.0
    assert outcome.precision == pytest.approx(0.1)
    # 90 false alarms over one day of origins.
    assert outcome.false_alarms_per_node_per_day == pytest.approx(90.0)


def test_the_nuisance_rate_accounts_for_the_stride() -> None:
    alerts = np.ones(100, dtype=bool)
    breaches = np.full(100, -1)
    dense = score_alerts(alerts, breaches, steps_per_day=100, n_origins_per_step=1.0)
    sparse = score_alerts(alerts, breaches, steps_per_day=100, n_origins_per_step=0.2)
    # Evaluating every fifth step means the same origin count spans five days, so the
    # same alerts are a fifth as much of a nuisance.
    assert sparse.false_alarms_per_node_per_day == pytest.approx(
        dense.false_alarms_per_node_per_day / 5.0
    )


def test_lead_time_converts_to_minutes() -> None:
    outcome = score_alerts(
        np.array([True]), np.array([6]), steps_per_day=1440
    )
    assert outcome.lead_time_minutes(60) == pytest.approx(6.0)
    assert outcome.as_dict(60)["median_lead_time_minutes"] == pytest.approx(6.0)


def test_a_detector_with_no_alerts_reports_nan_precision_not_zero() -> None:
    outcome = score_alerts(np.zeros(10, dtype=bool), np.full(10, -1), steps_per_day=10)
    # Nothing fired, so there is no precision to report. Zero would say the detector was
    # wrong every time it spoke, which is a different and false claim.
    assert np.isnan(outcome.precision)
    assert np.isnan(outcome.recall)


def test_misaligned_arrays_are_refused() -> None:
    with pytest.raises(CongestionError, match="must be aligned"):
        score_alerts(np.zeros(5, dtype=bool), np.zeros(4, dtype=int), steps_per_day=10)


def test_the_forecast_buys_lead_time_the_reactive_detector_cannot() -> None:
    # A room that fills at step 3 of every horizon. The forecast sees it coming; the
    # reactive detector only sees the room as it is now, which is still calm.
    n_origins = 40
    forecast = np.zeros((n_origins, 5, len(QUANTILES)))
    actual = np.zeros((n_origins, 5))
    context_last = np.full(n_origins, 5.0)
    for origin in range(n_origins):
        rising = origin % 4 == 0
        for step in range(5):
            level = 30.0 if (rising and step >= 3) else 5.0
            forecast[origin, step] = fan(level, 2.0)
            actual[origin, step] = level

    breaches = first_breach_step(actual, 20.0)
    predicted = score_alerts(
        forecast_alerts(forecast, QUANTILES, 20.0, 0.3), breaches, steps_per_day=1440
    )
    reactive = score_alerts(reactive_alerts(context_last, 20.0), breaches, steps_per_day=1440)

    assert predicted.recall == 1.0
    assert predicted.median_lead_time_steps == pytest.approx(3.0)
    assert reactive.n_alerts == 0
    assert np.isnan(reactive.recall) or reactive.recall == 0.0


def test_thresholds_scale_with_capacity() -> None:
    thresholds = node_thresholds({"study": 8.0, "great_hall": 200.0}, occupancy_fraction=0.75)
    assert thresholds == {"study": 6.0, "great_hall": 150.0}


def test_an_out_of_range_occupancy_fraction_is_refused() -> None:
    with pytest.raises(CongestionError, match=r"must lie in \(0, 1\]"):
        node_thresholds({"study": 8.0}, occupancy_fraction=1.5)


# --------------------------------------------------------------------------- #
# Anomaly
# --------------------------------------------------------------------------- #


@pytest.fixture
def stream():
    generator = np.random.default_rng(7)
    base = 10.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, 600))
    series = np.stack([base + generator.normal(0, 0.5, 600) for _ in range(4)])
    prediction = np.repeat(series[:, :, None], len(QUANTILES), axis=2)
    offsets = (np.asarray(QUANTILES) - 0.5) * 4.0
    prediction = prediction + offsets
    return series, prediction


def test_a_clean_stream_scores_zero_inside_the_interval(stream) -> None:
    series, prediction = stream
    assert anomaly_score(series, prediction, QUANTILES).max() == 0.0


def test_the_score_grows_with_the_excursion(stream) -> None:
    series, prediction = stream
    mild = series.copy()
    mild[0, 100] += 3.0
    wild = series.copy()
    wild[0, 100] += 30.0
    assert (
        anomaly_score(wild, prediction, QUANTILES)[0, 100]
        > anomaly_score(mild, prediction, QUANTILES)[0, 100]
        > 0.0
    )


def test_a_confident_empty_room_cannot_dominate_the_threshold() -> None:
    # Every quantile is zero, so the interval has no width. Without the spread floor one
    # person arriving would score infinity and set the threshold for the whole site.
    actual = np.array([[1.0]])
    prediction = np.zeros((1, 1, len(QUANTILES)))
    score = anomaly_score(actual, prediction, QUANTILES, floor=0.5)
    assert np.isfinite(score).all()
    assert score[0, 0] == pytest.approx(2.0)


def test_a_shape_mismatch_is_refused(stream) -> None:
    series, prediction = stream
    with pytest.raises(AnomalyError, match="does not cover"):
        anomaly_score(series[:, :10], prediction, QUANTILES)


def test_the_threshold_holds_the_false_alarm_budget(stream) -> None:
    series, prediction = stream
    generator = np.random.default_rng(8)
    noisy = series + generator.normal(0, 2.0, series.shape)
    clean_score = anomaly_score(noisy, prediction, QUANTILES)
    threshold = calibrate_threshold(clean_score, 0.01)
    assert (clean_score > threshold).mean() <= 0.011


def test_an_impossible_budget_is_refused(stream) -> None:
    series, prediction = stream
    with pytest.raises(AnomalyError, match="strictly inside"):
        calibrate_threshold(anomaly_score(series, prediction, QUANTILES), 1.0)


def test_a_dwell_cluster_is_injected_where_it_is_reported(stream) -> None:
    series, _ = stream
    contaminated, injected = inject_anomalies(
        series, ["dwell_cluster"], seed=1, n_per_kind=3, duration=15
    )
    assert len(injected) == 3
    for anomaly in injected:
        window = slice(anomaly.start, anomaly.stop)
        row = anomaly.series_index
        np.testing.assert_allclose(
            contaminated[row, window] - series[row, window], anomaly.magnitude
        )
        assert anomaly.covers(anomaly.start)
        assert not anomaly.covers(anomaly.stop)


def test_a_stuck_sensor_freezes_the_stream(stream) -> None:
    series, _ = stream
    contaminated, injected = inject_anomalies(
        series, ["stuck_sensor"], seed=2, n_per_kind=2, duration=20
    )
    for anomaly in injected:
        window = contaminated[anomaly.series_index, anomaly.start : anomaly.stop]
        assert len(np.unique(window)) == 1


def test_injection_is_deterministic(stream) -> None:
    series, _ = stream
    first, a = inject_anomalies(series, ["dwell_cluster"], seed=3, n_per_kind=2, duration=10)
    second, b = inject_anomalies(series, ["dwell_cluster"], seed=3, n_per_kind=2, duration=10)
    np.testing.assert_array_equal(first, second)
    assert a == b


def test_a_closed_gallery_anomaly_needs_the_closure_mask(stream) -> None:
    series, _ = stream
    with pytest.raises(AnomalyError, match="needs closed_mask"):
        inject_anomalies(series, ["closed_gallery_flow"], seed=4, n_per_kind=1, duration=5)


def test_a_closed_gallery_anomaly_lands_while_the_site_is_shut(stream) -> None:
    series, _ = stream
    closed = np.zeros(series.shape, dtype=bool)
    closed[:, 300:400] = True
    contaminated, injected = inject_anomalies(
        series,
        ["closed_gallery_flow"],
        seed=5,
        n_per_kind=3,
        duration=10,
        closed_mask=closed,
        flow_rows=np.array([2, 3]),
    )
    for anomaly in injected:
        assert anomaly.series_index in (2, 3)
        assert closed[anomaly.series_index, anomaly.start : anomaly.stop].all()
    assert not np.allclose(contaminated, series)


def test_a_site_that_never_closes_cannot_host_the_anomaly(stream) -> None:
    series, _ = stream
    with pytest.raises(AnomalyError, match="no 10-step window"):
        inject_anomalies(
            series,
            ["closed_gallery_flow"],
            seed=6,
            n_per_kind=1,
            duration=10,
            closed_mask=np.zeros(series.shape, dtype=bool),
            flow_rows=np.array([2, 3]),
        )


def test_a_duration_longer_than_the_window_is_refused(stream) -> None:
    series, _ = stream
    with pytest.raises(AnomalyError, match="does not fit"):
        inject_anomalies(series, ["dwell_cluster"], seed=7, duration=10_000)


def test_large_anomalies_are_detected_at_a_one_percent_budget(stream) -> None:
    series, prediction = stream
    contaminated, injected = inject_anomalies(
        series, ["dwell_cluster"], seed=9, n_per_kind=6, duration=20, magnitude=8.0
    )
    clean_score = anomaly_score(series, prediction, QUANTILES)
    threshold = calibrate_threshold(clean_score, 0.01)
    outcome = score_detections(
        anomaly_score(contaminated, prediction, QUANTILES),
        injected,
        threshold,
        clean_score,
    )
    assert outcome.detection_rate == 1.0
    assert outcome.median_time_to_detect_steps == 0.0
    assert outcome.false_alarm_rate <= 0.01
    assert outcome.n_detected == outcome.n_injected == 6


def test_a_threshold_nothing_can_reach_detects_nothing(stream) -> None:
    series, prediction = stream
    contaminated, injected = inject_anomalies(
        series, ["dwell_cluster"], seed=10, n_per_kind=3, duration=10
    )
    clean_score = anomaly_score(series, prediction, QUANTILES)
    outcome = score_detections(
        anomaly_score(contaminated, prediction, QUANTILES), injected, 1e9, clean_score
    )
    assert outcome.detection_rate == 0.0
    assert np.isnan(outcome.median_time_to_detect_steps)
    assert outcome.as_dict(60)["n_injected"] == 3.0


# --------------------------------------------------------------------------- #
# Exposure
# --------------------------------------------------------------------------- #


def test_a_budget_needs_a_stated_rationale() -> None:
    with pytest.raises(ExposureError, match="no rationale"):
        ExposureBudget(node_id="study", person_minutes_per_day=100.0, rationale="  ")


def test_a_budget_must_be_positive() -> None:
    with pytest.raises(ExposureError, match="must be positive"):
        ExposureBudget(node_id="study", person_minutes_per_day=0.0, rationale="set by curator")


def test_person_minutes_integrates_occupancy() -> None:
    # Ten people held for six intervals of sixty seconds is sixty person-minutes.
    assert person_minutes(np.full(6, 10.0), 60) == pytest.approx(60.0)
    assert person_minutes(np.full(6, 10.0), 300) == pytest.approx(300.0)


def test_gaps_contribute_nothing_and_are_reported() -> None:
    observed = np.array([10.0, np.nan, 10.0, np.nan])
    budget = ExposureBudget(node_id="study", person_minutes_per_day=100.0, rationale="curator")
    projection = project_exposure(
        observed, np.zeros(0), budget, pd.Timestamp("2026-01-01"), 60
    )
    assert projection.observed_person_minutes == pytest.approx(20.0)
    assert projection.observed_fraction == pytest.approx(0.5)


def test_a_projection_adds_the_forecast_to_what_has_happened() -> None:
    budget = ExposureBudget(node_id="study", person_minutes_per_day=100.0, rationale="curator")
    projection = project_exposure(
        np.full(3, 10.0), np.full(3, 10.0), budget, pd.Timestamp("2026-01-01"), 60
    )
    assert projection.observed_person_minutes == pytest.approx(30.0)
    assert projection.forecast_person_minutes == pytest.approx(30.0)
    assert projection.projected_person_minutes == pytest.approx(60.0)
    assert projection.budget_utilisation == pytest.approx(0.6)
    assert not projection.exceeds_budget


def test_exceeding_the_budget_is_flagged() -> None:
    budget = ExposureBudget(node_id="study", person_minutes_per_day=50.0, rationale="curator")
    projection = project_exposure(
        np.full(3, 10.0), np.full(3, 10.0), budget, pd.Timestamp("2026-01-01"), 60
    )
    assert projection.exceeds_budget
    assert projection.budget_utilisation > 1.0


def test_a_two_dimensional_input_is_refused() -> None:
    budget = ExposureBudget(node_id="study", person_minutes_per_day=50.0, rationale="curator")
    with pytest.raises(ExposureError, match="one series each"):
        project_exposure(
            np.zeros((2, 3)), np.zeros(3), budget, pd.Timestamp("2026-01-01"), 60
        )


def test_the_worked_example_declares_itself_illustrative(toy_site) -> None:
    budget, daily = worked_example(toy_site.occupancy, "gallery_a", 60)
    assert "Illustrative only" in budget.rationale
    assert "not derived from any measured damage relationship" in budget.rationale
    assert set(daily.columns) >= {"day", "person_minutes", "budget_utilisation"}
    # The budget is 1.2x the median, so the median day sits at about 0.83 utilisation and
    # no more than half the days can be over.
    assert daily["budget_utilisation"].median() == pytest.approx(1 / 1.2, rel=1e-6)


def test_the_worked_example_rejects_an_unknown_room(toy_site) -> None:
    with pytest.raises(ExposureError, match="not in the occupancy frame"):
        worked_example(toy_site.occupancy, "no_such_room", 60)
