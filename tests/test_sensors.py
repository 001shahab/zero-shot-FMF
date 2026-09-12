"""M3 acceptance: sensor degradation.

The stated criterion is that under the ``realistic`` profile the correlation between
simulated CO2 and true occupancy, and the measured lag, fall within the ranges reported
in the occupancy sensing literature cited in :mod:`mflow.sensors.environment`. The lag
band comes from Fan, Ding and Sun (2022): 10-20 min in Meyn et al. and 30-45 min in
Rahman et al., so 10-45 min overall.

The correlation floor is a design choice rather than a quoted figure. The literature is
consistent that CO2 is the environmental variable most strongly related to occupancy, but
the reported coefficient varies with zone volume, ventilation and sensor placement, so no
single published number would be honest to test against. 0.7 at the best lag is asserted
as the threshold below which the proxy would not be worth calling one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mflow.paths import configs_dir
from mflow.schema import conservation_residual, load_site, validate_site, write_site
from mflow.sensors import (
    ENVIRONMENT_VARIABLES,
    CountingSensorModel,
    EnvironmentSensorModel,
    FaultModel,
    co2_lag_minutes,
    counting_error_summary,
    degrade,
    load_sensor_profile,
)
from mflow.sensors.config import FaultConfig, SensorProfile
from mflow.sensors.environment import humidity_ratio, relative_humidity
from mflow.sim import GraphSimulator, aggregate, load_site_config

PROFILES = ["clean", "realistic", "degraded", "harsh"]

#: Fan, Ding and Sun (2022): Meyn et al. report 10-20 min, Rahman et al. 30-45 min.
CITED_LAG_MINUTES = (10.0, 45.0)


@pytest.fixture(scope="module")
def clean_site():
    """A few days of the house museum, simulated once and shared by the whole module."""
    config = load_site_config(configs_dir("sites") / "house_museum.yaml")
    simulator = GraphSimulator(config, seed=13)
    result = simulator.run("2026-03-03", days=4)
    return aggregate(simulator, result, drop_warmup_days=1)


@pytest.fixture(scope="module")
def realistic():
    return load_sensor_profile(configs_dir("sensors") / "realistic.yaml")


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PROFILES)
def test_shipped_profiles_load(name: str) -> None:
    profile = load_sensor_profile(configs_dir("sensors") / f"{name}.yaml")
    assert profile.name == name
    assert profile.description.strip()


def test_miss_probabilities_cannot_exceed_certainty(realistic: SensorProfile) -> None:
    payload = realistic.model_dump()
    payload["counting"]["base_miss_probability"] = 0.9
    payload["counting"]["congestion_miss_probability"] = 0.5
    with pytest.raises(ValueError, match="must not exceed 1"):
        SensorProfile.model_validate(payload)


def test_a_dropout_that_never_ends_is_rejected() -> None:
    with pytest.raises(ValueError, match="never ends"):
        FaultConfig(
            dropout_enter_probability=0.01,
            dropout_exit_probability=0.0,
            stuck_enter_probability=0.0,
            stuck_exit_probability=0.05,
            clock_skew_intervals=0.0,
            clock_drift_intervals=0.0,
        )


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


def test_counting_is_lossy_and_biased(clean_site, realistic: SensorProfile) -> None:
    model = CountingSensorModel(
        realistic.counting, clean_site.edge_ids, np.random.default_rng(0)
    )
    observed = model.apply(clean_site.flow, np.random.default_rng(1))
    errors = counting_error_summary(clean_site.flow, observed)
    relative = errors["relative_error"].dropna()
    # Under-counting dominates, as occlusion is the larger of the two effects.
    assert relative.mean() < 0.0
    # Gruber et al. (2014) measured directional errors of 3.3% and 7.2% over 36 days;
    # Cokbas et al. (2020) report 80-90% of events classified correctly. A profile whose
    # aggregate error falls outside that bracket is not describing a real counter.
    assert 0.02 < relative.abs().mean() < 0.20


def test_counting_bias_is_fixed_per_sensor(clean_site, realistic: SensorProfile) -> None:
    model = CountingSensorModel(
        realistic.counting, clean_site.edge_ids, np.random.default_rng(0)
    )
    assert len(set(model.bias.values())) > 1  # sensors differ from each other
    # A second pass over the same data reuses the same biases, so the systematic part of
    # the error does not average out across days.
    again = CountingSensorModel(
        realistic.counting, clean_site.edge_ids, np.random.default_rng(0)
    )
    assert model.bias == again.bias


def test_counting_misses_more_when_busy(realistic: SensorProfile) -> None:
    quiet = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2000, freq="1min", tz="UTC"),
            "edge_id": "e1",
            "count": 1.0,
        }
    )
    busy = quiet.assign(count=40.0)
    model = CountingSensorModel(realistic.counting, ["e1"], np.random.default_rng(0))
    model.bias["e1"] = 1.0  # isolate the occlusion term from the fixed bias
    quiet_rate = 1 - model.apply(quiet, np.random.default_rng(2))["count"].sum() / 2000
    busy_rate = 1 - model.apply(busy, np.random.default_rng(2))["count"].sum() / 80_000
    assert busy_rate > quiet_rate + 0.05


def test_clean_profile_leaves_counts_untouched(clean_site) -> None:
    profile = load_sensor_profile(configs_dir("sensors") / "clean.yaml")
    model = CountingSensorModel(profile.counting, clean_site.edge_ids, np.random.default_rng(0))
    observed = model.apply(clean_site.flow, np.random.default_rng(0))
    pd.testing.assert_series_equal(
        observed["count"], clean_site.flow["count"], check_dtype=False
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def test_humidity_conversions_round_trip() -> None:
    temperature = np.array([18.0, 21.0, 26.0])
    ratio = humidity_ratio(temperature, 55.0)
    np.testing.assert_allclose(relative_humidity(temperature, ratio), 55.0, rtol=1e-9)


def test_co2_rises_towards_the_mass_balance_steady_state(realistic: SensorProfile) -> None:
    # A 100 m2 room at 4 m and 2 air changes per hour, held at 20 people. The steady state
    # follows directly from the mass balance: C_out + 1e6 * G * n / Q.
    model = EnvironmentSensorModel(realistic.environment, {"hall": 100.0}, np.random.default_rng(0))
    volume, airflow = model.zone_constants("hall")
    assert volume == pytest.approx(400.0)
    assert airflow == pytest.approx(400.0 * 2 / 3600)

    occupancy = np.full(600, 20.0)
    quiet = realistic.environment.model_copy(
        update={"co2_noise_ppm": 0.0, "co2_noise_fraction": 0.0, "co2_offset_ppm": 0.0}
    )
    noiseless = EnvironmentSensorModel(quiet, {"hall": 100.0}, np.random.default_rng(0))
    series = noiseless.simulate_node("hall", occupancy, 60, np.random.default_rng(0))

    expected = quiet.outdoor_co2_ppm + 1e6 * (quiet.co2_generation_l_per_s / 1000.0) * 20 / airflow
    assert series["co2_ppm"][-1] == pytest.approx(expected, rel=1e-6)
    # A sanity check on the units rather than on the arithmetic. Twenty people in 100 m2
    # is close to the ASHRAE default museum density of 4.6 m2 per person, and 2 air
    # changes per hour supplies them 11 L/s each, so the room should settle a few hundred
    # ppm above ambient -- not at ambient, and not at the 5000 ppm a unit slip would give.
    assert 800.0 < expected < 1500.0


def test_co2_is_independent_of_the_sampling_interval(realistic: SensorProfile) -> None:
    # The balance is integrated exactly, so an hour of constant occupancy reaches the same
    # concentration whether it is stepped in 1-minute or 5-minute intervals.
    quiet = realistic.environment.model_copy(
        update={
            "co2_noise_ppm": 0.0,
            "co2_noise_fraction": 0.0,
            "co2_offset_ppm": 0.0,
            "co2_response_seconds": 0.0,
        }
    )
    model = EnvironmentSensorModel(quiet, {"hall": 100.0}, np.random.default_rng(0))
    fine = model.simulate_node("hall", np.full(60, 10.0), 60, np.random.default_rng(0))
    coarse = model.simulate_node("hall", np.full(12, 10.0), 300, np.random.default_rng(0))
    assert fine["co2_ppm"][-1] == pytest.approx(coarse["co2_ppm"][-1], rel=1e-9)


def test_an_empty_room_returns_to_baseline(realistic: SensorProfile) -> None:
    quiet = realistic.environment.model_copy(
        update={
            "co2_noise_ppm": 0.0,
            "co2_noise_fraction": 0.0,
            "co2_offset_ppm": 0.0,
            "temperature_noise_c": 0.0,
            "humidity_noise_pct": 0.0,
        }
    )
    model = EnvironmentSensorModel(quiet, {"hall": 100.0}, np.random.default_rng(0))
    occupancy = np.concatenate([np.full(300, 30.0), np.zeros(600)])
    series = model.simulate_node("hall", occupancy, 60, np.random.default_rng(0))
    assert series["co2_ppm"][-1] == pytest.approx(quiet.outdoor_co2_ppm, abs=1.0)
    assert series["temp_c"][-1] == pytest.approx(quiet.baseline_temperature_c, abs=0.05)
    assert series["rh_pct"][-1] == pytest.approx(quiet.baseline_humidity_pct, abs=0.5)


def test_occupancy_warms_a_room_and_adds_moisture(realistic: SensorProfile) -> None:
    quiet = realistic.environment.model_copy(
        update={"temperature_noise_c": 0.0, "humidity_noise_pct": 0.0}
    )
    model = EnvironmentSensorModel(quiet, {"hall": 60.0}, np.random.default_rng(0))
    series = model.simulate_node("hall", np.full(600, 25.0), 60, np.random.default_rng(0))
    assert series["temp_c"][-1] > quiet.baseline_temperature_c
    # Relative humidity can fall in a crowded room even though the air is taking up water,
    # because it warms faster than it moistens. The moisture content is what must rise.
    ratio = humidity_ratio(series["temp_c"][-1], series["rh_pct"][-1])
    baseline = humidity_ratio(quiet.baseline_temperature_c, quiet.baseline_humidity_pct)
    assert ratio > baseline
    assert series["rh_pct"].max() <= 100.0


def test_climate_control_damps_the_thermal_response(realistic: SensorProfile) -> None:
    # The plant fights the heat load but does not scrub CO2, so conditioning weakens the
    # temperature channel without touching the CO2 channel.
    occupancy = np.full(600, 25.0)
    quiet = realistic.environment.model_copy(
        update={"temperature_noise_c": 0.0, "co2_noise_ppm": 0.0, "co2_noise_fraction": 0.0}
    )
    uncontrolled = quiet.model_copy(update={"hvac_rejection_fraction": 0.0})
    controlled = quiet.model_copy(update={"hvac_rejection_fraction": 0.9})

    warm = EnvironmentSensorModel(uncontrolled, {"hall": 60.0}, np.random.default_rng(0))
    cool = EnvironmentSensorModel(controlled, {"hall": 60.0}, np.random.default_rng(0))
    warm_series = warm.simulate_node("hall", occupancy, 60, np.random.default_rng(0))
    cool_series = cool.simulate_node("hall", occupancy, 60, np.random.default_rng(0))

    warm_rise = warm_series["temp_c"][-1] - quiet.baseline_temperature_c
    cool_rise = cool_series["temp_c"][-1] - quiet.baseline_temperature_c
    assert cool_rise == pytest.approx(0.1 * warm_rise, rel=1e-6)
    assert warm_series["co2_ppm"][-1] == pytest.approx(cool_series["co2_ppm"][-1], rel=1e-9)


def test_a_node_without_area_fails_loudly(realistic: SensorProfile) -> None:
    model = EnvironmentSensorModel(realistic.environment, {"hall": 0.0}, np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive air volume"):
        model.zone_constants("hall")


def test_co2_lag_needs_a_varying_signal() -> None:
    with pytest.raises(ValueError, match="constant series"):
        co2_lag_minutes(np.ones(100), np.arange(100.0), 60)


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #


def test_dropouts_come_in_runs_not_isolated_samples(realistic: SensorProfile) -> None:
    model = FaultModel(realistic.faults, ["s"], np.random.default_rng(0))
    _, record = model.apply("s", np.arange(200_000.0), np.random.default_rng(1))
    assert record.dropout_fraction > 0.0
    gaps = np.diff(record.dropped)
    # In a Markov chain the dropped samples are mostly consecutive; under independent
    # per-sample dropout at the same rate they would almost all be isolated.
    assert float(np.mean(gaps == 1)) > 0.9


def test_dropout_rate_matches_the_configured_chain(realistic: SensorProfile) -> None:
    model = FaultModel(realistic.faults, ["s"], np.random.default_rng(0))
    _, record = model.apply("s", np.zeros(500_000), np.random.default_rng(1))
    enter = realistic.faults.dropout_enter_probability
    exit_ = realistic.faults.dropout_exit_probability
    assert record.dropout_fraction == pytest.approx(enter / (enter + exit_), rel=0.15)


def test_stuck_values_hold_the_previous_reading() -> None:
    config = FaultConfig(
        dropout_enter_probability=0.0,
        dropout_exit_probability=0.01,
        stuck_enter_probability=0.01,
        stuck_exit_probability=0.05,
        clock_skew_intervals=0.0,
        clock_drift_intervals=0.0,
    )
    model = FaultModel(config, ["s"], np.random.default_rng(0))
    values = np.arange(5000.0)
    observed, record = model.apply("s", values, np.random.default_rng(2))
    assert len(record.stuck) > 0
    for index in record.stuck:
        if index > 0 and np.isfinite(observed[index]):
            assert observed[index] == observed[index - 1]


def test_clock_shift_blanks_the_exposed_end() -> None:
    config = FaultConfig(
        dropout_enter_probability=0.0,
        dropout_exit_probability=0.01,
        stuck_enter_probability=0.0,
        stuck_exit_probability=0.01,
        clock_skew_intervals=3.0,
        clock_drift_intervals=0.0,
    )
    model = FaultModel(config, [f"s{i}" for i in range(50)], np.random.default_rng(0))
    shifted = [s for s in model.clock_skew if round(model.clock_skew[s]) != 0]
    assert shifted, "no sensor drew a non-zero clock skew"
    sensor = shifted[0]
    observed, record = model.apply(sensor, np.arange(100.0), np.random.default_rng(0))
    shift = int(record.clock_shift_intervals)
    # A shifted stream never invents a value: the exposed end is NaN rather than wrapped.
    exposed = observed[:shift] if shift > 0 else observed[shift:]
    assert np.isnan(exposed).all()


def test_an_unknown_sensor_is_an_error(realistic: SensorProfile) -> None:
    model = FaultModel(realistic.faults, ["s"], np.random.default_rng(0))
    with pytest.raises(KeyError, match="not declared to the fault model"):
        model.apply("other", np.zeros(10), np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# The pipeline, and the M3 acceptance criterion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PROFILES)
def test_observed_site_is_a_valid_canonical_site(
    clean_site, name: str, tmp_path: Path
) -> None:
    profile = load_sensor_profile(configs_dir("sensors") / f"{name}.yaml")
    observed, _ = degrade(clean_site, profile, seed=0)
    write_site(observed, tmp_path / name)
    validate_site(tmp_path / name)
    # Missing readings survive the round trip rather than becoming zeros.
    reloaded = load_site(tmp_path / name)
    assert len(reloaded.occupancy) == len(clean_site.occupancy)


def test_degradation_is_deterministic(clean_site, realistic: SensorProfile) -> None:
    first, first_report = degrade(clean_site, realistic, seed=5)
    second, second_report = degrade(clean_site, realistic, seed=5)
    pd.testing.assert_frame_equal(first.occupancy, second.occupancy)
    pd.testing.assert_frame_equal(first.flow, second.flow)
    pd.testing.assert_frame_equal(first.covariates_past, second.covariates_past)
    assert first_report.counting_bias == second_report.counting_bias


def test_a_different_seed_gives_different_damage(clean_site, realistic: SensorProfile) -> None:
    first, _ = degrade(clean_site, realistic, seed=5)
    second, _ = degrade(clean_site, realistic, seed=6)
    assert not first.flow.equals(second.flow)


def test_the_clean_site_is_not_modified(clean_site, realistic: SensorProfile) -> None:
    before = clean_site.flow.copy()
    degrade(clean_site, realistic, seed=0)
    pd.testing.assert_frame_equal(clean_site.flow, before)


def test_observation_breaks_conservation(clean_site, realistic: SensorProfile) -> None:
    # The clean site balances exactly; that is what makes the reconciliation testable.
    assert float(np.nanmax(np.abs(conservation_residual(clean_site).to_numpy()))) == 0.0
    observed, _ = degrade(clean_site, realistic, seed=0)
    assert observed.meta.has_ground_truth_flow is False
    residual = conservation_residual(observed)
    assert float(np.nanmax(np.abs(residual.to_numpy()))) > 0.0


def test_the_profile_is_recorded_on_the_observed_site(clean_site, realistic) -> None:
    observed, report = degrade(clean_site, realistic, seed=9)
    assert observed.meta.provenance["sensor_profile"] == "realistic"
    assert observed.meta.provenance["sensor_seed"] == 9
    assert observed.meta.provenance["clean_site_id"] == clean_site.meta.site_id
    assert report.summary()["profile"] == "realistic"


def test_every_stream_has_a_fault_record(clean_site, realistic: SensorProfile) -> None:
    _, report = degrade(clean_site, realistic, seed=0)
    expected = set(clean_site.series_ids) | {
        f"{node}|{variable}"
        for node in clean_site.interior_nodes
        for variable in ENVIRONMENT_VARIABLES
    }
    assert set(report.faults) == expected


def test_missing_data_is_reported_not_imputed(clean_site, realistic: SensorProfile) -> None:
    observed, report = degrade(clean_site, realistic, seed=0)
    wide = observed.wide_series()
    for series_id, record in report.faults.items():
        if series_id not in wide.columns:
            continue
        gaps = np.flatnonzero(wide[series_id].isna().to_numpy())
        # The record is exactly the set of gaps, not an approximation of it.
        np.testing.assert_array_equal(np.sort(gaps), np.sort(record.missing))


def test_dropout_severity_increases_across_the_profiles(clean_site) -> None:
    fractions = []
    for name in PROFILES:
        profile = load_sensor_profile(configs_dir("sensors") / f"{name}.yaml")
        _, report = degrade(clean_site, profile, seed=0)
        fractions.append(report.missing_fraction)
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions)


def test_realistic_co2_lag_and_correlation_match_the_literature(
    clean_site, realistic: SensorProfile
) -> None:
    """The M3 acceptance criterion."""
    observed, _ = degrade(clean_site, realistic, seed=0)
    co2 = observed.covariates_past.query("variable == 'co2_ppm'")
    truth = clean_site.occupancy.pivot(index="timestamp", columns="node_id", values="count")

    measured: list[tuple[str, float, float, float]] = []
    for node_id, group in co2.groupby("scope"):
        series = group.sort_values("timestamp")["value"].to_numpy()
        counts = truth[str(node_id)].to_numpy()
        finite = np.isfinite(series) & np.isfinite(counts)
        lag, correlation = co2_lag_minutes(
            counts[finite], series[finite], clean_site.meta.interval_seconds
        )
        measured.append((str(node_id), lag, correlation, float(np.std(counts[finite]))))

    assert len(measured) == len(clean_site.interior_nodes)
    low, high = CITED_LAG_MINUTES
    # Meyn et al. report "an average lag of 10-20 min ... among all the zones", so the
    # cited band is a building-level figure and the median across rooms is what compares
    # to it. Individual rooms spread either side of it, as they do in the field studies.
    median_lag = float(np.median([lag for _, lag, _, _ in measured]))
    assert low <= median_lag <= high, f"median CO2 lag {median_lag:.1f} min outside {low}-{high}"
    assert float(np.median([r for _, _, r, _ in measured])) > 0.7

    for node_id, lag, correlation, spread in measured:
        assert 5.0 <= lag <= 60.0, f"{node_id}: CO2 lag {lag:.1f} min is implausible"
        # A room that is almost always empty has no CO2 signal to correlate with, which is
        # the documented failure mode of CO2 as a proxy in sparsely occupied zones rather
        # than a defect of this model. The threshold applies where there is a signal.
        if spread >= 1.0:
            assert correlation > 0.7, f"{node_id}: CO2 correlation {correlation:.2f} too weak"


def test_co2_is_a_better_proxy_in_busier_rooms(clean_site, realistic: SensorProfile) -> None:
    observed, _ = degrade(clean_site, realistic, seed=0)
    co2 = observed.covariates_past.query("variable == 'co2_ppm'")
    truth = clean_site.occupancy.pivot(index="timestamp", columns="node_id", values="count")

    spreads, correlations = [], []
    for node_id, group in co2.groupby("scope"):
        series = group.sort_values("timestamp")["value"].to_numpy()
        counts = truth[str(node_id)].to_numpy()
        finite = np.isfinite(series) & np.isfinite(counts)
        _, correlation = co2_lag_minutes(
            counts[finite], series[finite], clean_site.meta.interval_seconds
        )
        spreads.append(float(np.std(counts[finite])))
        correlations.append(correlation)

    # This is what makes experiment E5 worth running: the environment-only condition is
    # not uniformly worse, it is worse exactly in the quiet rooms.
    assert np.corrcoef(spreads, correlations)[0, 1] > 0.5
