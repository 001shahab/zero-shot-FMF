"""M2 acceptance: the Tier A graph simulator.

Three criteria from the specification:

1. aggregate statistics fall inside the configured target ranges;
2. the conservation residual on raw simulator output is exactly zero;
3. two runs with the same seed are byte-identical.

The ranges are part of each site YAML rather than hard-coded here, so a new site declares
what "plausible" means for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mflow.graph import BuildingGraph
from mflow.paths import configs_dir
from mflow.schema import conservation_residual, validate_site, write_site
from mflow.sim import GraphSimulator, aggregate, load_site_config, summarise
from mflow.sim.arrivals import intensity_per_step, sample_day_arrivals
from mflow.sim.config import SiteConfig, StyleMix
from mflow.sim.visitors import STYLE_PARAMETERS, build_policies, sample_styles

SITES = ["house_museum", "palazzo", "national"]


@pytest.fixture(scope="module")
def house_config() -> SiteConfig:
    return load_site_config(configs_dir("sites") / "house_museum.yaml")


@pytest.fixture(scope="module")
def house_run(house_config: SiteConfig):
    simulator = GraphSimulator(house_config, seed=101)
    result = simulator.run("2026-03-03", days=2)
    return simulator, result


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", SITES)
def test_shipped_sites_load_and_are_connected(name: str) -> None:
    config = load_site_config(configs_dir("sites") / f"{name}.yaml")
    simulator = GraphSimulator(config, seed=0)
    # Every interior room must be reachable from outside and able to reach it again.
    for node in config.nodes:
        if node.id == config.outside_node:
            continue
        assert node.id in simulator._next_hop_to_exit, f"{node.id} cannot reach the exit"


def test_site_sizes_match_the_specification() -> None:
    sizes = {
        name: len(load_site_config(configs_dir("sites") / f"{name}.yaml").nodes)
        for name in SITES
    }
    # Counts include the virtual outside node.
    assert sizes["house_museum"] == 9
    assert sizes["palazzo"] == 25
    assert sizes["national"] >= 60


def test_style_mix_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        StyleMix(ant=0.5, fish=0.2, butterfly=0.2, grasshopper=0.2)


def test_step_must_divide_the_interval(house_config: SiteConfig) -> None:
    payload = house_config.model_dump()
    payload["step_seconds"] = 7
    with pytest.raises(ValueError, match="must divide interval_seconds"):
        SiteConfig.model_validate(payload)


# --------------------------------------------------------------------------- #
# Behaviour models
# --------------------------------------------------------------------------- #


def test_style_parameters_encode_the_typology() -> None:
    # A grasshopper dwells longest and is most interest-driven; a fish dwells least and
    # is most drawn to large open rooms. These orderings are what the styles mean.
    dwell = {k: v.dwell_multiplier for k, v in STYLE_PARAMETERS.items()}
    assert dwell["grasshopper"] == max(dwell.values())
    assert dwell["fish"] == min(dwell.values())
    assert (
        STYLE_PARAMETERS["grasshopper"].interest_exponent
        > STYLE_PARAMETERS["ant"].interest_exponent
    )
    assert STYLE_PARAMETERS["fish"].area_weight == max(
        p.area_weight for p in STYLE_PARAMETERS.values()
    )
    assert STYLE_PARAMETERS["ant"].route_weight == max(
        p.route_weight for p in STYLE_PARAMETERS.values()
    )


def test_ant_prefers_the_suggested_route(house_config: SiteConfig) -> None:
    policies = build_policies(house_config)
    # From the parlour the route continues to the dining room; the shop is off-route.
    candidates = ["dining", "shop"]
    ant = policies["ant"].next_node_weights("parlour", candidates, None, set())
    grasshopper = policies["grasshopper"].next_node_weights("parlour", candidates, None, set())
    assert ant[0] / ant[1] > grasshopper[0] / grasshopper[1]


def test_exit_probability_rises_with_fatigue(house_config: SiteConfig) -> None:
    policy = build_policies(house_config)["butterfly"]
    assert policy.exit_probability(10.0, 0.2) < policy.exit_probability(90.0, 0.2)


def test_style_mix_is_respected(house_config: SiteConfig) -> None:
    generator = np.random.default_rng(3)
    styles = sample_styles(20_000, house_config, generator)
    share = styles.count("ant") / len(styles)
    assert abs(share - house_config.style_mix.ant) < 0.02


# --------------------------------------------------------------------------- #
# Arrivals
# --------------------------------------------------------------------------- #


def test_arrivals_only_happen_during_opening_hours(house_config: SiteConfig) -> None:
    day = pd.Timestamp("2026-03-07", tz=house_config.timezone)
    intensity = intensity_per_step(house_config, day)
    steps_per_hour = 3600 // house_config.step_seconds
    assert intensity[: 10 * steps_per_hour].sum() == 0.0  # shut before 10:00
    assert intensity[18 * steps_per_hour :].sum() == 0.0  # shut after 18:00
    assert intensity[10 * steps_per_hour : 18 * steps_per_hour].sum() > 0.0


def test_monday_closure_produces_no_walk_ins(house_config: SiteConfig) -> None:
    monday = pd.Timestamp("2026-03-02", tz=house_config.timezone)
    generator = np.random.default_rng(0)
    arrivals = sample_day_arrivals(house_config, monday, generator)
    # The weekday multiplier is zero on Monday, so only scheduled groups can appear, and
    # the house museum schedules none on a Monday.
    assert all(a.group_id is not None for a in arrivals)


def test_scheduled_groups_arrive_together(house_config: SiteConfig) -> None:
    tuesday = pd.Timestamp("2026-03-03", tz=house_config.timezone)
    arrivals = sample_day_arrivals(house_config, tuesday, np.random.default_rng(5))
    school = [a for a in arrivals if a.group_id == "school_morning"]
    assert len(school) > 10
    assert len({a.step for a in school}) == 1


# --------------------------------------------------------------------------- #
# The acceptance criteria
# --------------------------------------------------------------------------- #


def test_conservation_is_exactly_zero(house_run) -> None:
    simulator, result = house_run
    site = aggregate(simulator, result, drop_warmup_days=0)
    residual = conservation_residual(site)
    assert float(np.max(np.abs(residual.to_numpy()))) == 0.0


def test_aggregated_site_passes_canonical_validation(house_run, tmp_path: Path) -> None:
    simulator, result = house_run
    site = aggregate(simulator, result, drop_warmup_days=1)
    write_site(site, tmp_path / "sim")
    validate_site(tmp_path / "sim")


def test_occupancy_never_exceeds_capacity(house_run) -> None:
    simulator, result = house_run
    site = aggregate(simulator, result, drop_warmup_days=0)
    capacity = site.node_capacity()
    over = site.occupancy[
        site.occupancy["count"] > site.occupancy["node_id"].map(capacity)
    ]
    assert len(over) == 0


def test_same_seed_is_byte_identical(house_config: SiteConfig) -> None:
    first = GraphSimulator(house_config, seed=42).run("2026-03-03", days=1)
    second = GraphSimulator(house_config, seed=42).run("2026-03-03", days=1)
    pd.testing.assert_frame_equal(first.events, second.events)
    pd.testing.assert_frame_equal(first.visits, second.visits)


def test_different_seeds_differ(house_config: SiteConfig) -> None:
    first = GraphSimulator(house_config, seed=42).run("2026-03-03", days=1)
    second = GraphSimulator(house_config, seed=43).run("2026-03-03", days=1)
    assert not first.events.equals(second.events)


def test_statistics_fall_inside_the_configured_targets(house_config: SiteConfig) -> None:
    simulator = GraphSimulator(house_config, seed=7)
    result = simulator.run("2026-03-03", days=3)
    site = aggregate(simulator, result, drop_warmup_days=1)
    statistics = summarise(result, site)
    targets = house_config.targets
    assert targets is not None
    for name, (low, high) in (
        ("mean_visit_minutes", targets.mean_visit_minutes),
        ("node_visit_fraction", targets.node_visit_fraction),
        ("occupancy_peak_to_mean", targets.occupancy_peak_to_mean),
    ):
        assert low <= statistics[name] <= high, f"{name}={statistics[name]:.3f}"


def test_congestion_appears_at_the_narrow_staircase(house_config: SiteConfig) -> None:
    simulator = GraphSimulator(house_config, seed=11)
    result = simulator.run("2026-03-07", days=1)  # a Saturday, the busiest day
    blocked = result.blocked_requests
    stair_edges = [e for e in blocked if e.startswith("e_sta_cor")]
    assert stair_edges, "the staircase edge is missing from the simulator"
    # The narrow stair must queue at least sometimes; if it never does, the capacity
    # model is not binding anywhere and the congestion experiment has no signal.
    assert sum(blocked[e] for e in stair_edges) > 0


def test_groups_show_up_as_known_future_covariates(house_run) -> None:
    simulator, result = house_run
    site = aggregate(simulator, result, drop_warmup_days=0)
    future = site.covariates_future
    assert set(future["variable"].unique()) == {
        "is_open",
        "timed_slot_admissions",
        "tour_departure",
        "group_size_booked",
    }
    tours = future[(future["variable"] == "tour_departure") & (future["value"] > 0)]
    assert len(tours) > 0


def test_series_adjacency_covers_occupancy_and_flow(house_run) -> None:
    simulator, result = house_run
    site = aggregate(simulator, result, drop_warmup_days=0)
    graph = BuildingGraph.from_site(site)
    adjacency = graph.series_adjacency(site.series_ids)
    assert adjacency.shape == (len(site.series_ids), len(site.series_ids))
    np.testing.assert_array_equal(adjacency, adjacency.T)
    assert adjacency.diagonal().min() == 1.0
    # A flow series is connected to the occupancy of the rooms it joins.
    index = {sid: i for i, sid in enumerate(site.series_ids)}
    flow = next(s for s in site.series_ids if s.startswith("flow:e_ent_par"))
    assert adjacency[index[flow], index["occ:entrance"]] == 1.0
    assert adjacency[index[flow], index["occ:parlour"]] == 1.0
