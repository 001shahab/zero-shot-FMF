"""The ROBOD adapter.

These tests run against a miniature built to the format verified from the published
dataset -- the same column names, the same ``YYYY-MM-DD HH:MM +08:00`` timestamps, the
same five-minute interval, and the same weekday-block structure with weekends absent.
They do not download the real 20 MB record, so they say nothing about whether ROBOD is
still published in that format. That question is the download manifest's job: the
checksums recorded by ``scripts/fetch_robod.py`` are what detects a re-release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mflow.data import AdapterError, DownloadManifest, FileRecord
from mflow.data.robod import (
    ROBOD_INTERVAL_SECONDS,
    ROOM_COVARIATES,
    ROOMS,
    WEATHER_COVARIATES,
    RobodAdapter,
)
from mflow.schema import load_site

#: Two weekdays, a weekend gap, then one more weekday -- the shape of the real record.
BLOCKS = [("2021-09-07", 2), ("2021-09-13", 1)]


def room_frame(rows: int, start: str, *, seed: int, wide: bool) -> pd.DataFrame:
    """One block of one room file, in the published column layout."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start, periods=rows, freq="5min", tz="Asia/Singapore")
    frame = pd.DataFrame(
        {
            "timestamp": stamps.strftime("%Y-%m-%d %H:%M %z").str.replace(
                r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True
            ),
            "occupant_presence": rng.integers(0, 2, rows),
            "occupant_count": rng.integers(0, 6, rows),
            "indoor_co2": rng.uniform(420, 900, rows),
            "air_temperature": rng.uniform(22, 27, rows),
            "indoor_relative_humidity": rng.uniform(55, 80, rows),
            "illuminance": rng.uniform(0, 500, rows),
            "wifi_connected_devices": rng.integers(0, 12, rows),
            "dry_bulb_temp": rng.uniform(24, 33, rows),
            "outdoor_relative_humidity": rng.uniform(60, 99, rows),
            "outdoor_co2": rng.uniform(400, 470, rows),
            "global_horizontal_solar_radiation": rng.uniform(0, 900, rows),
            "rainfall_raw": rng.uniform(0, 3, rows),
            "baromatic_pressure": rng.uniform(1000, 1010, rows),
        }
    )
    if wide:
        # Rooms 3, 4 and 5 are AHU-served and carry eight extra HVAC columns the adapter
        # does not use. It must ignore them rather than trip over them.
        for extra in ("supply_air_flow", "damper_position", "ahu_fan_speed"):
            frame[extra] = rng.uniform(0, 100, rows)
    return frame


@pytest.fixture
def raw(tmp_path):
    """A raw ROBOD directory: five room files plus a provenance manifest."""
    directory = tmp_path / "robod"
    directory.mkdir()
    for index, room in enumerate(ROOMS, start=1):
        blocks = BLOCKS if room in {"room1", "room2", "room3"} else [*BLOCKS, ("2021-09-20", 1)]
        frame = pd.concat(
            [room_frame(288 * days, start, seed=index, wide=index >= 3) for start, days in blocks],
            ignore_index=True,
        )
        frame.to_csv(directory / f"combined_Room{index}.csv", index=False)

    DownloadManifest(
        source="robod",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[FileRecord(name="combined_Room1.csv", url="https://x/1.csv", sha256="ab", bytes=1)],
        licence="CC BY 4.0",
        citation="Tekler et al. (2023)",
    ).write(directory)
    return directory


@pytest.fixture
def site(raw, tmp_path):
    """The converted, validated canonical site."""
    result = RobodAdapter(raw=raw, destination=tmp_path / "canonical").run()
    return load_site(result.sites[0])


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_the_five_rooms_become_one_building(site) -> None:
    assert site.meta.site_id == "robod_bldg1"
    assert site.meta.source == "real"
    assert site.meta.timezone == "Asia/Singapore"
    assert site.meta.interval_seconds == ROBOD_INTERVAL_SECONDS
    assert set(site.interior_nodes) == set(ROOMS)
    assert (site.nodes["kind"] == "outside").sum() == 1


def test_room_geometry_comes_from_the_published_table(site) -> None:
    nodes = site.nodes.set_index("node_id")
    assert nodes.loc["room5", "area_m2"] == pytest.approx(182.8)
    assert nodes.loc["room3", "capacity_persons"] == 15
    # Volumes are not part of the node schema, so they travel in the provenance where the
    # CO2 mass balance can find them without re-deriving a published number.
    assert site.meta.provenance["room_volumes_m3"]["room5"] == pytest.approx(1363.3)


def test_each_room_is_joined_only_to_outside(site) -> None:
    # The five rooms sit on four different levels and the dataset documents no doorway
    # between any pair of them, so no adjacency is invented.
    pairs = {(r.src_node, r.dst_node) for r in site.edges.itertuples()}
    assert pairs == {(room, "outside") for room in ROOMS} | {
        ("outside", room) for room in ROOMS
    }


def test_timestamps_are_converted_from_singapore_time(site) -> None:
    # Singapore is UTC+8 year round, so 00:00 local on the first day is 16:00 UTC the day
    # before. Getting this backwards would shift every diurnal pattern by sixteen hours.
    assert site.occupancy["timestamp"].min() == pd.Timestamp("2021-09-06 16:00", tz="UTC")


# --------------------------------------------------------------------------- #
# Gaps
# --------------------------------------------------------------------------- #


def test_the_weekend_gap_becomes_missing_values_not_fabricated_ones(site) -> None:
    counts = site.occupancy.pivot(index="timestamp", columns="node_id", values="count")
    # The grid is regular and gapless, as the contract demands...
    assert len(counts) == len(
        pd.date_range(counts.index.min(), counts.index.max(), freq="5min", tz="UTC")
    )
    # ...but the weekend is missing rather than interpolated, zero-filled or dropped.
    assert counts["room1"].isna().any()
    saturday = counts.loc["2021-09-09 04:00":"2021-09-10 04:00", "room1"]
    assert saturday.isna().all()


def test_a_room_with_a_shorter_record_does_not_truncate_the_others(site) -> None:
    counts = site.occupancy.pivot(index="timestamp", columns="node_id", values="count")
    # Rooms 4 and 5 were instrumented for an extra block. That is a hole in rooms 1-3,
    # not a reason to throw the extra block away.
    assert counts["room4"].notna().sum() > counts["room1"].notna().sum()


def test_the_missing_fraction_is_recorded(site) -> None:
    provenance = site.meta.provenance
    assert 0.0 < provenance["missing_occupancy_fraction"] < 1.0
    assert provenance["observed_occupancy_cells"] < provenance["grid_steps"] * len(ROOMS)


# --------------------------------------------------------------------------- #
# What ROBOD does not have
# --------------------------------------------------------------------------- #


def test_flow_is_dense_and_entirely_missing(site) -> None:
    # Deriving flow from successive occupancy differences would manufacture exactly the
    # quantity reconciliation is meant to be evaluated on.
    assert site.meta.has_ground_truth_flow is False
    assert site.flow["count"].isna().all()
    assert len(site.flow) == len(set(site.flow["timestamp"])) * len(site.edge_ids)
    assert "never derived" in site.meta.provenance["flow"]


def test_no_schedule_is_invented(site) -> None:
    assert site.meta.opening_hours == {}
    assert site.covariates_future.empty


# --------------------------------------------------------------------------- #
# Covariates
# --------------------------------------------------------------------------- #


def test_sensed_channels_are_scoped_to_their_room(site) -> None:
    past = site.covariates_past
    per_room = past[past["scope"] != "global"]
    assert set(per_room["variable"]) == set(ROOM_COVARIATES.values())
    assert set(per_room["scope"]) == set(ROOMS)


def test_outdoor_weather_is_global_and_stays_in_the_past(site) -> None:
    past = site.covariates_past
    weather = past[past["scope"] == "global"]
    assert set(weather["variable"]) == set(WEATHER_COVARIATES.values())
    # Measured outdoor temperature is an observation, not a forecast. Filing it as
    # known-future would leak it into every zero-shot evaluation.
    assert site.covariates_future.empty


def test_the_site_becomes_a_usable_panel(site) -> None:
    panel = site.to_panel()
    assert panel.series.shape[0] == len(ROOMS) + len(site.edge_ids)
    assert panel.past_covariates is not None
    assert np.isfinite(panel.series[: len(ROOMS)]).any()


# --------------------------------------------------------------------------- #
# Failure
# --------------------------------------------------------------------------- #


def test_a_renamed_column_stops_the_adapter(raw, tmp_path) -> None:
    path = raw / "combined_Room2.csv"
    frame = pd.read_csv(path)
    frame.rename(columns={"indoor_co2": "co2"}).to_csv(path, index=False)
    with pytest.raises(AdapterError, match="re-read from the new documentation"):
        RobodAdapter(raw=raw, destination=tmp_path / "canonical").run()


def test_a_missing_room_file_stops_the_adapter(raw, tmp_path) -> None:
    (raw / "combined_Room4.csv").unlink()
    with pytest.raises(AdapterError, match=r"re-run scripts/fetch_robod\.py"):
        RobodAdapter(raw=raw, destination=tmp_path / "canonical").run()


def test_duplicate_timestamps_stop_the_adapter(raw, tmp_path) -> None:
    path = raw / "combined_Room1.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(AdapterError, match="duplicate timestamps"):
        RobodAdapter(raw=raw, destination=tmp_path / "canonical").run()
