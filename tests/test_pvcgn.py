"""The PVCGN (HZMetro / SHMetro) adapter.

Built against a miniature in the verified layout: overlapping ``(T, 4, N, 2)`` windows
with matching ``xtime`` and ``ytime``, naive local timestamps on a fifteen-minute grid,
a service day that stops overnight, and a symmetric adjacency matrix with a non-zero
diagonal. Whether the real release still looks like this is the download manifest's job.

The channel-order test is the important one. The dataset README says ``(inflow/outflow)``
and the data says the reverse; the conservation constraint's sign depends on which is
right, so the adapter's choice is pinned here.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mflow.data import AdapterError, DownloadManifest, FileRecord
from mflow.data.pvcgn import (
    ENTRY_CHANNEL,
    EXIT_CHANNEL,
    EXTRACTED,
    PVCGN_INTERVAL_SECONDS,
    PvcgnAdapter,
)
from mflow.schema import load_site

N_STATIONS = 4
STEPS_PER_DAY = 8  # a short "service day", stopping overnight like the real record
DAYS = 3
WINDOW = 4


def service_grid() -> pd.DatetimeIndex:
    """Naive local timestamps: a run of steps each morning, nothing overnight."""
    days = [
        pd.date_range(f"2019-01-0{day + 1} 05:30", periods=STEPS_PER_DAY, freq="15min")
        for day in range(DAYS)
    ]
    return pd.DatetimeIndex(np.concatenate(days))


def ridership(grid: pd.DatetimeIndex) -> np.ndarray:
    """``(T, N, 2)`` with channel 0 as exits and channel 1 as entries.

    Shaped like the real thing: the day opens with entries and no exits, and closes with
    exits and no entries. That asymmetry is what the adapter's channel mapping is read
    from, so the fixture has to carry it.
    """
    values = np.zeros((len(grid), N_STATIONS, 2))
    for step in range(len(grid)):
        position = step % STEPS_PER_DAY
        entries = max(STEPS_PER_DAY - position, 0) * 10
        exits = position * 10
        values[step, :, ENTRY_CHANNEL] = entries + np.arange(N_STATIONS)
        values[step, :, EXIT_CHANNEL] = exits + np.arange(N_STATIONS)
    return values


def windows(grid: pd.DatetimeIndex, values: np.ndarray) -> dict[str, np.ndarray]:
    """Overlapping sliding windows, exactly as PVCGN ships them."""
    x, y, xt, yt = [], [], [], []
    for start in range(len(grid) - 2 * WINDOW + 1):
        mid = start + WINDOW
        x.append(values[start:mid])
        y.append(values[mid : mid + WINDOW])
        xt.append(grid[start:mid].to_numpy())
        yt.append(grid[mid : mid + WINDOW].to_numpy())
    return {
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "xtime": np.asarray(xt),
        "ytime": np.asarray(yt),
    }


@pytest.fixture
def grid():
    return service_grid()


@pytest.fixture
def values(grid):
    return ridership(grid)


@pytest.fixture
def raw(tmp_path, grid, values):
    """A raw PVCGN directory: unpacked Hangzhou pickles plus a provenance manifest."""
    directory = tmp_path / "pvcgn"
    city = directory / EXTRACTED / "hangzhou"
    city.mkdir(parents=True)

    payload = windows(grid, values)
    n = len(payload["x"])
    bounds = [(0, n - 4), (n - 4, n - 2), (n - 2, n)]
    for split, (lo, hi) in zip(("train", "val", "test"), bounds, strict=True):
        with (city / f"{split}.pkl").open("wb") as handle:
            pickle.dump({key: array[lo:hi] for key, array in payload.items()}, handle)

    # A line of stations, plus the non-zero diagonal the real matrix carries.
    adjacency = np.eye(N_STATIONS, dtype=int)
    for i in range(N_STATIONS - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    with (city / "graph_hz_conn.pkl").open("wb") as handle:
        pickle.dump(adjacency, handle)

    DownloadManifest(
        source="pvcgn",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[FileRecord(name="data.tar.gz", url="https://x/d.tgz", sha256="ab", bytes=1)],
        licence="not stated by the publisher",
        citation="Liu et al. (2020)",
    ).write(directory)
    return directory


@pytest.fixture
def site(raw, tmp_path):
    result = PvcgnAdapter(raw=raw, destination=tmp_path / "canonical").run()
    return load_site(result.sites[0])


# --------------------------------------------------------------------------- #
# Channel order
# --------------------------------------------------------------------------- #


def test_channel_one_is_entries_and_channel_zero_is_exits(site, grid, values) -> None:
    # The single fact the whole adapter turns on, and the one the dataset README gets
    # backwards. Entries belong on the edge running from outside into the station.
    flow = site.flow.set_index(["timestamp", "edge_id"])["count"]
    first = pd.Timestamp(grid[0], tz="Asia/Shanghai").tz_convert("UTC")
    assert flow[(first, "e_enter_station_000")] == values[0, 0, ENTRY_CHANNEL]
    assert flow[(first, "e_exit_station_000")] == values[0, 0, EXIT_CHANNEL]


def test_the_day_opens_with_entries_and_closes_with_exits(site) -> None:
    # The physical check the channel mapping was established from: nobody can alight
    # before anybody has boarded.
    flow = site.flow.dropna(subset=["count"])
    per_step = flow.assign(
        kind=np.where(flow["edge_id"].str.startswith("e_enter_"), "entries", "exits")
    ).pivot_table(index="timestamp", columns="kind", values="count", aggfunc="sum")
    opening = per_step.iloc[0]
    closing = per_step.iloc[len(per_step) // DAYS - 1]
    assert opening["entries"] > opening["exits"]
    assert closing["exits"] > closing["entries"]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_the_network_becomes_one_site(site) -> None:
    assert site.meta.site_id == "hzmetro"
    assert site.meta.source == "real"
    assert site.meta.interval_seconds == PVCGN_INTERVAL_SECONDS
    assert len(site.interior_nodes) == N_STATIONS
    assert len(site.edges) == 2 * N_STATIONS


def test_timestamps_are_converted_from_local_time(site, grid) -> None:
    assert site.occupancy["timestamp"].min() == pd.Timestamp(
        grid[0], tz="Asia/Shanghai"
    ).tz_convert("UTC")


def test_overlapping_windows_are_de_overlapped_consistently(site, grid, values) -> None:
    flow = site.flow.dropna(subset=["count"])
    # Each timestamp appears in many windows but must survive as one value per edge.
    assert len(flow) == len(grid) * 2 * N_STATIONS
    entries = flow[flow["edge_id"] == "e_enter_station_001"].sort_values("timestamp")
    assert np.allclose(entries["count"].to_numpy(), values[:, 1, ENTRY_CHANNEL])


def test_windows_that_disagree_stop_the_adapter(raw, tmp_path) -> None:
    path = raw / EXTRACTED / "hangzhou" / "train.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["x"][1, 0, 0, 0] += 999.0
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    # De-overlapping assumes the windows are slices of one series. If they are not, every
    # number built on the recovered series would be wrong.
    with pytest.raises(AdapterError, match="two windows disagree"):
        PvcgnAdapter(raw=raw, destination=tmp_path / "canonical").run()


# --------------------------------------------------------------------------- #
# Gaps
# --------------------------------------------------------------------------- #


def test_the_overnight_gap_becomes_missing_values(site) -> None:
    counts = site.flow.pivot(index="timestamp", columns="edge_id", values="count")
    assert len(counts) == len(
        pd.date_range(counts.index.min(), counts.index.max(), freq="15min", tz="UTC")
    )
    assert counts["e_enter_station_000"].isna().any()
    assert 0.0 < site.meta.provenance["missing_step_fraction"] < 1.0


# --------------------------------------------------------------------------- #
# What PVCGN does not have
# --------------------------------------------------------------------------- #


def test_occupancy_is_dense_and_entirely_missing(site) -> None:
    # Fare gates count crossings, not people standing in a station, and the running net
    # of entries minus exits goes negative at most real stations.
    assert site.meta.has_ground_truth_flow is False
    assert site.occupancy["count"].isna().all()
    assert "not derived" in site.meta.provenance["occupancy"].lower()


def test_an_unpublished_capacity_means_unbounded_not_guessed(site) -> None:
    assert site.nodes.loc[site.nodes["node_id"] != "outside", "capacity_persons"].isna().all()
    capacity = site.node_capacity()
    assert capacity["station_000"] == np.inf
    assert site.edge_capacity_per_interval()["e_enter_station_000"] == np.inf


def test_the_physical_adjacency_travels_in_the_provenance(site) -> None:
    # It is real topology and the graph baselines want it, but it carries no observed
    # flow -- that is the origin-destination matrix, which this release omits -- so it is
    # not added as edges whose flow would be permanently missing.
    pairs = site.meta.provenance["physical_adjacency"]
    assert ["station_000", "station_001"] in pairs
    assert len(pairs) == N_STATIONS - 1
    assert all(
        "outside" in {row.src_node, row.dst_node} for row in site.edges.itertuples()
    ), "every edge must be a fare gate; no station-to-station edge is measured"


def test_station_names_and_conservation_limits_are_recorded(site) -> None:
    provenance = site.meta.provenance
    assert "not published" in provenance["station_names"]
    assert "origin-destination" in provenance["conservation"]
    assert "README" in provenance["channel_order"]


# --------------------------------------------------------------------------- #
# Failure
# --------------------------------------------------------------------------- #


def test_an_unknown_city_is_refused(raw, tmp_path) -> None:
    with pytest.raises(AdapterError, match="unknown PVCGN cities"):
        PvcgnAdapter(raw=raw, destination=tmp_path / "canonical", cities=("kyoto",))


def test_unpacked_data_that_is_not_there_says_to_fetch_it(tmp_path) -> None:
    directory = tmp_path / "pvcgn"
    directory.mkdir()
    DownloadManifest(
        source="pvcgn",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[],
        licence="x",
        citation="y",
    ).write(directory)
    with pytest.raises(AdapterError, match=r"Run scripts/fetch_pvcgn\.py"):
        PvcgnAdapter(raw=directory, destination=tmp_path / "canonical").run()


def test_an_unexpected_array_layout_stops_the_adapter(raw, tmp_path) -> None:
    path = raw / EXTRACTED / "hangzhou" / "val.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["x"] = payload["x"][..., :1]
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(AdapterError, match="channel meaning has to be re-established"):
        PvcgnAdapter(raw=raw, destination=tmp_path / "canonical").run()


def test_an_adjacency_of_the_wrong_size_stops_the_adapter(raw, tmp_path) -> None:
    path = raw / EXTRACTED / "hangzhou" / "graph_hz_conn.pkl"
    with path.open("wb") as handle:
        pickle.dump(np.eye(N_STATIONS + 1, dtype=int), handle)
    with pytest.raises(AdapterError, match="adjacency matrix covers"):
        PvcgnAdapter(raw=raw, destination=tmp_path / "canonical").run()
