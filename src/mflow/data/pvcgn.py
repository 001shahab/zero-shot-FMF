"""HZMetro and SHMetro, as released with PVCGN.

Written against the published release, not against a remembered format:

* repository: https://github.com/HCPLab-SYSU/PVCGN, archive ``data/data.tar.gz``
* paper: Liu, L., Chen, J., Wu, H., Zhen, J., Li, G. and Lin, L. "Physical-Virtual
  Collaboration Modeling for Intra- and Inter-Station Metro Ridership Prediction."
  *IEEE Transactions on Intelligent Transportation Systems* (2020).
  https://doi.org/10.1109/TITS.2020.3036057

Verified facts that this adapter depends on
-------------------------------------------
* ``{train,val,test}.pkl`` each hold ``x`` and ``y`` of shape ``(T, 4, N, 2)`` with
  matching ``xtime`` and ``ytime`` of shape ``(T, 4)``. The windows overlap, so the raw
  series is recovered by de-overlapping; this adapter checks that every repeated
  ``(timestamp, station)`` cell agrees before accepting it.
* ``N`` is 80 for Hangzhou and 288 for Shanghai. Timestamps are naive local time: Asia/
  Shanghai for both cities, which is also Hangzhou's zone.
* The interval is 900 s. Service runs 05:15-23:30, giving 73 labelled steps per day and
  **no data overnight**. A timestamp label denotes the *end* of its interval, so 05:30
  covers 05:15-05:30; that is stated in the dataset's own README.
* Values are non-negative integers stored as float32.
* ``graph_{hz,sh}_conn.pkl`` is the physical station adjacency: a symmetric integer
  matrix with a non-zero diagonal (each station is adjacent to itself).

The channel order is the opposite of what the README says
---------------------------------------------------------
The dataset README describes ``D = 2 (inflow/outflow)``, which reads as channel 0 being
entries. The data says otherwise, and the conservation constraint's sign depends on
getting this right, so it was checked rather than assumed. Over the whole Hangzhou
record:

* In the first interval of the service day, channel 0 sums to 29 across the network while
  channel 1 sums to 240. Nobody can alight before anybody has boarded, so channel 1 is
  entries.
* In the last interval, channel 0 sums to 1495 and channel 1 to 118 -- the last trains
  emptying out after entry has all but stopped. Same conclusion.
* Within every service day the running total of channel 1 leads that of channel 0, and
  the two converge to within 0.37% by the end of the day, which is what a network that
  fills each morning and empties each night looks like.

So **channel 0 is exits and channel 1 is entries**, and :data:`EXIT_CHANNEL` and
:data:`ENTRY_CHANNEL` say so at the one place the mapping is applied.

What this adapter does not have, and does not invent
----------------------------------------------------
* **No station occupancy.** Fare-gate counts say how many people crossed a boundary, not
  how many are standing in the station, and PVCGN publishes nothing else.
  ``occupancy.parquet`` is written dense and entirely missing and
  ``has_ground_truth_flow`` is false.

  The tempting derivation -- occupancy as the running total of entries minus exits since
  service began -- was tried and rejected on the evidence. On 76 of Hangzhou's 80
  stations it goes negative at some point in the record, reaching -26068 at the worst,
  because a commuter station discharges far more people in the morning than it takes in.
  A quantity that is routinely negative is not a count of people in a room, and feeding
  it to a reconciler whose whole premise is a non-negative bounded occupancy would
  produce coherence numbers about a quantity that does not exist.

  The consequence is that **per-station conservation is not measurable here**. Closing
  the identity at a station needs the flows between stations, which is the
  origin-destination matrix; this release does not include it. The specification listed
  HZMetro as "a closed graph with a conservation identity of exactly the assumed form",
  and that turns out not to be true of the published data. The runner refuses to
  reconcile a site with no measured occupancy, so this cannot be forgotten.

* **No station names.** The Hangzhou graph pickle is a bare adjacency matrix with no
  identifiers, so stations are numbered in the dataset's own column order.
* **No capacity.** Nothing in the release states how many people a platform holds, so
  ``capacity_persons`` is left blank, which the contract reads as an unbounded node.

What it is good for is real multivariate flow forecasting at a scale no museum dataset
reaches: 160 series for Hangzhou and 576 for Shanghai, on a real physical graph.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from mflow.data.adapter import Adapter, AdapterError
from mflow.schema import SiteData, SiteMeta

#: Index of the exit (tap-out) channel in the ``(T, n, N, 2)`` arrays. See the module
#: docstring: this is channel 0 despite the dataset README's wording.
EXIT_CHANNEL: Final[int] = 0

#: Index of the entry (tap-in) channel.
ENTRY_CHANNEL: Final[int] = 1

PVCGN_ARCHIVE_URL: Final[str] = (
    "https://raw.githubusercontent.com/HCPLab-SYSU/PVCGN/master/data/data.tar.gz"
)

PVCGN_CITATION: Final[str] = (
    "Liu, L., Chen, J., Wu, H., Zhen, J., Li, G. and Lin, L. Physical-Virtual "
    "Collaboration Modeling for Intra- and Inter-Station Metro Ridership Prediction. "
    "IEEE Transactions on Intelligent Transportation Systems (2020). "
    "https://doi.org/10.1109/TITS.2020.3036057"
)

#: The repository carries no licence file. Recorded as the absence it is, rather than as
#: an assumed permissive licence.
PVCGN_LICENCE: Final[str] = (
    "not stated by the publisher; the repository has no LICENSE file. Used here for "
    "research benchmarking with attribution to Liu et al. (2020)."
)

PVCGN_INTERVAL_SECONDS: Final[int] = 900
PVCGN_TIMEZONE: Final[str] = "Asia/Shanghai"

#: Directory inside the archive, the site it becomes, and its adjacency pickle.
CITIES: Final[dict[str, dict[str, str]]] = {
    "hangzhou": {"site_id": "hzmetro", "graph": "graph_hz_conn.pkl", "city": "Hangzhou"},
    "shanghai": {"site_id": "shmetro", "graph": "graph_sh_conn.pkl", "city": "Shanghai"},
}

#: Where the fetch script unpacks ``data.tar.gz`` inside ``data/raw/pvcgn/``.
EXTRACTED: Final[str] = "extracted"

_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


class PvcgnAdapter(Adapter):
    """Convert the PVCGN release into one canonical site per city."""

    source = "pvcgn"
    version = "1"

    def __init__(
        self,
        raw: Path | None = None,
        destination: Path | None = None,
        cities: tuple[str, ...] = ("hangzhou",),
    ) -> None:
        """
        Args:
            cities: which cities to convert. Shanghai is 288 stations and 576 series over
                five months; it is off by default because converting it is expensive and
                nothing in the experiment plan needs it yet.
        """
        super().__init__(raw=raw, destination=destination)
        unknown = set(cities) - set(CITIES)
        if unknown:
            raise AdapterError(f"unknown PVCGN cities {sorted(unknown)}; known {sorted(CITIES)}")
        self.cities = cities

    def build(self) -> tuple[list[SiteData], dict[str, str]]:
        sites = []
        skipped = {
            name: "not requested; pass cities=(...) to convert it"
            for name in CITIES
            if name not in self.cities
        }
        for city in self.cities:
            sites.append(self._site(city))
        return sites, skipped

    # ----------------------------------------------------------------- reading

    def _directory(self, city: str) -> Path:
        """The extracted city directory, wherever the archive was unpacked."""
        for candidate in (self.raw / EXTRACTED / city, self.raw / city):
            if candidate.is_dir():
                return candidate
        raise AdapterError(
            f"no {city!r} directory under {self.raw}. Run scripts/fetch_pvcgn.py, which "
            "downloads data.tar.gz and unpacks it."
        )

    def _series(self, city: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """De-overlap the sliding windows into one ``(T, N, 2)`` array.

        Raises:
            AdapterError: if two windows disagree about the same ``(timestamp, station)``
                cell. They should not -- they are slices of one underlying series -- and
                if they do, the de-overlapping assumption is wrong and every number built
                on it would be wrong with it.
        """
        directory = self._directory(city)
        cells: dict[pd.Timestamp, np.ndarray] = {}
        for split in _SPLITS:
            path = directory / f"{split}.pkl"
            if not path.is_file():
                raise AdapterError(f"{path} is missing; re-run scripts/fetch_pvcgn.py")
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            for values_key, time_key in (("x", "xtime"), ("y", "ytime")):
                values = np.asarray(payload[values_key])
                stamps = np.asarray(payload[time_key])
                if values.ndim != 4 or values.shape[-1] != 2:
                    raise AdapterError(
                        f"{path.name} {values_key!r} has shape {values.shape}; the adapter "
                        "expects (T, n, N, 2). If PVCGN has been re-released in another "
                        "layout, the channel meaning has to be re-established from the "
                        "new data rather than carried over."
                    )
                for window in range(values.shape[0]):
                    for step in range(values.shape[1]):
                        stamp = pd.Timestamp(stamps[window, step])
                        cell = values[window, step]
                        previous = cells.get(stamp)
                        if previous is not None and not np.array_equal(previous, cell):
                            raise AdapterError(
                                f"{city}: two windows disagree about {stamp}. The sliding "
                                "windows are supposed to be slices of one series, so "
                                "de-overlapping them is not safe here."
                            )
                        cells[stamp] = cell

        order = sorted(cells)
        index = pd.DatetimeIndex(order).tz_localize(PVCGN_TIMEZONE).tz_convert("UTC")
        return index, np.stack([cells[stamp] for stamp in order])

    def _adjacency(self, city: str) -> np.ndarray:
        """The physical station adjacency, as a boolean off-diagonal matrix."""
        path = self._directory(city) / CITIES[city]["graph"]
        if not path.is_file():
            raise AdapterError(f"{path} is missing; re-run scripts/fetch_pvcgn.py")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        # Shanghai's loader unpacks a triple; Hangzhou's is a bare matrix.
        matrix = np.asarray(payload[-1] if isinstance(payload, list | tuple) else payload)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise AdapterError(f"{path.name} is not a square adjacency matrix: {matrix.shape}")
        adjacency = matrix != 0
        np.fill_diagonal(adjacency, False)
        return adjacency

    # ------------------------------------------------------------------- site

    def _site(self, city: str) -> SiteData:
        stamps, values = self._series(city)
        n_stations = values.shape[1]
        stations = [f"station_{i:03d}" for i in range(n_stations)]
        adjacency = self._adjacency(city)
        if adjacency.shape[0] != n_stations:
            raise AdapterError(
                f"{city}: the adjacency matrix covers {adjacency.shape[0]} stations but "
                f"the ridership arrays cover {n_stations}"
            )

        grid = pd.date_range(
            stamps.min(),
            stamps.max(),
            freq=pd.Timedelta(seconds=PVCGN_INTERVAL_SECONDS),
            tz="UTC",
        )
        positions = grid.get_indexer(stamps)
        dense = np.full((len(grid), n_stations, 2), np.nan)
        dense[positions] = values

        nodes = self._nodes(stations)
        edges = self._edges(stations)
        flow = self._flow(grid, stations, dense)
        occupancy = pd.DataFrame(
            {
                "timestamp": np.repeat(grid.to_numpy(), n_stations),
                "node_id": np.tile(np.asarray(stations), len(grid)),
                "count": np.nan,
            }
        ).sort_values(["timestamp", "node_id"], ignore_index=True)

        meta = SiteMeta(
            site_id=CITIES[city]["site_id"],
            interval_seconds=PVCGN_INTERVAL_SECONDS,
            timezone=PVCGN_TIMEZONE,
            source="real",
            has_ground_truth_flow=False,
            opening_hours={},
            provenance=self._provenance(city, grid, stamps, adjacency, n_stations),
        )
        return SiteData(
            meta=meta, nodes=nodes, edges=edges, occupancy=occupancy, flow=flow
        )

    def _nodes(self, stations: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "node_id": "outside",
                    "name": "Street",
                    "kind": "outside",
                    "area_m2": 0.0,
                    "capacity_persons": 0,
                },
                *(
                    {
                        "node_id": station,
                        "name": f"Station {station.removeprefix('station_')}",
                        "kind": "gallery",
                        "area_m2": np.nan,
                        # PVCGN states no platform capacity. Blank means unbounded; a
                        # plausible-looking number would make a box constraint bind where
                        # the data never said it should.
                        "capacity_persons": np.nan,
                    }
                    for station in stations
                ),
            ]
        )

    def _edges(self, stations: list[str]) -> pd.DataFrame:
        """Two fare-gate edges per station.

        Only the boundary between the street and the station is measured. The physical
        adjacency between stations is real but carries no observed flow -- that is the
        origin-destination matrix, which this release does not include -- so it is
        recorded in the provenance for the graph baselines rather than added as edges
        whose flow would be permanently missing.
        """
        rows: list[dict[str, Any]] = []
        for station in stations:
            for edge_id, src, dst in (
                (f"e_enter_{station}", "outside", station),
                (f"e_exit_{station}", station, "outside"),
            ):
                rows.append(
                    {
                        "edge_id": edge_id,
                        "src_node": src,
                        "dst_node": dst,
                        "width_m": np.nan,
                        # No published gate throughput either, and blank says so.
                        "capacity_persons_per_min": np.nan,
                        "reverse_edge_id": (
                            f"e_exit_{station}" if src == "outside" else f"e_enter_{station}"
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _flow(
        self, grid: pd.DatetimeIndex, stations: list[str], dense: np.ndarray
    ) -> pd.DataFrame:
        """Entries and exits, in canonical long form."""
        frames = []
        for channel, prefix in ((ENTRY_CHANNEL, "e_enter_"), (EXIT_CHANNEL, "e_exit_")):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": np.repeat(grid.to_numpy(), len(stations)),
                        "edge_id": np.tile(
                            np.asarray([prefix + station for station in stations]), len(grid)
                        ),
                        "count": dense[:, :, channel].reshape(-1),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True).sort_values(
            ["timestamp", "edge_id"], ignore_index=True
        )

    def _provenance(
        self,
        city: str,
        grid: pd.DatetimeIndex,
        stamps: pd.DatetimeIndex,
        adjacency: np.ndarray,
        n_stations: int,
    ) -> dict[str, Any]:
        pairs = sorted(
            (f"station_{i:03d}", f"station_{j:03d}")
            for i, j in zip(*np.nonzero(np.triu(adjacency)), strict=True)
        )
        return {
            "city": CITIES[city]["city"],
            "channel_order": (
                "channel 0 is exits and channel 1 is entries, established from the data: "
                "channel 1 dominates the first interval of the service day and channel 0 "
                "the last, and the running totals converge by close of service. The "
                "dataset README's '(inflow/outflow)' wording implies the opposite order."
            ),
            "occupancy": (
                "not measured; fare gates count crossings, not people standing in a "
                "station. Not derived from the running net either: that quantity goes "
                "negative at most stations, so it is not an occupancy."
            ),
            "conservation": (
                "per-station conservation is not measurable without the "
                "origin-destination matrix, which this release does not include"
            ),
            "station_names": "not published; stations are numbered in dataset column order",
            "capacity_persons": "not published; left blank, read as unbounded",
            "n_stations": n_stations,
            "physical_adjacency": [list(pair) for pair in pairs],
            "grid_steps": len(grid),
            "observed_steps": len(stamps),
            "missing_step_fraction": round(1.0 - len(stamps) / len(grid), 4),
            "service_window_local": "05:15-23:30; no data overnight",
        }
