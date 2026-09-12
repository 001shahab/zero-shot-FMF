"""ROBOD: Room-level Occupancy and Building Operation Dataset.

Written against the published dataset, not against a remembered API. Everything below was
read from the source before any code was written:

* repository: https://github.com/ideas-lab-nus/robod, files ``Data/combined_Room{1..5}.csv``
* paper: Tekler, Z. D. et al. "ROBOD, room-level occupancy and building operation
  dataset." *Building Simulation* 16, 397-405 (2023). https://doi.org/10.1007/s12273-022-0925-9
* Figshare record 19234530.

Verified facts that this adapter depends on
-------------------------------------------
* Five rooms of the SDE4 building, National University of Singapore.
* ``timestamp`` is formatted ``YYYY-MM-DD HH:MM +08:00`` -- already offset-aware, so no
  local-time localisation is needed and the daylight-saving question does not arise
  (Singapore has no DST).
* Sampling interval is 300 s.
* ``occupant_count`` is ground truth from a ceiling camera; ``occupant_presence`` is its
  binary form.
* Units, from the dataset's own sensor table: ``indoor_co2`` ppm, ``air_temperature`` degC,
  ``indoor_relative_humidity`` %RH, ``illuminance`` lux, ``wifi_connected_devices`` count.
* Rooms 1 and 2 are FCU-served and carry 28 columns; rooms 3, 4 and 5 are AHU-served and
  carry 36. The eight extra columns are HVAC channels this adapter does not use.
* The record is **not contiguous**. Collection ran in weekday blocks with weekends and
  long vacation breaks absent: rooms 1-3 cover 29 dates and rooms 4-5 cover 47, over a
  span of 108 days from 2021-09-07 to 2021-12-23.

What this adapter does not have, and does not invent
----------------------------------------------------
* **No flow.** ROBOD counts people in rooms and nothing crossing a doorway. The canonical
  contract requires ``flow.parquet`` to be dense over the grid, so it is written dense and
  entirely missing, and ``has_ground_truth_flow`` is false. Deriving a flow from successive
  occupancy differences would manufacture exactly the quantity reconciliation is supposed
  to be evaluated on, and every E3 and E5 number computed over it would be circular.
* **No schedule.** The publication states no opening hours for SDE4, so ``opening_hours``
  is left empty and no ``is_open`` covariate is emitted. An invented timetable would make
  the calendar arm of the covariate ablation measure a fiction.
* **No room adjacency.** The five rooms sit on four different levels and the dataset
  documents no doorway between any pair of them. Each is therefore connected only to
  ``outside``, which is the minimal graph that closes and the only one the source
  supports.

The gaps are carried through as missing values rather than filled. Reconstruction is the
evaluation protocol's business, and it handles it explicitly: see the ``require_observed``
option of :func:`mflow.eval.protocol.build_plan`, which is what E7 sets.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from mflow.data.adapter import Adapter, AdapterError, long_covariates
from mflow.schema import SiteData, SiteMeta

#: Downloads for ``scripts/fetch_robod.py``, keyed by the filename written under
#: ``data/raw/robod/``.
ROBOD_FILES: Final[dict[str, str]] = {
    f"combined_Room{room}.csv": (
        f"https://raw.githubusercontent.com/ideas-lab-nus/robod/master/Data/combined_Room{room}.csv"
    )
    for room in range(1, 6)
}

ROBOD_CITATION: Final[str] = (
    "Tekler, Z. D., Low, R., Zhou, Y., Yuen, C., Blessing, L. and Spanos, C. "
    "ROBOD, room-level occupancy and building operation dataset. "
    "Building Simulation 16, 397-405 (2023). https://doi.org/10.1007/s12273-022-0925-9"
)

#: The repository states no licence file; the paper is open access under CC BY 4.0 and the
#: Figshare record is published under CC BY 4.0. Recorded as stated, not as assumed.
ROBOD_LICENCE: Final[str] = "CC BY 4.0 (as published on Figshare record 19234530)"

ROBOD_INTERVAL_SECONDS: Final[int] = 300
ROBOD_TIMEZONE: Final[str] = "Asia/Singapore"

#: As printed in the dataset's own room-description table. ``capacity_persons`` is the
#: published seating capacity, which bounds every observed count in the record.
ROOMS: Final[dict[str, dict[str, object]]] = {
    "room1": {
        "name": "Lecture room (level 4)",
        "kind": "gallery",
        "area_m2": 118.6,
        "capacity_persons": 40,
    },
    "room2": {
        "name": "Lecture room (level 4)",
        "kind": "gallery",
        "area_m2": 53.7,
        "capacity_persons": 40,
    },
    "room3": {
        "name": "Office, administrative staff (level 5)",
        "kind": "gallery",
        "area_m2": 98.4,
        "capacity_persons": 15,
    },
    "room4": {
        "name": "Office, researchers (level 3)",
        "kind": "gallery",
        "area_m2": 141.9,
        "capacity_persons": 25,
    },
    "room5": {
        "name": "Library (level 2)",
        "kind": "gallery",
        "area_m2": 182.8,
        "capacity_persons": 36,
    },
}

#: Room volumes in cubic metres, from the same table. Not part of the canonical node
#: schema, so they travel in the provenance, where the CO2 mass-balance work can find
#: them without re-deriving a number the publication already states.
ROOM_VOLUMES_M3: Final[dict[str, float]] = {
    "room1": 486.2,
    "room2": 220.2,
    "room3": 413.2,
    "room4": 581.7,
    "room5": 1363.3,
}

#: Per-room sensed channels, mapped onto the canonical covariate names.
ROOM_COVARIATES: Final[dict[str, str]] = {
    "indoor_co2": "co2_ppm",
    "air_temperature": "temp_c",
    "indoor_relative_humidity": "rh_pct",
    "illuminance": "lux",
    "wifi_connected_devices": "wifi_devices",
}

#: Outdoor weather, identical across the five files because one station served them all.
#: Scoped ``global`` and kept as past covariates: these are measurements, not forecasts,
#: and treating a measured outdoor temperature as known-future would leak.
WEATHER_COVARIATES: Final[dict[str, str]] = {
    "dry_bulb_temp": "outdoor_temp_c",
    "outdoor_relative_humidity": "outdoor_rh_pct",
    "outdoor_co2": "outdoor_co2_ppm",
    "global_horizontal_solar_radiation": "solar_wm2",
    "rainfall_raw": "rainfall_mm",
    "baromatic_pressure": "pressure_hpa",
}

_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M %z"

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"timestamp", "occupant_count", *ROOM_COVARIATES, *WEATHER_COVARIATES}
)


class RobodAdapter(Adapter):
    """Convert the five ROBOD room files into one canonical site."""

    source = "robod"
    version = "1"
    site_id = "robod_bldg1"

    def build(self) -> tuple[list[SiteData], dict[str, str]]:
        frames = {room: self._read(room) for room in ROOMS}

        # One grid for the whole building. Rooms 4 and 5 were instrumented for longer than
        # rooms 1-3, so the shorter rooms are missing over the extra weeks; that is a hole
        # in those channels, not a reason to throw the extra weeks away.
        grid = pd.date_range(
            min(frame.index.min() for frame in frames.values()),
            max(frame.index.max() for frame in frames.values()),
            freq=pd.Timedelta(seconds=ROBOD_INTERVAL_SECONDS),
            tz="UTC",
        )

        occupancy = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": grid,
                        "node_id": room,
                        "count": frame["occupant_count"].reindex(grid).to_numpy(dtype=float),
                    }
                )
                for room, frame in frames.items()
            ],
            ignore_index=True,
        ).sort_values(["timestamp", "node_id"], ignore_index=True)

        nodes, edges = self._graph()
        flow = pd.DataFrame(
            {
                "timestamp": np.repeat(grid.to_numpy(), len(edges)),
                "edge_id": np.tile(edges["edge_id"].to_numpy(), len(grid)),
                "count": np.nan,
            }
        ).sort_values(["timestamp", "edge_id"], ignore_index=True)

        observed = int(occupancy["count"].notna().sum())
        meta = SiteMeta(
            site_id=self.site_id,
            interval_seconds=ROBOD_INTERVAL_SECONDS,
            timezone=ROBOD_TIMEZONE,
            source="real",
            has_ground_truth_flow=False,
            opening_hours={},
            provenance={
                "building": "SDE4, National University of Singapore",
                "room_volumes_m3": ROOM_VOLUMES_M3,
                "occupancy_ground_truth": "ceiling camera person count, per the dataset",
                "flow": "not measured by ROBOD; written dense and missing, never derived",
                "future_covariates": (
                    "none; ROBOD publishes no schedule for SDE4 and an invented timetable "
                    "would make the calendar covariate ablation measure a fiction"
                ),
                "grid_steps": len(grid),
                "observed_occupancy_cells": observed,
                "missing_occupancy_fraction": round(1.0 - observed / (len(grid) * len(ROOMS)), 4),
            },
        )

        site = SiteData(
            meta=meta,
            nodes=nodes,
            edges=edges,
            occupancy=occupancy,
            flow=flow,
            covariates_past=self._covariates(frames, grid),
            # covariates_future is left at its empty default: see the module docstring on
            # why no schedule is invented for SDE4.
        )
        return [site], {}

    # ----------------------------------------------------------------- reading

    def _read(self, room: str) -> pd.DataFrame:
        """One room file, indexed by UTC timestamp and checked against the header."""
        path = self.raw / f"combined_Room{room.removeprefix('room')}.csv"
        if not path.is_file():
            raise AdapterError(f"{path} is missing; re-run scripts/fetch_robod.py")

        frame = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise AdapterError(
                f"{path.name} is missing column(s) {sorted(missing)}. The adapter was "
                "written against the 2022 release; if ROBOD has been re-published with a "
                "different header, the mapping has to be re-read from the new "
                "documentation rather than guessed."
            )

        stamps = pd.to_datetime(frame["timestamp"], format=_TIMESTAMP_FORMAT, utc=True)
        if stamps.duplicated().any():
            raise AdapterError(f"{path.name} has duplicate timestamps")
        return frame.drop(columns=["timestamp"]).set_index(pd.DatetimeIndex(stamps))

    # ------------------------------------------------------------------ graph

    def _graph(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Five rooms, each joined to ``outside`` and to nothing else."""
        nodes = pd.DataFrame(
            [
                {
                    "node_id": "outside",
                    "name": "Outside",
                    "kind": "outside",
                    "area_m2": 0.0,
                    "capacity_persons": 0,
                },
                *({"node_id": room, **attributes} for room, attributes in ROOMS.items()),
            ]
        )

        rows = []
        for room in ROOMS:
            # The dataset documents no doorway geometry, so the width is left at NaN and
            # the per-minute capacity is set from the room's own occupancy bound: the most
            # that can cross in a minute is at most the number the room can hold.
            capacity = float(ROOMS[room]["capacity_persons"])  # type: ignore[arg-type]
            rows.append(
                {
                    "edge_id": f"e_in_{room}",
                    "src_node": "outside",
                    "dst_node": room,
                    "width_m": np.nan,
                    "capacity_persons_per_min": capacity,
                    "reverse_edge_id": f"e_out_{room}",
                }
            )
            rows.append(
                {
                    "edge_id": f"e_out_{room}",
                    "src_node": room,
                    "dst_node": "outside",
                    "width_m": np.nan,
                    "capacity_persons_per_min": capacity,
                    "reverse_edge_id": f"e_in_{room}",
                }
            )
        return nodes, pd.DataFrame(rows)

    # ------------------------------------------------------------- covariates

    def _covariates(self, frames: dict[str, pd.DataFrame], grid: pd.DatetimeIndex) -> pd.DataFrame:
        """Per-room sensed channels plus one shared outdoor weather record."""
        parts = [
            long_covariates(
                frame.reindex(grid).reset_index(names="timestamp").assign(scope=room),
                ROOM_COVARIATES,
                scope_column="scope",
            )
            for room, frame in frames.items()
        ]

        # The weather columns are identical in all five files because one station served
        # the building. Room 5 is used because it has the longest record; where a shorter
        # room would have had a reading the longer one has it too.
        longest = max(frames.values(), key=len)
        parts.append(
            long_covariates(
                longest.reindex(grid).reset_index(names="timestamp").assign(scope="global"),
                WEATHER_COVARIATES,
                scope_column="scope",
            )
        )
        return pd.concat(parts, ignore_index=True).sort_values(
            ["timestamp", "scope", "variable"], ignore_index=True
        )
