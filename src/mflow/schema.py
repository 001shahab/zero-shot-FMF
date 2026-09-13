"""Canonical data contract (M1).

Everything in this project -- simulated or real -- is converted to the on-disk layout
described here, and every forecaster sees exactly one thing: a :class:`Panel`.

On-disk layout of a canonical site::

    data/canonical/<site_id>/
        meta.json
        nodes.csv
        edges.csv
        occupancy.parquet
        flow.parquet
        covariates_past.parquet
        covariates_future.parquet

Series identifiers are ``occ:<node_id>`` for node occupancy and ``flow:<edge_id>`` for
directed edge crossings. The canonical ordering of a site's series is all occupancy
series in ``nodes.csv`` order followed by all flow series in ``edges.csv`` order; the
reconciliation operators in :mod:`mflow.reconcile` depend on that ordering, so it is
produced in exactly one place (:meth:`SiteData.series_ids`) and nowhere else.

The virtual ``outside`` node absorbs every entry and exit so the graph is closed.
Without it conservation cannot hold at the boundary. ``outside`` carries no occupancy
series: its occupancy is unbounded and meaningless, and it is excluded from the
conservation constraints.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NodeKind = Literal["gallery", "corridor", "stair", "entrance", "exit", "outside"]
NODE_KINDS: Final[frozenset[str]] = frozenset(
    {"gallery", "corridor", "stair", "entrance", "exit", "outside"}
)

OUTSIDE_NODE: Final[str] = "outside"
GLOBAL_SCOPE: Final[str] = "global"

OCC_PREFIX: Final[str] = "occ:"
FLOW_PREFIX: Final[str] = "flow:"

META_FILE: Final[str] = "meta.json"
NODES_FILE: Final[str] = "nodes.csv"
EDGES_FILE: Final[str] = "edges.csv"
OCCUPANCY_FILE: Final[str] = "occupancy.parquet"
FLOW_FILE: Final[str] = "flow.parquet"
COVARIATES_PAST_FILE: Final[str] = "covariates_past.parquet"
COVARIATES_FUTURE_FILE: Final[str] = "covariates_future.parquet"

REQUIRED_FILES: Final[tuple[str, ...]] = (
    META_FILE,
    NODES_FILE,
    EDGES_FILE,
    OCCUPANCY_FILE,
    FLOW_FILE,
)

NODE_COLUMNS: Final[tuple[str, ...]] = (
    "node_id",
    "name",
    "kind",
    "area_m2",
    "capacity_persons",
)
EDGE_COLUMNS: Final[tuple[str, ...]] = (
    "edge_id",
    "src_node",
    "dst_node",
    "width_m",
    "capacity_persons_per_min",
    "reverse_edge_id",
)

#: Variables permitted in ``covariates_future.parquet``. Anything whose value is not
#: known at forecast time belongs in ``covariates_past.parquet`` instead; mixing the two
#: is the most common way to leak the future into a zero-shot evaluation.
FUTURE_COVARIATE_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "is_open",
        "timed_slot_admissions",
        "tour_departure",
        "group_size_booked",
        "is_holiday",
        "weather_precip_mm",
        "weather_temp_c",
        "exhibition_id",
    }
)

#: Variables expected in ``covariates_past.parquet``. Not enforced as a closed set --
#: adapters for real datasets legitimately carry extra channels -- but the known names
#: are listed so that typos surface in review.
KNOWN_PAST_COVARIATE_VARIABLES: Final[frozenset[str]] = frozenset(
    {"co2_ppm", "temp_c", "rh_pct", "lux", "vibration_rms", "wifi_devices"}
)

DAY_KEYS: Final[tuple[str, ...]] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#: Absolute tolerance, in persons, for the per-node conservation residual of a site that
#: claims ``has_ground_truth_flow``. Raw simulator output is exact; adapters that derive
#: flow from counted crossings are allowed a small slack.
DEFAULT_CONSERVATION_TOLERANCE: Final[float] = 1e-6


class SiteValidationError(ValueError):
    """Raised when a canonical site violates the data contract.

    The message always names the specific violation and, where the violation is
    row-level, a bounded sample of offending rows. Silent repair is never attempted:
    a site either satisfies the contract or the pipeline stops.
    """


def occ_series_id(node_id: str) -> str:
    """Return the canonical series identifier for a node's occupancy series."""
    return f"{OCC_PREFIX}{node_id}"


def flow_series_id(edge_id: str) -> str:
    """Return the canonical series identifier for an edge's flow series."""
    return f"{FLOW_PREFIX}{edge_id}"


def parse_series_id(series_id: str) -> tuple[Literal["occupancy", "flow"], str]:
    """Split a canonical series identifier into its kind and the entity it refers to.

    Raises:
        ValueError: if ``series_id`` carries neither the occupancy nor the flow prefix.
    """
    if series_id.startswith(OCC_PREFIX):
        return "occupancy", series_id[len(OCC_PREFIX) :]
    if series_id.startswith(FLOW_PREFIX):
        return "flow", series_id[len(FLOW_PREFIX) :]
    raise ValueError(
        f"series id {series_id!r} has neither the {OCC_PREFIX!r} nor the {FLOW_PREFIX!r} prefix"
    )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


class SiteMeta(BaseModel):
    """Contents of ``meta.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    site_id: str = Field(min_length=1)
    interval_seconds: int = Field(gt=0)
    timezone: str
    source: Literal["simulated", "real"]
    has_ground_truth_flow: bool
    opening_hours: dict[str, list[str] | None] = Field(default_factory=dict)
    #: Free-form provenance recorded by adapters: download date, checksum, licence.
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {value!r}") from exc
        return value

    @field_validator("opening_hours")
    @classmethod
    def _wellformed_hours(cls, value: dict[str, list[str] | None]) -> dict[str, list[str] | None]:
        for day, hours in value.items():
            if day not in DAY_KEYS:
                raise ValueError(f"opening_hours key {day!r} is not one of {DAY_KEYS}")
            if hours is None:
                continue
            if len(hours) != 2:
                raise ValueError(f"opening_hours[{day!r}] must be null or [open, close]")
            for stamp in hours:
                hh, _, mm = stamp.partition(":")
                if not (
                    hh.isdigit() and mm.isdigit() and 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59
                ):
                    raise ValueError(f"opening_hours[{day!r}] entry {stamp!r} is not HH:MM")
        return value

    @property
    def interval(self) -> pd.Timedelta:
        """Sampling interval as a pandas offset."""
        return pd.Timedelta(seconds=self.interval_seconds)


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Panel:
    """The only view of the data a forecaster ever receives.

    Attributes:
        series: ``(n_series, T)`` float32, NaN where the observation is missing.
        series_ids: canonical identifiers, aligned with the rows of ``series``.
        timestamps: ``T`` timezone-aware UTC timestamps on a regular grid.
        past_covariates: ``(n_past_cov, T)`` or None.
        past_covariate_ids: identifiers of the form ``<scope>|<variable>``.
        future_covariates: ``(n_fut_cov, T + H)`` or None -- the trailing ``H`` columns
            are the known-future values for the forecast window.
        future_covariate_ids: identifiers of the form ``<scope>|<variable>``.
        interval_seconds: sampling interval.
        site_id: the site these series belong to.
    """

    series: np.ndarray
    series_ids: list[str]
    timestamps: pd.DatetimeIndex
    past_covariates: np.ndarray | None
    past_covariate_ids: list[str]
    future_covariates: np.ndarray | None
    future_covariate_ids: list[str]
    interval_seconds: int
    site_id: str

    def __post_init__(self) -> None:
        if self.series.ndim != 2:
            raise ValueError(f"Panel.series must be 2-D (n_series, T), got {self.series.shape}")
        if self.series.dtype != np.float32:
            raise ValueError(f"Panel.series must be float32, got {self.series.dtype}")
        n_series, n_time = self.series.shape
        if len(self.series_ids) != n_series:
            raise ValueError(
                f"Panel has {n_series} series rows but {len(self.series_ids)} series ids"
            )
        if len(set(self.series_ids)) != n_series:
            raise ValueError("Panel.series_ids contains duplicates")
        if len(self.timestamps) != n_time:
            raise ValueError(
                f"Panel has {n_time} time columns but {len(self.timestamps)} timestamps"
            )
        if self.timestamps.tz is None:
            raise ValueError("Panel.timestamps must be timezone-aware (UTC)")
        self._check_covariates(
            self.past_covariates, self.past_covariate_ids, n_time, "past_covariates"
        )
        if self.future_covariates is not None:
            if self.future_covariates.shape[1] < n_time:
                raise ValueError(
                    "Panel.future_covariates must span at least the context window: "
                    f"{self.future_covariates.shape[1]} columns < T={n_time}"
                )
            self._check_covariates(
                self.future_covariates,
                self.future_covariate_ids,
                self.future_covariates.shape[1],
                "future_covariates",
            )
        elif self.future_covariate_ids:
            raise ValueError("Panel.future_covariate_ids is non-empty but the array is None")

    @staticmethod
    def _check_covariates(
        array: np.ndarray | None, ids: Sequence[str], n_cols: int, label: str
    ) -> None:
        if array is None:
            if ids:
                raise ValueError(f"Panel.{label}_ids is non-empty but the array is None")
            return
        if array.ndim != 2:
            raise ValueError(f"Panel.{label} must be 2-D, got {array.shape}")
        if array.shape[0] != len(ids):
            raise ValueError(
                f"Panel.{label} has {array.shape[0]} rows but {len(ids)} ids were given"
            )
        if array.shape[1] != n_cols:
            raise ValueError(
                f"Panel.{label} must have {n_cols} columns, got {array.shape[1]}"
            )

    # -- convenience ------------------------------------------------------- #

    @property
    def n_series(self) -> int:
        """Number of series in the panel."""
        return self.series.shape[0]

    @property
    def n_timesteps(self) -> int:
        """Length of the context window."""
        return self.series.shape[1]

    @property
    def horizon_covered(self) -> int:
        """Number of future steps covered by ``future_covariates`` (0 if there are none)."""
        if self.future_covariates is None:
            return 0
        return int(self.future_covariates.shape[1] - self.n_timesteps)

    def index_of(self, series_id: str) -> int:
        """Row index of ``series_id``.

        Raises:
            KeyError: if the panel does not carry that series.
        """
        try:
            return self.series_ids.index(series_id)
        except ValueError as exc:
            raise KeyError(f"panel for site {self.site_id!r} has no series {series_id!r}") from exc

    def slice_time(self, start: int, stop: int, horizon: int = 0) -> Panel:
        """Return the sub-panel covering ``[start, stop)`` plus ``horizon`` future covariates.

        This is the single place where a forecast context is carved out of a longer
        history, so that no experiment can accidentally hand a model observations from
        beyond its origin.
        """
        if not 0 <= start < stop <= self.n_timesteps:
            raise ValueError(
                f"invalid slice [{start}, {stop}) for a panel of length {self.n_timesteps}"
            )
        future = None
        if self.future_covariates is not None:
            end = stop + horizon
            if end > self.future_covariates.shape[1]:
                raise ValueError(
                    f"future covariates cover {self.future_covariates.shape[1]} columns, "
                    f"need {end} for a context ending at {stop} with horizon {horizon}"
                )
            future = self.future_covariates[:, start:end]
        return Panel(
            series=self.series[:, start:stop],
            series_ids=list(self.series_ids),
            timestamps=self.timestamps[start:stop],
            past_covariates=(
                None if self.past_covariates is None else self.past_covariates[:, start:stop]
            ),
            past_covariate_ids=list(self.past_covariate_ids),
            future_covariates=future,
            future_covariate_ids=list(self.future_covariate_ids),
            interval_seconds=self.interval_seconds,
            site_id=self.site_id,
        )

    def select_covariates(
        self,
        *,
        keep: Callable[[str], bool] | None = None,
    ) -> Panel:
        """Return a panel carrying only the covariates ``keep`` accepts.

        The covariate ablation works by removing channels before the panel reaches the
        forecaster, rather than by asking each wrapper to ignore some of them. A wrapper
        that was told to ignore a channel could still normalise against it; a channel
        that is not in the panel cannot be used at all.

        Args:
            keep: predicate on the ``<scope>|<variable>`` identifier. ``None`` keeps
                everything, which makes the no-ablation case go down the same code path
                as every other one.
        """
        if keep is None:
            return self
        past_rows = [i for i, cid in enumerate(self.past_covariate_ids) if keep(cid)]
        future_rows = [i for i, cid in enumerate(self.future_covariate_ids) if keep(cid)]
        return Panel(
            series=self.series,
            series_ids=list(self.series_ids),
            timestamps=self.timestamps,
            past_covariates=(
                self.past_covariates[past_rows, :]
                if self.past_covariates is not None and past_rows
                else None
            ),
            past_covariate_ids=[self.past_covariate_ids[i] for i in past_rows],
            future_covariates=(
                self.future_covariates[future_rows, :]
                if self.future_covariates is not None and future_rows
                else None
            ),
            future_covariate_ids=[self.future_covariate_ids[i] for i in future_rows],
            interval_seconds=self.interval_seconds,
            site_id=self.site_id,
        )

    def subset_series(self, series_ids: Sequence[str]) -> Panel:
        """Return a panel restricted to ``series_ids``, preserving the given order."""
        rows = [self.index_of(sid) for sid in series_ids]
        return Panel(
            series=self.series[rows, :],
            series_ids=list(series_ids),
            timestamps=self.timestamps,
            past_covariates=self.past_covariates,
            past_covariate_ids=list(self.past_covariate_ids),
            future_covariates=self.future_covariates,
            future_covariate_ids=list(self.future_covariate_ids),
            interval_seconds=self.interval_seconds,
            site_id=self.site_id,
        )


# --------------------------------------------------------------------------- #
# Site container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SiteData:
    """A fully loaded canonical site."""

    meta: SiteMeta
    nodes: pd.DataFrame
    edges: pd.DataFrame
    occupancy: pd.DataFrame
    flow: pd.DataFrame
    covariates_past: pd.DataFrame = field(default_factory=lambda: _empty_covariates())
    covariates_future: pd.DataFrame = field(default_factory=lambda: _empty_covariates())

    # -- derived views ------------------------------------------------------ #

    @property
    def interior_nodes(self) -> list[str]:
        """Node ids excluding the virtual ``outside`` node, in ``nodes.csv`` order."""
        return [
            str(n)
            for n, k in zip(self.nodes["node_id"], self.nodes["kind"], strict=True)
            if k != OUTSIDE_NODE
        ]

    @property
    def edge_ids(self) -> list[str]:
        """Edge ids in ``edges.csv`` order."""
        return [str(e) for e in self.edges["edge_id"]]

    @property
    def series_ids(self) -> list[str]:
        """Canonical series ordering: occupancy of interior nodes, then edge flows."""
        return [occ_series_id(n) for n in self.interior_nodes] + [
            flow_series_id(e) for e in self.edge_ids
        ]

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """The regular UTC time grid shared by every series in the site."""
        stamps = pd.DatetimeIndex(sorted(set(self.occupancy["timestamp"].unique())))
        return stamps

    def node_capacity(self) -> dict[str, float]:
        """Map node id to capacity in persons, with an unknown capacity as infinity.

        A real source may simply not publish a capacity: PVCGN gives metro ridership
        without any statement of how many people a station platform holds. A blank
        ``capacity_persons`` records that, and the correct upper bound for an unknown
        capacity is ``inf`` rather than a plausible-looking guess, which would make the
        box constraint bind somewhere the data never said it should.
        """
        return {
            str(n): math.inf if pd.isna(c) else float(c)
            for n, c in zip(self.nodes["node_id"], self.nodes["capacity_persons"], strict=True)
        }

    def edge_capacity_per_interval(self) -> dict[str, float]:
        """Map edge id to the maximum crossings in one interval, unknown meaning infinity.

        As with :meth:`node_capacity`: a source that does not publish a doorway or
        fare-gate throughput leaves the column blank, and an unknown bound is infinite.
        """
        minutes = self.meta.interval_seconds / 60.0
        return {
            str(e): math.inf if pd.isna(c) else float(c) * minutes
            for e, c in zip(
                self.edges["edge_id"], self.edges["capacity_persons_per_min"], strict=True
            )
        }

    # -- wide matrices ------------------------------------------------------ #

    def wide_series(self) -> pd.DataFrame:
        """Return a ``(T, n_series)`` frame in canonical series order, NaN where missing."""
        occ = self.occupancy.pivot(index="timestamp", columns="node_id", values="count")
        occ.columns = [occ_series_id(str(c)) for c in occ.columns]
        flw = self.flow.pivot(index="timestamp", columns="edge_id", values="count")
        flw.columns = [flow_series_id(str(c)) for c in flw.columns]
        wide = occ.join(flw, how="outer")
        return wide.reindex(columns=self.series_ids).sort_index()

    def wide_covariates(self, which: Literal["past", "future"]) -> pd.DataFrame:
        """Return a ``(T, n_cov)`` frame of covariates keyed by ``<scope>|<variable>``."""
        frame = self.covariates_past if which == "past" else self.covariates_future
        if frame.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        keyed = frame.assign(key=frame["scope"].astype(str) + "|" + frame["variable"].astype(str))
        wide = keyed.pivot(index="timestamp", columns="key", values="value")
        return wide.sort_index().sort_index(axis=1)

    def to_panel(self, horizon: int = 0) -> Panel:
        """Materialise the whole site as a :class:`Panel`.

        Args:
            horizon: number of future steps of known-future covariates to append beyond
                the observed window. Future covariate rows must already exist on disk for
                those timestamps; missing rows are an error rather than a zero fill.
        """
        wide = self.wide_series()
        timestamps = pd.DatetimeIndex(wide.index)
        past_wide = self.wide_covariates("past").reindex(timestamps)
        fut_index = timestamps
        if horizon > 0:
            step = self.meta.interval
            extra = pd.date_range(
                timestamps[-1] + step, periods=horizon, freq=step, tz=timestamps.tz
            )
            fut_index = pd.DatetimeIndex(timestamps.append(extra))
        future_wide = self.wide_covariates("future")
        if not future_wide.empty:
            missing = fut_index.difference(pd.DatetimeIndex(future_wide.index))
            if len(missing) > 0:
                raise SiteValidationError(
                    f"site {self.meta.site_id!r}: future covariates are missing "
                    f"{len(missing)} timestamps required for horizon {horizon}, "
                    f"first missing {missing[0]}"
                )
            future_wide = future_wide.reindex(fut_index)

        return Panel(
            series=wide.to_numpy(dtype=np.float32).T,
            series_ids=list(wide.columns),
            timestamps=timestamps,
            past_covariates=(
                None if past_wide.empty else past_wide.to_numpy(dtype=np.float32).T
            ),
            past_covariate_ids=[] if past_wide.empty else [str(c) for c in past_wide.columns],
            future_covariates=(
                None if future_wide.empty else future_wide.to_numpy(dtype=np.float32).T
            ),
            future_covariate_ids=(
                [] if future_wide.empty else [str(c) for c in future_wide.columns]
            ),
            interval_seconds=self.meta.interval_seconds,
            site_id=self.meta.site_id,
        )


def _empty_covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([], tz="UTC"),
            "scope": pd.Series([], dtype="object"),
            "variable": pd.Series([], dtype="object"),
            "value": pd.Series([], dtype="float64"),
        }
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_site(path: str | Path, *, validate: bool = True) -> SiteData:
    """Load a canonical site from disk.

    Args:
        path: directory containing ``meta.json`` and friends.
        validate: run :func:`validate_site` on the loaded site. Only turn this off inside
            the validator itself or in tests that deliberately construct broken sites.

    Raises:
        SiteValidationError: if required files are missing or, when ``validate`` is set,
            if the site violates the data contract.
    """
    root = Path(path)
    if not root.is_dir():
        raise SiteValidationError(f"canonical site directory does not exist: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise SiteValidationError(f"site {root} is missing required files: {sorted(missing)}")

    meta = SiteMeta.model_validate_json((root / META_FILE).read_text(encoding="utf-8"))
    nodes = pd.read_csv(root / NODES_FILE, dtype={"node_id": str, "name": str, "kind": str})
    edges = pd.read_csv(
        root / EDGES_FILE,
        dtype={
            "edge_id": str,
            "src_node": str,
            "dst_node": str,
            "reverse_edge_id": str,
        },
    )
    _require_columns(nodes, NODE_COLUMNS, NODES_FILE, root)
    _require_columns(edges, EDGE_COLUMNS, EDGES_FILE, root)

    occupancy = _read_counts(root / OCCUPANCY_FILE, "node_id")
    flow = _read_counts(root / FLOW_FILE, "edge_id")
    covariates_past = _read_covariates(root / COVARIATES_PAST_FILE)
    covariates_future = _read_covariates(root / COVARIATES_FUTURE_FILE)

    site = SiteData(
        meta=meta,
        nodes=nodes,
        edges=edges,
        occupancy=occupancy,
        flow=flow,
        covariates_past=covariates_past,
        covariates_future=covariates_future,
    )
    if validate:
        validate_site(root, site=site)
    return site


def write_site(site: SiteData, path: str | Path, *, validate: bool = True) -> Path:
    """Write a site to disk in the canonical layout.

    Every producer -- the simulator, the sensor degradation stage and each adapter --
    goes through this function, so the on-disk format has exactly one implementation.

    Args:
        site: the site to write.
        path: destination directory, created if needed.
        validate: validate after writing. Leave this on except when deliberately
            materialising a broken site for a test.

    Returns:
        The directory written to.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / META_FILE).write_text(site.meta.model_dump_json(indent=2), encoding="utf-8")
    site.nodes[list(NODE_COLUMNS)].to_csv(root / NODES_FILE, index=False)
    site.edges[list(EDGE_COLUMNS)].to_csv(root / EDGES_FILE, index=False)

    # Counts are nullable integers on disk. People are countable, so a float column would
    # invite silently fractional occupancy, but a sensor that drops out has no reading at
    # all and that gap has to survive the round trip rather than become a zero.
    occ = site.occupancy[["timestamp", "node_id", "count"]].copy()
    occ["count"] = occ["count"].astype("Int32")
    occ.sort_values(["timestamp", "node_id"]).reset_index(drop=True).to_parquet(
        root / OCCUPANCY_FILE, index=False
    )

    flw = site.flow[["timestamp", "edge_id", "count"]].copy()
    flw["count"] = flw["count"].astype("Int32")
    flw.sort_values(["timestamp", "edge_id"]).reset_index(drop=True).to_parquet(
        root / FLOW_FILE, index=False
    )

    for frame, filename in (
        (site.covariates_past, COVARIATES_PAST_FILE),
        (site.covariates_future, COVARIATES_FUTURE_FILE),
    ):
        frame[["timestamp", "scope", "variable", "value"]].sort_values(
            ["timestamp", "scope", "variable"]
        ).reset_index(drop=True).to_parquet(root / filename, index=False)

    if validate:
        validate_site(root)
    return root


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], filename: str, root: Path
) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SiteValidationError(f"{root / filename} is missing columns {missing}")


def _read_counts(path: Path, key: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ("timestamp", key, "count"):
        if column not in frame.columns:
            raise SiteValidationError(f"{path} is missing column {column!r}")
    frame = frame[["timestamp", key, "count"]].copy()
    frame[key] = frame[key].astype(str)
    # Counts are float in memory whatever their on-disk dtype, so that a dropped reading
    # is a NaN every consumer already knows how to see rather than a pandas NA that only
    # some of them do.
    frame["count"] = frame["count"].astype("float64")
    frame["timestamp"] = _as_utc(frame["timestamp"], path)
    return frame


def _read_covariates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return _empty_covariates()
    frame = pd.read_parquet(path)
    for column in ("timestamp", "scope", "variable", "value"):
        if column not in frame.columns:
            raise SiteValidationError(f"{path} is missing column {column!r}")
    frame = frame[["timestamp", "scope", "variable", "value"]].copy()
    frame["scope"] = frame["scope"].astype(str)
    frame["variable"] = frame["variable"].astype(str)
    frame["value"] = frame["value"].astype("float64")
    frame["timestamp"] = _as_utc(frame["timestamp"], path)
    return frame


def _as_utc(column: pd.Series, path: Path) -> pd.Series:
    """Require timezone-aware timestamps and normalise them to UTC."""
    values = pd.to_datetime(column)
    if values.dt.tz is None:
        raise SiteValidationError(
            f"{path} has timezone-naive timestamps; the contract requires tz-aware UTC"
        )
    return values.dt.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_site(
    path: str | Path,
    *,
    site: SiteData | None = None,
    conservation_tolerance: float = DEFAULT_CONSERVATION_TOLERANCE,
) -> SiteData:
    """Validate a canonical site and return it.

    Checks, in order, and failing loudly on the first violation:

    1. structural integrity of ``nodes.csv`` and ``edges.csv`` (kinds, exactly one
       ``outside`` node, unique ids, non-negative capacities);
    2. referential integrity (edges, occupancy rows, flow rows and covariate scopes all
       reference known entities, and reverse edges are mutually consistent);
    3. duplicate ``(timestamp, entity)`` rows;
    4. a complete regular time grid with no gaps, shared by occupancy and flow;
    5. non-negative counts and occupancy at or below node capacity;
    6. conservation of persons at every interior node, where the site claims ground
       truth flow.

    Args:
        path: the site directory, used for error messages and to load when ``site`` is None.
        site: an already-loaded site, to avoid a second read from disk.
        conservation_tolerance: absolute tolerance in persons for the flow balance.

    Returns:
        The validated site.

    Raises:
        SiteValidationError: on the first violation found.
    """
    root = Path(path)
    data = site if site is not None else load_site(root, validate=False)

    _validate_nodes(data, root)
    _validate_edges(data, root)
    _validate_references(data, root)
    _validate_duplicates(data, root)
    grid = _validate_time_grid(data, root)
    _validate_values(data, root)
    _validate_covariates(data, root, grid)
    if data.meta.has_ground_truth_flow:
        _validate_conservation(data, root, conservation_tolerance)
    return data


def _fail(root: Path, message: str) -> None:
    raise SiteValidationError(f"site {root.name!r} ({root}): {message}")


def _sample(values: Iterable[Any], limit: int = 5) -> list[Any]:
    out = sorted({str(v) for v in values})
    return out[:limit]


def _validate_nodes(data: SiteData, root: Path) -> None:
    nodes = data.nodes
    if nodes.empty:
        _fail(root, "nodes.csv is empty")
    duplicated = nodes.loc[nodes["node_id"].duplicated(), "node_id"]
    if len(duplicated) > 0:
        _fail(root, f"duplicate node ids in nodes.csv: {_sample(duplicated)}")
    bad_kind = nodes.loc[~nodes["kind"].isin(NODE_KINDS), "kind"]
    if len(bad_kind) > 0:
        _fail(root, f"unknown node kinds {_sample(bad_kind)}; allowed {sorted(NODE_KINDS)}")
    n_outside = int((nodes["kind"] == OUTSIDE_NODE).sum())
    if n_outside != 1:
        _fail(
            root,
            f"the graph must be closed by exactly one node of kind {OUTSIDE_NODE!r}, found "
            f"{n_outside}",
        )
    for column in ("area_m2", "capacity_persons"):
        negative = nodes.loc[pd.to_numeric(nodes[column], errors="coerce") < 0, "node_id"]
        if len(negative) > 0:
            _fail(root, f"negative {column} for nodes {_sample(negative)}")
    # A blank capacity is permitted and means the source publishes none -- PVCGN gives
    # metro ridership without saying how many people a platform holds. It is read as an
    # unbounded node. A capacity of zero is a different claim, namely that nobody fits,
    # and for an interior node that is always a mistake.
    interior = nodes[nodes["kind"] != OUTSIDE_NODE]
    capacity = pd.to_numeric(interior["capacity_persons"], errors="coerce")
    zero_cap = interior.loc[capacity.notna() & (capacity <= 0), "node_id"]
    if len(zero_cap) > 0:
        _fail(
            root,
            f"interior nodes must have positive capacity_persons, or none at all if the "
            f"source does not publish one: {_sample(zero_cap)}",
        )


def _validate_edges(data: SiteData, root: Path) -> None:
    edges = data.edges
    if edges.empty:
        _fail(root, "edges.csv is empty")
    duplicated = edges.loc[edges["edge_id"].duplicated(), "edge_id"]
    if len(duplicated) > 0:
        _fail(root, f"duplicate edge ids in edges.csv: {_sample(duplicated)}")
    known_nodes = set(data.nodes["node_id"].astype(str))
    for column in ("src_node", "dst_node"):
        unknown = edges.loc[~edges[column].isin(known_nodes), column]
        if len(unknown) > 0:
            _fail(root, f"edges.csv {column} references unknown nodes {_sample(unknown)}")
    self_loops = edges.loc[edges["src_node"] == edges["dst_node"], "edge_id"]
    if len(self_loops) > 0:
        _fail(root, f"self-loop edges are not part of the contract: {_sample(self_loops)}")
    # Blank means the source publishes no throughput for this doorway, read as unbounded.
    # Zero is the different and always mistaken claim that nobody can cross it.
    throughput = pd.to_numeric(edges["capacity_persons_per_min"], errors="coerce")
    nonpositive = edges.loc[throughput.notna() & (throughput <= 0), "edge_id"]
    if len(nonpositive) > 0:
        _fail(
            root,
            "edges must have positive capacity_persons_per_min, or none at all if the "
            f"source does not publish one: {_sample(nonpositive)}",
        )

    known_edges = set(edges["edge_id"].astype(str))
    reverse = {
        str(edge_id): ("" if pd.isna(rev) else str(rev))
        for edge_id, rev in zip(edges["edge_id"], edges["reverse_edge_id"], strict=True)
    }
    for edge_id, rev in reverse.items():
        if rev == "":
            continue
        if rev not in known_edges:
            _fail(root, f"edge {edge_id!r} has reverse_edge_id {rev!r} which does not exist")
        if reverse[rev] != edge_id:
            _fail(
                root,
                f"reverse_edge_id is not symmetric: {edge_id!r} -> {rev!r} -> {reverse[rev]!r}",
            )
        src = edges.loc[edges["edge_id"] == edge_id, "src_node"].iloc[0]
        dst = edges.loc[edges["edge_id"] == edge_id, "dst_node"].iloc[0]
        rsrc = edges.loc[edges["edge_id"] == rev, "src_node"].iloc[0]
        rdst = edges.loc[edges["edge_id"] == rev, "dst_node"].iloc[0]
        if (src, dst) != (rdst, rsrc):
            _fail(
                root,
                f"edge {edge_id!r} ({src}->{dst}) and its reverse {rev!r} ({rsrc}->{rdst}) "
                "do not connect the same node pair",
            )


def _validate_references(data: SiteData, root: Path) -> None:
    interior = set(data.interior_nodes)
    outside_ids = set(data.nodes.loc[data.nodes["kind"] == OUTSIDE_NODE, "node_id"].astype(str))

    observed_nodes = set(data.occupancy["node_id"].astype(str))
    unknown = observed_nodes - interior - outside_ids
    if unknown:
        _fail(root, f"occupancy.parquet references unknown node ids {_sample(unknown)}")
    leaked_outside = observed_nodes & outside_ids
    if leaked_outside:
        _fail(
            root,
            f"occupancy.parquet contains the virtual outside node {_sample(leaked_outside)}; "
            "its occupancy is unbounded and must not be modelled",
        )
    absent = interior - observed_nodes
    if absent:
        _fail(root, f"occupancy.parquet has no rows for interior nodes {_sample(absent)}")

    known_edges = set(data.edge_ids)
    observed_edges = set(data.flow["edge_id"].astype(str))
    unknown_edges = observed_edges - known_edges
    if unknown_edges:
        _fail(root, f"flow.parquet references unknown edge ids {_sample(unknown_edges)}")
    absent_edges = known_edges - observed_edges
    if absent_edges:
        _fail(root, f"flow.parquet has no rows for edges {_sample(absent_edges)}")


def _validate_duplicates(data: SiteData, root: Path) -> None:
    for frame, key, filename in (
        (data.occupancy, "node_id", OCCUPANCY_FILE),
        (data.flow, "edge_id", FLOW_FILE),
    ):
        dupes = frame.duplicated(subset=["timestamp", key])
        if bool(dupes.any()):
            offending = frame.loc[dupes, ["timestamp", key]].astype(str).agg(" ".join, axis=1)
            _fail(root, f"{filename} has duplicate (timestamp, {key}) rows: {_sample(offending)}")
    for frame, filename in (
        (data.covariates_past, COVARIATES_PAST_FILE),
        (data.covariates_future, COVARIATES_FUTURE_FILE),
    ):
        if frame.empty:
            continue
        dupes = frame.duplicated(subset=["timestamp", "scope", "variable"])
        if bool(dupes.any()):
            offending = (
                frame.loc[dupes, ["timestamp", "scope", "variable"]]
                .astype(str)
                .agg(" ".join, axis=1)
            )
            _fail(root, f"{filename} has duplicate (timestamp, scope, variable) rows: "
                        f"{_sample(offending)}")


def _validate_time_grid(data: SiteData, root: Path) -> pd.DatetimeIndex:
    occ_stamps = pd.DatetimeIndex(sorted(set(data.occupancy["timestamp"])))
    flow_stamps = pd.DatetimeIndex(sorted(set(data.flow["timestamp"])))
    if len(occ_stamps) < 2:
        _fail(root, "occupancy.parquet must cover at least two timestamps")
    if not occ_stamps.equals(flow_stamps):
        only_occ = occ_stamps.difference(flow_stamps)
        only_flow = flow_stamps.difference(occ_stamps)
        _fail(
            root,
            "occupancy and flow are on different time grids: "
            f"{len(only_occ)} occupancy-only ({_sample(only_occ, 3)}), "
            f"{len(only_flow)} flow-only ({_sample(only_flow, 3)})",
        )

    step = data.meta.interval
    expected = pd.date_range(occ_stamps[0], occ_stamps[-1], freq=step)
    if len(expected) != len(occ_stamps) or not expected.equals(occ_stamps):
        gaps = expected.difference(occ_stamps)
        extra = occ_stamps.difference(expected)
        _fail(
            root,
            f"timestamps do not form a regular {data.meta.interval_seconds}s grid: "
            f"{len(gaps)} missing (e.g. {_sample(gaps, 3)}), "
            f"{len(extra)} off-grid (e.g. {_sample(extra, 3)})",
        )

    # Every entity must be present at every timestamp: a ragged panel silently changes
    # what "missing" means downstream, so it is rejected here instead.
    n_stamps = len(occ_stamps)
    for frame, key, filename, expected_n in (
        (data.occupancy, "node_id", OCCUPANCY_FILE, len(data.interior_nodes)),
        (data.flow, "edge_id", FLOW_FILE, len(data.edge_ids)),
    ):
        if len(frame) != n_stamps * expected_n:
            counts = frame.groupby(key).size()
            ragged = counts[counts != n_stamps]
            _fail(
                root,
                f"{filename} is ragged: expected {expected_n} entities x {n_stamps} timestamps "
                f"= {n_stamps * expected_n} rows, found {len(frame)}; "
                f"entities with wrong row counts: {_sample(ragged.index)}",
            )
    return occ_stamps


def _validate_values(data: SiteData, root: Path) -> None:
    for frame, key, filename in (
        (data.occupancy, "node_id", OCCUPANCY_FILE),
        (data.flow, "edge_id", FLOW_FILE),
    ):
        counts = pd.to_numeric(frame["count"], errors="coerce")
        negative = frame.loc[counts < 0, key]
        if len(negative) > 0:
            _fail(root, f"{filename} contains negative counts for {_sample(negative)}")
        finite = counts.dropna()
        if len(finite) > 0 and not np.isfinite(finite.to_numpy()).all():
            _fail(root, f"{filename} contains non-finite counts")

    capacity = data.node_capacity()
    occ = data.occupancy
    over = occ[
        pd.to_numeric(occ["count"], errors="coerce")
        > occ["node_id"].map(capacity).astype("float64")
    ]
    if len(over) > 0:
        worst = over.assign(
            excess=pd.to_numeric(over["count"]) - over["node_id"].map(capacity)
        ).nlargest(3, "excess")
        detail = ", ".join(
            f"{row.node_id}@{row.timestamp}: {row.count} > {capacity[str(row.node_id)]:g}"
            for row in worst.itertuples()
        )
        _fail(root, f"occupancy exceeds node capacity in {len(over)} rows ({detail})")


def _validate_covariates(data: SiteData, root: Path, grid: pd.DatetimeIndex) -> None:
    known_scopes = set(data.nodes["node_id"].astype(str)) | {GLOBAL_SCOPE}
    for frame, filename in (
        (data.covariates_past, COVARIATES_PAST_FILE),
        (data.covariates_future, COVARIATES_FUTURE_FILE),
    ):
        if frame.empty:
            continue
        unknown = frame.loc[~frame["scope"].isin(known_scopes), "scope"]
        if len(unknown) > 0:
            _fail(root, f"{filename} references unknown scopes {_sample(unknown)}")

    if not data.covariates_past.empty:
        off_grid = pd.DatetimeIndex(data.covariates_past["timestamp"].unique()).difference(grid)
        if len(off_grid) > 0:
            _fail(
                root,
                f"{COVARIATES_PAST_FILE} has {len(off_grid)} timestamps outside the observed "
                f"grid (e.g. {_sample(off_grid, 3)}); past covariates are observations, "
                "they cannot extend beyond the data",
            )

    if not data.covariates_future.empty:
        illegal = set(data.covariates_future["variable"].unique()) - FUTURE_COVARIATE_VARIABLES
        if illegal:
            _fail(
                root,
                f"{COVARIATES_FUTURE_FILE} contains variables that are not known at forecast "
                f"time: {sorted(illegal)}; allowed {sorted(FUTURE_COVARIATE_VARIABLES)}",
            )
        step = data.meta.interval
        stamps = pd.DatetimeIndex(sorted(set(data.covariates_future["timestamp"])))
        expected = pd.date_range(stamps[0], stamps[-1], freq=step)
        if not expected.equals(stamps):
            _fail(root, f"{COVARIATES_FUTURE_FILE} timestamps are not on the regular grid")
        if stamps[0] > grid[0]:
            _fail(
                root,
                f"{COVARIATES_FUTURE_FILE} starts at {stamps[0]} but the observed grid starts "
                f"at {grid[0]}; known-future covariates must cover the context as well",
            )


def _validate_conservation(data: SiteData, root: Path, tolerance: float) -> None:
    """Check that persons are conserved at every interior node.

    For every interior node ``v`` and every step ``t`` after the first::

        o_v(t) - o_v(t-1) - sum_{e in in(v)} f_e(t) + sum_{e in out(v)} f_e(t) == 0
    """
    residual = conservation_residual(data)
    if residual.size == 0:
        return
    magnitude = np.abs(residual.to_numpy())
    if not np.isfinite(magnitude).any():
        # Every cell is missing, so the identity is not satisfied or violated -- it is
        # simply unverifiable. Saying so beats numpy's "All-NaN slice encountered", which
        # tells a reader nothing about which of their files is wrong.
        _fail(
            root,
            "the conservation residual is missing at every node and step, so the identity "
            "cannot be checked at all. Occupancy or flow is entirely absent; a site with "
            "no observations is not a site.",
        )
    worst = float(np.nanmax(magnitude))
    if worst > tolerance:
        stacked = residual.stack()
        worst_time, worst_node = cast("tuple[Any, Any]", stacked.abs().idxmax())
        _fail(
            root,
            f"conservation residual {worst:.6g} exceeds tolerance {tolerance:g}; worst at "
            f"node {worst_node!r} time {worst_time} "
            f"({int((residual.abs() > tolerance).to_numpy().sum())} violating cells)",
        )


def conservation_residual(data: SiteData) -> pd.DataFrame:
    """Per-node, per-step conservation residual as a ``(T-1, n_interior_nodes)`` frame.

    A positive value means the node gained more people than its inflow minus outflow
    can explain. The first timestamp has no predecessor and is dropped.
    """
    occ = data.occupancy.pivot(index="timestamp", columns="node_id", values="count").sort_index()
    flw = data.flow.pivot(index="timestamp", columns="edge_id", values="count").sort_index()
    interior = data.interior_nodes
    occ = occ.reindex(columns=interior)

    delta = occ.diff().iloc[1:]
    net = pd.DataFrame(0.0, index=delta.index, columns=interior)
    for edge_id, src, dst in zip(
        data.edges["edge_id"], data.edges["src_node"], data.edges["dst_node"], strict=True
    ):
        crossings = flw[edge_id].iloc[1:]
        if dst in net.columns:
            net[dst] = net[dst] + crossings
        if src in net.columns:
            net[src] = net[src] - crossings
    return delta - net
