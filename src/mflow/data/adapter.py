"""The shared shape of a dataset adapter.

An adapter turns one external source into one or more canonical sites. Everything that is
true of every adapter lives here; everything specific to a source lives in that source's
module, next to a docstring citing the documentation it was written against.

Two rules apply to every adapter and are enforced by :func:`finalise`:

* **The graph must close.** The canonical contract requires a single virtual ``outside``
  node and a conservation identity with no sources or sinks. A real dataset almost never
  arrives that way -- ROBOD has occupancy but no doorway counts, the metro data has
  entries and exits but no explicit outside -- so each adapter has to say how it closes
  the graph, and :func:`finalise` checks that it did.
* **Provenance travels with the data.** ``meta.provenance`` records the source, the
  download manifest's digest for the files the site was built from, and the adapter
  version. A canonical site whose origin cannot be recovered is not usable in a paper.

Where a source cannot supply something the contract requires, the adapter must fail
rather than invent it. An adapter that fabricates flows to satisfy the conservation check
would make every reconciliation number meaningless, and nothing downstream could tell.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from mflow.data.fetch import DownloadManifest
from mflow.paths import canonical_dir, raw_dir
from mflow.schema import SiteData, write_site


class AdapterError(RuntimeError):
    """Raised when a source cannot be expressed in the canonical contract."""


@dataclass(frozen=True)
class AdapterResult:
    """What one adapter run produced.

    Attributes:
        sites: the canonical directories written.
        skipped: source entities that could not be converted, with the reason. A site
            that was skipped is not a site that scored badly, and the distinction has to
            reach the reader of the results.
    """

    sites: list[Path]
    skipped: dict[str, str]

    def summary(self) -> str:
        """One line per site and per skip, for a fetch script to print."""
        lines = [f"wrote {len(self.sites)} site(s):"]
        lines.extend(f"  {path}" for path in self.sites)
        if self.skipped:
            lines.append(f"skipped {len(self.skipped)}:")
            lines.extend(f"  {name}: {why}" for name, why in sorted(self.skipped.items()))
        return "\n".join(lines)


class Adapter(ABC):
    """Base class for the conversion of one external source.

    Subclasses implement :meth:`build`, which reads from ``self.raw`` and returns the
    sites it could construct. They must not write anything: :meth:`run` handles writing,
    validation and provenance, so that no adapter can accidentally skip the contract
    check on its way out.
    """

    #: Short name, matching ``data/raw/<source>/``.
    source: str

    #: Bumped when a change to the adapter would change the canonical output. It goes
    #: into the provenance so that two sites built by different adapter versions are
    #: distinguishable without re-deriving them.
    version: str = "1"

    def __init__(self, raw: Path | None = None, destination: Path | None = None) -> None:
        self.raw = Path(raw) if raw is not None else raw_dir(self.source)
        self.destination = Path(destination) if destination is not None else canonical_dir()

    @abstractmethod
    def build(self) -> tuple[list[SiteData], dict[str, str]]:
        """Read the raw files and return ``(sites, skipped)``.

        Raises:
            AdapterError: if the raw files are absent or are not in the documented
                format. A format that has changed under the adapter must stop the run;
                parsing it optimistically would produce a plausible-looking site built
                from the wrong columns.
        """

    def run(self, *, validate: bool = True) -> AdapterResult:
        """Build the sites, stamp their provenance and write them.

        Args:
            validate: run the full canonical validation on write. Leave it on; the
                conservation check is the only thing standing between a real dataset and
                a silently wrong graph.
        """
        if not self.raw.is_dir():
            raise AdapterError(
                f"{self.raw} does not exist. Run the fetch script for {self.source!r} "
                "first; the adapter does not download anything itself."
            )
        manifest = DownloadManifest.read(self.raw)
        sites, skipped = self.build()
        written: list[Path] = []
        for site in sites:
            stamped = self.stamp(site, manifest)
            target = self.destination / stamped.meta.site_id
            write_site(stamped, target, validate=validate)
            written.append(target)
        return AdapterResult(sites=written, skipped=skipped)

    def stamp(self, site: SiteData, manifest: DownloadManifest) -> SiteData:
        """Attach the source's provenance to a site's metadata."""
        provenance: dict[str, Any] = {
            **site.meta.provenance,
            "source": self.source,
            "adapter_version": self.version,
            "downloaded_at": manifest.downloaded_at,
            "licence": manifest.licence,
            "citation": manifest.citation,
            "raw_files": {record.name: record.sha256 for record in manifest.files},
        }
        return SiteData(
            meta=site.meta.model_copy(update={"provenance": provenance}),
            nodes=site.nodes,
            edges=site.edges,
            occupancy=site.occupancy,
            flow=site.flow,
            covariates_past=site.covariates_past,
            covariates_future=site.covariates_future,
        )


# --------------------------------------------------------------------------- #
# Helpers every adapter needs
# --------------------------------------------------------------------------- #


def _localise_errors() -> tuple[type[Exception], ...]:
    """What ``tz_localize`` raises on an ambiguous or non-existent local time.

    It depends on what else is installed, which is a trap. On its own pandas raises a
    ``ValueError``; when pytz is present -- and it arrives transitively, in this project
    through gluonts behind the Toto wrapper -- pandas raises
    ``pytz.exceptions.InvalidTimeError`` instead, which is not a ``ValueError`` at all.
    Catching only one of them means the guard below quietly stops guarding the moment an
    unrelated dependency changes, so both are named.
    """
    try:
        from pytz.exceptions import InvalidTimeError
    except ImportError:
        return (ValueError,)
    return (ValueError, InvalidTimeError)


_LOCALISE_ERRORS: Final[tuple[type[Exception], ...]] = _localise_errors()


def to_utc(series: pd.Series, timezone: str) -> pd.Series:
    """Localise naive local timestamps to ``timezone`` and convert to UTC.

    Real datasets publish local wall-clock time. Converting through the named zone rather
    than a fixed offset is what makes a record spanning a daylight-saving transition come
    out right.

    Raises:
        AdapterError: on an ambiguous or non-existent local time, which is what a record
            sampled through a DST transition produces. Silently picking one
            interpretation would shift an hour of data by an hour.
    """
    stamps = pd.to_datetime(series)
    if stamps.dt.tz is not None:
        return stamps.dt.tz_convert("UTC")
    try:
        localised = stamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    except _LOCALISE_ERRORS as error:
        raise AdapterError(
            f"could not localise timestamps to {timezone!r}: {error}. The record spans a "
            "daylight-saving transition; the adapter must say which side each ambiguous "
            "timestamp falls on rather than letting an hour of data shift by an hour."
        ) from error
    return localised.dt.tz_convert("UTC")


def regular_grid(stamps: pd.DatetimeIndex, interval_seconds: int) -> tuple[pd.DatetimeIndex, int]:
    """The complete regular grid spanning ``stamps``, and how many steps are missing.

    The canonical contract requires a regular, gapless time index. Real records have
    gaps, so an adapter reindexes onto this grid and leaves the missing steps as NaN --
    which is the honest representation, and the one the fault-aware metrics already
    handle. The returned count is what the adapter should report so the gap size is
    visible rather than buried.
    """
    if len(stamps) == 0:
        raise AdapterError("cannot build a grid from an empty timestamp index")
    grid = pd.date_range(
        stamps.min(), stamps.max(), freq=pd.Timedelta(seconds=interval_seconds), tz="UTC"
    )
    return grid, len(grid) - len(stamps.unique())


def long_counts(
    frame: pd.DataFrame, *, entity_column: str, value_column: str, entity_name: str
) -> pd.DataFrame:
    """Reshape a source frame into the canonical ``timestamp, <entity>, count`` form."""
    missing = {"timestamp", entity_column, value_column} - set(frame.columns)
    if missing:
        raise AdapterError(f"frame is missing column(s) {sorted(missing)}")
    return (
        frame[["timestamp", entity_column, value_column]]
        .rename(columns={entity_column: entity_name, value_column: "count"})
        .sort_values(["timestamp", entity_name], ignore_index=True)
    )


def long_covariates(
    frame: pd.DataFrame, variables: dict[str, str], *, scope_column: str
) -> pd.DataFrame:
    """Melt wide environmental columns into the canonical covariate form.

    Args:
        frame: must carry ``timestamp``, the scope column, and the source column names.
        variables: source column name to canonical variable name.
        scope_column: the column naming the node (or ``global``) each row applies to.
    """
    present = {src: dst for src, dst in variables.items() if src in frame.columns}
    if not present:
        return pd.DataFrame(columns=["timestamp", "scope", "variable", "value"])
    melted = frame.melt(
        id_vars=["timestamp", scope_column],
        value_vars=list(present),
        var_name="variable",
        value_name="value",
    )
    melted["variable"] = melted["variable"].map(present)
    return (
        melted.rename(columns={scope_column: "scope"})
        .dropna(subset=["value"])
        .sort_values(["timestamp", "scope", "variable"], ignore_index=True)
    )
