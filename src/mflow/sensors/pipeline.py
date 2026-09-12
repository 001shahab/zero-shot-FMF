"""Clean canonical site in, observed canonical site out (M3).

The degradation stage is deliberately a separate pass over a finished canonical site
rather than an option inside the simulator. The clean site is kept as ground truth, the
observed site is what a forecaster is allowed to see, and both are ordinary canonical
sites that everything downstream reads through the same loader.

Order of operations, which matters:

1. the counting model damages the flow series, because a miscount is a property of the
   crossing and happens before anything reaches a data logger;
2. the environmental model is driven by the *true* occupancy, since CO2 does not care
   what the people counter thought it saw;
3. the fault model runs last on every stream alike -- occupancy, flow and environment --
   because a dead gateway drops whatever was on the wire.

The observed site does not satisfy conservation, and that is the point: reconciliation
has something to repair. ``meta.has_ground_truth_flow`` is therefore cleared on the
observed site, which is what stops the validator from demanding a zero residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mflow.manifest import rng
from mflow.schema import (
    FLOW_PREFIX,
    OCC_PREFIX,
    SiteData,
    flow_series_id,
    occ_series_id,
)
from mflow.sensors.config import SensorProfile
from mflow.sensors.counting import CountingSensorModel
from mflow.sensors.environment import ENVIRONMENT_VARIABLES, EnvironmentSensorModel
from mflow.sensors.faults import FaultModel, FaultRecord


@dataclass(frozen=True)
class DegradationReport:
    """What the measurement model did to a site.

    Attributes:
        profile: the name of the profile applied.
        seed: the seed the degradation was run with.
        counting_bias: the fixed multiplicative bias drawn per edge counter.
        faults: the fault record per damaged stream, keyed by canonical series id for the
            count series and by ``<node>|<variable>`` for the environmental ones.
    """

    profile: str
    seed: int
    counting_bias: dict[str, float]
    faults: dict[str, FaultRecord]

    @property
    def missing_fraction(self) -> float:
        """Share of all samples across all streams with no reading."""
        total = sum(record.length for record in self.faults.values())
        if total == 0:
            return 0.0
        return sum(len(record.missing) for record in self.faults.values()) / total

    def summary(self) -> dict[str, float | str]:
        """Compact view for a run manifest."""
        shifted = sum(1 for r in self.faults.values() if r.clock_shift_intervals != 0)
        return {
            "profile": self.profile,
            "seed": self.seed,
            "n_streams": len(self.faults),
            "missing_fraction": self.missing_fraction,
            "stuck_fraction": (
                sum(len(r.stuck) for r in self.faults.values())
                / max(sum(r.length for r in self.faults.values()), 1)
            ),
            "n_clock_shifted_streams": shifted,
        }


def degrade(
    site: SiteData, profile: SensorProfile, *, seed: int
) -> tuple[SiteData, DegradationReport]:
    """Apply a sensor profile to a clean canonical site.

    Args:
        site: the clean site, treated as ground truth and never modified.
        profile: the measurement model to apply.
        seed: run seed. Every draw derives from it, so the same clean site, profile and
            seed always produce the same observed site.

    Returns:
        The observed site and a record of every fault that was injected.
    """
    site_id = site.meta.site_id
    interval = site.meta.interval_seconds

    counting = CountingSensorModel(
        profile.counting, site.edge_ids, rng(seed, site_id, "counting", "bias")
    )
    observed_flow = counting.apply(site.flow, rng(seed, site_id, "counting", "apply"))

    environment = EnvironmentSensorModel(
        profile.environment,
        _node_areas(site),
        rng(seed, site_id, "environment", "offset"),
    )
    environment_frame = environment.apply(
        site.occupancy, interval, rng(seed, site_id, "environment", "apply")
    )

    # Every stream that a logger can lose, named once so the fault model and the report
    # agree on what exists.
    count_streams = site.series_ids
    environment_streams = [
        f"{scope}|{variable}"
        for scope in sorted(str(s) for s in environment_frame["scope"].unique())
        for variable in ENVIRONMENT_VARIABLES
    ]
    faults = FaultModel(
        profile.faults,
        count_streams + environment_streams,
        rng(seed, site_id, "faults", "clock"),
    )

    occupancy, flow, records = _damage_counts(
        site, observed_flow, faults, rng(seed, site_id, "faults", "counts")
    )
    covariates_past, environment_records = _damage_environment(
        site, environment_frame, faults, rng(seed, site_id, "faults", "environment")
    )
    records.update(environment_records)

    observed = SiteData(
        # The observed flow no longer balances the observed occupancy, so the site must
        # stop claiming ground-truth flow or validation will reject it for a residual the
        # measurement model deliberately introduced.
        meta=site.meta.model_copy(
            update={
                "has_ground_truth_flow": False,
                "provenance": {
                    **site.meta.provenance,
                    "sensor_profile": profile.name,
                    "sensor_seed": seed,
                    "clean_site_id": site_id,
                },
            }
        ),
        nodes=site.nodes.copy(),
        edges=site.edges.copy(),
        occupancy=occupancy,
        flow=flow,
        covariates_past=covariates_past,
        covariates_future=site.covariates_future.copy(),
    )
    report = DegradationReport(
        profile=profile.name,
        seed=seed,
        counting_bias=dict(counting.bias),
        faults=records,
    )
    return observed, report


def _node_areas(site: SiteData) -> dict[str, float]:
    """Floor area per interior node, which the environmental balances need."""
    interior = set(site.interior_nodes)
    return {
        str(node): float(area)
        for node, area in zip(site.nodes["node_id"], site.nodes["area_m2"], strict=True)
        if str(node) in interior
    }


def _damage_counts(
    site: SiteData,
    observed_flow: pd.DataFrame,
    faults: FaultModel,
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FaultRecord]]:
    """Run the fault model over the occupancy and flow streams."""
    records: dict[str, FaultRecord] = {}

    occ_wide = site.occupancy.pivot(
        index="timestamp", columns="node_id", values="count"
    ).sort_index()
    flow_wide = observed_flow.pivot(
        index="timestamp", columns="edge_id", values="count"
    ).sort_index()

    for node_id in site.interior_nodes:
        series_id = occ_series_id(node_id)
        damaged, record = faults.apply(series_id, occ_wide[node_id].to_numpy(), generator)
        occ_wide[node_id] = damaged
        records[series_id] = record
    for edge_id in site.edge_ids:
        series_id = flow_series_id(edge_id)
        damaged, record = faults.apply(series_id, flow_wide[edge_id].to_numpy(), generator)
        flow_wide[edge_id] = damaged
        records[series_id] = record

    return _melt(occ_wide, "node_id"), _melt(flow_wide, "edge_id"), records


def _damage_environment(
    site: SiteData,
    environment_frame: pd.DataFrame,
    faults: FaultModel,
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, FaultRecord]]:
    """Run the fault model over the environmental streams and merge them into covariates."""
    records: dict[str, FaultRecord] = {}
    if environment_frame.empty:
        return site.covariates_past.copy(), records

    pieces: list[pd.DataFrame] = []
    for (scope, variable), group in environment_frame.groupby(
        ["scope", "variable"], sort=True
    ):
        ordered = group.sort_values("timestamp")
        stream_id = f"{scope}|{variable}"
        damaged, record = faults.apply(stream_id, ordered["value"].to_numpy(), generator)
        records[stream_id] = record
        pieces.append(ordered.assign(value=damaged))

    generated = pd.concat(pieces, ignore_index=True)
    # Environmental channels the site already carried win over generated ones, so that a
    # real dataset with measured CO2 is never overwritten by a simulated proxy.
    existing = site.covariates_past
    if existing.empty:
        return generated, records
    occupied = set(
        zip(existing["scope"].astype(str), existing["variable"].astype(str), strict=True)
    )
    keep = ~pd.Series(
        list(
            zip(generated["scope"].astype(str), generated["variable"].astype(str), strict=True)
        ),
        index=generated.index,
    ).isin(occupied)
    return pd.concat([existing, generated[keep]], ignore_index=True), records


def _melt(wide: pd.DataFrame, key: str) -> pd.DataFrame:
    """Turn a ``(T, n_entities)`` frame back into the canonical long form."""
    long = wide.reset_index().melt(id_vars="timestamp", var_name=key, value_name="count")
    long[key] = long[key].astype(str)
    return long.sort_values(["timestamp", key]).reset_index(drop=True)


def observed_series_ids(site: SiteData) -> list[str]:
    """Canonical ids of the count streams a sensor profile can damage."""
    return [
        sid
        for sid in site.series_ids
        if sid.startswith(OCC_PREFIX) or sid.startswith(FLOW_PREFIX)
    ]
