"""Turn raw simulator output into a canonical site.

Occupancy is derived from the event log by integration rather than sampled from the
simulator's internal counters. Both would agree, but integrating the events makes the
conservation identity true *by construction*: the occupancy change of a room over an
interval is, definitionally, the crossings in minus the crossings out. That is what lets
the M2 acceptance test assert a residual of exactly zero and treat any non-zero residual
downstream as a real defect rather than an artefact of how the series were sampled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mflow.schema import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    SiteData,
    SiteMeta,
)
from mflow.sim.arrivals import scheduled_group_covariates
from mflow.sim.graph_sim import GraphSimulator, SimulationResult


def _canonical_nodes(result: SimulationResult) -> pd.DataFrame:
    rows = [
        (node.id, node.name, node.kind, node.area_m2, node.capacity_persons)
        for node in result.config.nodes
    ]
    return pd.DataFrame(rows, columns=list(NODE_COLUMNS))


def _canonical_edges(simulator: GraphSimulator, result: SimulationResult) -> pd.DataFrame:
    """Expand the undirected doorway declarations into the canonical directed pairs."""
    directed = simulator.directed_edges
    by_pair = {(src, dst): edge_id for (src, dst), edge_id in directed.items()}
    rows = []
    for edge in result.config.edges:
        pairs = [(edge.src, edge.dst)]
        if edge.bidirectional:
            pairs.append((edge.dst, edge.src))
        for src, dst in pairs:
            edge_id = by_pair[(src, dst)]
            reverse = by_pair.get((dst, src), "")
            rows.append(
                (
                    edge_id,
                    src,
                    dst,
                    edge.width_m,
                    edge.capacity_persons_per_min,
                    reverse,
                )
            )
    return pd.DataFrame(rows, columns=list(EDGE_COLUMNS))


def aggregate(
    simulator: GraphSimulator, result: SimulationResult, *, drop_warmup_days: int = 0
) -> SiteData:
    """Convert an event log into canonical occupancy and flow series.

    Args:
        simulator: the simulator that produced the result, for the directed edge map.
        result: the raw run output.
        drop_warmup_days: leading days to discard. The building starts empty, so the
            first morning is not representative; discarding it is preferable to seeding
            an arbitrary initial occupancy.

    Returns:
        A :class:`~mflow.schema.SiteData` whose conservation residual is exactly zero.
    """
    config = result.config
    steps_per_interval = config.steps_per_interval
    n_intervals = result.n_steps // steps_per_interval
    if n_intervals < 2:
        raise ValueError(
            f"the run covers {result.n_steps} steps, which is fewer than two "
            f"{config.interval_seconds}s intervals"
        )

    nodes = _canonical_nodes(result)
    edges = _canonical_edges(simulator, result)
    interior = [str(n) for n, k in zip(nodes["node_id"], nodes["kind"], strict=True)
                if k != config.outside_node and k != "outside"]
    edge_ids = [str(e) for e in edges["edge_id"]]
    pair_to_edge = {
        (str(s), str(d)): str(e)
        for e, s, d in zip(edges["edge_id"], edges["src_node"], edges["dst_node"], strict=True)
    }

    # Crossings per (interval, directed edge).
    events = result.events
    flow_matrix = np.zeros((n_intervals, len(edge_ids)), dtype=np.int64)
    edge_position = {edge_id: i for i, edge_id in enumerate(edge_ids)}
    if len(events) > 0:
        interval_index = np.minimum(
            events["step"].to_numpy() // steps_per_interval, n_intervals - 1
        )
        for interval, src, dst in zip(
            interval_index, events["from_node"], events["to_node"], strict=True
        ):
            edge_id = pair_to_edge.get((str(src), str(dst)))
            if edge_id is None:
                raise ValueError(
                    f"the event log contains a crossing {src!r} -> {dst!r} that is not a "
                    "declared doorway; the simulator and the site configuration disagree"
                )
            flow_matrix[interval, edge_position[edge_id]] += 1

    # Occupancy by integrating net flow. Starting from an empty building, the cumulative
    # sum is the headcount at the end of each interval.
    node_position = {node: i for i, node in enumerate(interior)}
    net = np.zeros((n_intervals, len(interior)), dtype=np.int64)
    for edge_id, src, dst in zip(
        edges["edge_id"], edges["src_node"], edges["dst_node"], strict=True
    ):
        column = flow_matrix[:, edge_position[str(edge_id)]]
        if str(dst) in node_position:
            net[:, node_position[str(dst)]] += column
        if str(src) in node_position:
            net[:, node_position[str(src)]] -= column
    occupancy_matrix = np.cumsum(net, axis=0)
    if occupancy_matrix.min() < 0:
        raise ValueError(
            "integrating the event log produced a negative occupancy, which means the "
            "log is not a valid sequence of crossings"
        )

    step = pd.Timedelta(seconds=config.interval_seconds)
    local_start = result.start + step  # the first row is the interval *ending* here
    timestamps = pd.date_range(local_start, periods=n_intervals, freq=step).tz_convert("UTC")

    warmup = drop_warmup_days * (24 * 3600 // config.interval_seconds)
    if warmup >= n_intervals:
        raise ValueError(
            f"drop_warmup_days={drop_warmup_days} would discard the entire run"
        )
    keep = slice(warmup, n_intervals)
    timestamps = timestamps[keep]
    occupancy_matrix = occupancy_matrix[keep]
    flow_matrix = flow_matrix[keep]

    occupancy = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, len(interior)),
            "node_id": np.tile(interior, len(timestamps)),
            "count": occupancy_matrix.reshape(-1).astype(np.int32),
        }
    )
    flow = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, len(edge_ids)),
            "edge_id": np.tile(edge_ids, len(timestamps)),
            "count": flow_matrix.reshape(-1).astype(np.int32),
        }
    )

    covariates_future = scheduled_group_covariates(config, pd.DatetimeIndex(timestamps))
    meta = SiteMeta(
        site_id=config.site_id,
        interval_seconds=config.interval_seconds,
        timezone=config.timezone,
        source="simulated",
        has_ground_truth_flow=True,
        opening_hours=_opening_hours(config.arrivals.open_time, config.arrivals.close_time),
        provenance={
            "simulator": "mflow.sim.graph_sim.GraphSimulator",
            "seed": simulator.seed,
            "days": result.n_steps * config.step_seconds / 86_400,
            "drop_warmup_days": drop_warmup_days,
            "ejected_agents": result.ejected_agents,
            "blocked_doorway_ticks": int(sum(result.blocked_requests.values())),
        },
    )
    return SiteData(
        meta=meta,
        nodes=nodes,
        edges=edges,
        occupancy=occupancy,
        flow=flow,
        covariates_future=covariates_future,
    )


def _opening_hours(open_time: str, close_time: str) -> dict[str, list[str] | None]:
    """The same hours every day; Monday is the traditional Italian museum closing day."""
    hours: dict[str, list[str] | None] = {
        day: [open_time, close_time]
        for day in ("tue", "wed", "thu", "fri", "sat", "sun")
    }
    hours["mon"] = None
    return hours


def summarise(result: SimulationResult, site: SiteData) -> dict[str, float]:
    """Aggregate statistics checked against the site's configured target ranges.

    Returns:
        ``mean_visit_minutes``, ``median_visit_minutes``, ``node_visit_fraction`` (the
        mean fraction of interior rooms a visitor enters), ``occupancy_peak_to_mean``
        (over interior rooms, using the building total), ``n_visitors`` and
        ``mean_dwell_minutes``.
    """
    config = result.config
    step_minutes = config.step_seconds / 60.0
    visits = result.visits
    durations = (visits["exited_step"] - visits["entered_step"]).to_numpy() * step_minutes
    n_interior = max(len([n for n in config.nodes if n.kind != "outside"]), 1)

    total = (
        site.occupancy.groupby("timestamp")["count"].sum().to_numpy().astype(np.float64)
    )
    open_mask = total > 0
    mean_open = float(total[open_mask].mean()) if open_mask.any() else 0.0

    dwell = result.dwells
    dwell_minutes = (
        (dwell["step_out"] - dwell["step_in"]).to_numpy() * step_minutes
        if len(dwell)
        else np.array([0.0])
    )

    return {
        "n_visitors": float(len(visits)),
        "mean_visit_minutes": float(durations.mean()) if durations.size else 0.0,
        "median_visit_minutes": float(np.median(durations)) if durations.size else 0.0,
        "node_visit_fraction": (
            float(visits["n_nodes"].mean() / n_interior) if len(visits) else 0.0
        ),
        "occupancy_peak_to_mean": float(total.max() / mean_open) if mean_open > 0 else 0.0,
        "mean_dwell_minutes": float(dwell_minutes.mean()),
        "ejected_agents": float(result.ejected_agents),
    }
