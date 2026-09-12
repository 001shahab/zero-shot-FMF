#!/usr/bin/env python3
"""Regenerate the committed three-room toy site under ``tests/fixtures/toy_site``.

The fixture is the reference implementation of the canonical data contract: it is small
enough to read by eye, and it satisfies conservation exactly, so any residual a test sees
comes from the code under test rather than from the data.

The generator walks a tiny deterministic crowd around the graph. Crossings are drawn
edge by edge and immediately clipped to what the source node actually holds and what the
destination node can still take, which is why the resulting occupancy series need no
repair afterwards.

Usage::

    python scripts/build_toy_fixture.py [--out tests/fixtures/toy_site] [--seed 20260912]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mflow.graph import BuildingGraph
from mflow.manifest import rng
from mflow.paths import repo_root
from mflow.schema import SiteData, SiteMeta, write_site

SITE_ID = "toy_three_room"
INTERVAL_SECONDS = 60
N_STEPS = 240  # four hours at one minute

NODES = pd.DataFrame(
    [
        ("outside", "Outside", "outside", 0.0, 0),
        ("foyer", "Entrance foyer", "entrance", 60.0, 40),
        ("gallery_a", "Gallery A", "gallery", 120.0, 60),
        ("gallery_b", "Gallery B", "gallery", 80.0, 30),
    ],
    columns=["node_id", "name", "kind", "area_m2", "capacity_persons"],
)

EDGES = pd.DataFrame(
    [
        ("e_in", "outside", "foyer", 2.4, 60.0, "e_out"),
        ("e_out", "foyer", "outside", 2.4, 60.0, "e_in"),
        ("e_fa", "foyer", "gallery_a", 1.6, 40.0, "e_af"),
        ("e_af", "gallery_a", "foyer", 1.6, 40.0, "e_fa"),
        ("e_ab", "gallery_a", "gallery_b", 1.2, 25.0, "e_ba"),
        ("e_ba", "gallery_b", "gallery_a", 1.2, 25.0, "e_ab"),
    ],
    columns=[
        "edge_id",
        "src_node",
        "dst_node",
        "width_m",
        "capacity_persons_per_min",
        "reverse_edge_id",
    ],
)


def _arrival_intensity(step: int, n_steps: int) -> float:
    """Persons per minute arriving from outside: a single smooth midday peak."""
    phase = (step / n_steps) * np.pi
    return 6.0 * float(np.sin(phase) ** 2)


def simulate(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate conservation-exact occupancy and flow tables.

    Returns:
        ``(occupancy, flow)`` long frames with UTC timestamps.
    """
    graph = BuildingGraph.from_frames(NODES, EDGES)
    graph.check_connectivity()
    generator = rng(seed, "toy_fixture")

    interior = list(graph.interior_nodes)
    occupancy = dict.fromkeys(interior, 0)
    capacity = {n: int(graph.node_capacity[n]) for n in interior}
    per_interval_cap = {
        e: int(graph.edge_capacity_per_min[e] * INTERVAL_SECONDS / 60) for e in graph.edge_ids
    }

    timestamps = pd.date_range("2026-03-01T09:00:00Z", periods=N_STEPS, freq="60s")
    occ_rows: list[tuple[pd.Timestamp, str, int]] = []
    flow_rows: list[tuple[pd.Timestamp, str, int]] = []

    # Edge order is fixed so that the greedy feasibility clipping is deterministic.
    for step, stamp in enumerate(timestamps):
        arrivals = int(generator.poisson(_arrival_intensity(step, N_STEPS)))
        crossings: dict[str, int] = {}

        for edge_id, src, dst in zip(graph.edge_ids, graph.src, graph.dst, strict=True):
            if src == graph.outside_node:
                wanted = arrivals
            else:
                # A fraction of whoever stands in the source room walks through.
                share = {"e_out": 0.10, "e_fa": 0.35, "e_af": 0.12, "e_ab": 0.30, "e_ba": 0.20}[
                    edge_id
                ]
                wanted = int(generator.binomial(max(occupancy[src], 0), share))
            feasible = min(wanted, per_interval_cap[edge_id])
            if src != graph.outside_node:
                feasible = min(feasible, occupancy[src])
            if dst != graph.outside_node:
                feasible = min(feasible, capacity[dst] - occupancy[dst])
            feasible = max(feasible, 0)
            crossings[edge_id] = feasible
            if src != graph.outside_node:
                occupancy[src] -= feasible
            if dst != graph.outside_node:
                occupancy[dst] += feasible

        for node in interior:
            occ_rows.append((stamp, node, occupancy[node]))
        for edge_id in graph.edge_ids:
            flow_rows.append((stamp, edge_id, crossings[edge_id]))

    occ = pd.DataFrame(occ_rows, columns=["timestamp", "node_id", "count"])
    flw = pd.DataFrame(flow_rows, columns=["timestamp", "edge_id", "count"])
    return occ, flw


def build_covariates(occupancy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past CO2 proxies per node and a known-future opening flag.

    The CO2 proxy here is a deliberately crude affine map of occupancy; the physically
    motivated mass-balance model lives in :mod:`mflow.sensors.environment` and is what
    the experiments use. The fixture only needs a covariate column with the right shape.
    """
    past = occupancy.assign(
        scope=occupancy["node_id"],
        variable="co2_ppm",
        value=420.0 + 9.0 * occupancy["count"].astype(float),
    )[["timestamp", "scope", "variable", "value"]]

    stamps = pd.DatetimeIndex(sorted(occupancy["timestamp"].unique()))
    # Known-future covariates must also cover the horizon beyond the observed window.
    extended = stamps.append(
        pd.date_range(stamps[-1] + pd.Timedelta(seconds=INTERVAL_SECONDS), periods=60, freq="60s")
    )
    future = pd.DataFrame(
        {
            "timestamp": extended,
            "scope": "global",
            "variable": "is_open",
            "value": 1.0,
        }
    )
    return past, future


def main() -> None:
    """Build the fixture and write it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "tests" / "fixtures" / "toy_site",
        help="destination directory",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    occupancy, flow = simulate(args.seed)
    past, future = build_covariates(occupancy)
    site = SiteData(
        meta=SiteMeta(
            site_id=SITE_ID,
            interval_seconds=INTERVAL_SECONDS,
            timezone="Europe/Rome",
            source="simulated",
            has_ground_truth_flow=True,
            opening_hours={"mon": None, "tue": ["09:00", "18:00"]},
            provenance={"generator": "scripts/build_toy_fixture.py", "seed": args.seed},
        ),
        nodes=NODES,
        edges=EDGES,
        occupancy=occupancy,
        flow=flow,
        covariates_past=past,
        covariates_future=future,
    )
    out = write_site(site, args.out)
    print(f"wrote validated toy site to {out}")


if __name__ == "__main__":
    main()
