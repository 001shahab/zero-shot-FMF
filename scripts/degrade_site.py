#!/usr/bin/env python
"""Apply a sensor profile to a clean canonical site (M3).

Example:
    python scripts/degrade_site.py --site data/canonical/sim_palazzo \\
        --profile configs/sensors/realistic.yaml --seed 0

Writes ``<site>__<profile>`` next to the clean site and prints what the measurement model
did, including the CO2 lag and correlation that the M3 acceptance criterion is stated in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mflow.paths import canonical_dir, configs_dir
from mflow.schema import load_site, write_site
from mflow.sensors import co2_lag_minutes, degrade, load_sensor_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="clean canonical site")
    parser.add_argument(
        "--profile",
        type=Path,
        default=configs_dir("sensors") / "realistic.yaml",
        help="sensor profile YAML",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    clean = load_site(args.site)
    profile = load_sensor_profile(args.profile)
    observed, report = degrade(clean, profile, seed=args.seed)

    destination = args.out or canonical_dir(f"{clean.meta.site_id}__{profile.name}")
    write_site(observed, destination)
    print(f"wrote {destination}")

    summary = dict(report.summary())
    summary.update(_environment_diagnostics(clean, observed))
    print(json.dumps(summary, indent=2, default=float))


def _environment_diagnostics(clean, observed) -> dict[str, float]:
    """Median CO2 lag and correlation across nodes, for checking against the literature."""
    covariates = observed.covariates_past
    co2 = covariates[covariates["variable"] == "co2_ppm"]
    if co2.empty:
        return {}
    truth = clean.occupancy.pivot(index="timestamp", columns="node_id", values="count")
    lags: list[float] = []
    correlations: list[float] = []
    for node_id, group in co2.groupby("scope"):
        if node_id not in truth.columns:
            continue
        series = group.sort_values("timestamp")["value"].to_numpy()
        counts = truth[node_id].to_numpy()
        finite = np.isfinite(series) & np.isfinite(counts)
        if finite.sum() < 100 or np.std(counts[finite]) == 0.0:
            continue
        lag, correlation = co2_lag_minutes(
            counts[finite], series[finite], clean.meta.interval_seconds
        )
        lags.append(lag)
        correlations.append(correlation)
    if not lags:
        return {}
    return {
        "co2_lag_minutes_median": float(np.median(lags)),
        "co2_correlation_median": float(np.median(correlations)),
        "n_nodes_measured": len(lags),
    }


if __name__ == "__main__":
    main()
