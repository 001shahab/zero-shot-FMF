#!/usr/bin/env python3
"""Simulate a site and write it as a canonical dataset.

Usage::

    python scripts/simulate_site.py --site configs/sites/house_museum.yaml --days 30
    python scripts/simulate_site.py --site configs/sites/palazzo.yaml --days 120 --seed 7

The clean series are written to ``data/canonical/<site_id>/``. Sensor degradation is a
separate step (``scripts/degrade_site.py``) so that the ground truth is always retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mflow.paths import canonical_dir
from mflow.schema import write_site
from mflow.sim import GraphSimulator, aggregate, load_site_config, summarise


def main() -> None:
    """Run the simulator and write the canonical site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="site YAML")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=str, default="2026-03-02")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=1,
        help="leading days to discard; the building starts empty",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_site_config(args.site)
    simulator = GraphSimulator(config, seed=args.seed)
    result = simulator.run(args.start, args.days)
    site = aggregate(simulator, result, drop_warmup_days=args.warmup_days)

    destination = args.out or canonical_dir(config.site_id)
    write_site(site, destination)

    statistics = summarise(result, site)
    print(f"wrote {destination}")
    print(json.dumps(statistics, indent=2))

    if config.targets is not None:
        checks = {
            "mean_visit_minutes": config.targets.mean_visit_minutes,
            "node_visit_fraction": config.targets.node_visit_fraction,
            "occupancy_peak_to_mean": config.targets.occupancy_peak_to_mean,
        }
        failures = [
            f"{name}={statistics[name]:.3f} outside target {low}-{high}"
            for name, (low, high) in checks.items()
            if not low <= statistics[name] <= high
        ]
        if failures:
            raise SystemExit(
                "simulated statistics fall outside the configured target ranges:\n  "
                + "\n  ".join(failures)
            )
        print("all configured target ranges satisfied")


if __name__ == "__main__":
    main()
