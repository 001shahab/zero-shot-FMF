"""Measure what each forecaster actually costs in memory on this machine.

The sweep's own ``peak_memory_mb`` column is a ``tracemalloc`` figure, which counts only
the Python heap. On Apple Silicon every tensor lives in the MPS allocator instead, so that
column understates a transformer's footprint by orders of magnitude and must not be quoted
as a model's memory cost. Rather than re-run the whole evaluation to collect a corrected
column, this script measures the quantity a deployment decision actually needs: the peak
resident set size of a process that loads one model and runs a single forward pass at the
largest shape in the paper.

Each model is measured in its own subprocess. That is not fastidiousness. ``ru_maxrss`` is
a monotonic high-water mark for the whole process, so measuring several models in one
process credits the first model to allocate heavily with the peak and reports zero for
every model after it, which is how the first version of this script produced
\"chronos2 uses 0 MiB\".

Usage:
    python paper/Arxiv/measure_memory.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TABLES = Path(__file__).resolve().parent / "tables"

# The largest shape in the paper: HZMetro's 240 series at two seasons of context.
N_SERIES = 240
CONTEXT = 192
HORIZON = 8
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MODELS = [
    "timesfm3_univariate",
    "timesfm3_multivariate",
    "chronos2_multivariate",
    "toto2",
    "global_lightgbm",
    "historical_average",
    "seasonal_naive",
    "last_value",
]


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in mebibytes."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kibibytes.
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def synthetic_panel():
    """A panel with HZMetro's shape, so the measurement matches the reported runs."""
    from mflow.schema import Panel

    rng = np.random.default_rng(0)
    phase = np.arange(CONTEXT) % 96
    profile = 40.0 + 30.0 * np.sin(2 * np.pi * phase / 96)
    series = np.maximum(
        profile[None, :] + rng.normal(0.0, 5.0, size=(N_SERIES, CONTEXT)), 0.0
    ).astype(np.float32)
    timestamps = pd.date_range("2019-01-07", periods=CONTEXT, freq="15min", tz="UTC")
    return Panel(
        series=series,
        series_ids=[
            f"flow:s{i // 2}:{'in' if i % 2 else 'out'}" for i in range(N_SERIES)
        ],
        timestamps=timestamps,
        past_covariates=None,
        past_covariate_ids=[],
        future_covariates=None,
        future_covariate_ids=[],
        interval_seconds=900,
        site_id="synthetic",
    )


def measure_one(name: str) -> dict[str, float]:
    """Load one model, run one forward pass, and report the process peak."""
    from mflow.forecast import build_forecaster

    panel = synthetic_panel()
    baseline = peak_rss_mb()
    model = build_forecaster(name)
    if getattr(model, "requires_training", False):
        model.fit(panel)
    after_load = peak_rss_mb()
    model.predict(panel, HORIZON, QUANTILES)
    return {
        "baseline_rss_mb": round(baseline, 1),
        "peak_after_load_mb": round(after_load, 1),
        "peak_total_mb": round(peak_rss_mb(), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="measure one model and print JSON (internal)")
    arguments = parser.parse_args()

    if arguments.model:
        print(json.dumps(measure_one(arguments.model)))
        return 0

    rows = []
    for name in MODELS:
        finished = subprocess.run(
            [sys.executable, __file__, "--model", name],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        line = finished.stdout.strip().splitlines()
        if finished.returncode != 0 or not line:
            print(f"  {name:24s} could not be measured")
            continue
        record = json.loads(line[-1])
        rows.append(
            {
                "method": name,
                "peak_rss_mb": record["peak_total_mb"],
                "weights_mb": round(
                    record["peak_after_load_mb"] - record["baseline_rss_mb"], 1
                ),
                "forward_pass_mb": round(
                    record["peak_total_mb"] - record["peak_after_load_mb"], 1
                ),
            }
        )
        print(
            f"  {name:24s} peak {record['peak_total_mb']:8.1f} MiB   "
            f"(weights {rows[-1]['weights_mb']:7.1f}, "
            f"forward pass {rows[-1]['forward_pass_mb']:7.1f})"
        )

    if not rows:
        raise RuntimeError("nothing could be measured")
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "memory.csv", index=False)
    print(f"\nwrote {TABLES / 'memory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
