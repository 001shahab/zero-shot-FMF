"""Time one forecast per model at each candidate context length.

Sizing probe for the context sweep the reviewers asked for. Prints seconds per origin so
the sweep can be budgeted before it is launched rather than after.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mflow.forecast import build_forecaster  # noqa: E402
from mflow.schema import Panel, load_site  # noqa: E402

SITE = sys.argv[1] if len(sys.argv) > 1 else "hzmetro"
CONTEXTS = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [192, 672]
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MODELS = ["timesfm3_univariate", "timesfm3_multivariate", "chronos2_multivariate", "toto2"]


def slice_panel(panel: Panel, end: int, length: int) -> Panel:
    start = end - length
    past = panel.past_covariates
    future = panel.future_covariates
    return Panel(
        series=panel.series[:, start:end].astype(np.float32),
        series_ids=panel.series_ids,
        timestamps=panel.timestamps[start:end],
        past_covariates=None if past is None else past[:, start:end],
        past_covariate_ids=panel.past_covariate_ids,
        future_covariates=None if future is None else future[:, start:end],
        future_covariate_ids=panel.future_covariate_ids,
        interval_seconds=panel.interval_seconds,
        site_id=panel.site_id,
    )


def main() -> int:
    site = load_site(ROOT / "data" / "canonical" / SITE)
    panel = site.to_panel()
    end = panel.n_timesteps - HORIZON - 1
    print(f"{SITE}: {panel.n_series} series, horizon {HORIZON}")
    print(f"{'model':28s}" + "".join(f"{c:>10d}" for c in CONTEXTS))
    for name in MODELS:
        model = build_forecaster(name, seed=0)
        row = []
        for context in CONTEXTS:
            window = slice_panel(panel, end, context)
            model.predict(window, HORIZON)  # warm the graph
            clock = time.perf_counter()
            model.predict(window, HORIZON)
            row.append(time.perf_counter() - clock)
        print(f"{name:28s}" + "".join(f"{v:>10.2f}" for v in row))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
