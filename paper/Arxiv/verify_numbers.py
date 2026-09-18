"""Check that every headline number in main.tex still matches the logged runs.

The paper quotes figures that were computed by ``analysis.py`` from ``results/``. This
script re-reads ``summary.json`` and asserts the manuscript agrees with it, so that a
rerun which shifts a number cannot silently leave a stale claim in the text.

Timings are wall clock and drift between machines, so they carry a looser tolerance than
the statistics, which must agree to the digits printed.

Usage:
    python paper/Arxiv/verify_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TFM = "timesfm3_multivariate"
LGB = "global_lightgbm"
ABLATION = "proposed_vs_uniform"
SUMMARY = HERE / "summary.json"


def main() -> int:
    if not SUMMARY.is_file():
        raise SystemExit(f"{SUMMARY} is missing; run analysis.py first")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    recon = summary["reconciliation"]
    significance = summary["reconciliation_significance"]
    gain = summary["zero_shot_gain_percent"]
    strength = summary["seasonal_strength"]
    regression = next(
        r for r in summary["within_dataset_regression"] if "hzmetro" in r["experiment"]
    )

    def change(arm: str, forecaster: str) -> float:
        """Mean percentage change in MAE for one reconciler and one forecaster."""
        return significance[arm][forecaster]["mean_change_percent"]

    # (label, value printed in the paper, value in the artefacts, relative tolerance)
    checks: list[tuple[str, float, float, float]] = [
        ("raw conservation residual", 1.049, recon["none"]["residual_before"], 0.006),
        ("MinT bound violation", 1.738, recon["mint"]["violation_after"], 0.006),
        ("MAE, raw", 0.3538, recon["none"]["mae_before"], 0.006),
        ("MAE, MinT", 0.3556, recon["mint"]["mae_after"], 0.006),
        ("MAE, uniform", 0.3548, recon["uniform_weight"]["mae_after"], 0.006),
        ("MAE, proposed", 0.3511, recon["proposed"]["mae_after"], 0.006),
        ("CRPS, raw", 0.2723, recon["none"]["crps_before"], 0.006),
        ("CRPS, proposed", 0.2729, recon["proposed"]["crps_after"], 0.006),
        # Per forecaster, because the two share every origin and pooling them would
        # treat correlated observations as independent.
        ("dMAE, MinT, TimesFM (%)", 0.37, change("mint", TFM), 0.03),
        ("dMAE, MinT, LightGBM (%)", 0.69, change("mint", LGB), 0.03),
        ("dMAE, uniform, TimesFM (%)", 0.22, change("uniform_weight", TFM), 0.05),
        ("dMAE, uniform, LightGBM (%)", 0.34, change("uniform_weight", LGB), 0.05),
        ("dMAE, proposed, TimesFM (%)", -0.98, change("proposed", TFM), 0.03),
        ("dMAE, proposed, LightGBM (%)", -0.52, change("proposed", LGB), 0.03),
        ("proposed vs uniform, TimesFM (%)", -1.20, change(ABLATION, TFM), 0.03),
        ("proposed vs uniform, LightGBM (%)", -0.86, change(ABLATION, LGB), 0.03),
        (
            "origins worsened, proposed+TimesFM",
            0.70,
            significance["proposed"][TFM]["fraction_worsened"],
            0.02,
        ),
        ("zero-shot gain, ROBOD (%)", 30.5, gain["ROBOD"], 0.01),
        ("zero-shot gain, HZMetro (%)", 16.2, gain["HZMetro"], 0.01),
        ("zero-shot gain, simulated (%)", -4.0, gain["Simulated"], 0.02),
        ("seasonal strength, ROBOD", 0.44, strength["ROBOD"][0], 0.01),
        ("seasonal strength, HZMetro", 0.79, strength["HZMetro"][0], 0.01),
        ("seasonal strength, simulated", 0.55, strength["Simulated"][0], 0.01),
        # The within-dataset regression, which is the test that refuted the hypothesis.
        ("H2 slope, HZMetro", -17.0, regression["slope"], 0.05),
        ("H2 slope CI low", -38.0, regression["slope_ci_low"], 0.05),
        ("H2 slope CI high", 4.0, regression["slope_ci_high"], 0.10),
        ("H2 p-value", 0.12, regression["p_value"], 0.01),
        ("H2 R squared", 0.015, regression["r_squared"], 0.002),
        ("H2 series count", 160, regression["n_series"], 0.0),
        # Wall clock, so a wider band. A drift beyond this is worth looking at.
        ("solve time, proposed (ms)", 10.1, recon["proposed"]["solve_time_ms"], 0.20),
        ("solve time, uniform (ms)", 6.2, recon["uniform_weight"]["solve_time_ms"], 0.20),
    ]

    failures = 0
    for label, printed, actual, tolerance in checks:
        ok = abs(printed - actual) <= tolerance * max(1.0, abs(actual))
        if not ok:
            failures += 1
        print(
            f"{'ok  ' if ok else 'STALE'}  {label:32s} paper={printed:<10g} "
            f"logged={actual:.6g}"
        )

    if failures:
        print(f"\n{failures} number(s) in main.tex no longer match results/.")
        return 1
    print(f"\nall {len(checks)} headline numbers match the logged runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
