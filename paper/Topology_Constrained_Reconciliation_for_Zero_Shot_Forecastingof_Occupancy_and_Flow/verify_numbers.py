"""Check that every headline number in main.tex still matches the logged runs.

The paper quotes figures that were computed by ``analysis.py`` from ``results/``. This
script re-reads ``summary.json`` and asserts the manuscript agrees with it, so that a
rerun which shifts a number cannot silently leave a stale claim in the text.

Timings are wall clock and drift between machines, so they carry a looser tolerance than
the statistics, which must agree to the digits printed.

The check list is keyed to the revised manuscript. Several entries exist because the
submitted version quoted a number that a rerun moved, and the revision corrected it; the
point of listing them is that the next rerun cannot reintroduce the stale value quietly.

Usage:
    python verify_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "summary.json"
TFM = "timesfm3_multivariate"
LGB = "global_lightgbm"


def percent_gain(better: float, worse: float) -> float:
    """How much better ``better`` is than ``worse``, as a percentage of ``worse``."""
    return 100.0 * (1.0 - better / worse)


def main() -> int:
    if not SUMMARY.is_file():
        raise SystemExit(f"{SUMMARY} is missing; run analysis.py first")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    recon = summary["reconciliation"]
    base = recon["baseline"]
    crowded = recon["crowded"]
    significance = summary["reconciliation_significance"]["baseline"]
    gain = summary["zero_shot_gain_percent"]
    strength = summary["seasonal_strength"]
    regression = next(
        r for r in summary["within_dataset_regression"] if "hzmetro" in r["experiment"]
    )
    lgbm = summary["lightgbm"]
    calibration = summary["calibration"]
    context = summary["context_sweep"]
    sigma = summary["sigma_sensitivity"]
    dependence = summary["dependence"]
    coherence = {
        (row["strategy"], f"{row['level']:.1f}"): row for row in summary["quantile_coherence"]
    }

    def change(arm: str, forecaster: str) -> float:
        """Mean percentage change in MAE for one reconciler and one forecaster."""
        return significance[arm][forecaster]["mean_change_percent"]

    def mase(site: str, method: str, horizon: str) -> float:
        return lgbm[site]["mase"][method][horizon]

    # (label, value printed in the paper, value in the artefacts, absolute tolerance)
    checks: list[tuple[str, float, float, float]] = [
        # --- Reconciliation, baseline operating point ------------------------ #
        ("raw conservation residual", 1.05, base["none"]["residual_before"], 0.006),
        ("MinT bound violation", 2.06, base["mint"]["violation_after"], 0.006),
        ("MinT bound violation, crowded", 2.94, crowded["mint"]["violation_after"], 0.006),
        ("raw bound violation", 0.006, base["none"]["violation_before"], 0.001),
        ("raw bound violation, crowded", 0.451, crowded["none"]["violation_before"], 0.002),
        ("proposed bound violation", 0.0, base["proposed"]["violation_after"], 1e-9),
        ("CRPS, raw", 0.2687, base["none"]["crps_before"], 0.0006),
        ("CRPS, proposed", 0.2697, base["proposed"]["crps_after"], 0.0006),
        ("peak error, raw", 1.298, base["none"]["peak_error_before"], 0.002),
        ("peak error, proposed", 1.217, base["proposed"]["peak_error_after"], 0.002),
        ("peak error, crowded raw", 1.113, crowded["none"]["peak_error_before"], 0.002),
        ("peak error, crowded proposed", 1.068, crowded["proposed"]["peak_error_after"], 0.002),
        (
            "peak error reduction (%)",
            6.3,
            percent_gain(base["proposed"]["peak_error_after"], base["none"]["peak_error_before"]),
            0.06,
        ),
        (
            "MAE reduction (%)",
            0.56,
            percent_gain(base["proposed"]["mae_after"], base["none"]["mae_before"]),
            0.006,
        ),
        (
            "MAE reduction, crowded (%)",
            0.57,
            percent_gain(crowded["proposed"]["mae_after"], crowded["none"]["mae_before"]),
            0.006,
        ),
        # --- The three ablations, per forecaster ----------------------------- #
        ("proposed vs uniform, TimesFM (%)", -1.20, change("proposed_vs_uniform", TFM), 0.02),
        ("proposed vs uniform, LightGBM (%)", -0.56, change("proposed_vs_uniform", LGB), 0.02),
        ("proposed vs unbounded, TimesFM (%)", -0.28, change("proposed_vs_unbounded", TFM), 0.02),
        ("proposed vs unbounded, LightGBM (%)", -0.30, change("proposed_vs_unbounded", LGB), 0.02),
        ("unbounded vs MinT, TimesFM (%)", -1.07, change("unbounded_vs_mint", TFM), 0.02),
        ("unbounded vs MinT, LightGBM (%)", -0.69, change("unbounded_vs_mint", LGB), 0.02),
        ("proposed vs raw, TimesFM (%)", -0.98, change("proposed", TFM), 0.02),
        ("proposed vs raw, LightGBM (%)", -0.10, change("proposed", LGB), 0.02),
        ("proposed vs raw p-value, TimesFM", 0.18, significance["proposed"][TFM]["p_value"], 0.01),
        (
            "origins worsened, proposed+TimesFM",
            0.70,
            significance["proposed"][TFM]["fraction_worsened"],
            0.02,
        ),
        # --- Solve time, wall clock, so a wider band ------------------------- #
        ("solve time, proposed (ms)", 6.2, base["proposed"]["solve_time_ms"], 1.5),
        ("solve time, proposed crowded (ms)", 7.2, crowded["proposed"]["solve_time_ms"], 1.5),
        # --- H1 and the gradient-boosted ladder ------------------------------ #
        ("zero-shot gain, ROBOD (%)", 30.5, gain["ROBOD"], 0.06),
        ("zero-shot gain, HZMetro (%)", 16.2, gain["HZMetro"], 0.06),
        ("zero-shot gain, simulated (%)", -5.3, gain["Simulated"], 0.06),
        (
            "LightGBM per series, HZMetro 120 min",
            0.511,
            mase("hzmetro", "lightgbm_per_series", "120"),
            0.0006,
        ),
        (
            "LightGBM per series, ROBOD 60 min",
            0.512,
            mase("robod", "lightgbm_per_series", "60"),
            0.0006,
        ),
        (
            "LightGBM as submitted, ROBOD 60 min",
            0.601,
            mase("robod", "lightgbm_as_submitted", "60"),
            0.0006,
        ),
        ("Toto-2.0, ROBOD 60 min", 0.374, mase("robod", "toto2", "60"), 0.0006),
        (
            "re-specification gain, ROBOD (%)",
            14.8,
            percent_gain(
                mase("robod", "lightgbm_per_series", "60"),
                mase("robod", "lightgbm_as_submitted", "60"),
            ),
            0.06,
        ),
        (
            "zero-shot advantage over strong local, ROBOD (%)",
            26.8,
            percent_gain(mase("robod", "toto2", "60"), mase("robod", "lightgbm_per_series", "60")),
            0.06,
        ),
        # --- H2, the refuted predictability hypothesis ----------------------- #
        ("seasonal strength, ROBOD", 0.44, strength["ROBOD"][0], 0.01),
        ("seasonal strength, HZMetro", 0.79, strength["HZMetro"][0], 0.01),
        ("seasonal strength, simulated", 0.55, strength["Simulated"][0], 0.01),
        ("H2 slope, HZMetro", -17.0, regression["slope"], 0.6),
        ("H2 slope CI low", -38.0, regression["slope_ci_low"], 0.6),
        ("H2 slope CI high", 4.0, regression["slope_ci_high"], 0.6),
        ("H2 R squared", 0.015, regression["r_squared"], 0.002),
        ("H2 series count", 160, regression["n_series"], 0.0),
        # --- H3, calibration separated from sharpness ------------------------ #
        ("ROBOD coverage, Toto-2.0", 0.901, calibration["robod"]["toto2"]["coverage"], 0.002),
        (
            "ROBOD coverage, seasonal naive",
            0.839,
            calibration["robod"]["seasonal_naive"]["coverage"],
            0.002,
        ),
        (
            "matched width, Toto-2.0",
            0.77,
            calibration["robod"]["toto2"]["width_at_matched_coverage"],
            0.01,
        ),
        (
            "matched width, seasonal naive",
            3.75,
            calibration["robod"]["seasonal_naive"]["width_at_matched_coverage"],
            0.01,
        ),
        (
            "sharpness ratio at matched coverage",
            4.9,
            calibration["robod"]["seasonal_naive"]["width_at_matched_coverage"]
            / calibration["robod"]["toto2"]["width_at_matched_coverage"],
            0.06,
        ),
        # --- The context sweep ------------------------------------------------ #
        (
            "HZMetro context 1 season",
            1.037,
            context["hzmetro"]["mase"]["timesfm3_univariate"]["1"],
            0.0006,
        ),
        (
            "HZMetro context 14 seasons",
            0.414,
            context["hzmetro"]["mase"]["timesfm3_univariate"]["14"],
            0.0006,
        ),
        ("ROBOD context 1 season", 0.454, context["robod"]["mase"]["toto2"]["1"], 0.0006),
        ("ROBOD context 14 seasons", 0.309, context["robod"]["mase"]["toto2"]["14"], 0.0006),
        # --- Quantile coherence under the two strategies ---------------------- #
        ("shift residual at q95", 0.085, coherence[("shift", "0.9")]["residual"], 0.001),
        (
            "per-quantile bound violation at q05",
            0.0,
            coherence[("per_quantile", "0.1")]["violation"],
            1e-9,
        ),
        ("shift bound violation at q05", 9.832, coherence[("shift", "0.1")]["violation"], 0.002),
        # --- Invariance to the scale of the predictive spread ----------------- #
        ("gaussian_80 MASE", 0.31760, sigma["gaussian_80"]["mae_after"], 1e-5),
        ("raw_width_80 MASE", 0.31760, sigma["raw_width_80"]["mae_after"], 1e-5),
        ("gaussian_40 MASE", 0.31898, sigma["gaussian_40"]["mae_after"], 1e-5),
        # --- Dependence between overlapping forecast origins ------------------ #
        (
            "HZMetro Newey-West lag published",
            7,
            dependence["hzmetro"]["newey_west_lag_published"],
            0.0,
        ),
        (
            "HZMetro Newey-West lag implied",
            3,
            dependence["hzmetro"]["newey_west_lag_design_implied"],
            0.0,
        ),
        (
            "ROBOD Newey-West lag published",
            11,
            dependence["robod"]["newey_west_lag_published"],
            0.0,
        ),
        (
            "ROBOD Newey-West lag implied",
            0,
            dependence["robod"]["newey_west_lag_design_implied"],
            0.0,
        ),
    ]

    booleans: list[tuple[str, bool, bool]] = [
        (
            "HZMetro block bootstrap CI includes zero",
            True,
            not dependence["hzmetro"]["block_bootstrap_excludes_zero"],
        ),
        (
            "ROBOD block bootstrap CI excludes zero",
            True,
            dependence["robod"]["block_bootstrap_excludes_zero"],
        ),
        (
            "no quantile crossing under shift",
            True,
            summary["quantile_crossing_rate"]["shift"] == 0.0,
        ),
        (
            "no quantile crossing under per_quantile",
            True,
            summary["quantile_crossing_rate"]["per_quantile"] == 0.0,
        ),
    ]

    failures = 0
    for label, printed, actual, tolerance in checks:
        ok = abs(printed - actual) <= tolerance
        failures += not ok
        print(f"{'ok   ' if ok else 'STALE'}  {label:48s} paper={printed:<10g} logged={actual:.6g}")
    for label, printed_claim, actual_claim in booleans:
        ok = printed_claim == actual_claim
        failures += not ok
        print(
            f"{'ok   ' if ok else 'STALE'}  {label:48s} paper={printed_claim} logged={actual_claim}"
        )

    total = len(checks) + len(booleans)
    if failures:
        print(f"\n{failures} of {total} claims in main.tex no longer match results/.")
        return 1
    print(f"\nall {total} headline claims match the logged runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
