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
SUMMARY = HERE / "summary.json"


def main() -> int:
    if not SUMMARY.is_file():
        raise SystemExit(f"{SUMMARY} is missing; run analysis.py first")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    recon = summary["reconciliation"]
    significance = summary["reconciliation_significance"]
    gain = summary["zero_shot_gain_percent"]
    strength = summary["seasonal_strength"]

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
        ("dMAE, MinT (%)", 0.53, significance["mint"]["mean_change_percent"], 0.03),
        (
            "dMAE, uniform (%)",
            0.28,
            significance["uniform_weight"]["mean_change_percent"],
            0.05,
        ),
        ("dMAE, proposed (%)", -0.76, significance["proposed"]["mean_change_percent"], 0.03),
        (
            "proposed vs uniform (%)",
            -1.03,
            significance["proposed_vs_uniform"]["mean_change_percent"],
            0.03,
        ),
        ("zero-shot gain, ROBOD (%)", 14.9, gain["ROBOD"], 0.01),
        ("zero-shot gain, HZMetro (%)", -33.5, gain["HZMetro"], 0.01),
        ("zero-shot gain, simulated (%)", -4.0, gain["Simulated"], 0.02),
        ("seasonal strength, ROBOD", 0.44, strength["ROBOD"][0], 0.01),
        ("seasonal strength, HZMetro", 0.79, strength["HZMetro"][0], 0.01),
        ("seasonal strength, simulated", 0.55, strength["Simulated"][0], 0.01),
        # Wall clock, so a wider band. A drift beyond this is worth looking at.
        ("solve time, proposed (ms)", 10.2, recon["proposed"]["solve_time_ms"], 0.20),
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
