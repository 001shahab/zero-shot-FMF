"""Check the second-round manuscript against the logged second-round analysis.

``analysis_round2.py`` writes ``summary_round2.json``. This script rebuilds, from that
file alone, every table row and every figure that the second revision quotes, formats
each exactly as the manuscript prints it, and fails if the manuscript does not contain
it. A number in the text that no logged value produces therefore cannot survive, and a
rerun that moves a value cannot leave the old one in the text unnoticed.

The check is one-directional on purpose: it proves that what is listed here is in the
manuscript and agrees with the artefacts. It does not prove that the manuscript quotes
nothing else, which is what reading it is for.

Usage:
    python verify_round2.py [path/to/MainText.tex]
"""

# Each expected string below is a phrase of the manuscript, kept on one line so that it
# can be searched for in the source as it stands. Wrapping them would make the check
# harder to read against the text it checks.
# ruff: noqa: E501

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "summary_round2.json"
FIRST_ROUND = HERE / "summary.json"
DEFAULT_TEX = HERE.parent / "Zero_Shot_FMF_v3" / "MainText.tex"

TFM = "timesfm3_multivariate"
LGB = "global_lightgbm"
FORECASTER_NAME = {TFM: "TimesFM-3", LGB: "LightGBM"}
BLOCKS: tuple[int, ...] = (1, 5, 10, 20, 40)

MATCHED_ORDER: tuple[tuple[str, str], ...] = (
    ("toto2", "Toto-2.0"),
    ("timesfm3_multivariate", "TimesFM-3 (mv)"),
    ("timesfm3_univariate", "TimesFM-3 (uv)"),
    ("chronos2_multivariate", "Chronos-2 (mv)"),
    ("global_lightgbm", "LightGBM (trained)"),
    ("seasonal_naive", "Seasonal naive"),
    ("historical_average", "Historical average"),
    ("last_value", "Last value"),
)


class VerificationError(RuntimeError):
    """The manuscript and the logged analysis disagree."""


def squash(text: str) -> str:
    """Collapse runs of whitespace so that column padding does not matter."""
    return re.sub(r"\s+", " ", text)


def fixed(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def scientific(value: float) -> str:
    """``1.4e-12`` in the form the manuscript passes to ``\\num``."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def signed(value: float, digits: int) -> str:
    """A signed number, with an exact zero printed without a sign."""
    if abs(value) < 0.5 * 10 ** (-digits):
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}"


def interval(bounds: list[float], digits: int) -> str:
    low, high = bounds
    return f"$[{signed(low, digits)},{signed(high, digits)}]$"


def matched_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, target in (("robod", "0.8"), ("robod", "0.9"), ("hzmetro", "0.8"), ("hzmetro", "0.9")):
        width = 2 if key == "robod" else 1
        table = summary["matched_coverage"][key][target]
        for method, name in MATCHED_ORDER:
            row = table[method]
            cells = [
                fixed(row["coverage"], 3),
                fixed(row["width"], width),
                fixed(row["factor_in_sample"], 2),
                fixed(row["coverage_in_sample"], 3),
                fixed(row["width_in_sample"], width),
                fixed(row["factor_calibration"], 2),
                fixed(row["evaluation_coverage"], 3),
                fixed(row["evaluation_width"], width),
                fixed(row["expanding_coverage"], 3),
                fixed(row["expanding_width"], width),
            ]
            out.append(
                (
                    f"Table matched, {key} at {target}, {name}",
                    f"{name} & " + " & ".join(cells) + r" \\",
                )
            )
    return out


def typical_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    inference = summary["reconciliation_inference"]
    layout = (
        ("baseline", "proposed_vs_raw", "Proposed vs raw"),
        ("baseline", "proposed_vs_uniform", "Proposed vs uniform"),
        ("baseline", "proposed_vs_unbounded", "Proposed vs unbounded"),
        ("baseline", "unbounded_vs_mint", "Unbounded vs MinT"),
        ("crowded", "proposed_vs_raw", "Proposed vs raw"),
    )
    out: list[tuple[str, str]] = []
    for label, comparison, name in layout:
        for forecaster in (TFM, LGB):
            row = inference[label][comparison][forecaster]
            at_ten = row["by_block_length"]["10"]
            cells = [
                name,
                FORECASTER_NAME[forecaster],
                f"${signed(row['mean_change_percent'], 2)}$",
                interval(at_ten["mean_ci_percent"], 2),
                f"${signed(row['median_change_percent'], 3)}$",
                interval(at_ten["median_ci_percent"], 3),
                f"{row['n_improved']} / {row['n_worsened']} / {row['n_tied']}",
                rf"\num{{{scientific(row['sign_test_p_value'])}}}",
                rf"\num{{{scientific(row['signed_rank_p_value'])}}}",
            ]
            out.append(
                (f"Table typical, {label}, {name}, {forecaster}", " & ".join(cells) + r" \\")
            )
    return out


def block_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    inference = summary["reconciliation_inference"]
    out: list[tuple[str, str]] = []
    for label, title in (("baseline", "Baseline"), ("crowded", "Crowded")):
        for forecaster, quantity, key, digits in (
            (TFM, "TimesFM-3, mean", "mean_ci_percent", 2),
            (TFM, "TimesFM-3, median", "median_ci_percent", 3),
            (LGB, "LightGBM, mean", "mean_ci_percent", 2),
        ):
            by_block = inference[label]["proposed_vs_raw"][forecaster]["by_block_length"]
            cells = [interval(by_block[str(block)][key], digits) for block in BLOCKS]
            out.append(
                (
                    f"Table block, {label}, {quantity}",
                    f"{title} & {quantity} & " + " & ".join(cells) + r" \\",
                )
            )
    return out


def risk_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    risk = summary["risk"]
    arms = {label: risk[label]["arms"] for label in ("baseline", "crowded")}

    def level(name: str, getter: Any, formatter: Any) -> tuple[str, str]:
        cells = [
            formatter(getter(arms[label][arm]))
            for label in ("baseline", "crowded")
            for arm in ("none", "proposed")
        ]
        return (f"Table risk, {name}", f"{name} & " + " & ".join(cells) + r" \\")

    def percent(value: float) -> str:
        return rf"\SI{{{100.0 * value:.1f}}}{{\percent}}"

    def count(value: float) -> str:
        return rf"\num{{{int(value)}}}"

    out = [
        level("Peak error, persons", lambda a: a["peak_error"], lambda v: f"{v:.3f}"),
        level(
            "High-occupancy MAE, mean over origins",
            lambda a: a["high_occupancy_mae"],
            lambda v: f"{v:.3f}",
        ),
        level(
            "High-occupancy MAE, pooled over room-steps",
            lambda a: a["high_occupancy_mae_pooled"],
            lambda v: f"{v:.3f}",
        ),
    ]
    for rule in ("near_capacity", "at_capacity_rounded", "at_capacity"):
        for name, field, formatter in (
            (r"\quad Hit rate", "hit_rate", percent),
            (r"\quad False alarms", "false_alarms", count),
            (r"\quad Brier score", "brier", lambda v: f"{v:.4f}"),
        ):
            label, text = level(name, lambda a, r=rule, f=field: a[r][f], formatter)
            out.append((f"{label}, {rule}", text))

    changes = {label: risk[label]["arms"]["proposed"]["change_against_raw"] for label in arms}
    for metric, title, scale, digits in (
        ("peak_error", "Peak error", 1.0, 3),
        ("high_occupancy_mae", "High-occupancy MAE", 1.0, 3),
        ("near_capacity_brier", "Near-capacity Brier", 1000.0, 2),
        ("at_capacity_rounded_brier", "At-capacity Brier", 1000.0, 2),
    ):
        for forecaster in (LGB, TFM):
            cells = []
            for label in ("baseline", "crowded"):
                entry = changes[label][metric][forecaster]
                low, high = entry["ci"]
                cells.append(
                    rf"\multicolumn{{2}}{{c}}{{${signed(scale * entry['mean_change'], digits)}\ "
                    f"[{signed(scale * low, digits)},{signed(scale * high, digits)}]$}}"
                )
            name = f"{title}, {FORECASTER_NAME[forecaster]}"
            out.append((f"Table risk, change, {name}", f"{name} & " + " & ".join(cells) + r" \\"))
    return out


def quoted_values(summary: dict[str, Any], first: dict[str, Any]) -> list[tuple[str, str]]:
    """Numbers quoted in running text, each formatted as it is printed."""
    out: list[tuple[str, str]] = []

    def add(label: str, text: str) -> None:
        out.append((label, text))

    hz = summary["lightgbm"]["hzmetro"]
    ro = summary["lightgbm"]["robod"]
    add(
        "HZMetro MASE reduction",
        rf"\SI{{{hz['mase_reduction_winner_vs_loser_percent']:.1f}}}{{\percent}} in favour of the local model",
    )
    add(
        "HZMetro MAE reduction",
        rf"a reduction of \SI{{{hz['mae_reduction_winner_vs_loser_percent']:.1f}}}{{\percent}} under definition",
    )
    add(
        "HZMetro paired MAE",
        rf"\num{{{hz['local_mae']:.2f}}} passengers per interval for the local model and \num{{{hz['zero_shot_mae']:.2f}}}",
    )
    add("HZMetro origins", rf"over the \num{{{hz['n_origins']}}} origins")
    by_horizon = hz["local_reduction_by_horizon_percent_mase"]
    add(
        "HZMetro reduction by horizon",
        ", ".join(rf"\SI{{{by_horizon[h]:.1f}}}{{\percent}}" for h in ("15", "30", "60"))
        + rf" and \SI{{{by_horizon['120']:.1f}}}{{\percent}}",
    )
    add(
        "ROBOD reductions",
        rf"\SI{{{ro['mase_reduction_winner_vs_loser_percent']:.1f}}}{{\percent}} on MASE "
        rf"(\SI{{{ro['mae_reduction_winner_vs_loser_percent']:.1f}}}{{\percent}} on the paired per-origin MAE, "
        rf"\num{{{ro['zero_shot_mae']:.3f}}} against \num{{{ro['local_mae']:.3f}}} persons)",
    )

    robod80 = summary["matched_coverage"]["robod"]["0.8"]
    robod90 = summary["matched_coverage"]["robod"]["0.9"]
    hz80 = summary["matched_coverage"]["hzmetro"]["0.8"]
    hz90 = summary["matched_coverage"]["hzmetro"]["0.9"]
    toto, naive = robod90["toto2"], robod90["seasonal_naive"]
    add(
        "ROBOD published width ratio",
        rf"\num{{{robod80['seasonal_naive']['width'] / robod80['toto2']['width']:.1f}}} times as wide and covers less",
    )
    add(
        "ROBOD retrospective ratio",
        rf"\num{{{naive['width_in_sample'] / toto['width_in_sample']:.1f}}} times the foundation model's width",
    )
    add(
        "ROBOD out-of-sample ratio",
        rf"\num{{{naive['evaluation_width'] / toto['evaluation_width']:.1f}}} times as wide at coverages that differ by \num{{{abs(naive['evaluation_coverage'] - toto['evaluation_coverage']):.3f}}}",
    )
    for method, share in (
        ("toto2", "under Toto-2.0"),
        ("chronos2_multivariate", "under Chronos-2"),
        ("global_lightgbm", "under LightGBM"),
        ("seasonal_naive", "under seasonal naive"),
    ):
        add(
            f"endpoint mass, {method}",
            rf"\SI{{{100 * robod80[method]['fraction_on_endpoint']:.1f}}}{{\percent}} {share}",
        )
    add(
        "Toto coverage below one",
        rf"from \num{{{robod80['toto2']['coverage']:.3f}}} at $f=1$ to \num{{{robod80['toto2']['coverage_just_below_one']:.3f}}} immediately below it",
    )
    add(
        "seasonal coverage below one",
        rf"from \num{{{robod80['seasonal_naive']['coverage']:.3f}}} to \num{{{robod80['seasonal_naive']['coverage_just_below_one']:.3f}}}",
    )
    add(
        "last value exact share",
        rf"exactly right on \SI{{{100 * robod80['last_value']['fraction_exact']:.1f}}}{{\percent}} of observations",
    )
    add(
        "last value uncoverable",
        rf"\SI{{{100 * robod80['last_value']['fraction_uncoverable']:.1f}}}{{\percent}} of its outcomes lie on a side of zero width",
    )
    add(
        "historical average uncoverable",
        rf"Historical average has \SI{{{100 * robod80['historical_average']['fraction_uncoverable']:.1f}}}{{\percent}} of outcomes",
    )
    add(
        "HZMetro endpoint ceiling",
        rf"at most \SI{{{100 * max(v['fraction_on_endpoint'] for v in hz80.values()):.1f}}}{{\percent}} of outcomes lie on an endpoint",
    )

    def reduction(better: float, worse: float) -> float:
        return 100.0 * (1.0 - better / worse)

    tfm80, sn80 = hz80[TFM], hz80["seasonal_naive"]
    tfm90, sn90 = hz90[TFM], hz90["seasonal_naive"]
    add(
        "HZMetro retrospective 0.80",
        rf"a reduction of \SI{{{reduction(tfm80['width_in_sample'], sn80['width_in_sample']):.1f}}}{{\percent}}; at \num{{0.90}}",
    )
    add(
        "HZMetro retrospective 0.90",
        rf"against \num{{{sn90['width_in_sample']:.1f}}}, a reduction of \SI{{{reduction(tfm90['width_in_sample'], sn90['width_in_sample']):.1f}}}{{\percent}}",
    )
    add(
        "HZMetro split 0.80",
        rf"\num{{{sn80['evaluation_width']:.1f}}} against \num{{{tfm80['evaluation_width']:.1f}}} passengers, by \SI{{{reduction(sn80['evaluation_width'], tfm80['evaluation_width']):.1f}}}{{\percent}}",
    )
    add(
        "HZMetro out of sample 0.90",
        rf"narrower by \SI{{{reduction(tfm90['evaluation_width'], sn90['evaluation_width']):.1f}}}{{\percent}} and "
        rf"\SI{{{reduction(tfm90['expanding_width'], sn90['expanding_width']):.1f}}}{{\percent}} while covering "
        rf"\num{{{sn90['evaluation_coverage'] - tfm90['evaluation_coverage']:.3f}}} and "
        rf"\num{{{sn90['expanding_coverage'] - tfm90['expanding_coverage']:.3f}}} less",
    )

    regression = next(r for r in first["within_dataset_regression"] if "hzmetro" in r["experiment"])
    add("rank correlation p-value", rf"$p=\num{{{regression['spearman_p']:.3f}}}$ taken alone")
    add(
        "seasonal strength range",
        rf"a range of only \num{{{regression['strength_max'] - regression['strength_min']:.2f}}} on this panel",
    )

    quantile = summary["per_quantile"]
    add(
        "crossing rate",
        rf"\SI{{{100 * quantile['crossing_rate_before_sort']:.1f}}}{{\percent}} of adjacent quantile pairs cross",
    )
    add(
        "pairs with a crossing",
        rf"\SI{{{100 * quantile['fraction_of_pairs_with_a_crossing']:.0f}}}{{\percent}} of the origin and forecaster pairs",
    )
    add(
        "median agreement",
        rf"within \num{{{scientific(quantile['median_gap_to_shift_before_sort_max'])}}} persons",
    )
    add(
        "median gap after sort",
        rf"by up to \num{{{quantile['median_gap_to_shift_after_sort_max']:.1f}}} persons at a single coordinate",
    )
    add(
        "pairs measured",
        rf"the same \num{{{quantile['n_origin_forecaster_pairs']}}} origin and forecaster pairs",
    )
    levels = sorted(quantile["by_level"], key=lambda r: r["level"])
    add(
        "per-quantile residual row",
        r"\rev{Per quantile, rearranged} & "
        + " & ".join(f"{r['residual_after_sort']:.3f}" for r in levels)
        + r" \\",
    )
    add(
        "share moved row",
        r"\rev{Per quantile, rearranged} & "
        + " & ".join(rf"\rev{{{100 * r['fraction_moved_by_sort']:.1f}}}" for r in levels)
        + r" \\",
    )
    largest = max(r["max_residual_before_sort"] for r in levels)
    add("largest single residual", rf"is \num{{{scientific(largest)}}} persons")
    means = [r["residual_before_sort"] for r in levels]
    if max(means) >= 1e-9:
        raise VerificationError(
            "a per-quantile level has a mean residual of 1e-9 or more before sorting"
        )
    add(
        "mean residuals before sorting",
        r"the mean residual at each of the nine levels is below \num{e-9} persons",
    )
    if (
        max(r["violation_before_sort"] for r in levels) > 0
        or max(r["violation_after_sort"] for r in levels) > 0
    ):
        raise VerificationError("the per-quantile bound violation is not zero, as the table states")

    for label in ("baseline", "crowded"):
        sample = summary["risk"][label]["sample"]
        stability = summary["risk"][label]["solver_start_stability"]
        total = sample["n_room_steps_per_forecaster"]
        hot, full = (
            sample["n_high_occupancy_steps_per_forecaster"],
            sample["n_at_capacity_steps_per_forecaster"],
        )
        add(f"{label}: high-occupancy steps", rf"\num{{{hot}}}")
        add(f"{label}: high-occupancy share", rf"\SI{{{100 * hot / total:.1f}}}{{\percent}}")
        add(f"{label}: at-capacity steps", rf"\num{{{full}}}")
        add(f"{label}: at-capacity share", rf"\SI{{{100 * full / total:.1f}}}{{\percent}}")
        add(
            f"{label}: origins with high occupancy",
            rf"\num{{{sample['n_origins_with_high_occupancy']}}}",
        )
        add(f"{label}: origins at capacity", rf"\num{{{sample['n_origins_with_at_capacity']}}}")
        add(
            f"{label}: strict-rule hits",
            rf"from \num{{{stability['at_capacity']['hits_cold_start']}}} to \num{{{stability['at_capacity']['hits_warm_start']}}}",
        )
        if stability["max_abs_mae_difference"] >= 1e-6:
            raise VerificationError(f"{label}: warm and cold starts differ in MAE by 1e-6 or more")
        for rule in ("near_capacity", "at_capacity_rounded"):
            if stability[rule]["hits_cold_start"] != stability[rule]["hits_warm_start"]:
                raise VerificationError(
                    f"{label}: the {rule} rule is not stable to the solver start"
                )
        arms = summary["risk"][label]["arms"]
        events = {
            rule: arms["none"][rule]["events"] for rule in ("near_capacity", "at_capacity_rounded")
        }
        add(f"{label}: near-capacity events", rf"\num{{{events['near_capacity']}}}")
        add(f"{label}: at-capacity events", rf"\num{{{events['at_capacity_rounded']}}}")

    # The headline real-data interval at each block length, in the order of the sentence.
    headline = summary["headline_block_sensitivity"]
    for key, digits in (("robod", 3), ("hzmetro", 2)):
        by_block = headline[key]["by_block_length"]
        intervals = [interval(by_block[str(block)]["mean_ci"], digits) for block in BLOCKS]
        add(
            f"headline block sensitivity, {key}",
            ", ".join(intervals[:-1]) + " and " + intervals[-1],
        )
    robod_excludes = [
        entry["excludes_zero"] for entry in headline["robod"]["by_block_length"].values()
    ]
    if not all(robod_excludes):
        raise VerificationError("the ROBOD headline interval contains zero at some block length")
    for block in BLOCKS:
        excludes = headline["hzmetro"]["by_block_length"][str(block)]["excludes_zero"]
        if excludes != (block == 1):
            raise VerificationError(
                f"the HZMetro headline interval at block {block} does not behave as the text says"
            )
    add(
        "HZMetro autocorrelation at ten origins",
        rf"still \num{{{headline['hzmetro']['autocorrelation']['10']:.2f}}} at a lag of ten origins",
    )
    return out


def main() -> int:
    tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEX
    for path in (SUMMARY, FIRST_ROUND, tex_path):
        if not path.is_file():
            raise SystemExit(f"{path} is missing")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    first = json.loads(FIRST_ROUND.read_text(encoding="utf-8"))
    tex = squash(tex_path.read_text(encoding="utf-8"))

    checks = (
        matched_rows(summary)
        + typical_rows(summary)
        + block_rows(summary)
        + risk_rows(summary)
        + quoted_values(summary, first)
    )
    failures = 0
    for label, expected in checks:
        if squash(expected) not in tex:
            failures += 1
            print(f"MISSING  {label}\n         expected: {expected}")
    total = len(checks)
    if failures:
        print(f"\n{failures} of {total} second-round claims are not in {tex_path.name} as logged.")
        return 1
    print(f"all {total} second-round claims in {tex_path.name} match summary_round2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
