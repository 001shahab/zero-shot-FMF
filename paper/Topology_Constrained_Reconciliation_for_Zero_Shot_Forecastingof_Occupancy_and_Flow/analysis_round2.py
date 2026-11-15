"""Second-round re-analysis, answering the seven points of the final referee report.

Nothing here runs a forecaster. Every number is recomputed from the predictions that the
logged runs already wrote, or from the per-origin reconciliation table that
``analysis.py`` wrote, and the script raises instead of substituting a value when an
input is missing. Outputs go to ``summary_round2.json`` and to ``tables/round2_*.csv``.

The five analyses, in the order of the referee's comments:

1. Interval width at matched empirical coverage, redone so that the scale factor and the
   coverage are computed on the same observations by the same rule, and repeated with the
   factor estimated on a calibration period that precedes the evaluation period.
2. The gradient-boosted comparison, restated with one definition of relative improvement.
3. Inference on the reconciliation effect: the mean and the typical origin separately,
   with the sensitivity of both intervals to the bootstrap block length, and the same
   sensitivity for the headline real-data test, where serial dependence is strong.
4. Conservation residuals under per-quantile projection, measured before and after the
   monotone rearrangement that repairs crossings.
5. Risk-sensitive errors with their thresholds and sample sizes.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as base  # noqa: E402

TABLES = HERE / "tables"
BLOCK_LENGTHS: tuple[int, ...] = (1, 5, 10, 20, 40)
DRAWS = 4000
ENDPOINT_TOLERANCE = 1e-6


class RoundTwoError(RuntimeError):
    """An input the re-analysis needs is absent or inconsistent."""


# --------------------------------------------------------------------------- #
# Shared: a moving-block bootstrap for an arbitrary statistic
# --------------------------------------------------------------------------- #


def block_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    block: int,
    draws: int = DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """A 95 percent moving-block bootstrap interval for ``statistic``.

    ``block=1`` is the ordinary independent bootstrap and is included only as the
    reference against which the effect of dependence can be read.
    """
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 2 * block:
        raise RoundTwoError(f"{series.size} observations cannot support blocks of {block} origins")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(series.size / block)
    starts = rng.integers(0, series.size - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    out = np.empty(draws, dtype=float)
    for draw in range(draws):
        index = (starts[draw][:, None] + offsets[None, :]).reshape(-1)[: series.size]
        out[draw] = statistic(series[index])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# --------------------------------------------------------------------------- #
# 1. Width at matched coverage
# --------------------------------------------------------------------------- #


def _scores(frame: pd.DataFrame) -> pd.DataFrame:
    """The factor each observation needs for its interval to cover the outcome.

    The interval is scaled about the median, each side separately, so a factor of one is
    the published interval exactly and coverage is non-decreasing in the factor. The
    score of an observation is the smallest factor at which it is covered: zero when the
    outcome equals the median, and infinite when the outcome lies on a side whose
    half-width is zero, because no finite multiple of zero reaches it.
    """
    q10 = frame["q10"].to_numpy(dtype=float)
    q50 = frame["q50"].to_numpy(dtype=float)
    q90 = frame["q90"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    lower = np.maximum(q50 - q10, 0.0)
    upper = np.maximum(q90 - q50, 0.0)
    below = np.maximum(q50 - actual, 0.0)
    above = np.maximum(actual - q50, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score_below = np.where(below > 0.0, below / lower, 0.0)
        score_above = np.where(above > 0.0, above / upper, 0.0)
    score = np.maximum(score_below, score_above)
    # An outcome that sits on an interval endpoint has a score of exactly one in exact
    # arithmetic, and the quantiles of a trained model reach it only to within rounding,
    # which leaves scores such as 0.9999998. Left alone, those split one atom of the
    # coverage curve into two and make a factor that prints as 1.00 cover either 80 or
    # 94 percent. They are snapped upwards only, so that an observation the published
    # interval misses is never moved inside it.
    score = np.where((score >= 1.0 - ENDPOINT_TOLERANCE) & (score < 1.0), 1.0, score)
    return pd.DataFrame(
        {
            "origin": frame["origin"].to_numpy(),
            "score": score,
            "width": q90 - q10,
            "covered": (actual >= q10) & (actual <= q90),
        }
    )


def _factor_for(scores: np.ndarray, target: float, *, conformal: bool) -> float:
    """Smallest factor whose coverage on ``scores`` is at least ``target``."""
    ordered = np.sort(scores)
    n = ordered.size
    rank = math.ceil(target * (n + 1)) if conformal else math.ceil(target * n)
    if rank > n:
        return float("inf")
    return float(ordered[rank - 1])


def _expanding_window(
    table: pd.DataFrame, origins: np.ndarray, gap: int, target: float
) -> tuple[float, float, int]:
    """Coverage and width when every origin is calibrated on strictly earlier origins.

    The single split above uses each observation once and so depends on where the split
    falls. This is the complementary check: after a burn-in of a quarter of the origins,
    the factor applied at an origin is estimated from all origins at least ``gap``
    positions before it, so no forecast window used for calibration overlaps the one
    being scored. Origins at which no finite factor exists are counted as uncovered at
    infinite width and make the result undefined, which is reported as such.
    """
    burn_in = max(origins.size // 4, 1)
    by_origin = dict(tuple(table.groupby("origin")))
    covered = 0
    total = 0
    width_sum = 0.0
    used = 0
    history: list[np.ndarray] = []
    for position, origin in enumerate(origins):
        if position >= burn_in + gap:
            available = np.concatenate(history[: position - gap + 1])
            factor = _factor_for(available, target, conformal=True)
            block = by_origin[origin]
            if not np.isfinite(factor):
                return float("nan"), float("nan"), used
            covered += int((block["score"].to_numpy() <= factor + 1e-12).sum())
            width_sum += float(factor * block["width"].sum())
            total += len(block)
            used += 1
        history.append(by_origin[origin]["score"].to_numpy())
    if total == 0:
        return float("nan"), float("nan"), used
    return covered / total, width_sum / total, used


def matched_coverage(key: str, horizon: int, target: float) -> pd.DataFrame:
    """Width needed for a common empirical coverage, in sample and out of sample."""
    directories = [
        d for d in base.runs_for(base.REAL[key]) if (d / "predictions.parquet").is_file()
    ]
    if not directories:
        raise RoundTwoError(f"no predictions for {base.REAL[key]}")
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    frame = frame[frame["step"] <= horizon]
    prefix = "occ:" if base.CONTEXT_GROUP[key] == "occupancy" else "flow:"
    frame = frame[frame["series_id"].str.startswith(prefix)]
    finite = np.isfinite(frame["q10"]) & np.isfinite(frame["q90"]) & np.isfinite(frame["actual"])
    frame = frame[finite]

    rows: list[dict[str, Any]] = []
    for method, block in frame.groupby("method"):
        table = _scores(block)
        n = len(table)
        raw_coverage = float(table["covered"].mean())
        # The published interval is the factor-one interval, so the two ways of counting
        # coverage have to agree. If they do not, the score is not what it claims to be.
        at_one = float((table["score"] <= 1.0 + 1e-9).mean())
        if abs(at_one - raw_coverage) > 1e-9:
            raise RoundTwoError(
                f"{method}: coverage at factor one is {at_one}, raw coverage is "
                f"{raw_coverage}; the score does not reproduce the published interval"
            )

        factor = _factor_for(table["score"].to_numpy(), target, conformal=False)
        mean_width = float(table["width"].mean())
        achieved = (
            float((table["score"] <= factor + 1e-12).mean())
            if np.isfinite(factor)
            else float("nan")
        )

        # Out of sample: the factor comes from the earlier origins and is applied,
        # unchanged, to the later ones. Origins whose forecast window could overlap the
        # boundary are dropped from the evaluation half.
        origins = np.sort(table["origin"].unique())
        stride = int(np.min(np.diff(origins))) if origins.size > 1 else 1
        gap = math.ceil(horizon / stride)
        half = origins.size // 2
        calibration_origins = set(origins[:half].tolist())
        evaluation_origins = set(origins[half + gap :].tolist())
        calibration = table[table["origin"].isin(calibration_origins)]
        evaluation = table[table["origin"].isin(evaluation_origins)]
        factor_cal = _factor_for(calibration["score"].to_numpy(), target, conformal=True)
        if np.isfinite(factor_cal):
            eval_coverage = float((evaluation["score"] <= factor_cal + 1e-12).mean())
            eval_width = float(factor_cal * evaluation["width"].mean())
        else:
            eval_coverage = float("nan")
            eval_width = float("nan")

        expanding_coverage, expanding_width, expanding_origins = _expanding_window(
            table, origins, gap, target
        )
        score = table["score"].to_numpy()
        rows.append(
            {
                "method": base.base_method(str(method)),
                "n_observations": n,
                "n_origins": int(origins.size),
                "stride_steps": stride,
                # How much of the published coverage sits exactly on an interval
                # endpoint. This mass is covered at factor one and lost at any factor
                # below it, so it is the size of the step in the coverage curve at one.
                "coverage_just_below_one": float((score < 1.0 - 1e-9).mean()),
                "fraction_on_endpoint": float((np.abs(score - 1.0) <= 1e-9).mean()),
                "expanding_coverage": expanding_coverage,
                "expanding_width": expanding_width,
                "expanding_origins": expanding_origins,
                "coverage": raw_coverage,
                "width": mean_width,
                "fraction_degenerate": float((table["width"] <= 1e-6).mean()),
                "fraction_uncoverable": float(np.isinf(table["score"]).mean()),
                "fraction_exact": float((table["score"] == 0.0).mean()),
                "factor_in_sample": factor,
                "coverage_in_sample": achieved,
                "width_in_sample": factor * mean_width if np.isfinite(factor) else float("nan"),
                "n_calibration_origins": len(calibration_origins),
                "n_evaluation_origins": len(evaluation_origins),
                "factor_calibration": factor_cal,
                "evaluation_raw_coverage": float(evaluation["covered"].mean()),
                "evaluation_raw_width": float(evaluation["width"].mean()),
                "evaluation_coverage": eval_coverage,
                "evaluation_width": eval_width,
            }
        )
    out = pd.DataFrame(rows).set_index("method")
    return out.sort_values("width_in_sample")


# --------------------------------------------------------------------------- #
# 2. One definition of relative improvement for the gradient-boosted comparison
# --------------------------------------------------------------------------- #


def lightgbm_restated(key: str) -> dict[str, Any]:
    """The strong local arm against the best zero-shot arm, on one stated definition.

    The relative improvement of a winner over a loser is ``100 (1 - winner / loser)``,
    so the loser is always the reference. It is given on MASE, which is what the tables
    print, and on the paired per-origin MAE that the Diebold-Mariano test runs on.
    """
    ladder = base.table_lightgbm(key)
    tests = base.lightgbm_significance(key)
    best = tests["best_zero_shot"]
    stat = tests["comparisons"]["lightgbm_per_series"]
    longest = ladder.columns[-1]
    column = ladder[longest].astype(float)
    zero_mase = float(column[best])
    local_mase = float(column["lightgbm_per_series"])
    zero_mae = float(stat["zero_shot_mae"])
    local_mae = float(stat["local_mae"])

    def reduction(winner: float, loser: float) -> float:
        return 100.0 * (1.0 - winner / loser)

    local_wins = local_mase < zero_mase

    def signed_local_reduction(minutes: Any) -> float:
        """Positive when the local arm wins at this horizon, negative when it loses."""
        local = float(ladder.loc["lightgbm_per_series", minutes])
        zero = float(ladder.loc[best, minutes])
        return reduction(local, zero) if local < zero else -reduction(zero, local)

    by_horizon = {str(minutes): signed_local_reduction(minutes) for minutes in ladder.columns}
    return {
        "best_zero_shot": best,
        "horizon_minutes": int(longest),
        "zero_shot_mase": zero_mase,
        "local_mase": local_mase,
        "zero_shot_mae": zero_mae,
        "local_mae": local_mae,
        "winner": "local" if local_wins else "zero_shot",
        "mase_reduction_winner_vs_loser_percent": reduction(
            min(zero_mase, local_mase), max(zero_mase, local_mase)
        ),
        "mae_reduction_winner_vs_loser_percent": reduction(
            min(zero_mae, local_mae), max(zero_mae, local_mae)
        ),
        # What the previous revision printed for HZMetro: the loser's excess over the
        # winner, on MAE. Kept so that the old figure can be reconciled with the new.
        "mae_excess_loser_over_winner_percent": 100.0
        * (max(zero_mae, local_mae) / min(zero_mae, local_mae) - 1.0),
        "local_reduction_by_horizon_percent_mase": by_horizon,
        "p_value": float(stat["p_value"]),
        "holm_significant": bool(stat["holm_significant"]),
        "n_origins": int(stat["n_origins"]),
    }


# --------------------------------------------------------------------------- #
# 3. Mean effect and typical-origin effect of reconciliation
# --------------------------------------------------------------------------- #

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("proposed_vs_raw", "proposed", "none"),
    ("proposed_vs_uniform", "proposed", "uniform_weight"),
    ("proposed_vs_unbounded", "proposed", "unbounded_uncertainty"),
    ("unbounded_vs_mint", "unbounded_uncertainty", "mint"),
)


def _paired(
    study: pd.DataFrame, forecaster: str, left: str, right: str
) -> tuple[np.ndarray, float]:
    """Per-origin MAE of ``left`` minus ``right``, in origin order, and the reference."""
    a = study[(study["reconciler"] == left) & (study["method"] == forecaster)]
    b = study[(study["reconciler"] == right) & (study["method"] == forecaster)]
    merged = a.merge(b, on="origin", suffixes=("_l", "_r")).sort_values("origin")
    if merged.empty:
        raise RoundTwoError(f"no paired origins for {forecaster}: {left} against {right}")
    difference = (merged["mae_after_l"] - merged["mae_after_r"]).to_numpy(dtype=float)
    return difference, float(merged["mae_after_r"].mean())


def reconciliation_inference(label: str) -> dict[str, Any]:
    """Separate statements about the mean change and about the typical origin."""
    path = TABLES / f"reconciliation_raw_{label}.csv"
    if not path.is_file():
        raise RoundTwoError(f"{path} is missing; run analysis.py first")
    study = pd.read_csv(path)
    study = study[study["failed"].isna() | (study["failed"] == "")]

    out: dict[str, Any] = {}
    for name, left, right in COMPARISONS:
        per_forecaster: dict[str, Any] = {}
        for forecaster in sorted(study["method"].unique()):
            difference, reference = _paired(study, forecaster, left, right)
            scale = 100.0 / reference
            nonzero = difference[np.abs(difference) > 1e-12]
            improved = int((nonzero < 0).sum())
            worsened = int((nonzero > 0).sum())
            sign = stats.binomtest(improved, improved + worsened, 0.5)
            wilcoxon = stats.wilcoxon(difference)
            by_block: dict[str, Any] = {}
            for block in BLOCK_LENGTHS:
                mean_low, mean_high = block_bootstrap(difference, np.mean, block=block)
                median_low, median_high = block_bootstrap(
                    difference, lambda v: float(np.median(v)), block=block
                )
                by_block[str(block)] = {
                    "mean_ci_percent": [scale * mean_low, scale * mean_high],
                    "median_ci_percent": [scale * median_low, scale * median_high],
                }
            per_forecaster[forecaster] = {
                "n_origins": int(difference.size),
                "mean_change_percent": float(scale * difference.mean()),
                "median_change_percent": float(scale * np.median(difference)),
                "n_improved": improved,
                "n_worsened": worsened,
                "n_tied": int(difference.size - nonzero.size),
                "sign_test_p_value": float(sign.pvalue),
                "signed_rank_p_value": float(wilcoxon.pvalue),
                "skewness": float(stats.skew(difference)),
                "lag1_autocorrelation": float(np.corrcoef(difference[:-1], difference[1:])[0, 1]),
                "by_block_length": by_block,
            }
        out[name] = per_forecaster
    return out


def headline_block_sensitivity(key: str) -> dict[str, Any]:
    """The dependence-robust interval of the headline real-data test, by block length.

    The reconciliation differences above are nearly uncorrelated between origins, so they
    say little about whether a block of ten is long enough. The loss differential of the
    headline comparison on the real sites is the opposite case, with a lag-one
    autocorrelation of 0.57 on ROBOD and 0.92 on HZMetro, and it is where the choice of
    block could change a conclusion. The differential is rebuilt exactly as
    ``analysis.dependence_check`` builds it, and the interval at ten origins is required
    to reproduce the one that function logged.
    """
    directories = [
        d for d in base.runs_for(base.REAL[key]) if (d / "predictions.parquet").is_file()
    ]
    if not directories:
        raise RoundTwoError(f"no logged predictions for {base.REAL[key]}")
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    frame = frame.assign(method=[base.base_method(str(m)) for m in frame["method"]])
    frame = frame[frame["step"] <= base.CONTEXT_HORIZON[key]]
    prefix = "occ:" if base.CONTEXT_GROUP[key] == "occupancy" else "flow:"
    frame = frame[frame["series_id"].str.startswith(prefix)]
    frame = frame.assign(ae=(frame["q50"] - frame["actual"]).abs())
    per_origin = frame.groupby(["method", "origin"])["ae"].mean().unstack(0)
    zero_shot = min(
        (c for c in per_origin.columns if c in base.ZERO_SHOT),
        key=lambda c: per_origin[c].mean(),
    )
    baseline = min(
        (c for c in per_origin.columns if c not in base.ZERO_SHOT),
        key=lambda c: per_origin[c].mean(),
    )
    pair = per_origin[[zero_shot, baseline]].dropna()
    difference = pair[zero_shot].to_numpy() - pair[baseline].to_numpy()

    by_block: dict[str, Any] = {}
    for block in BLOCK_LENGTHS:
        low, high = block_bootstrap(difference, lambda v: float(v.mean()), block=block)
        by_block[str(block)] = {
            "mean_ci": [low, high],
            "excludes_zero": bool(high < 0.0 or low > 0.0),
        }
    logged = base.block_bootstrap_ci(difference)
    ten = by_block["10"]["mean_ci"]
    if not np.allclose(ten, logged, rtol=0.0, atol=1e-12):
        raise RoundTwoError(
            f"{key}: the interval at ten origins {ten} does not reproduce the logged {logged}"
        )
    series = pd.Series(difference)
    return {
        "zero_shot": str(zero_shot),
        "baseline": str(baseline),
        "n_origins": int(difference.size),
        "mean_differential": float(difference.mean()),
        "autocorrelation": {str(lag): float(series.autocorr(lag=lag)) for lag in (1, 2, 5, 10)},
        "by_block_length": by_block,
    }


# --------------------------------------------------------------------------- #
# 4. Per-quantile projection, before and after the monotone rearrangement
# --------------------------------------------------------------------------- #


def per_quantile_diagnosis(max_origins: int) -> dict[str, Any]:
    """Where the per-quantile conservation residual comes from.

    The published table measured the residual after the levels had been sorted to repair
    crossings. Sorting acts coordinate by coordinate, so wherever two levels cross it
    exchanges coordinates between two paths that each satisfied conservation, and the
    exchanged paths need not. This function measures the residual on both sides of the
    sort, and compares the per-quantile median with the shift median before the sort,
    where the two are the same optimisation problem.
    """
    from mflow.reconcile import default_registry
    from mflow.reconcile.projection import sigma_from_quantiles
    from mflow.reconcile.quantiles import crossing_rate, enforce_monotone

    experiment, site_id, variant = base.RECON_SITES["baseline"]
    system, _graph, _panel, origins = base._origin_cubes(
        experiment, site_id, variant, 0, max_origins
    )
    registry = default_registry()
    reconciler = registry.create("proposed")
    levels = list(base.QUANTILES)
    median_index = levels.index(0.5)

    rows: list[dict[str, Any]] = []
    per_origin: list[dict[str, Any]] = []
    for method, origin, cube, _actual, rhs in origins:
        sigma = sigma_from_quantiles(cube, levels)
        shift = reconciler.reconcile(cube[:, :, median_index], system, rhs, sigma=sigma)
        shift_median = np.asarray(shift.reconciled, dtype=float)
        projected = np.stack(
            [
                np.asarray(
                    reconciler.reconcile(cube[:, :, i], system, rhs, sigma=sigma).reconciled,
                    dtype=float,
                )
                for i in range(len(levels))
            ],
            axis=-1,
        )
        repaired = enforce_monotone(projected)
        moved = np.abs(repaired - projected) > 1e-9
        for index, level in enumerate(levels):
            before = projected[:, :, index].reshape(-1)
            after = repaired[:, :, index].reshape(-1)
            rows.append(
                {
                    "method": method,
                    "origin": origin,
                    "level": level,
                    "residual_before_sort": float(np.abs(system.residual(before, rhs)).mean()),
                    "residual_after_sort": float(np.abs(system.residual(after, rhs)).mean()),
                    "max_residual_before_sort": float(np.abs(system.residual(before, rhs)).max()),
                    "violation_before_sort": float(system.violation(before).sum()),
                    "violation_after_sort": float(system.violation(after).sum()),
                    "fraction_moved_by_sort": float(moved[:, :, index].mean()),
                }
            )
        per_origin.append(
            {
                "method": method,
                "origin": origin,
                "crossing_rate_before_sort": crossing_rate(projected),
                "crossing_rate_after_sort": crossing_rate(repaired),
                "median_gap_to_shift_before_sort": float(
                    np.abs(projected[:, :, median_index] - shift_median).max()
                ),
                "median_gap_to_shift_after_sort": float(
                    np.abs(repaired[:, :, median_index] - shift_median).max()
                ),
                "shift_median_residual": float(
                    np.abs(system.residual(shift_median.reshape(-1), rhs)).mean()
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "round2_per_quantile_raw.csv", index=False)
    by_level = frame.groupby("level").agg(
        residual_before_sort=("residual_before_sort", "mean"),
        residual_after_sort=("residual_after_sort", "mean"),
        max_residual_before_sort=("max_residual_before_sort", "max"),
        violation_before_sort=("violation_before_sort", "mean"),
        violation_after_sort=("violation_after_sort", "mean"),
        fraction_moved_by_sort=("fraction_moved_by_sort", "mean"),
    )
    by_level.to_csv(TABLES / "round2_per_quantile.csv")
    origin_frame = pd.DataFrame(per_origin)
    options = dict(reconciler.solver_options)
    return {
        "n_origin_forecaster_pairs": len(origin_frame),
        "by_level": json.loads(by_level.reset_index().to_json(orient="records")),
        "crossing_rate_before_sort": float(origin_frame["crossing_rate_before_sort"].mean()),
        "crossing_rate_after_sort": float(origin_frame["crossing_rate_after_sort"].mean()),
        "fraction_of_pairs_with_a_crossing": float(
            (origin_frame["crossing_rate_before_sort"] > 0).mean()
        ),
        "median_gap_to_shift_before_sort_max": float(
            origin_frame["median_gap_to_shift_before_sort"].max()
        ),
        "median_gap_to_shift_after_sort_max": float(
            origin_frame["median_gap_to_shift_after_sort"].max()
        ),
        "shift_median_residual_mean": float(origin_frame["shift_median_residual"].mean()),
        "solver": str(reconciler.solver),
        "solver_options": {
            k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in options.items()
        },
        "post_solve_residual_tolerance": float(reconciler.residual_tolerance),
    }


# --------------------------------------------------------------------------- #
# 5. Risk-sensitive errors with thresholds and sample sizes
# --------------------------------------------------------------------------- #

RISK_RULES: dict[str, tuple[float, float]] = {
    # name: (fraction of capacity that defines the event, tolerance in persons)
    # ``at_capacity`` is the rule the previous revision logged. ``at_capacity_rounded``
    # reads the forecast to the nearest person, which is the resolution occupancy is
    # counted at; it is reported because a box-constrained forecast sits on the bound to
    # within solver tolerance and the strict rule then turns on the last digit.
    "at_capacity": (1.0, 0.0),
    "at_capacity_rounded": (1.0, 0.5),
    "near_capacity": (0.8, 0.0),
}


def _risk_rows(
    label: str, *, arms: tuple[str, ...] = base.RECONCILERS, warm_start: bool = False
) -> pd.DataFrame:
    """Per-origin risk-sensitive errors for each reconciler.

    ``warm_start=False`` builds a fresh reconciler at every origin, which is how
    ``analysis.py`` ran the logged study, so the two agree to the last digit.
    ``warm_start=True`` reuses one reconciler per arm, so each solve starts from the
    previous origin's solution. The two reach the same optimum to solver tolerance, and
    comparing them shows which of the event rules is stable under that tolerance.
    """
    from mflow.reconcile import default_registry
    from mflow.reconcile.quantiles import QuantileReconciler

    experiment, site_id, variant = base.RECON_SITES[label]
    system, graph, _panel, origins = base._origin_cubes(experiment, site_id, variant, 0, 10_000)
    registry = default_registry()
    shared = {name: registry.create(name) for name in arms}
    capacity = np.array(
        [
            graph.node_capacity[sid.split(":", 1)[1]] if sid.startswith("occ:") else np.nan
            for sid in system.series_ids
        ],
        dtype=float,
    )
    rooms = np.isfinite(capacity) & (capacity > 0)
    if not rooms.any():
        raise RoundTwoError(f"{site_id} declares no room capacity")
    cap = capacity[rooms][:, None]
    median_index = base.QUANTILES.index(0.5)

    rows: list[dict[str, Any]] = []
    for method, origin, cube, actual, rhs in origins:
        observed = actual[rooms]
        for name in arms:
            reconciler = shared[name] if warm_start else registry.create(name)
            reconciled, _ = QuantileReconciler(reconciler, strategy="shift").reconcile(
                cube, list(base.QUANTILES), system, rhs
            )
            point = reconciled[:, :, median_index]
            predicted = point[rooms]
            hot = observed >= 0.8 * cap
            row: dict[str, Any] = {
                "method": method,
                "reconciler": name,
                "origin": origin,
                "mae": float(np.mean(np.abs(point - actual))),
                "peak_error": float(np.mean(np.abs(predicted.max(axis=1) - observed.max(axis=1)))),
                "n_room_steps": int(observed.size),
                "n_high_occupancy_steps": int(hot.sum()),
                "high_occupancy_abs_error_sum": float(np.abs(predicted[hot] - observed[hot]).sum()),
            }
            for rule, (fraction, tolerance) in RISK_RULES.items():
                event = observed >= fraction * cap
                alarm = predicted >= fraction * cap - tolerance
                row[f"{rule}_events"] = int(event.sum())
                row[f"{rule}_hits"] = int((event & alarm).sum())
                row[f"{rule}_misses"] = int((event & ~alarm).sum())
                row[f"{rule}_false_alarms"] = int((~event & alarm).sum())
                row[f"{rule}_brier"] = float(
                    np.mean((alarm.astype(float) - event.astype(float)) ** 2)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def risk_study(label: str) -> dict[str, Any]:
    """Peak error, high-occupancy error and event detection, with what they rest on."""
    rows = _risk_rows(label)
    rows.to_csv(TABLES / f"round2_risk_raw_{label}.csv", index=False)

    # The reconciliation arms must reproduce the per-origin errors that analysis.py
    # logged; if they do not, the two scripts are not describing the same experiment.
    logged = pd.read_csv(TABLES / f"reconciliation_raw_{label}.csv")
    check = rows.merge(logged, on=["method", "reconciler", "origin"], how="inner")
    if len(check) != len(rows):
        raise RoundTwoError(f"{label}: the risk rows do not align with the logged study")
    drift = float(np.abs(check["mae"] - check["mae_after"]).max())
    if drift > 1e-6:
        raise RoundTwoError(f"{label}: recomputed MAE differs from the logged MAE by {drift}")

    rows["high_occupancy_mae"] = np.where(
        rows["n_high_occupancy_steps"] > 0,
        rows["high_occupancy_abs_error_sum"] / rows["n_high_occupancy_steps"].clip(lower=1),
        np.nan,
    )
    raw = rows[rows["reconciler"] == "none"]
    first = raw[raw["method"] == raw["method"].iloc[0]]
    sample = {
        "n_origins_per_forecaster": int(first["origin"].nunique()),
        "n_forecasters": int(rows["method"].nunique()),
        "n_room_steps_per_forecaster": int(first["n_room_steps"].sum()),
        "n_high_occupancy_steps_per_forecaster": int(first["n_high_occupancy_steps"].sum()),
        "n_origins_with_high_occupancy": int((first["n_high_occupancy_steps"] > 0).sum()),
        "n_at_capacity_steps_per_forecaster": int(first["at_capacity_events"].sum()),
        "n_origins_with_at_capacity": int((first["at_capacity_events"] > 0).sum()),
    }

    arms: dict[str, Any] = {}
    for name in base.RECONCILERS:
        arm = rows[rows["reconciler"] == name].sort_values(["method", "origin"])
        reference = raw.sort_values(["method", "origin"])
        entry: dict[str, Any] = {
            "mae": float(arm["mae"].mean()),
            "peak_error": float(arm["peak_error"].mean()),
            "high_occupancy_mae": float(arm["high_occupancy_mae"].mean()),
            "high_occupancy_mae_pooled": float(
                arm["high_occupancy_abs_error_sum"].sum() / arm["n_high_occupancy_steps"].sum()
            ),
        }
        for rule in RISK_RULES:
            events = int(arm[f"{rule}_events"].sum())
            hits = int(arm[f"{rule}_hits"].sum())
            false_alarms = int(arm[f"{rule}_false_alarms"].sum())
            non_events = int(arm["n_room_steps"].sum()) - events
            entry[rule] = {
                "events": events,
                "hits": hits,
                "misses": int(arm[f"{rule}_misses"].sum()),
                "false_alarms": false_alarms,
                "hit_rate": hits / events if events else float("nan"),
                "false_alarm_rate": false_alarms / non_events if non_events else float("nan"),
                "precision": hits / (hits + false_alarms) if hits + false_alarms else float("nan"),
                "brier": float(arm[f"{rule}_brier"].mean()),
            }
        if name != "none":
            # Paired per-origin changes against the raw forecast, pooled over the two
            # forecasters in origin order within each, with a block bootstrap interval.
            changes: dict[str, Any] = {}
            for metric in (
                "peak_error",
                "high_occupancy_mae",
                "at_capacity_brier",
                "at_capacity_rounded_brier",
                "near_capacity_brier",
            ):
                per_forecaster: dict[str, Any] = {}
                for forecaster in sorted(arm["method"].unique()):
                    a = arm[arm["method"] == forecaster][metric].to_numpy(dtype=float)
                    b = reference[reference["method"] == forecaster][metric].to_numpy(dtype=float)
                    difference = a - b
                    difference = difference[np.isfinite(difference)]
                    low, high = block_bootstrap(difference, np.mean, block=10)
                    per_forecaster[forecaster] = {
                        "n": int(difference.size),
                        "mean_change": float(difference.mean()),
                        "ci": [low, high],
                        "reference_mean": float(np.nanmean(b)),
                    }
                changes[metric] = per_forecaster
            entry["change_against_raw"] = changes
        arms[name] = entry

    # The same projection, started from the previous origin's solution instead of from
    # scratch. Any rule whose score moves here is reading solver noise.
    warm = _risk_rows(label, arms=("proposed",), warm_start=True)
    cold = rows[rows["reconciler"] == "proposed"]
    stability: dict[str, Any] = {
        "max_abs_mae_difference": float(
            np.abs(
                warm.sort_values(["method", "origin"])["mae"].to_numpy()
                - cold.sort_values(["method", "origin"])["mae"].to_numpy()
            ).max()
        ),
    }
    for rule in RISK_RULES:
        stability[rule] = {
            "brier_cold_start": float(cold[f"{rule}_brier"].mean()),
            "brier_warm_start": float(warm[f"{rule}_brier"].mean()),
            "hits_cold_start": int(cold[f"{rule}_hits"].sum()),
            "hits_warm_start": int(warm[f"{rule}_hits"].sum()),
        }
    return {"sample": sample, "arms": arms, "solver_start_stability": stability}


# --------------------------------------------------------------------------- #


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    print("1. width at matched empirical coverage")
    summary["matched_coverage"] = {}
    for key, horizon in (("robod", 12), ("hzmetro", 4)):
        summary["matched_coverage"][key] = {}
        for target in (0.8, 0.9):
            table = matched_coverage(key, horizon, target)
            table.to_csv(TABLES / f"round2_matched_{key}_{round(100 * target)}.csv")
            summary["matched_coverage"][key][str(target)] = json.loads(
                table.to_json(orient="index")
            )
            print(f"  {key}, target coverage {target}")
            print(
                table[
                    [
                        "coverage",
                        "width",
                        "fraction_degenerate",
                        "fraction_uncoverable",
                        "factor_in_sample",
                        "width_in_sample",
                        "factor_calibration",
                        "evaluation_coverage",
                        "evaluation_width",
                        "coverage_just_below_one",
                        "expanding_coverage",
                        "expanding_width",
                    ]
                ]
                .round(3)
                .to_string()
                .replace("\n", "\n    ")
            )

    print("\n2. the gradient-boosted comparison on one definition")
    summary["lightgbm"] = {key: lightgbm_restated(key) for key in ("robod", "hzmetro")}
    for key, stat in summary["lightgbm"].items():
        print(
            f"  {key}: winner {stat['winner']}, MASE {stat['local_mase']:.4f} local against "
            f"{stat['zero_shot_mase']:.4f} zero-shot, reduction "
            f"{stat['mase_reduction_winner_vs_loser_percent']:.1f}% on MASE, "
            f"{stat['mae_reduction_winner_vs_loser_percent']:.1f}% on paired MAE "
            f"(excess {stat['mae_excess_loser_over_winner_percent']:.1f}%), "
            f"p={stat['p_value']:.3g}, holm={stat['holm_significant']}"
        )

    print("\n3. mean effect and typical-origin effect of reconciliation")
    summary["reconciliation_inference"] = {
        label: reconciliation_inference(label) for label in ("baseline", "crowded")
    }
    for label, comparisons in summary["reconciliation_inference"].items():
        print(f"  --- {label}")
        for name, per_forecaster in comparisons.items():
            for forecaster, stat in per_forecaster.items():
                ten = stat["by_block_length"]["10"]
                print(
                    f"    {name:22s} {forecaster:22s} mean {stat['mean_change_percent']:+6.2f}% "
                    f"[{ten['mean_ci_percent'][0]:+.2f},{ten['mean_ci_percent'][1]:+.2f}]  "
                    f"median {stat['median_change_percent']:+7.4f}% "
                    f"[{ten['median_ci_percent'][0]:+.4f},{ten['median_ci_percent'][1]:+.4f}]  "
                    f"better/worse/tied {stat['n_improved']}/{stat['n_worsened']}/{stat['n_tied']} "
                    f"sign p={stat['sign_test_p_value']:.2e}"
                )

    print("\n3b. the headline real-data interval against the block length")
    summary["headline_block_sensitivity"] = {
        key: headline_block_sensitivity(key) for key in ("robod", "hzmetro")
    }
    for key, stat in summary["headline_block_sensitivity"].items():
        print(f"  --- {key}: {stat['zero_shot']} against {stat['baseline']}")
        for block, entry in stat["by_block_length"].items():
            low, high = entry["mean_ci"]
            print(
                f"    block {block:>2s}  [{low:+.3f}, {high:+.3f}]  "
                f"excludes zero: {entry['excludes_zero']}"
            )

    print("\n4. per-quantile projection before and after the monotone rearrangement")
    summary["per_quantile"] = per_quantile_diagnosis(max_origins=60)
    for row in summary["per_quantile"]["by_level"]:
        print(
            f"    level {row['level']:.1f}  residual before sort "
            f"{row['residual_before_sort']:.2e}  after {row['residual_after_sort']:.4f}  "
            f"moved {100 * row['fraction_moved_by_sort']:.2f}%"
        )
    diagnosis = summary["per_quantile"]
    print(
        f"    crossing rate before sort {diagnosis['crossing_rate_before_sort']:.5f}; "
        f"median gap to shift before sort "
        f"{diagnosis['median_gap_to_shift_before_sort_max']:.2e}, after "
        f"{diagnosis['median_gap_to_shift_after_sort_max']:.4f}"
    )

    print("\n5. risk-sensitive errors")
    summary["risk"] = {label: risk_study(label) for label in ("baseline", "crowded")}
    for label, study in summary["risk"].items():
        print(f"  --- {label}: {study['sample']}")
        print(f"      solver start stability: {study['solver_start_stability']}")
        for name, arm in study["arms"].items():
            strict = arm["at_capacity"]
            near = arm["near_capacity"]
            print(
                f"    {name:22s} peak {arm['peak_error']:.4f}  high-occ MAE "
                f"{arm['high_occupancy_mae']:.4f}  at-cap hits "
                f"{strict['hits']}/{strict['events']} FA {strict['false_alarms']} "
                f"brier {strict['brier']:.5f}  near-cap hits {near['hits']}/{near['events']} "
                f"FA {near['false_alarms']} brier {near['brier']:.5f}"
            )

    (HERE / "summary_round2.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'summary_round2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
