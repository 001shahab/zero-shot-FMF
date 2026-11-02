"""Build every figure and table in the paper from the logged runs.

Nothing here invents a number. Each artefact is derived from ``results/<run_id>/``, which
is written only by ``mflow run``, and the script fails loudly if a run it needs is absent
rather than emitting a placeholder.

Three things in here are worth knowing before reading the output.

Seeds are not replicates. Every forecaster in this study is deterministic, which the
script checks rather than assumes (:func:`assert_seeds_are_replicates`). Pooling the three
seeds would triple the sample without adding information and would understate every
p-value, so all inference runs on distinct origins from a single seed.

Two analyses are computed offline from stored forecasts rather than from a fresh sweep.
The conservation residual of a raw forecast and the effect of reconciling it are both
functions of the quantile forecast and the building graph alone, so they can be evaluated
from ``predictions.parquet`` without re-running any model. This is exact, not an
approximation: the reconciler sees precisely the vector the harness would have handed it.

The headline hypothesis is tested within datasets, not across them. A three-point ordering
over three datasets confounds the predictor with dataset identity, so the real test
regresses per-series advantage on per-series seasonal strength inside one panel.

Revision note. This script grew substantially in response to the journal referees, and
the additions are worth naming because several of them change a reported conclusion.

The gradient-boosted competitor is now a ladder of five specifications rather than one
    30|(:func:`table_lightgbm`), because the submitted configuration was handicapped in two
ways that had nothing to do with gradient boosting. The per-series arm beats every
foundation model on HZMetro, which falsifies the first hypothesis on that dataset.

Context length is swept over five values on both real datasets rather than two on one
(:func:`table_context`, :func:`figure_context`), and the sweep is restricted to the
origins common to every context so the comparison stays paired.

The reconciliation study gains an arm that applies the proposed diagonal metric without
the box (:func:`reconciliation_study`), which is what isolates the inequality constraints
from the choice of weighting matrix. It also reports confidence intervals, risk-sensitive
    40|errors, the behaviour of the non-median quantile levels under each lifting strategy, and
the sensitivity of the result to the Gaussian mapping from interval width to sigma.

Calibration and sharpness are separated (:func:`table_calibration`), because a method
closer to nominal coverage and a method with a narrower interval are not comparable until
one of them is recalibrated.

Usage:
    python paper/<manuscript>/analysis.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
TABLES = Path(__file__).resolve().parent / "tables"

QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QCOLS = [f"q{round(q * 100)}" for q in QUANTILES]
NORMAL_80_SPREAD = 2.5631

# Which experiment supplies the real-data results. E7 used a one-season context, which
# left seasonal_naive and historical_average with no residuals to build a predictive fan
# from and so scored them with zero-width intervals. E8 is the same protocol at two
# seasons, and adds the trained local baseline.
REAL = {"robod": "E8_robod", "hzmetro": "E8_hzmetro"}
ROBOD_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
HZ_MINUTES = {1: 15, 2: 30, 4: 60, 8: 120}

# The gradient-boosted re-specification, E10. The submitted paper ran one configuration;
# the referees asked for the training window, the lag selection, the tuning procedure and
# the forecast strategy, and reading our own code turned up two handicaps in it.
LIGHTGBM = {"robod": "E10_lightgbm_robod", "hzmetro": "E10_lightgbm_hzmetro"}
LIGHTGBM_LADDER = (
    "lightgbm_as_submitted",
    "lightgbm_horizon_matched",
    "lightgbm_seasonal_lags",
    "lightgbm_tuned",
    "lightgbm_per_series",
)

# The context sweep, E9. Keys are the context in daily seasons.
CONTEXT_SWEEP = {
    "hzmetro": {1: 96, 2: 192, 3: 288, 7: 672, 14: 1344},
    "robod": {1: 288, 2: 576, 3: 864, 7: 2016, 14: 4032},
}
CONTEXT_SITE = {"hzmetro": "hzmetro", "robod": "robod_bldg1"}
CONTEXT_GROUP = {"hzmetro": "flow", "robod": "occupancy"}
CONTEXT_HORIZON = {"hzmetro": 8, "robod": 12}

# Data efficiency and training-budget sensitivity, E11. E11_efficiency_sim supersedes E4:
# the training budget used to be padded by imputation, so a small budget trained the
# booster largely on a back-filled constant.
EFFICIENCY_SIM = "E11_efficiency_sim"
EFFICIENCY_ROBOD = "E11_efficiency_robod"

# The reconciliation sites. The baseline museum is nearly always inside its capacity
# bounds, so the box is rarely active there and the comparison against MinT is made in a
# regime with little to separate. The crowded site holds the building fixed and raises
# demand until the box binds.
RECON_SITES = {
    "baseline": (EFFICIENCY_SIM, "sim_house_museum", "d090"),
    "crowded": ("E12_crowded", "sim_house_museum_crowded", "default"),
}

#: Reconcilers compared. ``unbounded_uncertainty`` is new in the revision and is the arm
#: that isolates the inequality constraints: it uses exactly the proposed diagonal metric
#: and drops only the box, so ``proposed`` minus ``unbounded_uncertainty`` is the effect
#: of the box and ``unbounded_uncertainty`` minus ``mint`` is the effect of the weighting
#: matrix. Without it, the two effects are confounded, which is what a referee observed.
RECONCILERS = ("none", "mint", "unbounded_uncertainty", "uniform_weight", "proposed")

#: Three ablations, each isolating one design decision while holding the others fixed.
RECON_ABLATION_ARMS: tuple[tuple[str, str, str], ...] = (
    # The weighting, with the box held on.
    ("proposed_vs_uniform", "proposed", "uniform_weight"),
    # The box, with the proposed weighting held on. This is the comparison the submitted
    # paper should have made instead of comparing against MinT, which changes the metric
    # and the constraint set at the same time.
    ("proposed_vs_unbounded", "proposed", "unbounded_uncertainty"),
    # The weighting matrix, with the constraint set held at equalities only.
    ("unbounded_vs_mint", "unbounded_uncertainty", "mint"),
)
RECON_ABLATIONS = tuple(name for name, _left, _right in RECON_ABLATION_ARMS)

ZERO_SHOT = {
    "toto2",
    "chronos2_multivariate",
    "chronos2_univariate",
    "timesfm3_univariate",
    "timesfm3_multivariate",
}

# A referee asked for larger legends and axis labels. These sizes are set against the
# width the figures are included at in the two-column layout rather than against the
# figure's own width, which is why they look large for a 3.4 inch panel.
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)

STYLE: dict[str, tuple[str, str]] = {
    "toto2": ("#d62728", "o"),
    "chronos2_multivariate": ("#9467bd", "s"),
    "timesfm3_univariate": ("#1f77b4", "^"),
    "timesfm3_multivariate": ("#17becf", "v"),
    "last_value": ("#7f7f7f", "x"),
    "seasonal_naive": ("#2ca02c", "d"),
    "historical_average": ("#8c564b", "P"),
    "global_lightgbm": ("#ff7f0e", "*"),
    "seasonal_naive_weekly": ("#bcbd22", "D"),
    "lightgbm_as_submitted": ("#ff7f0e", "*"),
    "lightgbm_horizon_matched": ("#e377c2", "1"),
    "lightgbm_seasonal_lags": ("#e377c2", "2"),
    "lightgbm_tuned": ("#e377c2", "3"),
    # The per-series arm appears in the headline figure alongside the foundation models,
    # so it needs a colour of its own rather than one borrowed from a baseline.
    "lightgbm_per_series": ("#000000", "h"),
}
PRETTY = {
    "toto2": "Toto-2.0",
    "chronos2_multivariate": "Chronos-2 (mv)",
    "timesfm3_univariate": "TimesFM-3 (uv)",
    "timesfm3_multivariate": "TimesFM-3 (mv)",
    "last_value": "Last value",
    "seasonal_naive": "Seasonal naive (daily)",
    "seasonal_naive_weekly": "Seasonal naive (weekly)",
    "historical_average": "Historical average",
    "global_lightgbm": "LightGBM (global)",
    "lightgbm_as_submitted": "LightGBM (as submitted)",
    "lightgbm_horizon_matched": "LightGBM (horizon-matched)",
    "lightgbm_seasonal_lags": "LightGBM (seasonal lags)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "lightgbm_per_series": "LightGBM (per series)",
}


class AnalysisError(RuntimeError):
    """Raised when a required run is missing, rather than guessing at its contents."""


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #


def save(figure: plt.Figure, filename: str) -> None:
    """Write a figure as flat RGB.

    matplotlib emits RGBA by default, and pdfTeX turns every alpha channel into a soft
    mask, which on a document with this many panels does not finish in any reasonable
    time. Compositing onto white here costs nothing and keeps the paper buildable.
    """
    path = FIGURES / filename
    figure.savefig(path)
    plt.close(figure)
    from PIL import Image

    with Image.open(path) as image:
        if image.mode != "RGBA":
            return
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.split()[3])
    flat.save(path, optimize=True)


def legend_below(figure: plt.Figure, axis: plt.Axes, ncol: int, y: float = -0.04) -> None:
    """Put the legend under the plot.

    Every one of these panels is a bundle of monotone curves that fills its own axes, so
    an inside legend lands on the data whichever corner it picks. Below the axes it never
    collides and the panels stay directly comparable.
    """
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
        handletextpad=0.5,
    )


def base_method(name: str) -> str:
    """Strip the reconciler suffix from a method label."""
    return name.split("+")[0]


def order_for_legend(table: pd.DataFrame) -> list[str]:
    """Methods ordered by mean value, so the legend reads in the curves' own order."""
    return list(table.mean().sort_values().index)


# --------------------------------------------------------------------------- #
# Loading, with the seed question settled up front
# --------------------------------------------------------------------------- #


def runs_for(experiment: str) -> list[Path]:
    """Every completed run directory of one experiment."""
    found = sorted(
        d
        for d in RESULTS.glob(f"{experiment}_*")
        if (d / "metrics.parquet").is_file() and (d / "manifest.json").is_file()
    )
    if not found:
        raise AnalysisError(
            f"no completed run of {experiment!r} under {RESULTS}. Run it with "
            f"`mflow run configs/experiments/{experiment}.yaml` first; this script does "
            "not fabricate results."
        )
    return found


def assert_seeds_are_replicates(experiment: str) -> dict[str, bool]:
    """Report, per method, whether the seeds produced bit-identical forecasts.

    A method whose seeds agree to the bit has not been replicated by running it again:
    pooling those origins multiplies the sample size without adding information and
    understates every p-value computed from it. This is checked rather than assumed,
    because which methods are stochastic is a property of the implementations and not
    something to be recalled.
    """
    directories = [d for d in runs_for(experiment) if (d / "predictions.parquet").is_file()]
    if len(directories) < 2:
        raise AnalysisError(f"{experiment} has fewer than two seeds to compare")
    key = ["method", "origin", "series_id", "step"]
    first = pd.read_parquet(directories[0] / "predictions.parquet")
    spread: dict[str, float] = {}
    for other in directories[1:]:
        second = pd.read_parquet(other / "predictions.parquet")
        merged = first.merge(second, on=key, suffixes=("_a", "_b"))
        if merged.empty:
            raise AnalysisError(f"{experiment} seeds share no common origins")
        gap = (merged["q50_a"] - merged["q50_b"]).abs().groupby(merged["method"]).max()
        for method, value in gap.items():
            spread[str(method)] = max(spread.get(str(method), 0.0), float(value))
    return {base_method(m): v == 0.0 for m, v in sorted(spread.items())}


def metrics_for(experiment: str, *, distinct: bool = True) -> pd.DataFrame:
    """Per-origin metrics for one experiment.

    With ``distinct`` the frame is restricted to a single seed, which is correct whenever
    the seeds are bit-identical replicates: pooling them would triple the sample size
    without adding information.
    """
    from mflow.eval.report import combine, load_runs

    frame = combine(load_runs(experiment=experiment))
    if distinct and "seed" in frame.columns:
        return frame[frame["seed"] == frame["seed"].min()].copy()
    return frame


# --------------------------------------------------------------------------- #
# Seasonal strength
# --------------------------------------------------------------------------- #


def seasonal_strength(series: np.ndarray, period: int) -> float:
    """Fraction of variance explained by the mean seasonal profile.

    For a series :math:`y_t` with period :math:`m`, let :math:`\\bar{y}^{(m)}_{t \\bmod m}`
    be the mean over all observations sharing a phase. The statistic is

    .. math:: S_m = 1 - \\frac{\\operatorname{Var}(y_t - \\bar{y}^{(m)}_{t \\bmod m})}
                              {\\operatorname{Var}(y_t)}.

    It is zero when the seasonal profile carries no information and approaches one when
    the series is a deterministic function of phase.
    """
    if period < 2:
        raise AnalysisError(f"a seasonal period of {period} is not a season")
    finite = np.isfinite(series)
    if finite.sum() < 2 * period:
        return float("nan")
    phase = np.arange(series.size) % period
    profile = np.full(period, np.nan)
    for p in range(period):
        take = finite & (phase == p)
        if take.any():
            profile[p] = series[take].mean()
    residual = series - profile[phase]
    ok = np.isfinite(residual) & finite
    if ok.sum() < 2:
        return float("nan")
    total = float(np.var(series[ok]))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.var(residual[ok]) / total)


def per_series_strength(site_id: str, prefix: str) -> pd.Series:
    """Daily seasonal strength of every series of one site sharing a prefix."""
    from mflow.schema import load_site

    site = load_site(ROOT / "data" / "canonical" / site_id)
    panel = site.to_panel()
    period = round(24 * 3600 / site.meta.interval_seconds)
    values = {
        sid: seasonal_strength(panel.series[i], period)
        for i, sid in enumerate(panel.series_ids)
        if sid.startswith(prefix)
    }
    out = pd.Series(values, dtype=float)
    return out[np.isfinite(out)]


# --------------------------------------------------------------------------- #
# The within-dataset test of the predictability hypothesis
# --------------------------------------------------------------------------- #


def per_series_advantage(
    experiment: str, zero_shot: str, baseline: str
) -> pd.Series:
    """Percentage MAE advantage of one fixed method over another, per series.

    Both arms are fixed in advance rather than chosen per series, because taking the
    per-series minimum over methods would select on the noise being regressed.
    """
    directories = [d for d in runs_for(experiment) if (d / "predictions.parquet").is_file()]
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    wanted = [f"{zero_shot}+none", f"{baseline}+none"]
    frame = frame[frame["method"].isin(wanted)]
    if frame.empty:
        raise AnalysisError(f"{experiment} has no runs of {wanted}")
    frame = frame.assign(ae=(frame["q50"] - frame["actual"]).abs())
    table = frame.groupby(["method", "series_id"])["ae"].mean().unstack(0)
    advantage = 100.0 * (1.0 - table[wanted[0]] / table[wanted[1]])
    return advantage[np.isfinite(advantage)]


def within_dataset_regression(
    experiment: str, site_id: str, prefix: str, zero_shot: str, baseline: str
) -> dict[str, Any]:
    """Regress per-series zero-shot advantage on per-series seasonal strength."""
    advantage = per_series_advantage(experiment, zero_shot, baseline)
    strength = per_series_strength(site_id, prefix)
    frame = pd.concat(
        [advantage.rename("advantage"), strength.rename("strength")], axis=1
    ).dropna()
    if len(frame) < 4:
        raise AnalysisError(f"{experiment} has only {len(frame)} usable series")
    fit = stats.linregress(frame["strength"], frame["advantage"])
    rho, rho_p = stats.spearmanr(frame["strength"], frame["advantage"])
    return {
        "experiment": experiment,
        "zero_shot": zero_shot,
        "baseline": baseline,
        "n_series": len(frame),
        "slope": float(fit.slope),
        "slope_se": float(fit.stderr),
        "slope_ci_low": float(fit.slope - 1.96 * fit.stderr),
        "slope_ci_high": float(fit.slope + 1.96 * fit.stderr),
        "p_value": float(fit.pvalue),
        "r_squared": float(fit.rvalue**2),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "strength_min": float(frame["strength"].min()),
        "strength_max": float(frame["strength"].max()),
        "frame": frame,
    }


def figure_within_dataset(results: list[dict[str, Any]], filename: str) -> None:
    """The real test of the predictability hypothesis, inside each panel."""
    figure, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 2.7))
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results, strict=True):
        frame = result["frame"]
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.scatter(
            frame["strength"], frame["advantage"], s=14, alpha=0.65,
            color="#1f77b4", edgecolors="none",
        )
        grid = np.linspace(frame["strength"].min(), frame["strength"].max(), 50)
        intercept = frame["advantage"].mean() - result["slope"] * frame["strength"].mean()
        axis.plot(
            grid, result["slope"] * grid + intercept,
            color="#d62728", linewidth=1.4,
        )
        name = result["experiment"].split("_")[-1].upper()
        axis.set_title(
            f"{name}, $n={result['n_series']}$ series\n"
            f"slope $={result['slope']:.0f}$, $p={result['p_value']:.2f}$, "
            f"$R^2={result['r_squared']:.3f}$"
        )
        axis.set_xlabel("per-series seasonal strength $S_m$")
        axis.set_ylabel(
            f"MAE advantage of\n{PRETTY[result['zero_shot']]} over "
            f"{PRETTY[result['baseline']]} (%)"
        )
    figure.tight_layout()
    save(figure, filename)


# --------------------------------------------------------------------------- #
# Reconciliation, evaluated offline from the stored forecasts
# --------------------------------------------------------------------------- #


def _origin_cubes(
    experiment: str, site_id: str, variant: str, seed: int, max_origins: int
) -> tuple[Any, Any, Any, list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Every scorable origin of one run as a quantile cube, truth and constraint rhs.

    Factored out because four separate analyses now walk the same origins: the reconciler
    comparison, the quantile-lifting comparison, the sigma sensitivity and the
    risk-sensitive errors. Re-deriving the cubes four times would quadruple the cost of
    the slowest part of this script for no benefit.
    """
    from mflow.graph import BuildingGraph
    from mflow.reconcile.constraints import build_constraints
    from mflow.schema import load_site

    directory = RESULTS / f"{experiment}_{site_id}_{variant}_s{seed}"
    if not (directory / "predictions.parquet").is_file():
        raise AnalysisError(
            f"{directory} has no predictions.parquet. Run "
            f"`mflow run configs/experiments/{experiment}.yaml` first."
        )
    frame = pd.read_parquet(directory / "predictions.parquet")

    site = load_site(ROOT / "data" / "canonical" / site_id)
    graph = BuildingGraph.from_site(site)
    panel = site.to_panel()
    series_ids = list(panel.series_ids)
    index = {sid: i for i, sid in enumerate(series_ids)}
    occupancy_rows = {
        node: index[f"occ:{node}"]
        for node in graph.interior_index
        if f"occ:{node}" in index
    }

    horizon = int(frame["step"].max())
    system = build_constraints(graph, series_ids, horizon, site.meta.interval_seconds)

    out: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for method, by_method in frame.groupby("method"):
        origins = sorted(by_method["origin"].unique())[:max_origins]
        for origin in origins:
            block = by_method[by_method["origin"] == origin]
            wide = block.pivot_table(index="series_id", columns="step", values=QCOLS)
            truth = block.pivot_table(index="series_id", columns="step", values="actual")
            if len([s for s in series_ids if s in wide.index]) != len(series_ids):
                continue
            cube = np.stack(
                [wide[q].loc[series_ids].to_numpy(dtype=float) for q in QCOLS], axis=-1
            )
            actual = truth.loc[series_ids].to_numpy(dtype=float)
            if not np.isfinite(cube).all() or not np.isfinite(actual).all():
                continue
            anchor = {
                node: float(panel.series[row, origin])
                for node, row in occupancy_rows.items()
            }
            if not all(np.isfinite(v) for v in anchor.values()):
                continue
            try:
                rhs = system.rhs(anchor)
            except Exception:  # noqa: BLE001
                continue
            out.append((base_method(str(method)), int(origin), cube, actual, rhs))
    if not out:
        raise AnalysisError(f"{directory} yielded no scorable origin")
    return system, graph, panel, out


def _risk_metrics(
    point: np.ndarray, actual: np.ndarray, capacity: np.ndarray
) -> dict[str, float]:
    """Errors that a crowd-management rule actually reads.

    A referee observed that average MAE is the wrong summary for a method sold on crowd
    safety: an occasional large improvement at a critical moment is worth more than a
    uniform small one, and the submitted paper could not say whether the improvements
    landed near high occupancy or anywhere at all. These three quantities answer that.

    ``peak_error`` is the absolute error in the maximum occupancy reached over the
    horizon, which is the number an operator compares against capacity.
    ``high_occupancy_mae`` restricts the error to the steps where the truth is above
    eighty percent of capacity. ``exceedance_brier`` scores the forecast as a binary
    predictor of whether capacity will be exceeded, read off the quantile fan.
    """
    occupied = np.isfinite(capacity) & (capacity > 0)
    if not occupied.any():
        return {
            "peak_error": float("nan"),
            "high_occupancy_mae": float("nan"),
            "exceedance_brier": float("nan"),
        }
    cap = capacity[occupied][:, None]
    predicted = point[occupied]
    observed = actual[occupied]

    peak = float(np.mean(np.abs(predicted.max(axis=1) - observed.max(axis=1))))
    hot = observed >= 0.8 * cap
    high = float(np.mean(np.abs(predicted[hot] - observed[hot]))) if hot.any() else float("nan")
    # A point forecast at or above capacity is read as predicting exceedance. This is a
    # deliberately crude rule: the comparison across reconcilers is what matters, and all
    # arms are scored by the same rule.
    forecast_event = (predicted >= cap).astype(float)
    true_event = (observed >= cap).astype(float)
    brier = float(np.mean((forecast_event - true_event) ** 2))
    return {
        "peak_error": peak,
        "high_occupancy_mae": high,
        "exceedance_brier": brier,
    }


def reconciliation_study(
    experiment: str,
    site_id: str,
    variant: str,
    seed: int,
    max_origins: int,
    *,
    strategy: str = "shift",
) -> pd.DataFrame:
    """Apply each reconciler to forecasts that were produced without one.

    Five arms rather than four. ``unbounded_uncertainty`` applies the proposed diagonal
    metric with the box removed, which is the arm that makes the comparison attributable:
    against ``mint`` it isolates the weighting matrix, and against ``proposed`` it
    isolates the inequality constraints. Without it, the published MinT comparison
    changes the metric and the constraint set simultaneously.
    """
    from mflow.reconcile import default_registry
    from mflow.reconcile.quantiles import QuantileReconciler, crossing_rate

    system, graph, _panel, origins = _origin_cubes(
        experiment, site_id, variant, seed, max_origins
    )
    registry = default_registry()

    capacity = np.array(
        [
            graph.node_capacity[sid.split(":", 1)[1]]
            if sid.startswith("occ:")
            else np.nan
            for sid in system.series_ids
        ],
        dtype=float,
    )

    median_index = QUANTILES.index(0.5)
    rows: list[dict[str, Any]] = []
    for method, origin, cube, actual, rhs in origins:
        median = cube[:, :, median_index]
        for name in RECONCILERS:
            reconciler = registry.create(name)
            try:
                reconciled, result = QuantileReconciler(
                    reconciler, strategy=strategy
                ).reconcile(cube, list(QUANTILES), system, rhs)
            except Exception as error:  # noqa: BLE001
                rows.append(
                    {
                        "method": method,
                        "reconciler": name,
                        "origin": origin,
                        "failed": str(error)[:80],
                    }
                )
                continue
            point = reconciled[:, :, median_index]
            row = {
                "method": method,
                "reconciler": name,
                "strategy": strategy,
                "origin": origin,
                "residual_before": result.residual_before,
                "residual_after": result.residual_after,
                "violation_before": result.violation_before,
                "violation_after": result.violation_after,
                "mae_before": float(np.mean(np.abs(median - actual))),
                "mae_after": float(np.mean(np.abs(point - actual))),
                "crps_before": _crps(cube, actual),
                "crps_after": _crps(reconciled, actual),
                "solve_time_ms": result.solve_time_ms,
                "crossing_rate_after": crossing_rate(reconciled),
                "failed": "",
            }
            for key, value in _risk_metrics(median, actual, capacity).items():
                row[f"{key}_before"] = value
            for key, value in _risk_metrics(point, actual, capacity).items():
                row[f"{key}_after"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_coherence_study(
    experiment: str, site_id: str, variant: str, seed: int, max_origins: int
) -> pd.DataFrame:
    """How coherent the non-median quantile levels are under each lifting strategy.

    Both referees raised this and both were right. The submitted paper claimed the shift
    strategy keeps the whole predictive distribution coherent. It does not, and the
    reason is one line of algebra: shifting level :math:`\\alpha` by the median's
    correction :math:`c` gives :math:`A(y^{\\alpha} + c) = Ay^{\\alpha} + Ac`, and since
    :math:`Ac = b - Ay^{0.5}`, the residual at level :math:`\\alpha` is
    :math:`Ay^{\\alpha} - Ay^{0.5}`, which vanishes only when the level agrees with the
    median. The median path is coherent; no other level is.

    This function measures the size of that residual instead of arguing about it, and
    compares it against ``per_quantile``, which projects every level separately and so is
    coherent and feasible at every level at the cost that the levels no longer come from
    one joint law and crossings have to be repaired by sorting.
    """
    from mflow.reconcile import default_registry
    from mflow.reconcile.quantiles import QuantileReconciler, crossing_rate

    system, _graph, _panel, origins = _origin_cubes(
        experiment, site_id, variant, seed, max_origins
    )
    registry = default_registry()

    rows: list[dict[str, Any]] = []
    for method, origin, cube, _actual, rhs in origins:
        for strategy in ("shift", "per_quantile"):
            reconciler = registry.create("proposed")
            try:
                reconciled, _ = QuantileReconciler(
                    reconciler, strategy=strategy
                ).reconcile(cube, list(QUANTILES), system, rhs)
            except Exception:  # noqa: BLE001
                continue
            for index, level in enumerate(QUANTILES):
                y = reconciled[:, :, index].reshape(-1)
                rows.append(
                    {
                        "method": method,
                        "origin": origin,
                        "strategy": strategy,
                        "level": level,
                        "residual": float(np.abs(system.residual(y, rhs)).mean()),
                        "violation": float(system.violation(y).sum()),
                    }
                )
            rows.append(
                {
                    "method": method,
                    "origin": origin,
                    "strategy": strategy,
                    "level": float("nan"),
                    "residual": float("nan"),
                    "violation": float("nan"),
                    "crossing_rate": crossing_rate(reconciled),
                }
            )
    return pd.DataFrame(rows)


def sigma_sensitivity_study(
    experiment: str, site_id: str, variant: str, seed: int, max_origins: int
) -> pd.DataFrame:
    """Does the projection depend on the Gaussian mapping from interval width to sigma?

    A referee noted that Equation (6) converts the eighty percent interval to a standard
    deviation through a Gaussian quantile, while occupancy counts are sparse and not
    Gaussian, and asked whether the Gaussian assumption is necessary and how sensitive
    the result is to the choice of interval.

    The honest answer is that it is not necessary, and the reason is visible in the
    objective: the projection minimises a quadratic form in which only the relative
    weights matter, so multiplying every sigma by a positive constant leaves the solution
    unchanged. The Gaussian constant 2.5631 is exactly such a constant, so it divides out
    and the Gaussian reading of the interval has no effect on the projected point. What
    could matter is the *shape* of the map from the fan to the weights, meaning which
    interval is read. These four variants probe that: the published Gaussian mapping, the
    plain eighty percent width, a sixty percent interval, and a forty percent one. The
    first two differ only by a constant and must agree to solver accuracy; the last two
    read a different part of the fan and are free to disagree.
    """
    from mflow.reconcile import ProjectionReconciler
    from mflow.reconcile.projection import DEFAULT_SIGMA_FLOOR

    system, _graph, _panel, origins = _origin_cubes(
        experiment, site_id, variant, seed, max_origins
    )
    median_index = QUANTILES.index(0.5)

    def width(cube: np.ndarray, low: float, high: float) -> np.ndarray:
        return cube[:, :, QUANTILES.index(high)] - cube[:, :, QUANTILES.index(low)]

    # Each variant carries the factor its floor must be multiplied by to stay comparable.
    # The floor is in persons, so a mapping that returns a sigma k times larger needs a
    # floor k times larger, otherwise it clips a different set of coordinates and the
    # comparison would measure the floor rather than the mapping.
    variants: dict[str, tuple[Any, float]] = {
        "gaussian_80": (lambda c: width(c, 0.1, 0.9) / NORMAL_80_SPREAD, 1.0),
        "raw_width_80": (lambda c: width(c, 0.1, 0.9), NORMAL_80_SPREAD),
        "gaussian_60": (lambda c: width(c, 0.2, 0.8) / 1.6832, 1.0),
        "gaussian_40": (lambda c: width(c, 0.3, 0.7) / 1.0488, 1.0),
    }

    rows: list[dict[str, Any]] = []
    for method, origin, cube, actual, rhs in origins:
        median = cube[:, :, median_index]
        reference: np.ndarray | None = None
        for name, (mapping, floor_scale) in variants.items():
            sigma = np.asarray(mapping(cube), dtype=float)
            floor = DEFAULT_SIGMA_FLOOR * floor_scale
            reconciler = ProjectionReconciler(weighting="uncertainty", sigma_floor=floor)
            try:
                result = reconciler.reconcile(median, system, rhs, sigma=sigma)
            except Exception:  # noqa: BLE001
                continue
            point = result.reconciled
            if name == "gaussian_80":
                reference = point
            rows.append(
                {
                    "method": method,
                    "origin": origin,
                    "sigma_variant": name,
                    "mae_after": float(np.mean(np.abs(point - actual))),
                    "max_abs_difference_from_published": (
                        float("nan")
                        if reference is None
                        else float(np.max(np.abs(point - reference)))
                    ),
                    "fraction_at_sigma_floor": float(np.mean(sigma <= floor)),
                    "residual_after": result.residual_after,
                }
            )
        # Scale invariance is a property of the objective, so it is asserted rather than
        # tabulated: if it ever fails, the weighting is not entering as claimed.
        if reference is not None:
            scale = 1000.0
            scaled = ProjectionReconciler(
                weighting="uncertainty",
                # The floor is denominated in persons, so it has to travel with the
                # rescaling or it would clip a different set of coordinates and change
                # the shape of the weights rather than only their level.
                sigma_floor=DEFAULT_SIGMA_FLOOR * scale,
            ).reconcile(
                median, system, rhs,
                sigma=scale * np.asarray(variants["gaussian_80"][0](cube)),
            )
            drift = float(np.max(np.abs(scaled.reconciled - reference)))
            if drift > 1e-4:
                raise AnalysisError(
                    f"scaling every sigma by 1000 moved the projection by {drift:.2e}; "
                    "the objective is supposed to depend on relative weights only"
                )
    return pd.DataFrame(rows)


def _crps(cube: np.ndarray, actual: np.ndarray) -> float:
    """CRPS approximated by the mean pinball loss over the nine quantile levels."""
    levels = np.asarray(QUANTILES)
    error = actual[:, :, None] - cube
    loss = np.maximum(levels * error, (levels - 1.0) * error)
    return float(2.0 * loss.mean())


def block_bootstrap_ci(
    values: np.ndarray, *, block: int = 10, draws: int = 4000, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval for a mean under serial dependence between origins.

    Rolling origins overlap, so consecutive loss differentials are dependent and an
    independent bootstrap understates the interval. A moving-block bootstrap resamples
    contiguous runs of ``block`` origins instead of single origins, which preserves the
    local dependence whatever its form. A referee asked whether the significance results
    survive a dependence adjustment of this kind, and this is what answers it.
    """
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(series.size / block))
    starts = rng.integers(0, series.size - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        index = (starts[draw][:, None] + offsets[None, :]).reshape(-1)[: series.size]
        means[draw] = series[index].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _paired_summary(
    difference: np.ndarray, reference: float, *, alternative: str = "two-sided"
) -> dict[str, Any]:
    """Mean effect, its block-bootstrap interval, and a signed-rank p-value."""
    test = stats.wilcoxon(difference, alternative=alternative)
    low, high = block_bootstrap_ci(difference)
    return {
        "mean_change_percent": float(100.0 * difference.mean() / reference),
        "ci_low_percent": float(100.0 * low / reference),
        "ci_high_percent": float(100.0 * high / reference),
        "p_value": float(test.pvalue),
        "n_origins": int(difference.size),
        "fraction_worsened": float((difference > 0).mean()),
        "median_change_percent": float(100.0 * np.median(difference) / reference),
        "worst_origin_change_percent": float(100.0 * difference.max() / reference),
        "best_origin_change_percent": float(100.0 * difference.min() / reference),
    }


def reconciliation_significance(ok: pd.DataFrame) -> dict[str, Any]:
    """Paired tests for the reconcilers, per forecaster rather than pooled.

    The two forecasters share every origin, so pooling their occasions would treat
    correlated observations as independent. Each forecaster is therefore tested on its
    own, and the pooled figure is reported only as a descriptive summary.

    Every effect now carries a moving-block bootstrap interval alongside its p-value. A
    referee asked for intervals on the mean paired change, and they matter more here than
    usual because the per-origin distribution is strongly asymmetric: the largest mean
    improvement in the submitted table carried the weakest p-value, which a mean and a
    p-value alone cannot explain.
    """
    out: dict[str, Any] = {}
    for name in RECONCILERS[1:]:
        arm = ok[ok["reconciler"] == name]
        per_forecaster: dict[str, Any] = {}
        for forecaster, block in arm.groupby("method"):
            difference = (block["mae_after"] - block["mae_before"]).to_numpy()
            per_forecaster[str(forecaster)] = _paired_summary(
                difference, float(block["mae_before"].mean())
            )
        out[name] = per_forecaster

    for label, left_arm, right_arm in RECON_ABLATION_ARMS:
        ablation: dict[str, Any] = {}
        for forecaster in sorted(ok["method"].unique()):
            left = ok[(ok["reconciler"] == left_arm) & (ok["method"] == forecaster)]
            right = ok[(ok["reconciler"] == right_arm) & (ok["method"] == forecaster)]
            merged = left.merge(right, on="origin", suffixes=("_l", "_r"))
            if merged.empty:
                continue
            difference = (merged["mae_after_l"] - merged["mae_after_r"]).to_numpy()
            ablation[str(forecaster)] = _paired_summary(
                difference, float(merged["mae_after_r"].mean()), alternative="less"
            )
        out[label] = ablation
    return out


def figure_recon_distribution(study: pd.DataFrame, filename: str) -> None:
    """Where the reconciliation gain comes from, origin by origin."""
    ok = study[study["failed"] == ""]
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    names = {"uniform_weight": "uniform", "proposed": "proposed"}
    colours = {"uniform_weight": "#7f7f7f", "proposed": "#2ca02c"}
    for name in ("uniform_weight", "proposed"):
        arm = ok[ok["reconciler"] == name]
        change = 100.0 * (arm["mae_after"] - arm["mae_before"]) / arm["mae_before"].mean()
        axes[0].hist(
            change, bins=60, range=(-8, 8), histtype="step", linewidth=1.3,
            color=colours[name], label=names[name],
        )
    axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("change in MAE at one origin (%)")
    axes[0].set_ylabel("origins")
    axes[0].set_title("(a) the whole distribution, not just its median")
    axes[0].legend(frameon=False, loc="upper left")

    for name in ("uniform_weight", "proposed"):
        arm = ok[ok["reconciler"] == name].sort_values("origin")
        change = 100.0 * (arm["mae_after"] - arm["mae_before"]) / arm["mae_before"].mean()
        axes[1].plot(
            np.sort(change), np.linspace(0, 1, len(change)),
            color=colours[name], linewidth=1.4, label=names[name],
        )
    axes[1].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(-8, 8)
    axes[1].set_xlabel("change in MAE at one origin (%)")
    axes[1].set_ylabel("cumulative fraction of origins")
    axes[1].set_title("(b) how often each one helps")
    axes[1].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    save(figure, filename)


def figure_reconciliation(study: pd.DataFrame, filename: str) -> None:
    """Three panels, because the method has to win on three things at once."""
    ok = study[study["failed"] == ""]
    order = ["none", "mint", "uniform_weight", "proposed"]
    names = {
        "none": "none",
        "mint": "MinT",
        "uniform_weight": "uniform",
        "proposed": "proposed",
    }
    grouped = ok.groupby("reconciler")
    present = [r for r in order if r in grouped.groups]
    labels = [names[r] for r in present]
    good, bad, neutral = "#2ca02c", "#d62728", "#7f7f7f"

    figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))

    residual = [grouped.get_group(r)["residual_after"].mean() for r in present]
    axes[0].bar(
        labels, [max(v, 1e-11) for v in residual],
        color=[bad if v > 1e-6 else good for v in residual], width=0.62,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-11, 10)
    axes[0].set_ylabel("conservation residual (persons)")
    axes[0].set_title("(a) does it balance?")

    violation = [grouped.get_group(r)["violation_after"].mean() for r in present]
    axes[1].bar(
        labels, violation,
        color=[bad if v > 1e-3 else good for v in violation], width=0.62,
    )
    axes[1].set_ylabel("capacity and sign violation (persons)")
    axes[1].set_title("(b) at what cost to the bounds?")

    reference = grouped.get_group("none")["mae_before"].mean()
    change = [
        100.0 * (grouped.get_group(r)["mae_after"].mean() / reference - 1.0)
        for r in present
    ]
    axes[2].bar(
        labels, change,
        color=[neutral if abs(v) < 1e-9 else (good if v < 0 else bad) for v in change],
        width=0.62,
    )
    axes[2].axhline(0.0, color="black", linewidth=0.9)
    axes[2].set_ylabel("change in MAE (%)")
    axes[2].set_title("(c) and to accuracy?")

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    save(figure, filename)


# --------------------------------------------------------------------------- #
# Real-data figures
# --------------------------------------------------------------------------- #


def figure_skill_by_horizon(
    experiment: str, group: str, minutes: dict[int, int], filename: str, title: str,
    *, overlay: str | None = None, overlay_methods: Sequence[str] = (),
) -> pd.DataFrame:
    """MASE against horizon, one line per method, legend clear of the data.

    Args:
        overlay: a second experiment to draw curves from, used to bring the
            re-specified gradient-boosted arms into the headline figure. It is legitimate
            to combine the two because E10 runs the identical site, context, stride and
            horizon ladder as E8, so the curves are over the same origins.
        overlay_methods: which methods to take from ``overlay``.
    """
    metrics = metrics_for(experiment)
    cell = metrics[metrics["series_group"] == group]
    if cell.empty:
        raise AnalysisError(f"{experiment} has no {group!r} metrics")
    table = cell.pivot_table(index="horizon", columns="method", values="mase")
    table.columns = [base_method(str(c)) for c in table.columns]
    if overlay is not None:
        extra = metrics_for(overlay)
        extra = extra[extra["series_group"] == group]
        wide = extra.pivot_table(index="horizon", columns="method", values="mase")
        wide.columns = [base_method(str(c)) for c in wide.columns]
        for name in overlay_methods:
            if name not in wide.columns:
                raise AnalysisError(f"{overlay} has no method {name!r}")
            table[name] = wide[name]
        # The submitted global arm is superseded by the ladder and would otherwise appear
        # twice under two names.
        table = table.drop(columns=[c for c in ("global_lightgbm",) if c in table.columns])

    # Taller than the other panels: the legend sits underneath and nine entries at the
    # enlarged font would otherwise take a third of the height.
    figure, axis = plt.subplots(figsize=(3.4, 3.2))
    for method in order_for_legend(table):
        base = base_method(str(method))
        colour, marker = STYLE.get(base, ("#000000", "."))
        axis.plot(
            [minutes[int(h)] for h in table.index],
            table[method].to_numpy(),
            marker=marker, color=colour, markersize=4, linewidth=1.2,
            label=PRETTY.get(base, base),
        )
    axis.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    axis.set_xlabel("forecast horizon (minutes)")
    axis.set_ylabel("MASE")
    axis.set_title(title)
    legend_below(figure, axis, ncol=2)
    save(figure, filename)
    return table


def _skill_overlay(key: str) -> dict[str, Any]:
    """The overlay arguments that bring the re-specified local arms into Figure 1."""
    # Only the strongest local arm goes into the figure; the full ladder is a table, and
    # drawing all five would bury the comparison the figure exists to make.
    return {"overlay": LIGHTGBM[key], "overlay_methods": ("lightgbm_per_series",)}


def figure_point_versus_probabilistic(filename: str) -> None:
    """MASE and CRPS side by side on HZMetro, which is where they disagree."""
    metrics = metrics_for(REAL["hzmetro"])
    cell = metrics[metrics["series_group"] == "flow"]

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    reference = None
    for axis, column, label, title in (
        (axes[0], "mase", "MASE (point accuracy)", "(a) point accuracy"),
        (axes[1], "crps", "CRPS (probabilistic)", "(b) probabilistic accuracy"),
    ):
        table = cell.pivot_table(index="horizon", columns="method", values=column)
        for method in order_for_legend(table):
            base = base_method(str(method))
            colour, marker = STYLE.get(base, ("#000000", "."))
            axis.plot(
                [HZ_MINUTES[int(h)] for h in table.index],
                table[method].to_numpy(),
                marker=marker, color=colour, markersize=4, linewidth=1.2,
                label=PRETTY.get(base, base),
            )
        axis.set_xlabel("forecast horizon (minutes)")
        axis.set_ylabel(label)
        axis.set_title(title)
        reference = axis
    assert reference is not None
    legend_below(figure, reference, ncol=4, y=-0.02)
    save(figure, filename)


def figure_calibration(filename: str) -> None:
    """Empirical coverage of the nominal 80 percent interval, both real sites."""
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    reference = None
    for axis, key, group, minutes, title in (
        (axes[0], "robod", "occupancy", ROBOD_MINUTES, "(a) ROBOD"),
        (axes[1], "hzmetro", "flow", HZ_MINUTES, "(b) HZMetro"),
    ):
        metrics = metrics_for(REAL[key])
        cell = metrics[metrics["series_group"] == group]
        table = cell.pivot_table(index="horizon", columns="method", values="coverage_80")
        for method in order_for_legend(table):
            base = base_method(str(method))
            colour, marker = STYLE.get(base, ("#000000", "."))
            axis.plot(
                [minutes[int(h)] for h in table.index],
                table[method].to_numpy(),
                marker=marker, color=colour, markersize=4, linewidth=1.2,
                label=PRETTY.get(base, base),
            )
        axis.axhline(0.8, color="black", linewidth=0.9, linestyle="--")
        axis.set_ylim(-0.02, 1.05)
        axis.set_xlabel("forecast horizon (minutes)")
        axis.set_title(title)
        reference = axis
    # Shared axis, so only the left panel is labelled; a label on the right panel would
    # print over the left panel's tick numbers.
    axes[0].set_ylabel("coverage of the 80% interval")
    assert reference is not None
    legend_below(figure, reference, ncol=4, y=-0.02)
    save(figure, filename)


def efficiency_table(experiment: str, group: str, horizon: int) -> pd.DataFrame:
    """Accuracy against the local training budget."""
    metrics = metrics_for(experiment)
    cell = metrics[(metrics["series_group"] == group) & (metrics["horizon"] == horizon)]
    if cell.empty:
        raise AnalysisError(f"{experiment} has no {group!r} metrics at horizon {horizon}")
    table = cell.pivot_table(index="train_days", columns="method", values="mase")
    table.columns = [base_method(str(c)) for c in table.columns]
    return table


def figure_data_efficiency(filename: str) -> dict[str, pd.DataFrame]:
    """Accuracy against the local training budget, on the simulator and on ROBOD.

    The ROBOD panel is new in the revision and exists because a referee objected that the
    submitted paper explained LightGBM's weakness on real occupancy by the shortness of
    the record without ever varying the record. Here the specification is held fixed and
    only the budget moves, so the curve says how much of the gap data quantity can close.
    """
    tables = {
        "simulated": efficiency_table(EFFICIENCY_SIM, "all", 60),
        "robod": efficiency_table(EFFICIENCY_ROBOD, "occupancy", 12),
    }
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for axis, (label, table), title in zip(
        axes,
        tables.items(),
        (
            "(a) simulated museum, 60 min",
            "(b) ROBOD room occupancy, 60 min",
        ),
        strict=True,
    ):
        for method in table.columns:
            colour, marker = STYLE.get(method, ("#000000", "."))
            axis.plot(
                table.index.to_numpy(), table[method].to_numpy(),
                marker=marker, color=colour, markersize=5, linewidth=1.3,
                label=PRETTY.get(method, method),
            )
        axis.set_xscale("log")
        # A log axis adds its own minor labels, which overprint the budget labels.
        axis.minorticks_off()
        axis.set_xticks(list(table.index))
        axis.set_xticklabels([str(int(d)) for d in table.index])
        axis.set_xlabel("days of site-specific training data")
        axis.set_ylabel("MASE")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8, loc="center right")
        del label
    figure.tight_layout()
    save(figure, filename)
    return tables


def figure_between_dataset(
    strengths: dict[str, tuple[float, float]], gains: dict[str, float], filename: str
) -> None:
    """The three-point ordering, shown with the caveat that it is three points."""
    figure, axis = plt.subplots(figsize=(3.6, 2.8))
    labels = sorted(strengths, key=lambda k: strengths[k][0])
    x = np.array([strengths[k][0] for k in labels])
    err = np.array([strengths[k][1] for k in labels])
    y = np.array([gains[k] for k in labels])

    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.plot(x, y, color="#d62728", linewidth=1.2, alpha=0.4, zorder=2, linestyle=":")
    axis.errorbar(
        x, y, xerr=err, fmt="o", markersize=7, color="#d62728",
        ecolor="#d62728", elinewidth=1.0, capsize=3, zorder=3,
    )
    for label, xi, yi in zip(labels, x, y, strict=True):
        axis.annotate(
            label, (xi, yi), textcoords="offset points",
            xytext=(0, -14 if yi > 0 else 10), ha="center", fontsize=8,
        )
    axis.text(
        0.97, 0.92, "zero-shot better", transform=axis.transAxes, ha="right",
        fontsize=7, style="italic", color="#666666",
    )
    axis.text(
        0.97, 0.05, "local baseline better", transform=axis.transAxes, ha="right",
        fontsize=7, style="italic", color="#666666",
    )
    axis.set_xlabel("dataset mean seasonal strength $S_m$")
    axis.set_ylabel("zero-shot MASE gain over\nthe best local baseline (%)")
    axis.set_title("Three datasets, three points")
    axis.margins(x=0.16, y=0.22)
    save(figure, filename)


# --------------------------------------------------------------------------- #
# The gradient-boosted re-specification
# --------------------------------------------------------------------------- #


def table_lightgbm(key: str) -> pd.DataFrame:
    """MASE of the five gradient-boosted specifications against the reference arms.

    The ladder adds one factor at a time, so the table attributes the change rather than
    merely reporting that a better configuration exists: horizon matching, then the
    site's own daily lag, then a hyperparameter search, then removal of the pooling
    across series.
    """
    metrics = metrics_for(LIGHTGBM[key], distinct=True)
    group = CONTEXT_GROUP[key]
    cell = metrics[metrics["series_group"] == group]
    if cell.empty:
        raise AnalysisError(f"{LIGHTGBM[key]} has no {group!r} metrics")
    table = cell.pivot_table(index="method", columns="horizon", values="mase")
    table.index = [base_method(str(m)) for m in table.index]
    minutes = ROBOD_MINUTES if key == "robod" else HZ_MINUTES
    table.columns = [minutes[int(h)] for h in table.columns]
    return table.sort_values(table.columns[-1])


def lightgbm_significance(key: str) -> dict[str, Any]:
    """Is the best local specification beaten by the best foundation model, or not?

    This is the comparison the first hypothesis stands or falls on, and after the
    re-specification it falls on HZMetro. Reporting it needs a paired test on the same
    origins, not a comparison of two column means.
    """
    from mflow.eval.significance import diebold_mariano, holm_bonferroni

    directories = [
        d for d in runs_for(LIGHTGBM[key]) if (d / "predictions.parquet").is_file()
    ]
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    frame = frame.assign(method=[base_method(str(m)) for m in frame["method"]])
    horizon = CONTEXT_HORIZON[key]
    frame = frame[frame["step"] <= horizon]
    frame = frame.assign(ae=(frame["q50"] - frame["actual"]).abs())

    # One loss per (method, origin): the mean absolute error over every series and every
    # step up to the horizon. Aggregating across series before testing is deliberate and
    # is what a referee asked to have stated: series from the same station are dependent,
    # so treating each series as an independent observation would inflate the sample.
    per_origin = frame.groupby(["method", "origin"])["ae"].mean().unstack(0)

    best_zero_shot = min(
        (c for c in per_origin.columns if c in ZERO_SHOT),
        key=lambda c: per_origin[c].mean(),
    )
    locals_ = [c for c in per_origin.columns if c not in ZERO_SHOT]
    raw: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for name in locals_:
        pair = per_origin[[best_zero_shot, name]].dropna()
        result = diebold_mariano(
            pair[best_zero_shot].to_numpy(), pair[name].to_numpy(), horizon=horizon
        )
        raw[name] = float(result.p_value)
        difference = (pair[best_zero_shot] - pair[name]).to_numpy()
        low, high = block_bootstrap_ci(difference)
        detail[name] = {
            "zero_shot_mae": float(pair[best_zero_shot].mean()),
            "local_mae": float(pair[name].mean()),
            "advantage_percent": float(
                100.0 * (1.0 - pair[best_zero_shot].mean() / pair[name].mean())
            ),
            "p_value": float(result.p_value),
            "dm_statistic": float(result.statistic),
            "ci_low": float(low),
            "ci_high": float(high),
            "n_origins": len(pair),
        }
    rejected = holm_bonferroni(raw, alpha=0.05)
    for name, flag in rejected.items():
        detail[name]["holm_significant"] = bool(flag)
    return {"best_zero_shot": best_zero_shot, "comparisons": detail}


# --------------------------------------------------------------------------- #
# Context length
# --------------------------------------------------------------------------- #


def context_sweep(key: str) -> tuple[pd.DataFrame, list[int]]:
    """MASE at the longest horizon against context length, on common origins.

    The sweep's value is that only the amount of history read changes. On HZMetro every
    context scores the identical origin set; on gappy ROBOD a longer context rescues two
    origins a shorter one had to drop, because origin eligibility asks each live series
    for one finite observation inside the context. The frame is therefore restricted to
    the origins common to every context, which is what keeps the comparison paired.
    """
    site = CONTEXT_SITE[key]
    group = CONTEXT_GROUP[key]
    horizon = CONTEXT_HORIZON[key]

    predictions: dict[int, pd.DataFrame] = {}
    for seasons, context in CONTEXT_SWEEP[key].items():
        directory = RESULTS / f"E9_{site}_c{context}_{site}_default_s0"
        path = directory / "predictions.parquet"
        if not path.is_file():
            raise AnalysisError(
                f"{path} is missing. Generate the sweep configs with "
                "scripts/make_context_sweep.py and run them."
            )
        frame = pd.read_parquet(path)
        predictions[seasons] = frame[frame["step"] <= horizon]

    common = set.intersection(
        *(set(frame["origin"].unique()) for frame in predictions.values())
    )
    if not common:
        raise AnalysisError(f"the {key} context sweep shares no origin across contexts")

    metrics: dict[int, pd.Series] = {}
    for seasons, frame in predictions.items():
        kept = frame[frame["origin"].isin(common)]
        prefix = "occ:" if group == "occupancy" else "flow:"
        kept = kept[kept["series_id"].str.startswith(prefix)]
        kept = kept.assign(ae=(kept["q50"] - kept["actual"]).abs())
        metrics[seasons] = kept.groupby("method")["ae"].mean()

    table = pd.DataFrame(metrics)
    table.index = [base_method(str(m)) for m in table.index]
    # Scale to MASE using the same in-training seasonal denominator the harness used, so
    # the numbers are comparable with the main tables rather than being raw MAE.
    reference = metrics_for(f"E9_{site}_c{CONTEXT_SWEEP[key][2]}")
    reference = reference[
        (reference["series_group"] == group) & (reference["horizon"] == horizon)
    ]
    scale = reference.groupby("method").apply(
        lambda block: float(block["mase"].mean() / block["mae"].mean()),
        include_groups=False,
    )
    scale.index = [base_method(str(m)) for m in scale.index]
    factor = float(scale.dropna().median())
    return table * factor, sorted(common)


def figure_context(filename: str) -> dict[str, Any]:
    """Accuracy against context length on both real datasets."""
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    summary: dict[str, Any] = {}
    reference = None
    for axis, key, title in (
        (axes[0], "robod", "ROBOD, room occupancy, 60 min"),
        (axes[1], "hzmetro", "HZMetro, station flow, 120 min"),
    ):
        table, common = context_sweep(key)
        summary[key] = {
            "n_common_origins": len(common),
            "mase": json.loads(table.to_json(orient="index")),
        }
        for method in order_for_legend(table.T):
            colour, marker = STYLE.get(method, ("#000000", "."))
            row = table.loc[method].dropna()
            axis.plot(
                [int(s) for s in row.index],
                row.to_numpy(),
                marker=marker, color=colour, markersize=5, linewidth=1.3,
                label=PRETTY.get(method, method),
            )
        axis.set_xscale("log")
        axis.minorticks_off()
        axis.set_xticks(sorted(CONTEXT_SWEEP[key]))
        axis.set_xticklabels([str(s) for s in sorted(CONTEXT_SWEEP[key])])
        axis.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
        axis.set_xlabel("context length (daily seasons)")
        axis.set_ylabel("MASE")
        axis.set_title(title)
        reference = axis
    assert reference is not None
    # The two panels have very different vertical ranges, so they cannot share an axis.
    # Padding between them keeps the right panel's label off the left panel's ticks.
    figure.subplots_adjust(wspace=0.33)
    legend_below(figure, reference, ncol=4, y=-0.03)
    save(figure, filename)
    return summary


# --------------------------------------------------------------------------- #
# Calibration separated from sharpness
# --------------------------------------------------------------------------- #


def calibration_table(key: str, horizon: int) -> pd.DataFrame:
    """Coverage, width, and width rescaled to a common empirical coverage.

    A referee observed, correctly, that the submitted table did not establish better
    calibration: on ROBOD the seasonal baseline covered 83.9 percent against a nominal
    80 while Toto-2.0 covered 90.1, so the baseline was the closer to nominal and the
    foundation model was merely sharper. Those are two different virtues and the table
    conflated them.

    The fix is to report both separately and then to compare sharpness at matched
    coverage. Each method's interval is rescaled by the single factor that would bring
    its own empirical coverage to eighty percent, estimated from its own absolute errors
    relative to its half-width, and the resulting widths are comparable because every
    method is then covering the same fraction of outcomes.
    """
    directories = [d for d in runs_for(REAL[key]) if (d / "predictions.parquet").is_file()]
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    frame = frame[frame["step"] <= horizon]
    prefix = "occ:" if CONTEXT_GROUP[key] == "occupancy" else "flow:"
    frame = frame[frame["series_id"].str.startswith(prefix)]

    rows: list[dict[str, Any]] = []
    for method, block in frame.groupby("method"):
        low = block["q10"].to_numpy(dtype=float)
        high = block["q90"].to_numpy(dtype=float)
        actual = block["actual"].to_numpy(dtype=float)
        median = block["q50"].to_numpy(dtype=float)
        ok = np.isfinite(low) & np.isfinite(high) & np.isfinite(actual)
        low, high, actual, median = low[ok], high[ok], actual[ok], median[ok]
        if actual.size == 0:
            continue
        covered = (actual >= low) & (actual <= high)
        width = high - low
        half = width / 2.0
        # The factor that would make exactly eighty percent of outcomes fall inside the
        # interval, if the interval were scaled about its median. A method whose interval
        # has collapsed cannot be rescaled at all, because scaling zero leaves zero, and
        # for it the matched-coverage comparison is undefined rather than favourable.
        # Last-value on ROBOD is exactly this case: its interval is degenerate and it
        # reaches 82 percent coverage only because room occupancy is often unchanged from
        # one step to the next, so it scores well on coverage without quantifying
        # anything. Reporting a matched width of zero for it would be an artefact.
        usable = half > 1e-6
        if usable.mean() < 0.5:
            factor = float("nan")
            matched = float("nan")
            achieved = float("nan")
        else:
            standardised = np.abs(actual[usable] - median[usable]) / half[usable]
            factor = float(np.quantile(standardised, 0.8))
            matched = float(width.mean() * factor)
            rescaled_half = factor * half
            achieved = float(
                (np.abs(actual - median) <= rescaled_half + 1e-9).mean()
            )
        rows.append(
            {
                "method": base_method(str(method)),
                "coverage": float(covered.mean()),
                "width": float(width.mean()),
                "calibration_error": float(abs(covered.mean() - 0.8)),
                "scale_to_nominal": factor,
                "width_at_matched_coverage": matched,
                "coverage_after_rescaling": achieved,
                "fraction_of_intervals_degenerate": float(1.0 - usable.mean()),
            }
        )
    table = pd.DataFrame(rows).set_index("method")
    return table.sort_values("width_at_matched_coverage")


# --------------------------------------------------------------------------- #
# Dependence between origins
# --------------------------------------------------------------------------- #


def dependence_check(key: str) -> dict[str, Any]:
    """How the headline p-value moves with the dependence adjustment.

    A referee asked three things: how per-origin loss differentials were aggregated
    across series, how the Newey-West truncation was chosen relative to the origin
    stride, and whether the conclusions survive a block-based interval. All three are
    answered here rather than argued.

    Aggregation is across series first, giving one loss per origin, because series from
    the same station are dependent and treating each series as an independent observation
    would inflate the sample by the number of series.

    The truncation deserves care, and checking it turned up something worth stating. The
    published choice is ``h - 1``, which is the standard rule for overlapping multi-step
    forecasts and is expressed in *steps*. The differentials being tested are indexed by
    *origin*, and consecutive origins are ``stride`` steps apart, so the overlap between
    forecast windows extends only ``ceil(h / stride) - 1`` origins. On HZMetro that is
    three origins against a truncation of seven, and on ROBOD it is zero against eleven.
    The published lag is therefore the more conservative of the two, not the looser one:
    it smooths over more autocovariances than the design implies and so inflates the
    variance estimate and the p-value. Both are reported.
    """
    from mflow.eval.significance import diebold_mariano

    directories = [d for d in runs_for(REAL[key]) if (d / "predictions.parquet").is_file()]
    frame = pd.read_parquet(directories[0] / "predictions.parquet")
    frame = frame.assign(method=[base_method(str(m)) for m in frame["method"]])
    horizon = CONTEXT_HORIZON[key]
    stride = 12 if key == "robod" else 2
    frame = frame[frame["step"] <= horizon]
    prefix = "occ:" if CONTEXT_GROUP[key] == "occupancy" else "flow:"
    frame = frame[frame["series_id"].str.startswith(prefix)]
    frame = frame.assign(ae=(frame["q50"] - frame["actual"]).abs())
    per_origin = frame.groupby(["method", "origin"])["ae"].mean().unstack(0)

    zero_shot = min(
        (c for c in per_origin.columns if c in ZERO_SHOT),
        key=lambda c: per_origin[c].mean(),
    )
    baseline = min(
        (c for c in per_origin.columns if c not in ZERO_SHOT),
        key=lambda c: per_origin[c].mean(),
    )
    pair = per_origin[[zero_shot, baseline]].dropna()
    left = pair[zero_shot].to_numpy()
    right = pair[baseline].to_numpy()
    difference = left - right

    # Overlap between consecutive origins, counted in origins rather than steps.
    overlap_lag = max(0, int(np.ceil(horizon / stride)) - 1)
    published = diebold_mariano(left, right, horizon=horizon)
    design_implied = diebold_mariano(left, right, horizon=overlap_lag + 1)
    low, high = block_bootstrap_ci(difference)
    autocorr = (
        float(pd.Series(difference).autocorr(lag=1)) if difference.size > 2 else float("nan")
    )
    return {
        "zero_shot": zero_shot,
        "baseline": baseline,
        "n_origins": int(difference.size),
        "stride_steps": stride,
        "horizon_steps": horizon,
        "newey_west_lag_published": int(published.lag),
        "newey_west_lag_design_implied": int(design_implied.lag),
        "p_value_published_lag": float(published.p_value),
        "p_value_design_implied_lag": float(design_implied.p_value),
        "lag1_autocorrelation_of_differential": autocorr,
        "mean_differential": float(difference.mean()),
        "block_bootstrap_ci": [float(low), float(high)],
        "block_bootstrap_excludes_zero": bool(high < 0.0 or low > 0.0),
    }


# --------------------------------------------------------------------------- #
# The simulator, described rather than asserted
# --------------------------------------------------------------------------- #


def simulator_profile(site_id: str) -> dict[str, Any]:
    """Distributional summaries of a simulated site, for the plausibility check.

    A referee asked how arrival rates, route choices, dwell times, room capacities and
    peak crowding were chosen, and for evidence that the generated distributions are
    plausible. The choices live in the site YAML and are reported in the paper; what this
    function adds is the realised behaviour, which is the part that can be compared
    against the published measurements the configuration was calibrated against.
    """
    from mflow.graph import BuildingGraph
    from mflow.schema import load_site

    site = load_site(ROOT / "data" / "canonical" / site_id)
    graph = BuildingGraph.from_site(site)
    panel = site.to_panel()
    per_day = round(24 * 3600 / site.meta.interval_seconds)

    rooms: dict[str, Any] = {}
    for i, sid in enumerate(panel.series_ids):
        if not sid.startswith("occ:"):
            continue
        node = sid.split(":", 1)[1]
        capacity = float(graph.node_capacity[node])
        values = panel.series[i]
        values = values[np.isfinite(values)]
        if values.size == 0 or capacity <= 0:
            continue
        rooms[node] = {
            "capacity": capacity,
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "fraction_at_capacity": float((values >= capacity).mean()),
            "fraction_above_80pc": float((values >= 0.8 * capacity).mean()),
            "peak_to_mean": (
                float(values.max() / values.mean()) if values.mean() > 0 else float("nan")
            ),
        }

    occupancy_rows = [
        i for i, sid in enumerate(panel.series_ids) if sid.startswith("occ:")
    ]
    total = np.nansum(panel.series[occupancy_rows], axis=0)
    daily_peak = [
        float(np.nanmax(total[d * per_day : (d + 1) * per_day]))
        for d in range(total.size // per_day)
        if np.isfinite(total[d * per_day : (d + 1) * per_day]).any()
    ]
    entries = [
        i
        for i, sid in enumerate(panel.series_ids)
        if sid.startswith("flow:") and "door" in sid and sid.endswith("_fwd")
    ]
    arrivals = (
        [
            float(np.nansum(panel.series[entries][:, d * per_day : (d + 1) * per_day]))
            for d in range(total.size // per_day)
        ]
        if entries
        else []
    )
    return {
        "site_id": site_id,
        "interval_seconds": site.meta.interval_seconds,
        "n_days": total.size // per_day,
        "rooms": rooms,
        "building_peak_occupancy_mean": float(np.mean(daily_peak)) if daily_peak else float("nan"),
        "building_peak_occupancy_p95": (
            float(np.quantile(daily_peak, 0.95)) if daily_peak else float("nan")
        ),
        "visitors_per_day_mean": (
            float(np.mean([a for a in arrivals if a > 0])) if arrivals else float("nan")
        ),
        "fraction_of_room_steps_at_capacity": float(
            np.mean([r["fraction_at_capacity"] for r in rooms.values()])
        ),
    }


def figure_simulator(filename: str) -> None:
    """What the simulated buildings actually produce, at both operating points."""
    figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    from mflow.graph import BuildingGraph
    from mflow.schema import load_site

    colours = {"sim_house_museum": "#1f77b4", "sim_house_museum_crowded": "#d62728"}
    labels = {"sim_house_museum": "baseline", "sim_house_museum_crowded": "crowded"}

    for site_id, colour in colours.items():
        site = load_site(ROOT / "data" / "canonical" / site_id)
        graph = BuildingGraph.from_site(site)
        panel = site.to_panel()
        per_day = round(24 * 3600 / site.meta.interval_seconds)

        occupancy_rows = [
            i for i, sid in enumerate(panel.series_ids) if sid.startswith("occ:")
        ]
        total = np.nansum(panel.series[occupancy_rows], axis=0)
        profile = np.nanmean(
            total[: (total.size // per_day) * per_day].reshape(-1, per_day), axis=0
        )
        hours = np.arange(per_day) * site.meta.interval_seconds / 3600.0
        axes[0].plot(hours, profile, color=colour, linewidth=1.3, label=labels[site_id])

        utilisation = []
        for i, sid in enumerate(panel.series_ids):
            if not sid.startswith("occ:"):
                continue
            capacity = float(graph.node_capacity[sid.split(":", 1)[1]])
            values = panel.series[i]
            values = values[np.isfinite(values)]
            if capacity > 0 and values.size:
                utilisation.append(values / capacity)
        pooled = np.concatenate(utilisation)
        axes[1].hist(
            np.clip(pooled, 0, 1.05), bins=40, histtype="step", density=True,
            linewidth=1.3, color=colour, label=labels[site_id],
        )

        dwell = np.nanmean(panel.series[occupancy_rows], axis=1)
        axes[2].plot(
            np.arange(1, dwell.size + 1), np.sort(dwell)[::-1],
            marker="o", markersize=3, linewidth=1.2, color=colour, label=labels[site_id],
        )

    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("building occupancy (persons)")
    axes[0].set_title("(a) mean daily profile")
    axes[0].set_xlim(0, 24)
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("room occupancy / capacity")
    axes[1].set_ylabel("density")
    axes[1].set_title("(b) how often capacity binds")

    axes[2].set_xlabel("room, ranked")
    axes[2].set_ylabel("mean occupancy (persons)")
    axes[2].set_title("(c) load across rooms")

    figure.tight_layout()
    save(figure, filename)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    print("checking, per method, whether seeds are independent replicates")
    determinism = {
        name: assert_seeds_are_replicates(experiment)
        for name, experiment in REAL.items()
    }
    summary["seeds_are_bit_identical"] = determinism
    for name, per_method in determinism.items():
        identical = [m for m, same in per_method.items() if same]
        varying = [m for m, same in per_method.items() if not same]
        print(f"  {name:8s} bit-identical across seeds: {', '.join(identical)}")
        print(f"  {name:8s} genuinely stochastic:      {', '.join(varying) or 'none'}")

    print("\nfigures: skill by horizon")
    robod = figure_skill_by_horizon(
        REAL["robod"], "occupancy", ROBOD_MINUTES, "skill_robod.png",
        "ROBOD, room occupancy", **_skill_overlay("robod"),
    )
    hz = figure_skill_by_horizon(
        REAL["hzmetro"], "flow", HZ_MINUTES, "skill_hzmetro.png",
        "HZMetro, station flow", **_skill_overlay("hzmetro"),
    )
    robod.to_csv(TABLES / "mase_robod.csv")
    hz.to_csv(TABLES / "mase_hzmetro.csv")

    print("figures: point versus probabilistic, calibration, data efficiency")
    figure_point_versus_probabilistic("point_vs_probabilistic.png")
    figure_calibration("calibration.png")
    for label, table in figure_data_efficiency("data_efficiency.png").items():
        table.to_csv(TABLES / f"data_efficiency_{label}.csv")
        summary.setdefault("data_efficiency", {})[label] = json.loads(
            table.to_json(orient="index")
        )

    print("\nthe gradient-boosted re-specification (E10)")
    summary["lightgbm"] = {}
    for key in ("robod", "hzmetro"):
        ladder = table_lightgbm(key)
        ladder.to_csv(TABLES / f"lightgbm_{key}.csv")
        print(f"  {key}: MASE by horizon")
        print(ladder.round(3).to_string().replace("\n", "\n    "))
        tests = lightgbm_significance(key)
        summary["lightgbm"][key] = {
            "mase": json.loads(ladder.to_json(orient="index")),
            "significance": tests,
        }
        best = tests["best_zero_shot"]
        for name, stat in sorted(
            tests["comparisons"].items(), key=lambda kv: kv[1]["local_mae"]
        ):
            verdict = "zero-shot ahead" if stat["advantage_percent"] > 0 else "LOCAL AHEAD"
            print(
                f"    {best} vs {name:26s} {stat['advantage_percent']:+6.1f}%  "
                f"p={stat['p_value']:.2e}  holm={stat['holm_significant']}  {verdict}"
            )

    print("\ncontext length sweep (E9)")
    summary["context_sweep"] = figure_context("context_sweep.png")
    for key in ("robod", "hzmetro"):
        table, common = context_sweep(key)
        table.to_csv(TABLES / f"context_{key}.csv")
        print(f"  {key}: MASE at the longest horizon, {len(common)} common origins")
        print(table.round(3).to_string().replace("\n", "\n    "))

    print("\ncalibration separated from sharpness")
    summary["calibration"] = {}
    for key, horizon in (("robod", 12), ("hzmetro", 4)):
        table = calibration_table(key, horizon)
        table.to_csv(TABLES / f"calibration_{key}.csv")
        summary["calibration"][key] = json.loads(table.to_json(orient="index"))
        print(f"  {key} at horizon {horizon}")
        print(table.round(3).to_string().replace("\n", "\n    "))

    print("\ndependence between origins")
    summary["dependence"] = {key: dependence_check(key) for key in ("robod", "hzmetro")}
    for key, stat in summary["dependence"].items():
        print(
            f"  {key}: {stat['zero_shot']} vs {stat['baseline']}, n={stat['n_origins']}, "
            f"lag1 rho={stat['lag1_autocorrelation_of_differential']:+.3f}"
        )
        print(
            f"    p at published lag {stat['newey_west_lag_published']} = "
            f"{stat['p_value_published_lag']:.3e}; at design-implied lag "
            f"{stat['newey_west_lag_design_implied']} = "
            f"{stat['p_value_design_implied_lag']:.3e}; block bootstrap CI "
            f"{stat['block_bootstrap_ci'][0]:+.4f} to {stat['block_bootstrap_ci'][1]:+.4f} "
            f"(excludes zero: {stat['block_bootstrap_excludes_zero']})"
        )

    print("\nsimulator profile, at both operating points")
    summary["simulator"] = {
        name: simulator_profile(site)
        for name, site in (
            ("baseline", "sim_house_museum"),
            ("crowded", "sim_house_museum_crowded"),
        )
    }
    figure_simulator("simulator_profile.png")
    for name, profile in summary["simulator"].items():
        print(
            f"  {name:9s} {profile['n_days']:4d} days, peak occupancy "
            f"{profile['building_peak_occupancy_mean']:.1f} persons, "
            f"{100 * profile['fraction_of_room_steps_at_capacity']:.1f}% of room-steps "
            f"at capacity"
        )

    print("\nbetween-dataset ordering (three points, reported as such)")
    strengths = {}
    for label, site, group in (
        ("ROBOD", "robod_bldg1", "occ:"),
        ("HZMetro", "hzmetro", "flow:"),
        ("Simulated", "sim_house_museum", "occ:"),
    ):
        values = per_series_strength(site, group)
        strengths[label] = (
            float(values.mean()),
            float(values.std(ddof=1) / np.sqrt(len(values))),
        )
    summary["seasonal_strength"] = strengths

    def gain(metrics: pd.DataFrame, group: str, horizon: int) -> float:
        cell = metrics[(metrics["series_group"] == group) & (metrics["horizon"] == horizon)]
        by = cell.groupby("method")["mase"].mean()
        fm = min(v for k, v in by.items() if base_method(str(k)) in ZERO_SHOT)
        local = min(v for k, v in by.items() if base_method(str(k)) not in ZERO_SHOT)
        return 100.0 * (1.0 - fm / local)

    gains = {
        "ROBOD": gain(metrics_for(REAL["robod"]), "occupancy", 12),
        "HZMetro": gain(metrics_for(REAL["hzmetro"]), "flow", 8),
        "Simulated": gain(metrics_for(EFFICIENCY_SIM), "all", 60),
    }
    summary["zero_shot_gain_percent"] = gains
    for name, (mean, err) in strengths.items():
        print(f"  {name:10s} S_m = {mean:.3f} +/- {err:.3f}   gain = {gains[name]:+.1f}%")
    figure_between_dataset(strengths, gains, "between_dataset.png")

    print("\nwithin-dataset regression: the real test of the hypothesis")
    regressions = [
        within_dataset_regression(
            REAL["hzmetro"], "hzmetro", "flow:", "timesfm3_multivariate", "seasonal_naive"
        ),
        within_dataset_regression(
            REAL["robod"], "robod_bldg1", "occ:", "toto2", "last_value"
        ),
    ]
    for result in regressions:
        print(
            f"  {result['experiment']:12s} n={result['n_series']:3d}  "
            f"slope={result['slope']:8.1f}  "
            f"95% CI [{result['slope_ci_low']:.0f}, {result['slope_ci_high']:.0f}]  "
            f"p={result['p_value']:.3f}  R2={result['r_squared']:.3f}  "
            f"rho={result['spearman_rho']:+.3f}"
        )
    figure_within_dataset(regressions, "within_dataset.png")
    summary["within_dataset_regression"] = [
        {k: v for k, v in r.items() if k != "frame"} for r in regressions
    ]
    pd.DataFrame(summary["within_dataset_regression"]).to_csv(
        TABLES / "within_dataset.csv", index=False
    )

    print("\nreconciliation study (offline, from stored forecasts)")
    summary["reconciliation"] = {}
    summary["reconciliation_significance"] = {}
    for label, (experiment, site_id, variant) in RECON_SITES.items():
        study = reconciliation_study(experiment, site_id, variant, 0, max_origins=10_000)
        study.to_csv(TABLES / f"reconciliation_raw_{label}.csv", index=False)
        ok = study[study["failed"] == ""]
        if ok.empty:
            raise AnalysisError(
                f"every reconciliation failed on the {label} site; see "
                f"reconciliation_raw_{label}.csv"
            )
        digest = ok.groupby("reconciler").agg(
            residual_before=("residual_before", "mean"),
            residual_after=("residual_after", "mean"),
            violation_before=("violation_before", "mean"),
            violation_after=("violation_after", "mean"),
            mae_before=("mae_before", "mean"),
            mae_after=("mae_after", "mean"),
            crps_before=("crps_before", "mean"),
            crps_after=("crps_after", "mean"),
            peak_error_before=("peak_error_before", "mean"),
            peak_error_after=("peak_error_after", "mean"),
            high_occupancy_mae_before=("high_occupancy_mae_before", "mean"),
            high_occupancy_mae_after=("high_occupancy_mae_after", "mean"),
            exceedance_brier_before=("exceedance_brier_before", "mean"),
            exceedance_brier_after=("exceedance_brier_after", "mean"),
            solve_time_ms=("solve_time_ms", "median"),
            n=("origin", "count"),
        )
        digest = digest.reindex([r for r in RECONCILERS if r in digest.index])
        digest.to_csv(TABLES / f"reconciliation_{label}.csv")
        summary["reconciliation"][label] = json.loads(digest.to_json(orient="index"))
        print(f"  --- {label} site ({site_id})")
        print(
            digest[
                [
                    "residual_after",
                    "violation_after",
                    "mae_after",
                    "crps_after",
                    "peak_error_after",
                    "high_occupancy_mae_after",
                    "solve_time_ms",
                ]
            ]
            .round(4)
            .to_string()
            .replace("\n", "\n    ")
        )
        significance = reconciliation_significance(ok)
        summary["reconciliation_significance"][label] = significance
        # Change in MAE as a percentage of the unreconciled MAE, so negative is better.
        # The mean and the median are both shown because they disagree in sign here, and
        # that disagreement is the finding: most origins are nudged slightly worse while
        # the average improves, so the gain comes from a minority of large corrections.
        # The interval is a moving-block bootstrap of the mean; the p-value is a signed
        # rank test, which is a statement about the median.
        print("    (change in MAE, negative is better; CI is on the mean)")
        print(
            f"    {'arm':22s} {'forecaster':22s} {'mean':>7s} "
            f"{'95% CI':>17s} {'median':>7s} {'p':>9s} {'worse':>6s}"
        )
        for name in (*RECONCILERS[1:], *RECON_ABLATIONS):
            for forecaster, stat in significance[name].items():
                print(
                    f"    {name:22s} {forecaster:22s} "
                    f"{stat['mean_change_percent']:+6.2f}% "
                    f"[{stat['ci_low_percent']:+6.2f},{stat['ci_high_percent']:+6.2f}] "
                    f"{stat['median_change_percent']:+6.2f}% "
                    f"{stat['p_value']:9.2e} "
                    f"{100 * stat['fraction_worsened']:5.0f}%"
                )
        if label == "baseline":
            figure_reconciliation(study, "reconciliation.png")
            figure_recon_distribution(study, "reconciliation_distribution.png")

    print("\nhow coherent are the non-median quantile levels?")
    experiment, site_id, variant = RECON_SITES["baseline"]
    coherence = quantile_coherence_study(
        experiment, site_id, variant, 0, max_origins=60
    )
    coherence.to_csv(TABLES / "quantile_coherence_raw.csv", index=False)
    levels = coherence[np.isfinite(coherence["level"])]
    digest = levels.groupby(["strategy", "level"]).agg(
        residual=("residual", "mean"), violation=("violation", "mean")
    )
    digest.to_csv(TABLES / "quantile_coherence.csv")
    summary["quantile_coherence"] = json.loads(
        digest.reset_index().to_json(orient="records")
    )
    print(digest.round(6).to_string().replace("\n", "\n    "))
    crossings = (
        coherence[coherence["crossing_rate"].notna()]
        if "crossing_rate" in coherence
        else pd.DataFrame()
    )
    if not crossings.empty:
        rates = crossings.groupby("strategy")["crossing_rate"].mean()
        summary["quantile_crossing_rate"] = json.loads(rates.to_json())
        print("  crossing rate before the monotone repair:")
        print(rates.round(6).to_string().replace("\n", "\n    "))

    print("\nsensitivity of the projection to the sigma mapping")
    sigma = sigma_sensitivity_study(experiment, site_id, variant, 0, max_origins=60)
    sigma.to_csv(TABLES / "sigma_sensitivity_raw.csv", index=False)
    digest = sigma.groupby("sigma_variant").agg(
        mae_after=("mae_after", "mean"),
        max_difference_from_published=("max_abs_difference_from_published", "max"),
        fraction_at_sigma_floor=("fraction_at_sigma_floor", "mean"),
        residual_after=("residual_after", "mean"),
        n=("origin", "count"),
    )
    digest.to_csv(TABLES / "sigma_sensitivity.csv")
    summary["sigma_sensitivity"] = json.loads(digest.to_json(orient="index"))
    print(digest.round(6).to_string().replace("\n", "\n    "))

    (Path(__file__).resolve().parent / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote figures to {FIGURES} and tables to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
