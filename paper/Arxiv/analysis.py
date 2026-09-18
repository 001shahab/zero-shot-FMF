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

Usage:
    python paper/Arxiv/analysis.py
"""

from __future__ import annotations

import json
import sys
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

ZERO_SHOT = {
    "toto2",
    "chronos2_multivariate",
    "chronos2_univariate",
    "timesfm3_univariate",
    "timesfm3_multivariate",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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
}
PRETTY = {
    "toto2": "Toto-2.0",
    "chronos2_multivariate": "Chronos-2 (mv)",
    "timesfm3_univariate": "TimesFM-3 (uv)",
    "timesfm3_multivariate": "TimesFM-3 (mv)",
    "last_value": "Last value",
    "seasonal_naive": "Seasonal naive",
    "historical_average": "Historical average",
    "global_lightgbm": "LightGBM (trained)",
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


def reconciliation_study(
    experiment: str, site_id: str, variant: str, seed: int, max_origins: int
) -> pd.DataFrame:
    """Apply each reconciler to forecasts that were produced without one."""
    from mflow.graph import BuildingGraph
    from mflow.reconcile import default_registry
    from mflow.reconcile.constraints import build_constraints
    from mflow.reconcile.quantiles import QuantileReconciler
    from mflow.schema import load_site

    directory = RESULTS / f"{experiment}_{site_id}_{variant}_s{seed}"
    if not (directory / "predictions.parquet").is_file():
        raise AnalysisError(f"{directory} has no predictions.parquet")
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
    registry = default_registry()

    rows: list[dict[str, Any]] = []
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

            median = cube[:, :, QUANTILES.index(0.5)]
            for name in ("none", "mint", "uniform_weight", "proposed"):
                reconciler = registry.create(name)
                try:
                    reconciled, result = QuantileReconciler(
                        reconciler, strategy="shift"
                    ).reconcile(cube, list(QUANTILES), system, rhs)
                except Exception as error:  # noqa: BLE001
                    rows.append(
                        {
                            "method": base_method(str(method)),
                            "reconciler": name,
                            "origin": int(origin),
                            "failed": str(error)[:80],
                        }
                    )
                    continue
                point = reconciled[:, :, QUANTILES.index(0.5)]
                rows.append(
                    {
                        "method": base_method(str(method)),
                        "reconciler": name,
                        "origin": int(origin),
                        "residual_before": result.residual_before,
                        "residual_after": result.residual_after,
                        "violation_before": result.violation_before,
                        "violation_after": result.violation_after,
                        "mae_before": float(np.mean(np.abs(median - actual))),
                        "mae_after": float(np.mean(np.abs(point - actual))),
                        "crps_before": _crps(cube, actual),
                        "crps_after": _crps(reconciled, actual),
                        "solve_time_ms": result.solve_time_ms,
                        "failed": "",
                    }
                )
    return pd.DataFrame(rows)


def _crps(cube: np.ndarray, actual: np.ndarray) -> float:
    """CRPS approximated by the mean pinball loss over the nine quantile levels."""
    levels = np.asarray(QUANTILES)
    error = actual[:, :, None] - cube
    loss = np.maximum(levels * error, (levels - 1.0) * error)
    return float(2.0 * loss.mean())


def reconciliation_significance(ok: pd.DataFrame) -> dict[str, Any]:
    """Paired tests for the reconcilers, per forecaster rather than pooled.

    The two forecasters share every origin, so pooling their occasions would treat
    correlated observations as independent. Each forecaster is therefore tested on its
    own, and the pooled figure is reported only as a descriptive summary.
    """
    out: dict[str, Any] = {}
    for name in ("mint", "uniform_weight", "proposed"):
        arm = ok[ok["reconciler"] == name]
        per_forecaster: dict[str, Any] = {}
        for forecaster, block in arm.groupby("method"):
            difference = (block["mae_after"] - block["mae_before"]).to_numpy()
            test = stats.wilcoxon(difference, alternative="two-sided")
            per_forecaster[str(forecaster)] = {
                "mean_change_percent": float(
                    100.0 * difference.mean() / block["mae_before"].mean()
                ),
                "p_value": float(test.pvalue),
                "n_origins": int(difference.size),
                "fraction_worsened": float((difference > 0).mean()),
                "worst_origin_change_percent": float(
                    100.0 * difference.max() / block["mae_before"].mean()
                ),
                "best_origin_change_percent": float(
                    100.0 * difference.min() / block["mae_before"].mean()
                ),
            }
        out[name] = per_forecaster
    # The ablation the design rests on, again per forecaster.
    ablation: dict[str, Any] = {}
    for forecaster in sorted(ok["method"].unique()):
        left = ok[(ok["reconciler"] == "proposed") & (ok["method"] == forecaster)]
        right = ok[(ok["reconciler"] == "uniform_weight") & (ok["method"] == forecaster)]
        merged = left.merge(right, on="origin", suffixes=("_p", "_u"))
        difference = (merged["mae_after_p"] - merged["mae_after_u"]).to_numpy()
        test = stats.wilcoxon(difference, alternative="less")
        ablation[str(forecaster)] = {
            "mean_change_percent": float(
                100.0 * difference.mean() / merged["mae_after_u"].mean()
            ),
            "p_value": float(test.pvalue),
            "n_origins": int(difference.size),
        }
    out["proposed_vs_uniform"] = ablation
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
    experiment: str, group: str, minutes: dict[int, int], filename: str, title: str
) -> pd.DataFrame:
    """MASE against horizon, one line per method, legend clear of the data."""
    metrics = metrics_for(experiment)
    cell = metrics[metrics["series_group"] == group]
    if cell.empty:
        raise AnalysisError(f"{experiment} has no {group!r} metrics")
    table = cell.pivot_table(index="horizon", columns="method", values="mase")

    figure, axis = plt.subplots(figsize=(3.4, 2.7))
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
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
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
        axis.set_ylabel("coverage of the 80% interval")
        axis.set_title(title)
        reference = axis
    assert reference is not None
    legend_below(figure, reference, ncol=4, y=-0.02)
    save(figure, filename)


def figure_data_efficiency(filename: str) -> pd.DataFrame:
    """E4: accuracy against the local training budget."""
    metrics = metrics_for("E4")
    cell = metrics[(metrics["series_group"] == "all") & (metrics["horizon"] == 60)]
    table = cell.pivot_table(index="train_days", columns="method", values="mase")

    figure, axis = plt.subplots(figsize=(3.4, 2.7))
    for method in table.columns:
        base = base_method(str(method))
        colour, marker = STYLE.get(base, ("#000000", "."))
        axis.plot(
            table.index.to_numpy(), table[method].to_numpy(),
            marker=marker, color=colour, markersize=4, linewidth=1.2,
            label=PRETTY.get(base, base),
        )
    axis.set_xscale("log")
    axis.set_xticks(list(table.index))
    axis.set_xticklabels([str(int(d)) for d in table.index])
    axis.set_xlabel("days of site-specific training data")
    axis.set_ylabel("MASE at 60 minutes")
    axis.set_title("Data efficiency, simulated museum")
    legend_below(figure, axis, ncol=2)
    save(figure, filename)
    return table


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
        "ROBOD, room occupancy",
    )
    hz = figure_skill_by_horizon(
        REAL["hzmetro"], "flow", HZ_MINUTES, "skill_hzmetro.png",
        "HZMetro, station flow",
    )
    robod.to_csv(TABLES / "mase_robod.csv")
    hz.to_csv(TABLES / "mase_hzmetro.csv")

    print("figures: point versus probabilistic, calibration, data efficiency")
    figure_point_versus_probabilistic("point_vs_probabilistic.png")
    figure_calibration("calibration.png")
    efficiency = figure_data_efficiency("data_efficiency.png")
    efficiency.to_csv(TABLES / "data_efficiency.csv")

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
        "Simulated": gain(metrics_for("E4"), "all", 60),
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
    study = reconciliation_study("E4", "sim_house_museum", "d090", 0, max_origins=10_000)
    study.to_csv(TABLES / "reconciliation_raw.csv", index=False)
    ok = study[study["failed"] == ""]
    if ok.empty:
        raise AnalysisError("every reconciliation failed; see reconciliation_raw.csv")
    digest = ok.groupby("reconciler").agg(
        residual_before=("residual_before", "mean"),
        residual_after=("residual_after", "mean"),
        violation_after=("violation_after", "mean"),
        mae_before=("mae_before", "mean"),
        mae_after=("mae_after", "mean"),
        crps_before=("crps_before", "mean"),
        crps_after=("crps_after", "mean"),
        solve_time_ms=("solve_time_ms", "median"),
        n=("origin", "count"),
    )
    digest.to_csv(TABLES / "reconciliation.csv")
    print(digest.round(4).to_string())
    summary["reconciliation"] = json.loads(digest.to_json(orient="index"))
    figure_reconciliation(study, "reconciliation.png")
    figure_recon_distribution(study, "reconciliation_distribution.png")

    print("\npaired tests, per forecaster (the two share every origin)")
    significance = reconciliation_significance(ok)
    for name in ("mint", "uniform_weight", "proposed"):
        for forecaster, stat in significance[name].items():
            print(
                f"  {name:15s} {forecaster:24s} {stat['mean_change_percent']:+.2f}%  "
                f"p={stat['p_value']:.2e}  n={stat['n_origins']}  "
                f"worsened {100 * stat['fraction_worsened']:.0f}% of origins"
            )
    for forecaster, stat in significance["proposed_vs_uniform"].items():
        print(
            f"  proposed vs uniform {forecaster:20s} {stat['mean_change_percent']:+.2f}%  "
            f"p={stat['p_value']:.2e}  n={stat['n_origins']}"
        )
    summary["reconciliation_significance"] = significance

    (Path(__file__).resolve().parent / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote figures to {FIGURES} and tables to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
