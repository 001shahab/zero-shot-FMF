"""Build every figure and table in the paper from the logged runs.

Nothing here invents a number. Each artefact is derived from ``results/<run_id>/``, which
is written only by ``mflow run``, and the script fails loudly if a run it needs is absent
rather than emitting a placeholder.

Two analyses are computed offline from stored forecasts rather than from a fresh sweep.
The conservation residual of a raw forecast and the effect of reconciling it are both
functions of the quantile forecast and the building graph alone, so they can be evaluated
from ``predictions.parquet`` without re-running any model. This is exact, not an
approximation: the reconciler sees precisely the vector the harness would have handed it.

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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
TABLES = Path(__file__).resolve().parent / "tables"

QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]
# The 10-90 interval of a normal covers 2 * 1.2816 standard deviations.
NORMAL_80_SPREAD = 2.5631

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
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

# A fixed colour and marker per method, so that a reader who has learned the legend of one
# figure has learned the legend of all of them.
STYLE: dict[str, tuple[str, str]] = {
    "toto2": ("#d62728", "o"),
    "chronos2_multivariate": ("#9467bd", "s"),
    "chronos2_univariate": ("#9467bd", "s"),
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
    "chronos2_univariate": "Chronos-2 (uv)",
    "timesfm3_univariate": "TimesFM-3 (uv)",
    "timesfm3_multivariate": "TimesFM-3 (mv)",
    "last_value": "Last value",
    "seasonal_naive": "Seasonal naive",
    "historical_average": "Historical average",
    "global_lightgbm": "LightGBM (trained)",
}


class AnalysisError(RuntimeError):
    """Raised when a required run is missing, rather than guessing at its contents."""


def save(figure: "plt.Figure", filename: str) -> None:
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


# --------------------------------------------------------------------------- #
# Loading
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


def metrics_for(experiment: str) -> pd.DataFrame:
    """Per-origin metrics pooled over the seeds of one experiment."""
    return pd.concat(
        [pd.read_parquet(d / "metrics.parquet") for d in runs_for(experiment)],
        ignore_index=True,
    )


def base_method(name: str) -> str:
    """Strip the reconciler suffix from a method label."""
    return name.split("+")[0]


# --------------------------------------------------------------------------- #
# Seasonal strength, the quantity the central hypothesis is stated in
# --------------------------------------------------------------------------- #


def seasonal_strength(series: np.ndarray, period: int) -> float:
    """Fraction of variance explained by the mean seasonal profile.

    For a series :math:`y_t` with period :math:`m`, let :math:`\\bar{y}^{(m)}_{t \\bmod m}`
    be the mean over all observations sharing a phase. The statistic is

    .. math:: S_m = 1 - \\frac{\\operatorname{Var}(y_t - \\bar{y}^{(m)}_{t \\bmod m})}
                              {\\operatorname{Var}(y_t)},

    which is zero when the seasonal profile carries no information and approaches one
    when the series is a deterministic function of phase. It is the population quantity
    a seasonal-naive forecaster exploits, which is why the hypothesis is stated in it.

    NaNs are dropped pairwise, so a gappy record contributes only its observed steps.
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


def site_seasonal_strength(site_id: str, group: str) -> tuple[float, float]:
    """Mean and standard error of daily seasonal strength over a site's series."""
    from mflow.schema import load_site

    site = load_site(ROOT / "data" / "canonical" / site_id)
    panel = site.to_panel()
    period = int(round(24 * 3600 / site.meta.interval_seconds))
    prefix = "occ:" if group == "occupancy" else "flow:"
    values = [
        seasonal_strength(panel.series[i], period)
        for i, sid in enumerate(panel.series_ids)
        if sid.startswith(prefix)
    ]
    good = np.array([v for v in values if np.isfinite(v)])
    if good.size == 0:
        raise AnalysisError(f"no usable {group} series on {site_id}")
    return float(good.mean()), float(good.std(ddof=1) / np.sqrt(good.size))


# --------------------------------------------------------------------------- #
# Reconciliation, evaluated offline from the stored forecasts
# --------------------------------------------------------------------------- #


def reconciliation_study(
    experiment: str, site_id: str, variant: str, seed: int, max_origins: int
) -> pd.DataFrame:
    """Apply each reconciler to forecasts that were produced without one.

    The projection is a function of the quantile forecast, the graph and the occupancy
    anchor, so it can be applied after the fact to a stored forecast and gives exactly
    the vector the harness would have produced online.
    """
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
    # Only interior nodes carry a conservation constraint; the virtual exterior absorbs
    # whatever crosses the building boundary and is deliberately unconstrained.
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
            present = [s for s in series_ids if s in wide.index]
            if len(present) != len(series_ids):
                continue
            cube = np.stack(
                [wide[q].loc[series_ids].to_numpy(dtype=float) for q in QCOLS], axis=-1
            )
            actual = truth.loc[series_ids].to_numpy(dtype=float)
            if not np.isfinite(cube).all() or not np.isfinite(actual).all():
                continue

            # The anchor is the occupancy actually observed at the origin, which is the
            # step before the first forecast step.
            anchor = {
                node: float(panel.series[row, origin])
                for node, row in occupancy_rows.items()
            }
            if not all(np.isfinite(v) for v in anchor.values()):
                continue
            try:
                rhs = system.rhs(anchor)
            except Exception:  # noqa: BLE001 - a missing anchor is not an analysis error
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


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def figure_skill_by_horizon(
    experiment: str, group: str, minutes: dict[int, int], filename: str, title: str
) -> pd.DataFrame:
    """MASE against horizon, one line per method."""
    metrics = metrics_for(experiment)
    cell = metrics[metrics["series_group"] == group]
    if cell.empty:
        raise AnalysisError(f"{experiment} has no {group!r} metrics")
    table = cell.pivot_table(index="horizon", columns="method", values="mase")

    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    order = table.mean().sort_values().index
    for method in order:
        base = base_method(str(method))
        colour, marker = STYLE.get(base, ("#000000", "."))
        axis.plot(
            [minutes[int(h)] for h in table.index],
            table[method].to_numpy(),
            marker=marker,
            color=colour,
            markersize=4,
            linewidth=1.2,
            label=PRETTY.get(base, base),
        )
    axis.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    axis.set_xlabel("forecast horizon (minutes)")
    axis.set_ylabel("MASE")
    axis.set_title(title)
    axis.legend(frameon=False, ncol=1, loc="best")
    save(figure, filename)
    return table


def figure_point_versus_probabilistic(filename: str) -> None:
    """MASE and CRPS side by side on HZMetro, which is where they disagree."""
    metrics = metrics_for("E7_hzmetro")
    cell = metrics[metrics["series_group"] == "flow"]
    minutes = {1: 15, 2: 30, 4: 60, 8: 120}

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for axis, column, label in (
        (axes[0], "mase", "MASE (point accuracy)"),
        (axes[1], "crps", "CRPS (probabilistic)"),
    ):
        table = cell.pivot_table(index="horizon", columns="method", values=column)
        for method in table.mean().sort_values().index:
            base = base_method(str(method))
            colour, marker = STYLE.get(base, ("#000000", "."))
            axis.plot(
                [minutes[int(h)] for h in table.index],
                table[method].to_numpy(),
                marker=marker,
                color=colour,
                markersize=4,
                linewidth=1.2,
                label=PRETTY.get(base, base),
            )
        axis.set_xlabel("forecast horizon (minutes)")
        axis.set_ylabel(label)
    axes[0].set_title("(a) point accuracy: the baseline wins")
    axes[1].set_title("(b) probabilistic: it does not, below one hour")
    axes[1].legend(frameon=False, loc="best")
    save(figure, filename)


def figure_calibration(filename: str) -> None:
    """Empirical coverage of the nominal 80 percent interval, both real sites."""
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for axis, experiment, group, minutes, title in (
        (axes[0], "E7_robod", "occupancy", {1: 5, 3: 15, 6: 30, 12: 60}, "(a) ROBOD"),
        (
            axes[1],
            "E7_hzmetro",
            "flow",
            {1: 15, 2: 30, 4: 60, 8: 120},
            "(b) HZMetro",
        ),
    ):
        metrics = metrics_for(experiment)
        cell = metrics[metrics["series_group"] == group]
        table = cell.pivot_table(index="horizon", columns="method", values="coverage_80")
        for method in table.columns:
            base = base_method(str(method))
            colour, marker = STYLE.get(base, ("#000000", "."))
            axis.plot(
                [minutes[int(h)] for h in table.index],
                table[method].to_numpy(),
                marker=marker,
                color=colour,
                markersize=4,
                linewidth=1.2,
                label=PRETTY.get(base, base),
            )
        axis.axhline(0.8, color="black", linewidth=0.9, linestyle="--")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("forecast horizon (minutes)")
        axis.set_ylabel("empirical coverage of the 80% interval")
        axis.set_title(title)
    axes[1].legend(frameon=False, loc="center right")
    save(figure, filename)


def figure_data_efficiency(filename: str) -> pd.DataFrame:
    """E4: accuracy against the local training budget."""
    metrics = metrics_for("E4")
    cell = metrics[(metrics["series_group"] == "all") & (metrics["horizon"] == 60)]
    table = cell.pivot_table(index="train_days", columns="method", values="mase")

    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    for method in table.columns:
        base = base_method(str(method))
        colour, marker = STYLE.get(base, ("#000000", "."))
        axis.plot(
            table.index.to_numpy(),
            table[method].to_numpy(),
            marker=marker,
            color=colour,
            markersize=4,
            linewidth=1.2,
            label=PRETTY.get(base, base),
        )
    axis.set_xscale("log")
    axis.set_xticks(list(table.index))
    axis.set_xticklabels([str(int(d)) for d in table.index])
    axis.set_xlabel("days of site-specific training data")
    axis.set_ylabel("MASE at 60 minutes")
    axis.set_title("Data efficiency, simulated museum")
    axis.legend(frameon=False, loc="best")
    save(figure, filename)
    return table


def figure_hypothesis(
    strengths: dict[str, tuple[float, float]],
    gains: dict[str, float],
    filename: str,
) -> None:
    """The central claim: zero-shot gain falls as seasonal strength rises."""
    figure, axis = plt.subplots(figsize=(3.6, 2.8))
    labels = sorted(strengths, key=lambda k: strengths[k][0])
    x = np.array([strengths[k][0] for k in labels])
    err = np.array([strengths[k][1] for k in labels])
    y = np.array([gains[k] for k in labels])

    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.plot(x, y, color="#d62728", linewidth=1.2, alpha=0.45, zorder=2)
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
        0.97, 0.90, "zero-shot better", transform=axis.transAxes, ha="right",
        fontsize=7, style="italic", color="#666666",
    )
    axis.text(
        0.97, 0.05, "local baseline better", transform=axis.transAxes, ha="right",
        fontsize=7, style="italic", color="#666666",
    )
    axis.set_xlabel("daily seasonal strength $S_m$ of the target series")
    axis.set_ylabel("zero-shot MASE gain over\nthe best local baseline (%)")
    axis.set_title("Zero-shot advantage falls as periodicity rises")
    axis.margins(x=0.16, y=0.22)
    save(figure, filename)


def figure_reconciliation(study: pd.DataFrame, filename: str) -> None:
    """Three panels, because the method has to win on three things at once.

    Coherence alone is easy, and MinT gets it. The question is what else a reconciler
    breaks on the way, and whether the correction is placed where the forecast was
    actually weak.
    """
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
        labels,
        [max(v, 1e-11) for v in residual],
        color=[bad if v > 1e-6 else good for v in residual],
        width=0.62,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-11, 10)
    axes[0].set_ylabel("conservation residual (persons)")
    axes[0].set_title("(a) does it balance?")

    violation = [grouped.get_group(r)["violation_after"].mean() for r in present]
    axes[1].bar(
        labels,
        violation,
        color=[bad if v > 1e-3 else good for v in violation],
        width=0.62,
    )
    axes[1].set_ylabel("capacity and sign violation (persons)")
    axes[1].set_title("(b) at what cost to the bounds?")

    reference = grouped.get_group("none")["mae_before"].mean()
    change = [
        100.0 * (grouped.get_group(r)["mae_after"].mean() / reference - 1.0)
        for r in present
    ]
    axes[2].bar(
        labels,
        change,
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
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    robod_minutes = {1: 5, 3: 15, 6: 30, 12: 60}
    hz_minutes = {1: 15, 2: 30, 4: 60, 8: 120}

    print("figure: skill by horizon")
    robod = figure_skill_by_horizon(
        "E7_robod", "occupancy", robod_minutes, "skill_robod.png", "ROBOD, room occupancy"
    )
    hz = figure_skill_by_horizon(
        "E7_hzmetro", "flow", hz_minutes, "skill_hzmetro.png", "HZMetro, station flow"
    )
    robod.to_csv(TABLES / "mase_robod.csv")
    hz.to_csv(TABLES / "mase_hzmetro.csv")

    print("figure: point versus probabilistic")
    figure_point_versus_probabilistic("point_vs_probabilistic.png")

    print("figure: calibration")
    figure_calibration("calibration.png")

    print("figure: data efficiency")
    efficiency = figure_data_efficiency("data_efficiency.png")
    efficiency.to_csv(TABLES / "data_efficiency.csv")

    print("seasonal strength")
    strengths = {
        "ROBOD": site_seasonal_strength("robod_bldg1", "occupancy"),
        "HZMetro": site_seasonal_strength("hzmetro", "flow"),
        "Simulated": site_seasonal_strength("sim_house_museum", "occupancy"),
    }
    summary["seasonal_strength"] = {k: v for k, v in strengths.items()}
    for name, (mean, err) in strengths.items():
        print(f"  {name:10s} S_m = {mean:.3f} +/- {err:.3f}")

    # The gain is measured at the longest horizon each experiment reports, between the
    # best zero-shot model and the best method that needs no foundation model at all.
    def gain(metrics: pd.DataFrame, group: str, horizon: int, zero_shot: set[str]) -> float:
        cell = metrics[(metrics["series_group"] == group) & (metrics["horizon"] == horizon)]
        by = cell.groupby("method")["mase"].mean()
        fm = min(v for k, v in by.items() if base_method(str(k)) in zero_shot)
        local = min(v for k, v in by.items() if base_method(str(k)) not in zero_shot)
        return 100.0 * (1.0 - fm / local)

    zero_shot = {
        "toto2",
        "chronos2_multivariate",
        "chronos2_univariate",
        "timesfm3_univariate",
        "timesfm3_multivariate",
    }
    gains = {
        "ROBOD": gain(metrics_for("E7_robod"), "occupancy", 12, zero_shot),
        "HZMetro": gain(metrics_for("E7_hzmetro"), "flow", 8, zero_shot),
        "Simulated": gain(metrics_for("E4"), "all", 60, zero_shot),
    }
    summary["zero_shot_gain_percent"] = gains
    for name, value in gains.items():
        print(f"  {name:10s} gain = {value:+.1f}%")

    figure_hypothesis(strengths, gains, "hypothesis.png")

    print("reconciliation study (offline, from stored forecasts)")
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

    # Paired over forecast occasions, because every reconciler saw the same raw forecast
    # at the same origin. An unpaired test here would be answering a different question.
    from scipy import stats

    print("\npaired tests against the raw forecast (per-origin MAE)")
    significance: dict[str, dict[str, float]] = {}
    for name in ("mint", "uniform_weight", "proposed"):
        arm = ok[ok["reconciler"] == name].set_index(["method", "origin"])
        difference = (arm["mae_after"] - arm["mae_before"]).to_numpy()
        result = stats.wilcoxon(difference, alternative="two-sided")
        significance[name] = {
            "mean_change_percent": float(
                100.0 * difference.mean() / arm["mae_before"].mean()
            ),
            "p_value": float(result.pvalue),
            "n_pairs": int(difference.size),
        }
        print(
            f"  {name:15s} {significance[name]['mean_change_percent']:+.2f}%  "
            f"p={result.pvalue:.2e}  n={difference.size}"
        )
    # And the ablation that the whole design rests on: does weighting by the model's own
    # uncertainty beat weighting everything equally?
    left = ok[ok["reconciler"] == "proposed"].set_index(["method", "origin"])["mae_after"]
    right = ok[ok["reconciler"] == "uniform_weight"].set_index(["method", "origin"])[
        "mae_after"
    ]
    shared = left.index.intersection(right.index)
    ablation = stats.wilcoxon(
        left.loc[shared].to_numpy() - right.loc[shared].to_numpy(), alternative="less"
    )
    significance["proposed_vs_uniform"] = {
        "mean_change_percent": float(
            100.0 * (left.loc[shared].mean() / right.loc[shared].mean() - 1.0)
        ),
        "p_value": float(ablation.pvalue),
        "n_pairs": int(shared.size),
    }
    print(
        f"  proposed vs uniform "
        f"{significance['proposed_vs_uniform']['mean_change_percent']:+.2f}%  "
        f"p={ablation.pvalue:.2e}  n={shared.size}"
    )
    summary["reconciliation_significance"] = significance
    figure_reconciliation(study, "reconciliation.png")
    summary["reconciliation"] = json.loads(digest.to_json(orient="index"))

    (Path(__file__).resolve().parent / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote figures to {FIGURES} and tables to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
