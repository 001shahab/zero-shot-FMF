"""Build the paper's tables and figures from ``results/`` (M6).

Everything here reads logged runs and writes nothing but derivations of them. There is no
path through this module that can produce a number which did not come out of a run: a
requested table whose runs are absent raises, and an absent run is never filled in with a
placeholder, a dash or an interpolation. That is ground rule 2 of the specification, and
it is enforced by :func:`load_runs` refusing to return an empty collection.

Each artefact is written twice, as CSV and as LaTeX, so that the manuscript never contains
a number retyped by hand. Figures are written as PDF and PNG beside them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from mflow.eval.significance import SignificanceError, diebold_mariano, holm_bonferroni
from mflow.manifest import MANIFEST_FILE, RunManifest
from mflow.paths import ensure_dir, results_dir

#: Files a completed run is expected to contain.
PREDICTIONS_FILE = "predictions.parquet"
METRICS_FILE = "metrics.parquet"
RESOURCES_FILE = "resources.parquet"
RECONCILIATION_FILE = "reconciliation.parquet"
RISK_FILE = "risk.parquet"


class ReportError(RuntimeError):
    """Raised when a requested artefact cannot be built from the logged runs."""


@dataclass(frozen=True)
class Run:
    """One logged run on disk.

    Attributes:
        directory: the ``results/<run_id>/`` directory.
        manifest: its manifest.
        metrics: the per-origin metrics frame.
    """

    directory: Path
    manifest: RunManifest
    metrics: pd.DataFrame

    @property
    def run_id(self) -> str:
        """The run identifier."""
        return self.manifest.run_id

    @property
    def experiment(self) -> str:
        """The experiment this run belongs to, e.g. ``E1``."""
        return self.manifest.experiment

    def frame(self, filename: str) -> pd.DataFrame:
        """Load one of the run's parquet files.

        Raises:
            ReportError: if the file is absent. A run that did not write its resources
                frame did not measure latency, and a latency table assembled from the
                runs that happen to have one would silently be about a different subset
                of methods than the main table.
        """
        path = self.directory / filename
        if not path.is_file():
            raise ReportError(f"run {self.run_id!r} has no {filename}: {path} does not exist")
        return pd.read_parquet(path)

    def config_value(self, key: str, default: Any = None) -> Any:
        """Read a key out of the run's recorded configuration."""
        return self.manifest.config.get(key, default)


def load_runs(
    root: Path | None = None,
    *,
    experiment: str | None = None,
    require: bool = True,
) -> list[Run]:
    """Load every completed run under ``results/``.

    A directory counts as a completed run when it holds both a manifest and a metrics
    file. A directory with a manifest but no metrics is a run that was started and did
    not finish; it is skipped rather than partially reported.

    Args:
        root: the results directory. Defaults to ``results/``.
        experiment: keep only runs of this experiment.
        require: raise when nothing matches. Leave this on for anything that feeds the
            manuscript.

    Raises:
        ReportError: if ``require`` is set and no run matches.
    """
    base = results_dir() if root is None else Path(root)
    runs: list[Run] = []
    if base.is_dir():
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            if not (directory / MANIFEST_FILE).is_file():
                continue
            metrics_path = directory / METRICS_FILE
            if not metrics_path.is_file():
                continue
            manifest = RunManifest.read(directory)
            if experiment is not None and manifest.experiment != experiment:
                continue
            runs.append(
                Run(
                    directory=directory,
                    manifest=manifest,
                    metrics=pd.read_parquet(metrics_path),
                )
            )
    if require and not runs:
        scope = "any experiment" if experiment is None else f"experiment {experiment!r}"
        raise ReportError(
            f"no completed run for {scope} under {base}. Run the experiment first; this "
            "module reports results, it does not invent them."
        )
    return runs


def combine(runs: Iterable[Run], filename: str = METRICS_FILE) -> pd.DataFrame:
    """Stack one frame across runs, tagging each row with its run and seed."""
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = run.metrics if filename == METRICS_FILE else run.frame(filename)
        frames.append(
            frame.assign(
                run_id=run.run_id,
                experiment=run.experiment,
                seed=run.manifest.seed,
                site_id=run.config_value("site_id"),
            )
        )
    if not frames:
        raise ReportError(f"no run supplied a {filename} to combine")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_table(frame: pd.DataFrame, directory: Path, name: str, *, caption: str) -> list[Path]:
    """Write one table as CSV and as LaTeX.

    Args:
        frame: the table.
        directory: where to write it.
        name: base filename, without an extension.
        caption: LaTeX caption, which must state what the numbers are and which runs
            produced them.
    """
    target = ensure_dir(Path(directory))
    csv_path = target / f"{name}.csv"
    tex_path = target / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        frame.to_latex(
            index=False,
            float_format="%.3f",
            caption=caption,
            label=f"tab:{name}",
            escape=True,
        ),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def write_figure(figure: Any, directory: Path, name: str) -> list[Path]:
    """Write one figure as PDF and PNG and close it."""
    target = ensure_dir(Path(directory))
    paths = []
    for suffix in ("pdf", "png"):
        path = target / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    figure.clf()
    return paths


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def main_table(
    metrics: pd.DataFrame,
    *,
    reference: str,
    horizons: Sequence[int] | None = None,
    groups: Sequence[str] = ("occupancy", "flow"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Methods against horizons, MAE and WQL, occupancy and flow separately.

    Every method is compared against ``reference`` by a Diebold-Mariano test on the
    paired per-origin MAE, and the family of comparisons within each group and horizon is
    corrected by Holm-Bonferroni. The ``significant`` column has three states, because
    they are three different facts:

    ``*``
        better than the reference, and the difference survives the correction.
    `` `` (empty)
        tested, and the difference did not survive. The number is still reported; it
        simply carries no claim.
    ``n/a``
        the test could not be run, almost always because the run has too few origins to
        support the Harvey-Leybourne-Newbold correction at that horizon. The reason is
        recorded verbatim in the ``significance_note`` column. This is deliberately not
        collapsed into the empty state: "we tested and found nothing" and "we could not
        test" must not look the same in a manuscript.

    Args:
        metrics: per-origin metrics, as written by the harness.
        reference: the method every other is tested against.
        horizons: which horizons to report. Defaults to all present.
        groups: which series groups to report.
        alpha: family-wise error rate.

    Raises:
        ReportError: if the reference method is absent from the metrics, or a requested
            group and horizon has no metrics at all.
    """
    _require_method(metrics, reference)
    wanted = sorted(metrics["horizon"].unique()) if horizons is None else list(horizons)

    rows: list[dict[str, Any]] = []
    for group in groups:
        for horizon in wanted:
            cell = metrics[
                (metrics["series_group"] == group) & (metrics["horizon"] == horizon)
            ]
            if cell.empty:
                raise ReportError(
                    f"no metrics for group {group!r} at horizon {horizon}; the main table "
                    "cannot be assembled from runs that did not evaluate it"
                )
            verdicts, note = _significance_against(cell, reference, alpha=alpha)
            for method, sub in cell.groupby("method"):
                if str(method) == reference:
                    marker, why = "", "reference method"
                elif str(method) in verdicts:
                    better, significant = verdicts[str(method)]
                    marker = "*" if (better and significant) else ""
                    why = ""
                else:
                    marker, why = "n/a", note
                rows.append(
                    {
                        "series_group": group,
                        "horizon": horizon,
                        "method": method,
                        "mae": sub["mae"].mean(),
                        "wql": sub["wql"].mean(),
                        "mase": sub["mase"].mean(),
                        "crps": sub["crps"].mean(),
                        "coverage_80": sub["coverage_80"].mean(),
                        "n_origins": len(sub.drop_duplicates(pair_index(sub))),
                        "significant": marker,
                        "significance_note": why,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["series_group", "horizon", "mae"], ignore_index=True
    )


#: Columns that, taken together, identify one forecast occasion. A paired test must line
#: methods up on all of them: origin index 34566 of one site is not the same occasion as
#: origin 34566 of another, and pooling runs without these keys would average unrelated
#: losses into a single row and shrink the variance the test depends on.
OCCASION_KEYS = ("run_id", "site_id", "seed", "variant", "origin")


def pair_index(frame: pd.DataFrame) -> list[str]:
    """The occasion keys present in ``frame``, for use as a pivot index."""
    return [key for key in OCCASION_KEYS if key in frame.columns]


def _require_method(metrics: pd.DataFrame, method: str) -> None:
    available = sorted(metrics["method"].unique())
    if method not in available:
        raise ReportError(f"method {method!r} is not in the metrics; available: {available}")


def _significance_against(
    cell: pd.DataFrame, reference: str, *, alpha: float, loss: str = "mae"
) -> tuple[dict[str, tuple[bool, bool]], str]:
    """Per-method ``(is_better, is_significant)`` against the reference, Holm-corrected.

    Returns:
        The verdicts for the methods that could be tested, and a note explaining why the
        others could not. A method absent from the mapping was not tested; the caller
        must render it as ``n/a`` rather than as a failed test.

    When several runs are pooled the differentials are ordered by run and then by origin,
    so each run's block is contiguous and the Newey-West window only strays across a
    boundary at the joins. That is the usual treatment of a panel of forecast origins and
    it is mildly conservative here, since losses from different sites are less correlated
    than losses from adjacent origins of one site.
    """
    pivot = cell.pivot_table(index=pair_index(cell), columns="method", values=loss)
    if reference not in pivot.columns:
        return {}, f"the reference {reference!r} was not run at this group and horizon"
    horizon = int(cell["horizon"].iloc[0])
    p_values: dict[str, float] = {}
    better: dict[str, bool] = {}
    note = ""
    for method in pivot.columns:
        if method == reference:
            continue
        paired = pivot[[str(method), reference]].dropna()
        if paired.shape[0] < 3:
            note = note or (
                f"only {paired.shape[0]} paired origin(s) against {reference!r}"
            )
            continue
        try:
            result = diebold_mariano(
                paired[str(method)].to_numpy(), paired[reference].to_numpy(), horizon=horizon
            )
        except SignificanceError as error:
            # The test genuinely cannot be run here -- almost always too few origins for
            # the horizon. That is a fact about the run, so it is carried into the table
            # rather than raised: refusing the whole table would also suppress the
            # horizons where the test was perfectly well defined.
            note = note or str(error)
            continue
        p_values[str(method)] = result.p_value
        better[str(method)] = bool(result.mean_difference < 0)
    verdicts = holm_bonferroni(p_values, alpha=alpha)
    return {name: (better[name], verdicts[name]) for name in p_values}, note


def reconciliation_table(
    metrics: pd.DataFrame, reconciliation: pd.DataFrame
) -> pd.DataFrame:
    """Error, coherence residual and violation rate before and after reconciliation.

    The point of the table is that the projection buys coherence without costing
    accuracy, so both columns have to appear side by side; reporting the residual alone
    would hide a projection that achieved it by moving the forecast somewhere silly.
    """
    accuracy = (
        metrics[metrics["series_group"] == "all"]
        .groupby(["method", "horizon"], as_index=False)
        .agg(mae=("mae", "mean"), wql=("wql", "mean"), violation_rate=("violation_rate", "mean"))
    )
    coherence = reconciliation.groupby("method", as_index=False).agg(
        residual_before=("residual_before", "mean"),
        residual_after=("residual_after", "mean"),
        violation_before=("violation_before", "mean"),
        violation_after=("violation_after", "mean"),
        adjustment_norm=("adjustment_norm", "mean"),
        solve_time_ms=("solve_time_ms", "median"),
        # A run under a sensor profile may have had to carry the conservation anchor
        # forward over a dropout. The mean staleness belongs beside the residual, because
        # a projection anchored on a stale observation is coherent with respect to a
        # different starting point than the one the reader will assume.
        anchor_staleness_steps=("anchor_staleness_steps", "mean"),
    )
    merged = accuracy.merge(coherence, on="method", how="left")
    return merged.sort_values(["horizon", "mae"], ignore_index=True)


def latency_table(resources: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Median and tail latency and peak memory per method, beside its accuracy.

    The median is what a deployment feels on an average tick and the 95th percentile is
    what decides whether it keeps up on a busy one, so both are reported.
    """
    timing = resources.groupby("method", as_index=False).agg(
        latency_ms_median=("latency_ms", "median"),
        latency_ms_p95=("latency_ms", lambda s: float(np.percentile(s, 95))),
        peak_memory_mb=("peak_memory_mb", "max"),
        n_calls=("latency_ms", "size"),
    )
    accuracy = (
        metrics[metrics["series_group"] == "all"]
        .groupby("method", as_index=False)
        .agg(mae=("mae", "mean"))
    )
    return timing.merge(accuracy, on="method", how="left").sort_values(
        "latency_ms_median", ignore_index=True
    )


def covariate_table(metrics: pd.DataFrame, *, condition_column: str = "covariates") -> pd.DataFrame:
    """Accuracy by covariate set, for the E2 ablation.

    Raises:
        ReportError: if the runs did not record which covariate set they used, because a
            table whose rows cannot be attributed to a condition says nothing.
    """
    if condition_column not in metrics.columns:
        raise ReportError(
            f"the metrics carry no {condition_column!r} column, so the covariate ablation "
            "cannot be attributed to a condition. Record it in the run config."
        )
    return (
        metrics[metrics["series_group"] != "all"]
        .groupby(["method", condition_column, "series_group", "horizon"], as_index=False)
        .agg(mae=("mae", "mean"), wql=("wql", "mean"), crps=("crps", "mean"))
        .sort_values(["series_group", "horizon", "method", condition_column], ignore_index=True)
    )


# --------------------------------------------------------------------------- #
# Data efficiency
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Crossover:
    """Where a trained method overtakes a zero-shot one.

    Attributes:
        days_below: the largest training budget at which the zero-shot method is still
            ahead, or None when it is never ahead.
        days_above: the smallest budget at which the trained method is ahead, or None
            when it never gets ahead within the range that was run.
        significant: whether the difference at ``days_above`` survives the paired test.
            A crossover that is not significant is a crossing of two noisy curves and
            must be drawn, and described, as such.
    """

    days_below: int | None
    days_above: int | None
    significant: bool

    def describe(self) -> str:
        """One sentence for a caption."""
        if self.days_above is None:
            return "no crossover within the budgets that were run"
        if self.days_below is None:
            return f"the trained method leads from the smallest budget run ({self.days_above} days)"
        marker = "significant" if self.significant else "not significant"
        return (
            f"crossover between {self.days_below} and {self.days_above} days ({marker} "
            "at the first budget where the trained method leads)"
        )


def data_efficiency(
    metrics: pd.DataFrame,
    *,
    zero_shot: str,
    trained: str,
    horizon: int,
    series_group: str = "all",
    days_column: str = "train_days",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, Crossover]:
    """The E4 curve and the budget at which the trained method overtakes.

    Args:
        metrics: per-origin metrics across the data-efficiency runs.
        zero_shot: the method whose accuracy does not depend on the budget.
        trained: the method being given more local history.
        horizon: which horizon the curve is drawn at.
        series_group: which group.
        days_column: the column recording each run's training budget in days.
        alpha: level for the significance flag on the crossover.

    Returns:
        A frame with one row per budget, carrying both methods' mean error and a
        bootstrap confidence interval, and the crossover.

    Raises:
        ReportError: if the budget column is missing or a method is absent.
    """
    if days_column not in metrics.columns:
        raise ReportError(
            f"the metrics carry no {days_column!r} column; a data-efficiency curve cannot "
            "be drawn without knowing how much history each run was given"
        )
    for method in (zero_shot, trained):
        _require_method(metrics, method)

    cell = metrics[
        (metrics["series_group"] == series_group) & (metrics["horizon"] == horizon)
    ]
    rows: list[dict[str, Any]] = []
    verdicts: dict[int, tuple[float, bool]] = {}
    for key, sub in cell.groupby(days_column):
        days = int(cast("int", key))
        pivot = sub.pivot_table(index=pair_index(sub), columns="method", values="mae")
        if zero_shot not in pivot.columns or trained not in pivot.columns:
            raise ReportError(
                f"budget {days} was not run for both {zero_shot!r} and {trained!r}"
            )
        paired = pivot[[zero_shot, trained]].dropna()
        difference = paired[trained] - paired[zero_shot]
        low, high = _bootstrap_ci(difference.to_numpy())
        try:
            test = diebold_mariano(
                paired[trained].to_numpy(), paired[zero_shot].to_numpy(), horizon=horizon
            )
            p_value = test.p_value
        except SignificanceError as error:
            raise ReportError(
                f"cannot test the crossover at {days} days: {error}"
            ) from error
        rows.append(
            {
                days_column: days,
                f"mae_{zero_shot}": paired[zero_shot].mean(),
                f"mae_{trained}": paired[trained].mean(),
                "difference": difference.mean(),
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "n_origins": int(paired.shape[0]),
            }
        )
        verdicts[days] = (float(difference.mean()), p_value < alpha)

    curve = pd.DataFrame(rows).sort_values(days_column, ignore_index=True)
    return curve, _crossover(verdicts)


def _crossover(verdicts: dict[int, tuple[float, bool]]) -> Crossover:
    """Locate the first budget at which the trained method's error drops below."""
    budgets = sorted(verdicts)
    below: int | None = None
    for days in budgets:
        difference, significant = verdicts[days]
        if difference < 0:
            return Crossover(days_below=below, days_above=days, significant=significant)
        below = days
    return Crossover(days_below=below, days_above=None, significant=False)


def _bootstrap_ci(
    values: np.ndarray, *, level: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean, over a fixed seed."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.choice(finite, size=(n_resamples, finite.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1.0 - tail)),
    )


def degradation_curve(
    metrics: pd.DataFrame,
    *,
    horizon: int,
    series_group: str = "all",
    profile_column: str = "sensor_profile",
    order: Sequence[str] = ("clean", "realistic", "degraded", "harsh"),
) -> pd.DataFrame:
    """Accuracy per method against sensor quality, for E5.

    Raises:
        ReportError: if the profile column is missing, or a profile in ``order`` has no
            run. A degradation curve with a gap in it invites the reader to interpolate
            across a condition nobody measured.
    """
    if profile_column not in metrics.columns:
        raise ReportError(
            f"the metrics carry no {profile_column!r} column; record the sensor profile "
            "in the run config before building a degradation figure"
        )
    cell = metrics[
        (metrics["series_group"] == series_group) & (metrics["horizon"] == horizon)
    ]
    present = set(cell[profile_column].dropna().unique())
    missing = [name for name in order if name not in present]
    if missing:
        raise ReportError(f"no run for sensor profile(s) {missing}; the curve would have a gap")
    curve = (
        cell.groupby(["method", profile_column], as_index=False)
        .agg(mae=("mae", "mean"), wql=("wql", "mean"), coverage_80=("coverage_80", "mean"))
    )
    curve[profile_column] = pd.Categorical(curve[profile_column], categories=list(order))
    return curve.sort_values(["method", profile_column], ignore_index=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def data_efficiency_figure(
    curve: pd.DataFrame,
    crossover: Crossover,
    *,
    zero_shot: str,
    trained: str,
    days_column: str = "train_days",
) -> Any:
    """Both error curves against the training budget, with the crossover marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    days = curve[days_column]
    axis.plot(days, curve[f"mae_{zero_shot}"], marker="o", label=zero_shot)
    axis.plot(days, curve[f"mae_{trained}"], marker="s", label=trained)
    axis.fill_between(
        days,
        curve[f"mae_{zero_shot}"] + curve["ci_low"],
        curve[f"mae_{zero_shot}"] + curve["ci_high"],
        alpha=0.15,
        label="95% CI on the difference",
    )
    if crossover.days_above is not None:
        axis.axvline(
            crossover.days_above,
            linestyle="--" if crossover.significant else ":",
            color="0.3",
        )
    axis.set_xscale("log")
    axis.set_xticks(list(days))
    axis.set_xticklabels([str(int(d)) for d in days])
    axis.set_xlabel("days of local training history")
    axis.set_ylabel("MAE (persons)")
    axis.set_title(crossover.describe(), fontsize=9)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def degradation_figure(
    curve: pd.DataFrame, *, profile_column: str = "sensor_profile"
) -> Any:
    """One line per method across the sensor profiles, worst to best."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    for method, sub in curve.groupby("method"):
        axis.plot(sub[profile_column].astype(str), sub["mae"], marker="o", label=str(method))
    axis.set_xlabel("sensor profile")
    axis.set_ylabel("MAE (persons)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def build_report(
    output: Path,
    *,
    root: Path | None = None,
    reference: str | None = None,
) -> dict[str, list[Path]]:
    """Build every artefact the logged runs can support, and no others.

    Tables whose runs are absent are skipped with their reason recorded in
    ``report.json``, rather than emitted empty. A reader of the output directory can
    therefore tell the difference between "this was measured and is here" and "this was
    not measured", which is the distinction ground rule 2 exists to protect.

    Args:
        output: directory to write into.
        root: the results directory. Defaults to ``results/``.
        reference: the baseline method for the significance markers. Defaults to the
            method with the highest mean MAE, i.e. the weakest baseline present.

    Returns:
        A mapping from artefact name to the files written for it.
    """
    runs = load_runs(root)
    target = ensure_dir(Path(output))
    metrics = combine(runs)
    written: dict[str, list[Path]] = {}
    skipped: dict[str, str] = {}

    if reference is None:
        reference = str(
            metrics[metrics["series_group"] == "all"]
            .groupby("method")["mae"]
            .mean()
            .idxmax()
        )

    written["main"] = write_table(
        main_table(metrics, reference=reference),
        target,
        "main",
        caption=(
            f"Forecast accuracy by method and horizon over {len(runs)} logged run(s). "
            f"A star marks a method significantly better than {reference} by a "
            "Diebold-Mariano test on paired per-origin MAE, Holm-Bonferroni corrected "
            "within each group and horizon."
        ),
    )

    for name, builder, caption in (
        (
            "reconciliation",
            lambda: reconciliation_table(metrics, combine(runs, RECONCILIATION_FILE)),
            "Accuracy and coherence before and after reconciliation.",
        ),
        (
            "latency",
            lambda: latency_table(combine(runs, RESOURCES_FILE), metrics),
            "Inference latency and peak memory per method, measured by the harness.",
        ),
        ("covariates", lambda: covariate_table(metrics), "Covariate ablation."),
    ):
        try:
            written[name] = write_table(builder(), target, name, caption=caption)
        except ReportError as error:
            skipped[name] = str(error)

    (target / "report.json").write_text(
        json.dumps(
            {
                "runs": [run.run_id for run in runs],
                "reference_method": reference,
                "written": {k: [str(p) for p in v] for k, v in written.items()},
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return written
