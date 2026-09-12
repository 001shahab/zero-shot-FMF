"""The evaluation harness: run methods over a plan and score them (M6).

One function, :func:`run_evaluation`, drives everything. It takes a site, a frozen
:class:`~mflow.eval.protocol.RollingOriginPlan` and a list of method/reconciler
combinations, and returns tidy predictions and metrics frames plus the resource
measurements that the edge-deployment claim rests on.

Three properties are enforced here rather than left to the caller:

* **Identical conditions.** Every combination is driven from the same plan and the same
  context panels, so a difference in the results table is a difference between methods.
* **Per-origin losses.** Metrics are recorded per origin, not just aggregated, because
  the Diebold-Mariano test needs the paired series and an aggregate cannot be
  un-aggregated afterwards.
* **Fail loudly.** A forecaster that raises is not skipped. The run stops, because a
  results table with a silently missing method invites exactly the wrong conclusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from mflow.eval import metrics as M
from mflow.eval.protocol import (
    ForecastTask,
    RollingOriginPlan,
    context_panel,
    training_panel,
    truth_for,
)
from mflow.forecast.base import DEFAULT_QUANTILES, Forecaster, season_length
from mflow.graph import BuildingGraph
from mflow.profiling import measure
from mflow.reconcile import (
    ConstraintSystem,
    QuantileReconciler,
    Reconciler,
    build_constraints,
    flatten,
    unflatten,
)
from mflow.schema import OCC_PREFIX, Panel, SiteData, parse_series_id

#: Series groups reported separately, because occupancy and flow have different scales
#: and the paper's claims are made about each.
SERIES_GROUPS = ("occupancy", "flow", "all")

#: How a point reconciliation is carried through the quantile fan.
QuantileStrategy = Literal["shift", "per_quantile"]


class HarnessError(RuntimeError):
    """Raised when an evaluation cannot be carried out as specified."""


@dataclass(frozen=True)
class MethodSpec:
    """One cell of the experiment grid.

    Attributes:
        forecaster: the model. Trained models are fitted once per run, not per origin.
        reconciler: the projection applied to its output, or None for the raw forecast.
        quantile_strategy: how the reconciliation is carried through the quantiles.
        label: name used in the results tables. Defaults to ``method+reconciler``.
    """

    forecaster: Forecaster
    reconciler: Reconciler | None = None
    quantile_strategy: QuantileStrategy = "shift"
    label: str | None = None

    @property
    def name(self) -> str:
        """Results-table label for this combination."""
        if self.label is not None:
            return self.label
        if self.reconciler is None:
            return f"{self.forecaster.name}+none"
        return f"{self.forecaster.name}+{self.reconciler.name}:{self.quantile_strategy}"


@dataclass
class EvaluationResult:
    """Everything one run produced.

    Attributes:
        predictions: one row per method, origin, series and horizon step, with a column
            per quantile and the actual value.
        metrics: one row per method, origin, series group and horizon, with every metric.
        resources: one row per method and origin with latency and peak memory.
        reconciliation: one row per method and origin with the coherence diagnostics.
        plan_description: the frozen plan, for the manifest.
    """

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    resources: pd.DataFrame
    reconciliation: pd.DataFrame
    plan_description: dict[str, object] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Metrics averaged over origins, which is what goes into the main table."""
        keys = ["method", "series_group", "horizon"]
        numeric = self.metrics.select_dtypes(include="number").columns.difference(["origin"])
        return (
            self.metrics.groupby(keys, as_index=False)[list(numeric)]
            .mean()
            .sort_values(keys, ignore_index=True)
        )


def run_evaluation(
    site: SiteData,
    plan: RollingOriginPlan,
    specs: Sequence[MethodSpec],
    *,
    panel: Panel | None = None,
    truth_panel: Panel | None = None,
    include_validation_in_training: bool = True,
    enforce_bounds: bool = True,
    mase_season: int | None = None,
) -> EvaluationResult:
    """Run every method over every origin and score the results.

    Args:
        site: the site being evaluated, which supplies the graph and the interval.
        plan: the frozen rolling-origin plan.
        specs: the method and reconciler combinations to run.
        panel: the observed panel models see. Defaults to ``site.to_panel()``.
        truth_panel: the panel forecasts are scored against. Defaults to ``panel``. The
            two differ when a sensor profile has been applied: models see the degraded
            series and are scored against the clean ones, which is the only honest way to
            measure what degradation costs.
        include_validation_in_training: whether trained models may fit on the validation
            window as well as the training window.
        enforce_bounds: pass box bounds to the constraint system.
        mase_season: seasonal lag for the MASE denominator, in steps. Defaults to one
            day, which is the right season for a museum. A record whose training window
            is shorter than one season has to say what season it wants instead of getting
            a quietly substituted one, because MASE values computed against different
            denominators are not comparable and nothing in the output would show it.

    Raises:
        HarnessError: if the panels disagree, or if a site cannot supply the conservation
            anchor a reconciler needs.
    """
    # No horizon is appended: every origin lies inside the test window, so the known-future
    # covariates a task needs are already on the observed grid. Asking for more would
    # demand rows past the end of the record that no site has.
    observed = panel if panel is not None else site.to_panel()
    actual = truth_panel if truth_panel is not None else observed
    if actual.series.shape != observed.series.shape:
        raise HarnessError(
            f"the observed panel is {observed.series.shape} but the truth panel is "
            f"{actual.series.shape}; they must describe the same series and grid"
        )
    if actual.series_ids != observed.series_ids:
        raise HarnessError("the observed and truth panels are not in the same series order")

    quantiles = plan.quantiles or DEFAULT_QUANTILES
    needs_constraints = any(spec.reconciler is not None for spec in specs)
    system = (
        build_constraints(
            BuildingGraph.from_site(site),
            observed.series_ids,
            plan.horizon,
            site.meta.interval_seconds,
            enforce_bounds=enforce_bounds,
        )
        if needs_constraints
        else None
    )

    season = (
        season_length(site.meta.interval_seconds) if mase_season is None else int(mase_season)
    )
    scale = M.seasonal_naive_scale(actual.series[:, : plan.split.train_end], season)
    groups = _series_groups(observed.series_ids)
    lower, upper = _bounds(site, observed.series_ids)

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []

    fitted: dict[int, None] = {}
    for spec in specs:
        forecaster = spec.forecaster
        if forecaster.requires_training and id(forecaster) not in fitted:
            forecaster.fit(
                training_panel(
                    observed, plan.split, include_validation=include_validation_in_training
                )
            )
            fitted[id(forecaster)] = None

        quantile_reconciler = (
            None
            if spec.reconciler is None
            else QuantileReconciler(spec.reconciler, strategy=spec.quantile_strategy)
        )

        for task in plan:
            # Latency and peak memory are measured here rather than read back from the
            # forecaster, so that every method is timed by the same clock over the same
            # boundary. A wrapper that measured only its own inner call would look faster
            # than one that measured its preprocessing too, and the edge-deployment claim
            # rests on this table being comparable across methods.
            with measure() as usage:
                raw = forecaster.predict(
                    context_panel(observed, task), task.horizon, quantiles
                )
            prediction = raw
            if quantile_reconciler is not None:
                if system is None:  # pragma: no cover - guarded by needs_constraints
                    raise HarnessError("a reconciler was requested without a constraint system")
                prediction, diagnostics = _reconcile(
                    quantile_reconciler, system, raw, quantiles, observed, task, site
                )
                reconciliation_rows.append(
                    {"method": spec.name, "origin": task.origin, **diagnostics}
                )

            target = truth_for(actual, task)
            prediction_rows.append(
                _prediction_frame(spec.name, task, observed.series_ids, prediction, target, plan)
            )
            metric_rows.extend(
                _score(
                    spec.name,
                    task,
                    prediction,
                    target,
                    quantiles,
                    plan,
                    groups,
                    scale,
                    lower,
                    upper,
                    system,
                    observed,
                    site,
                )
            )
            resource_rows.append(
                {"method": spec.name, "origin": task.origin, **usage.as_dict()}
            )

    return EvaluationResult(
        predictions=pd.concat(prediction_rows, ignore_index=True),
        metrics=pd.DataFrame(metric_rows),
        resources=pd.DataFrame(resource_rows),
        reconciliation=pd.DataFrame(reconciliation_rows),
        plan_description=plan.describe(),
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _series_groups(series_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Boolean row masks for each reported series group."""
    kinds = np.array([parse_series_id(sid)[0] for sid in series_ids])
    return {
        "occupancy": kinds == "occupancy",
        "flow": kinds == "flow",
        "all": np.ones(len(series_ids), dtype=bool),
    }


def _bounds(site: SiteData, series_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Physical lower and upper bounds per series, for the violation rate."""
    capacity = site.node_capacity()
    per_interval = site.edge_capacity_per_interval()
    lower = np.zeros(len(series_ids))
    upper = np.empty(len(series_ids))
    for index, series_id in enumerate(series_ids):
        kind, entity = parse_series_id(series_id)
        upper[index] = capacity[entity] if kind == "occupancy" else per_interval[entity]
    return lower, upper


def _last_occupancy(panel: Panel, task: ForecastTask) -> tuple[dict[str, float], int]:
    """The conservation anchor for one origin, and how stale it is.

    The identity is anchored on the occupancy observed at the step before the origin.
    Under a sensor profile that reading may be missing, and a missing anchor is not the
    same as a zero one: substituting zero would manufacture a surge of arrivals at the
    first horizon step. The anchor is therefore taken from the most recent finite
    observation inside the context window, and the number of steps it was carried
    forward is returned so that the run records where it imputed.

    Returns:
        ``(anchor, staleness)`` where ``staleness`` is the largest number of steps any
        node's anchor was carried forward: zero when every room was observed at the
        origin.

    Raises:
        HarnessError: if a node has no finite observation anywhere in its context. That
            is a dead sensor, not a gap, and no amount of carrying forward will anchor
            the identity for it.
    """
    window = panel.series[:, task.context_start : task.origin]
    anchor: dict[str, float] = {}
    staleness = 0
    for row, series_id in enumerate(panel.series_ids):
        if not series_id.startswith(OCC_PREFIX):
            continue
        observed = np.flatnonzero(np.isfinite(window[row]))
        if observed.size == 0:
            raise HarnessError(
                f"series {series_id!r} has no finite observation in the "
                f"{window.shape[1]}-step context before origin {task.origin}; the "
                "conservation identity cannot be anchored on a dead sensor"
            )
        last = int(observed[-1])
        staleness = max(staleness, window.shape[1] - 1 - last)
        anchor[parse_series_id(series_id)[1]] = float(window[row, last])
    return anchor, staleness


def _reconcile(
    reconciler: QuantileReconciler,
    system: ConstraintSystem,
    raw: np.ndarray,
    quantiles: Sequence[float],
    panel: Panel,
    task: ForecastTask,
    site: SiteData,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Project one quantile forecast and report what the projection changed."""
    anchor, staleness = _last_occupancy(panel, task)
    rhs = system.rhs(anchor)
    reconciled, result = reconciler.reconcile(raw, list(quantiles), system, rhs)
    median = M.point_forecast(reconciled, quantiles)
    diagnostics: dict[str, float | str] = dict(result.as_dict())
    diagnostics["site_id"] = site.meta.site_id
    # How far the anchor had to be carried forward. Zero on a clean site; under a sensor
    # profile it is the record of where the run imputed, which ground rule 5 requires.
    diagnostics["anchor_staleness_steps"] = float(staleness)
    diagnostics["residual_after"] = float(
        np.abs(system.residual(flatten(median), rhs)).mean()
    )
    diagnostics["violation_after"] = float(
        (system.violation(flatten(median)) > 1e-9).mean()
    )
    return reconciled, diagnostics


def _prediction_frame(
    method: str,
    task: ForecastTask,
    series_ids: Sequence[str],
    prediction: np.ndarray,
    target: np.ndarray,
    plan: RollingOriginPlan,
) -> pd.DataFrame:
    """Tidy predictions for one method at one origin.

    Quantile columns are written as float64 whatever the forecaster produced. A
    reconciler solves in double precision while a raw forecaster may return float32, and
    without this the dtype of a results file would depend on which method wrote it.
    """
    n_series, horizon, _ = prediction.shape
    prediction = prediction.astype(np.float64, copy=False)
    frame = pd.DataFrame(
        {
            "method": method,
            "origin": task.origin,
            "origin_time": plan.split.timestamps[task.origin],
            "series_id": np.repeat(np.asarray(series_ids), horizon),
            "step": np.tile(np.arange(1, horizon + 1), n_series),
            "actual": target.reshape(-1),
        }
    )
    for index, level in enumerate(plan.quantiles):
        frame[f"q{round(level * 100):02d}"] = prediction[:, :, index].reshape(-1)
    return frame


def _score(
    method: str,
    task: ForecastTask,
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: Sequence[float],
    plan: RollingOriginPlan,
    groups: dict[str, np.ndarray],
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    system: ConstraintSystem | None,
    panel: Panel,
    site: SiteData,
) -> list[dict[str, Any]]:
    """Every metric for one method at one origin, per group and per horizon."""
    rows: list[dict[str, Any]] = []
    for horizon in plan.horizons:
        # A shorter horizon is scored on a prefix of the same forecast, so the comparison
        # across horizons is within a single model call rather than across separate ones.
        for group, mask in groups.items():
            truth = target[mask, :horizon]
            forecast = prediction[mask, :horizon, :]
            if not np.isfinite(truth).any():
                # Nothing was observed for this group at this origin, which happens
                # overnight. Scoring it would divide by zero; recording a row of NaN
                # would quietly enter the averages.
                continue
            row: dict[str, Any] = {
                "method": method,
                "origin": task.origin,
                "origin_time": plan.split.timestamps[task.origin],
                "series_group": group,
                "horizon": horizon,
                "n_series": int(mask.sum()),
                "mae": M.mae(truth, forecast, quantiles),
                "rmse": M.rmse(truth, forecast, quantiles),
                "mase": M.mase(truth, forecast, quantiles, scale[mask]),
                "crps": M.crps(truth, forecast, quantiles),
                "coverage_80": M.coverage(truth, forecast, quantiles),
                "interval_width_80": M.interval_width(truth, forecast, quantiles),
                "violation_rate": M.violation_rate(forecast, quantiles, lower[mask], upper[mask]),
                "quantile_crossing_rate": M.quantile_crossing_rate(forecast),
            }
            try:
                row["wql"] = M.wql(truth, forecast, quantiles)
            except M.MetricError:
                # An all-zero window has no scale to normalise by. The other metrics are
                # still meaningful, so the row is kept with WQL marked absent rather than
                # dropped or filled with a number nobody computed.
                row["wql"] = float("nan")
            if group == "all" and system is not None and horizon == plan.horizon:
                rhs = system.rhs(_last_occupancy(panel, task)[0])
                row["conservation_residual_mae"] = M.conservation_residual_mae(
                    prediction, quantiles, system.A.toarray(), rhs
                )
            rows.append(row)
    _ = site
    return rows


def paired_losses(
    metrics: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    series_group: str = "all",
    horizon: int | None = None,
    loss: str = "mae",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-origin losses for two methods, aligned by origin.

    Raises:
        HarnessError: if either method is absent, or the two do not share any origin,
            which would otherwise produce a significance test over a handful of
            accidentally overlapping rows.
    """
    frame = metrics[metrics["series_group"] == series_group]
    if horizon is not None:
        frame = frame[frame["horizon"] == horizon]
    pivot = frame.pivot_table(index="origin", columns="method", values=loss)
    for method in (method_a, method_b):
        if method not in pivot.columns:
            raise HarnessError(
                f"method {method!r} is not in the metrics; available: {sorted(pivot.columns)}"
            )
    paired = pivot[[method_a, method_b]].dropna()
    if paired.empty:
        raise HarnessError(
            f"{method_a!r} and {method_b!r} share no origin with a {loss} value for "
            f"group {series_group!r}"
        )
    return paired[method_a].to_numpy(), paired[method_b].to_numpy()


def reconciled_unflatten(y: np.ndarray, n_series: int, horizon: int) -> np.ndarray:
    """Re-export of :func:`mflow.reconcile.unflatten` for callers of this module."""
    return unflatten(y, n_series, horizon)
