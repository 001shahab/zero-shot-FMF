"""Score the three risk heads against a completed evaluation (M6 / E6).

The heads live in :mod:`mflow.risk`; this module is the adapter that feeds them from the
harness's tidy predictions frame. Keeping the two apart means the heads can be tested on
hand-built arrays, and the reshaping -- which is where the off-by-one errors live -- is
tested once here rather than in every head.

Three conventions are worth stating, because each is a modelling choice:

* **Congestion** is scored per occupancy series. The horizon is the full forecast, so an
  alert says "this room breaches at some point in the next H minutes", which is the form
  an operator can act on.
* **Anomaly detection** runs on the one-step-ahead forecast. A live monitor compares each
  arriving observation against what was predicted for it a moment earlier, and a
  multi-step forecast would conflate a detector's sensitivity with the model's decay.
* **Exposure** is reported as a formulation and a worked example, never as a validated
  claim, because no dataset links visitor load to a conservation outcome. The output
  carries that caveat in a column so it survives into any table built from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mflow.eval.harness import EvaluationResult
from mflow.eval.protocol import RollingOriginPlan
from mflow.risk import (
    ANOMALY_KINDS,
    AnomalyKind,
    anomaly_score,
    calibrate_threshold,
    exceedance_probability,
    first_breach_step,
    inject_anomalies,
    node_thresholds,
    score_alerts,
    score_detections,
    worked_example,
)
from mflow.schema import OCC_PREFIX, SiteData, parse_series_id

#: Covariate identifier carrying the opening schedule, used to place the closed-gallery
#: anomaly at a time when nothing should be moving.
IS_OPEN_COVARIATE = "global|is_open"


class RiskEvalError(RuntimeError):
    """Raised when a risk head cannot be scored against a run."""


class RiskConfig(BaseModel):
    """How the three heads are scored for one experiment."""

    model_config = ConfigDict(extra="forbid")

    #: Share of a room's capacity at which it counts as congested.
    congestion_occupancy_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    #: Exceedance probability at which the forecast raises an alert.
    alert_probability: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: Share of clean cells the anomaly threshold is allowed to flag.
    anomaly_false_alarm_budget: float = Field(default=0.01, gt=0.0, lt=1.0)
    #: Which anomalies to inject. Validated against the Literal, so a typo fails at
    #: config load rather than after the forecasts have been computed.
    anomaly_kinds: list[AnomalyKind] = Field(default_factory=lambda: list(ANOMALY_KINDS))
    #: How many of each.
    anomaly_per_kind: int = Field(default=6, gt=0)
    #: Length of each, in origins.
    anomaly_duration: int = Field(default=5, gt=0)
    #: Room the exposure worked example uses. ``None`` picks the busiest.
    exposure_node: str | None = None
    #: Multiple of a room's median daily load used as the illustrative budget.
    exposure_headroom: float = Field(default=1.2, gt=0.0)


def evaluate_risk(
    result: EvaluationResult,
    site: SiteData,
    plan: RollingOriginPlan,
    config: RiskConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    """Score every head for every method in ``result``.

    Returns:
        A long frame with columns ``head``, ``method``, ``detector``, ``subject``,
        ``metric``, ``value`` and ``caveat``. Long rather than wide because the heads
        report different metrics and a wide frame would be mostly empty cells that a
        reader could mistake for zeros. ``value`` is always numeric, so anything
        categorical -- which room the exposure example used, for instance -- goes in
        ``subject`` rather than being stringly typed into the measurement column.
    """
    rows: list[dict[str, object]] = []
    rows.extend(_congestion_rows(result, site, plan, config))
    rows.extend(_anomaly_rows(result, site, plan, config, seed=seed))
    rows.extend(_exposure_rows(site, config))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #


def _quantile_columns(predictions: pd.DataFrame, plan: RollingOriginPlan) -> list[str]:
    columns = [f"q{round(level * 100):02d}" for level in plan.quantiles]
    missing = [c for c in columns if c not in predictions.columns]
    if missing:
        raise RiskEvalError(f"the predictions frame is missing quantile column(s) {missing}")
    return columns


def _cube(
    predictions: pd.DataFrame, plan: RollingOriginPlan, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Reshape one method's tidy predictions into arrays.

    Returns:
        ``(forecast, actual, series_ids, origins)`` where ``forecast`` is
        ``(n_series, n_origins, horizon, n_quantiles)`` and ``actual`` is
        ``(n_series, n_origins, horizon)``.
    """
    ordered = predictions.sort_values(["series_id", "origin", "step"])
    series_ids = sorted(ordered["series_id"].unique())
    origins = sorted(ordered["origin"].unique())
    horizon = plan.horizon
    expected = len(series_ids) * len(origins) * horizon
    if len(ordered) != expected:
        raise RiskEvalError(
            f"expected {expected} prediction rows for {len(series_ids)} series over "
            f"{len(origins)} origins at horizon {horizon}, got {len(ordered)}"
        )
    forecast = ordered[columns].to_numpy().reshape(
        len(series_ids), len(origins), horizon, len(columns)
    )
    actual = ordered["actual"].to_numpy().reshape(len(series_ids), len(origins), horizon)
    return forecast, actual, [str(s) for s in series_ids], [int(o) for o in origins]


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #


def _congestion_rows(
    result: EvaluationResult,
    site: SiteData,
    plan: RollingOriginPlan,
    config: RiskConfig,
) -> list[dict[str, object]]:
    """Alerting performance per method, pooled over the occupancy series."""
    columns = _quantile_columns(result.predictions, plan)
    thresholds = node_thresholds(
        site.node_capacity(), occupancy_fraction=config.congestion_occupancy_fraction
    )
    steps_per_day = 24 * 3600 // site.meta.interval_seconds
    stride = _stride(plan)

    rows: list[dict[str, object]] = []
    truth_panel = site.to_panel()
    for method, frame in result.predictions.groupby("method"):
        forecast, actual, series_ids, origins = _cube(frame, plan, columns)
        occupancy_rows = [
            index for index, sid in enumerate(series_ids) if sid.startswith(OCC_PREFIX)
        ]
        if not occupancy_rows:
            raise RiskEvalError("the run carries no occupancy series to alert on")

        alerts: list[np.ndarray] = []
        breaches: list[np.ndarray] = []
        reactive: list[np.ndarray] = []
        for index in occupancy_rows:
            node = parse_series_id(series_ids[index])[1]
            threshold = thresholds[node]
            probability = exceedance_probability(forecast[index], plan.quantiles, threshold)
            alerts.append(np.asarray((probability >= config.alert_probability).any(axis=1)))
            breaches.append(first_breach_step(actual[index], threshold))
            row = truth_panel.index_of(series_ids[index])
            last = np.array(
                [truth_panel.series[row, origin - 1] for origin in origins], dtype=np.float64
            )
            reactive.append(np.isfinite(last) & (last > threshold))

        n_nodes = len(occupancy_rows)
        pooled_breaches = np.concatenate(breaches)
        forecast_outcome = score_alerts(
            np.concatenate(alerts),
            pooled_breaches,
            steps_per_day=steps_per_day,
            n_nodes=n_nodes,
            n_origins_per_step=1.0 / stride,
        )
        reactive_outcome = score_alerts(
            np.concatenate(reactive),
            pooled_breaches,
            steps_per_day=steps_per_day,
            n_nodes=n_nodes,
            n_origins_per_step=1.0 / stride,
        )
        for detector, outcome in (
            ("forecast", forecast_outcome),
            ("reactive", reactive_outcome),
        ):
            for metric, value in outcome.as_dict(site.meta.interval_seconds).items():
                rows.append(
                    {
                        "head": "congestion",
                        "method": str(method),
                        "detector": detector,
                        "subject": "all_rooms",
                        "metric": metric,
                        "value": value,
                        "caveat": "",
                    }
                )
    return rows


def _anomaly_rows(
    result: EvaluationResult,
    site: SiteData,
    plan: RollingOriginPlan,
    config: RiskConfig,
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Detection rate and time to detect per method, at the configured budget."""
    columns = _quantile_columns(result.predictions, plan)
    rows: list[dict[str, object]] = []
    for method, frame in result.predictions.groupby("method"):
        forecast, actual, series_ids, origins = _cube(frame, plan, columns)
        # One step ahead: what a live monitor would have had for each arriving reading.
        clean = actual[:, :, 0]
        prediction = forecast[:, :, 0, :]
        closed = _closed_mask(site, origins, len(series_ids))
        flow_rows = np.array(
            [i for i, sid in enumerate(series_ids) if not sid.startswith(OCC_PREFIX)]
        )
        contaminated, injected = inject_anomalies(
            clean,
            config.anomaly_kinds,
            seed=seed,
            n_per_kind=config.anomaly_per_kind,
            duration=config.anomaly_duration,
            closed_mask=closed,
            flow_rows=flow_rows,
        )
        clean_score = anomaly_score(clean, prediction, plan.quantiles)
        threshold = calibrate_threshold(clean_score, config.anomaly_false_alarm_budget)
        outcome = score_detections(
            anomaly_score(contaminated, prediction, plan.quantiles),
            injected,
            threshold,
            clean_score,
        )
        interval = site.meta.interval_seconds
        stride = _stride(plan)
        for metric, value in outcome.as_dict(interval * stride).items():
            rows.append(
                {
                    "head": "anomaly",
                    "method": str(method),
                    "detector": "forecast",
                    "subject": "all_series",
                    "metric": metric,
                    "value": value,
                    "caveat": "",
                }
            )
    return rows


def _stride(plan: RollingOriginPlan) -> int:
    """Steps between consecutive origins, which sets the nuisance-rate denominator."""
    return plan.tasks[1].origin - plan.tasks[0].origin if len(plan.tasks) > 1 else 1


def _closed_mask(
    site: SiteData, origins: list[int], n_series: int
) -> np.ndarray | None:
    """Which origins fall while the site is shut, broadcast over the series.

    Returns None when the site does not publish an opening schedule, in which case the
    closed-gallery anomaly cannot be placed and :func:`inject_anomalies` says so.
    """
    panel = site.to_panel()
    if IS_OPEN_COVARIATE not in panel.future_covariate_ids:
        return None
    if panel.future_covariates is None:
        return None
    row = panel.future_covariate_ids.index(IS_OPEN_COVARIATE)
    schedule = panel.future_covariates[row]
    shut = np.array([schedule[origin] < 0.5 for origin in origins], dtype=bool)
    return np.broadcast_to(shut, (n_series, len(origins))).copy()


def _exposure_rows(site: SiteData, config: RiskConfig) -> list[dict[str, object]]:
    """The worked example, carrying its caveat in every row."""
    node = config.exposure_node
    if node is None:
        totals = site.occupancy.groupby("node_id")["count"].sum()
        if totals.empty:
            raise RiskEvalError("the site has no occupancy counts to illustrate exposure with")
        node = str(totals.idxmax())
    budget, daily = worked_example(
        site.occupancy,
        node,
        site.meta.interval_seconds,
        headroom=config.exposure_headroom,
    )
    return [
        {
            "head": "exposure",
            "method": "n/a",
            "detector": "n/a",
            "subject": node,
            "metric": metric,
            "value": value,
            "caveat": budget.rationale,
        }
        for metric, value in {
            "budget_person_minutes": budget.person_minutes_per_day,
            "median_daily_person_minutes": float(daily["person_minutes"].median()),
            "max_daily_person_minutes": float(daily["person_minutes"].max()),
            "days_over_budget": float(daily["exceeds_budget"].sum()),
            "n_days": float(len(daily)),
        }.items()
    ]
