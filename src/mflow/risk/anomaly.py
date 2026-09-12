"""Anomaly detection against a forecast (M6).

The detector is deliberately simple: an observation is anomalous when it falls far
outside the predictive interval the forecast issued for it, measured in units of the
forecast's own spread. A better detector could certainly be built, but the point of the
experiment is whether a zero-shot forecast is a good enough *normal model* to detect
things worth detecting, not whether a detector can be tuned.

Three anomaly types are injected, matching the specification:

* ``closed_gallery_flow`` -- a room that is shut still records people crossing into it.
  This is the one that only a topology-aware system can see, because the counts
  themselves look ordinary; what is wrong is that they contradict the room's state.
* ``dwell_cluster`` -- occupancy in one room climbs far above its usual level and stays
  there, as when a tour group stalls or an incident draws a crowd.
* ``stuck_sensor`` -- the stream freezes at its last value. Unlike the other two this is
  an instrument fault rather than a crowd event, and it is included because operators
  cannot act on an alert until they know which of the two it is.

Detection is scored at a fixed false alarm budget, chosen by calibrating the threshold on
clean data. Reporting a detection rate without pinning the false alarm rate would let any
detector look good by alerting constantly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mflow.manifest import rng

AnomalyKind = Literal["closed_gallery_flow", "dwell_cluster", "stuck_sensor"]

ANOMALY_KINDS: tuple[AnomalyKind, ...] = (
    "closed_gallery_flow",
    "dwell_cluster",
    "stuck_sensor",
)


class AnomalyError(ValueError):
    """Raised when an anomaly cannot be injected or scored as specified."""


@dataclass(frozen=True)
class InjectedAnomaly:
    """One anomaly written into a test stream.

    Attributes:
        kind: which of the three types.
        series_index: row of the panel that was altered.
        start: first altered step.
        stop: one past the last altered step.
        magnitude: how large the alteration was, in the units of the series.
    """

    kind: AnomalyKind
    series_index: int
    start: int
    stop: int
    magnitude: float

    def covers(self, step: int) -> bool:
        """Whether ``step`` falls inside this anomaly."""
        return self.start <= step < self.stop


@dataclass(frozen=True)
class DetectionOutcome:
    """Detection performance at a fixed false alarm budget.

    Attributes:
        detection_rate: share of injected anomalies flagged at any point while active.
        median_time_to_detect_steps: median steps from the start of an anomaly to the
            first flag inside it, over the anomalies that were detected.
        false_alarm_rate: share of clean cells flagged, which the threshold was
            calibrated to hold at the budget.
        threshold: the calibrated score above which a cell is flagged.
        n_injected: how many anomalies were injected.
        n_detected: how many were found.
    """

    detection_rate: float
    median_time_to_detect_steps: float
    false_alarm_rate: float
    threshold: float
    n_injected: int
    n_detected: int

    def as_dict(self, interval_seconds: int) -> dict[str, float]:
        """Flat mapping for a results table."""
        return {
            "detection_rate": self.detection_rate,
            "median_time_to_detect_minutes": (
                self.median_time_to_detect_steps * interval_seconds / 60.0
            ),
            "false_alarm_rate": self.false_alarm_rate,
            "threshold": self.threshold,
            "n_injected": float(self.n_injected),
            "n_detected": float(self.n_detected),
        }


def inject_anomalies(
    series: np.ndarray,
    kinds: Sequence[AnomalyKind],
    *,
    seed: int,
    n_per_kind: int = 4,
    duration: int = 20,
    magnitude: float = 4.0,
    closed_mask: np.ndarray | None = None,
    flow_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, list[InjectedAnomaly]]:
    """Write synthetic anomalies into a copy of ``series``.

    Args:
        series: ``(n_series, T)`` of clean test-window actuals.
        kinds: which anomaly types to inject.
        seed: run seed; the same seed always injects the same anomalies.
        n_per_kind: how many of each type.
        duration: length of each anomaly in steps.
        magnitude: size of the ``dwell_cluster`` anomaly, in units of the series'
            standard deviation, and the headcount injected into a closed gallery.
        closed_mask: ``(n_series, T)`` marking steps where the site is shut. Required for
            ``closed_gallery_flow``, because the whole point of that anomaly is that it
            occurs when nothing should be moving.
        flow_rows: indices of the flow series, required for ``closed_gallery_flow``.

    Returns:
        The altered series and the list of what was injected.

    Raises:
        AnomalyError: if an anomaly cannot be placed. Silently injecting fewer than asked
            would make the detection rate a ratio with an unknown denominator.
    """
    if series.ndim != 2:
        raise AnomalyError(f"series must be (n_series, T), got {series.shape}")
    n_series, n_steps = series.shape
    if duration < 1 or duration >= n_steps:
        raise AnomalyError(f"duration {duration} does not fit in a window of {n_steps}")

    altered = np.array(series, dtype=np.float64, copy=True)
    injected: list[InjectedAnomaly] = []
    spread = np.nanstd(series, axis=1)

    for kind in kinds:
        generator = rng(seed, "anomaly", kind)
        for _ in range(n_per_kind):
            if kind == "closed_gallery_flow":
                if closed_mask is None or flow_rows is None:
                    raise AnomalyError(
                        "closed_gallery_flow needs closed_mask and flow_rows: the anomaly "
                        "is defined by movement while the site is shut"
                    )
                row, start = _pick_closed_window(
                    closed_mask, flow_rows, duration, generator
                )
                altered[row, start : start + duration] += magnitude
                size = magnitude
            elif kind == "dwell_cluster":
                row = int(generator.integers(n_series))
                start = int(generator.integers(0, n_steps - duration))
                size = float(magnitude * max(spread[row], 1.0))
                altered[row, start : start + duration] += size
            elif kind == "stuck_sensor":
                row = int(generator.integers(n_series))
                start = int(generator.integers(0, n_steps - duration))
                frozen = altered[row, start]
                altered[row, start : start + duration] = frozen
                size = float(
                    np.nanmax(np.abs(series[row, start : start + duration] - frozen))
                )
            else:  # pragma: no cover - the Literal makes this unreachable
                raise AnomalyError(f"unknown anomaly kind {kind!r}")
            injected.append(
                InjectedAnomaly(
                    kind=kind,
                    series_index=row,
                    start=start,
                    stop=start + duration,
                    magnitude=size,
                )
            )
    return altered, injected


def _pick_closed_window(
    closed_mask: np.ndarray,
    flow_rows: np.ndarray,
    duration: int,
    generator: np.random.Generator,
) -> tuple[int, int]:
    """Find a flow series and a window during which the site is shut throughout."""
    if flow_rows.size == 0:
        raise AnomalyError("the site has no flow series to inject a closed-gallery anomaly into")
    candidates: list[tuple[int, int]] = []
    n_steps = closed_mask.shape[1]
    for row in flow_rows:
        shut = closed_mask[row]
        for start in range(0, n_steps - duration):
            if shut[start : start + duration].all():
                candidates.append((int(row), start))
    if not candidates:
        raise AnomalyError(
            f"no {duration}-step window in which the site is shut; a closed-gallery "
            "anomaly cannot be placed"
        )
    choice = int(generator.integers(len(candidates)))
    return candidates[choice]


def anomaly_score(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile_levels: Sequence[float],
    *,
    floor: float = 0.5,
) -> np.ndarray:
    """How far each observation falls outside its predictive interval, in spread units.

    Args:
        actual: ``(n_series, T)`` of observations.
        prediction: ``(n_series, T, n_quantiles)`` of one-step forecasts for them.
        quantile_levels: the levels.
        floor: minimum spread, in persons. Without it a room the model is certain about --
            an empty store cupboard, where every quantile is zero -- produces an infinite
            score the moment one person walks in, and that single cell then sets the
            detection threshold for the whole site.

    Returns:
        A non-negative ``(n_series, T)`` score; zero inside the interval.
    """
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if prediction.shape[:2] != actual.shape:
        raise AnomalyError(
            f"prediction {prediction.shape} does not cover actual {actual.shape}"
        )
    lower_index = int(np.argmin(np.abs(levels - 0.1)))
    upper_index = int(np.argmin(np.abs(levels - 0.9)))
    lower = prediction[:, :, lower_index]
    upper = prediction[:, :, upper_index]
    spread = np.maximum((upper - lower) / 2.0, floor)
    below = np.maximum(lower - actual, 0.0)
    above = np.maximum(actual - upper, 0.0)
    score = (below + above) / spread
    return np.where(np.isfinite(actual), score, 0.0)


def calibrate_threshold(clean_score: np.ndarray, false_alarm_budget: float) -> float:
    """The score above which at most ``false_alarm_budget`` of clean cells are flagged.

    Raises:
        AnomalyError: if the budget is not a proper fraction.
    """
    if not 0.0 < false_alarm_budget < 1.0:
        raise AnomalyError(
            f"false_alarm_budget must lie strictly inside (0, 1), got {false_alarm_budget}"
        )
    finite = clean_score[np.isfinite(clean_score)]
    if finite.size == 0:
        raise AnomalyError("no finite clean scores to calibrate against")
    return float(np.quantile(finite, 1.0 - false_alarm_budget))


def score_detections(
    score: np.ndarray,
    injected: Sequence[InjectedAnomaly],
    threshold: float,
    clean_score: np.ndarray,
) -> DetectionOutcome:
    """Detection rate and time to detect at a calibrated threshold.

    Args:
        score: ``(n_series, T)`` anomaly score of the contaminated stream.
        injected: what was injected, from :func:`inject_anomalies`.
        threshold: from :func:`calibrate_threshold`.
        clean_score: the same score on the uncontaminated stream, used to report the
            false alarm rate actually achieved rather than the one asked for.
    """
    flagged = score > threshold
    times: list[int] = []
    detected = 0
    for anomaly in injected:
        window = flagged[anomaly.series_index, anomaly.start : anomaly.stop]
        if window.any():
            detected += 1
            times.append(int(np.argmax(window)))

    finite = np.isfinite(clean_score)
    false_alarm_rate = (
        float((clean_score[finite] > threshold).mean()) if finite.any() else float("nan")
    )
    return DetectionOutcome(
        detection_rate=detected / len(injected) if injected else float("nan"),
        median_time_to_detect_steps=float(np.median(times)) if times else float("nan"),
        false_alarm_rate=false_alarm_rate,
        threshold=threshold,
        n_injected=len(injected),
        n_detected=detected,
    )
