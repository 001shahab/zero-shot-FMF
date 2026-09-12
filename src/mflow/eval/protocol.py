"""Rolling-origin evaluation protocol (M6).

This module owns the one thing the whole paper rests on: that no model ever sees data
from beyond its forecast origin, and that every method sees exactly the same origins with
exactly the same context. Both properties are enforced here rather than trusted to each
forecaster, because a leak in a single wrapper would invalidate every number.

The design is deliberately rigid:

* the split into train, validation and test is by time, computed once per site;
* the splits are separated by a gap of at least one horizon, so that a training target
  never overlaps a test context;
* origins are generated from the test window alone and are frozen in a
  :class:`RollingOriginPlan` before any model runs;
* a context is carved out with :meth:`Panel.slice_time`, which is the only code path that
  can produce a model input, and which cannot reach past the origin.

A trained model is fitted on the training panel, which ends before the gap. A zero-shot
model is handed the context window and nothing else. Normalisation statistics are not
computed here and must not be computed by the harness from the full series, because that
is leakage that no unit test on the forecaster would catch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from mflow.schema import Panel, SiteData

#: Default fractions of the record given to train, validation and test.
DEFAULT_SPLIT: Final[tuple[float, float, float]] = (0.6, 0.2, 0.2)


class ProtocolError(ValueError):
    """Raised when an evaluation plan is impossible or unsafe."""


@dataclass(frozen=True)
class TimeSplit:
    """Index boundaries of the three time windows, as half-open intervals.

    Attributes:
        train: ``[0, train_end)``.
        validation: ``[validation_start, validation_end)``.
        test: ``[test_start, n_timesteps)``.
        gap: number of steps left empty between consecutive windows.
        timestamps: the grid the indices refer to, kept so that a split can be reported
            in wall-clock terms without the caller having to hold on to the panel.
    """

    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    gap: int
    timestamps: pd.DatetimeIndex

    def describe(self) -> dict[str, str]:
        """Human-readable boundaries, for a run manifest."""
        stamps = self.timestamps
        return {
            "train": f"{stamps[0]} .. {stamps[self.train_end - 1]}",
            "validation": f"{stamps[self.validation_start]} .. {stamps[self.validation_end - 1]}",
            "test": f"{stamps[self.test_start]} .. {stamps[self.test_end - 1]}",
            "gap_steps": str(self.gap),
        }

    @property
    def n_train(self) -> int:
        """Length of the training window in steps."""
        return self.train_end

    @property
    def n_test(self) -> int:
        """Length of the test window in steps."""
        return self.test_end - self.test_start


def make_split(
    n_timesteps: int,
    timestamps: pd.DatetimeIndex,
    *,
    horizon: int,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    gap: int | None = None,
) -> TimeSplit:
    """Split a record into train, validation and test windows separated by a gap.

    Args:
        n_timesteps: length of the record.
        timestamps: the grid, used only for reporting.
        horizon: the longest horizon that will be evaluated. The gap defaults to this,
            which is the smallest value that stops a training target from overlapping a
            test context.
        fractions: relative sizes of the three windows, before the gaps are removed.
        gap: override the gap in steps.

    Raises:
        ProtocolError: if the fractions are invalid, or the record is too short to hold
            three windows and two gaps.
    """
    if len(timestamps) != n_timesteps:
        raise ProtocolError(
            f"split was given {n_timesteps} steps but {len(timestamps)} timestamps"
        )
    if any(f <= 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ProtocolError(f"split fractions must be positive and sum to 1, got {fractions}")
    if horizon < 1:
        raise ProtocolError(f"horizon must be at least 1, got {horizon}")

    separation = horizon if gap is None else gap
    if separation < 0:
        raise ProtocolError(f"gap must not be negative, got {separation}")

    usable = n_timesteps - 2 * separation
    if usable < 3:
        raise ProtocolError(
            f"a record of {n_timesteps} steps cannot hold three windows separated by "
            f"{separation}-step gaps"
        )
    n_train = int(usable * fractions[0])
    n_validation = int(usable * fractions[1])
    n_test = usable - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ProtocolError(
            f"split {fractions} of {usable} usable steps leaves an empty window: "
            f"train={n_train}, validation={n_validation}, test={n_test}"
        )

    validation_start = n_train + separation
    validation_end = validation_start + n_validation
    test_start = validation_end + separation
    return TimeSplit(
        train_end=n_train,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_start + n_test,
        gap=separation,
        timestamps=timestamps,
    )


@dataclass(frozen=True)
class ForecastTask:
    """One forecast origin.

    Attributes:
        origin: index of the first step to be predicted. The context is everything
            strictly before it.
        context_start: index of the first step of the context window.
        horizon: number of steps to predict.
    """

    origin: int
    context_start: int
    horizon: int

    @property
    def target_slice(self) -> slice:
        """Index range of the values being predicted."""
        return slice(self.origin, self.origin + self.horizon)


@dataclass(frozen=True)
class RollingOriginPlan:
    """A frozen list of forecast tasks, shared by every method in a run.

    Attributes:
        tasks: the origins, in time order.
        split: the time split they were drawn from.
        context_length: context length in steps, identical for every task.
        horizon: the longest horizon evaluated.
        horizons: the horizons reported separately, in steps.
        quantiles: the quantile levels every method must produce.
        dropped: origins that were enumerated and then found unusable, mapped to why.
            Recorded rather than discarded: a run over a record with a two-month hole in
            it evaluates on fewer origins than the stride implies, and the reader of the
            results has to be able to see that.
    """

    tasks: tuple[ForecastTask, ...]
    split: TimeSplit
    context_length: int
    horizon: int
    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]
    dropped: Mapping[int, str] = MappingProxyType({})

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[ForecastTask]:
        return iter(self.tasks)

    def origins(self) -> list[int]:
        """Origin indices, in order."""
        return [task.origin for task in self.tasks]

    def origin_timestamps(self) -> pd.DatetimeIndex:
        """Wall-clock time of the first predicted step at each origin."""
        return pd.DatetimeIndex([self.split.timestamps[task.origin] for task in self.tasks])

    def describe(self) -> dict[str, object]:
        """Summary for a run manifest."""
        reasons = Counter(self.dropped.values())
        return {
            "n_origins": len(self.tasks),
            "context_length": self.context_length,
            "horizon": self.horizon,
            "horizons": list(self.horizons),
            "quantiles": list(self.quantiles),
            "n_origins_dropped": len(self.dropped),
            "dropped_reasons": dict(sorted(reasons.items())),
            **self.split.describe(),
        }


def build_plan(
    panel: Panel,
    *,
    context_length: int,
    horizons: Sequence[int],
    stride: int,
    quantiles: Sequence[float],
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    gap: int | None = None,
    max_origins: int | None = None,
    require_observed: bool = False,
) -> RollingOriginPlan:
    """Enumerate the forecast origins for one site.

    Every method in a run is driven from the plan returned here, so a method cannot
    quietly evaluate on an easier subset of the test window.

    Args:
        panel: the full site panel.
        context_length: steps of history each origin is given.
        horizons: horizons to report, in steps. The longest is what is actually
            forecast; the shorter ones are prefixes of it, which is what makes the
            per-horizon comparison fair.
        stride: spacing between origins, in steps.
        quantiles: quantile levels every method must produce.
        fractions: train/validation/test fractions.
        gap: gap between splits, defaulting to the longest horizon.
        max_origins: keep at most this many origins, evenly spaced, which is how a
            smoke run is made cheap without changing the protocol.
        require_observed: drop origins whose context or target is too empty to evaluate.
            Off by default, because simulated sites are complete and dropping nothing is
            the stricter guarantee. Real records are not complete -- ROBOD, for instance,
            was collected in weekday blocks and has a two-month hole in the middle -- and
            over those an origin in the hole has no anchor for the conservation identity
            and no truth to score against. Every dropped origin is recorded in
            :attr:`RollingOriginPlan.dropped` and counted in the run manifest.

    Raises:
        ProtocolError: if the arguments cannot produce a single valid origin.
    """
    if context_length < 1:
        raise ProtocolError(f"context_length must be at least 1, got {context_length}")
    if stride < 1:
        raise ProtocolError(f"stride must be at least 1, got {stride}")
    if not horizons:
        raise ProtocolError("at least one horizon is required")
    if any(h < 1 for h in horizons):
        raise ProtocolError(f"horizons must be positive, got {sorted(horizons)}")

    ordered_horizons = tuple(sorted({int(h) for h in horizons}))
    horizon = ordered_horizons[-1]
    split = make_split(
        panel.n_timesteps,
        panel.timestamps,
        horizon=horizon,
        fractions=fractions,
        gap=gap,
    )

    # An origin is valid when it has a full context behind it and a full horizon ahead of
    # it. Allowing a short context would make the comparison between methods depend on
    # how each one pads, which is a property of the wrapper, not of the model.
    first = max(split.test_start, context_length)
    last = split.test_end - horizon
    if first > last:
        raise ProtocolError(
            f"no valid origin: the test window is [{split.test_start}, {split.test_end}) "
            f"but an origin needs {context_length} steps of context and {horizon} ahead"
        )

    origins = list(range(first, last + 1, stride))

    dropped: dict[int, str] = {}
    if require_observed:
        # A channel that is NaN across the entire record is a site-level absence, not an
        # origin-level hole: ROBOD measures no doorway flows at all, so every flow series
        # is empty everywhere and no choice of origin would fix that. Judging origins
        # against such a channel would reject all of them for a reason that has nothing to
        # do with the origin.
        live = np.isfinite(panel.series).any(axis=1)
        kept = []
        for origin in origins:
            reason = _unusable(panel, origin, context_length, horizon, live)
            if reason is None:
                kept.append(origin)
            else:
                dropped[origin] = reason
        if not kept:
            raise ProtocolError(
                f"all {len(origins)} enumerated origins are unusable: "
                f"{dict(sorted(Counter(dropped.values()).items()))}. The record has no "
                "window with both a full context and an observed target."
            )
        origins = kept

    # Thinning comes after the eligibility filter so that a capped run keeps `max_origins`
    # usable origins rather than `max_origins` candidates of which most are holes.
    if max_origins is not None and len(origins) > max_origins:
        keep = np.linspace(0, len(origins) - 1, max_origins).round().astype(int)
        origins = [origins[i] for i in dict.fromkeys(keep.tolist())]

    tasks = tuple(
        ForecastTask(origin=o, context_start=o - context_length, horizon=horizon)
        for o in origins
    )
    return RollingOriginPlan(
        tasks=tasks,
        split=split,
        context_length=context_length,
        horizon=horizon,
        horizons=ordered_horizons,
        quantiles=tuple(float(q) for q in quantiles),
        dropped=MappingProxyType(dropped),
    )


def _unusable(
    panel: Panel, origin: int, context_length: int, horizon: int, live: np.ndarray
) -> str | None:
    """Why this origin cannot be evaluated, or None if it can.

    Two things have to hold, judged only over the ``live`` channels. Every live series
    needs at least one finite observation in its context, because that is what anchors
    the conservation identity and what any forecaster conditions on. And the target
    window needs at least one finite value somewhere, because an origin scored entirely
    against NaN contributes nothing to any metric while still counting as an origin in
    every table and every paired test.
    """
    context = panel.series[live, origin - context_length : origin]
    if not np.isfinite(context).any(axis=1).all():
        return "a target series has no observation anywhere in its context"
    if not np.isfinite(panel.series[live, origin : origin + horizon]).any():
        return "the target window is entirely unobserved"
    return None


def context_panel(panel: Panel, task: ForecastTask) -> Panel:
    """Carve out the model input for one origin.

    This is the only way a forecaster is given data during evaluation. The slice ends at
    the origin, so the returned panel physically cannot contain a target value; the
    known-future covariates for the horizon are appended because they are, by
    construction, known at the origin.
    """
    return panel.slice_time(task.context_start, task.origin, horizon=task.horizon)


def training_panel(panel: Panel, split: TimeSplit, *, include_validation: bool) -> Panel:
    """The panel a trained model is allowed to fit on.

    Args:
        panel: the full site panel.
        split: the time split.
        include_validation: whether to include the validation window. Fitting on it is
            legitimate once hyperparameters are fixed, and it is what the data-efficiency
            experiment varies.
    """
    stop = split.validation_end if include_validation else split.train_end
    return panel.slice_time(0, stop)


def truth_for(panel: Panel, task: ForecastTask) -> np.ndarray:
    """The ``(n_series, horizon)`` actuals a forecast at ``task`` is scored against."""
    return panel.series[:, task.target_slice]


def limit_days(site: SiteData, panel: Panel, split: TimeSplit, days: int) -> Panel:
    """The last ``days`` days of the training window, for the data-efficiency sweep.

    Args:
        site: the site, for its sampling interval.
        panel: the full site panel.
        split: the time split.
        days: how many days of local history the model is allowed.

    Raises:
        ProtocolError: if the training window is shorter than the requested budget, which
            would silently turn a 90-day condition into a 40-day one and invent a
            crossover point that the data does not support.
    """
    steps_per_day = 24 * 3600 // site.meta.interval_seconds
    wanted = days * steps_per_day
    if wanted > split.train_end:
        raise ProtocolError(
            f"{days} days is {wanted} steps but the training window is only "
            f"{split.train_end} steps for site {site.meta.site_id!r}"
        )
    return panel.slice_time(split.train_end - wanted, split.train_end)
