"""Cumulative visitor load against a conservation budget (M6).

Preventive conservation limits how much visitor exposure an object or a room can take:
people bring heat, moisture, dust and vibration, and a gallery that hosts a thousand
people a day ages faster than one that hosts a hundred. A forecast of cumulative load is
the input to a decision about capping admissions or rerouting a tour.

**This module reports a formulation and a worked example. It makes no validation claim.**

There is no dataset in this project, and to the author's knowledge no public dataset, that
links a measured conservation outcome to a measured visitor load at room level. The
constants below therefore express a budget the caller sets, not a damage function anyone
has fitted. Nothing here should be read as evidence that a particular load causes a
particular amount of harm, and the paper must say so wherever these numbers appear.

What the code does support is the operational question: given a forecast, is a room on
track to exceed the budget its curator has set for it, and by when? That is a statement
about the forecast, not about conservation science, and it is testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ExposureError(ValueError):
    """Raised when an exposure projection cannot be computed as specified."""


@dataclass(frozen=True)
class ExposureBudget:
    """A curator-set limit on visitor load for one room.

    Attributes:
        node_id: the room.
        person_minutes_per_day: the budget, in person-minutes. Person-minutes rather than
            a headcount because a room that holds ten people all day is under more load
            than one that ten people walk through.
        rationale: free text recording who set the budget and why. Required, because a
            budget without a stated origin will be mistaken for a measurement.
    """

    node_id: str
    person_minutes_per_day: float
    rationale: str

    def __post_init__(self) -> None:
        if self.person_minutes_per_day <= 0:
            raise ExposureError(
                f"budget for {self.node_id!r} must be positive, got "
                f"{self.person_minutes_per_day}"
            )
        if not self.rationale.strip():
            raise ExposureError(
                f"budget for {self.node_id!r} has no rationale; an unexplained budget "
                "will be read as a measured limit"
            )


def person_minutes(
    occupancy: np.ndarray, interval_seconds: int, *, axis: int = -1
) -> np.ndarray:
    """Integrate occupancy into person-minutes.

    Args:
        occupancy: headcount per interval.
        interval_seconds: sampling interval.
        axis: the time axis.

    Missing steps contribute nothing, which under-counts rather than over-counts the
    load. That direction is the safe one for a conservation limit only if the caller
    knows it: a budget check on a series with gaps reports less exposure than really
    occurred, so :func:`project_exposure` reports the observed fraction alongside.
    """
    minutes = interval_seconds / 60.0
    return np.nansum(occupancy, axis=axis) * minutes


@dataclass(frozen=True)
class ExposureProjection:
    """Projected load for one room on one day.

    Attributes:
        node_id: the room.
        day: the local date.
        observed_person_minutes: load already accumulated from observations.
        forecast_person_minutes: load the forecast adds over the remainder.
        budget_person_minutes: the curator's limit.
        observed_fraction: share of the day's steps that carried a reading. A projection
            built on a half-observed day understates the load and this is how the reader
            knows.
    """

    node_id: str
    day: pd.Timestamp
    observed_person_minutes: float
    forecast_person_minutes: float
    budget_person_minutes: float
    observed_fraction: float

    @property
    def projected_person_minutes(self) -> float:
        """Total load expected by the end of the day."""
        return self.observed_person_minutes + self.forecast_person_minutes

    @property
    def budget_utilisation(self) -> float:
        """Projected load as a fraction of the budget."""
        return self.projected_person_minutes / self.budget_person_minutes

    @property
    def exceeds_budget(self) -> bool:
        """Whether the projection is over the limit."""
        return self.budget_utilisation > 1.0


def project_exposure(
    observed: np.ndarray,
    forecast: np.ndarray,
    budget: ExposureBudget,
    day: pd.Timestamp,
    interval_seconds: int,
) -> ExposureProjection:
    """Project one room's load for one day from what has happened and what is forecast.

    Args:
        observed: occupancy already observed today, one value per interval.
        forecast: median occupancy forecast for the rest of the day.
        budget: the room's limit.
        day: the local date the projection is for.
        interval_seconds: sampling interval.
    """
    if observed.ndim != 1 or forecast.ndim != 1:
        raise ExposureError(
            f"expected one series each, got {observed.shape} and {forecast.shape}"
        )
    total = observed.size
    return ExposureProjection(
        node_id=budget.node_id,
        day=day,
        observed_person_minutes=float(person_minutes(observed, interval_seconds)),
        forecast_person_minutes=float(person_minutes(forecast, interval_seconds)),
        budget_person_minutes=budget.person_minutes_per_day,
        observed_fraction=float(np.isfinite(observed).mean()) if total else 0.0,
    )


def worked_example(
    site_occupancy: pd.DataFrame,
    node_id: str,
    interval_seconds: int,
    *,
    headroom: float = 1.2,
) -> tuple[ExposureBudget, pd.DataFrame]:
    """Build an illustrative budget from a room's own history and apply it.

    The budget is set at ``headroom`` times the room's median daily load, which is a
    statement about the room's normal operation and nothing more. It is offered so the
    formulation can be demonstrated on real numbers; it is not a conservation limit and
    the returned :attr:`ExposureBudget.rationale` says so.

    Args:
        site_occupancy: canonical occupancy frame.
        node_id: the room to illustrate with.
        interval_seconds: sampling interval.
        headroom: multiple of the median daily load used as the budget.

    Returns:
        The budget and a frame of daily load against it.

    Raises:
        ExposureError: if the room is not in the frame or has no full day of data.
    """
    frame = site_occupancy[site_occupancy["node_id"].astype(str) == node_id]
    if frame.empty:
        raise ExposureError(f"node {node_id!r} is not in the occupancy frame")
    daily = (
        frame.assign(day=frame["timestamp"].dt.floor("D"))
        .groupby("day")["count"]
        .agg(person_minutes=lambda s: float(np.nansum(s)) * interval_seconds / 60.0)
        .reset_index()
    )
    if daily.empty:
        raise ExposureError(f"node {node_id!r} has no daily totals to summarise")

    median = float(daily["person_minutes"].median())
    if median <= 0:
        raise ExposureError(
            f"node {node_id!r} has a median daily load of {median}; there is nothing to "
            "illustrate a budget against"
        )
    budget = ExposureBudget(
        node_id=node_id,
        person_minutes_per_day=headroom * median,
        rationale=(
            f"Illustrative only: {headroom:g} times this room's own median daily load "
            "over the observed record. Not a conservation limit and not derived from any "
            "measured damage relationship."
        ),
    )
    daily["budget_person_minutes"] = budget.person_minutes_per_day
    daily["budget_utilisation"] = daily["person_minutes"] / budget.person_minutes_per_day
    daily["exceeds_budget"] = daily["budget_utilisation"] > 1.0
    return budget, daily
