"""Arrival processes and group injection.

Individual visitors arrive as a non-homogeneous Poisson process whose intensity is the
site's peak rate scaled by an hour-of-day profile and a weekday multiplier. Groups --
coach parties, school parties, guided tours -- are injected deterministically at their
scheduled times with a random size, because that is how they actually happen: a coach
booked for 10:30 arrives at 10:30 whatever the Poisson process is doing, and it is
exactly those step changes that break naive counting models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mflow.sim.config import GroupConfig, SiteConfig


@dataclass(frozen=True)
class Arrival:
    """One person entering the building.

    Attributes:
        step: simulation step of arrival.
        entrance: node they enter at.
        group_id: the scheduled group they belong to, or None for a walk-in.
        cohesion: probability of following the group rather than moving independently.
        guided: whether they are on a guided tour and so follow the suggested route.
    """

    step: int
    entrance: str
    group_id: str | None
    cohesion: float
    guided: bool


def _minutes_since_midnight(stamp: str) -> int:
    hours, _, minutes = stamp.partition(":")
    return int(hours) * 60 + int(minutes)


def opening_window(config: SiteConfig, day: pd.Timestamp) -> tuple[int, int]:
    """First and last simulation step of the opening window on ``day``."""
    del day
    step = config.step_seconds
    open_step = _minutes_since_midnight(config.arrivals.open_time) * 60 // step
    close_step = _minutes_since_midnight(config.arrivals.close_time) * 60 // step
    if close_step <= open_step:
        raise ValueError(
            f"site {config.site_id!r} closes at {config.arrivals.close_time} which is not "
            f"after it opens at {config.arrivals.open_time}"
        )
    return open_step, close_step


def intensity_per_step(config: SiteConfig, day: pd.Timestamp) -> np.ndarray:
    """Arrival intensity for every step of one day, in persons per step.

    The hourly profile is interpolated onto the step grid rather than held piecewise
    constant, so that the arrival rate does not jump discontinuously on the hour.
    """
    arrivals = config.arrivals
    steps_per_day = 24 * 3600 // config.step_seconds
    intensity = np.zeros(steps_per_day, dtype=np.float64)
    open_step, close_step = opening_window(config, day)

    profile = np.asarray(arrivals.hourly_profile, dtype=np.float64)
    open_steps = close_step - open_step
    positions = np.linspace(0.0, 1.0, len(profile))
    query = np.linspace(0.0, 1.0, open_steps)
    shape = np.interp(query, positions, profile)

    weekday = int(day.dayofweek)
    per_minute = arrivals.peak_rate_per_min * arrivals.weekday_multiplier[weekday]
    intensity[open_step:close_step] = shape * per_minute * config.step_seconds / 60.0
    return intensity


def _group_occurs(group: GroupConfig, day: pd.Timestamp) -> bool:
    return not group.weekdays or int(day.dayofweek) in group.weekdays


def sample_day_arrivals(
    config: SiteConfig, day: pd.Timestamp, generator: np.random.Generator
) -> list[Arrival]:
    """Generate every arrival for one day, walk-ins and groups together.

    Args:
        config: the site.
        day: midnight of the day being simulated, in local time.
        generator: the day's random stream.

    Returns:
        Arrivals sorted by step.
    """
    entrances = config.arrivals.entrances or {
        node.id: 1.0 for node in config.nodes if node.kind == "entrance"
    }
    if not entrances:
        raise ValueError(
            f"site {config.site_id!r} declares no entrances, so nobody can get in; add "
            "arrivals.entrances or a node of kind 'entrance'"
        )
    names = list(entrances)
    weights = np.array([entrances[name] for name in names], dtype=np.float64)
    weights = weights / weights.sum()

    arrivals: list[Arrival] = []
    intensity = intensity_per_step(config, day)
    counts = generator.poisson(intensity)
    for step in np.nonzero(counts)[0]:
        for _ in range(int(counts[step])):
            entrance = names[int(generator.choice(len(names), p=weights))]
            arrivals.append(
                Arrival(
                    step=int(step),
                    entrance=entrance,
                    group_id=None,
                    cohesion=0.0,
                    guided=False,
                )
            )

    for group in config.groups:
        if not _group_occurs(group, day):
            continue
        size = max(1, round(float(generator.normal(group.size_mean, group.size_sd))))
        group_step = int(_minutes_since_midnight(group.time) * 60 // config.step_seconds)
        entrance = group.entrance or names[0]
        for _ in range(size):
            arrivals.append(
                Arrival(
                    step=group_step,
                    entrance=entrance,
                    group_id=group.id,
                    cohesion=group.cohesion,
                    guided=group.guided,
                )
            )

    arrivals.sort(key=lambda a: (a.step, a.group_id or "", a.entrance))
    return arrivals


def scheduled_group_covariates(
    config: SiteConfig, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Known-future covariates describing the booked schedule.

    These are legitimately known in advance -- the museum sold the tickets -- so they go
    into ``covariates_future.parquet``: ``timed_slot_admissions`` (booked heads arriving
    in the interval), ``tour_departure`` (a guided tour leaves in this interval) and
    ``group_size_booked`` (expected heads in the largest group of the interval).

    Args:
        config: the site.
        index: the canonical UTC time grid.

    Returns:
        A long frame with columns ``timestamp, scope, variable, value``.
    """
    local = index.tz_convert(config.timezone)
    interval = pd.Timedelta(seconds=config.interval_seconds)
    slot_admissions = np.zeros(len(index), dtype=np.float64)
    tour_departure = np.zeros(len(index), dtype=np.float64)
    group_size = np.zeros(len(index), dtype=np.float64)

    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    interval_minutes = int(interval.total_seconds() // 60)
    for group in config.groups:
        group_minute = _minutes_since_midnight(group.time)
        slot = (group_minute // interval_minutes) * interval_minutes
        mask = minutes == slot
        if group.weekdays:
            mask = mask & np.isin(local.dayofweek.to_numpy(), group.weekdays)
        slot_admissions[mask] += group.size_mean
        group_size[mask] = np.maximum(group_size[mask], group.size_mean)
        if group.guided:
            tour_departure[mask] = 1.0

    open_minute = _minutes_since_midnight(config.arrivals.open_time)
    close_minute = _minutes_since_midnight(config.arrivals.close_time)
    is_open = ((minutes >= open_minute) & (minutes < close_minute)).astype(np.float64)

    frames = [
        pd.DataFrame(
            {"timestamp": index, "scope": "global", "variable": name, "value": values}
        )
        for name, values in (
            ("is_open", is_open),
            ("timed_slot_admissions", slot_admissions),
            ("tour_departure", tour_departure),
            ("group_size_booked", group_size),
        )
    ]
    return pd.concat(frames, ignore_index=True)
