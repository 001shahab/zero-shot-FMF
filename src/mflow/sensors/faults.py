"""Sensor faults: dropout, stuck values and clock error (M3).

The three faults here are the ones that actually appear in deployed building sensor
records, and each breaks a different assumption a forecaster might make.

**Dropout** is bursty, not independent across time. A gateway that loses its uplink loses
it for minutes or hours, so it is modelled as a two-state Markov chain per sensor rather
than as a per-sample coin flip. A dropped sample is a NaN and stays a NaN: nothing here
fills a gap, because a zero written into an occupancy series is indistinguishable from an
empty room and would quietly become training data.

**Stuck values** are the failure mode that survives a null check. The sensor keeps
reporting, but reports the last value it saw. Modelled as a second Markov chain, so that
stuck episodes also come in runs.

**Clock error** is a fixed offset per sensor plus a drift that accumulates over the
record. It is applied by shifting the series, which means the first or last few samples
of a shifted sensor have no data to draw from and become NaN rather than being wrapped
around from the other end of the record.

Every fault is recorded. :class:`FaultRecord` carries the exact index of every sample
that was dropped, frozen or shifted, so a downstream imputation step can be evaluated
against where the gaps actually were.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mflow.sensors.config import FaultConfig


@dataclass(frozen=True)
class FaultRecord:
    """Where a sensor stream was damaged, and how.

    ``missing`` is every index with no reading, whatever the cause, so that it can be
    checked against the gaps actually present in the observed series. ``dropped`` is the
    subset attributable to the dropout chain; the remainder is the exposed end of a
    shifted clock.

    Attributes:
        missing: indices with no observation.
        dropped: indices lost to the dropout chain.
        stuck: indices holding a stale value, excluding any that ended up missing.
        clock_shift_intervals: the offset applied to this sensor, in sampling intervals,
            positive meaning the sensor reports the past as the present.
        length: number of samples in the stream.
    """

    missing: np.ndarray
    dropped: np.ndarray
    stuck: np.ndarray
    clock_shift_intervals: float
    length: int

    @property
    def missing_fraction(self) -> float:
        """Share of the record with no reading."""
        return float(len(self.missing) / self.length) if self.length else 0.0

    @property
    def dropout_fraction(self) -> float:
        """Share of the record lost to the dropout chain specifically."""
        return float(len(self.dropped) / self.length) if self.length else 0.0

    @property
    def stuck_fraction(self) -> float:
        """Share of the record reporting a stale value."""
        return float(len(self.stuck) / self.length) if self.length else 0.0


def _markov_episodes(
    n_steps: int,
    enter_probability: float,
    exit_probability: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample a boolean run-length process from a two-state Markov chain.

    The chain starts in the healthy state. Starting in the faulty state would make the
    fraction of damaged samples depend on the record length in a way that is hard to
    reason about when comparing profiles.
    """
    if enter_probability <= 0.0:
        return np.zeros(n_steps, dtype=bool)
    draws = generator.random(n_steps)
    state = np.zeros(n_steps, dtype=bool)
    faulty = False
    for index in range(n_steps):
        threshold = exit_probability if faulty else enter_probability
        if draws[index] < threshold:
            faulty = not faulty
        state[index] = faulty
    return state


class FaultModel:
    """Apply dropout, stuck values and clock error to one sensor stream at a time.

    Args:
        config: the faults section of a sensor profile.
        sensor_ids: every stream that can fail, in canonical order.
        generator: source of randomness for the per-sensor clock errors, drawn once.
    """

    def __init__(
        self,
        config: FaultConfig,
        sensor_ids: list[str],
        generator: np.random.Generator,
    ) -> None:
        self.config = config
        self.sensor_ids = list(sensor_ids)
        skew = generator.normal(0.0, config.clock_skew_intervals, size=len(self.sensor_ids))
        drift = generator.normal(0.0, config.clock_drift_intervals, size=len(self.sensor_ids))
        self.clock_skew: dict[str, float] = dict(
            zip(self.sensor_ids, map(float, skew), strict=True)
        )
        self.clock_drift: dict[str, float] = dict(
            zip(self.sensor_ids, map(float, drift), strict=True)
        )

    def apply(
        self,
        sensor_id: str,
        values: np.ndarray,
        generator: np.random.Generator,
    ) -> tuple[np.ndarray, FaultRecord]:
        """Damage one stream and report exactly what was damaged.

        Args:
            sensor_id: the stream being degraded, which selects its clock error.
            values: the clean series.
            generator: source of randomness for the dropout and stuck episodes.

        Returns:
            The observed series and the record of what happened to it.

        Raises:
            KeyError: if ``sensor_id`` was not declared at construction, which would mean
                the stream has no clock error and the caller has lost track of its
                sensor inventory.
        """
        if sensor_id not in self.clock_skew:
            raise KeyError(
                f"sensor {sensor_id!r} was not declared to the fault model; known sensors "
                f"are {len(self.sensor_ids)} streams starting {self.sensor_ids[:3]}"
            )
        observed = np.array(values, dtype=np.float64, copy=True)
        n_steps = observed.shape[0]

        shift = self._total_shift(sensor_id)
        if shift != 0:
            observed = _shift(observed, shift)

        stuck = _markov_episodes(
            n_steps,
            self.config.stuck_enter_probability,
            self.config.stuck_exit_probability,
            generator,
        )
        if stuck.any():
            observed = _freeze(observed, stuck)

        dropped = _markov_episodes(
            n_steps,
            self.config.dropout_enter_probability,
            self.config.dropout_exit_probability,
            generator,
        )
        observed[dropped] = np.nan
        # A shifted clock also leaves the record short at one end. Every gap is reported,
        # whatever produced it, so that a downstream imputation step can be scored against
        # where the data really was absent.
        missing = np.isnan(observed) & ~np.isnan(values)
        # A sample that was frozen and then lost is only reported as stuck if a consumer
        # could actually see the stale value.
        stuck &= ~missing

        return observed, FaultRecord(
            missing=np.flatnonzero(missing),
            dropped=np.flatnonzero(dropped),
            stuck=np.flatnonzero(stuck),
            clock_shift_intervals=shift,
            length=n_steps,
        )

    def _total_shift(self, sensor_id: str) -> int:
        """Whole-interval clock error for one sensor.

        Skew and drift are summed and rounded, because the canonical grid has no
        sub-interval resolution and pretending otherwise would mean resampling the
        series, which is itself a modelling choice this stage should not be making.
        """
        total = self.clock_skew[sensor_id] + self.clock_drift[sensor_id]
        return int(np.rint(total))


def _shift(values: np.ndarray, shift: int) -> np.ndarray:
    """Shift a series in time, padding the exposed end with NaN."""
    out = np.full_like(values, np.nan)
    if shift > 0:
        out[shift:] = values[:-shift]
    elif shift < 0:
        out[:shift] = values[-shift:]
    else:
        out[:] = values
    return out


def _freeze(values: np.ndarray, stuck: np.ndarray) -> np.ndarray:
    """Hold the last healthy value across each stuck episode."""
    out = np.array(values, copy=True)
    last = np.nan
    for index in range(out.shape[0]):
        if stuck[index]:
            out[index] = last
        else:
            last = out[index]
    return out
