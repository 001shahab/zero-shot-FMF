"""Sensor degradation profiles (M3).

A profile is the complete description of how a clean canonical site is turned into an
observed one. Every parameter that changes the observed data lives here and nowhere
else, so an experiment can record the profile name and config hash and have that be a
complete account of the measurement model it ran under.

Shipped profiles are in ``configs/sensors/``: ``clean``, ``realistic``, ``degraded`` and
``harsh``. The published values the defaults are drawn from are cited in the modules that
use them -- :mod:`mflow.sensors.counting`, :mod:`mflow.sensors.environment` and
:mod:`mflow.sensors.faults`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CountingConfig(BaseModel):
    """Doorway counter error model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Probability a crossing is missed when the doorway is otherwise empty.
    base_miss_probability: float = Field(ge=0.0, le=1.0)
    #: Additional miss probability at ``congestion_reference`` simultaneous crossings,
    #: which is how group occlusion enters the model.
    congestion_miss_probability: float = Field(ge=0.0, le=1.0)
    #: Crossings per interval at which the congestion term reaches its full value.
    congestion_reference: float = Field(gt=0.0)
    #: Probability that a single crossing is registered twice.
    double_count_probability: float = Field(ge=0.0, le=1.0)
    #: Standard deviation of the fixed multiplicative bias drawn once per sensor.
    bias_sigma: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _miss_probability_is_bounded(self) -> CountingConfig:
        total = self.base_miss_probability + self.congestion_miss_probability
        if total > 1.0:
            raise ValueError(
                "base_miss_probability + congestion_miss_probability must not exceed 1, "
                f"got {total}"
            )
        return self


class EnvironmentConfig(BaseModel):
    """CO2, temperature and humidity generation from occupancy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Emit environmental series at all.
    enabled: bool = True
    #: Ceiling height used to turn node floor area into air volume, metres.
    ceiling_height_m: float = Field(gt=0.0)
    #: Air changes per hour.
    air_changes_per_hour: float = Field(gt=0.0)
    #: CO2 generation per person, litres per second at 101 kPa and 273 K.
    co2_generation_l_per_s: float = Field(gt=0.0)
    #: Outdoor CO2 concentration, ppm.
    outdoor_co2_ppm: float = Field(gt=0.0)
    #: Sensor noise, as the fixed and proportional parts of the datasheet accuracy.
    co2_noise_ppm: float = Field(ge=0.0)
    co2_noise_fraction: float = Field(ge=0.0)
    #: Lifetime calibration offset drawn once per sensor, ppm standard deviation.
    co2_offset_ppm: float = Field(ge=0.0)
    #: Sensor response time constant, seconds, applied as a first-order lag.
    co2_response_seconds: float = Field(ge=0.0)
    #: Baseline conditions with the room empty.
    baseline_temperature_c: float
    baseline_humidity_pct: float = Field(ge=0.0, le=100.0)
    #: Sensible and latent heat per person, watts.
    sensible_heat_w: float = Field(gt=0.0)
    latent_heat_w: float = Field(gt=0.0)
    #: Share of the occupant heat and moisture gain removed by conditioning. Heritage
    #: museums control temperature and humidity tightly for the collection's sake, which
    #: is precisely what makes those two channels weak occupancy proxies and CO2 a usable
    #: one. Zero is an unconditioned building; one is a perfect thermostat.
    hvac_rejection_fraction: float = Field(ge=0.0, le=1.0)
    #: Measurement noise for the temperature and humidity channels.
    temperature_noise_c: float = Field(ge=0.0)
    humidity_noise_pct: float = Field(ge=0.0)


class FaultConfig(BaseModel):
    """Dropout, clock skew and stuck-value faults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Two-state Markov chain, per sampling interval.
    dropout_enter_probability: float = Field(ge=0.0, le=1.0)
    dropout_exit_probability: float = Field(ge=0.0, le=1.0)
    #: Probability per interval of entering a stuck-value episode, and of leaving one.
    stuck_enter_probability: float = Field(ge=0.0, le=1.0)
    stuck_exit_probability: float = Field(ge=0.0, le=1.0)
    #: Constant clock offset per sensor, in sampling intervals, standard deviation.
    clock_skew_intervals: float = Field(ge=0.0)
    #: Clock drift accumulated over the record, in sampling intervals, standard deviation.
    clock_drift_intervals: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _absorbing_states_are_rejected(self) -> FaultConfig:
        if self.dropout_enter_probability > 0.0 and self.dropout_exit_probability <= 0.0:
            raise ValueError(
                "dropout_exit_probability must be positive when dropouts can start, "
                "otherwise the first dropout never ends"
            )
        if self.stuck_enter_probability > 0.0 and self.stuck_exit_probability <= 0.0:
            raise ValueError(
                "stuck_exit_probability must be positive when stuck episodes can start, "
                "otherwise the first one never ends"
            )
        return self


class SensorProfile(BaseModel):
    """A named measurement model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    counting: CountingConfig
    environment: EnvironmentConfig
    faults: FaultConfig

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, for hashing into a run manifest."""
        return self.model_dump(mode="json")


def load_sensor_profile(path: str | Path) -> SensorProfile:
    """Read a profile from YAML.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the profile is malformed.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"sensor profile not found: {file}")
    payload = yaml.safe_load(file.read_text(encoding="utf-8"))
    return SensorProfile.model_validate(payload)
