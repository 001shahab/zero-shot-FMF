"""Sensor degradation: clean canonical series to observed canonical series (M3)."""

from mflow.sensors.config import (
    CountingConfig,
    EnvironmentConfig,
    FaultConfig,
    SensorProfile,
    load_sensor_profile,
)
from mflow.sensors.counting import CountingSensorModel, counting_error_summary
from mflow.sensors.environment import (
    ENVIRONMENT_VARIABLES,
    EnvironmentSensorModel,
    co2_lag_minutes,
)
from mflow.sensors.faults import FaultModel, FaultRecord
from mflow.sensors.pipeline import DegradationReport, degrade

__all__ = [
    "ENVIRONMENT_VARIABLES",
    "CountingConfig",
    "CountingSensorModel",
    "DegradationReport",
    "EnvironmentConfig",
    "EnvironmentSensorModel",
    "FaultConfig",
    "FaultModel",
    "FaultRecord",
    "SensorProfile",
    "co2_lag_minutes",
    "counting_error_summary",
    "degrade",
    "load_sensor_profile",
]
