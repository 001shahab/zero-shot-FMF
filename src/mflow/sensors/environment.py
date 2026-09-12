"""Environmental proxy sensors driven by occupancy (M3).

Museums instrument rooms for conservation reasons long before they instrument them for
crowd management, so CO2, temperature and humidity are often the only per-room signals a
heritage site already has. They are proxies: they respond to occupancy with a lag set by
the room's air volume and ventilation rate, which is exactly why a forecaster that only
sees them is at a disadvantage, and exactly what experiment E5 is meant to quantify.

Physics
-------

CO2 follows the single-zone mass balance used throughout the occupancy-estimation
literature (Wang et al. 1999; Zuraimi et al. 2017, *Predicting occupancy counts using
physical and statistical CO2-based modelling methodologies*, Building and Environment
123, 517-528):

.. math::

    V \\frac{dC}{dt} = 10^6 G n(t) - Q (C - C_\\text{out})

with :math:`V` the zone air volume in m^3, :math:`Q` the outdoor airflow in m^3/s,
:math:`G` the CO2 generation per person in m^3/s, :math:`C` in ppm and :math:`n(t)` the
headcount. Because :math:`n` is constant within one sampling interval, the equation is
integrated exactly rather than stepped with a forward difference, so the result does not
depend on the interval length.

Temperature and humidity use the same first-order form driven by the sensible and latent
heat of the occupants.

Parameters
----------

*CO2 generation.* Persily and de Jonge (2017), *Carbon dioxide generation rates for
building occupants*, Indoor Air 27(5), 868-879, give the generation rate as
:math:`V_{CO_2} = 0.000484 \\cdot BMR \\cdot M` L/s at 101 kPa and 273 K, with BMR in
MJ/day and M in met. Their worked example, an 85 kg male aged 30-60 with BMR 7.73 MJ/day
at 1.5 met, gives 0.0056 L/s, "close to the value of 0.0052 L/s cited in ASHRAE Standard
62.1 and ASTM D6245 for an adult". Museum visiting sits between their "standing quietly"
(1.3 met) and "walking, less than 2 mph, level surface, very slow" (2.0 met), so the
shipped profiles use 0.0060 L/s.

*Ventilation.* ASHRAE 62.1 Table 6-1 sets the breathing-zone outdoor air rate for
"Museums/galleries" at 3.8 L/s per person plus 0.3 L/s per m^2, at a default occupant
density of 40 people per 1000 ft^2 (4.6 m^2 per person). At that density the design
outdoor airflow is about 1.13 L/s per m^2 of floor, which for the 4 m ceilings the
profiles assume is close to one air change per hour. Real installations run above the
minimum, and the shipped profiles use 2 air changes per hour, which puts the CO2 response
time in the range the field studies report.

*Sensor error.* The Sensirion SCD30 datasheet, a representative NDIR module, specifies
accuracy of +/-(30 ppm + 3% of measured value) over 400-10000 ppm, repeatability
+/-10 ppm, accuracy drift over lifetime +/-50 ppm, and a response time tau_63 of 20 s.
The 20 s sensor time constant is an order of magnitude shorter than the transport lag,
consistent with Rahman and Han's finding, quoted in Fan et al. (2022), that "the time
delay of the change in CO2 concentration is mainly due to the CO2 dispersion time rather
than the sensor response time".

*Heat gain.* ASHRAE Handbook -- Fundamentals, Chapter 18, Table 4, "Standing, light work;
walking" (department store; retail store): adjusted total heat 132 W, of which 73 W
sensible and 59 W latent, at a 23.9 C room dry-bulb temperature. Most of that gain is
removed by the conditioning a heritage museum runs for the collection's sake, which is
why ``hvac_rejection_fraction`` exists and why temperature and humidity are much weaker
occupancy proxies here than CO2: the plant fights the thermal load but does not scrub CO2.
Note also that a crowded room's relative humidity can *fall* even as its moisture content
rises, because the air warms faster than it takes up water.

*Expected behaviour.* The reported transport lag between an occupancy change and the CO2
response is 10-20 min averaged over the zones of a multi-zone office building (Meyn
et al.) and 30-45 min under six ventilation schemes (Rahman et al.), both quoted in Fan,
Ding and Sun (2022), *The nexus of the indoor CO2 concentration and ventilation demands
underlying CO2-based demand-controlled ventilation in commercial buildings: a critical
review*, Building and Environment 217, 109063. :func:`co2_lag_minutes` measures the lag
this module produces so that a profile can be checked against that range.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from mflow.sensors.config import EnvironmentConfig

#: Volumetric heat capacity of air at room conditions, J/(m^3 K): 1.2 kg/m^3 x 1005 J/(kg K).
_AIR_HEAT_CAPACITY: Final[float] = 1206.0
#: Density of air at room conditions, kg/m^3.
_AIR_DENSITY: Final[float] = 1.2
#: Latent heat of vaporisation of water near room temperature, J/kg.
_LATENT_HEAT_WATER: Final[float] = 2.45e6
#: Standard atmospheric pressure, Pa.
_ATMOSPHERIC_PRESSURE: Final[float] = 101325.0
#: Ratio of the molar masses of water vapour and dry air.
_MOLAR_RATIO: Final[float] = 0.62198

ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = ("co2_ppm", "temp_c", "rh_pct")


def saturation_vapour_pressure(temperature_c: np.ndarray | float) -> np.ndarray:
    """Saturation vapour pressure of water in Pa, by the Magnus-Tetens approximation.

    Alduchov and Eskridge (1996), *Improved Magnus form approximation of saturation
    vapor pressure*, Journal of Applied Meteorology 35(4), 601-609.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    return 610.94 * np.exp(17.625 * t / (t + 243.04))


def humidity_ratio(temperature_c: np.ndarray | float, relative_humidity_pct: float) -> np.ndarray:
    """Mass of water vapour per mass of dry air, kg/kg."""
    vapour = saturation_vapour_pressure(temperature_c) * relative_humidity_pct / 100.0
    return _MOLAR_RATIO * vapour / (_ATMOSPHERIC_PRESSURE - vapour)


def relative_humidity(temperature_c: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Invert :func:`humidity_ratio` back to a relative humidity in per cent."""
    vapour = _ATMOSPHERIC_PRESSURE * ratio / (_MOLAR_RATIO + ratio)
    return 100.0 * vapour / saturation_vapour_pressure(temperature_c)


class EnvironmentSensorModel:
    """Generate CO2, temperature and humidity series from true occupancy.

    Args:
        config: the environment section of a sensor profile.
        node_area_m2: floor area per node, from ``nodes.csv``.
        generator: source of randomness for the per-sensor calibration offsets.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        node_area_m2: dict[str, float],
        generator: np.random.Generator,
    ) -> None:
        self.config = config
        self.node_area_m2 = dict(node_area_m2)
        nodes = sorted(self.node_area_m2)
        offsets = generator.normal(0.0, config.co2_offset_ppm, size=len(nodes))
        #: One calibration offset per sensor, drawn once and held for the whole record.
        self.co2_offset: dict[str, float] = dict(zip(nodes, map(float, offsets), strict=True))

    def zone_constants(self, node_id: str) -> tuple[float, float]:
        """Air volume in m^3 and outdoor airflow in m^3/s for one node.

        Raises:
            KeyError: if the node has no recorded floor area.
            ValueError: if the area is not positive, which would make the mass balance
                singular rather than merely inaccurate.
        """
        area = self.node_area_m2[node_id]
        if not area > 0.0:
            raise ValueError(
                f"node {node_id!r} has floor area {area}; the CO2 mass balance needs a "
                "positive air volume"
            )
        volume = area * self.config.ceiling_height_m
        airflow = self.config.air_changes_per_hour * volume / 3600.0
        return volume, airflow

    def simulate_node(
        self,
        node_id: str,
        occupancy: np.ndarray,
        interval_seconds: int,
        generator: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Run the mass and energy balances for one node.

        Args:
            node_id: the node whose area and calibration offset to use.
            occupancy: true headcount per interval, length ``T``.
            interval_seconds: sampling interval.
            generator: source of randomness for measurement noise.

        Returns:
            One array per variable in :data:`ENVIRONMENT_VARIABLES`, length ``T``.
        """
        volume, airflow = self.zone_constants(node_id)
        counts = np.nan_to_num(np.asarray(occupancy, dtype=np.float64), nan=0.0)
        n_steps = counts.shape[0]

        # Exact integration of the first-order balance over one interval of constant
        # occupancy. Both the CO2 and the thermal balances share this time constant,
        # because both are governed by the same air exchange.
        decay = float(np.exp(-airflow * interval_seconds / volume))

        generation_m3_per_s = self.config.co2_generation_l_per_s / 1000.0
        co2_steady = self.config.outdoor_co2_ppm + 1e6 * generation_m3_per_s * counts / airflow

        # Conditioning removes heat and moisture but not CO2, which is the asymmetry that
        # makes CO2 the informative channel of the three in a climate-controlled building.
        retained = 1.0 - self.config.hvac_rejection_fraction
        temperature_steady = self.config.baseline_temperature_c + (
            retained * self.config.sensible_heat_w * counts / (_AIR_HEAT_CAPACITY * airflow)
        )
        baseline_ratio = float(
            humidity_ratio(self.config.baseline_temperature_c, self.config.baseline_humidity_pct)
        )
        ratio_steady = baseline_ratio + retained * self.config.latent_heat_w * counts / (
            _LATENT_HEAT_WATER * _AIR_DENSITY * airflow
        )

        co2 = _first_order(co2_steady, decay, self.config.outdoor_co2_ppm)
        temperature = _first_order(
            temperature_steady, decay, self.config.baseline_temperature_c
        )
        ratio = _first_order(ratio_steady, decay, baseline_ratio)

        # Sensor response lag, on top of the transport lag. Short by comparison, but it is
        # the part a different sensor choice would change.
        if self.config.co2_response_seconds > 0.0:
            alpha = 1.0 - float(
                np.exp(-interval_seconds / self.config.co2_response_seconds)
            )
            co2 = _lag(co2, alpha)

        measured_co2 = (
            co2
            + self.co2_offset[node_id]
            + generator.normal(
                0.0,
                self.config.co2_noise_ppm + self.config.co2_noise_fraction * co2,
                size=n_steps,
            )
        )
        measured_temperature = temperature + generator.normal(
            0.0, self.config.temperature_noise_c, size=n_steps
        )
        measured_humidity = relative_humidity(temperature, ratio) + generator.normal(
            0.0, self.config.humidity_noise_pct, size=n_steps
        )

        return {
            "co2_ppm": np.maximum(measured_co2, 0.0),
            "temp_c": measured_temperature,
            "rh_pct": np.clip(measured_humidity, 0.0, 100.0),
        }

    def apply(
        self,
        occupancy: pd.DataFrame,
        interval_seconds: int,
        generator: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate a canonical past-covariate frame from a clean occupancy frame.

        Args:
            occupancy: canonical occupancy with ``timestamp``, ``node_id`` and ``count``.
            interval_seconds: sampling interval.
            generator: source of randomness for measurement noise.

        Returns:
            A frame with ``timestamp``, ``scope``, ``variable`` and ``value``, one scope
            per instrumented node.
        """
        if not self.config.enabled:
            return pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex([], tz="UTC"),
                    "scope": pd.Series([], dtype="object"),
                    "variable": pd.Series([], dtype="object"),
                    "value": pd.Series([], dtype="float64"),
                }
            )

        wide = occupancy.pivot(index="timestamp", columns="node_id", values="count").sort_index()
        frames: list[pd.DataFrame] = []
        for node_id in sorted(str(c) for c in wide.columns):
            series = self.simulate_node(
                node_id, wide[node_id].to_numpy(), interval_seconds, generator
            )
            for variable in ENVIRONMENT_VARIABLES:
                frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": wide.index,
                            "scope": node_id,
                            "variable": variable,
                            "value": series[variable].astype(np.float64),
                        }
                    )
                )
        return pd.concat(frames, ignore_index=True)


def _first_order(steady_state: np.ndarray, decay: float, initial: float) -> np.ndarray:
    """Integrate ``x' = (x_ss - x) / tau`` across a series of piecewise-constant targets."""
    out = np.empty_like(steady_state)
    previous = initial
    for index, target in enumerate(steady_state):
        previous = target + (previous - target) * decay
        out[index] = previous
    return out


def _lag(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average with weight ``alpha`` on the new sample."""
    out = np.empty_like(signal)
    previous = float(signal[0])
    for index, value in enumerate(signal):
        previous += alpha * (value - previous)
        out[index] = previous
    return out


def co2_lag_minutes(
    occupancy: np.ndarray,
    co2: np.ndarray,
    interval_seconds: int,
    max_lag_minutes: float = 120.0,
) -> tuple[float, float]:
    """Measure the lag between occupancy and CO2, and the correlation at that lag.

    The lag is the shift that maximises the cross-correlation, which is the quantity the
    demand-controlled ventilation studies report.

    Args:
        occupancy: true headcount per interval.
        co2: the CO2 series for the same node and window.
        interval_seconds: sampling interval.
        max_lag_minutes: largest shift considered.

    Returns:
        The lag in minutes and the Pearson correlation at that lag.

    Raises:
        ValueError: if either series is constant, in which case no correlation is defined
            and silently returning zero would hide a broken profile.
    """
    x = np.asarray(occupancy, dtype=np.float64)
    y = np.asarray(co2, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"occupancy and co2 must have the same shape, got {x.shape} and {y.shape}")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        raise ValueError("cannot measure a lag against a constant series")

    max_shift = int(max_lag_minutes * 60 / interval_seconds)
    best_lag, best_correlation = 0, -np.inf
    for shift in range(max_shift + 1):
        if shift == 0:
            a, b = x, y
        else:
            a, b = x[:-shift], y[shift:]
        if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
            continue
        correlation = float(np.corrcoef(a, b)[0, 1])
        if correlation > best_correlation:
            best_lag, best_correlation = shift, correlation
    return best_lag * interval_seconds / 60.0, best_correlation
