"""Doorway counter error model (M3).

Every parameter default in ``configs/sensors/*.yaml`` traces to a measured error rate,
not to a guess. The three effects modelled here are the ones the counting literature
consistently reports:

**Misses that grow with traffic.** Occlusion is the dominant failure mode of doorway
counters, and the error rate is a function of how many people cross together rather than
a constant. Cokbas et al. (2020), *Low-Resolution Overhead Thermal Tripwire for Occupancy
Estimation*, CVPR Workshops, report that "typically 80-90% entry and exit events are
correctly classified" across challenging scenarios, "while in simpler, less-active
scenarios even 100% correct classification can be reached", with their high-activity
sequence the worst case. Gruber et al. (2014), *Evaluation of Visitor Counting
Technologies and Their Energy Saving Potential through Demand-Controlled Ventilation*,
Energies 7(3), 1685-1705, measured a stereoscopic overhead camera over 36 days of free
people flow at directional errors of 3.3% and 7.2%, and their best light-beam sensor at
4.6% and 5.2%. Gade et al. (2016), *Pedestrian Counting with Occlusion Handling Using
Stereo Thermal Cameras*, Sensors 16(1), 62, report success rates of 95.4% and 99.1% under
"moderate density of pedestrians and heavy occlusions".

Together these bracket a quiet-doorway miss rate of a few per cent rising towards 15-20%
when a group crosses at once, which is what ``base_miss_probability`` and
``congestion_miss_probability`` encode.

**Double counts.** A person who lingers in the detection zone, or a coat carried at body
temperature, can register twice. This is a smaller effect than occlusion and enters as a
flat per-crossing probability.

**Fixed per-sensor bias.** The directional errors in Gruber et al. are not symmetric --
the same unit under-counts one direction and over-counts the other -- so each sensor gets
a multiplicative bias drawn once and held for the whole record. This is the component
that a forecaster cannot average away, and the one that breaks conservation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mflow.sensors.config import CountingConfig


class CountingSensorModel:
    """Turn true edge crossings into counter readings.

    Args:
        config: the counting section of a sensor profile.
        edge_ids: every edge that carries a counter, in canonical order.
        generator: source of randomness. The caller owns the seed.
    """

    def __init__(
        self,
        config: CountingConfig,
        edge_ids: list[str],
        generator: np.random.Generator,
    ) -> None:
        self.config = config
        self.edge_ids = list(edge_ids)
        # One bias per sensor, drawn once and held. Centred on 1.0 and clipped away from
        # zero so that a sensor can under- or over-count but never invert.
        draws = generator.normal(1.0, config.bias_sigma, size=len(self.edge_ids))
        self.bias: dict[str, float] = {
            edge_id: float(np.clip(value, 0.5, 1.5))
            for edge_id, value in zip(self.edge_ids, draws, strict=True)
        }

    def apply(self, flow: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
        """Return observed crossing counts for a clean ``flow`` frame.

        Args:
            flow: canonical flow frame with ``timestamp``, ``edge_id`` and ``count``.
            generator: source of randomness for the per-crossing draws.

        Returns:
            A new frame with the same index and an observed ``count`` column. Counts stay
            non-negative integers; no row is dropped, because a counter that is working
            reports zero rather than nothing.
        """
        observed = flow.copy()
        counts = observed["count"].to_numpy(dtype=np.float64)
        edges = observed["edge_id"].to_numpy()

        # Miss probability rises with the instantaneous crossing rate and saturates, so
        # that a very busy doorway does not end up with a probability above one.
        saturation = np.minimum(counts / self.config.congestion_reference, 1.0)
        miss_probability = (
            self.config.base_miss_probability
            + self.config.congestion_miss_probability * saturation
        )

        true_counts = np.nan_to_num(counts, nan=0.0).astype(np.int64)
        missed = generator.binomial(true_counts, miss_probability)
        doubled = generator.binomial(
            true_counts - missed, self.config.double_count_probability
        )
        bias = np.array([self.bias[str(e)] for e in edges], dtype=np.float64)

        reported = (true_counts - missed + doubled) * bias
        result = np.rint(reported)
        # A NaN in the clean series means the truth is unknown, and inventing a reading
        # for it would be fabricating data.
        result[np.isnan(counts)] = np.nan
        observed["count"] = np.maximum(result, 0.0)
        return observed


def counting_error_summary(clean: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Per-edge relative counting error, for checking a profile against the literature.

    Args:
        clean: the true flow frame.
        observed: the same frame after :meth:`CountingSensorModel.apply`.

    Returns:
        One row per edge with the true total, the observed total and the signed relative
        error, which is the quantity the cited counter evaluations report.
    """
    merged = clean.merge(
        observed, on=["timestamp", "edge_id"], suffixes=("_true", "_observed")
    )
    totals = merged.groupby("edge_id")[["count_true", "count_observed"]].sum()
    totals["relative_error"] = (
        totals["count_observed"] - totals["count_true"]
    ) / totals["count_true"].replace(0.0, np.nan)
    return totals.reset_index()
