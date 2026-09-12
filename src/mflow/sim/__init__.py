"""Museum simulation (M2).

Tier A (:mod:`~mflow.sim.graph_sim`) is a graph-level discrete-event simulator and is
required: it is enough for every forecasting experiment in the study and costs a fraction
of a continuous-space run.

Tier B (:mod:`~mflow.sim.continuous`) wraps JuPedSim and is optional. It exists only to
produce one higher-fidelity site for the density-based crowd safety head, and emits an
event log with the identical schema so that everything downstream is unchanged.
"""

from __future__ import annotations

from mflow.sim.aggregate import aggregate, summarise
from mflow.sim.arrivals import Arrival, sample_day_arrivals
from mflow.sim.config import SiteConfig, load_site_config
from mflow.sim.graph_sim import GraphSimulator, SimulationResult
from mflow.sim.visitors import STYLE_PARAMETERS, StylePolicy

__all__ = [
    "STYLE_PARAMETERS",
    "Arrival",
    "GraphSimulator",
    "SimulationResult",
    "SiteConfig",
    "StylePolicy",
    "aggregate",
    "load_site_config",
    "sample_day_arrivals",
    "summarise",
]
