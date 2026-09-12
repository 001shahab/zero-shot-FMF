"""mflow: zero-shot foundation model forecasting of museum visitor flow.

The package is organised around a single canonical data contract (:mod:`mflow.schema`).
Every dataset -- simulated or real -- is converted into that contract by an adapter, and
no model code ever reads a raw dataset.

Sub-packages
------------
``adapters``  external dataset readers that emit canonical sites
``sim``       Tier A discrete-event graph simulator and Tier B continuous wrapper
``sensors``   degradation models turning clean series into observed series
``forecast``  the :class:`~mflow.forecast.base.Forecaster` interface and implementations
``reconcile`` topology-constrained reconciliation of incoherent forecasts
``risk``      congestion, exposure and anomaly decision heads
``eval``      rolling-origin protocol, metrics, significance tests and reporting
"""

from __future__ import annotations

from mflow.schema import (
    Panel,
    SiteData,
    SiteMeta,
    SiteValidationError,
    load_site,
    validate_site,
    write_site,
)

__version__ = "0.1.0"

__all__ = [
    "Panel",
    "SiteData",
    "SiteMeta",
    "SiteValidationError",
    "__version__",
    "load_site",
    "validate_site",
    "write_site",
]
