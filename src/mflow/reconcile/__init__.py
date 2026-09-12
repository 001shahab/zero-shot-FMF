"""Topology-constrained reconciliation of incoherent forecasts (M5).

A forecaster produces each series independently, so nothing makes its occupancy and flow
predictions agree with the physics of the building: people who leave a room must appear
in the next one. This package turns that physics into a linear constraint system and
projects the raw forecast onto the feasible set.

* :mod:`~mflow.reconcile.constraints` builds ``A y = b`` together with the box bounds.
* :mod:`~mflow.reconcile.projection` solves the uncertainty-weighted constrained QP that
  the paper proposes, plus the ``uniform_weight`` ablation.
* :mod:`~mflow.reconcile.mint` is the minimum-trace baseline, equality constraints only.
* :mod:`~mflow.reconcile.quantiles` lifts a point reconciler to the full quantile fan.
"""

from __future__ import annotations

from mflow.reconcile.constraints import (
    ConstraintError,
    ConstraintSystem,
    build_constraints,
    build_constraints_cached,
    flatten,
    unflatten,
)
from mflow.reconcile.mint import MinTReconciler
from mflow.reconcile.projection import (
    IdentityReconciler,
    ProjectionReconciler,
    Reconciler,
    ReconcilerRegistry,
    ReconciliationResult,
    default_registry,
    sigma_from_quantiles,
)
from mflow.reconcile.quantiles import QuantileReconciler, crossing_rate, enforce_monotone

__all__ = [
    "ConstraintError",
    "ConstraintSystem",
    "IdentityReconciler",
    "MinTReconciler",
    "ProjectionReconciler",
    "QuantileReconciler",
    "Reconciler",
    "ReconcilerRegistry",
    "ReconciliationResult",
    "build_constraints",
    "build_constraints_cached",
    "crossing_rate",
    "default_registry",
    "enforce_monotone",
    "flatten",
    "sigma_from_quantiles",
    "unflatten",
]
