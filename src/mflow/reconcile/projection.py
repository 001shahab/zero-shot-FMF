"""Uncertainty-weighted constrained projection (M5.2).

The proposed reconciler solves, for each forecast origin::

    minimise   (y - yhat)^T W (y - yhat)
    subject to A y = b,  lb <= y <= ub

with ``W = diag(1 / sigma_i^2)`` and ``sigma_i = (q90_i - q10_i) / 2.5631``, the spread a
normal distribution would need to produce the model's own 10-90 interval. A series the
model is confident about is therefore held close to its raw value, and the correction is
pushed onto the series the model is unsure of. The ``uniform_weight`` ablation replaces
``W`` with the identity and isolates whether that weighting is doing any work.

The problem is a convex QP with a fixed constraint matrix. It is expressed once as a
CVXPY problem with parameters, so the canonicalisation and OSQP's symbolic factorisation
of ``A`` are reused across every origin of a rolling-origin experiment; only the
parameter values change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from mflow.reconcile.constraints import ConstraintSystem, flatten, unflatten

#: z(0.9) - z(0.1) = 2.5631, the width of the central 80% interval of a standard normal.
#: Dividing the model's own 10-90 spread by it converts that spread into the sigma a
#: normal distribution would need, which is what the quadratic weight wants.
NORMAL_80_SPREAD: Final[float] = 2.5631

#: Floor on sigma. A series predicted with literally zero spread -- a shut gallery, or a
#: quantile head that collapsed -- would otherwise carry infinite weight and freeze the
#: whole projection. The floor is in persons and deliberately small.
DEFAULT_SIGMA_FLOOR: Final[float] = 1e-2

#: Ridge added to the diagonal of W, keeping the QP strongly convex when every weight is
#: tiny. Expressed relative to the mean weight so that it does not depend on the units.
DEFAULT_RIDGE: Final[float] = 1e-6

WeightMode = Literal["uncertainty", "uniform"]


class ReconciliationError(RuntimeError):
    """Raised when the projection cannot be solved to the requested tolerance."""


def _polish_option() -> dict[str, bool]:
    """Return the solution-polishing flag under the name the installed OSQP expects."""
    from importlib import metadata

    try:
        version = metadata.version("osqp")
    except metadata.PackageNotFoundError:  # pragma: no cover - OSQP is a hard dependency
        return {}
    return {"polishing": True} if int(version.split(".")[0]) >= 1 else {"polish": True}


@dataclass(frozen=True)
class ReconciliationResult:
    """Outcome of one reconciliation call.

    Attributes:
        reconciled: ``(n_series, H)`` coherent forecast.
        residual_before: max absolute conservation residual of the raw forecast.
        residual_after: max absolute conservation residual after projection.
        violation_before: total bound violation of the raw forecast, in persons.
        violation_after: total bound violation after projection.
        adjustment_norm: Euclidean norm of the correction applied.
        solver_status: status string reported by the solver.
        solve_time_ms: time spent inside the solver.
    """

    reconciled: np.ndarray
    residual_before: float
    residual_after: float
    violation_before: float
    violation_after: float
    adjustment_norm: float
    solver_status: str
    solve_time_ms: float

    def as_dict(self) -> dict[str, float | str]:
        """Diagnostics for the metrics table (everything except the array itself)."""
        return {
            "residual_before": self.residual_before,
            "residual_after": self.residual_after,
            "violation_before": self.violation_before,
            "violation_after": self.violation_after,
            "adjustment_norm": self.adjustment_norm,
            "solver_status": self.solver_status,
            "solve_time_ms": self.solve_time_ms,
        }


class Reconciler(ABC):
    """Interface shared by every reconciliation strategy, including the no-op."""

    name: str

    @abstractmethod
    def reconcile(
        self,
        forecast: np.ndarray,
        system: ConstraintSystem,
        rhs: np.ndarray,
        *,
        sigma: np.ndarray | None = None,
    ) -> ReconciliationResult:
        """Project a raw forecast onto the coherent, feasible set.

        Args:
            forecast: ``(n_series, H)`` median forecast in canonical series order.
            system: the constraint system for this site and horizon.
            rhs: right-hand side from :meth:`ConstraintSystem.rhs`.
            sigma: ``(n_series, H)`` predictive spread, required by weighted strategies.

        Returns:
            The reconciled forecast together with before/after diagnostics.
        """


def sigma_from_quantiles(
    quantile_forecast: np.ndarray,
    quantile_levels: list[float] | tuple[float, ...],
    *,
    floor: float = DEFAULT_SIGMA_FLOOR,
) -> np.ndarray:
    """Normal-equivalent spread from the 10th and 90th predictive quantiles.

    Args:
        quantile_forecast: ``(n_series, H, n_quantiles)``.
        quantile_levels: the levels, ascending, matching the last axis.
        floor: minimum sigma, in persons.

    Returns:
        ``(n_series, H)`` positive spreads.

    Raises:
        ValueError: if the 0.1 and 0.9 levels are not both present.
    """
    levels = [round(float(q), 6) for q in quantile_levels]
    try:
        lo = levels.index(0.1)
        hi = levels.index(0.9)
    except ValueError as exc:
        raise ValueError(
            f"the uncertainty weighting needs the 0.1 and 0.9 quantiles; got {levels}"
        ) from exc
    spread = quantile_forecast[..., hi] - quantile_forecast[..., lo]
    return np.maximum(np.asarray(spread, dtype=np.float64) / NORMAL_80_SPREAD, floor)


class IdentityReconciler(Reconciler):
    """The ``none`` baseline: return the forecast untouched, but still measure it."""

    name = "none"

    def reconcile(
        self,
        forecast: np.ndarray,
        system: ConstraintSystem,
        rhs: np.ndarray,
        *,
        sigma: np.ndarray | None = None,
    ) -> ReconciliationResult:
        del sigma
        y = flatten(forecast)
        residual = float(np.max(np.abs(system.residual(y, rhs)))) if system.n_constraints else 0.0
        violation = float(np.sum(system.violation(y)))
        return ReconciliationResult(
            reconciled=np.asarray(forecast, dtype=np.float64).copy(),
            residual_before=residual,
            residual_after=residual,
            violation_before=violation,
            violation_after=violation,
            adjustment_norm=0.0,
            solver_status="not_applicable",
            solve_time_ms=0.0,
        )


@dataclass
class _CompiledProblem:
    """A parameterised CVXPY problem retained across forecast origins."""

    problem: cp.Problem
    y: cp.Variable
    w_sqrt: cp.Parameter
    z: cp.Parameter
    b: cp.Parameter


class ProjectionReconciler(Reconciler):
    """The proposed reconciler: an uncertainty-weighted, bound-constrained projection.

    Args:
        weighting: ``uncertainty`` uses ``W = diag(1 / sigma^2)``; ``uniform`` uses
            ``W = I`` and is the ablation that isolates the weighting.
        enforce_bounds: honour the capacity box. Turning it off leaves a pure equality
            projection and is only useful for diagnosing solver behaviour.
        ridge: relative ridge added to the weights to keep the QP strongly convex.
        sigma_floor: minimum predictive spread, in persons.
        residual_tolerance: post-solve check on ``max |A y - b|``. Exceeding it raises,
            because a silently incoherent "reconciled" forecast would invalidate every
            claim the reconciliation table makes.
        solver: CVXPY solver name; OSQP is the intended one.
        solver_options: forwarded to the solver.
    """

    name = "proposed"

    def __init__(
        self,
        *,
        weighting: WeightMode = "uncertainty",
        enforce_bounds: bool = True,
        ridge: float = DEFAULT_RIDGE,
        sigma_floor: float = DEFAULT_SIGMA_FLOOR,
        residual_tolerance: float = 1e-4,
        solver: str = "OSQP",
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        if weighting not in ("uncertainty", "uniform"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.weighting: WeightMode = weighting
        self.enforce_bounds = enforce_bounds
        self.ridge = ridge
        self.sigma_floor = sigma_floor
        self.residual_tolerance = residual_tolerance
        self.solver = solver
        self.solver_options = {
            "eps_abs": 1e-7,
            "eps_rel": 1e-7,
            "max_iter": 50_000,
            # OSQP renamed "polish" to "polishing" in 1.0; solution polishing is what
            # brings the equality residual down to solver tolerance rather than merely
            # to the ADMM convergence threshold, so it is on deliberately.
            **_polish_option(),
            **(solver_options or {}),
        }
        self.name = "proposed" if weighting == "uncertainty" else "uniform_weight"
        self._compiled: dict[tuple[int, int, bool], _CompiledProblem] = {}

    # -- problem construction ------------------------------------------------ #

    def _compile(self, system: ConstraintSystem) -> _CompiledProblem:
        """Build (or fetch) the parameterised QP for this constraint system.

        The problem is written so that every parameter enters affinely -- ``w_sqrt``
        multiplies the variable and ``z`` is precomputed outside -- which keeps it
        DPP-compliant and lets CVXPY reuse the compiled form instead of recanonicalising
        at every origin.
        """
        key = (system.n_vars, system.n_constraints, self.enforce_bounds)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached

        n = system.n_vars
        y = cp.Variable(n, name="y")
        w_sqrt = cp.Parameter(n, name="w_sqrt", nonneg=True)
        z = cp.Parameter(n, name="z")
        b = cp.Parameter(system.n_constraints, name="b")

        constraints: list[cp.Constraint] = [sp.csr_matrix(system.A) @ y == b]
        if self.enforce_bounds:
            finite_ub = np.where(np.isfinite(system.ub), system.ub, 1e12)
            constraints.append(y >= system.lb)
            constraints.append(y <= finite_ub)

        objective = cp.Minimize(cp.sum_squares(cp.multiply(w_sqrt, y) - z))
        compiled = _CompiledProblem(
            problem=cp.Problem(objective, constraints), y=y, w_sqrt=w_sqrt, z=z, b=b
        )
        self._compiled[key] = compiled
        return compiled

    def _weights(self, sigma: np.ndarray | None, n_vars: int) -> np.ndarray:
        if self.weighting == "uniform":
            return np.ones(n_vars, dtype=np.float64)
        if sigma is None:
            raise ReconciliationError(
                "the uncertainty-weighted projection needs a predictive spread; pass "
                "sigma=sigma_from_quantiles(...) or use weighting='uniform'"
            )
        spread = np.maximum(flatten(sigma), self.sigma_floor)
        weights = 1.0 / np.square(spread)
        return weights + self.ridge * float(np.mean(weights))

    # -- the projection ------------------------------------------------------ #

    def reconcile(
        self,
        forecast: np.ndarray,
        system: ConstraintSystem,
        rhs: np.ndarray,
        *,
        sigma: np.ndarray | None = None,
    ) -> ReconciliationResult:
        """Solve the constrained QP for one forecast origin."""
        yhat = flatten(forecast)
        if yhat.size != system.n_vars:
            raise ReconciliationError(
                f"forecast has {yhat.size} values but the constraint system expects "
                f"{system.n_vars}; the panel and the graph disagree"
            )
        if not np.isfinite(yhat).all():
            raise ReconciliationError("the forecast contains non-finite values")

        weights = self._weights(sigma, system.n_vars)
        compiled = self._compile(system)
        w_sqrt = np.sqrt(weights)
        compiled.w_sqrt.value = w_sqrt
        compiled.z.value = w_sqrt * yhat
        compiled.b.value = np.asarray(rhs, dtype=np.float64)

        compiled.problem.solve(solver=self.solver, warm_start=True, **self.solver_options)
        status = str(compiled.problem.status)
        if status not in ("optimal", "optimal_inaccurate"):
            raise ReconciliationError(
                f"the projection did not solve: status {status!r}. The constraint set is "
                "infeasible or the solver tolerance is too tight; do not fall back to the "
                "raw forecast silently."
            )
        solution = compiled.y.value
        if solution is None:  # pragma: no cover - defensive, cvxpy sets it on success
            raise ReconciliationError(f"solver reported {status!r} but returned no solution")

        y = np.asarray(solution, dtype=np.float64)
        if self.enforce_bounds:
            # OSQP satisfies the box to its own tolerance; clipping removes the last
            # 1e-9 of slack so that the reported violation rate is exactly zero rather
            # than "zero up to solver noise".
            y = np.clip(y, system.lb, system.ub)

        residual_after = float(np.max(np.abs(system.residual(y, rhs))))
        if residual_after > self.residual_tolerance:
            raise ReconciliationError(
                f"reconciled forecast still violates conservation by {residual_after:.3e}, "
                f"above the tolerance {self.residual_tolerance:.1e} (solver status {status})"
            )

        solve_time = compiled.problem.solver_stats.solve_time
        return ReconciliationResult(
            reconciled=unflatten(y, len(system.series_ids), system.horizon),
            residual_before=float(np.max(np.abs(system.residual(yhat, rhs)))),
            residual_after=residual_after,
            violation_before=float(np.sum(system.violation(yhat))),
            violation_after=float(np.sum(system.violation(y))),
            adjustment_norm=float(np.linalg.norm(y - yhat)),
            solver_status=status,
            solve_time_ms=float(solve_time or 0.0) * 1000.0,
        )


@dataclass
class ReconcilerRegistry:
    """Named reconcilers, so experiment configs can refer to them as strings."""

    factories: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, factory: Any) -> None:
        """Add a factory under ``name``."""
        if name in self.factories:
            raise ValueError(f"reconciler {name!r} is already registered")
        self.factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Reconciler:
        """Instantiate the reconciler registered under ``name``."""
        if name not in self.factories:
            raise KeyError(f"unknown reconciler {name!r}; known: {sorted(self.factories)}")
        return self.factories[name](**kwargs)  # type: ignore[no-any-return]


def default_registry() -> ReconcilerRegistry:
    """The four strategies compared in experiment E3."""
    from mflow.reconcile.mint import MinTReconciler

    registry = ReconcilerRegistry()
    registry.register("none", IdentityReconciler)
    registry.register("mint", MinTReconciler)
    registry.register(
        "uniform_weight", lambda **kw: ProjectionReconciler(weighting="uniform", **kw)
    )
    registry.register(
        "proposed", lambda **kw: ProjectionReconciler(weighting="uncertainty", **kw)
    )
    return registry
