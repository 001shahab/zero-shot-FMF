"""Minimum trace reconciliation baseline (M5.4).

MinT (Wickramasuriya, Athanasopoulos and Hyndman, 2019, *JASA* 114(526):804-819) is the
standard hierarchical reconciliation method. It is usually written for a summing matrix
``S``, but the same estimator has an equivalent projection form -- see Panagiotelis et
al. (2021), *International Journal of Forecasting* 37(1):343-359 -- which is the one used
here because a building graph has no single aggregation hierarchy::

    y_rec = yhat - W A^T (A W A^T)^{-1} (A yhat - b)

``W`` is an estimate of the base forecast error covariance. Following the MinT(Shrink)
variant, it is the Schafer-Strimmer shrinkage estimator (Schafer and Strimmer, 2005,
*Statistical Applications in Genetics and Molecular Biology* 4(1)): the sample covariance
shrunk towards its own diagonal, with the intensity chosen analytically.

Two differences from the proposed projection are deliberate and are what the E3
comparison is about: MinT uses equality constraints only, so nothing stops it returning a
negative occupancy or a flow beyond a doorway's capacity; and its ``W`` comes from
historical residuals rather than from the model's own predictive spread at this origin.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from mflow.reconcile.constraints import ConstraintSystem, flatten, unflatten
from mflow.reconcile.projection import ReconciliationResult, Reconciler


class MinTReconciler(Reconciler):
    """MinT with a shrinkage covariance, equality constraints only.

    Args:
        shrinkage: fixed shrinkage intensity in ``[0, 1]``, or None to estimate it from
            the residuals with the Schafer-Strimmer formula.
        ols_fallback: use ``W = I`` (that is, OLS reconciliation) until residuals have
            been supplied through :meth:`fit`. Without it the first origin of a run would
            have no covariance to work with.
    """

    name = "mint"

    def __init__(self, *, shrinkage: float | None = None, ols_fallback: bool = True) -> None:
        if shrinkage is not None and not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage}")
        self.shrinkage = shrinkage
        self.ols_fallback = ols_fallback
        self._covariance: np.ndarray | None = None
        self._series_ids: tuple[str, ...] | None = None
        self._estimated_intensity: float | None = None

    # -- covariance estimation ----------------------------------------------- #

    def fit(self, residuals: np.ndarray, series_ids: list[str] | tuple[str, ...]) -> None:
        """Estimate the base error covariance from in-sample one-step residuals.

        Args:
            residuals: ``(n_series, n_observations)`` of ``actual - forecast`` on the
                training window. NaN columns are dropped.
            series_ids: series order of the rows.

        Raises:
            ValueError: if fewer than two usable observations remain.
        """
        if residuals.ndim != 2:
            raise ValueError(f"residuals must be (n_series, n_obs), got {residuals.shape}")
        usable = residuals[:, ~np.isnan(residuals).any(axis=0)]
        if usable.shape[1] < 2:
            raise ValueError(
                f"need at least two complete residual observations to estimate a "
                f"covariance, got {usable.shape[1]}"
            )
        self._covariance, self._estimated_intensity = _shrunk_covariance(
            usable.T, intensity=self.shrinkage
        )
        self._series_ids = tuple(str(s) for s in series_ids)

    @property
    def shrinkage_intensity(self) -> float | None:
        """The intensity actually used, once :meth:`fit` has run."""
        return self._estimated_intensity

    def _weight_matrix(self, system: ConstraintSystem) -> np.ndarray:
        """Expand the per-series covariance to the stacked variable space.

        The covariance is estimated across series at a single step. Lifting it to the
        horizon by a Kronecker product with the identity assumes errors at different
        steps are uncorrelated, which is the assumption MinT is normally applied under
        for multi-step forecasts; it is stated here rather than hidden.
        """
        n_series = len(system.series_ids)
        horizon = system.horizon
        if self._covariance is None:
            if not self.ols_fallback:
                raise RuntimeError(
                    "MinTReconciler.fit has not been called and ols_fallback is off"
                )
            return np.eye(n_series * horizon)
        if self._series_ids != system.series_ids:
            raise ValueError(
                "the covariance was estimated for a different series ordering than the "
                "constraint system uses"
            )
        return np.kron(self._covariance, np.eye(horizon))

    # -- the projection ------------------------------------------------------- #

    def reconcile(
        self,
        forecast: np.ndarray,
        system: ConstraintSystem,
        rhs: np.ndarray,
        *,
        sigma: np.ndarray | None = None,
    ) -> ReconciliationResult:
        """Apply the MinT projection. ``sigma`` is ignored: MinT uses historical residuals."""
        del sigma
        yhat = flatten(forecast)
        weight = self._weight_matrix(system)
        matrix = sp.csr_matrix(system.A)
        b = np.asarray(rhs, dtype=np.float64)

        discrepancy = matrix @ yhat - b
        wat = weight @ matrix.T.toarray()  # (n_vars, n_constraints)
        awat = matrix @ wat  # (n_constraints, n_constraints)
        # A W A^T is symmetric positive definite for a full-rank A and a proper W; a
        # small jitter covers the rank-deficient case that arises when a site has an
        # isolated node pair.
        jitter = 1e-10 * float(np.trace(awat)) / max(awat.shape[0], 1)
        awat = awat + jitter * np.eye(awat.shape[0])
        correction = sla.solve(awat, discrepancy, assume_a="pos")
        y = yhat - wat @ correction

        residual_after = float(np.max(np.abs(system.residual(y, b))))
        return ReconciliationResult(
            reconciled=unflatten(y, len(system.series_ids), system.horizon),
            residual_before=float(np.max(np.abs(system.residual(yhat, b)))),
            residual_after=residual_after,
            violation_before=float(np.sum(system.violation(yhat))),
            # MinT has no bound constraints, so this number is the point of the baseline.
            violation_after=float(np.sum(system.violation(y))),
            adjustment_norm=float(np.linalg.norm(y - yhat)),
            solver_status="closed_form",
            solve_time_ms=0.0,
        )


def _shrunk_covariance(
    observations: np.ndarray, *, intensity: float | None
) -> tuple[np.ndarray, float]:
    """Schafer-Strimmer shrinkage of the sample covariance towards its diagonal.

    Args:
        observations: ``(n_obs, n_series)``.
        intensity: fixed intensity, or None to estimate it.

    Returns:
        ``(covariance, intensity)``.
    """
    n_obs, n_series = observations.shape
    centred = observations - observations.mean(axis=0, keepdims=True)
    sample = centred.T @ centred / (n_obs - 1)
    target = np.diag(np.diag(sample))

    if intensity is None:
        # Variance of each sample covariance entry across observations, per eq. (2) of
        # Schafer and Strimmer; the diagonal is excluded because it is the target.
        products = centred[:, :, None] * centred[:, None, :]
        var_entries = products.var(axis=0, ddof=1) * n_obs / (n_obs - 1) ** 2
        off_diagonal = ~np.eye(n_series, dtype=bool)
        numerator = float(var_entries[off_diagonal].sum())
        denominator = float(np.square(sample[off_diagonal]).sum())
        intensity = 1.0 if denominator <= 0 else float(np.clip(numerator / denominator, 0.0, 1.0))

    covariance = intensity * target + (1.0 - intensity) * sample
    # Keep the estimate usable even when a series is constant on the training window.
    floor = 1e-8 * max(float(np.trace(covariance)) / max(n_series, 1), 1e-8)
    covariance = covariance + floor * np.eye(n_series)
    return covariance, float(intensity)
