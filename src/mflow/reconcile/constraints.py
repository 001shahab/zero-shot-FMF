"""The conservation constraint system (M5.1).

For every forecast step ``h`` and every interior node ``v``::

    o_v(t+h) - o_v(t+h-1) - sum_{e in in(v)} f_e(t+h) + sum_{e in out(v)} f_e(t+h) = 0

At ``h = 0`` the previous occupancy ``o_v(t)`` is the last observed value, a known
constant, so it moves to the right-hand side. Stacking over nodes and steps gives
``A y = b`` where ``y`` is the flattened forecast.

Variable layout
---------------
``y`` is the row-major flattening of the ``(n_series, H)`` forecast array in the
canonical series order of :meth:`mflow.schema.SiteData.series_ids`, so the entry for
series ``i`` at step ``h`` sits at index ``i * H + h``. Nothing else in the project is
allowed to invent a different layout: :func:`flatten` and :func:`unflatten` are the only
sanctioned conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.sparse as sp

from mflow.graph import BuildingGraph
from mflow.schema import FLOW_PREFIX, OCC_PREFIX, parse_series_id

#: Occupancy is bounded above by the node capacity; flow by what the doorway can pass in
#: one interval. An unbounded variable would let the projection trade a small residual
#: for a physically absurd value, which is exactly what the bounds are there to prevent.
_UNBOUNDED: Final[float] = np.inf


class ConstraintError(ValueError):
    """Raised when a constraint system cannot be built for the given panel."""


@dataclass(frozen=True)
class ConstraintSystem:
    """``A y = b`` with box bounds, for one site and one horizon.

    Attributes:
        A: sparse ``(n_interior * H, n_series * H)`` conservation operator.
        b: right-hand side; zero except at ``h = 0`` where it carries the last observed
            occupancy. Stored as the ``h = 0`` template so that a new forecast origin only
            needs :meth:`rhs`, not a rebuild.
        lb: lower bounds on ``y``, length ``n_series * H``.
        ub: upper bounds on ``y``.
        series_ids: canonical series order this system was built for.
        horizon: number of forecast steps.
        node_order: interior nodes in constraint-row order.
    """

    A: sp.csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    series_ids: tuple[str, ...]
    horizon: int
    node_order: tuple[str, ...]

    @property
    def n_vars(self) -> int:
        """Length of the stacked forecast vector ``y``."""
        return len(self.series_ids) * self.horizon

    @property
    def n_constraints(self) -> int:
        """Number of equality constraints."""
        return len(self.node_order) * self.horizon

    def rhs(self, last_occupancy: dict[str, float]) -> np.ndarray:
        """Assemble ``b`` for a forecast origin.

        Args:
            last_occupancy: observed occupancy of every interior node at the origin,
                i.e. at ``h = 0`` minus one step.

        Returns:
            The ``(n_constraints,)`` right-hand side.

        Raises:
            ConstraintError: if an interior node has no observation at the origin. An
                unobserved origin makes the ``h = 0`` balance unverifiable, and silently
                substituting zero would manufacture a spurious surge of arrivals.
        """
        missing = [n for n in self.node_order if n not in last_occupancy]
        if missing:
            raise ConstraintError(
                f"no observed occupancy at the forecast origin for nodes {missing[:5]}"
            )
        b = np.zeros(self.n_constraints, dtype=np.float64)
        for i, node in enumerate(self.node_order):
            value = last_occupancy[node]
            if not np.isfinite(value):
                raise ConstraintError(
                    f"occupancy of node {node!r} at the forecast origin is {value}; the "
                    "conservation anchor must be a finite observation"
                )
            b[i * self.horizon] = float(value)
        return b

    def residual(self, y: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return ``A y - b`` for a flattened forecast."""
        return np.asarray(self.A @ y) - b

    def violation(self, y: np.ndarray) -> np.ndarray:
        """Per-variable amount by which ``y`` falls outside its box bounds (>= 0)."""
        return np.maximum(self.lb - y, 0.0) + np.maximum(y - self.ub, 0.0)


def flatten(forecast: np.ndarray) -> np.ndarray:
    """Flatten an ``(n_series, H)`` forecast into the stacked vector ``y``."""
    if forecast.ndim != 2:
        raise ConstraintError(f"expected a 2-D (n_series, H) forecast, got {forecast.shape}")
    return np.ascontiguousarray(forecast, dtype=np.float64).reshape(-1)


def unflatten(y: np.ndarray, n_series: int, horizon: int) -> np.ndarray:
    """Inverse of :func:`flatten`."""
    if y.size != n_series * horizon:
        raise ConstraintError(
            f"cannot reshape {y.size} values into ({n_series}, {horizon})"
        )
    return np.asarray(y, dtype=np.float64).reshape(n_series, horizon)


def build_constraints(
    graph: BuildingGraph,
    series_ids: list[str] | tuple[str, ...],
    horizon: int,
    interval_seconds: int,
    *,
    enforce_bounds: bool = True,
) -> ConstraintSystem:
    """Build the conservation system for a site and horizon.

    Args:
        graph: the building graph.
        series_ids: canonical series order, occupancy series then flow series. Every
            interior node and every edge must appear exactly once; a partial panel
            cannot be reconciled because the missing series are precisely the ones the
            balance needs.
        horizon: number of forecast steps.
        interval_seconds: sampling interval, used to convert edge capacity from persons
            per minute to persons per interval.
        enforce_bounds: when False the box bounds are infinite, which isolates the effect
            of the equality constraints alone.

    Returns:
        The assembled :class:`ConstraintSystem`.

    Raises:
        ConstraintError: if the panel does not carry exactly the expected series.
    """
    if horizon < 1:
        raise ConstraintError(f"horizon must be at least 1, got {horizon}")

    series = tuple(str(s) for s in series_ids)
    position = {sid: i for i, sid in enumerate(series)}
    if len(position) != len(series):
        raise ConstraintError("series_ids contains duplicates")

    expected_occ = {f"{OCC_PREFIX}{n}" for n in graph.interior_nodes}
    expected_flow = {f"{FLOW_PREFIX}{e}" for e in graph.edge_ids}
    present = set(series)
    missing = sorted((expected_occ | expected_flow) - present)
    if missing:
        raise ConstraintError(
            f"cannot build conservation constraints: the panel is missing {len(missing)} "
            f"required series, e.g. {missing[:5]}"
        )
    unexpected = sorted(present - expected_occ - expected_flow)
    if unexpected:
        raise ConstraintError(f"panel carries series unknown to the graph: {unexpected[:5]}")

    n_series = len(series)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def var(series_id: str, step: int) -> int:
        return position[series_id] * horizon + step

    for i, node in enumerate(graph.interior_nodes):
        occ_id = f"{OCC_PREFIX}{node}"
        for h in range(horizon):
            row = i * horizon + h
            rows.append(row)
            cols.append(var(occ_id, h))
            vals.append(1.0)
            if h > 0:
                rows.append(row)
                cols.append(var(occ_id, h - 1))
                vals.append(-1.0)
            for edge in graph.in_edges[node]:
                rows.append(row)
                cols.append(var(f"{FLOW_PREFIX}{edge}", h))
                vals.append(-1.0)
            for edge in graph.out_edges[node]:
                rows.append(row)
                cols.append(var(f"{FLOW_PREFIX}{edge}", h))
                vals.append(1.0)

    matrix = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(graph.n_interior * horizon, n_series * horizon),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()

    lb = np.zeros(n_series * horizon, dtype=np.float64)
    ub = np.full(n_series * horizon, _UNBOUNDED, dtype=np.float64)
    if enforce_bounds:
        minutes = interval_seconds / 60.0
        for sid in series:
            kind, entity = parse_series_id(sid)
            cap = (
                graph.node_capacity[entity]
                if kind == "occupancy"
                else graph.edge_capacity_per_min[entity] * minutes
            )
            start = position[sid] * horizon
            ub[start : start + horizon] = cap

    return ConstraintSystem(
        A=matrix,
        lb=lb,
        ub=ub,
        series_ids=series,
        horizon=horizon,
        node_order=tuple(graph.interior_nodes),
    )


def build_constraints_cached(
    graph: BuildingGraph,
    series_ids: list[str] | tuple[str, ...],
    horizon: int,
    interval_seconds: int,
    *,
    enforce_bounds: bool = True,
    _cache: dict[tuple[object, ...], ConstraintSystem] = {},  # noqa: B006 - module-level cache
) -> ConstraintSystem:
    """Memoised :func:`build_constraints`.

    ``A`` is fixed for a given site and horizon, so a rolling-origin experiment with
    hundreds of origins builds it once. Only the right-hand side changes per origin, and
    that is what :meth:`ConstraintSystem.rhs` is for.
    """
    key = (
        graph.node_ids,
        graph.edge_ids,
        tuple(series_ids),
        horizon,
        interval_seconds,
        enforce_bounds,
    )
    cached = _cache.get(key)
    if cached is None:
        cached = build_constraints(
            graph,
            series_ids,
            horizon,
            interval_seconds,
            enforce_bounds=enforce_bounds,
        )
        _cache[key] = cached
    return cached
