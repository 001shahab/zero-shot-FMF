"""Building graph and the incidence operators the reconciliation depends on.

The building is a directed graph whose nodes are spaces (galleries, corridors, stairs,
entrances, exits) plus one virtual ``outside`` node that closes the boundary, and whose
edges are one-way doorway crossings. Two objects are derived from it:

* the **node-edge incidence matrix** ``B``, which turns edge flows into the net gain of
  each interior node and is the algebraic core of the conservation constraint in
  :mod:`mflow.reconcile.constraints`;
* the **adjacency matrix** ``A``, which the spatio-temporal graph baselines in
  :mod:`mflow.forecast.graphnn` consume.

Both are built here and nowhere else, so that every part of the project agrees on node
and edge ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from mflow.schema import FLOW_PREFIX, OCC_PREFIX, OUTSIDE_NODE, SiteData


class GraphError(ValueError):
    """Raised when the building graph is structurally unusable."""


@dataclass(frozen=True)
class BuildingGraph:
    """Directed graph view of a canonical site.

    Attributes:
        node_ids: every node including ``outside``, in ``nodes.csv`` order.
        interior_nodes: nodes excluding ``outside``, in ``nodes.csv`` order. These are
            the nodes that carry an occupancy series and a conservation constraint.
        edge_ids: directed edges in ``edges.csv`` order.
        src: source node of each edge, aligned with ``edge_ids``.
        dst: destination node of each edge, aligned with ``edge_ids``.
        node_capacity: persons, per node.
        edge_capacity_per_min: persons per minute, per edge.
        node_kind: canonical kind, per node.
        node_area_m2: floor area, per node.
    """

    node_ids: tuple[str, ...]
    interior_nodes: tuple[str, ...]
    edge_ids: tuple[str, ...]
    src: tuple[str, ...]
    dst: tuple[str, ...]
    node_capacity: dict[str, float]
    edge_capacity_per_min: dict[str, float]
    node_kind: dict[str, str]
    node_area_m2: dict[str, float]
    outside_node: str

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_site(cls, site: SiteData) -> BuildingGraph:
        """Build the graph from a loaded canonical site."""
        return cls.from_frames(site.nodes, site.edges)

    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> BuildingGraph:
        """Build the graph from ``nodes.csv`` and ``edges.csv`` frames."""
        node_ids = tuple(str(n) for n in nodes["node_id"])
        kinds = {str(n): str(k) for n, k in zip(nodes["node_id"], nodes["kind"], strict=True)}
        outside = [n for n in node_ids if kinds[n] == OUTSIDE_NODE]
        if len(outside) != 1:
            raise GraphError(
                f"expected exactly one {OUTSIDE_NODE!r} node to close the graph, found "
                f"{len(outside)}"
            )
        return cls(
            node_ids=node_ids,
            interior_nodes=tuple(n for n in node_ids if kinds[n] != OUTSIDE_NODE),
            edge_ids=tuple(str(e) for e in edges["edge_id"]),
            src=tuple(str(s) for s in edges["src_node"]),
            dst=tuple(str(d) for d in edges["dst_node"]),
            node_capacity={
                str(n): float(c)
                for n, c in zip(nodes["node_id"], nodes["capacity_persons"], strict=True)
            },
            edge_capacity_per_min={
                str(e): float(c)
                for e, c in zip(
                    edges["edge_id"], edges["capacity_persons_per_min"], strict=True
                )
            },
            node_kind=kinds,
            node_area_m2={
                str(n): float(a) for n, a in zip(nodes["node_id"], nodes["area_m2"], strict=True)
            },
            outside_node=outside[0],
        )

    # -- basic accessors ---------------------------------------------------- #

    @property
    def n_nodes(self) -> int:
        """Number of nodes including ``outside``."""
        return len(self.node_ids)

    @property
    def n_interior(self) -> int:
        """Number of nodes that carry an occupancy series."""
        return len(self.interior_nodes)

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return len(self.edge_ids)

    @cached_property
    def interior_index(self) -> dict[str, int]:
        """Map interior node id to its row in the incidence matrix."""
        return {node: i for i, node in enumerate(self.interior_nodes)}

    @cached_property
    def edge_index(self) -> dict[str, int]:
        """Map edge id to its column in the incidence matrix."""
        return {edge: i for i, edge in enumerate(self.edge_ids)}

    @cached_property
    def in_edges(self) -> dict[str, tuple[str, ...]]:
        """Edges entering each node."""
        out: dict[str, list[str]] = {n: [] for n in self.node_ids}
        for edge, dst in zip(self.edge_ids, self.dst, strict=True):
            out[dst].append(edge)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def out_edges(self) -> dict[str, tuple[str, ...]]:
        """Edges leaving each node."""
        out: dict[str, list[str]] = {n: [] for n in self.node_ids}
        for edge, src in zip(self.edge_ids, self.src, strict=True):
            out[src].append(edge)
        return {k: tuple(v) for k, v in out.items()}

    # -- operators ---------------------------------------------------------- #

    @cached_property
    def incidence(self) -> sp.csr_matrix:
        """Signed node-edge incidence ``B`` of shape ``(n_interior, n_edges)``.

        ``B[v, e]`` is ``+1`` when edge ``e`` enters node ``v``, ``-1`` when it leaves,
        and ``0`` otherwise, so that ``B @ f`` is the net number of persons gained by
        each interior node during one interval. The ``outside`` row is dropped: it is
        the redundant constraint implied by the other rows in a closed graph.
        """
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for j, (edge, s, d) in enumerate(zip(self.edge_ids, self.src, self.dst, strict=True)):
            del edge
            if d in self.interior_index:
                rows.append(self.interior_index[d])
                cols.append(j)
                vals.append(1.0)
            if s in self.interior_index:
                rows.append(self.interior_index[s])
                cols.append(j)
                vals.append(-1.0)
        matrix = sp.coo_matrix(
            (vals, (rows, cols)), shape=(self.n_interior, self.n_edges), dtype=np.float64
        )
        return matrix.tocsr()

    def adjacency(
        self,
        *,
        mode: Literal["binary", "capacity", "distance"] = "binary",
        symmetric: bool = True,
        self_loops: bool = True,
        include_outside: bool = False,
    ) -> np.ndarray:
        """Dense node adjacency for the graph neural network baselines.

        Args:
            mode: ``binary`` gives 1 for a connected pair; ``capacity`` weights the pair
                by edge throughput normalised to its maximum; ``distance`` applies the
                thresholded Gaussian kernel of Li et al. (2018) to the inverse capacity,
                which is the convention the DCRNN baselines were tuned under.
            symmetric: fold the two directions of a doorway into one undirected weight.
            self_loops: put 1 on the diagonal, as the diffusion convolutions expect.
            include_outside: keep the virtual ``outside`` node. The GNN baselines never
                forecast it, so it is dropped by default.

        Returns:
            A ``(n, n)`` float32 array whose node order matches ``interior_nodes``
            (or ``node_ids`` when ``include_outside``).
        """
        nodes = self.node_ids if include_outside else self.interior_nodes
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        weights = np.zeros((n, n), dtype=np.float64)

        for edge, s, d in zip(self.edge_ids, self.src, self.dst, strict=True):
            if s not in index or d not in index:
                continue
            capacity = self.edge_capacity_per_min[edge]
            weights[index[s], index[d]] = max(weights[index[s], index[d]], capacity)

        mask = weights > 0
        if mode == "binary":
            out = mask.astype(np.float64)
        elif mode == "capacity":
            peak = weights.max() if mask.any() else 1.0
            out = weights / peak
        elif mode == "distance":
            # Cost of a doorway is the reciprocal of its capacity; the standard
            # thresholded Gaussian kernel then turns costs into weights.
            costs = np.where(mask, 1.0 / np.maximum(weights, 1e-9), np.inf)
            finite = costs[np.isfinite(costs)]
            sigma = float(finite.std()) if finite.size > 1 else 1.0
            sigma = sigma if sigma > 0 else 1.0
            out = np.exp(-((costs / sigma) ** 2))
            out[~np.isfinite(costs)] = 0.0
        else:  # pragma: no cover - guarded by Literal
            raise GraphError(f"unknown adjacency mode {mode!r}")

        if symmetric:
            out = np.maximum(out, out.T)
        if self_loops:
            np.fill_diagonal(out, 1.0)
        return out.astype(np.float32)

    def series_adjacency(
        self,
        series_ids: Sequence[str],
        *,
        symmetric: bool = True,
        self_loops: bool = True,
    ) -> np.ndarray:
        """Adjacency over *series* rather than over nodes.

        The graph baselines have to forecast the same targets as every other method, and
        that set contains both node occupancy and edge flow. The building graph alone
        cannot index a flow series, so the operator here is the standard augmentation of
        a graph by its line graph: an occupancy series is adjacent to the occupancy
        series of every neighbouring room, and a flow series is adjacent to the occupancy
        series of the two rooms it connects and to the flow series that share those
        rooms. The alternative -- running the baselines on occupancy only -- would make
        them incomparable with the foundation models on the flow half of the table.

        Args:
            series_ids: canonical series order to build the operator for.
            symmetric: fold direction, as the diffusion convolutions expect.
            self_loops: put 1 on the diagonal.

        Returns:
            ``(n_series, n_series)`` float32.
        """
        index = {sid: i for i, sid in enumerate(series_ids)}
        n = len(series_ids)
        weights = np.zeros((n, n), dtype=np.float32)

        def connect(a: str, b: str) -> None:
            if a in index and b in index:
                weights[index[a], index[b]] = 1.0

        for edge, s, d in zip(self.edge_ids, self.src, self.dst, strict=True):
            flow = f"{FLOW_PREFIX}{edge}"
            connect(f"{OCC_PREFIX}{s}", f"{OCC_PREFIX}{d}")
            connect(flow, f"{OCC_PREFIX}{s}")
            connect(f"{OCC_PREFIX}{s}", flow)
            connect(flow, f"{OCC_PREFIX}{d}")
            connect(f"{OCC_PREFIX}{d}", flow)

        for node in self.node_ids:
            touching = [*self.in_edges[node], *self.out_edges[node]]
            for a in touching:
                for b in touching:
                    if a != b:
                        connect(f"{FLOW_PREFIX}{a}", f"{FLOW_PREFIX}{b}")

        if symmetric:
            weights = np.maximum(weights, weights.T)
        if self_loops:
            np.fill_diagonal(weights, 1.0)
        return weights

    def to_networkx(self) -> nx.DiGraph:
        """Return the graph as a NetworkX object for topology queries and plotting."""
        graph = nx.DiGraph()
        for node in self.node_ids:
            graph.add_node(
                node,
                kind=self.node_kind[node],
                capacity=self.node_capacity[node],
                area_m2=self.node_area_m2[node],
            )
        for edge, s, d in zip(self.edge_ids, self.src, self.dst, strict=True):
            graph.add_edge(s, d, edge_id=edge, capacity=self.edge_capacity_per_min[edge])
        return graph

    # -- structural checks --------------------------------------------------- #

    def check_connectivity(self) -> None:
        """Assert that every interior node is reachable from and can reach ``outside``.

        A node that cannot be reached from outside can never be occupied, and a node
        from which outside cannot be reached traps agents; both are configuration bugs
        rather than interesting topologies, so they fail here.

        Raises:
            GraphError: naming the unreachable nodes.
        """
        graph = self.to_networkx()
        forward = nx.descendants(graph, self.outside_node) | {self.outside_node}
        backward = nx.ancestors(graph, self.outside_node) | {self.outside_node}
        unreachable = [n for n in self.interior_nodes if n not in forward]
        trapped = [n for n in self.interior_nodes if n not in backward]
        if unreachable:
            raise GraphError(f"nodes unreachable from {self.outside_node!r}: {unreachable}")
        if trapped:
            raise GraphError(f"nodes from which {self.outside_node!r} is unreachable: {trapped}")

    def net_inflow(self, flows: np.ndarray) -> np.ndarray:
        """Apply the incidence operator to edge flows.

        Args:
            flows: ``(n_edges,)`` or ``(n_edges, k)`` crossings during one interval.

        Returns:
            Net persons gained by each interior node, shaped ``(n_interior,)`` or
            ``(n_interior, k)``.
        """
        if flows.shape[0] != self.n_edges:
            raise GraphError(
                f"expected {self.n_edges} edge flows, got {flows.shape[0]}"
            )
        return np.asarray(self.incidence @ flows)
