"""Tier A: graph-level discrete-event simulator (M2).

Agents occupy nodes and traverse edges on a fixed tick. Each tick, in this order:

1. agents whose dwell has expired choose a next node and join that doorway's queue;
2. every doorway is served simultaneously, up to its per-tick capacity and the
   destination's remaining headroom, and whoever is not served stays queued;
3. new arrivals are placed at their entrance.

Congestion emerges from step 2 rather than being imposed: a doorway that cannot pass
everyone leaves a queue, a full room ahead throttles the queue, and the blockage
propagates upstream. That is the mechanism the crowd-safety head is supposed to see
coming.

At graph level a doorway crossing is instantaneous: an agent leaves one room and enters
the next within the same tick, and the configured walking time is charged as a minimum
dwell at the destination. Modelling the walk as time spent *on* the edge would leave the
agent in no room at all, and the canonical contract has nowhere to put them -- occupancy
would drift below the true headcount, capacity checks would pass on rooms that are
actually full, and conservation would fail at interval boundaries.

The event log is ``(agent_id, step, from_node, to_node)``, with ``from_node`` equal to the
outside node on entry and ``to_node`` equal to it on exit, so the graph stays closed and
:mod:`mflow.sim.aggregate` can produce a conservation-exact canonical site.

Determinism: everything is drawn from generators derived from the run seed and a stable
stream name, agents are processed in arrival order, and doorways are served in a fixed
order. Two runs with the same seed produce byte-identical output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mflow.manifest import rng
from mflow.sim.arrivals import Arrival, sample_day_arrivals
from mflow.sim.config import SiteConfig, VisitingStyle
from mflow.sim.visitors import StylePolicy, build_policies, sample_styles

#: Hard cap on how long an agent may stay, as a multiple of the opening window. An agent
#: that exceeds it is ejected and the ejection is counted, because an unbounded visit
#: means the exit logic has a bug and silently leaving them in the building would corrupt
#: every occupancy series.
_MAX_VISIT_MULTIPLE: Final[float] = 1.5


@dataclass
class Agent:
    """One visitor inside the simulation."""

    agent_id: int
    style: VisitingStyle
    entered_step: int
    node: str
    previous: str | None = None
    dwell_remaining: int = 0
    visited: set[str] = field(default_factory=set)
    group_id: str | None = None
    cohesion: float = 0.0
    guided: bool = False
    leaving: bool = False
    #: ``dwelling`` while in a room, ``waiting`` while queued at a doorway.
    state: str = "dwelling"


@dataclass
class SimulationResult:
    """Raw simulator output.

    Attributes:
        events: ``agent_id, step, from_node, to_node`` for every traversal.
        dwells: ``agent_id, node, step_in, step_out, style`` for every stay.
        visits: ``agent_id, style, entered_step, exited_step, n_nodes`` per visitor.
        config: the configuration that produced this run.
        start: local midnight of the first simulated day.
        n_steps: total simulation ticks.
        blocked_requests: doorway requests refused for capacity, per edge.
        ejected_agents: agents removed by the maximum-visit guard.
    """

    events: pd.DataFrame
    dwells: pd.DataFrame
    visits: pd.DataFrame
    config: SiteConfig
    start: pd.Timestamp
    n_steps: int
    blocked_requests: dict[str, int]
    ejected_agents: int


class GraphSimulator:
    """Discrete-event simulator over the building graph.

    Args:
        config: the site to simulate.
        seed: run seed. All randomness derives from it.
    """

    def __init__(self, config: SiteConfig, *, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.policies: dict[VisitingStyle, StylePolicy] = build_policies(config)
        self.outside = config.outside_node

        self._interior = [node.id for node in config.nodes if node.id != self.outside]
        self._capacity = {node.id: node.capacity_persons for node in config.nodes}
        self._capacity[self.outside] = 10**9  # the world absorbs everyone

        # Directed adjacency, expanded from the undirected doorway declarations. Edge
        # endpoints are kept in explicit maps rather than encoded in the id, so that a
        # node id containing an underscore cannot corrupt the topology.
        self._directed: dict[tuple[str, str], str] = {}
        self._edge_src: dict[str, str] = {}
        self._edge_dst: dict[str, str] = {}
        self._traversal_steps: dict[str, int] = {}
        self._edge_capacity: dict[str, int] = {}
        self._neighbours: dict[str, list[str]] = {node.id: [] for node in config.nodes}
        for edge in config.edges:
            pairs = [(edge.src, edge.dst, f"{edge.id}_fwd")]
            if edge.bidirectional:
                pairs.append((edge.dst, edge.src, f"{edge.id}_rev"))
            for src, dst, edge_id in pairs:
                self._directed[(src, dst)] = edge_id
                self._edge_src[edge_id] = src
                self._edge_dst[edge_id] = dst
                self._traversal_steps[edge_id] = max(
                    1, round(edge.traversal_seconds / config.step_seconds)
                )
                per_tick = edge.capacity_persons_per_min * config.step_seconds / 60.0
                self._edge_capacity[edge_id] = max(1, round(per_tick))
                self._neighbours[src].append(dst)

        # Fixed iteration order for the doorways, and the in/out edge lists used by the
        # per-tick capacity resolution.
        self._edge_order: list[str] = sorted(self._edge_src)
        self._in_edges: dict[str, list[str]] = {node.id: [] for node in config.nodes}
        self._out_edges: dict[str, list[str]] = {node.id: [] for node in config.nodes}
        for edge_id in self._edge_order:
            self._out_edges[self._edge_src[edge_id]].append(edge_id)
            self._in_edges[self._edge_dst[edge_id]].append(edge_id)

        # Each two-way doorway listed once, as the pair of directed edges that can trade
        # places within a tick.
        self._reverse_pairs: list[tuple[str, str]] = []
        for edge_id in self._edge_order:
            reverse_id = self._directed.get(
                (self._edge_dst[edge_id], self._edge_src[edge_id])
            )
            if reverse_id is not None and edge_id < reverse_id:
                self._reverse_pairs.append((edge_id, reverse_id))

        self._next_hop_to_exit = self._route_to_exit()

    def _route_to_exit(self) -> dict[str, str]:
        """Next hop from every room along a shortest path to the outside.

        A visitor who decides to leave has to walk to a door, and in a real building that
        is a navigation problem, not a teleport. Without this, agents can only leave from
        rooms that happen to touch the outside; everyone else wanders until the entrance
        fills up and the museum stops admitting people.

        Returns:
            Room to next room. Rooms from which the outside is unreachable are absent,
            which :meth:`BuildingGraph.check_connectivity` already rejects at load time.
        """
        distance = {self.outside: 0}
        next_hop: dict[str, str] = {}
        frontier = deque([self.outside])
        # Breadth-first search over reversed edges: whoever can reach node v in one step
        # inherits v as their next hop.
        incoming: dict[str, list[str]] = {node.id: [] for node in self.config.nodes}
        for src, dst in self._directed:
            incoming[dst].append(src)
        while frontier:
            node = frontier.popleft()
            for predecessor in sorted(incoming[node]):
                if predecessor in distance:
                    continue
                distance[predecessor] = distance[node] + 1
                next_hop[predecessor] = node
                frontier.append(predecessor)
        return next_hop

    # -- public ----------------------------------------------------------------- #

    @property
    def directed_edges(self) -> dict[tuple[str, str], str]:
        """Map ``(src, dst)`` to the directed edge id used in the event log."""
        return dict(self._directed)

    def run(self, start: str | pd.Timestamp, days: int) -> SimulationResult:
        """Simulate ``days`` consecutive days from local midnight on ``start``.

        Args:
            start: first day, interpreted in the site's timezone.
            days: number of days.

        Returns:
            The raw event, dwell and visit logs.
        """
        if days < 1:
            raise ValueError(f"days must be at least 1, got {days}")
        first_day = pd.Timestamp(start, tz=self.config.timezone).normalize()
        steps_per_day = 24 * 3600 // self.config.step_seconds
        n_steps = steps_per_day * days

        schedule: dict[int, list[Arrival]] = {}
        styles_by_arrival: list[VisitingStyle] = []
        for day_index in range(days):
            day = first_day + pd.Timedelta(days=day_index)
            generator = rng(self.seed, self.config.site_id, "arrivals", day.isoformat())
            day_arrivals = sample_day_arrivals(self.config, day, generator)
            day_styles = sample_styles(len(day_arrivals), self.config, generator)
            for arrival, style in zip(day_arrivals, day_styles, strict=True):
                absolute = day_index * steps_per_day + arrival.step
                schedule.setdefault(absolute, []).append(
                    Arrival(
                        step=absolute,
                        entrance=arrival.entrance,
                        group_id=arrival.group_id,
                        cohesion=arrival.cohesion,
                        guided=arrival.guided,
                    )
                )
                styles_by_arrival.append(style)

        return self._simulate(schedule, styles_by_arrival, first_day, n_steps)

    # -- the loop ---------------------------------------------------------------- #

    def _simulate(
        self,
        schedule: dict[int, list[Arrival]],
        styles: list[VisitingStyle],
        start: pd.Timestamp,
        n_steps: int,
    ) -> SimulationResult:
        generator = rng(self.seed, self.config.site_id, "movement")
        occupancy = dict.fromkeys(self._capacity, 0)
        agents: dict[int, Agent] = {}
        # Queues are FIFO per directed edge, so a congested doorway serves people in the
        # order they arrived rather than in whatever order the container iterates.
        queues: dict[str, deque[int]] = {e: deque() for e in self._traversal_steps}

        events: list[tuple[int, int, str, str]] = []
        dwells: list[tuple[int, str, int, int, str]] = []
        visits: list[tuple[int, str, int, int, int]] = []
        blocked: dict[str, int] = dict.fromkeys(self._traversal_steps, 0)
        ejected = 0

        dwell_start: dict[int, int] = {}
        next_agent_id = 0
        style_cursor = 0
        max_visit_steps = int(_MAX_VISIT_MULTIPLE * 24 * 3600 / self.config.step_seconds)

        for step in range(n_steps):
            # 1. dwell expiry: choose the next room and join that doorway's queue
            for agent_id in sorted(agents):
                agent = agents[agent_id]
                if agent.state != "dwelling":
                    continue
                if agent.dwell_remaining > 0:
                    agent.dwell_remaining -= 1
                    continue
                if step - agent.entered_step > max_visit_steps:
                    agent.leaving = True
                    ejected += 1
                target = self._choose_next(agent, generator, step)
                edge_id = self._directed.get((agent.node, target))
                if edge_id is None:
                    # The candidate list is built from the adjacency, so this cannot
                    # normally happen. Waiting one tick is a bounded safety net; it never
                    # teleports anyone and never fabricates a crossing.
                    agent.dwell_remaining = 1
                    continue
                agent.state = "waiting"
                queues[edge_id].append(agent_id)

            # 2. serve the doorways, up to edge capacity and destination headroom
            served_per_edge = self._resolve_service(queues, occupancy)
            for edge_id in self._edge_order:
                queue = queues[edge_id]
                source = self._edge_src[edge_id]
                destination = self._edge_dst[edge_id]
                served = 0
                allowed = served_per_edge[edge_id]
                blocked[edge_id] += len(queue) - allowed
                while queue and served < allowed:
                    agent_id = queue.popleft()
                    agent = agents[agent_id]
                    occupancy[source] -= 1
                    dwells.append(
                        (agent_id, source, dwell_start.get(agent_id, step), step, agent.style)
                    )
                    events.append((agent_id, step, source, destination))
                    served += 1

                    if destination == self.outside:
                        visits.append(
                            (
                                agent_id,
                                agent.style,
                                agent.entered_step,
                                step,
                                len(agent.visited),
                            )
                        )
                        del agents[agent_id]
                        continue

                    occupancy[destination] += 1
                    agent.state = "dwelling"
                    agent.previous = source
                    agent.node = destination
                    agent.visited.add(destination)
                    # Walking time is charged to the destination rather than modelled as
                    # time spent on the edge. A person who is halfway through a doorway
                    # has to be somewhere in the canonical occupancy series, and the
                    # alternative -- letting them be nowhere -- makes occupancy exceed
                    # capacity and breaks conservation at the interval boundary.
                    agent.dwell_remaining = self._traversal_steps[edge_id] + self.policies[
                        agent.style
                    ].dwell_steps(destination, generator, self.config.step_seconds)
                    dwell_start[agent_id] = step

            # 3. new arrivals
            for arrival in schedule.get(step, []):
                style = styles[style_cursor] if style_cursor < len(styles) else "butterfly"
                style_cursor += 1
                if occupancy[arrival.entrance] >= self._capacity[arrival.entrance]:
                    # Turned away at a full door. Recorded as never having entered, which
                    # is what a closed entrance means; the arrival is not deferred.
                    continue
                agent_id = next_agent_id
                next_agent_id += 1
                agents[agent_id] = Agent(
                    agent_id=agent_id,
                    style=style,
                    entered_step=step,
                    node=arrival.entrance,
                    previous=self.outside,
                    dwell_remaining=self.policies[style].dwell_steps(
                        arrival.entrance, generator, self.config.step_seconds
                    ),
                    visited={arrival.entrance},
                    group_id=arrival.group_id,
                    cohesion=arrival.cohesion,
                    guided=arrival.guided,
                )
                occupancy[arrival.entrance] += 1
                dwell_start[agent_id] = step
                events.append((agent_id, step, self.outside, arrival.entrance))

        # Visitors still inside when the run ends are recorded as open-ended visits. No
        # closing exit events are emitted: inventing them would put crossings in the last
        # interval that the simulation never produced.
        for agent_id in sorted(agents):
            agent = agents[agent_id]
            dwells.append(
                (agent_id, agent.node, dwell_start.get(agent_id, n_steps), n_steps, agent.style)
            )
            visits.append(
                (agent_id, agent.style, agent.entered_step, n_steps, len(agent.visited))
            )

        return SimulationResult(
            events=pd.DataFrame(
                events, columns=["agent_id", "step", "from_node", "to_node"]
            ).sort_values(["step", "agent_id"], ignore_index=True),
            dwells=pd.DataFrame(
                dwells, columns=["agent_id", "node", "step_in", "step_out", "style"]
            ),
            visits=pd.DataFrame(
                visits,
                columns=["agent_id", "style", "entered_step", "exited_step", "n_nodes"],
            ),
            config=self.config,
            start=start,
            n_steps=n_steps,
            blocked_requests=blocked,
            ejected_agents=ejected,
        )

    # -- doorway service --------------------------------------------------------- #

    def _resolve_service(
        self, queues: dict[str, deque[int]], occupancy: dict[str, int]
    ) -> dict[str, int]:
        """Decide how many people each doorway passes this tick.

        Crossings inside a tick are simultaneous, so a chain of full rooms leading to a
        door clears in one tick rather than one hop per tick. Serving each doorway once,
        in isolation, against the live occupancy deadlocks the building: a full room can
        only empty through a neighbour, and if that neighbour is also full nobody takes
        the first step, so visitors are stranded until the run ends.

        The resolution grants movement rather than rationing it. Each pass walks the
        doorways in a fixed order and lets through whoever fits in the destination as it
        stands, counting the departures already granted this tick. Granting a move frees
        space upstream, so passes repeat until nothing more can be granted. The outside
        always has room, which is what lets a queue rooted at a door pull the whole chain
        behind it forward. Grants only ever increase and are bounded by the queue lengths,
        so the loop terminates, and because a grant is only made against space that
        actually exists, no room can end the tick over capacity.

        What the grant pass cannot do is start a ring of full rooms moving, because no
        link in the ring has space until another link moves first. The common case by far
        is a pair of rooms whose visitors want to trade places, and two people passing
        each other in a doorway need no free space at all -- so a second pass matches
        opposing demand on each doorway and lets both sides through. Longer rings with no
        slack anywhere along them stay put for the tick, which is a genuine standstill
        rather than an artefact of the update order.

        Args:
            queues: per-doorway FIFO queues of waiting agents.
            occupancy: headcount per node at the start of the tick.

        Returns:
            Number of agents each doorway may pass, keyed by directed edge id.
        """
        demand = {
            edge_id: min(len(queues[edge_id]), self._edge_capacity[edge_id])
            for edge_id in self._edge_order
        }
        served = dict.fromkeys(self._edge_order, 0)
        projected = dict(occupancy)

        changed = True
        while changed:
            changed = False
            for edge_id in self._edge_order:
                outstanding = demand[edge_id] - served[edge_id]
                if outstanding <= 0:
                    continue
                destination = self._edge_dst[edge_id]
                headroom = self._capacity[destination] - projected[destination]
                grant = min(outstanding, headroom)
                if grant <= 0:
                    continue
                served[edge_id] += grant
                projected[destination] += grant
                projected[self._edge_src[edge_id]] -= grant
                changed = True

        # Opposing demand on the same doorway trades places; occupancy is unchanged on
        # both sides, so no headroom is consumed.
        for edge_id, reverse_id in self._reverse_pairs:
            exchange = min(
                demand[edge_id] - served[edge_id], demand[reverse_id] - served[reverse_id]
            )
            if exchange > 0:
                served[edge_id] += exchange
                served[reverse_id] += exchange
        return served

    # -- choice ------------------------------------------------------------------ #

    def _choose_next(
        self, agent: Agent, generator: np.random.Generator, step: int
    ) -> str:
        """Pick the room this agent moves to next, possibly the outside."""
        policy = self.policies[agent.style]
        candidates = [n for n in self._neighbours[agent.node] if n != self.outside]
        can_leave = self.outside in self._neighbours[agent.node]

        elapsed_minutes = (step - agent.entered_step) * self.config.step_seconds / 60.0
        visited_fraction = len(agent.visited) / max(len(self._interior), 1)
        if not agent.leaving:
            agent.leaving = bool(
                generator.random()
                < policy.exit_probability(elapsed_minutes, visited_fraction)
            )
        if agent.leaving:
            # The decision to leave is sticky, and from here the agent walks the shortest
            # path to a door rather than continuing to browse.
            if can_leave:
                return self.outside
            hop = self._next_hop_to_exit.get(agent.node)
            if hop is not None:
                return hop
        if not candidates:
            return self.outside if can_leave else agent.node

        # A group member with high cohesion follows the guided route instead of choosing
        # freely, which is what makes a coach party move as a block.
        if agent.guided and agent.cohesion > 0 and generator.random() < agent.cohesion:
            route = self.config.suggested_route
            if agent.node in route:
                position = route.index(agent.node)
                if position + 1 < len(route) and route[position + 1] in candidates:
                    return route[position + 1]

        weights = policy.next_node_weights(
            agent.node, candidates, agent.previous, agent.visited
        )
        probabilities = weights / weights.sum()
        return candidates[int(generator.choice(len(candidates), p=probabilities))]
