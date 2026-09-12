"""Site configuration schema for the simulator.

A site YAML in ``configs/sites/`` fully determines a simulated museum: its rooms and
doorways, the exhibits people come to see, the suggested route, the arrival profile, the
mix of visiting styles and the schedule of groups and guided tours. Nothing about a
simulated site is hard-coded in Python, so a new building is a new file.

The models are validated with pydantic so that a typo in a YAML key fails at load time
with the key name, rather than silently defaulting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mflow.schema import NODE_KINDS, OUTSIDE_NODE

VisitingStyle = Literal["ant", "fish", "butterfly", "grasshopper"]
VISITING_STYLES: tuple[VisitingStyle, ...] = ("ant", "fish", "butterfly", "grasshopper")


class NodeConfig(BaseModel):
    """One space in the building."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: str
    area_m2: float = Field(ge=0.0)
    capacity_persons: int = Field(ge=0)
    #: Relative pull of the exhibits in this room, driving dwell time and route choice.
    interest: float = Field(default=1.0, ge=0.0)
    #: Ceiling height, used by the CO2 mass-balance model in :mod:`mflow.sensors`.
    ceiling_height_m: float = Field(default=4.0, gt=0.0)

    @model_validator(mode="after")
    def _known_kind(self) -> NodeConfig:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"node {self.id!r} has unknown kind {self.kind!r}")
        return self


class EdgeConfig(BaseModel):
    """One doorway, declared undirected and expanded into a directed pair."""

    model_config = ConfigDict(extra="forbid")

    id: str
    src: str
    dst: str
    width_m: float = Field(gt=0.0)
    capacity_persons_per_min: float = Field(gt=0.0)
    #: Seconds to walk the connection, rounded to whole simulation steps.
    traversal_seconds: float = Field(default=10.0, ge=0.0)
    #: False makes the doorway one-way (a fire exit, a turnstile).
    bidirectional: bool = True


class ArrivalConfig(BaseModel):
    """Non-homogeneous Poisson arrivals plus scheduled group injections."""

    model_config = ConfigDict(extra="forbid")

    open_time: str = "09:00"
    close_time: str = "18:00"
    #: Peak arrival intensity, persons per minute.
    peak_rate_per_min: float = Field(gt=0.0)
    #: Hour-of-day multipliers on the peak rate, one per opening hour.
    hourly_profile: list[float] = Field(min_length=1)
    #: Multiplier per weekday, Monday first.
    weekday_multiplier: list[float] = Field(default_factory=lambda: [1.0] * 7, min_length=7)
    #: Entrance nodes arrivals are distributed over, with weights.
    entrances: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _positive_profile(self) -> ArrivalConfig:
        if any(value < 0 for value in self.hourly_profile):
            raise ValueError("hourly_profile entries must be non-negative")
        if any(value < 0 for value in self.weekday_multiplier):
            raise ValueError("weekday_multiplier entries must be non-negative")
        return self


class GroupConfig(BaseModel):
    """A scheduled coach party, school group or guided tour."""

    model_config = ConfigDict(extra="forbid")

    id: str
    #: ``HH:MM`` local time of arrival.
    time: str
    #: Mean and standard deviation of the group size.
    size_mean: float = Field(gt=0.0)
    size_sd: float = Field(default=2.0, ge=0.0)
    #: Probability that a member follows the group rather than moving independently.
    cohesion: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Days of the week the group arrives on, Monday first. Empty means every day.
    weekdays: list[int] = Field(default_factory=list)
    #: A guided tour follows the suggested route exactly and is exposed as the
    #: ``tour_departure`` known-future covariate.
    guided: bool = False
    entrance: str | None = None


class StyleMix(BaseModel):
    """Proportions of the four visiting styles.

    The typology is Veron and Levasseur's (1983) ant, fish, butterfly and grasshopper,
    as operationalised for tracking studies by Zancanaro, Kuflik, Boger, Goren-Bar and
    Goldwasser (2007). Proportions must sum to one.
    """

    model_config = ConfigDict(extra="forbid")

    ant: float = Field(ge=0.0)
    fish: float = Field(ge=0.0)
    butterfly: float = Field(ge=0.0)
    grasshopper: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> StyleMix:
        total = self.ant + self.fish + self.butterfly + self.grasshopper
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"style mix must sum to 1, got {total:.6f}")
        return self

    def as_dict(self) -> dict[str, float]:
        """Mapping from style name to proportion."""
        return {
            "ant": self.ant,
            "fish": self.fish,
            "butterfly": self.butterfly,
            "grasshopper": self.grasshopper,
        }


class TargetRanges(BaseModel):
    """Acceptance ranges the simulator output is checked against (M2).

    A simulator that produces plausible-looking series but an implausible mean visit
    duration is not a useful stand-in for a museum, so these are asserted rather than
    eyeballed.
    """

    model_config = ConfigDict(extra="forbid")

    mean_visit_minutes: tuple[float, float]
    node_visit_fraction: tuple[float, float]
    occupancy_peak_to_mean: tuple[float, float]


class SiteConfig(BaseModel):
    """A complete simulated site."""

    model_config = ConfigDict(extra="forbid")

    site_id: str
    timezone: str = "Europe/Rome"
    interval_seconds: int = Field(default=60, gt=0)
    #: Simulation tick. Must divide ``interval_seconds``; a finer tick resolves queueing
    #: at doorways but costs proportionally more time.
    step_seconds: int = Field(default=10, gt=0)
    nodes: list[NodeConfig] = Field(min_length=2)
    edges: list[EdgeConfig] = Field(min_length=1)
    suggested_route: list[str] = Field(default_factory=list)
    arrivals: ArrivalConfig
    style_mix: StyleMix
    groups: list[GroupConfig] = Field(default_factory=list)
    #: Rate at which the exit probability grows with elapsed visit time, per minute.
    fatigue_per_minute: float = Field(default=0.004, ge=0.0)
    #: Dwell time is log-normal; this is the geometric mean at interest 1.0, in minutes.
    base_dwell_minutes: float = Field(default=1.5, gt=0.0)
    targets: TargetRanges | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SiteConfig:
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids in the site configuration")
        outside = [node.id for node in self.nodes if node.kind == OUTSIDE_NODE]
        if len(outside) != 1:
            raise ValueError(
                f"the site must declare exactly one node of kind {OUTSIDE_NODE!r}, found "
                f"{len(outside)}"
            )
        known = set(ids)
        for edge in self.edges:
            for endpoint in (edge.src, edge.dst):
                if endpoint not in known:
                    raise ValueError(f"edge {edge.id!r} references unknown node {endpoint!r}")
        for node_id in self.suggested_route:
            if node_id not in known:
                raise ValueError(f"suggested_route references unknown node {node_id!r}")
        for node_id in self.arrivals.entrances:
            if node_id not in known:
                raise ValueError(f"arrivals.entrances references unknown node {node_id!r}")
        if self.interval_seconds % self.step_seconds != 0:
            raise ValueError(
                f"step_seconds ({self.step_seconds}) must divide interval_seconds "
                f"({self.interval_seconds}) so that aggregation lands on interval boundaries"
            )
        group_ids = [group.id for group in self.groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("duplicate group ids in the site configuration")
        return self

    @property
    def outside_node(self) -> str:
        """Id of the virtual node that closes the graph."""
        return next(node.id for node in self.nodes if node.kind == OUTSIDE_NODE)

    @property
    def steps_per_interval(self) -> int:
        """Simulation ticks per canonical sampling interval."""
        return self.interval_seconds // self.step_seconds

    def node(self, node_id: str) -> NodeConfig:
        """Look up a node by id."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"unknown node {node_id!r}")


def load_site_config(path: str | Path) -> SiteConfig:
    """Load and validate a site YAML.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: naming the offending key.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"site configuration not found: {target}")
    payload: dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8"))
    return SiteConfig.model_validate(payload)
