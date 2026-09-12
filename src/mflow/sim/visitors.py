"""Visiting-style behaviour models.

Four styles from the museum studies typology of Veron and Levasseur (1983), as
operationalised for tracking data by Zancanaro, Kuflik, Boger, Goren-Bar and Goldwasser
(2007), *Analyzing museum visitors' behavior patterns*, UM 2007, LNCS 4511:238-246.

Each style is a policy over the choice of next room given the current one, plus a dwell
distribution. Dwell is log-normal, which is the distribution repeatedly fitted to
observed museum dwell times (see Yoshimura et al., 2014, *An analysis of visitors'
behaviour in the Louvre Museum*, Environment and Planning B 41(6):1113-1131); its
geometric mean scales with the room's interest weight and with a per-style multiplier.

The styles differ along three axes and nothing else, so the effect of the mix on the
aggregate series is interpretable:

===========  ==========  ============  =========================================
style        dwell x     route pull    next-room rule
===========  ==========  ============  =========================================
ant          1.8         strong        follows the suggested route, visits most rooms
fish         0.5         weak          crosses open space, few stops, prefers large rooms
butterfly    1.0         moderate      frequent direction changes and revisits
grasshopper  2.5         weak          seeks out high-interest rooms, ignores the rest
===========  ==========  ============  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from mflow.sim.config import VISITING_STYLES, SiteConfig, VisitingStyle


@dataclass(frozen=True)
class StyleParameters:
    """Behavioural parameters of one visiting style.

    Attributes:
        dwell_multiplier: scales the geometric mean of the log-normal dwell.
        dwell_sigma: log-scale standard deviation of the dwell.
        route_weight: pull towards the next room on the suggested route.
        interest_exponent: how sharply room interest drives the choice. A grasshopper has
            a high exponent and effectively ignores low-interest rooms.
        area_weight: preference for large open rooms, which is what makes a fish a fish.
        backtrack_penalty: multiplier on returning to the room just left. Below one it
            discourages backtracking; a butterfly's is close to one.
        exit_bias: baseline pull towards the exit, before fatigue is added.
    """

    dwell_multiplier: float
    dwell_sigma: float
    route_weight: float
    interest_exponent: float
    area_weight: float
    backtrack_penalty: float
    exit_bias: float


STYLE_PARAMETERS: Final[dict[VisitingStyle, StyleParameters]] = {
    "ant": StyleParameters(
        dwell_multiplier=1.8,
        dwell_sigma=0.55,
        route_weight=6.0,
        interest_exponent=0.8,
        area_weight=0.0,
        backtrack_penalty=0.15,
        exit_bias=0.2,
    ),
    "fish": StyleParameters(
        dwell_multiplier=0.5,
        dwell_sigma=0.70,
        route_weight=0.8,
        interest_exponent=0.2,
        area_weight=1.2,
        backtrack_penalty=0.35,
        exit_bias=1.4,
    ),
    "butterfly": StyleParameters(
        dwell_multiplier=1.0,
        dwell_sigma=0.80,
        route_weight=1.5,
        interest_exponent=1.0,
        area_weight=0.3,
        backtrack_penalty=0.85,
        exit_bias=0.5,
    ),
    "grasshopper": StyleParameters(
        dwell_multiplier=2.5,
        dwell_sigma=0.65,
        route_weight=0.5,
        interest_exponent=3.0,
        area_weight=0.1,
        backtrack_penalty=0.4,
        exit_bias=0.8,
    ),
}


class StylePolicy:
    """Movement and dwell policy for one visiting style on one site.

    The static parts of the choice -- interest, area, route position -- are precomputed
    per style so that the simulator's inner loop only samples.
    """

    def __init__(self, style: VisitingStyle, config: SiteConfig) -> None:
        self.style = style
        self.parameters = STYLE_PARAMETERS[style]
        self.config = config
        self._interest = {node.id: max(node.interest, 1e-6) for node in config.nodes}
        max_area = max((node.area_m2 for node in config.nodes), default=1.0) or 1.0
        self._area = {node.id: node.area_m2 / max_area for node in config.nodes}
        self._route_position = {node_id: i for i, node_id in enumerate(config.suggested_route)}

    def dwell_steps(
        self, node_id: str, generator: np.random.Generator, step_seconds: int
    ) -> int:
        """Sample a dwell time, in simulation steps.

        The geometric mean is ``base_dwell_minutes * style multiplier * interest``, and
        the log-scale spread is the style's. Corridors and stairs get no interest bonus,
        so people pass through them.
        """
        interest = self._interest[node_id]
        median_minutes = (
            self.config.base_dwell_minutes * self.parameters.dwell_multiplier * interest
        )
        minutes = float(
            generator.lognormal(np.log(max(median_minutes, 1e-3)), self.parameters.dwell_sigma)
        )
        return max(1, round(minutes * 60.0 / step_seconds))

    def next_node_weights(
        self,
        current: str,
        candidates: list[str],
        previous: str | None,
        visited: set[str],
    ) -> np.ndarray:
        """Unnormalised preference over the rooms reachable from ``current``.

        Args:
            current: the room the visitor is in.
            candidates: rooms reachable by one doorway.
            previous: the room they came from, for the backtracking penalty.
            visited: rooms already seen, which an ant avoids repeating and a butterfly
                does not mind.

        Returns:
            Non-negative weights aligned with ``candidates``.
        """
        parameters = self.parameters
        route_index = self._route_position.get(current)
        weights = np.empty(len(candidates), dtype=np.float64)

        for i, candidate in enumerate(candidates):
            weight = 1.0
            weight *= self._interest[candidate] ** parameters.interest_exponent
            weight *= 1.0 + parameters.area_weight * self._area[candidate]

            candidate_index = self._route_position.get(candidate)
            if route_index is not None and candidate_index == route_index + 1:
                weight *= 1.0 + parameters.route_weight
            elif (
                candidate_index is not None
                and route_index is not None
                and candidate_index < route_index
            ):
                    weight *= 1.0 / (1.0 + parameters.route_weight)

            if previous is not None and candidate == previous:
                weight *= parameters.backtrack_penalty
            if candidate in visited and self.style == "ant":
                # An ant is systematic: it prefers rooms it has not covered yet.
                weight *= 0.25
            weights[i] = max(weight, 1e-9)
        return weights

    def exit_probability(self, elapsed_minutes: float, visited_fraction: float) -> float:
        """Probability of heading for the exit at the end of a dwell.

        Two terms: a fatigue term growing linearly with elapsed visit time, which
        reproduces the well-documented decline of attention over a visit (Falk and
        Dierking, 2013, *The Museum Experience Revisited*), and a completion term that
        rises once most of the building has been seen.
        """
        fatigue = self.config.fatigue_per_minute * elapsed_minutes * self.parameters.exit_bias
        completion = 0.3 * max(0.0, visited_fraction - 0.7) / 0.3
        return float(np.clip(fatigue + completion, 0.0, 1.0))


def build_policies(config: SiteConfig) -> dict[VisitingStyle, StylePolicy]:
    """Instantiate one policy per style for a site."""
    return {style: StylePolicy(style, config) for style in VISITING_STYLES}


def sample_styles(
    n: int, config: SiteConfig, generator: np.random.Generator
) -> list[VisitingStyle]:
    """Draw ``n`` visiting styles from the site's configured mix."""
    mix = config.style_mix.as_dict()
    probabilities = np.array([mix[style] for style in VISITING_STYLES], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    indices = generator.choice(len(VISITING_STYLES), size=n, p=probabilities)
    return [VISITING_STYLES[int(i)] for i in indices]
