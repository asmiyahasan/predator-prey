"""Run matches between two policies and collect statistics."""

from dataclasses import dataclass, field

import numpy as np

from predprey.agents import Policy
from predprey.env import PREDATOR, PREY, PredatorPreyEnv


@dataclass
class MatchResult:
    episodes: int
    catch_steps: list[int] = field(default_factory=list)  # step of each catch

    @property
    def catches(self) -> int:
        return len(self.catch_steps)

    @property
    def catch_rate(self) -> float:
        return self.catches / self.episodes

    @property
    def mean_catch_steps(self) -> float | None:
        """Average steps to a catch, or None if there were no catches."""
        if not self.catch_steps:
            return None
        return float(np.mean(self.catch_steps))


def run_match(
    predator_policy: Policy,
    prey_policy: Policy,
    episodes: int = 200,
    seed: int = 0,
    max_steps: int = 300,
) -> MatchResult:
    """Play `episodes` games between two policies, without rendering."""
    env = PredatorPreyEnv(max_steps=max_steps)
    result = MatchResult(episodes=episodes)

    for episode in range(episodes):
        # A different but fixed seed per episode: varied starts, reproducible results
        observations, _ = env.reset(seed=seed + episode)

        while env.agents:
            actions = {
                PREDATOR: predator_policy(observations[PREDATOR]),
                PREY: prey_policy(observations[PREY]),
            }
            observations, _, _, _, _ = env.step(actions)

        if env.caught:
            result.catch_steps.append(env.step_count)

    env.close()
    return result