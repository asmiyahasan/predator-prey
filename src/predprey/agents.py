"""Scripted (non-learning) agents, used as opponents and evaluation baselines."""

from collections.abc import Callable

import numpy as np

from predprey.observations import OPP_REL_POS, OPP_VISIBLE, RAYS
from predprey.physics import ACTION_DIRECTIONS, NUM_ACTIONS
from predprey.sensors import RAY_DIRECTIONS

# A policy takes an observation and returns an action.
# Scripted agents and trained networks all follow this shape.
Policy = Callable[[np.ndarray], int]

STAY = 0


def best_action_towards(direction: np.ndarray) -> int:
    """Return the action whose direction best matches the given vector."""
    # Dot product with all 9 action directions at once; highest = most aligned
    scores = ACTION_DIRECTIONS @ direction
    return int(np.argmax(scores))


def most_open_direction(obs: np.ndarray) -> np.ndarray:
    """The ray direction with the most free space ahead."""
    return RAY_DIRECTIONS[int(np.argmax(obs[RAYS]))]


def opponent_visible(obs: np.ndarray) -> bool:
    return obs[OPP_VISIBLE] > 0.5


def chaser(obs: np.ndarray) -> int:
    """Predator: run at the prey if visible, otherwise head for open space to search."""
    if opponent_visible(obs):
        return best_action_towards(obs[OPP_REL_POS])
    return best_action_towards(most_open_direction(obs))


def fleer(obs: np.ndarray) -> int:
    """Prey: run directly away from the predator if visible, otherwise stay hidden."""
    if opponent_visible(obs):
        return best_action_towards(-obs[OPP_REL_POS])
    return STAY


def make_random_policy(seed: int | None = None) -> Policy:
    """Create a policy that ignores the observation and acts randomly."""
    rng = np.random.default_rng(seed)

    def random_policy(obs: np.ndarray) -> int:
        return int(rng.integers(NUM_ACTIONS))

    return random_policy
