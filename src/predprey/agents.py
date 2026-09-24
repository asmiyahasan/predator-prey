"""Scripted (non-learning) agents, used as opponents and evaluation baselines."""

from collections.abc import Callable

import numpy as np

from predprey.physics import ACTION_DIRECTIONS, NUM_ACTIONS

# A policy takes an observation and returns an action.
# Scripted agents and (later) trained networks all follow this shape.
Policy = Callable[[np.ndarray], int]

# Where the opponent's relative position lives in the observation vector
REL_POS = slice(4, 6)


def best_action_towards(direction: np.ndarray) -> int:
    """Return the action whose direction best matches the given vector."""
    # Dot product with all 9 action directions at once; highest = most aligned
    scores = ACTION_DIRECTIONS @ direction
    return int(np.argmax(scores))


def chaser(obs: np.ndarray) -> int:
    """Predator: always move towards the prey."""
    return best_action_towards(obs[REL_POS])


def fleer(obs: np.ndarray) -> int:
    """Prey: always move directly away from the predator."""
    return best_action_towards(-obs[REL_POS])


def make_random_policy(seed: int | None = None) -> Policy:
    """Create a policy that ignores the observation and acts randomly."""
    rng = np.random.default_rng(seed)

    def random_policy(obs: np.ndarray) -> int:
        return int(rng.integers(NUM_ACTIONS))

    return random_policy