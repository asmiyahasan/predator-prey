import numpy as np
import pytest

from predprey.agents import chaser, fleer, make_random_policy
from predprey.physics import NUM_ACTIONS

# Action numbers (must match ACTION_DIRECTIONS in physics.py)
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 5, 6, 7, 8


def make_obs(rel_x: float, rel_y: float) -> np.ndarray:
    """Build an observation with only the opponent's relative position set."""
    obs = np.zeros(8, dtype=np.float32)
    obs[4:6] = [rel_x, rel_y]
    return obs


@pytest.mark.parametrize(
    "rel_pos, expected",
    [
        ((1.0, 0.0), RIGHT),
        ((-1.0, 0.0), LEFT),
        ((0.0, 1.0), UP),
        ((0.0, -1.0), DOWN),
        ((1.0, 1.0), UP_RIGHT),
        ((-1.0, -1.0), DOWN_LEFT),
    ],
    ids=["right", "left", "up", "down", "up_right", "down_left"],
)
def test_chaser_moves_towards_prey(rel_pos, expected):
    assert chaser(make_obs(*rel_pos)) == expected


@pytest.mark.parametrize(
    "rel_pos, expected",
    [
        ((1.0, 0.0), LEFT),
        ((-1.0, 0.0), RIGHT),
        ((0.0, 1.0), DOWN),
        ((1.0, 1.0), DOWN_LEFT),
    ],
    ids=["from_right", "from_left", "from_above", "from_up_right"],
)
def test_fleer_moves_away_from_predator(rel_pos, expected):
    assert fleer(make_obs(*rel_pos)) == expected


def test_scripted_agents_ignore_scale():
    # Direction matters, not distance: near and far prey give the same action
    assert chaser(make_obs(0.05, 0.0)) == chaser(make_obs(0.9, 0.0)) == RIGHT


def test_random_policy_is_reproducible():
    obs = make_obs(0.0, 0.0)
    policy_a = make_random_policy(seed=42)
    policy_b = make_random_policy(seed=42)
    actions_a = [policy_a(obs) for _ in range(100)]
    actions_b = [policy_b(obs) for _ in range(100)]
    assert actions_a == actions_b


def test_random_policy_returns_valid_actions():
    policy = make_random_policy(seed=0)
    obs = make_obs(0.0, 0.0)
    actions = {policy(obs) for _ in range(500)}
    assert actions <= set(range(NUM_ACTIONS))  # never out of range
    assert len(actions) == NUM_ACTIONS         # and uses every action eventually