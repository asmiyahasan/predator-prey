import json

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from predprey.env import PREDATOR, PREY, GameConfig, PredatorPreyEnv
from predprey.observations import (
    LAST_SEEN_REL_POS,
    MEMORY_HORIZON,
    OBS_SIZE,
    OPP_REL_POS,
    OPP_REL_VEL,
    OPP_VISIBLE,
    TIME_SINCE_SEEN,
)
from predprey.physics import NUM_ACTIONS, overlaps_obstacle

STAY = 0


def test_pettingzoo_api():
    env = PredatorPreyEnv()
    parallel_api_test(env, num_cycles=1000)


def test_rewards_are_zero_sum():
    env = PredatorPreyEnv()
    rng = np.random.default_rng(seed=0)

    for episode in range(20):
        env.reset(seed=episode)
        while env.agents:
            actions = {agent: rng.integers(NUM_ACTIONS) for agent in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            assert rewards[PREDATOR] + rewards[PREY] == pytest.approx(0.0)


def test_episode_truncates_at_max_steps():
    max_steps = 5
    env = PredatorPreyEnv(max_steps=max_steps)
    env.reset(seed=0)
    stay = {PREDATOR: STAY, PREY: STAY}

    # Steps before the limit: episode keeps going
    for _ in range(max_steps - 1):
        _, _, terminations, truncations, _ = env.step(stay)
        assert not any(terminations.values())
        assert not any(truncations.values())

    # The final step: truncated (time ran out), not terminated (no catch)
    _, _, terminations, truncations, _ = env.step(stay)
    assert all(truncations.values())
    assert not any(terminations.values())
    assert env.agents == []


def test_same_seed_same_episode():
    env_a = PredatorPreyEnv()
    env_b = PredatorPreyEnv()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)

    rng = np.random.default_rng(seed=7)
    while env_a.agents:
        for agent in env_a.possible_agents:
            assert np.array_equal(obs_a[agent], obs_b[agent])

        actions = {agent: rng.integers(NUM_ACTIONS) for agent in env_a.agents}
        obs_a, rewards_a, _, _, _ = env_a.step(actions)
        obs_b, rewards_b, _, _, _ = env_b.step(actions)
        assert rewards_a == rewards_b

    assert env_b.agents == []  # both episodes ended on the same step


# ----------------------------------------------------------------------
# Obstacles, observations and config
# ----------------------------------------------------------------------
def test_observation_size():
    env = PredatorPreyEnv()
    observations, _ = env.reset(seed=0)
    for agent in env.possible_agents:
        assert observations[agent].shape == (OBS_SIZE,)
        assert observations[agent].dtype == np.float32


def test_agents_never_spawn_inside_obstacles():
    env = PredatorPreyEnv()
    for seed in range(300):
        env.reset(seed=seed)
        for state in env.states.values():
            assert not overlaps_obstacle(state.position, state.radius, env.obstacles)


def test_hidden_opponent_is_zeroed_out():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    # Put a block between the agents: the left and right of the bottom-left block
    env.states[PREDATOR].position[:] = [1.0, 3.0]
    env.states[PREY].position[:] = [5.0, 3.0]
    env.states[PREY].velocity[:] = [2.0, 0.0]
    env._update_memory()
    obs = env._get_obs(PREDATOR)

    assert not env.visible
    assert obs[OPP_VISIBLE] == 0.0
    assert np.all(obs[OPP_REL_POS] == 0.0)
    assert np.all(obs[OPP_REL_VEL] == 0.0)


def test_visible_opponent_is_reported():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    env.states[PREDATOR].position[:] = [1.0, 5.0]   # middle corridor, clear view
    env.states[PREY].position[:] = [9.0, 5.0]
    env._update_memory()
    obs = env._get_obs(PREDATOR)

    assert env.visible
    assert obs[OPP_VISIBLE] == 1.0
    assert obs[OPP_REL_POS][0] == pytest.approx(0.8)  # 8 units right, / ARENA_SIZE


def test_no_obstacles_config():
    env = PredatorPreyEnv(config=GameConfig(obstacles=()))
    env.reset(seed=0)
    assert len(env.obstacles) == 0
    assert env.can_see_each_other()


def test_config_survives_json_round_trip():
    config = GameConfig(predator_max_speed=3.3, obstacles=((1.0, 2.0, 3.0, 4.0),))
    restored = GameConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_config_changes_species_speed():
    env = PredatorPreyEnv(config=GameConfig(predator_max_speed=5.0))
    env.reset(seed=0)
    assert env.states[PREDATOR].max_speed == 5.0


# ----------------------------------------------------------------------
# Memory: where the opponent was last seen
# ----------------------------------------------------------------------
def _place(env, predator_pos, prey_pos):
    """Move both agents and refresh visibility and memory, as a real step would."""
    env.states[PREDATOR].position[:] = predator_pos
    env.states[PREY].position[:] = prey_pos
    env._update_memory()


def test_memory_starts_empty_when_hidden_at_spawn():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    env.last_seen = {a: None for a in env.possible_agents}
    env.steps_since_seen = {a: None for a in env.possible_agents}
    _place(env, [1.0, 3.0], [5.0, 3.0])  # a block between them
    obs = env._get_obs(PREDATOR)
    assert obs[TIME_SINCE_SEEN] == 1.0
    assert np.all(obs[LAST_SEEN_REL_POS] == 0.0)


def test_memory_remembers_last_sighting():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    _place(env, [1.0, 5.0], [5.0, 5.0])       # clear view along the middle corridor
    obs = env._get_obs(PREDATOR)
    assert obs[TIME_SINCE_SEEN] == 0.0
    assert obs[LAST_SEEN_REL_POS][0] == pytest.approx(0.4)

    # Prey slips behind a block; predator should remember where it was
    for _ in range(10):
        _place(env, [1.0, 5.0], [5.0, 3.0])
    obs = env._get_obs(PREDATOR)
    assert obs[OPP_VISIBLE] == 0.0
    assert obs[LAST_SEEN_REL_POS] == pytest.approx([0.4, 0.0])   # still points at (5, 5)
    assert obs[TIME_SINCE_SEEN] == pytest.approx(10 / MEMORY_HORIZON)


def test_memory_is_relative_to_current_position():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    _place(env, [1.0, 5.0], [5.0, 5.0])       # seen at (5, 5)
    _place(env, [1.0, 3.0], [5.0, 3.0])       # both move, block in the way
    obs = env._get_obs(PREDATOR)
    assert obs[LAST_SEEN_REL_POS] == pytest.approx([0.4, 0.2])   # (5,5) - (1,3), / 10


def test_memory_caps_at_one():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    _place(env, [1.0, 5.0], [5.0, 5.0])
    for _ in range(MEMORY_HORIZON * 2):
        _place(env, [1.0, 3.0], [5.0, 3.0])
    assert env._get_obs(PREDATOR)[TIME_SINCE_SEEN] == 1.0
