import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from predprey.env import PREDATOR, PREY, PredatorPreyEnv
from predprey.physics import NUM_ACTIONS

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