import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from predprey.agents import chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY, GameConfig
from predprey.wrappers import SingleAgentEnv

STAY = 0


def stay_still(obs):
    return STAY


# ----------------------------------------------------------------------
# Single fixed opponent
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "agent, opponent",
    [(PREDATOR, fleer), (PREY, chaser)],
    ids=["train_predator", "train_prey"],
)
def test_gymnasium_api(agent, opponent):
    env = SingleAgentEnv(agent=agent, opponent=opponent)
    check_env(env, skip_render_check=True)


def test_invalid_agent_name_raises():
    with pytest.raises(ValueError):
        SingleAgentEnv(agent="wolf", opponent=fleer)


def test_predator_catches_stationary_prey():
    # Drive the wrapped predator with the chaser against a prey that never moves.
    # Open arena, so the straight-line chaser can't get stuck behind a block.
    env = SingleAgentEnv(agent=PREDATOR, opponent=stay_still, config=GameConfig(obstacles=()))
    obs, _ = env.reset(seed=0)

    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(chaser(obs))

    assert terminated and not truncated
    assert reward == 1.0
    assert info["caught"] is True


def test_prey_gets_positive_reward_while_surviving():
    env = SingleAgentEnv(agent=PREY, opponent=stay_still, max_steps=5)
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(STAY)
    assert reward > 0
    assert not terminated and not truncated


def test_opponent_sees_its_own_observation():
    # A random opponent must work every step without errors, until the episode ends
    env = SingleAgentEnv(agent=PREDATOR, opponent=make_random_policy(seed=0))
    env.reset(seed=1)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated


# ----------------------------------------------------------------------
# Opponent sampler (self-play)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"opponent": fleer, "opponent_sampler": lambda rng: fleer},  # both
        {},                                                           # neither
    ],
    ids=["both", "neither"],
)
def test_must_pass_exactly_one_opponent_source(kwargs):
    with pytest.raises(ValueError):
        SingleAgentEnv(agent=PREDATOR, **kwargs)


def test_sampler_called_on_every_reset():
    received = []

    def counting_sampler(rng):
        received.append(rng)
        return stay_still

    env = SingleAgentEnv(agent=PREDATOR, opponent_sampler=counting_sampler)
    for episode in range(3):
        env.reset(seed=episode)

    assert len(received) == 3
    # The sampler gets the env's random generator, so its choices are reproducible
    assert all(isinstance(rng, np.random.Generator) for rng in received)


def test_sampled_opponent_is_the_one_that_plays():
    calls = {"a": 0, "b": 0}

    def opponent_a(obs):
        calls["a"] += 1
        return STAY

    def opponent_b(obs):
        calls["b"] += 1
        return STAY

    choices = iter([opponent_a, opponent_b])
    env = SingleAgentEnv(agent=PREDATOR, opponent_sampler=lambda rng: next(choices))

    env.reset(seed=0)          # picks opponent_a
    env.step(STAY)
    env.step(STAY)
    env.reset(seed=1)          # picks opponent_b
    env.step(STAY)

    assert calls == {"a": 2, "b": 1}


def test_gymnasium_api_with_sampler():
    env = SingleAgentEnv(agent=PREY, opponent_sampler=lambda rng: chaser)
    check_env(env, skip_render_check=True)
