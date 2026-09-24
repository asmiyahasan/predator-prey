import pytest
from gymnasium.utils.env_checker import check_env

from predprey.agents import chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY
from predprey.wrappers import SingleAgentEnv

STAY = 0


def stay_still(obs):
    return STAY


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
    # Drive the wrapped predator with the chaser against a prey that never moves
    env = SingleAgentEnv(agent=PREDATOR, opponent=stay_still)
    obs, _ = env.reset(seed=0)

    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(chaser(obs))

    assert terminated and not truncated
    assert reward == 1.0
    assert info["caught"] is True


def test_prey_gets_positive_reward_while_surviving():
    env = SingleAgentEnv(agent=PREY, opponent=stay_still, max_steps=5)
    obs, _ = env.reset(seed=0)
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