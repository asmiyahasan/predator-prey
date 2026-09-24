import numpy as np
import pytest

from predprey.physics import ARENA_SIZE, DRAG, NUM_ACTIONS, AgentState, step_agent, is_caught

STAY = 0
RIGHT = 4


def test_speed_caps_at_max_speed():
    agent = AgentState(position=[5.0, 5.0])
    for _ in range(6):
        step_agent(agent, RIGHT)
    assert np.linalg.norm(agent.velocity) == pytest.approx(agent.max_speed)

def test_agent_stops_at_right_wall():
    agent = AgentState(position=[5.0, 5.0])
    for _ in range(40):
        step_agent(agent, RIGHT)
    assert np.linalg.norm(agent.velocity[0]) == 0
    assert agent.position[0] == pytest.approx(ARENA_SIZE - agent.radius)

def test_drag_slows_agent():
    agent = AgentState(position=[5.0, 5.0], velocity=[2.0, 0.0])
    for _ in range(1):
        step_agent(agent, STAY)
    assert np.linalg.norm(agent.velocity) == pytest.approx(2.0 * DRAG)  


def test_random_walk_stays_in_bounds():
    rng = np.random.default_rng(seed=42)
    agent = AgentState(position=[5.0, 5.0])
    low = agent.radius
    high = ARENA_SIZE - agent.radius

    for step in range(1000):
        step_agent(agent, rng.integers(NUM_ACTIONS))
        assert np.all(agent.position >= low), f"left arena at step {step}: {agent.position}"
        assert np.all(agent.position <= high), f"left arena at step {step}: {agent.position}"


def test_agents_have_separate_velocity_arrays():
    a = AgentState(position=[2.0, 2.0])
    b = AgentState(position=[8.0, 8.0])
    a.velocity += np.array([1.0, 0.5])
    assert np.array_equal(b.velocity, [0.0, 0.0])
    assert a.velocity is not b.velocity


@pytest.mark.parametrize(
    "prey_position, expected",
    [
        ([8.0, 8.0], False),   # far apart
        ([2.3, 2.0], True),    # overlapping (0.3 apart, radii sum 0.6)
        ([2.6, 2.0], True),    # exactly touching (0.6 apart)
        ([2.61, 2.0], False),  # just apart
    ],
    ids=["far_apart", "overlapping", "exactly_touching", "just_apart"],
)
def test_is_caught(prey_position, expected):
    predator = AgentState(position=[2.0, 2.0])
    prey = AgentState(position=prey_position)
    assert is_caught(predator, prey) is expected