import numpy as np
import pytest

from predprey.physics import (
    ARENA_SIZE,
    DRAG,
    NUM_ACTIONS,
    AgentState,
    is_caught,
    obstacle_bounds,
    overlaps_obstacle,
    step_agent,
)

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
UP_RIGHT = 6


# ----------------------------------------------------------------------
# Movement and arena walls
# ----------------------------------------------------------------------
def test_speed_caps_at_max_speed():
    agent = AgentState(position=[5.0, 5.0])
    for _ in range(6):  # enough steps to reach top speed, not enough to hit the wall
        step_agent(agent, RIGHT)
    assert np.linalg.norm(agent.velocity) == pytest.approx(agent.max_speed)


def test_agent_stops_at_right_wall():
    agent = AgentState(position=[5.0, 5.0])
    for _ in range(40):
        step_agent(agent, RIGHT)
    assert agent.position[0] == pytest.approx(ARENA_SIZE - agent.radius)
    assert agent.velocity[0] == 0.0
    assert agent.position[1] == pytest.approx(5.0)


def test_drag_slows_agent():
    agent = AgentState(position=[5.0, 5.0], velocity=[2.0, 0.0])
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


# ----------------------------------------------------------------------
# Catching
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Obstacles
# ----------------------------------------------------------------------
# One block in the middle of the arena: x from 4 to 6, y from 4 to 6
BLOCK = obstacle_bounds([(4.0, 4.0, 2.0, 2.0)])


def test_obstacle_bounds_converts_rectangles():
    assert np.array_equal(BLOCK, [[4.0, 4.0, 6.0, 6.0]])


def test_agent_stops_at_obstacle_face():
    # Start left of the block, run right into it
    agent = AgentState(position=[1.0, 5.0])
    for _ in range(40):
        step_agent(agent, RIGHT, BLOCK)
    assert agent.position[0] == pytest.approx(4.0 - agent.radius)
    assert agent.velocity[0] == pytest.approx(0.0)


def test_agent_slides_along_obstacle():
    # Pushing diagonally into the left face: x is blocked, but y keeps moving
    agent = AgentState(position=[3.5, 4.2])
    for _ in range(8):
        step_agent(agent, UP_RIGHT, BLOCK)
    assert agent.position[0] == pytest.approx(4.0 - agent.radius)  # pressed against the face
    assert agent.position[1] > 5.0                                 # but slid upwards along it
    assert agent.velocity[1] > 0


def test_random_walk_never_enters_obstacle():
    rng = np.random.default_rng(seed=3)
    agent = AgentState(position=[1.0, 1.0])
    for step in range(3000):
        step_agent(agent, rng.integers(NUM_ACTIONS), BLOCK)
        # Allow a hair of rounding error at the surface
        assert not overlaps_obstacle(agent.position, agent.radius - 1e-6, BLOCK), (
            f"inside obstacle at step {step}: {agent.position}"
        )


def test_agent_starting_inside_obstacle_is_pushed_out():
    agent = AgentState(position=[4.5, 5.0])  # inside, nearest the left side
    step_agent(agent, STAY, BLOCK)
    assert agent.position[0] == pytest.approx(4.0 - agent.radius)


def test_corner_collision_pushes_diagonally():
    # Just off the block's bottom-left corner, overlapping it
    agent = AgentState(position=[3.9, 3.9])
    step_agent(agent, STAY, BLOCK)
    distance_to_corner = np.linalg.norm(agent.position - np.array([4.0, 4.0]))
    assert distance_to_corner == pytest.approx(agent.radius)
