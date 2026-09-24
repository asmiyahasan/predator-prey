"""Movement, collisions and catch detection. No RL code in here.

This is the file that gets ported to TypeScript for the website, so it stays
small, pure and dependent only on NumPy.
"""

from dataclasses import dataclass, field

import numpy as np

ARENA_SIZE = 10.0
DT = 0.1                # seconds per step
DRAG = 0.9              # fraction of velocity kept each step
CATCH_EPSILON = 1e-9    # tolerance for floating-point rounding


@dataclass
class AgentState:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 0.3
    max_speed: float = 3.0
    acceleration: float = 8.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).copy()
        self.velocity = np.asarray(self.velocity, dtype=np.float64).copy()


# Action -> direction, using maths convention (+y is up)
# 0: stay, 1: up, 2: down, 3: left, 4: right,
# 5: up-left, 6: up-right, 7: down-left, 8: down-right
_RAW_DIRECTIONS = np.array([
    [ 0,  0],
    [ 0,  1],
    [ 0, -1],
    [-1,  0],
    [ 1,  0],
    [-1,  1],
    [ 1,  1],
    [-1, -1],
    [ 1, -1],
], dtype=np.float64)

# Normalise every non-zero row to length 1 (row 0 stays [0, 0])
_lengths = np.linalg.norm(_RAW_DIRECTIONS, axis=1, keepdims=True)
ACTION_DIRECTIONS = np.divide(
    _RAW_DIRECTIONS,
    _lengths,
    out=np.zeros_like(_RAW_DIRECTIONS),
    where=_lengths > 0,
)

NUM_ACTIONS = len(ACTION_DIRECTIONS)

# Obstacles are stored as an (N, 4) array of [x_min, y_min, x_max, y_max]
NO_OBSTACLES = np.zeros((0, 4))


def obstacle_bounds(rects) -> np.ndarray:
    """Convert (x, y, width, height) rectangles into [x_min, y_min, x_max, y_max] rows."""
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([
        rects[:, 0],
        rects[:, 1],
        rects[:, 0] + rects[:, 2],
        rects[:, 1] + rects[:, 3],
    ])


def resolve_obstacle_collision(agent: AgentState, bounds: np.ndarray) -> None:
    """Push the agent out of one rectangle and cancel velocity into its surface."""
    low, high = bounds[:2], bounds[2:]

    # The point on the rectangle closest to the agent's centre
    closest = np.clip(agent.position, low, high)
    offset = agent.position - closest
    distance = np.linalg.norm(offset)

    if distance >= agent.radius:
        return  # not touching

    if distance > 0:
        # Centre is outside the rectangle: push out along the offset direction
        normal = offset / distance
    else:
        # Centre is inside (or exactly on the edge): leave via the nearest side
        gaps = np.array([
            agent.position[0] - low[0],   # to left side
            high[0] - agent.position[0],  # to right side
            agent.position[1] - low[1],   # to bottom side
            high[1] - agent.position[1],  # to top side
        ])
        side = int(np.argmin(gaps))
        normal = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]][side])
        closest = agent.position + normal * gaps[side]

    agent.position[:] = closest + normal * agent.radius

    # Remove only the part of the velocity heading into the surface,
    # so the agent slides along the wall instead of stopping dead
    into_surface = agent.velocity @ normal
    if into_surface < 0:
        agent.velocity -= into_surface * normal


def step_agent(agent: AgentState, action: int, obstacles: np.ndarray = NO_OBSTACLES) -> None:
    """Advance one agent by one time step, applying the chosen action.

    `obstacles` is an (N, 4) array from `obstacle_bounds`.
    """
    # Accelerate in the chosen direction
    direction = ACTION_DIRECTIONS[action]
    agent.velocity += direction * agent.acceleration * DT

    # Apply drag first, so max_speed below is the true top speed
    agent.velocity *= DRAG

    # Limit the speed to max_speed (keep direction, shrink size)
    speed = np.linalg.norm(agent.velocity)
    if speed > agent.max_speed:
        agent.velocity *= agent.max_speed / speed

    # Move
    agent.position += agent.velocity * DT

    # Obstacles inside the arena
    for bounds in obstacles:
        resolve_obstacle_collision(agent, bounds)

    # Keep the agent inside the arena walls
    low = agent.radius
    high = ARENA_SIZE - agent.radius
    clipped = np.clip(agent.position, low, high)

    # On any axis where we hit a wall, stop movement along that axis only
    hit_wall = clipped != agent.position
    agent.velocity[hit_wall] = 0.0
    agent.position[:] = clipped


def overlaps_obstacle(position: np.ndarray, radius: float, obstacles: np.ndarray) -> bool:
    """True if a circle at `position` touches any obstacle."""
    for bounds in obstacles:
        closest = np.clip(position, bounds[:2], bounds[2:])
        if np.linalg.norm(position - closest) < radius:
            return True
    return False


def is_caught(predator: AgentState, prey: AgentState) -> bool:
    """Return True if the predator is touching or overlapping the prey."""
    distance = np.linalg.norm(predator.position - prey.position)
    return bool(distance <= predator.radius + prey.radius + CATCH_EPSILON)
