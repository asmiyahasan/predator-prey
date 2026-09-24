from dataclasses import dataclass, field
import numpy as np

ARENA_SIZE = 10.0
DT = 0.1
DRAG = 0.9
CATCH_EPSILON = 1e-9  # tolerance for floating-point rounding

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

def step_agent(agent: AgentState, action: int) -> None:
    """Update the agent's position and velocity based on the action taken."""
    # Get the direction vector for the action
    direction = ACTION_DIRECTIONS[action]

    # Accelerate in the chosen direction
    agent.velocity += direction * agent.acceleration * DT

    # Apply drag first, so max_speed below is the true top speed
    agent.velocity *= DRAG

    # Limit the speed to max_speed (keep direction, shrink size)
    speed = np.linalg.norm(agent.velocity)
    if speed > agent.max_speed:
        agent.velocity *= agent.max_speed / speed

    # Move
    agent.position += agent.velocity * DT

    # Keep the agent inside the arena walls
    low = agent.radius
    high = ARENA_SIZE - agent.radius
    clipped = np.clip(agent.position, low, high)

    # On any axis where we hit a wall, stop movement along that axis only
    hit_wall = clipped != agent.position
    agent.velocity[hit_wall] = 0.0
    agent.position[:] = clipped

def is_caught(predator: AgentState, prey: AgentState) -> bool:
    """Return True if the predator is touching or overlapping the prey."""
    distance = np.linalg.norm(predator.position - prey.position)
    return bool(distance <= predator.radius + prey.radius + CATCH_EPSILON)