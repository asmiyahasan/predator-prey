"""Raycast sensors: how agents 'see' walls, obstacles and each other.

Each agent casts NUM_RAYS rays evenly spaced around a full circle (fixed
world directions, since agents have no facing direction). Each ray reports
how far it travels before hitting an arena wall or an obstacle.

Line of sight decides whether an agent can see its opponent: if an obstacle
sits between them, the opponent is hidden.
"""

import numpy as np

from predprey.physics import ARENA_SIZE

NUM_RAYS = 16
MAX_RAY_DISTANCE = ARENA_SIZE * np.sqrt(2)  # the arena's diagonal: no ray can go further

_angles = np.arange(NUM_RAYS) * (2 * np.pi / NUM_RAYS)
RAY_DIRECTIONS = np.column_stack([np.cos(_angles), np.sin(_angles)])  # (NUM_RAYS, 2)


def _distance_to_arena_edge(origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """How far each ray travels from `origin` before leaving the square arena."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # For each axis, the wall the ray is heading towards depends on its sign
        target = np.where(directions > 0, ARENA_SIZE, 0.0)
        t = (target - origin) / directions
    t = np.where(directions == 0, np.inf, t)  # rays parallel to an axis never hit it
    return np.clip(t.min(axis=1), 0.0, None)


def _distance_to_boxes(origin: np.ndarray, directions: np.ndarray,
                       obstacles: np.ndarray) -> np.ndarray:
    """How far each ray travels before hitting any obstacle (inf if it misses all).

    Uses the 'slab' method: a ray is inside a box when it is between the box's
    x-limits AND its y-limits at the same time.
    """
    if len(obstacles) == 0:
        return np.full(len(directions), np.inf)

    d = directions[:, None, :]        # (rays, 1, 2)
    low = obstacles[None, :, :2]      # (1, boxes, 2)
    high = obstacles[None, :, 2:]
    o = origin[None, None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (low - o) / d
        t2 = (high - o) / d

    # Rays parallel to an axis: inside that slab forever, or never
    parallel = d == 0
    inside_slab = (o >= low) & (o <= high)
    t1 = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), t1)
    t2 = np.where(parallel, np.where(inside_slab, np.inf, np.inf), t2)

    t_enter = np.minimum(t1, t2).max(axis=2)   # (rays, boxes)
    t_exit = np.maximum(t1, t2).min(axis=2)

    hit = (t_enter <= t_exit) & (t_exit >= 0)
    distance = np.where(hit, np.maximum(t_enter, 0.0), np.inf)
    return distance.min(axis=1)


def cast_rays(origin: np.ndarray, obstacles: np.ndarray,
              directions: np.ndarray = RAY_DIRECTIONS) -> np.ndarray:
    """Distance along each ray to the nearest wall or obstacle."""
    return np.minimum(
        _distance_to_arena_edge(origin, directions),
        _distance_to_boxes(origin, directions, obstacles),
    )


def has_line_of_sight(a: np.ndarray, b: np.ndarray, obstacles: np.ndarray) -> bool:
    """True if no obstacle blocks the straight line between points a and b."""
    offset = b - a
    distance = np.linalg.norm(offset)
    if distance == 0 or len(obstacles) == 0:
        return True
    direction = (offset / distance)[None, :]
    return bool(_distance_to_boxes(a, direction, obstacles)[0] >= distance)
