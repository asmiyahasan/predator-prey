import numpy as np
import pytest

from predprey.physics import ARENA_SIZE, obstacle_bounds
from predprey.sensors import (
    MAX_RAY_DISTANCE,
    NUM_RAYS,
    RAY_DIRECTIONS,
    cast_rays,
    has_line_of_sight,
)

NO_OBSTACLES = obstacle_bounds([])
BLOCK = obstacle_bounds([(4.0, 4.0, 2.0, 2.0)])  # x 4..6, y 4..6

# Ray 0 points right (angle 0); with 16 rays, ray 4 points up, 8 left, 12 down
RIGHT_RAY, UP_RAY, LEFT_RAY, DOWN_RAY = 0, 4, 8, 12


def test_ray_directions_are_unit_vectors():
    assert RAY_DIRECTIONS.shape == (NUM_RAYS, 2)
    assert np.allclose(np.linalg.norm(RAY_DIRECTIONS, axis=1), 1.0)


def test_rays_in_empty_arena_hit_walls():
    distances = cast_rays(np.array([2.0, 3.0]), NO_OBSTACLES)
    assert distances[RIGHT_RAY] == pytest.approx(ARENA_SIZE - 2.0)
    assert distances[LEFT_RAY] == pytest.approx(2.0)
    assert distances[UP_RAY] == pytest.approx(ARENA_SIZE - 3.0)
    assert distances[DOWN_RAY] == pytest.approx(3.0)


def test_diagonal_ray_from_centre_reaches_corner():
    distances = cast_rays(np.array([5.0, 5.0]), NO_OBSTACLES)
    up_right = 2  # 45 degrees
    assert distances[up_right] == pytest.approx(5.0 * np.sqrt(2))


def test_ray_stops_at_obstacle():
    distances = cast_rays(np.array([1.0, 5.0]), BLOCK)
    assert distances[RIGHT_RAY] == pytest.approx(3.0)  # block's left face is at x = 4
    assert distances[LEFT_RAY] == pytest.approx(1.0)   # arena wall unaffected


def test_ray_misses_obstacle_it_does_not_point_at():
    distances = cast_rays(np.array([1.0, 1.0]), BLOCK)
    assert distances[UP_RAY] == pytest.approx(ARENA_SIZE - 1.0)


def test_all_rays_within_max_distance():
    rng = np.random.default_rng(0)
    for _ in range(200):
        distances = cast_rays(rng.uniform(0.3, 9.7, size=2), BLOCK)
        assert np.all(distances >= 0)
        assert np.all(distances <= MAX_RAY_DISTANCE + 1e-9)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1.0, 5.0], [9.0, 5.0], False),  # straight through the block
        ([1.0, 1.0], [9.0, 1.0], True),   # along the bottom, under the block
        ([1.0, 5.0], [3.0, 5.0], True),   # both on the same side
        ([5.0, 1.0], [5.0, 9.0], False),  # vertically through the block
        ([1.0, 1.0], [9.0, 9.0], False),  # diagonally through the block
    ],
    ids=["blocked", "clear_below", "same_side", "blocked_vertical", "blocked_diagonal"],
)
def test_line_of_sight(a, b, expected):
    assert has_line_of_sight(np.array(a), np.array(b), BLOCK) is expected


def test_line_of_sight_always_clear_without_obstacles():
    assert has_line_of_sight(np.array([1.0, 5.0]), np.array([9.0, 5.0]), NO_OBSTACLES)
