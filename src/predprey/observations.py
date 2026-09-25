"""The observation vector: what each agent sees every step.

Layout (28 numbers, all within -1..1):

    index  0-1    own position, rescaled to -1..1
    index  2-3    own velocity / own max speed
    index  4      1.0 if the opponent is in line of sight, else 0.0
    index  5-6    opponent position relative to me / ARENA_SIZE   (0 if hidden)
    index  7-8    opponent velocity relative to me / speed scale  (0 if hidden)
    index  9-10   where I LAST SAW the opponent, relative to me / ARENA_SIZE
                  (0 if never seen this episode)
    index  11     how long ago I last saw it: 0 = right now, 1 = MEMORY_HORIZON
                  steps or more ago (or never)
    index  12-27  16 ray distances to walls/obstacles / MAX_RAY_DISTANCE

Anything that reads observations (scripted agents, tests, the website port)
should use these named slices rather than raw numbers.
"""

import numpy as np

from predprey.physics import ARENA_SIZE, AgentState
from predprey.sensors import MAX_RAY_DISTANCE, NUM_RAYS, cast_rays

OWN_POS = slice(0, 2)
OWN_VEL = slice(2, 4)
OPP_VISIBLE = 4
OPP_REL_POS = slice(5, 7)
OPP_REL_VEL = slice(7, 9)
LAST_SEEN_REL_POS = slice(9, 11)
TIME_SINCE_SEEN = 11
RAYS = slice(12, 12 + NUM_RAYS)

OBS_SIZE = 12 + NUM_RAYS

# After this many steps (10 seconds) without a sighting, memory reads as "long ago"
MEMORY_HORIZON = 100


def build_observation(
    me: AgentState,
    other: AgentState,
    obstacles: np.ndarray,
    visible: bool,
    last_seen: np.ndarray | None,
    steps_since_seen: int | None,
) -> np.ndarray:
    """Build one agent's observation.

    `last_seen` is where this agent last saw its opponent (None if never), and
    `steps_since_seen` how many steps ago that was.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float64)

    obs[OWN_POS] = me.position / ARENA_SIZE * 2 - 1
    obs[OWN_VEL] = me.velocity / me.max_speed

    if visible:
        speed_scale = me.max_speed + other.max_speed
        obs[OPP_VISIBLE] = 1.0
        obs[OPP_REL_POS] = (other.position - me.position) / ARENA_SIZE
        obs[OPP_REL_VEL] = (other.velocity - me.velocity) / speed_scale

    if last_seen is None:
        obs[TIME_SINCE_SEEN] = 1.0
    else:
        obs[LAST_SEEN_REL_POS] = (last_seen - me.position) / ARENA_SIZE
        obs[TIME_SINCE_SEEN] = min(steps_since_seen / MEMORY_HORIZON, 1.0)

    obs[RAYS] = cast_rays(me.position, obstacles) / MAX_RAY_DISTANCE

    # Guard against tiny rounding pushing values just past +/-1
    return np.clip(obs, -1.0, 1.0).astype(np.float32)
