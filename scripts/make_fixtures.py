"""Record short games as JSON fixtures for testing the TypeScript port.

Each fixture game stores its exact start positions and every action both agents
took, plus what Python produced after each step: positions, velocities,
observations, rewards and whether the game ended. The website's tests replay
the same actions and check they get the same numbers.

Usage:
    uv run python scripts/make_fixtures.py

Re-run this (and commit the result) whenever physics, sensors or observations change.
"""

import json
from pathlib import Path

import numpy as np

from predprey.agents import chaser, fleer
from predprey.env import PREDATOR, PREY, GameConfig, PredatorPreyEnv
from predprey.observations import MEMORY_HORIZON, OBS_SIZE
from predprey.physics import (
    ACTION_DIRECTIONS,
    ARENA_SIZE,
    CATCH_EPSILON,
    DRAG,
    DT,
    NUM_ACTIONS,
    AgentState,
)
from predprey.sensors import MAX_RAY_DISTANCE, RAY_DIRECTIONS

OUT_PATH = Path("fixtures/parity.json")
FORMAT_VERSION = 1


def random_actions(seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(NUM_ACTIONS))


def stay(obs):
    return 0


# name, predator start, prey start, predator policy, prey policy, max steps
GAMES = [
    ("random_moves", [1.0, 1.0], [9.0, 9.0], random_actions(1), random_actions(2), 120),
    ("hide_and_seek", [1.0, 3.0], [5.0, 3.0], chaser, fleer, 150),  # starts hidden
    ("catch", [1.0, 5.0], [5.0, 5.0], chaser, stay, 100),
    ("time_limit", [1.0, 9.0], [9.0, 1.0], random_actions(3), random_actions(4), 20),
]


def agent_snapshot(state: AgentState) -> dict:
    return {"position": state.position.tolist(), "velocity": state.velocity.tolist()}


def record_game(name, predator_start, prey_start, predator_policy, prey_policy, max_steps):
    env = PredatorPreyEnv(max_steps=max_steps, config=GameConfig())
    observations, _ = env.reset(
        options={"predator_position": predator_start, "prey_position": prey_start}
    )

    game = {
        "name": name,
        "max_steps": max_steps,
        "predator_start": predator_start,
        "prey_start": prey_start,
        "initial_observations": {a: observations[a].tolist() for a in env.possible_agents},
        "steps": [],
    }

    while env.agents:
        actions = {
            PREDATOR: predator_policy(observations[PREDATOR]),
            PREY: prey_policy(observations[PREY]),
        }
        observations, rewards, terminations, truncations, _ = env.step(actions)
        game["steps"].append({
            "actions": actions,
            PREDATOR: agent_snapshot(env.states[PREDATOR]),
            PREY: agent_snapshot(env.states[PREY]),
            "visible": env.visible,
            "observations": {a: observations[a].tolist() for a in env.possible_agents},
            "rewards": {a: float(rewards[a]) for a in env.possible_agents},
            "caught": env.caught,
            "truncated": bool(truncations[PREDATOR]),
        })

    visibility = {step["visible"] for step in game["steps"]}
    print(f"  {name:<14} {len(game['steps']):>4} steps | caught: {env.caught!s:<5} | "
          f"visibility changed: {len(visibility) > 1}")
    return game


def main() -> None:
    fixtures = {
        "format_version": FORMAT_VERSION,
        "constants": {
            "arena_size": ARENA_SIZE,
            "dt": DT,
            "drag": DRAG,
            "agent_radius": AgentState(position=[0.0, 0.0]).radius,
            "catch_epsilon": CATCH_EPSILON,
            "action_directions": ACTION_DIRECTIONS.tolist(),
            "ray_directions": RAY_DIRECTIONS.tolist(),
            "max_ray_distance": float(MAX_RAY_DISTANCE),
            "memory_horizon": MEMORY_HORIZON,
            "obs_size": OBS_SIZE,
        },
        "game_config": GameConfig().to_dict(),
        "games": [],
    }

    print("Recording fixture games:")
    for game in GAMES:
        fixtures["games"].append(record_game(*game))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()