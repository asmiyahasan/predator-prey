"""Predator vs prey environment using PettingZoo's Parallel API."""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from predprey.physics import (
    ARENA_SIZE,
    NUM_ACTIONS,
    AgentState,
    is_caught,
    step_agent,
)

PREDATOR = "predator"
PREY = "prey"

# Per-step rewards (zero-sum: one agent's gain is the other's loss)
STEP_REWARD = 0.01
CATCH_REWARD = 1.0

# Minimum distance between agents when an episode starts
MIN_SPAWN_DISTANCE = 3.0

OBS_SIZE = 8


class PredatorPreyEnv(ParallelEnv):
    metadata = {"name": "predator_prey_v0", "render_modes": ["human"], "render_fps": 10}

    def __init__(self, max_steps: int = 300, render_mode: str | None = None):
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Species settings: predator is faster, prey is more agile
        self.species = {
            PREDATOR: {"max_speed": 3.0, "acceleration": 7.0},
            PREY: {"max_speed": 3.0, "acceleration": 9.0},
        }

        self.possible_agents = [PREDATOR, PREY]
        self.agents: list[str] = []

        # Build the spaces once, so every call returns the same object
        self._observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS) for agent in self.possible_agents
        }

        self._rng = np.random.default_rng()
        self.states: dict[str, AgentState] = {}
        self.step_count = 0

        self.caught = False
        self.renderer = None
        if render_mode == "human":
            from predprey.render import Renderer  # imported here so CI never needs pygame's display
            self.renderer = Renderer(fps=self.metadata["render_fps"])

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

        self.caught = False

        predator_pos = self._random_position()
        prey_pos = self._random_position()
        # Rejection sampling: re-roll the prey until it's far enough away
        while np.linalg.norm(prey_pos - predator_pos) < MIN_SPAWN_DISTANCE:
            prey_pos = self._random_position()

        self.states = {
            PREDATOR: AgentState(position=predator_pos, **self.species[PREDATOR]),
            PREY: AgentState(position=prey_pos, **self.species[PREY]),
        }

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        # Both agents move at the same time
        for agent in self.agents:
            step_agent(self.states[agent], int(actions[agent]))
        self.step_count += 1

        caught = is_caught(self.states[PREDATOR], self.states[PREY])
        self.caught = caught
        out_of_time = self.step_count >= self.max_steps

        if caught:
            rewards = {PREDATOR: CATCH_REWARD, PREY: -CATCH_REWARD}
        else:
            rewards = {PREDATOR: -STEP_REWARD, PREY: STEP_REWARD}

        # terminated = the game ended naturally (a catch)
        # truncated  = we cut it off (time limit) without a catch
        terminations = {agent: caught for agent in self.agents}
        truncations = {agent: out_of_time and not caught for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {
            agent: {"caught": caught, "steps": self.step_count}
            for agent in self.agents
        }

        if caught or out_of_time:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.renderer is not None and self.renderer.is_open:
            self.renderer.draw(self.states, self.step_count, self.caught)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_position(self) -> np.ndarray:
        radius = AgentState(position=[0.0, 0.0]).radius
        return self._rng.uniform(radius, ARENA_SIZE - radius, size=2)

    def _get_obs(self, agent: str) -> np.ndarray:
        me = self.states[agent]
        other = self.states[PREY if agent == PREDATOR else PREDATOR]

        speed_scale = me.max_speed + other.max_speed

        obs = np.concatenate([
            me.position / ARENA_SIZE * 2 - 1,                    # own position
            me.velocity / me.max_speed,                          # own velocity
            (other.position - me.position) / ARENA_SIZE,         # opponent, relative
            (other.velocity - me.velocity) / speed_scale,        # opponent velocity, relative
        ])
        # Guard against tiny rounding pushing values just past +/-1
        return np.clip(obs, -1.0, 1.0).astype(np.float32)