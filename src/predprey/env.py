"""Predator vs prey environment using PettingZoo's Parallel API."""

from dataclasses import asdict, dataclass

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from predprey.observations import OBS_SIZE, build_observation
from predprey.physics import (
    ARENA_SIZE,
    NUM_ACTIONS,
    AgentState,
    is_caught,
    obstacle_bounds,
    overlaps_obstacle,
    step_agent,
)
from predprey.sensors import has_line_of_sight

PREDATOR = "predator"
PREY = "prey"

# Per-step rewards (zero-sum: one agent's gain is the other's loss)
STEP_REWARD = 0.01
CATCH_REWARD = 1.0

# Minimum distance between agents when an episode starts
MIN_SPAWN_DISTANCE = 3.0

# Rectangles as (x, y, width, height), where (x, y) is the bottom-left corner.
# Four blocks around the centre, with corridors between them and along the walls.
# Blocks must be thicker than two agent radii (0.6), or catches could happen through them.
FOUR_BLOCKS = (
    (2.2, 2.2, 1.6, 1.6),
    (6.2, 2.2, 1.6, 1.6),
    (2.2, 6.2, 1.6, 1.6),
    (6.2, 6.2, 1.6, 1.6),
)

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class GameConfig:
    """Tunable game settings. Saved with every training run for reproducibility."""

    predator_max_speed: float = 3.0
    predator_acceleration: float = 7.0
    prey_max_speed: float = 3.0
    prey_acceleration: float = 9.0
    obstacles: tuple[Rect, ...] = FOUR_BLOCKS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        """Rebuild a config from JSON, where tuples come back as lists."""
        data = dict(data)
        if "obstacles" in data:
            data["obstacles"] = tuple(tuple(float(v) for v in rect) for rect in data["obstacles"])
        return cls(**data)


class PredatorPreyEnv(ParallelEnv):
    metadata = {"name": "predator_prey_v1", "render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        max_steps: int = 300,
        render_mode: str | None = None,
        config: GameConfig | None = None,
    ):
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.config = config or GameConfig()
        self.obstacles = obstacle_bounds(self.config.obstacles)

        self.species = {
            PREDATOR: {
                "max_speed": self.config.predator_max_speed,
                "acceleration": self.config.predator_acceleration,
            },
            PREY: {
                "max_speed": self.config.prey_max_speed,
                "acceleration": self.config.prey_acceleration,
            },
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
            from predprey.render import Renderer  # imported here so CI never needs a display
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

        predator_pos = self._random_free_position()
        prey_pos = self._random_free_position()
        # Rejection sampling: re-roll the prey until it's far enough away
        while np.linalg.norm(prey_pos - predator_pos) < MIN_SPAWN_DISTANCE:
            prey_pos = self._random_free_position()

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
            step_agent(self.states[agent], int(actions[agent]), self.obstacles)
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
        visible = self.can_see_each_other()
        infos = {
            agent: {"caught": caught, "steps": self.step_count, "visible": visible}
            for agent in self.agents
        }

        if caught or out_of_time:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.renderer is not None and self.renderer.is_open:
            self.renderer.draw(
                self.states,
                self.step_count,
                self.caught,
                obstacles=self.obstacles,
                visible=self.can_see_each_other(),
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def can_see_each_other(self) -> bool:
        return has_line_of_sight(
            self.states[PREDATOR].position, self.states[PREY].position, self.obstacles
        )

    def _random_free_position(self) -> np.ndarray:
        """A random position inside the arena that doesn't overlap an obstacle."""
        radius = AgentState(position=[0.0, 0.0]).radius
        while True:
            position = self._rng.uniform(radius, ARENA_SIZE - radius, size=2)
            if not overlaps_obstacle(position, radius, self.obstacles):
                return position

    def _get_obs(self, agent: str) -> np.ndarray:
        me = self.states[agent]
        other = self.states[PREY if agent == PREDATOR else PREDATOR]
        return build_observation(me, other, self.obstacles)
