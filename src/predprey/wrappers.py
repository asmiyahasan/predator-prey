"""Wrap the two-agent environment as a single-agent Gymnasium environment.

One side is controlled by the learner; the other is played by a fixed policy
(a scripted agent or a frozen snapshot of a trained network). This turns a
multi-agent problem into a normal single-agent one that standard RL code can
train on.

Pass either:
- `opponent`: one policy used for every episode, or
- `opponent_sampler`: a function that picks an opponent at each reset
  (used for self-play with an opponent pool).
"""

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from predprey.agents import Policy
from predprey.env import PREDATOR, PREY, PredatorPreyEnv

OpponentSampler = Callable[[np.random.Generator], Policy]


class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        agent: str,
        opponent: Policy | None = None,
        opponent_sampler: OpponentSampler | None = None,
        max_steps: int = 300,
        render_mode: str | None = None,
    ):
        if agent not in (PREDATOR, PREY):
            raise ValueError(f"agent must be '{PREDATOR}' or '{PREY}', got '{agent}'")
        if (opponent is None) == (opponent_sampler is None):
            raise ValueError("pass exactly one of `opponent` or `opponent_sampler`")

        self.agent = agent
        self.opponent_name = PREY if agent == PREDATOR else PREDATOR
        self.opponent = opponent
        self.opponent_sampler = opponent_sampler
        self.render_mode = render_mode

        self.env = PredatorPreyEnv(max_steps=max_steps, render_mode=render_mode)

        # In Gymnasium these are attributes, not methods
        self.observation_space = self.env.observation_space(agent)
        self.action_space = self.env.action_space(agent)

        self._opponent_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds Gymnasium's own self.np_random

        # With a pool, each episode is played against a freshly chosen opponent
        if self.opponent_sampler is not None:
            self.opponent = self.opponent_sampler(self.np_random)

        observations, infos = self.env.reset(seed=seed, options=options)
        self._opponent_obs = observations[self.opponent_name]
        return observations[self.agent], infos[self.agent]

    def step(self, action):
        actions = {
            self.agent: int(action),
            self.opponent_name: self.opponent(self._opponent_obs),
        }
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Remember what the opponent sees now, for its next decision
        self._opponent_obs = observations[self.opponent_name]

        return (
            observations[self.agent],
            float(rewards[self.agent]),
            bool(terminations[self.agent]),
            bool(truncations[self.agent]),
            infos[self.agent],
        )

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()