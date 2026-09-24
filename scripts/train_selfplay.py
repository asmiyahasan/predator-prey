"""Self-play training: predator and prey take turns learning against each other.

Each generation:
  1. train the predator against opponents sampled from the prey pool
  2. save a frozen snapshot of it into the predator pool
  3. train the prey against opponents sampled from the predator pool
  4. save a frozen snapshot of it into the prey pool
  5. evaluate and log results

The game settings used are saved to <run>/config.json, so models are always
watched and evaluated in the same game they were trained in.

Usage:
    uv run python scripts/train_selfplay.py
    uv run python scripts/train_selfplay.py --predator-speed 3.0 --run-name selfplay_equal_speed
    uv run python scripts/train_selfplay.py --generations 2 --steps 20000 --run-name smoke_test

Press Ctrl+C at any time: the latest models are saved before exiting.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from predprey.agents import Policy, chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY, GameConfig
from predprey.evaluation import run_match
from predprey.wrappers import SingleAgentEnv

N_ENVS = 8
SEED = 0
LATEST_OPPONENT_PROB = 0.5  # chance of facing the newest snapshot vs an older one
EVAL_EPISODES = 200
EVAL_SEED = 1000

MODEL_DIR = Path("models")
LOG_DIR = Path("runs")


class OpponentPool:
    """A growing collection of opponents to train against."""

    def __init__(self, members: list[tuple[str, Policy]]):
        self.members = list(members)

    def add(self, name: str, policy: Policy) -> None:
        self.members.append((name, policy))

    def sample(self, rng: np.random.Generator) -> Policy:
        # Often the newest opponent, otherwise any older one, so agents
        # don't forget how to beat strategies they've already seen
        if len(self.members) == 1 or rng.random() < LATEST_OPPONENT_PROB:
            return self.members[-1][1]
        index = rng.integers(len(self.members) - 1)
        return self.members[index][1]


def as_policy(model: PPO, deterministic: bool) -> Policy:
    def policy(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return policy


def frozen_snapshot(path: Path) -> Policy:
    """Load a saved model as a fixed opponent that will never change again.

    Opponents act stochastically (sampling from their action probabilities),
    which gives the learner more varied games to learn from.
    """
    return as_policy(PPO.load(path, device="cpu"), deterministic=False)


def parse_args() -> argparse.Namespace:
    defaults = GameConfig()
    parser = argparse.ArgumentParser(description="Self-play training with SB3 PPO.")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200_000,
                        help="training steps per side per generation")
    parser.add_argument("--run-name", default="selfplay")
    parser.add_argument("--predator-speed", type=float, default=defaults.predator_max_speed)
    parser.add_argument("--predator-accel", type=float, default=defaults.predator_acceleration)
    parser.add_argument("--prey-speed", type=float, default=defaults.prey_max_speed)
    parser.add_argument("--prey-accel", type=float, default=defaults.prey_acceleration)
    return parser.parse_args()


def format_steps(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def main() -> None:
    args = parse_args()
    config = GameConfig(
        predator_max_speed=args.predator_speed,
        predator_acceleration=args.predator_accel,
        prey_max_speed=args.prey_speed,
        prey_acceleration=args.prey_accel,
    )

    run_dir = MODEL_DIR / args.run_name
    (run_dir / PREDATOR).mkdir(parents=True, exist_ok=True)
    (run_dir / PREY).mkdir(parents=True, exist_ok=True)

    # Record exactly which game these models were trained in
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    # Pools start with simple opponents, so neither side forgets the basics
    predator_pool = OpponentPool([("random", make_random_policy(seed=1)), ("chaser", chaser)])
    prey_pool = OpponentPool([("random", make_random_policy(seed=2)), ("fleer", fleer)])

    predator_env = make_vec_env(
        lambda: SingleAgentEnv(agent=PREDATOR, opponent_sampler=prey_pool.sample, config=config),
        n_envs=N_ENVS, seed=SEED,
    )
    prey_env = make_vec_env(
        lambda: SingleAgentEnv(agent=PREY, opponent_sampler=predator_pool.sample, config=config),
        n_envs=N_ENVS, seed=SEED + 100,
    )

    predator_model = PPO("MlpPolicy", predator_env, seed=SEED, verbose=0,
                         tensorboard_log=str(LOG_DIR))
    prey_model = PPO("MlpPolicy", prey_env, seed=SEED + 1, verbose=0,
                     tensorboard_log=str(LOG_DIR))

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "generation",
            "latest_vs_latest_catch_rate", "latest_vs_latest_avg_catch_step",
            "predator_vs_fleer_catch_rate", "prey_vs_chaser_survival_rate",
        ])

    print(f"Self-play: {args.generations} generations x {args.steps:,} steps per side")
    print(f"Game config: {config.to_dict()}\n")
    print(f"{'gen':>4} | {'latest vs latest':>17} | {'avg catch':>9} | "
          f"{'pred vs fleer':>13} | {'prey vs chaser':>14}")
    print("-" * 72)

    generation = 0
    try:
        for generation in range(1, args.generations + 1):
            # --- Predator's turn ---
            predator_model.learn(args.steps, reset_num_timesteps=False,
                                 tb_log_name=f"{args.run_name}_predator")
            predator_path = run_dir / PREDATOR / f"gen_{generation:03d}"
            predator_model.save(predator_path)
            predator_pool.add(f"predator_gen{generation}", frozen_snapshot(predator_path))

            # --- Prey's turn ---
            prey_model.learn(args.steps, reset_num_timesteps=False,
                             tb_log_name=f"{args.run_name}_prey")
            prey_path = run_dir / PREY / f"gen_{generation:03d}"
            prey_model.save(prey_path)
            prey_pool.add(f"prey_gen{generation}", frozen_snapshot(prey_path))

            # --- Evaluate: head to head, and against the scripted baselines ---
            predator = as_policy(predator_model, deterministic=True)
            prey = as_policy(prey_model, deterministic=True)

            def match(pred, pry):
                return run_match(pred, pry, EVAL_EPISODES, seed=EVAL_SEED, config=config)

            head_to_head = match(predator, prey)
            vs_fleer = match(predator, fleer).catch_rate
            vs_chaser = 1 - match(chaser, prey).catch_rate

            avg_catch = head_to_head.mean_catch_steps
            print(f"{generation:>4} | {head_to_head.catch_rate:>16.1%} | "
                  f"{format_steps(avg_catch):>9} | {vs_fleer:>12.1%} | {vs_chaser:>13.1%}")
            with history_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    generation, head_to_head.catch_rate, avg_catch, vs_fleer, vs_chaser,
                ])

    except KeyboardInterrupt:
        print("\nInterrupted, saving the latest models...")
        predator_model.save(run_dir / PREDATOR / "interrupted")
        prey_model.save(run_dir / PREY / "interrupted")
    finally:
        predator_env.close()
        prey_env.close()

    print(f"\nFinished {generation} generation(s). Models and history in {run_dir}/")


if __name__ == "__main__":
    main()
