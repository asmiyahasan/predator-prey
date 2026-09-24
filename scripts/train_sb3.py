"""Train the predator against the scripted fleer using Stable-Baselines3 PPO.

This is a sanity check: if SB3 can learn here, the environment is learnable,
and later problems with our own PPO are in our code, not the env.

Usage:
    uv run python scripts/train_sb3.py
    uv run python scripts/train_sb3.py --timesteps 500000 --run-name quick_test

Press Ctrl+C at any time to stop early: the model trained so far is still
saved and evaluated.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from predprey.agents import fleer
from predprey.env import PREDATOR
from predprey.evaluation import run_match
from predprey.wrappers import SingleAgentEnv

N_ENVS = 8
SEED = 0
TOTAL_TIMESTEPS = 3_000_000
CHECKPOINT_EVERY = 250_000  # total environment steps between checkpoints
EVAL_EPISODES = 200
EVAL_SEED = 1000  # different from training seeds, so evaluation uses unseen starts

MODEL_DIR = Path("models")
LOG_DIR = Path("runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the predator with SB3 PPO.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--run-name", default="sb3_predator_vs_fleer")
    return parser.parse_args()


def make_env() -> SingleAgentEnv:
    return SingleAgentEnv(agent=PREDATOR, opponent=fleer)


def evaluate(model: PPO) -> None:
    """Play the trained predator against the fleer and print the results."""

    def trained_predator(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    result = run_match(trained_predator, fleer, episodes=EVAL_EPISODES, seed=EVAL_SEED)
    avg = result.mean_catch_steps
    avg_text = f"{avg:.1f}" if avg is not None else "-"

    print(f"\nTrained predator vs fleer over {EVAL_EPISODES} episodes:")
    print(f"  catch rate:      {result.catch_rate:.1%}")
    print(f"  avg catch step:  {avg_text}")
    print("  (chaser baseline: 100.0%, ~32 steps)")


def main() -> None:
    args = parse_args()
    run_dir = MODEL_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Several copies of the game side by side, to collect experience faster
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)

    model = PPO(
        "MlpPolicy",            # a small fully connected neural network
        vec_env,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    # save_freq counts calls per environment; each call steps all N_ENVS at once
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="predator",
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=args.run_name,
            callback=checkpoint_callback,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving the model trained so far...")
    finally:
        vec_env.close()

    final_path = run_dir / "final"
    model.save(final_path)
    print(f"\nSaved model to {final_path}.zip ({model.num_timesteps:,} steps trained)")

    evaluate(model)


if __name__ == "__main__":
    main()