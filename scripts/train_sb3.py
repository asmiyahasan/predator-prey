"""Train one agent (predator or prey) against a scripted opponent with SB3 PPO.

Usage:
    uv run python scripts/train_sb3.py --agent predator            # vs fleer (default)
    uv run python scripts/train_sb3.py --agent prey                # vs chaser (default)
    uv run python scripts/train_sb3.py --agent prey --opponent random --timesteps 500000

Press Ctrl+C at any time to stop early: the model trained so far is still
saved and evaluated.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from predprey.agents import Policy, chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY
from predprey.evaluation import run_match
from predprey.wrappers import SingleAgentEnv

N_ENVS = 8
SEED = 0
TOTAL_TIMESTEPS = 3_000_000  # total environment steps to train for
CHECKPOINT_EVERY = 250_000  # total environment steps between checkpoints
EVAL_EPISODES = 200
EVAL_SEED = 1000  # different from training seeds, so evaluation uses unseen starts

MODEL_DIR = Path("models")
LOG_DIR = Path("runs")

# Scripted policies the learner can train against
OPPONENTS = {
    "chaser": lambda: chaser,
    "fleer": lambda: fleer,
    "random": lambda: make_random_policy(seed=SEED),
    "stay": lambda: (lambda obs: 0),
}
DEFAULT_OPPONENT = {PREDATOR: "fleer", PREY: "chaser"}

# The scripted agent each side is compared against after training
BASELINE = {PREDATOR: ("chaser", chaser), PREY: ("fleer", fleer)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one agent with SB3 PPO.")
    parser.add_argument("--agent", choices=[PREDATOR, PREY], default=PREDATOR)
    parser.add_argument("--opponent", choices=OPPONENTS, default=None,
                        help="scripted opponent (default: fleer for predator, chaser for prey)")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="total environment steps to train for")
    parser.add_argument("--run-name", default=None,
                        help="default: sb3_<agent>_vs_<opponent>")
    args = parser.parse_args()

    if args.opponent is None:
        args.opponent = DEFAULT_OPPONENT[args.agent]
    if args.run_name is None:
        args.run_name = f"sb3_{args.agent}_vs_{args.opponent}"
    return args


def play(agent: str, learner: Policy, opponent: Policy, seed: int):
    """Run a match with the learner on the correct side."""
    if agent == PREDATOR:
        return run_match(learner, opponent, episodes=EVAL_EPISODES, seed=seed)
    return run_match(opponent, learner, episodes=EVAL_EPISODES, seed=seed)


def describe(agent: str, result) -> str:
    avg = result.mean_catch_steps
    avg_text = f"{avg:.1f}" if avg is not None else "-"
    if agent == PREDATOR:
        return f"catch rate {result.catch_rate:6.1%} | avg catch step {avg_text}"
    return f"survival rate {1 - result.catch_rate:6.1%} | avg catch step {avg_text}"


def evaluate(model: PPO, agent: str, opponent_name: str) -> None:
    """Compare the trained agent with the scripted baseline, same opponent and starts."""

    def trained(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    baseline_name, baseline_policy = BASELINE[agent]

    trained_result = play(agent, trained, OPPONENTS[opponent_name](), EVAL_SEED)
    baseline_result = play(agent, baseline_policy, OPPONENTS[opponent_name](), EVAL_SEED)

    print(f"\n{agent} vs {opponent_name}, {EVAL_EPISODES} episodes:")
    print(f"  trained {agent:<9} {describe(agent, trained_result)}")
    print(f"  {baseline_name:<17} {describe(agent, baseline_result)}")


def main() -> None:
    args = parse_args()
    run_dir = MODEL_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.agent} vs {args.opponent} -> {run_dir}")

    def make_env() -> SingleAgentEnv:
        return SingleAgentEnv(agent=args.agent, opponent=OPPONENTS[args.opponent]())

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
        name_prefix=args.agent,
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

    evaluate(model, args.agent, args.opponent)


if __name__ == "__main__":
    main()