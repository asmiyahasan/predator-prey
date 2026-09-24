"""Watch a trained agent (predator or prey) play against a scripted opponent.

Usage:
    uv run python scripts/watch_model.py --model models/sb3_prey_vs_chaser/final.zip --agent prey
    uv run python scripts/watch_model.py --model models/sb3_predator_vs_fleer_3m.zip --agent predator
    uv run python scripts/watch_model.py --model models/sb3_prey_vs_chaser/final.zip --agent prey --opponent random --fps 20
"""

import argparse

from stable_baselines3 import PPO

from predprey.agents import chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY, PredatorPreyEnv

OPPONENTS = {
    "chaser": lambda: chaser,
    "fleer": lambda: fleer,
    "random": lambda: make_random_policy(),
    "stay": lambda: (lambda obs: 0),
}
DEFAULT_OPPONENT = {PREDATOR: "fleer", PREY: "chaser"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained agent.")
    parser.add_argument("--model", required=True, help="path to a saved SB3 model (.zip)")
    parser.add_argument("--agent", choices=[PREDATOR, PREY], required=True,
                        help="which side the model controls")
    parser.add_argument("--opponent", choices=OPPONENTS, default=None,
                        help="default: fleer for predator, chaser for prey")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10, help="10 = real time")
    args = parser.parse_args()
    if args.opponent is None:
        args.opponent = DEFAULT_OPPONENT[args.agent]
    return args


def main() -> None:
    args = parse_args()
    opponent_name = PREY if args.agent == PREDATOR else PREDATOR

    model = PPO.load(args.model)

    def trained(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    policies = {args.agent: trained, opponent_name: OPPONENTS[args.opponent]()}

    env = PredatorPreyEnv(render_mode="human")
    env.renderer.fps = args.fps
    print(f"Trained {args.agent} vs scripted {args.opponent}")

    for episode in range(args.episodes):
        observations, _ = env.reset()
        env.render()

        while env.agents and env.renderer.is_open:
            actions = {name: policy(observations[name]) for name, policy in policies.items()}
            observations, _, _, _, _ = env.step(actions)
            env.render()

        if not env.renderer.is_open:
            print("Window closed, stopping.")
            break

        if env.caught:
            print(f"Episode {episode + 1}: caught at step {env.step_count}")
        else:
            print(f"Episode {episode + 1}: prey survived all {env.step_count} steps")

    env.close()


if __name__ == "__main__":
    main()