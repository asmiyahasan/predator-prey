"""Watch any predator against any prey: trained models or scripted agents.

Each side takes either a scripted agent name (chaser, fleer, random, stay)
or a path to a saved SB3 model (.zip).

Usage:
    # Final self-play predator vs final self-play prey
    uv run python scripts/watch_model.py \
        --predator models/selfplay/predator/gen_010.zip \
        --prey models/selfplay/prey/gen_010.zip

    # Early prey vs late predator, at double speed
    uv run python scripts/watch_model.py \
        --predator models/selfplay/predator/gen_010.zip \
        --prey models/selfplay/prey/gen_002.zip --fps 20

    # Trained prey vs the scripted chaser
    uv run python scripts/watch_model.py --predator chaser --prey models/selfplay/prey/gen_010.zip
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from predprey.agents import Policy, chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY, PredatorPreyEnv

SCRIPTED = {
    "chaser": lambda: chaser,
    "fleer": lambda: fleer,
    "random": lambda: make_random_policy(),
    "stay": lambda: (lambda obs: 0),
}


def load_policy(spec: str) -> Policy:
    """Turn a scripted agent name or a model path into a policy."""
    if spec in SCRIPTED:
        return SCRIPTED[spec]()

    path = Path(spec)
    if not path.exists():
        raise SystemExit(
            f"'{spec}' is neither a scripted agent ({', '.join(SCRIPTED)}) "
            f"nor an existing model file"
        )

    model = PPO.load(path, device="cpu")

    def trained(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return trained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a predator vs a prey.")
    parser.add_argument("--predator", required=True, help="scripted name or model path")
    parser.add_argument("--prey", required=True, help="scripted name or model path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10, help="10 = real time")
    parser.add_argument("--seed", type=int, default=None,
                        help="set to replay the same starting positions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = {PREDATOR: load_policy(args.predator), PREY: load_policy(args.prey)}

    env = PredatorPreyEnv(render_mode="human")
    env.renderer.fps = args.fps
    print(f"Predator: {args.predator}\nPrey:     {args.prey}\n")

    for episode in range(args.episodes):
        seed = None if args.seed is None else args.seed + episode
        observations, _ = env.reset(seed=seed)
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