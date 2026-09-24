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
import json
from pathlib import Path

from stable_baselines3 import PPO

from predprey.agents import Policy, chaser, fleer, make_random_policy
from predprey.env import PREDATOR, PREY, GameConfig, PredatorPreyEnv
from predprey.observations import OBS_SIZE

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

    # Models trained before obstacles and raycasts expect a different observation size
    if model.observation_space.shape != (OBS_SIZE,):
        raise SystemExit(
            f"'{spec}' expects observations of shape {model.observation_space.shape}, "
            f"but the current game produces ({OBS_SIZE},). It was trained on an older "
            f"version of the environment and needs retraining."
        )

    def trained(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return trained


def find_config(specs: list[str]) -> GameConfig:
    """Use the config.json saved with a model's training run, if there is one.

    Self-play models live at models/<run>/<side>/gen_XXX.zip, with the config
    at models/<run>/config.json, so we look one and two folders up.
    """
    for spec in specs:
        path = Path(spec)
        if spec in SCRIPTED or not path.exists():
            continue
        for folder in (path.parent, path.parent.parent):
            config_path = folder / "config.json"
            if config_path.exists():
                print(f"Using game config from {config_path}")
                return GameConfig.from_dict(json.loads(config_path.read_text()))
    print("No config.json found, using default game settings")
    return GameConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a predator vs a prey.")
    parser.add_argument("--predator", required=True, help="scripted name or model path")
    parser.add_argument("--prey", required=True, help="scripted name or model path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10, help="10 = real time")
    parser.add_argument("--seed", type=int, default=None,
                        help="set to replay the same starting positions")
    parser.add_argument("--show-rays", action="store_true",
                        help="draw each agent's raycasts (or press R in the window)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = {PREDATOR: load_policy(args.predator), PREY: load_policy(args.prey)}

    config = find_config([args.predator, args.prey])
    env = PredatorPreyEnv(render_mode="human", config=config)
    env.renderer.fps = args.fps
    env.renderer.show_rays = args.show_rays
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
