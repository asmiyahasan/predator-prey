"""Export a trained model to JSON for the website, and check the export is faithful.

Usage:
    uv run python scripts/export_model.py --model models/selfplay_memory/predator/gen_008.zip
    uv run python scripts/export_model.py --model models/selfplay_memory/prey/gen_008.zip --out exports/prey.json

After writing the file, the script plays real games and checks the exported
network picks exactly the same action as the original SB3 model at every step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from predprey.env import PREDATOR, PREY, GameConfig, PredatorPreyEnv
from predprey.export import choose_action, export_model, load_exported
from predprey.observations import OBS_SIZE

CHECK_EPISODES = 20


def find_run_config(model_path: Path) -> GameConfig:
    """Find the config.json saved next to a training run (one or two folders up)."""
    for folder in (model_path.parent, model_path.parent.parent):
        config_path = folder / "config.json"
        if config_path.exists():
            print(f"Using game config from {config_path}")
            return GameConfig.from_dict(json.loads(config_path.read_text()))
    print("No config.json found, using default game settings")
    return GameConfig()


def guess_agent(model_path: Path) -> str | None:
    """Self-play models live in .../predator/... or .../prey/... folders."""
    for part in reversed(model_path.parts):
        if part in (PREDATOR, PREY):
            return part
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained model to JSON.")
    parser.add_argument("--model", required=True, type=Path, help="path to a saved SB3 model (.zip)")
    parser.add_argument("--agent", choices=[PREDATOR, PREY], default=None,
                        help="which side the model plays (guessed from the path if omitted)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: exports/<run>_<agent>_<file name>.json")
    return parser.parse_args()


def check_export(model: PPO, exported: dict, agent: str, config: GameConfig) -> int:
    """Play games, comparing the two versions' action at every step. Returns steps checked."""
    env = PredatorPreyEnv(config=config)
    rng = np.random.default_rng(0)
    steps = 0

    for episode in range(CHECK_EPISODES):
        observations, _ = env.reset(seed=10_000 + episode)
        while env.agents:
            obs = observations[agent]
            original, _ = model.predict(obs, deterministic=True)
            ours = choose_action(exported["layers"], obs)
            if int(original) != ours:
                raise SystemExit(
                    f"MISMATCH at episode {episode}, step {env.step_count}: "
                    f"SB3 chose {int(original)}, export chose {ours}"
                )
            # The model plays its side; the other side moves randomly for variety
            other = PREY if agent == PREDATOR else PREDATOR
            actions = {agent: ours, other: int(rng.integers(9))}
            observations, _, _, _, _ = env.step(actions)
            steps += 1
    return steps


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"model file not found: {args.model}")

    agent = args.agent or guess_agent(args.model)
    if agent is None:
        raise SystemExit("couldn't tell which side this model plays; pass --agent predator or --agent prey")

    model = PPO.load(args.model, device="cpu")
    if model.observation_space.shape != (OBS_SIZE,):
        raise SystemExit(
            f"model expects observations of shape {model.observation_space.shape}, "
            f"but the current game produces ({OBS_SIZE},). Retrain it first."
        )

    config = find_run_config(args.model)

    run_name = args.model.parent.parent.name if guess_agent(args.model) else args.model.parent.name
    out = args.out or Path("exports") / f"{run_name}_{agent}_{args.model.stem}.json"

    export_model(
        model,
        out,
        agent=agent,
        game_config=config.to_dict(),
        metadata={"source_model": str(args.model), "run": run_name},
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.0f} KB)")

    steps = check_export(model, load_exported(out), agent, config)
    print(f"Check passed: identical actions on all {steps:,} steps of {CHECK_EPISODES} games")


if __name__ == "__main__":
    main()
