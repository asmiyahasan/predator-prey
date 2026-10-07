[![Tests](https://github.com/asmiyahasan/predator-prey/actions/workflows/tests.yml/badge.svg)](https://github.com/asmiyahasan/predator-prey/actions/workflows/tests.yml)

# Predator-Prey

A predator and a prey learn to chase and escape each other through self-play reinforcement learning (PPO, Stable-Baselines3) in a 2D arena with obstacles and line-of-sight sensors. The best models are exported to JSON and played on https://asmiyahasan.github.io.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Usage

```bash
# Train both agents with self-play
uv run python scripts/train_selfplay.py --run-name my_run

# Watch a match (model .zip paths or scripted agents: chaser, fleer, random, stay)
uv run python scripts/watch_model.py --predator models/my_run/predator/gen_010.zip --prey fleer

# Export a model to JSON (checks the export matches the original)
uv run python scripts/export_model.py --model models/my_run/predator/gen_010.zip --out exports/predator.json

# Gate challengers against the current champions, promoting any that win
uv run python scripts/gate.py --challenger-predator exports/predator.json --challenger-prey exports/prey.json --promote

# Publish the champions as a GitHub Release (models-vN)
uv run python scripts/release.py
```

Publishing a release triggers a workflow that opens a pull request on the website repo with the new models. If physics or observations change, re-run `scripts/make_fixtures.py` so the website's TypeScript port can be checked against Python.

## Layout

- `src/predprey/`: environment, physics, sensors, scripted agents, export
- `scripts/`: training, watching, export, gating, release
- `champion/`, `champion.json`: current champion models
- `release/`: release bundles
