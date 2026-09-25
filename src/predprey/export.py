"""Export a trained SB3 policy to plain JSON, so the website can run it.

Only the actor (the part that picks actions) is exported. The critic is only
needed during training. For SB3's default MlpPolicy the actor is:

    observation (28) -> Linear -> tanh -> Linear -> tanh -> Linear -> 9 logits

and the chosen action is the index of the largest logit.

The JSON file contains everything the website needs to run the agent:

    {
      "format_version": 1,
      "agent": "predator",
      "obs_size": 28,
      "num_actions": 9,
      "layers": [
        {"weights": [[...], ...], "bias": [...], "activation": "tanh"},
        ...
        {"weights": [[...], ...], "bias": [...], "activation": "none"}
      ],
      "game_config": {...},       # the game this model was trained in
      "metadata": {...}           # where it came from, when, etc.
    }

`weights` has shape [out_features][in_features], so a layer computes
    output[i] = bias[i] + sum_j weights[i][j] * input[j]

`forward` below is the reference implementation. The TypeScript version on the
website must produce the same actions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

FORMAT_VERSION = 1

_ACTIVATIONS = {
    torch.nn.Tanh: "tanh",
    torch.nn.ReLU: "relu",
}


def extract_actor_layers(model: PPO) -> list[dict]:
    """Pull the actor's weights out of an SB3 PPO model as plain Python lists."""
    policy = model.policy
    modules = list(policy.mlp_extractor.policy_net) + [policy.action_net]

    layers: list[dict] = []
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            layers.append({
                "weights": module.weight.detach().cpu().numpy().astype(np.float32).tolist(),
                "bias": module.bias.detach().cpu().numpy().astype(np.float32).tolist(),
                "activation": "none",
            })
        elif type(module) in _ACTIVATIONS:
            if not layers:
                raise ValueError("activation found before any Linear layer")
            layers[-1]["activation"] = _ACTIVATIONS[type(module)]
        else:
            raise ValueError(f"don't know how to export layer type {type(module).__name__}")
    return layers


def forward(layers: list[dict], obs: np.ndarray) -> np.ndarray:
    """Run the exported network: observation in, action logits out."""
    x = np.asarray(obs, dtype=np.float32)
    for layer in layers:
        weights = np.asarray(layer["weights"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        x = weights @ x + bias
        if layer["activation"] == "tanh":
            x = np.tanh(x)
        elif layer["activation"] == "relu":
            x = np.maximum(x, 0.0)
    return x


def choose_action(layers: list[dict], obs: np.ndarray) -> int:
    """The deterministic action: the one with the largest logit."""
    return int(np.argmax(forward(layers, obs)))


def export_model(
    model: PPO,
    out_path: Path,
    agent: str,
    game_config: dict,
    metadata: dict | None = None,
) -> dict:
    """Write the actor and its context to a JSON file, and return the data."""
    layers = extract_actor_layers(model)
    data = {
        "format_version": FORMAT_VERSION,
        "agent": agent,
        "obs_size": len(layers[0]["weights"][0]),
        "num_actions": len(layers[-1]["bias"]),
        "layers": layers,
        "game_config": game_config,
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_steps": int(model.num_timesteps),
            **(metadata or {}),
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    return data


def load_exported(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported export format: {data.get('format_version')}")
    return data
