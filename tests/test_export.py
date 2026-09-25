import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from predprey.agents import fleer
from predprey.env import PREDATOR, PREY, GameConfig, PredatorPreyEnv
from predprey.export import (
    choose_action,
    export_model,
    extract_actor_layers,
    forward,
    load_exported,
)
from predprey.observations import OBS_SIZE
from predprey.physics import NUM_ACTIONS
from predprey.wrappers import SingleAgentEnv


@pytest.fixture(scope="module")
def model():
    """A small PPO model with randomly initialised weights (no training needed)."""
    env = SingleAgentEnv(agent=PREDATOR, opponent=fleer)
    return PPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=0, device="cpu")


@pytest.fixture(scope="module")
def real_observations():
    """Observations from real games, so the test covers realistic inputs."""
    env = PredatorPreyEnv()
    rng = np.random.default_rng(0)
    collected = []
    for episode in range(5):
        observations, _ = env.reset(seed=episode)
        while env.agents:
            collected.append(observations[PREDATOR])
            collected.append(observations[PREY])
            observations, _, _, _, _ = env.step(
                {agent: int(rng.integers(NUM_ACTIONS)) for agent in env.agents}
            )
    return collected


def torch_logits(model: PPO, obs: np.ndarray) -> np.ndarray:
    """The original network's logits, computed by SB3/PyTorch itself."""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs[None, :])
        latent = model.policy.mlp_extractor.forward_actor(obs_tensor)
        return model.policy.action_net(latent)[0].numpy()


def test_layer_shapes_and_activations(model):
    layers = extract_actor_layers(model)
    shapes = [np.shape(layer["weights"]) for layer in layers]
    assert shapes == [(64, OBS_SIZE), (64, 64), (NUM_ACTIONS, 64)]
    assert [layer["activation"] for layer in layers] == ["tanh", "tanh", "none"]


def test_logits_match_pytorch(model, real_observations):
    layers = extract_actor_layers(model)
    for obs in real_observations[:500]:
        np.testing.assert_allclose(forward(layers, obs), torch_logits(model, obs), atol=1e-5)


def test_actions_match_sb3_predict(model, real_observations):
    layers = extract_actor_layers(model)
    for obs in real_observations:
        expected, _ = model.predict(obs, deterministic=True)
        assert choose_action(layers, obs) == int(expected)


def test_export_file_round_trip(model, tmp_path, real_observations):
    out = tmp_path / "predator.json"
    config = GameConfig()
    export_model(model, out, agent=PREDATOR, game_config=config.to_dict(),
                 metadata={"run": "test"})

    data = load_exported(out)
    assert data["agent"] == PREDATOR
    assert data["obs_size"] == OBS_SIZE
    assert data["num_actions"] == NUM_ACTIONS
    assert data["metadata"]["run"] == "test"
    assert GameConfig.from_dict(data["game_config"]) == config

    # Loaded from disk, it still picks the same actions as the live model
    for obs in real_observations[:200]:
        expected, _ = model.predict(obs, deterministic=True)
        assert choose_action(data["layers"], obs) == int(expected)


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format_version": 999}))
    with pytest.raises(ValueError):
        load_exported(path)
