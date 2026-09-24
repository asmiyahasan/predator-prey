"""Train the predator against the scripted fleer using Stable-Baselines3 PPO.

This is a sanity check: if SB3 can learn here, the environment is learnable,
and later problems with our own PPO are in our code, not the env.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from predprey.agents import fleer
from predprey.env import PREDATOR
from predprey.evaluation import run_match
from predprey.wrappers import SingleAgentEnv

TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 8
SEED = 0

RUN_NAME = "sb3_predator_vs_fleer_3m"
MODEL_DIR = Path("models")
LOG_DIR = Path("runs")


def make_env() -> SingleAgentEnv:
    return SingleAgentEnv(agent=PREDATOR, opponent=fleer)


def main() -> None:
    # Run several copies of the game side by side to collect experience faster
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)

    model = PPO(
        "MlpPolicy",            # a small fully connected neural network
        vec_env,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=RUN_NAME)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / RUN_NAME
    model.save(model_path)
    print(f"\nSaved model to {model_path}.zip")

    # Wrap the trained network as a Policy so run_match can use it
    def trained_predator(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    result = run_match(trained_predator, fleer, episodes=200, seed=1000)
    print(f"\nTrained predator vs fleer over 200 episodes:")
    print(f"  catch rate:      {result.catch_rate:.1%}")
    print(f"  avg catch step:  {result.mean_catch_steps}")
    print("  (chaser baseline: 100.0%, ~32 steps)")


if __name__ == "__main__":
    main()