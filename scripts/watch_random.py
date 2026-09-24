"""Watch a few episodes of predator vs prey with random actions."""

import numpy as np

from predprey.env import PredatorPreyEnv
from predprey.physics import NUM_ACTIONS

NUM_EPISODES = 5


def main() -> None:
    env = PredatorPreyEnv(render_mode="human")
    rng = np.random.default_rng()

    for episode in range(NUM_EPISODES):
        env.reset()
        env.render()  # show the starting positions

        while env.agents and env.renderer.is_open:
            actions = {agent: rng.integers(NUM_ACTIONS) for agent in env.agents}
            env.step(actions)
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