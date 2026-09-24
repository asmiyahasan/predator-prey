"""Evaluate scripted agents against each other and print a results table."""

from predprey.agents import chaser, fleer, make_random_policy
from predprey.evaluation import run_match

EPISODES = 200
SEED = 0


def main() -> None:
    matchups = {
        "random vs random": (make_random_policy(seed=1), make_random_policy(seed=2)),
        "chaser vs random": (chaser, make_random_policy(seed=2)),
        "random vs fleer": (make_random_policy(seed=1), fleer),
        "chaser vs fleer": (chaser, fleer),
    }

    print(f"{EPISODES} episodes per matchup (predator vs prey)\n")
    print(f"{'matchup':<20} {'catch rate':>11} {'avg catch step':>15}")
    print("-" * 48)

    for name, (predator, prey) in matchups.items():
        result = run_match(predator, prey, episodes=EPISODES, seed=SEED)
        avg = result.mean_catch_steps
        avg_text = f"{avg:.1f}" if avg is not None else "-"
        print(f"{name:<20} {result.catch_rate:>10.1%} {avg_text:>15}")


if __name__ == "__main__":
    main()