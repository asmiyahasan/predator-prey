"""Evaluation gate: should a newly trained model replace the one on the website?

The models currently live are the CHAMPIONS. Newly exported models are
CHALLENGERS. Each side is judged separately, on the same fixed set of starting
positions, so results are reproducible and comparisons are fair:

  Predator challenger is promoted if
    1. it catches the champion prey more often than the champion predator does,
       by at least --margin, and
    2. it still catches the scripted fleer at least --baseline-floor of the time.

  Prey challenger is promoted if
    1. it survives the champion predator more often than the champion prey does,
       by at least --margin, and
    2. it still survives the scripted chaser at least --baseline-floor of the time.

All four models must have been trained in the same game (same map and speeds).
If there is no champion yet, a challenger only needs to pass the baseline check.

Usage:
    uv run python scripts/gate.py \
        --challenger-predator exports/new_predator.json \
        --challenger-prey exports/new_prey.json

    # Promote whatever passes, copying it into champion/ and updating champion.json
    uv run python scripts/gate.py --challenger-predator ... --challenger-prey ... --promote

The report is written to reports/gate_<time>.json. The exit code is 0 if at
least one side was promoted and 1 if neither was, so CI can act on it.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from predprey.agents import Policy, chaser, fleer
from predprey.env import GameConfig
from predprey.evaluation import run_match
from predprey.export import load_exported, policy_from_export

CHAMPION_DIR = Path("champion")
CHAMPION_FILE = Path("champion.json")
REPORT_DIR = Path("reports")
EVAL_SEED = 50_000  # different from every seed used in training and other checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether to promote new models.")
    parser.add_argument("--challenger-predator", type=Path, required=True)
    parser.add_argument("--challenger-prey", type=Path, required=True)
    parser.add_argument("--games", type=int, default=300, help="games per matchup")
    parser.add_argument("--margin", type=float, default=0.03,
                        help="how much better a challenger must be (0.03 = 3 percentage points)")
    parser.add_argument("--baseline-floor", type=float, default=0.70,
                        help="minimum win rate against the scripted baseline")
    parser.add_argument("--promote", action="store_true",
                        help="copy winners into champion/ and update champion.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, expected_agent: str) -> tuple[Policy, dict]:
    data = load_exported(path)
    if data["agent"] != expected_agent:
        raise SystemExit(f"{path} is a {data['agent']} model, expected a {expected_agent}")
    return policy_from_export(data), data


def load_champion() -> dict | None:
    if not CHAMPION_FILE.exists():
        return None
    return json.loads(CHAMPION_FILE.read_text())


def main() -> None:
    args = parse_args()

    challenger = {
        "predator": load(args.challenger_predator, "predator"),
        "prey": load(args.challenger_prey, "prey"),
    }
    champion_info = load_champion()
    champion = None
    if champion_info:
        champion = {
            side: load(CHAMPION_DIR / f"{side}.json", side) for side in ("predator", "prey")
        }

    # Everything must come from the same game, or the comparison means nothing
    configs = {json.dumps(data["game_config"], sort_keys=True)
               for _, data in [*challenger.values(), *(champion or {}).values()]}
    if len(configs) != 1:
        raise SystemExit(
            "These models were trained in different games (different map or speeds). "
            "They can't be compared. Delete champion/ and champion.json to start fresh "
            "if the game itself has changed."
        )
    config = GameConfig.from_dict(challenger["predator"][1]["game_config"])

    def catch_rate(predator: Policy, prey: Policy) -> float:
        return run_match(predator, prey, episodes=args.games, seed=EVAL_SEED, config=config).catch_rate

    new_pred, new_prey = challenger["predator"][0], challenger["prey"][0]
    results: dict = {}

    print(f"Playing {args.games} games per matchup...\n")

    # --- Predator ---
    pred = {"vs_fleer": catch_rate(new_pred, fleer)}
    pred["baseline_ok"] = pred["vs_fleer"] >= args.baseline_floor
    if champion:
        old_pred, old_prey = champion["predator"][0], champion["prey"][0]
        pred["challenger_vs_champion_prey"] = catch_rate(new_pred, old_prey)
        pred["champion_vs_champion_prey"] = catch_rate(old_pred, old_prey)
        pred["improvement"] = pred["challenger_vs_champion_prey"] - pred["champion_vs_champion_prey"]
        pred["promote"] = pred["baseline_ok"] and pred["improvement"] >= args.margin
    else:
        pred["promote"] = pred["baseline_ok"]
    results["predator"] = pred

    # --- Prey (survival rate = 1 - catch rate) ---
    prey = {"vs_chaser": 1 - catch_rate(chaser, new_prey)}
    prey["baseline_ok"] = prey["vs_chaser"] >= args.baseline_floor
    if champion:
        prey["challenger_vs_champion_predator"] = 1 - catch_rate(old_pred, new_prey)
        prey["champion_vs_champion_predator"] = 1 - catch_rate(old_pred, old_prey)
        prey["improvement"] = prey["challenger_vs_champion_predator"] - prey["champion_vs_champion_predator"]
        prey["promote"] = prey["baseline_ok"] and prey["improvement"] >= args.margin
    else:
        prey["promote"] = prey["baseline_ok"]
    results["prey"] = prey

    # The very first champion needs both sides, so both must pass together
    if not champion and not (pred["promote"] and prey["promote"]):
        pred["promote"] = prey["promote"] = False

    # --- Print a readable summary ---
    def pct(x: float) -> str:
        return f"{x:6.1%}"

    print(f"{'':<10} {'vs baseline':>12} {'vs champion':>12} {'champion':>10} {'change':>8}   decision")
    print("-" * 72)
    for side, baseline_key, h2h_key, ref_key in [
        ("predator", "vs_fleer", "challenger_vs_champion_prey", "champion_vs_champion_prey"),
        ("prey", "vs_chaser", "challenger_vs_champion_predator", "champion_vs_champion_predator"),
    ]:
        r = results[side]
        h2h = pct(r[h2h_key]) if h2h_key in r else "     -"
        ref = pct(r[ref_key]) if ref_key in r else "     -"
        change = f"{r['improvement']:+.1%}" if "improvement" in r else "     -"
        decision = "PROMOTE" if r["promote"] else ("keep champion" if champion else "rejected")
        if not r["baseline_ok"]:
            decision += " (fails baseline)"
        print(f"{side:<10} {pct(r[baseline_key]):>12} {h2h:>12} {ref:>10} {change:>8}   {decision}")
    if not champion:
        print("\nNo champion yet: challengers only need to pass the baseline check.")

    # --- Save the report ---
    now = datetime.now(timezone.utc)
    report = {
        "evaluated_at": now.isoformat(timespec="seconds"),
        "settings": {
            "games_per_matchup": args.games,
            "margin": args.margin,
            "baseline_floor": args.baseline_floor,
            "eval_seed": EVAL_SEED,
        },
        "game_config": config.to_dict(),
        "challenger": {
            side: {"file": str(path), "sha256": sha256(path),
                   "metadata": challenger[side][1]["metadata"]}
            for side, path in [("predator", args.challenger_predator), ("prey", args.challenger_prey)]
        },
        "champion_version": champion_info["version"] if champion_info else None,
        "results": results,
    }
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"gate_{now.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")

    promoted = [side for side in ("predator", "prey") if results[side]["promote"]]

    # --- Optionally promote ---
    if args.promote and promoted:
        CHAMPION_DIR.mkdir(exist_ok=True)
        new_info = champion_info or {"version": 0, "predator": None, "prey": None, "history": []}
        new_info["version"] += 1
        for side in promoted:
            source = args.challenger_predator if side == "predator" else args.challenger_prey
            shutil.copy(source, CHAMPION_DIR / f"{side}.json")
            new_info[side] = {
                "sha256": sha256(source),
                "source": challenger[side][1]["metadata"].get("source_model"),
                "promoted_in_version": new_info["version"],
                "results": results[side],
            }
        new_info["history"].append({
            "version": new_info["version"],
            "promoted": promoted,
            "at": now.isoformat(timespec="seconds"),
            "report": str(report_path),
        })
        CHAMPION_FILE.write_text(json.dumps(new_info, indent=2))
        print(f"Promoted {', '.join(promoted)} -> champion version {new_info['version']}")
    elif args.promote:
        print("Nothing promoted; champion unchanged.")

    sys.exit(0 if promoted else 1)


if __name__ == "__main__":
    main()
