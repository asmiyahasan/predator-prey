"""Publish the current champions as a GitHub Release.

Each release (models-v1, models-v2, ...) is a bundle with everything the
website needs to run, check and describe the agents:

    predator.json      the champion predator (exported network + test vectors)
    prey.json          the champion prey
    parity.json        recorded games for checking the TypeScript game port
    model_card.json    version, where each model came from, and its gate results
    gate_report.json   the full report from the evaluation that promoted them

Usage:
    uv run python scripts/release.py --dry-run   # build the bundle, don't publish
    uv run python scripts/release.py             # build and publish with the gh CLI

Needs the GitHub CLI (`gh`), logged in with `gh auth login`.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

CHAMPION_DIR = Path("champion")
CHAMPION_FILE = Path("champion.json")
FIXTURES = Path("fixtures/parity.json")
BUNDLE_ROOT = Path("release")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"Can't release: {message}")


def tag_exists(tag: str) -> bool:
    result = subprocess.run(["gh", "release", "view", tag], capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the champions as a GitHub Release.")
    parser.add_argument("--dry-run", action="store_true", help="build the bundle but don't publish")
    args = parser.parse_args()

    if not CHAMPION_FILE.exists():
        fail("no champion.json. Promote models with scripts/gate.py --promote first.")
    champion = json.loads(CHAMPION_FILE.read_text())
    version = champion["version"]
    tag = f"models-v{version}"

    # --- Check everything we're about to publish is what champion.json says it is ---
    models = {}
    for side in ("predator", "prey"):
        path = CHAMPION_DIR / f"{side}.json"
        if not path.exists():
            fail(f"{path} is missing")
        if sha256(path) != champion[side]["sha256"]:
            fail(f"{path} doesn't match the checksum in champion.json (was it changed by hand?)")
        models[side] = json.loads(path.read_text())

    if not FIXTURES.exists():
        fail(f"{FIXTURES} is missing. Run scripts/make_fixtures.py.")
    fixtures = json.loads(FIXTURES.read_text())
    for side, model in models.items():
        if model["game_config"] != fixtures["game_config"]:
            fail(f"the {side} was trained on a different map or speeds than {FIXTURES} records. "
                 "Re-run scripts/make_fixtures.py if the game changed.")

    latest = champion["history"][-1]
    report_path = Path(latest["report"])
    if not report_path.exists():
        fail(f"gate report {report_path} is missing")

    # --- Build the bundle ---
    bundle = BUNDLE_ROOT / tag
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    for side in ("predator", "prey"):
        shutil.copy(CHAMPION_DIR / f"{side}.json", bundle / f"{side}.json")
    shutil.copy(FIXTURES, bundle / "parity.json")
    shutil.copy(report_path, bundle / "gate_report.json")

    card = {
        "version": version,
        "tag": tag,
        "released_from_history": latest,
        "game_config": fixtures["game_config"],
        "agents": {
            side: {
                "source_model": champion[side]["source"],
                "promoted_in_version": champion[side]["promoted_in_version"],
                "trained_steps": models[side]["metadata"].get("trained_steps"),
                "sha256": champion[side]["sha256"],
                "gate_results": champion[side]["results"],
            }
            for side in ("predator", "prey")
        },
    }
    (bundle / "model_card.json").write_text(json.dumps(card, indent=2))

    # --- Human-readable release notes ---
    def describe(side: str) -> str:
        info = champion[side]
        r = info["results"]
        baseline = (f"catches the scripted fleer {r['vs_fleer']:.1%}" if side == "predator"
                    else f"escapes the scripted chaser {r['vs_chaser']:.1%}")
        line = (f"- **{side.capitalize()}**: `{info['source']}` "
                f"(champion since v{info['promoted_in_version']}); {baseline}")
        if "improvement" in r:
            line += f"; {r['improvement']:+.1%} vs the previous champion"
        return line

    notes = "\n".join([
        f"Champion models, version {version}.",
        "",
        describe("predator"),
        describe("prey"),
        "",
        f"Promoted this version: {', '.join(latest['promoted'])}.",
        "Files: `predator.json`, `prey.json` (networks), `parity.json` (game parity fixtures), "
        "`model_card.json`, `gate_report.json`.",
    ])
    (bundle / "NOTES.md").write_text(notes)

    print(f"Built {bundle}/")
    for f in sorted(bundle.iterdir()):
        print(f"  {f.name:<18} {f.stat().st_size / 1024:6.0f} KB")
    print("\n" + notes + "\n")

    if args.dry_run:
        print("Dry run: nothing published.")
        return

    # --- Publish ---
    if shutil.which("gh") is None:
        fail("the GitHub CLI (gh) isn't installed. On a Mac: brew install gh, then gh auth login")
    if tag_exists(tag):
        fail(f"release {tag} already exists. Promote new models first (that bumps the version).")

    assets = [str(bundle / name) for name in
              ("predator.json", "prey.json", "parity.json", "model_card.json", "gate_report.json")]
    subprocess.run(
        ["gh", "release", "create", tag, *assets,
         "--title", f"Models v{version}", "--notes-file", str(bundle / "NOTES.md")],
        check=True,
    )
    print(f"Published {tag}")


if __name__ == "__main__":
    main()