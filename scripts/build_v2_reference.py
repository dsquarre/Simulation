"""Precompute the parameter-aware oracle for crop_planning_v2 seeds (needs numpy; ~20-60 s per seed).

    conda run -n base python scripts/build_v2_reference.py --seeds 12 --workers 6

Writes data/crop_v2_reference.json: per seed, the oracle's final wealth and the moves it played.
The harness reads this file, so it needs neither numpy nor the oracle's runtime. Existing seeds are kept.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.crop_v2 import Game, generate_instance  # noqa: E402
from harness.crop_v2_oracles import ParamAware  # noqa: E402
from harness.crop_v2_task import REFERENCE  # noqa: E402


def solve(seed):
    inst = generate_instance(seed)
    pa, game = ParamAware(inst), Game(inst)
    while not game.done:
        game.step({} if game.must_idle() else pa.act(game.t, game.nitrogen, game.capital))
    return seed, {"pa_wealth": game.capital, "pa_schedule": [e["cells"] for e in game.log]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12, help="build seeds 0..N-1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.nice(10)
    data = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {"seeds": {}}
    data["oracle"] = "ParamAware(scenarios=32, act_scenarios=256, grid=40, pair_step=2)"
    todo = [s for s in range(args.seeds) if str(s) not in data["seeds"]]
    with Pool(args.workers) as pool:
        for seed, entry in pool.imap_unordered(solve, todo):
            data["seeds"][str(seed)] = entry
            print(f"seed {seed}: oracle final wealth {entry['pa_wealth']}", flush=True)
    data["seeds"] = dict(sorted(data["seeds"].items(), key=lambda kv: int(kv[0])))
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(data, indent=1))
    print(f"{len(data['seeds'])} seeds in {REFERENCE}")


if __name__ == "__main__":
    main()
