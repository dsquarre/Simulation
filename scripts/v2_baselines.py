"""Baselines for crop planning v2: how simple strategies and the oracles score. No language model.

    conda run -n base python scripts/v2_baselines.py --seeds 200 --workers 6
    conda run -n base python scripts/v2_baselines.py --seeds 1 --workers 1     # timing check

Score of a game = final wealth / parameter-aware oracle's final wealth on the same farm and prices.
Every strategy plays through crop_v2.Game, so the game rules are enforced for the oracles too.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.crop_planning import CASH_CROPS, get_yield_multiplier  # noqa: E402
from harness.crop_v2 import Game, best_allocation, generate_instance, season_crops  # noqa: E402
from harness.crop_v2_oracles import ParamAware, candidate_allocations, clairvoyant  # noqa: E402

HEURISTICS = ["idle", "random", "greedy", "cautious"]


def last_prices(inst, game):
    """Most recent prices the player has seen: season t-1, or history season 0 before season 1."""
    return inst["prices"][game.t - 2] if game.t > 1 else inst["history"][-1]["prices"]


def greedy_choice(inst, game, allowed):
    crops = [c for c in season_crops(game.t) if c in allowed]
    y = get_yield_multiplier(game.nitrogen)
    seen = last_prices(inst, game)
    values = {c: int(round(seen[c] * y)) - inst["costs"][c] for c in crops}
    return best_allocation(values, crops, inst["land_cells"], inst["water_budgets"][game.t - 1],
                           game.capital, inst["waters"], inst["costs"])


def play(inst, choose):
    game, forced = Game(inst), []
    while not game.done:
        forced.append(game.must_idle())
        game.step({} if forced[-1] else choose(game))
    for entry, f in zip(game.log, forced):
        entry["forced_idle"] = f
    return {"wealth": game.capital, "ruined": game.ruined, "log": game.log}


def run_seed(seed):
    inst = generate_instance(seed)
    out = {"seed": seed, "events": inst["events"]}
    rng = random.Random(f"random-strategy-{seed}")

    def random_choice(game):
        options = [a for a in candidate_allocations(inst, game.t, pair_step=1)
                   if sum(n * inst["costs"][c] for c, n in a.items()) <= game.capital]
        return rng.choice(options)

    strategies = {
        "idle": lambda g: {},
        "random": random_choice,
        "greedy": lambda g: greedy_choice(inst, g, season_crops(g.t)),
        "cautious": lambda g: greedy_choice(inst, g, [c for c in season_crops(g.t) if c not in CASH_CROPS]),
    }
    for name, choose in strategies.items():
        r = play(inst, choose)
        out[name] = {"wealth": r["wealth"], "ruined": r["ruined"]}

    start = time.monotonic()
    cv_wealth, schedule = clairvoyant(inst)
    replay = play(inst, lambda g: schedule[g.t - 1])
    assert abs(replay["wealth"] - cv_wealth) < 1e-6, (seed, replay["wealth"], cv_wealth)
    out["clairvoyant"] = {"wealth": replay["wealth"], "ruined": replay["ruined"],
                          "seconds": round(time.monotonic() - start, 2)}

    start = time.monotonic()
    pa = ParamAware(inst)
    r = play(inst, lambda g: pa.act(g.t, g.nitrogen, g.capital))
    out["param_aware"] = {"wealth": r["wealth"], "ruined": r["ruined"],
                          "seconds": round(time.monotonic() - start, 2)}
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report(rows):
    n = len(rows)
    W = {s: [r[s]["wealth"] for r in rows] for s in HEURISTICS + ["clairvoyant", "param_aware"]}
    pa = W["param_aware"]
    print(f"\nseeds: {n}   score = W / W_param_aware on the same farm and prices\n")
    print(f"{'strategy':<13}{'mean W':>10}{'sd':>10}{'p5':>9}{'p50':>9}{'p95':>9}{'ruin %':>8}{'mean score':>12}")
    for s, ws in W.items():
        ruin = 100 * sum(r[s]["ruined"] for r in rows) / n
        ratio = statistics.mean(w / p for w, p in zip(ws, pa))
        print(f"{s:<13}{statistics.mean(ws):>10.0f}{statistics.pstdev(ws):>10.0f}{pct(ws, .05):>9.0f}"
              f"{pct(ws, .5):>9.0f}{pct(ws, .95):>9.0f}{ruin:>8.1f}{ratio:>12.3f}")
    print("\nparam-aware oracle settings: 32 sampled seasons per backward step, 256 per real decision, "
          "40-point wealth grid, nitrogen 0-100 exact; backward actions = idle, singles, pairs in steps of 2; "
          "real decisions add pairs in steps of 1 and 9 shadow-priced knapsack mixes")
    print("random strategy: uniform over idle/single/pair allocations that are affordable "
          "(not over every feasible allocation)")
    secs = [r["param_aware"]["seconds"] for r in rows]
    print(f"\nparam-aware oracle runtime per seed: mean {statistics.mean(secs):.1f}s, max {max(secs):.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "v2_baselines.json"))
    args = ap.parse_args()
    os.nice(10)
    start = time.monotonic()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(run_seed, range(args.start, args.start + args.seeds)), 1):
                rows.append(row)
                if i % 20 == 0:
                    print(f"  {i}/{args.seeds} seeds, {time.monotonic() - start:.0f}s", flush=True)
    else:
        rows = [run_seed(s) for s in range(args.start, args.start + args.seeds)]
    rows.sort(key=lambda r: r["seed"])
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    report(rows)
    print(f"\ntotal {time.monotonic() - start:.0f}s, rows in {args.out}")


if __name__ == "__main__":
    main()
