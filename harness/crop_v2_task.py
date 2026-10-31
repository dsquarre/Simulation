"""crop_planning_v2 as a harness task: the v2 game (crop_v2.py) played through the plant_season tool.

Score = (final wealth - starting capital) / (oracle's final wealth - starting capital), so idling scores 0
and the parameter-aware oracle scores 1. The oracle needs numpy and ~20-60 s per farm, so its wealth and
moves are precomputed per seed by scripts/build_v2_reference.py into data/crop_v2_reference.json.
Raw final wealth is reported too (the reward for RL training needs no oracle).
"""
import json
from pathlib import Path

from .crop_planning import CASH_CROPS, STAPLE_CROPS, get_yield_multiplier
from .crop_v2 import Game, generate_instance, season_crops
from .tasks import TaskInstance

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "crop_v2_reference.json"

SYSTEM_PROMPT = (
    "You are a farm planning agent managing a farm over 10 seasons. Each season, choose how many cells to "
    "plant with each allowed crop and call `plant_season`. Prices are revealed only after each harvest. "
    "Your goal is the highest possible capital after season 10. You may use `run_python` to calculate "
    "before you commit. After season 10, call `finish`."
)


def load_reference(seed):
    if REFERENCE.exists():
        entry = json.loads(REFERENCE.read_text())["seeds"].get(str(seed))
        if entry:
            return entry
    raise KeyError(f"crop_planning_v2 has no oracle reference for seed {seed}. Add it with: "
                   f"conda run -n base python scripts/build_v2_reference.py --seeds {seed + 1}")


def role(c):
    return "Cash Crop" if c in CASH_CROPS else ("Staple" if c in STAPLE_CROPS else "Pulse/Oilseed")


def crop_table(inst, crops):
    rows = []
    for c in crops:
        d = inst["nitrogen_deltas"][c]
        rows.append(f"  - {c:<10} [{role(c):<13}]: water={inst['waters'][c]:>2}, cost={inst['costs'][c]:>2}, "
                    f"nitrogen={'+' if d > 0 else ''}{d}/cell")
    return "\n".join(rows)


def price_line(prices):
    return ", ".join(f"{c}: {p}" for c, p in prices.items())


def instructions(inst):
    history = "\n".join(f"  Season {h['season']}: {price_line(h['prices'])}" for h in inst["history"])
    return f"""Crop Rotation with Compounding Capital (10 seasons):

FARM:
- Land: {inst['land_cells']} cells. Seasons alternate Kharif (odd: 1, 3, 5, 7, 9) and Rabi (even: 2, 4, 6, 8, 10).
- Only Kharif crops can be planted in Kharif seasons, and only Rabi crops in Rabi seasons.

CROPS:
* Kharif (odd seasons):
{crop_table(inst, season_crops(1))}
* Rabi (even seasons):
{crop_table(inst, season_crops(2))}

HARVEST:
- Revenue per cell = round(that season's market price x yield multiplier). Profit = revenue - planting cost.
- Market prices are revealed only after the harvest. They vary from season to season.

SOIL NITROGEN:
- Nitrogen S is in [0, 100] and starts at {inst['initial_nitrogen']}.
- Yield multiplier Y(S) = clamp(0.4 + 0.6 x S / 50, 0.4, 1.6); Y(50) = 1.0, Y(100) = 1.6.
- Each planted cell changes nitrogen by the crop's value above; each idle cell adds +1.
- If nitrogen reaches 0 the soil is exhausted: the next season must be completely idle (`{{}}`).

CAPITAL:
- You start with {inst['start_capital']}. Planting costs are paid from capital; revenue is added after the harvest.
- Capital compounds: next season's capital = this season's capital + this season's profit. There is no refill.
- Ruin: if at the start of a season your capital is below the cost of one cell of that season's cheapest crop,
  the farm is ruined and every remaining season must be idle.

WATER:
- Each season has its own water quota, announced at the start of that season.

PRICE HISTORY (the two seasons before season 1):
{history}

SCORE: your capital after season 10.

{status(inst, Game(inst))}"""


def status(inst, game):
    t = game.t
    name = "Kharif" if t % 2 == 1 else "Rabi"
    lines = [f"SEASON {t} ({name}) STATUS:",
             f"- Capital: {game.capital}",
             f"- Soil nitrogen: {game.nitrogen} / 100 (yield multiplier {get_yield_multiplier(game.nitrogen)}x)",
             f"- Water quota this season: {inst['water_budgets'][t - 1]}",
             f"- Land: {inst['land_cells']} cells",
             f"- Allowed crops: {', '.join(season_crops(t))}"]
    if game.must_idle():
        why = "RUINED: capital is below the cheapest crop's cost" if game.ruined else "SOIL EXHAUSTED"
        lines.append(f"- {why}. You must submit plant_season(allocation='{{}}').")
    lines.append(f"Call `plant_season(allocation='{{\"crop\": cells, ...}}')` to plant season {t}.")
    return "\n".join(lines)


class CropV2Task(TaskInstance):
    def __init__(self, seed):
        self.inst = generate_instance(seed)
        self.ref = load_reference(seed)
        super().__init__(task_id="crop_planning_v2", title=f"Crop planning v2 (seed {seed})",
                         difficulty="hard", instructions=instructions(self.inst), files={}, cases=[],
                         reference={}, system_prompt=SYSTEM_PROMPT,
                         closing="Plant Season 1 now by calling `plant_season`.")
        self.game = Game(self.inst)

    # used by the human mode in agents.MockClient
    @property
    def current_season(self):
        return self.game.t

    @property
    def is_finished(self):
        return self.game.done

    @property
    def hint_schedule(self):
        return self.ref["pa_schedule"]

    def oracle_actions(self):
        return [("plant_season", {"allocation": json.dumps(a)}) for a in self.ref["pa_schedule"]]

    def score_played(self):
        start, final, ref = self.inst["start_capital"], self.game.capital, self.ref["pa_wealth"]
        return {"final_wealth": final, "oracle_wealth": ref,
                "total": round(max(0.0, (final - start) / (ref - start)), 4)}

    def handle_tool(self, tool_name, args, env):
        if tool_name != "plant_season":
            return False, "", False
        game = self.game
        if game.done:
            return True, "error: all 10 seasons are finished. Call `finish`.", True
        raw = args.get("allocation", "")
        try:
            alloc = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            return True, f"error: allocation must be a JSON object like '{{\"rice\": 10}}' ({exc.msg})", True
        if not isinstance(alloc, dict):
            return True, "error: allocation must be a JSON object like '{\"rice\": 10}'", True
        problem = game.check(alloc)
        if problem:
            return True, f"error: {problem}", True

        t, before = game.t, game.capital
        prices = game.step(alloc)
        entry = game.log[-1]
        name = "Kharif" if t % 2 == 1 else "Rabi"
        msg = [f"--- Season {t} ({name}) harvest ---",
               f"* Planted: {json.dumps(entry['cells']) if entry['cells'] else 'nothing (idle)'}",
               f"* Profit: {entry['profit']}   Capital: {before} -> {game.capital}",
               f"* Soil nitrogen: {entry['nitrogen_before']} -> {game.nitrogen}",
               f"* Season {t} market prices: {price_line(prices)}"]
        if game.done:
            msg.append(f"\nAll 10 seasons are done. Final capital: {game.capital}. Call `finish`.")
        else:
            msg.append("\n" + status(self.inst, game))
        return True, "\n".join(msg), False


def crop_v2_task(seed):
    return CropV2Task(seed)
