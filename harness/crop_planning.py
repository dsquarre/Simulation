"""Crop Rotation & Market POMDP Environment (NP-Hard Interactive Multi-Step Challenge).

Server-Side Interactive POMDP with Dynamic Soil Nitrogen & Bipartite Kharif/Rabi Seasons:
  - Farm with A discrete land cells (A in [20, 28] randomly seeded).
  - 12 crops partitioned into seasons:
      Kharif (odd seasons: 1, 3, 5, 7, 9): rice, maize, cotton, soybean, groundnut, sugarcane
      Rabi (even seasons: 2, 4, 6, 8, 10):  wheat, mustard, chickpea, barley, potato, lentil
  - Categorized Economic Profiles:
      1. Cash Crops (cotton, sugarcane, potato):
         High cost, high water, massive profit upside (+40 to +110) or steep loss (-30),
         and heavy soil nitrogen depletion (-4 to -5 per cell).
      2. Staples (rice, maize, wheat, barley):
         Moderate cost and water, steady food demand (-15 to +25), mild nitrogen depletion (-1 per cell).
      3. Pulses & Oilseeds (soybean, groundnut, mustard, chickpea, lentil):
         Low cost, low water, tight margin (-10 to +15), actively replenishes soil nitrogen (+2 to +3 per cell).
  - Soil Nitrogen Scalar S_t in [0, 100] (starts at S_1 = 50):
      Each cell of crop c planted changes nitrogen by delta_N[c].
      Idle / fallow land gains +1 nitrogen per idle cell:
        S_{t+1} = clamp(S_t + sum(count[c] * delta_N[c]) + (A - sum(count)), 0, 100).
      CRITICAL RULE: If nitrogen drops to 0, soil is exhausted: no crops can be planted next season!
      The farm must remain completely idle for at least one season to recover nitrogen naturally.
      Yield Multiplier: Y(S_t) = clamp(0.4 + 0.6 * (S_t / 50.0), 0.4, 1.6) if S_t > 0 else 0.0.
      Harvest revenue = sum( count[c] * int(round(P_t[c] * Y(S_t))) ).
  - Native plant_season Tool (Zero files in sandbox):
      No farm.py, no secret price files, and no solution.json in the sandbox workspace.
      Turn-by-turn interaction:
        Agent calls `plant_season(allocation='{"cotton": 10, "soybean": 10}')` or `plant_season(allocation='{}')` to idle.
        Server validates land, water, budget, computes harvest revenue with yield multiplier,
        compounds 100% of profit/loss into next season's budget, updates soil nitrogen,
        reveals that season's true market prices, and returns Season t+1's state directly in the tool response!
  - Initial Prior History:
      Only Season -1 and Season 0 historical prices are provided up front.
      No future prices or weather forecasts are revealed ahead of time.
  - Verifier & Scoring:
      100% programmatic verification comparing agent profit against global DP optimum.
"""
import copy
import hashlib
import json
import random
import textwrap
import time

from .tasks import Case, TaskInstance, _case

KHARIF_CROPS = ["rice", "maize", "cotton", "soybean", "groundnut", "sugarcane"]
RABI_CROPS = ["wheat", "mustard", "chickpea", "barley", "potato", "lentil"]
ALL_CROPS = KHARIF_CROPS + RABI_CROPS

CASH_CROPS = {"cotton", "sugarcane", "potato"}
STAPLE_CROPS = {"rice", "maize", "wheat", "barley"}
PULSE_CROPS = {"soybean", "groundnut", "mustard", "chickpea", "lentil"}

NITROGEN_DELTA = {
    # Cash crops: heavy depletion (-4 to -5)
    "cotton": -5,
    "sugarcane": -5,
    "potato": -4,
    # Staples: mild depletion (-1)
    "rice": -1,
    "maize": -1,
    "wheat": -1,
    "barley": -1,
    # Pulses & Oilseeds: nitrogen fixing (+2 to +3)
    "soybean": 3,
    "groundnut": 2,
    "mustard": 2,
    "chickpea": 3,
    "lentil": 2,
}


SYSTEM_PROMPT = (
    "You are a farm planning agent managing land over 10 seasons. Each season, choose how many cells "
    "to plant with each allowed crop and call `plant_season`. Prices are revealed only after each "
    "harvest. You may use `run_python` to calculate before you commit. After season 10, call `finish`."
)


def get_yield_multiplier(nitrogen: float) -> float:
    """Compute crop yield multiplier based on soil nitrogen index in [0, 100]."""
    if nitrogen <= 0:
        return 0.0
    return max(0.4, min(1.6, round(0.4 + 0.6 * (nitrogen / 50.0), 2)))


def fast_knapsack(crops, A, W, B, eff_profits, costs, waters):
    """Solve single season knapsack over active seasonal crops."""
    pos = [c for c in crops if eff_profits.get(c, 0) > 0]
    if not pos:
        return 0, {}
    pos.sort(key=lambda c: eff_profits[c] / (1 + waters[c] + costs[c]), reverse=True)
    best_v = 0
    best_a = {}

    def search(idx, rem_a, rem_w, rem_b, cur_v, cur_a):
        nonlocal best_v, best_a
        if cur_v > best_v:
            best_v = cur_v
            best_a = dict(cur_a)
        if idx >= len(pos) or rem_a <= 0:
            return
        c = pos[idx]
        p = eff_profits[c]
        w = waters[c]
        b = costs[c]
        if cur_v + rem_a * p <= best_v:
            return
        max_take = min(rem_a, rem_w // max(1, w), rem_b // max(1, b))
        for take in range(max_take, -1, -1):
            if take > 0:
                cur_a[c] = take
            search(idx + 1, rem_a - take, rem_w - take * w, rem_b - take * b, cur_v + take * p, cur_a)
            if take > 0:
                del cur_a[c]

    search(0, A, W, B, 0, {})
    return best_v, best_a


def solve_crop_planning_optimal(instance):
    """Global multi-season DP solver with continuous soil nitrogen and budget compounding."""
    T = instance["num_seasons"]
    kharif_crops = instance["kharif_crops"]
    rabi_crops = instance["rabi_crops"]
    A = instance["land_cells"]
    costs = instance["costs"]
    waters = instance["waters"]
    prices = instance["prices"]
    n_deltas = instance["nitrogen_deltas"]
    w_budgets = instance["water_budgets"]
    b_budgets = instance["capital_budgets"]

    # dp state: (nitrogen_level) -> (cumulative_profit, available_budget, history)
    dp = {50: (0, b_budgets[0], [])}

    for t in range(1, T + 1):
        is_kharif = (t % 2 == 1)
        crops = kharif_crops if is_kharif else rabi_crops
        W_t = w_budgets[t - 1]
        P_t = prices[t - 1]

        # Build diverse candidate pool across shadow nitrogen prices
        candidates = [{}]  # Fallow / idle
        seen = {()}

        for y in [0.4, 0.7, 1.0, 1.3, 1.6]:
            for lam in [-12, -6, 0, 6, 12]:
                eff_p = {c: int(round(P_t[c] * y)) - costs[c] + lam * n_deltas[c] for c in crops}
                _, a = fast_knapsack(crops, A, W_t, 5000, eff_p, costs, waters)
                k = tuple(sorted(a.items()))
                if k not in seen:
                    seen.add(k)
                    candidates.append(a)

        # Also add greedy monocultures
        for c in crops:
            max_c = min(A, W_t // max(1, waters[c]))
            a = {c: max_c}
            k = tuple(sorted(a.items()))
            if k not in seen:
                seen.add(k)
                candidates.append(a)

        # Pairwise cash + pulse rotations
        cash_list = [c for c in crops if n_deltas[c] <= -4]
        pulse_list = [c for c in crops if n_deltas[c] >= 2]
        for c_cash in cash_list:
            for c_pulse in pulse_list:
                for frac in [0.3, 0.5, 0.7]:
                    n_cash = int(round(A * frac))
                    w_rem = W_t - n_cash * waters[c_cash]
                    if w_rem < 0:
                        continue
                    n_pulse = min(A - n_cash, w_rem // max(1, waters[c_pulse]))
                    a = {c_cash: n_cash, c_pulse: n_pulse}
                    k = tuple(sorted(a.items()))
                    if k not in seen:
                        seen.add(k)
                        candidates.append(a)

        next_dp = {}
        for cur_N, (profit, cur_b, hist) in dp.items():
            if cur_N <= 0:
                # Soil exhausted: Must idle completely!
                fallow = A
                next_N = min(100, cur_N + fallow)
                next_b = max(0, (b_budgets[t] if t < T else 0))
                bucket = (next_N // 5) * 5
                if bucket not in next_dp or profit > next_dp[bucket][0]:
                    next_dp[bucket] = (profit, next_b, hist + [{}])
                continue

            y = get_yield_multiplier(cur_N)
            for alloc in candidates:
                cur_land = sum(alloc.values())
                cur_water = sum(alloc[c] * waters[c] for c in alloc)
                cur_cost = sum(alloc[c] * costs[c] for c in alloc)

                if cur_land > A or cur_water > W_t or cur_cost > cur_b:
                    continue

                cur_rev = sum(alloc[c] * int(round(P_t[c] * y)) for c in alloc)
                season_p = cur_rev - cur_cost
                fallow = max(0, A - cur_land)
                # Idle land gains +1 nitrogen per cell
                delta_n = sum(alloc[c] * n_deltas[c] for c in alloc) + fallow
                next_N = max(0, min(100, cur_N + delta_n))
                next_b = max(0, (b_budgets[t] if t < T else 0) + season_p)
                new_p = profit + season_p

                bucket = (next_N // 5) * 5
                if bucket not in next_dp or new_p > next_dp[bucket][0]:
                    next_dp[bucket] = (new_p, next_b, hist + [alloc])

        dp = next_dp

    best_profit = -1
    best_schedule = []
    for bucket, (profit, cur_b, hist) in dp.items():
        if profit > best_profit:
            best_profit = profit
            best_schedule = hist

    return best_profit, best_schedule


def hashlib_seed(val):
    import hashlib
    return int(hashlib.sha256(str(val).encode()).hexdigest()[:8], 16)


def generate_crop_planning_instance(seed_val) -> dict:
    """Generate seeded crop planning instance with dynamic nitrogen and market properties."""
    seed_int = int(hashlib_seed(seed_val)) if not isinstance(seed_val, int) else seed_val
    rng = random.Random(seed_int)

    A = rng.randint(20, 28)
    num_seasons = 10

    waters = {}
    costs = {}
    base_prices = {}
    volatilities = {}

    for c in ALL_CROPS:
        is_kharif = c in KHARIF_CROPS
        if c in CASH_CROPS:
            waters[c] = rng.randint(16, 24) if is_kharif else rng.randint(12, 18)
            costs[c] = rng.randint(55, 80)
            # High risk & massive payoff upside: base margin -30 to +80
            base_prices[c] = costs[c] + rng.randint(-30, 80)
            volatilities[c] = rng.choice([0.50, 0.65, 0.80])
        elif c in STAPLE_CROPS:
            waters[c] = rng.randint(12, 20) if is_kharif else rng.randint(8, 14)
            costs[c] = rng.randint(30, 50)
            # Modest margin: -15 to +25
            base_prices[c] = costs[c] + rng.randint(-15, 25)
            volatilities[c] = rng.choice([0.25, 0.35, 0.45])
        else:  # PULSE_CROPS / Oilseeds
            waters[c] = rng.randint(6, 14) if is_kharif else rng.randint(4, 10)
            costs[c] = rng.randint(15, 30)
            # Low cost, tight margin: -10 to +15
            base_prices[c] = costs[c] + rng.randint(-10, 15)
            volatilities[c] = rng.choice([0.15, 0.25, 0.35])

    water_budgets = []
    capital_budgets = []

    for t in range(1, num_seasons + 1):
        is_kharif = (t % 2 == 1)
        w_factor = rng.uniform(0.55, 0.85) if is_kharif else rng.uniform(0.35, 0.65)
        avg_w = sum(waters[c] for c in (KHARIF_CROPS if is_kharif else RABI_CROPS)) / 6.0
        w_budget = int(A * avg_w * w_factor)

        avg_cost = sum(costs[c] for c in (KHARIF_CROPS if is_kharif else RABI_CROPS)) / 6.0
        b_budget = int(A * avg_cost * rng.uniform(0.65, 0.95))

        water_budgets.append(w_budget)
        capital_budgets.append(b_budget)

    # Historical prices for Season -1 and Season 0
    prior_price_history = []
    for past_t in [-1, 0]:
        past_p = {}
        for c in ALL_CROPS:
            fluc = rng.uniform(-volatilities[c], volatilities[c])
            past_p[c] = max(1, int(round(base_prices[c] * (1.0 + fluc))))
        prior_price_history.append({"past_season": past_t, "prices": past_p})

    # Deterministic price trajectory with macroeconomic swings
    prices = []
    for t in range(1, num_seasons + 1):
        p_t = {}
        shock = rng.choice([-0.40, -0.20, 0.0, 0.25, 0.45])
        for c in ALL_CROPS:
            vol = volatilities[c]
            wave = 0.4 * vol * (1.0 if (t % 2 == 1 and c in KHARIF_CROPS) else -0.4)
            # Occasional cash crop boom (+30%)
            boom = 0.30 if (c in CASH_CROPS and rng.random() < 0.30) else 0.0
            fluctuation = wave + shock * vol + boom + rng.uniform(-0.15, 0.15)
            p_val = int(round(base_prices[c] * (1.0 + fluctuation)))
            p_t[c] = max(1, p_val)
        prices.append(p_t)

    inst = {
        "seed": seed_int,
        "num_seasons": num_seasons,
        "land_cells": A,
        "kharif_crops": KHARIF_CROPS,
        "rabi_crops": RABI_CROPS,
        "waters": waters,
        "costs": costs,
        "base_prices": base_prices,
        "nitrogen_deltas": NITROGEN_DELTA,
        "initial_nitrogen": 50,
        "water_budgets": water_budgets,
        "capital_budgets": capital_budgets,
        "prior_price_history": prior_price_history,
        "prices": prices,
    }

    opt_profit, opt_schedule = solve_crop_planning_optimal(inst)
    inst["optimal_profit"] = opt_profit
    inst["optimal_schedule"] = opt_schedule
    return inst


class CropPlanningTaskInstance(TaskInstance):
    """TaskInstance with active server-side state tracking for plant_season tool."""

    def __init__(self, inst, task_id, title, difficulty, instructions, files, cases, reference,
                 system_prompt="", closing=""):
        super().__init__(
            task_id=task_id,
            title=title,
            difficulty=difficulty,
            instructions=instructions,
            files=files,
            cases=cases,
            reference=reference,
            system_prompt=system_prompt,
            closing=closing,
        )
        self.inst = inst
        self.A = self.inst["land_cells"]
        self.T = self.inst["num_seasons"]
        self.current_season = 1
        self.waters = self.inst["waters"]
        self.costs = self.inst["costs"]
        self.prices = self.inst["prices"]
        self.n_deltas = self.inst["nitrogen_deltas"]
        self.w_budgets = list(self.inst["water_budgets"])
        self.b_budgets = list(self.inst["capital_budgets"])
        self.kharif = set(self.inst["kharif_crops"])
        self.rabi = set(self.inst["rabi_crops"])

        self.nitrogen = self.inst["initial_nitrogen"]
        self.cumulative_profit = 0
        self.current_budget = self.b_budgets[0]
        self.log = []
        self.history = list(self.inst["prior_price_history"])
        self.is_finished = False

    def played_solution(self) -> str:
        """solution.json content for the seasons actually played through plant_season."""
        return json.dumps({"seasons": self.log}, indent=2)

    def oracle_actions(self):
        """The precomputed optimal schedule, submitted one season at a time like an agent would."""
        return [("plant_season", {"allocation": json.dumps(alloc)}) for alloc in self.inst["optimal_schedule"]]

    def handle_tool(self, tool_name: str, args: dict, env) -> tuple[bool, str, bool]:
        """Intercept plant_season tool call on the harness server side."""
        if tool_name != "plant_season":
            return False, "", False

        if self.is_finished:
            return True, "error: all seasons have already been completed. Call finish to complete the task.", True

        alloc_raw = args.get("allocation", "")
        if isinstance(alloc_raw, str):
            try:
                alloc = json.loads(alloc_raw)
            except Exception as e:
                return True, f"error: allocation must be valid JSON: {e}", True
        elif isinstance(alloc_raw, dict):
            alloc = alloc_raw
        else:
            return True, "error: allocation must be a JSON object or string", True

        t = self.current_season
        t_idx = t - 1
        is_kharif = (t % 2 == 1)
        active_set = self.kharif if is_kharif else self.rabi
        season_name = "Kharif" if is_kharif else "Rabi"

        # Check soil nitrogen depletion rule
        if self.nitrogen <= 0:
            # Soil is exhausted: no crops can be planted!
            planted_count = sum(v for v in alloc.values() if isinstance(v, int) and v > 0)
            if planted_count > 0:
                return True, f"error: Soil Nitrogen is {self.nitrogen} (Exhausted)! You cannot plant crops this season and must leave all land idle ('{{}}') to allow the soil to regenerate.", True

        cur_land = 0
        cur_water = 0
        cur_cost = 0

        for crop, count in alloc.items():
            if not isinstance(count, int) or count < 0:
                return True, f"error: invalid count for crop '{crop}': {count}", True
            if count == 0:
                continue
            if crop not in active_set:
                return True, f"error: crop '{crop}' is not allowed in {season_name} season {t}. Active crops: {', '.join(sorted(active_set))}", True

            cur_land += count
            cur_water += count * self.waters[crop]
            cur_cost += count * self.costs[crop]

        if cur_land > self.A:
            return True, f"error: land overflow ({cur_land} cells used, max capacity is {self.A})", True
        if cur_water > self.w_budgets[t_idx]:
            return True, f"error: water overflow ({cur_water} units used, budget is {self.w_budgets[t_idx]})", True
        if cur_cost > self.current_budget:
            return True, f"error: capital budget overflow (cost {cur_cost} exceeds working capital {self.current_budget})", True

        # Calculate revenue with current soil nitrogen yield multiplier
        y_mult = get_yield_multiplier(self.nitrogen)
        cur_revenue = sum(count * int(round(self.prices[t_idx][crop] * y_mult)) for crop, count in alloc.items() if count > 0)
        season_profit = cur_revenue - cur_cost
        self.cumulative_profit += season_profit

        # Soil Nitrogen Dynamics: idle land gains +1 per cell
        fallow_cells = max(0, self.A - cur_land)
        delta_n = sum(count * self.n_deltas[crop] for crop, count in alloc.items() if count > 0) + fallow_cells
        old_n = self.nitrogen
        self.nitrogen = max(0, min(100, self.nitrogen + delta_n))
        next_y_mult = get_yield_multiplier(self.nitrogen)

        # Compounding working capital budget
        next_base_budget = self.b_budgets[t] if t < self.T else 0
        self.current_budget = max(0, next_base_budget + season_profit)

        # Reveal this season's true market prices
        revealed_prices = self.prices[t_idx]
        self.history.append({
            "season": t,
            "season_type": season_name,
            "prices": revealed_prices,
        })

        self.log.append({
            "season": t,
            "cells": alloc,
            "profit": season_profit,
            "land_used": cur_land,
            "water_used": cur_water,
            "cost_spent": cur_cost,
            "revenue": cur_revenue,
            "nitrogen_before": old_n,
            "nitrogen_after": self.nitrogen,
            "yield_multiplier": y_mult,
        })

        # Sync state to sandbox solution.json so standard harness verifier can score it
        solution_data = {"seasons": self.log}
        try:
            env._post("/tool", {"name": "write_file", "args": {"path": "solution.json", "content": json.dumps(solution_data, indent=2)}}, env.tool_timeout)
        except Exception:
            pass

        if t == self.T:
            self.is_finished = True
            msg = textwrap.dedent(f"""
                --- Season {t} ({season_name}) Harvest Complete ---
                * Season Profit: {season_profit}
                * Total Cumulative Profit: {self.cumulative_profit}
                * Soil Nitrogen: {old_n} -> {self.nitrogen} (Yield Multiplier Applied: {y_mult}x)
                * Revealed Market Prices: {json.dumps(revealed_prices)}

                All {self.T} seasons have been successfully harvested!
                Call the `finish` tool now to complete your submission.
            """).strip()
            return True, msg, False
        else:
            self.current_season += 1
            next_t = self.current_season
            next_is_kharif = (next_t % 2 == 1)
            next_season_name = "Kharif" if next_is_kharif else "Rabi"
            next_active = sorted(list(self.kharif if next_is_kharif else self.rabi))

            warning_note = ""
            if self.nitrogen <= 0:
                warning_note = "\n* WARNING: Soil Nitrogen has dropped to 0! All crops are locked. You must allocate '{}' (all idle) for Season " + str(next_t) + " to restore soil."

            msg = textwrap.dedent(f"""
                --- Season {t} ({season_name}) Harvest Complete ---
                * Season Profit: {season_profit}
                * Cumulative Profit So Far: {self.cumulative_profit}
                * Soil Nitrogen: {old_n} -> {self.nitrogen} (Yield Multiplier Applied: {y_mult}x)
                * Revealed Season {t} Market Prices: {json.dumps(revealed_prices)}

                >>> ADVANCING TO SEASON {next_t} ({next_season_name}) <<<
                * Total Land: {self.A} cells
                * Available Water Quota: {self.w_budgets[next_t - 1]}
                * Available Capital Budget (Compounded): {self.current_budget}
                * Current Soil Nitrogen: {self.nitrogen}/100 (Upcoming Yield Multiplier: {next_y_mult}x)
                * Allowed Crop Varieties: {', '.join(next_active)}{warning_note}

                Call `plant_season(allocation='...')` to plant Season {next_t}.
            """).strip()
            return True, msg, False


def make_crop_planning_cases(inst):
    """Create verification cases evaluating the agent's multi-season crop plan."""
    A = inst["land_cells"]
    T = inst["num_seasons"]
    waters = inst["waters"]
    costs = inst["costs"]
    prices = inst["prices"]
    n_deltas = inst["nitrogen_deltas"]
    w_budgets = inst["water_budgets"]
    b_budgets = inst["capital_budgets"]
    opt_profit = inst["optimal_profit"]
    kharif = inst["kharif_crops"]
    rabi = inst["rabi_crops"]

    check_script = textwrap.dedent(f"""
        def verify_crop_solution():
            import json
            try:
                with open("solution.json") as f:
                    data = json.load(f)
            except Exception as e:
                return {{"feasible": False, "score": 0.0, "reason": f"file_error: {{e}}"}}

            if not isinstance(data, dict) or "seasons" not in data or not isinstance(data["seasons"], list):
                return {{"feasible": False, "score": 0.0, "reason": "schema_error: missing 'seasons' list"}}

            seasons = data["seasons"]
            if len(seasons) != {T}:
                return {{"feasible": False, "score": 0.0, "reason": f"expected {T} seasons, got {{len(seasons)}}"}}

            A = {A}
            waters = {waters}
            costs = {costs}
            prices = {prices}
            n_deltas = {n_deltas}
            w_budgets = {w_budgets}
            b_budgets = {b_budgets}
            opt_profit = {opt_profit}
            kharif = set({kharif})
            rabi = set({rabi})

            total_profit = 0
            cur_nitrogen = 50
            cur_budget = b_budgets[0]

            for t_idx, s_plan in enumerate(seasons):
                season_num = t_idx + 1
                is_kharif = (season_num % 2 == 1)
                active_set = kharif if is_kharif else rabi
                season_name = "Kharif" if is_kharif else "Rabi"

                if not isinstance(s_plan, dict) or "cells" not in s_plan or not isinstance(s_plan["cells"], dict):
                    return {{"feasible": False, "score": 0.0, "reason": f"season_{{season_num}}_invalid_cells_dict"}}

                alloc = s_plan["cells"]
                cur_land = 0
                cur_water = 0
                cur_cost = 0

                if cur_nitrogen <= 0:
                    planted_count = sum(v for v in alloc.values() if isinstance(v, int) and v > 0)
                    if planted_count > 0:
                        return {{"feasible": False, "score": 0.0, "reason": f"season_{{season_num}}_planted_while_nitrogen_exhausted"}}

                for crop, count in alloc.items():
                    if not isinstance(count, int) or count < 0:
                        return {{"feasible": False, "score": 0.0, "reason": f"invalid_crop_count_{{crop}}_{{count}}"}}
                    if count == 0:
                        continue
                    if crop not in active_set:
                        return {{"feasible": False, "score": 0.0, "reason": f"crop_{{crop}}_not_allowed_in_{{season_name}}_season_{{season_num}}"}}

                    cur_land += count
                    cur_water += count * waters[crop]
                    cur_cost += count * costs[crop]

                if cur_land > A:
                    return {{"feasible": False, "score": 0.0, "reason": f"land_overflow_season_{{season_num}}_{{cur_land}}_gt_{{A}}"}}
                if cur_water > w_budgets[t_idx]:
                    return {{"feasible": False, "score": 0.0, "reason": f"water_overflow_season_{{season_num}}_{{cur_water}}_gt_{{w_budgets[t_idx]}}"}}
                if cur_cost > cur_budget:
                    return {{"feasible": False, "score": 0.0, "reason": f"budget_overflow_season_{{season_num}}_{{cur_cost}}_gt_{{cur_budget}}"}}

                y_mult = 0.0 if cur_nitrogen <= 0 else max(0.4, min(1.6, round(0.4 + 0.6 * (cur_nitrogen / 50.0), 2)))
                cur_revenue = sum(count * int(round(prices[t_idx][c] * y_mult)) for c, count in alloc.items() if count > 0)
                season_p = cur_revenue - cur_cost
                total_profit += season_p

                fallow = max(0, A - cur_land)
                delta_n = sum(count * n_deltas[c] for c, count in alloc.items() if count > 0) + fallow
                cur_nitrogen = max(0, min(100, cur_nitrogen + delta_n))

                # Compound budget for next season
                if t_idx + 1 < len(b_budgets):
                    cur_budget = max(0, b_budgets[t_idx + 1] + season_p)

            ratio = total_profit / max(1, opt_profit)
            is_optimal = (ratio >= 0.90)
            is_good = (ratio >= 0.60)
            return {{
                "feasible": True,
                "profit": total_profit,
                "optimal_profit": opt_profit,
                "ratio": round(ratio, 4),
                "optimal": is_optimal,
                "good": is_good,
            }}

        v_res = verify_crop_solution()
    """)

    case_feasible = _case(
        "target",
        check_script + "\nresult = v_res.get('feasible', False)",
        True,
    )
    case_good = _case(
        "target",
        check_script + "\nresult = v_res.get('feasible', False) and v_res.get('good', False)",
        True,
    )
    case_optimal = _case(
        "target",
        check_script + "\nresult = v_res.get('feasible', False) and v_res.get('optimal', False)",
        True,
    )
    case_regression = _case(
        "regression",
        check_script + "\nresult = v_res.get('feasible', False)",
        True,
    )

    return [case_feasible, case_good, case_optimal, case_regression]


def crop_planning_task(seed) -> CropPlanningTaskInstance:
    """Build the Crop Planning POMDP TaskInstance."""
    inst = generate_crop_planning_instance(seed)

    A = inst["land_cells"]
    T = inst["num_seasons"]
    waters = inst["waters"]
    costs = inst["costs"]
    n_deltas = inst["nitrogen_deltas"]
    w_budgets = inst["water_budgets"]
    b_budgets = inst["capital_budgets"]
    prior_hist = inst["prior_price_history"]
    opt_schedule = inst["optimal_schedule"]

    kharif_rows = []
    for c in KHARIF_CROPS:
        role = "Cash Crop" if c in CASH_CROPS else ("Staple" if c in STAPLE_CROPS else "Pulse/Oilseed")
        n_sign = f"+{n_deltas[c]}" if n_deltas[c] > 0 else f"{n_deltas[c]}"
        kharif_rows.append(f"  - {c:<10} [{role:<13}]: water={waters[c]:>2}, cost={costs[c]:>2}, nitrogen={n_sign:>2}/cell")
    kharif_desc = "\n        ".join(kharif_rows)

    rabi_rows = []
    for c in RABI_CROPS:
        role = "Cash Crop" if c in CASH_CROPS else ("Staple" if c in STAPLE_CROPS else "Pulse/Oilseed")
        n_sign = f"+{n_deltas[c]}" if n_deltas[c] > 0 else f"{n_deltas[c]}"
        rabi_rows.append(f"  - {c:<10} [{role:<13}]: water={waters[c]:>2}, cost={costs[c]:>2}, nitrogen={n_sign:>2}/cell")
    rabi_desc = "\n        ".join(rabi_rows)

    prior_lines = []
    for entry in prior_hist:
        prior_lines.append(f"  Season {entry['past_season']}: " + ", ".join(f"{c}: {p}" for c, p in entry["prices"].items()))
    prior_desc = "\n        ".join(prior_lines)

    instructions = textwrap.dedent(f"""
        Multi-Season Crop Rotation & Market POMDP with Dynamic Soil Nitrogen:

        FARM SETUP:
        - Total Land Cells: A = {A} unit cells.
        - Planning Horizon: {T} sequential seasons alternating between Kharif (odd seasons 1, 3, 5, 7, 9) and Rabi (even seasons 2, 4, 6, 8, 10).
        - Seasonal Crop Exclusivity: Only Kharif crops can be planted in Kharif seasons, and only Rabi crops in Rabi seasons.

        CROP RESOURCE REQUIREMENTS & ECONOMIC PROFILES:
        * Kharif Season Crops (Odd Seasons 1, 3, 5, 7, 9):
        {kharif_desc}

        * Rabi Season Crops (Even Seasons 2, 4, 6, 8, 10):
        {rabi_desc}

        SOIL NITROGEN DYNAMICS & YIELD MULTIPLIER:
        - Soil Nitrogen S_t is a continuous index in [0, 100], starting at S_1 = 50.
        - Yield Multiplier: Y(S_t) = clamp(0.4 + 0.6 * (S_t / 50.0), 0.4, 1.6).
          * At S=50 (baseline fertility): 1.0x harvest yield.
          * At S=100 (rich fertility): 1.6x harvest yield (+60% bonus revenue on all crops!).
          * At S=10 (depleted): 0.52x harvest yield (heavy penalty).
        - Nitrogen Depletion & Regeneration:
          * Cash Crops (cotton, sugarcane, potato) severely deplete nitrogen (-4 to -5 per cell), but can yield huge windfall profits.
          * Staples (rice, maize, wheat, barley) mildly deplete nitrogen (-1 per cell).
          * Pulses & Oilseeds (soybean, groundnut, mustard, chickpea, lentil) fix nitrogen (+2 to +3 per cell).
          * Fallow / Idle Land: Every unplanted idle cell naturally gains +1 nitrogen!
        - ZERO NITROGEN BAN: If Soil Nitrogen drops to 0 (S <= 0), the soil is completely dead:
          * YOU CANNOT PLANT ANY CROPS NEXT SEASON!
          * You must submit `allocation='{{}}'` (all land idle) to let the land regenerate naturally.

        HISTORICAL MARKET PRICES (Past Records for Season -1 and Season 0):
        {prior_desc}

        SEASON 1 INITIAL STATUS (KHARIF):
        - Available Land: {A} cells
        - Available Water Quota: {w_budgets[0]} units
        - Starting Capital Budget: {b_budgets[0]}
        - Starting Soil Nitrogen: 50 / 100 (Yield Multiplier: 1.0x)
        - Allowed Kharif Crops: {', '.join(KHARIF_CROPS)}

        RULES & BUDGET REINVESTMENT:
        - True market prices for Season 1 are strictly unknown until harvest.
        - 100% of season profit/loss compounds directly into next season's capital budget.
        - To plant crops for Season 1, call the `plant_season` tool:
            `plant_season(allocation='{{"cotton": 8, "soybean": 12}}')`
        - The server will execute your planting, update soil nitrogen, reveal Season 1's actual prices and profit, and return Season 2's status.
        - Repeat sequentially across all {T} seasons, then call `finish`!
    """).strip()

    cases = make_crop_planning_cases(inst)
    reference_solution = json.dumps({"seasons": [{"season": t + 1, "cells": alloc} for t, alloc in enumerate(opt_schedule)]}, indent=2)

    return CropPlanningTaskInstance(
        inst=inst,
        task_id="crop_planning",
        title=f"Kharif/Rabi Soil Nitrogen POMDP (A={A}, T={T}, seed={inst['seed']})",
        difficulty="hard",
        instructions=instructions,
        files={},  # Completely empty sandbox! No secret price files, no farm.py!
        cases=cases,
        reference={
            "solution.json": reference_solution,
        },
        system_prompt=SYSTEM_PROMPT,
        closing="Plant Season 1 now by calling `plant_season`.",
    )
