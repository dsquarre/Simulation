"""Crop planning v2: generator and game rules (rules in docs/v2_test_rules.md).

Changes from v1 (harness/crop_planning.py, which is untouched):
  1. capital compounds: capital_{t+1} = capital_t + profit_t, no per-season refill
  2. ruin: capital below one cell of the season's cheapest crop -> idle for the rest of the game
  3. each Rabi season is bad with probability 0.2: every Rabi price x0.5, nothing else changes
  4. no seasonal Rabi price bias (v1 lowered Rabi prices by 0.16 x volatility every season)
  5. the two seasons of price history come from the same process as future prices

Pure Python on purpose: the game must run in the harness image without numpy.
The oracles live in crop_v2_oracles.py.
"""
import random

from .crop_planning import (ALL_CROPS, CASH_CROPS, KHARIF_CROPS, NITROGEN_DELTA, PULSE_CROPS, RABI_CROPS,
                            STAPLE_CROPS, get_yield_multiplier)

NUM_SEASONS = 10
SHOCKS = [-0.40, -0.20, 0.0, 0.25, 0.45]   # shock values, scaled by crop volatility
# Each season every crop's shock = MARKET_WEIGHT x one market-wide draw + CROP_WEIGHT x its own draw.
# 0.6^2 + 0.8^2 = 1, so each crop's shock spread is the same as a single draw; only the share that
# moves crops independently (and so can change which crop is best) is new. (rules version 2, 2026-10-31)
MARKET_WEIGHT, CROP_WEIGHT = 0.6, 0.8
BOOM_PROB, BOOM_SIZE = 0.30, 0.30         # each cash crop, each season
NOISE = 0.15                              # per crop, uniform in [-NOISE, NOISE]
KHARIF_WAVE = 0.4                         # Kharif crops in a Kharif season: +0.4 x volatility
BAD_RABI_PROB, BAD_RABI_FACTOR = 0.2, 0.5     # factor 0.7 in rules version 1


def season_crops(t):
    """Season t >= 1: odd = Kharif, even = Rabi. History seasons -1 and 0 follow the same parity."""
    return KHARIF_CROPS if t % 2 == 1 else RABI_CROPS


def generate_params(seed):
    """Everything about a farm that does not change during the game."""
    rng = random.Random(f"crop-v2-params-{seed}")
    A = rng.randint(20, 28)
    waters, costs, base, vol = {}, {}, {}, {}
    for c in ALL_CROPS:
        kharif = c in KHARIF_CROPS
        if c in CASH_CROPS:
            waters[c] = rng.randint(16, 24) if kharif else rng.randint(12, 18)
            costs[c] = rng.randint(55, 80)
            base[c] = costs[c] + rng.randint(-30, 80)
            vol[c] = rng.choice([0.50, 0.65, 0.80])
        elif c in STAPLE_CROPS:
            waters[c] = rng.randint(12, 20) if kharif else rng.randint(8, 14)
            costs[c] = rng.randint(30, 50)
            base[c] = costs[c] + rng.randint(-15, 25)
            vol[c] = rng.choice([0.25, 0.35, 0.45])
        else:
            waters[c] = rng.randint(6, 14) if kharif else rng.randint(4, 10)
            costs[c] = rng.randint(15, 30)
            base[c] = costs[c] + rng.randint(-10, 15)
            vol[c] = rng.choice([0.15, 0.25, 0.35])
    water_budgets = []
    for t in range(1, NUM_SEASONS + 1):
        crops = season_crops(t)
        factor = rng.uniform(0.55, 0.85) if t % 2 == 1 else rng.uniform(0.35, 0.65)
        water_budgets.append(int(A * sum(waters[c] for c in crops) / 6.0 * factor))
    avg_cost = sum(costs[c] for c in KHARIF_CROPS) / 6.0
    start_capital = int(A * avg_cost * rng.uniform(0.35, 0.55))   # 0.65-0.95 in rules version 1
    return {"seed": seed, "land_cells": A, "waters": waters, "costs": costs, "base_prices": base,
            "volatilities": vol, "nitrogen_deltas": dict(NITROGEN_DELTA), "water_budgets": water_budgets,
            "start_capital": start_capital, "initial_nitrogen": 50, "num_seasons": NUM_SEASONS}


def draw_season(params, t, rng):
    """One season's prices for all 12 crops. The single source of the price process: the generator,
    the history and the parameter-aware oracle's scenarios all call this."""
    kharif_season = t % 2 == 1
    bad = (not kharif_season) and rng.random() < BAD_RABI_PROB
    market = rng.choice(SHOCKS)
    prices = {}
    for c in ALL_CROPS:
        vol = params["volatilities"][c]
        wave = KHARIF_WAVE * vol if (kharif_season and c in KHARIF_CROPS) else 0.0
        boom = BOOM_SIZE if (c in CASH_CROPS and rng.random() < BOOM_PROB) else 0.0
        shock = MARKET_WEIGHT * market + CROP_WEIGHT * rng.choice(SHOCKS)
        p = max(1, int(round(params["base_prices"][c] * (1.0 + wave + shock * vol + boom
                                                          + rng.uniform(-NOISE, NOISE)))))
        if bad and c in RABI_CROPS:
            p = max(1, int(round(p * BAD_RABI_FACTOR)))
        prices[c] = p
    return prices, {"market_shock": market, "bad": bad}


def generate_instance(seed):
    """Parameters plus one realised draw: the history the agent sees and the prices of seasons 1-10."""
    params = generate_params(seed)
    rng = random.Random(f"crop-v2-draws-{seed}")
    history = [{"season": t, "prices": draw_season(params, t, rng)[0]} for t in (-1, 0)]
    prices, events = [], []
    for t in range(1, NUM_SEASONS + 1):
        p, e = draw_season(params, t, rng)
        prices.append(p)
        events.append(e)
    return {**params, "history": history, "prices": prices, "events": events}


# ------------------------------------------------------------------ rules
def min_cost(params, t):
    return min(params["costs"][c] for c in season_crops(t))


class Game:
    """One 10-season game. step() enforces every rule; strategies only choose allocations."""

    def __init__(self, inst):
        self.inst = inst
        self.t, self.nitrogen, self.capital = 1, inst["initial_nitrogen"], inst["start_capital"]
        self.ruined, self.log = False, []

    @property
    def done(self):
        return self.t > self.inst["num_seasons"]

    def must_idle(self):
        """Ruin is absorbing; exhausted soil forces one idle season."""
        if not self.ruined and self.capital < min_cost(self.inst, self.t):
            self.ruined = True
        return self.ruined or self.nitrogen <= 0

    def check(self, alloc):
        """Error text for an illegal allocation, else None."""
        inst, t = self.inst, self.t
        for c, n in alloc.items():
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                return f"cell count for '{c}' must be a whole number >= 0, got {n!r}"
        alloc = {c: n for c, n in alloc.items() if n}
        if self.must_idle() and alloc:
            why = "you are ruined (capital too low to plant)" if self.ruined else "soil nitrogen is 0"
            return f"{why}: all land must stay idle, submit {{}}"
        for c in alloc:
            if c not in season_crops(t):
                return f"'{c}' cannot be planted in season {t}; allowed: {', '.join(season_crops(t))}"
        land = sum(alloc.values())
        if land > inst["land_cells"]:
            return f"land overflow: {land} cells used, the farm has {inst['land_cells']}"
        water = sum(n * inst["waters"][c] for c, n in alloc.items())
        if water > inst["water_budgets"][t - 1]:
            return f"water overflow: {water} units used, this season's quota is {inst['water_budgets'][t - 1]}"
        cost = sum(n * inst["costs"][c] for c, n in alloc.items())
        if cost > self.capital:
            return f"capital overflow: planting costs {cost}, you have {self.capital}"
        return None

    def step(self, alloc):
        """Plant, harvest, advance one season. Returns this season's revealed prices."""
        error = self.check(alloc)
        if error:
            raise ValueError(f"season {self.t}: {error}")
        inst, t = self.inst, self.t
        alloc = {c: n for c, n in alloc.items() if n}
        prices = inst["prices"][t - 1]
        y = get_yield_multiplier(self.nitrogen)
        cost = sum(n * inst["costs"][c] for c, n in alloc.items())
        revenue = sum(n * int(round(prices[c] * y)) for c, n in alloc.items())
        planted = sum(alloc.values())
        delta = sum(n * inst["nitrogen_deltas"][c] for c, n in alloc.items()) + (inst["land_cells"] - planted)
        self.log.append({"season": t, "cells": alloc, "profit": revenue - cost, "capital_before": self.capital,
                         "nitrogen_before": self.nitrogen})
        self.capital += revenue - cost
        self.nitrogen = max(0, min(100, self.nitrogen + delta))
        self.t += 1
        return prices


# --------------------------------------------------------------- knapsack
def best_allocation(values, crops, land, water, capital, waters, costs):
    """Exact integer optimum of sum(n_c * values[c]) subject to land, water and capital.
    Branch and bound over the (at most 6) crops with a positive value per cell."""
    items = sorted((c for c in crops if values[c] > 0), key=lambda c: -values[c])
    best = [0, {}]

    def search(i, land_left, water_left, cap_left, value, chosen):
        if value > best[0]:
            best[0], best[1] = value, dict(chosen)
        if i == len(items):
            return
        # bound: every remaining cell earns the best remaining value per cell
        if value + land_left * values[items[i]] <= best[0]:
            return
        c = items[i]
        most = min(land_left, water_left // waters[c], cap_left // costs[c])
        for n in range(most, -1, -1):
            if n:
                chosen[c] = n
            search(i + 1, land_left - n, water_left - n * waters[c], cap_left - n * costs[c],
                   value + n * values[c], chosen)
            chosen.pop(c, None)

    search(0, land, water, capital, 0, {})
    return best[1]
