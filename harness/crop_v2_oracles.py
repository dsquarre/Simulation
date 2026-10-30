"""Oracles for crop planning v2 (needs numpy; not used by the game itself).

clairvoyant(inst)
    Knows every realised price. Forward DP over seasons keeping, for each nitrogen level, only the
    highest-wealth state: with equal nitrogen, more wealth is never worse (more allocations are
    affordable and wealth is the objective), so this pruning loses nothing. Exact with respect to its
    candidate allocations, not over every possible allocation: a lower bound on the true optimum.

ParamAware(inst)
    Knows the generator's parameters and probabilities, not the draws. Backward DP on a grid of
    (nitrogen, wealth) maximising expected final wealth, averaging over sampled seasons from
    crop_v2.draw_season (the same function that makes the real prices). Plays by choosing, at the real
    state, the allocation with the best expected (wealth after this season -> value table). Approximate:
    reduced allocation set, sampled scenarios, interpolated wealth.
"""
import itertools
import random

import numpy as np

from .crop_planning import get_yield_multiplier
from .crop_v2 import best_allocation, draw_season, min_cost, season_crops

# --------------------------------------------------------- allocation sets
def candidate_allocations(inst, t, pair_step=1, triple_step=None):
    """Idle, every single crop at every count, every pair of crops with counts in steps of pair_step,
    and optionally every triple with counts in steps of triple_step. Land and water feasible only;
    capital is checked by the caller because it depends on the state."""
    crops, A, W = season_crops(t), inst["land_cells"], inst["water_budgets"][t - 1]
    waters = inst["waters"]
    out = [{}]
    for c in crops:
        out += [{c: n} for n in range(1, min(A, W // waters[c]) + 1)]
    for a, b in itertools.combinations(crops, 2):
        for na in range(pair_step, A, pair_step):
            for nb in range(pair_step, A - na + 1, pair_step):
                if na * waters[a] + nb * waters[b] <= W:
                    out.append({a: na, b: nb})
    if triple_step:
        for a, b, c in itertools.combinations(crops, 3):
            for na in range(triple_step, A, triple_step):
                for nb in range(triple_step, A - na, triple_step):
                    for nc in range(triple_step, A - na - nb + 1, triple_step):
                        if na * waters[a] + nb * waters[b] + nc * waters[c] <= W:
                            out.append({a: na, b: nb, c: nc})
    return out


class Actions:
    """Allocations as arrays: counts [K, 6] over the season's crops, plus cost and nitrogen change."""

    def __init__(self, inst, t, allocs):
        self.crops = season_crops(t)
        self.allocs = allocs
        self.counts = np.array([[a.get(c, 0) for c in self.crops] for a in allocs], dtype=np.int64)
        self.cost = self.counts @ np.array([inst["costs"][c] for c in self.crops])
        planted = self.counts.sum(axis=1)
        self.dN = self.counts @ np.array([inst["nitrogen_deltas"][c] for c in self.crops]) \
            + (inst["land_cells"] - planted)

    def revenue(self, prices, y):
        """prices: [S, 6] array. Returns [K, S] revenue, with per-cell rounding as in the game."""
        per_cell = np.round(prices * y)          # numpy rounds halves to even, like Python's round()
        return self.counts @ per_cell.T


def lagrangian_allocations(inst, t, values_by_crop, capital, lambdas=(-20, -12, -6, -3, 0, 3, 6, 12, 20)):
    """Exact knapsack solutions for per-cell values plus a shadow price on nitrogen change.
    Adds multi-crop mixes that the pair/triple grids miss."""
    crops = season_crops(t)
    out = []
    for lam in lambdas:
        vals = {c: values_by_crop[c] + lam * (inst["nitrogen_deltas"][c] - 1) for c in crops}
        out.append(best_allocation(vals, crops, inst["land_cells"], inst["water_budgets"][t - 1],
                                   int(capital), inst["waters"], inst["costs"]))
    return out


def _dedupe(allocs):
    seen, out = set(), []
    for a in allocs:
        key = tuple(sorted((c, n) for c, n in a.items() if n))
        if key not in seen:
            seen.add(key)
            out.append(dict(key))
    return out


# ------------------------------------------------------------ clairvoyant
def clairvoyant(inst):
    """Returns (final wealth, schedule)."""
    T = inst["num_seasons"]
    # nitrogen -> (wealth, schedule, ruined); ruin is absorbing, so it is part of the state
    states = {inst["initial_nitrogen"]: (inst["start_capital"], [], False)}
    for t in range(1, T + 1):
        crops = season_crops(t)
        prices = np.array([[inst["prices"][t - 1][c] for c in crops]], dtype=float)
        base = candidate_allocations(inst, t, pair_step=1, triple_step=2)
        nxt = {}
        for N, (W, sched, was_ruined) in states.items():
            ruined = was_ruined or W < min_cost(inst, t)
            if ruined or N <= 0:
                allocs = [{}]
            else:
                y = get_yield_multiplier(N)
                vals = {c: int(round(inst["prices"][t - 1][c] * y)) - inst["costs"][c] for c in crops}
                allocs = _dedupe(base + lagrangian_allocations(inst, t, vals, W))
            acts = Actions(inst, t, allocs)
            y = get_yield_multiplier(N)
            Wn = W - acts.cost + acts.revenue(prices, y)[:, 0]
            Wn = np.where(acts.cost <= W, Wn, -np.inf)
            Nn = np.clip(N + acts.dN, 0, 100)
            for k in np.argsort(-Wn):
                if not np.isfinite(Wn[k]):
                    break
                n2 = int(Nn[k])
                if n2 not in nxt or Wn[k] > nxt[n2][0]:
                    nxt[n2] = (float(Wn[k]), sched + [allocs[k]], ruined)
        states = nxt
    W, sched, _ = max(states.values(), key=lambda s: s[0])
    return W, sched


# ------------------------------------------------------- parameter-aware
class ParamAware:
    def __init__(self, inst, scenarios=32, act_scenarios=256, grid=40, pair_step=2, seed_tag="pa"):
        self.inst = inst
        T, A = inst["num_seasons"], inst["land_cells"]
        rng = random.Random(f"{seed_tag}-{inst['seed']}")       # never the generator's draw stream
        self.samples, self.act_samples, self.actions = {}, {}, {}
        for t in range(1, T + 1):
            crops = season_crops(t)
            draw = lambda: [draw_season(inst, t, rng)[0][c] for c in crops]   # noqa: E731
            self.samples[t] = np.array([draw() for _ in range(scenarios)], dtype=float)
            self.act_samples[t] = np.array([draw() for _ in range(act_scenarios)], dtype=float)
            self.actions[t] = Actions(inst, t, candidate_allocations(inst, t, pair_step=pair_step))

        top = inst["start_capital"] + T * A * max(self.samples[t].max() for t in self.samples) * 1.6
        self.wgrid = np.concatenate([[0.0], np.geomspace(1.0, top, grid - 1)])
        self.V = {T + 1: np.tile(self.wgrid, (101, 1))}     # V[t][N, g]: expected final wealth
        for t in range(T, 0, -1):
            self.V[t] = self._backup(t)

    def _lookup(self, t, N, W):
        """Interpolate V[t] at integer nitrogen N (array) and wealth W (array, same shape).
        Above the grid, extra wealth carries over one for one (the land cannot use it)."""
        V, g = self.V[t], self.wgrid
        Wc = np.clip(W, 0, g[-1])
        i = np.clip(np.searchsorted(g, Wc, side="right") - 1, 0, len(g) - 2)
        frac = (Wc - g[i]) / (g[i + 1] - g[i])
        val = V[N, i] * (1 - frac) + V[N, i + 1] * frac
        return val + np.maximum(W - g[-1], 0)

    def _q(self, t, N, W, acts, samples):
        """Expected value of each action at wealth levels W ([G]). Returns [G, K]."""
        y = get_yield_multiplier(N)
        rev = acts.revenue(samples, y)                       # [K, S]
        Wn = W[:, None, None] - acts.cost[None, :, None] + rev[None, :, :]
        Nn = np.clip(N + acts.dN, 0, 100)[None, :, None]
        q = self._lookup(t + 1, np.broadcast_to(Nn, Wn.shape), Wn).mean(axis=2)
        return np.where(acts.cost[None, :] <= W[:, None], q, -np.inf)

    def _backup(self, t):
        V = np.empty((101, len(self.wgrid)))
        acts, g = self.actions[t], self.wgrid
        idle = Actions(self.inst, t, [{}])
        ruined = g < min_cost(self.inst, t)
        for N in range(101):
            q = self._q(t, N, g, idle if N == 0 else acts, self.samples[t]).max(axis=1)
            V[N] = np.where(ruined, g, q)     # ruin is absorbing: wealth stays as it is
        return V

    def _act_q(self, t, N, W):
        """Candidate allocations at the real state and their expected final wealth."""
        inst = self.inst
        crops = season_crops(t)
        y = get_yield_multiplier(N)
        mean_rev = np.round(self.act_samples[t] * y).mean(axis=0)
        vals = {c: mean_rev[i] - inst["costs"][c] for i, c in enumerate(crops)}
        allocs = _dedupe(self.actions[t].allocs + candidate_allocations(inst, t, pair_step=1)
                         + lagrangian_allocations(inst, t, vals, W))
        acts = Actions(inst, t, allocs)
        return allocs, self._q(t, N, np.array([float(W)]), acts, self.act_samples[t])[0]

    def act(self, t, N, W):
        """Best allocation at the real state, from a richer set than the backward pass used."""
        if W < min_cost(self.inst, t) or N <= 0:
            return {}
        allocs, q = self._act_q(t, N, W)
        return allocs[int(np.argmax(q))]
