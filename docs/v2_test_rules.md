# Benchmark v2: first test, rules fixed before running

Committed before any v2 code is run. The commit timestamp is the registration. Any change to these rules
after this commit is a new, dated version of this file, listed under "Changes" below with its reason.
Changes after the test has run must be reported as such, never as part of the original rules.

## What v2 changes (crop_planning)
1. **Compounding capital:** capital at season 1 = the instance's starting budget;
   capital_{t+1} = capital_t + profit_t. No per-season refill.
2. **Ruin:** if capital is below the cost of one cell of the cheapest crop allowed that season, the season
   and every later season of the game are forced idle.
3. **Bad Rabi seasons:** each Rabi season is independently bad with probability 0.2. In a bad season every
   Rabi crop's price is multiplied by 0.7. Yield and water are unchanged, so the agent cannot see it coming.
4. **No seasonal Rabi bias:** the v1 term that lowered every Rabi crop's price by 0.16 × volatility in every
   season is removed. The Kharif in-season term is kept.
5. **Honest history:** the two seasons of price history shown at the start are drawn from the same process
   as future prices.
6. Everything else as in v1: crop categories and volatility ranges, market-wide shock, cash-crop booms,
   noise, nitrogen dynamics, water budgets.

v1 stays unchanged; v2 lives in a separate module.

## The test (no language model)
Seeds 0–199 of the v2 generator. For each seed, each strategy plays one 10-season game on the same draws.
W = final wealth (capital after season 10).

Strategies:
- **idle:** plants nothing.
- **random:** each season a uniformly random feasible allocation (fixed RNG seed per game seed).
- **greedy:** each season maximises profit assuming the last observed prices repeat (the season-0 history for
  season 1), using the current nitrogen multiplier. Ignores future seasons.
- **cautious:** like greedy, but may only plant pulses/oilseeds and staples.
- **clairvoyant (CV):** exact DP that knows every realised price. W_cv.
- **parameter-aware oracle (PA):** knows the generator's per-seed parameters and probabilities but not the
  draws. Maximises expected final wealth by averaging over sampled scenarios, with a reduced action set.
  Approximate; the sample count and action set are recorded in the results. W_pa.

Per-game score: s = W / W_pa (paired, same draws).

## Kill rules
The test **fails** (redesign before building anything else) if any of these hold:

1. **Too easy:** mean over seeds of W_greedy / W_pa ≥ 0.95.
2. **Luck doesn't matter:** mean over seeds of W_cv / W_pa ≤ 1.05.
3. **Oracle wrong:** any of idle, random, greedy or cautious has a higher mean W than PA, with the paired
   difference more than 2 standard errors above 0.
4. **Too noisy for 5 games:** for greedy and cautious, let sd = standard deviation of s across seeds. A linear
   learning trend over games 1..5 has standard error sd / √10. A total improvement of 0.10 over 5 games is a
   slope of 0.025 per game. The rule fails if 2 × sd / √10 > 0.025, i.e. sd > 0.0395, for either strategy.
   Consequence (decided in advance): move to 10 games, or make luck-free decision-level regret the headline.

## Reported regardless of outcome
Mean, sd and the 5th/50th/95th percentiles of W for every strategy; mean W_cv / W_pa; share of games
ending in ruin per strategy; runtime of the PA oracle per seed.

## Not covered by this test
Scoring of real model runs (headline metric, the 5-game protocol, the notes file) gets its own rules file,
committed before the first model run.

## Changes
### Version 2 — 2026-10-31, after the first test (code 6293054) failed rules 2 and 4
First-test result: W_cv / W_pa = 1.034 (rule 2 fail); sd of W / W_pa = 0.17 greedy, 0.19 cautious
(rule 4 fail); rules 1 and 3 passed; ruin in 0% of games. Kept on record in results/v2_first_test.json.

Change 1 (rule 2): price shocks become partly crop-specific. Each season, each crop's shock =
0.6 × one market-wide draw + 0.8 × its own independent draw, both from {−0.40, −0.20, 0, +0.25, +0.45},
scaled by the crop's volatility. The spread of each crop's shock is unchanged (0.6² + 0.8² = 1); its
mean moves from +0.020 to +0.028. Hypothesis: with one market-wide shock all crops move together, so
knowing the future rarely changes which crop is best.

Change 2 (ruin never happened): starting capital = A × average Kharif cost × uniform(0.35, 0.55)
(was 0.65–0.95), and a bad Rabi season multiplies Rabi prices by 0.5 (was 0.7).

Change 3 (rule 4 failed): luck-free score, measured against the parameter-aware oracle (PA). For each
decision a strategy actually made (forced-idle seasons excluded), regret = PA's expected final wealth from
its best move − PA's expected final wealth from the strategy's move, at the strategy's real state
(season, nitrogen, wealth), before that season's prices are drawn. Luck-free score of a game =
1 − (sum of regrets) / (PA's expected final wealth from the starting state).

Rule 4, version 2: same formula and threshold, applied to the luck-free score instead of W / W_pa:
fails if the standard deviation across seeds of greedy's or cautious's luck-free score exceeds 0.0395.

New rule 5 (ruin range): fails if PA is ruined in 10% of games or more, or if greedy is ruined in fewer
than 15% or more than 55% of games.

What PA knows (stated so its ceiling is read correctly). PA knows: land, water use, costs and nitrogen
effects per crop, crop categories, starting capital and nitrogen, the water quota of all 10 seasons, each
crop's exact base price and volatility for the farm, and every constant of the price process (shock
values and 0.6/0.8 weights, boom probability and size, noise range, Kharif in-season term, bad-Rabi
probability and factor). PA does not know the random draws. An agent sees the per-crop facts, categories,
starting state and two seasons of price history, and learns each season's water quota and prices only as
the game reveals them. PA is therefore a full-knowledge ceiling: part of any strategy's regret is
information no agent can obtain within a game, and a luck-free score of 1.0 is not reachable in general.

Rules 1–3, the strategies and the seeds (0–199) are unchanged.
A smoke test on seeds 1000–1001 (outside the test range) was run before this version was committed,
only to check that the code runs.
