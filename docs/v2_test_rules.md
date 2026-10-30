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
(none)
