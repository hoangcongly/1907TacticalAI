# Survivorship and universe-selection study

Run on 2026-11-11 with `scripts/download_vision_history.py` and `scripts/survivorship_study.py`.
Data: 1h bars and funding for all 918 USDT perpetuals in the public Binance dump, 899 of them
with enough history to load, 57 of those no longer trading at the data cut-off (2026-09-10, the
same vintage as `strategy_v3.json`). Raw output: `artifacts/survivorship_study.json`.

## Question

`artifacts/universe_wide.json` lists the symbols that were trading, and liquid, on the day the
data was downloaded. Two things follow:

1. Contracts that were delisted before that day are absent from every backtest.
2. The liquidity filter uses volume from the end of the sample to choose symbols for 2021.

A third issue is in `simulate`: a held position whose next price is missing earns exactly zero,
so the last move of a contract that disappears between two rebalances is dropped.

## Design

The same 26 signals, combiner, costs and data cut-off, on four universes. No parameter was tuned.

| Name | Universe |
|---|---|
| `repo` | `universe_wide.json` as committed, `min_coverage = 0.15` (125 symbols load) |
| `repo_terminal` | same, with the last traded price carried one period so exits are priced |
| `pit_survivors` | point-in-time filter, restricted to symbols still trading at the cut-off |
| `pit_all` | point-in-time filter over every symbol ever listed |

The point-in-time filter admits a symbol at time t if its trailing 30-day dollar volume is at
least $20M a day (the threshold `download_wide_universe.py` uses), it has 200 bars of history, and
it has a price at t. It uses data up to t only.

## Results

Sharpe ratio, 72-hour periods, net of costs and funding.

| Portfolio | Universe | All (814 periods) | Since 2023-11 (347) | Since 2025-02 (194) | Holdout (219) |
|---|---|---:|---:|---:|---:|
| 12 positions | `repo` | 1.31 | 0.94 | 0.75 | 0.77 |
| 12 positions | `pit_survivors` | 1.40 | 1.07 | 1.24 | 1.35 |
| 12 positions | `pit_all` | 1.39 | 0.97 | 1.05 | 1.24 |
| 50 positions, equal weight | `repo` | 1.28 | 2.18 | 2.40 | 2.09 |
| 50 positions, equal weight | `pit_survivors` | 1.37 | 0.95 | 0.70 | 0.99 |
| 50 positions, equal weight | `pit_all` | 1.65 | 1.26 | 1.16 | 1.43 |

Sharpe differences with a paired block-bootstrap 95% interval:

| Effect | Portfolio | All periods | Since 2023-11 |
|---|---|---:|---:|
| Pricing the exit (`repo_terminal` − `repo`) | 12 | 0.00 [−0.01, 0.01] | 0.00 |
| | 50 | +0.02 [0.00, 0.04] | 0.00 |
| Point-in-time liquidity (`pit_survivors` − `repo`) | 12 | +0.09 [−0.80, 0.90] | +0.13 [−1.08, 1.72] |
| | 50 | +0.09 [−0.95, 1.10] | −1.24 [−2.44, 0.34] |
| Delisted symbols (`pit_all` − `pit_survivors`) | 12 | −0.01 [−0.39, 0.37] | −0.10 [−0.69, 0.52] |
| | 50 | +0.28 [−0.06, 0.67] | +0.32 [−0.33, 0.93] |
| Total (`pit_all` − `repo`) | 12 | +0.08 [−0.80, 0.88] | +0.03 [−1.16, 1.61] |
| | 50 | +0.37 [−0.65, 1.42] | −0.92 [−2.25, 0.78] |

## What this shows

**Delisting bias was not found.** The hypothesis going in was that missing delisted contracts
flatter the backtest. Adding them did not lower the Sharpe ratio in either portfolio; for 50
positions the point estimate is higher. The book is long-short, so a contract that decays before
delisting is as likely to be a profitable short as a losing long. Over the whole history the
strategy held a position into a missing price 19 times (12 positions) and 50 times (50 positions),
and the return `simulate` ignored sums to +1.5% and −1.2%. The simulator behavior is wrong in
principle and immaterial here.

**The 50-position result depends on the universe it was chosen on.** On the committed universe the
wide configuration scores 2.18 since 2023-11, which matches the figures in `CLAUDE.md`. On a
universe built without hindsight it scores 0.95 to 1.26. The interval on that difference includes
zero, so 347 periods cannot establish the drop statistically, and the comparison is not pure: the
point-in-time universe is also broader (a median of 168 eligible symbols in 2025 against 125).
The 12-position configuration does not show the drop.

**The leverage estimate moves with it.** Growth-optimal leverage is mean over variance. Since
2023-11 that ratio is 8.3 on the committed universe (57.0% return, 26.2% volatility), 5.2 on
`pit_all` (30.6%, 24.3%) and 3.7 on `pit_survivors` (23.9%, 25.3%). `strategy_v3_wide.json` runs
7.89x, which was sized from the first of these.

## Limits

- Every comparison here is in-sample. The holdout was already used twice; this study did not
  choose any parameter, but it also does not provide a clean out-of-sample test.
- Funding dumps exist for complete months only, so the last partial month carries the previous rate.
- Spreads measured from the live order book exist for the committed universe only; other symbols
  use the Corwin-Schultz estimate.
- Baseline numbers differ slightly from earlier reports (125 symbols load instead of 127) because
  the bars come from the public dump rather than the REST API.
