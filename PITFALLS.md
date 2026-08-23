# PITFALLS.md -- Documented Pitfalls in markov_vol

Twelve pitfalls encountered during development. Each entry includes what happened, why it was wrong, the correct approach, and how to verify the fix.

---

## Pitfall 1: TRACE avg_yield is YTM, not spread

**What happened.** Early versions used `avg_yield` from daily TRACE aggregates as the credit spread input to Merton inversion. The resulting CIV values were 7.4x too large (median CIV ~ 2.3 instead of ~ 0.31), and the correlation between TRACE-derived CIV and the correct bondret-derived CIV was 0.0004.

**Why it was wrong.** TRACE `yld_pt` is yield-to-maturity, not a credit spread. YTM includes the risk-free rate, which dominates the signal. Feeding YTM into the Merton model as if it were a spread produces nonsensical implied volatilities.

**The correct approach.** Compute the credit spread as end-of-month bond yield minus the maturity-matched treasury yield (stored in `wrds_panel_yieldspread.parquet`). The spread is monthly, so forward-fill to daily using merge_asof with a 45-day backward tolerance.

**Update (August 2026).** The original fix used `t_spread` from `wrdsapps_bondret.bondret`, believing it to be the treasury-matched credit spread. It is not: `t_spread` is a bid/ask (transaction-cost) spread. For a mean-reversion strategy this is worse than Pitfall 1, because bid/ask bounce is mechanically mean-reverting. All results were rebuilt on the yield-minus-treasury spread (see the Correction section in README.md).

**How to verify.** After building daily CIV, compare the median to the monthly CIV median from the companion portfolios. The ratio should be close to 1.0. If it is above 3.0, you are probably using YTM instead of spread. Also check the spread level against external benchmarks: median IG credit spreads run roughly 100-200 bps (ICE BofA indices); `t_spread` medians around 35 bps, an immediate red flag.

---

## Pitfall 2: Compustat CUSIP != bond issuer CUSIP

**What happened.** Attempted to join bond data to Compustat fundamentals by matching the first 6 characters of the bond CUSIP to the Compustat CUSIP. Match rates were below 30%.

**Why it was wrong.** Compustat uses its own CUSIP assignment, which does not reliably match bond issuer CUSIPs. The mapping is many-to-many and vintage-dependent.

**The correct approach.** Use a two-step permno bridge: (1) `bondcrsp_link` maps `issuer_cusip` to CRSP `permno`, (2) `ccm_link` (CCM linking history) maps `permno` to Compustat `gvkey`. This gives 85%+ match rates.

**How to verify.** After building the identifier map, check that the number of matched issuer_cusips is at least 1000. If below 500, the bridge is broken.

---

## Pitfall 3: Optimizer gets stuck without smart starts

**What happened.** Initial estimation used random starting points drawn uniformly from parameter bounds. The best loglikelihood was 20,000 points below the optimum found later with smart starts.

**Why it was wrong.** The likelihood surface has many local optima. Random starts that place `sigma_base` far from the data grand mean converge to poor local optima because the model CIV is nowhere near the observed CIV.

**The correct approach.** Anchor `sigma_base` to the grand mean of observed CIV for the first 3 starts, then perturb by +/-10% for remaining starts. This ensures at least one start is in the basin of the global optimum.

**How to verify.** After estimation, check that `sigma_base` is within 50% of the data grand mean. If it is more than 50% away, the optimizer likely found a local optimum. The estimation code prints a warning in this case.

---

## Pitfall 4: Intercepts needed (R2 2.8% -> 99.4%)

**What happened.** The first model version had no portfolio-specific intercepts. The Markov model fit the grand mean well but could not explain cross-portfolio variation. R-squared was 2.8%.

**Why it was wrong.** Different leverage-maturity portfolios have systematically different CIV levels. Without intercepts, the model forces all portfolios to have the same mean, which is a terrible fit to cross-sectional variation.

**The correct approach.** Estimate portfolio intercepts as `port_mean - grand_mean` and add them to the model CIV: `model_civ[j, p] = sigma_grid[j] + port_intercepts[p]`. This separates cross-sectional level differences from time-series regime variation. R-squared jumps to 99.4%.

**How to verify.** After estimation, compute R-squared with and without intercepts. The difference should be dramatic (typically 90+ percentage points).

---

## Pitfall 5: WRDS drops connection on yearly queries

**What happened.** Initial TRACE queries used full-year date ranges (e.g., `WHERE trd_exctn_dt BETWEEN '2020-01-01' AND '2020-12-31'`). The WRDS PostgreSQL server would drop the connection after several minutes with no error message -- just a closed socket.

**Why it was wrong.** TRACE Enhanced has hundreds of millions of rows. A full-year GROUP BY query exceeds the WRDS server's query timeout or memory limits.

**The correct approach.** Split queries into half-year chunks (Jan-Jun, Jul-Dec). Each chunk completes in under 60 seconds. Add auto-reconnect logic: on failure, sleep 5 seconds, close the engine, reconnect, and retry up to 3 times.

**How to verify.** Monitor the per-chunk timing in the console output. Each chunk should complete in under 120 seconds. If a chunk takes more than 300 seconds, the query is too large.

---

## Pitfall 6: Unicode prints fail on Windows

**What happened.** Print statements with Unicode characters (arrows, Greek letters, checkmarks) caused `UnicodeEncodeError` on Windows when stdout was redirected to a file or when the console code page was not UTF-8.

**Why it was wrong.** Windows cmd.exe and PowerShell default to code page 1252, not UTF-8. Python's default stdout encoding on Windows does not handle arbitrary Unicode.

**The correct approach.** Use only ASCII in all print statements. Replace Unicode arrows with `->`, Greek letters with their names (`sigma`, `lambda`), and checkmarks with `PASS` / `OK`.

**How to verify.** Run `python run_pipeline.py > log.txt 2>&1` on Windows. If it completes without `UnicodeEncodeError`, the fix is working.

---

## Pitfall 7: simulate n_port bug (cross-product vs parallel arrays)

**What happened.** The simulation function `simulate_markov_merton` originally computed `n_port = len(L_grid) * len(tau_grid)`, treating them as a cross-product. This produced 80 portfolios (4 lev x 5 mat x 4 = 80??) instead of the expected 20.

**Why it was wrong.** `L_grid` and `tau_grid` are parallel arrays of length `n_port`, not separate grids to be crossed. Each index `p` corresponds to one portfolio with leverage `L_grid[p]` and maturity `tau_grid[p]`.

**The correct approach.** Set `n_port = len(L_grid)` (which equals `len(tau_grid)`). The Hamilton filter, simulation, and all other functions treat these as parallel arrays.

**How to verify.** Check that `len(L_grid) == len(tau_grid)` and that `y_all.shape[1]` equals both. If `y_all.shape[1]` is 20 but `len(L_grid) * len(tau_grid)` is 80, you have the cross-product bug.

---

## Pitfall 8: rpt_side_cd = 'S' filter needed in TRACE

**What happened.** Early TRACE queries did not filter on `rpt_side_cd`, pulling both buy-side and sell-side reports. This double-counted many trades and inflated volume estimates.

**Why it was wrong.** TRACE reports both sides of each trade. Without filtering to one side, inter-dealer trades appear twice. The convention is to keep only sell-side reports (`rpt_side_cd = 'S'`).

**The correct approach.** Add `AND rpt_side_cd = 'S'` to all TRACE queries. This gives clean, non-duplicated trade counts and volumes.

**How to verify.** Compare the number of rows with and without the filter. The filtered count should be roughly 50-60% of the unfiltered count. If it is 100%, the filter is not being applied.

---

## Pitfall 9: bondret is monthly, not daily

**What happened.** Attempted to use bondret spreads directly as daily observations. The resulting CIV panel had only ~260 unique dates per year instead of ~252 trading days, and within each month all observations were identical.

**Why it was wrong.** `wrdsapps_bondret.bondret` is a monthly dataset. Each row represents a bond-month, not a bond-day. Using it as daily data produces stale, repeated values.

**The correct approach.** Accept that bondret spreads are monthly. For daily CIV, forward-fill the monthly spread to daily dates using merge_asof (backward, 45-day tolerance), and combine with genuinely daily leverage from CRSP. The daily variation comes from leverage, not from spread updates.

**How to verify.** After building daily CIV, check that the number of unique dates is close to 252 per year. If it is close to 12, you are treating monthly data as daily without proper forward-filling.

---

## Pitfall 10: model_civ under Merton is trivially sigma_j

**What happened.** Initially tried to compute model CIV for each state by running Merton inversion at each (sigma_grid[j], L, tau) combination. But the Merton model maps sigma_A -> spread -> invert back to sigma_A. The inversion recovers exactly sigma_A.

**Why it was wrong.** The round-trip sigma_A -> merton_spread(sigma_A, L, tau) -> invert_spread_to_civ(spread, L, tau) = sigma_A is an identity. The model CIV for state j is simply sigma_grid[j], regardless of L and tau.

**The correct approach.** Set `model_civ[j, p] = sigma_grid[j] + port_intercepts[p]`. The Merton structure enters through the data construction (observed CIV depends on leverage and maturity), not through the state-space model CIV.

**How to verify.** Run `invert_spread_to_civ(merton_spread(0.30, 0.50, 5.0), 0.50, 5.0)`. The result should be 0.30 (up to numerical tolerance).

---

## Pitfall 11: Forward-filling leverage is a shortcut

**What happened.** Early versions forward-filled quarterly Compustat leverage to daily frequency, ignoring daily equity price changes. This produced step-function leverage that jumped only 4 times per year.

**Why it was wrong.** Leverage = debt / (debt + equity). Debt changes quarterly, but equity market cap changes daily. Forward-filling leverage misses the daily variation that is the entire point of moving to daily frequency.

**The correct approach.** Forward-fill only the debt component (quarterly Compustat). Compute market cap daily from CRSP (abs(price) * shrout / 1000). Then compute leverage = debt / (debt + mcap) at daily frequency. This gives genuinely daily leverage variation.

**How to verify.** Compute the autocorrelation of daily leverage. With proper daily market cap, the AR(1) coefficient should be around 0.98-0.99. If it is 1.000 for weeks at a time, you are forward-filling the ratio instead of recomputing it.

---

## Pitfall 12: Copied data files break provenance

**What happened.** Manually copied parquet files between project directories (`impcredvol/data/` -> `markov_vol/data/`). Later, the source data was updated but the copy was stale, producing inconsistent results.

**Why it was wrong.** Manual file copying breaks the reproducibility chain. There is no way to verify which version of the data produced a given result.

**The correct approach.** Every data file in `markov_vol/data/` must be produced by a script in this project (`fetch_daily_trace.py`, `build_daily_civ.py`). Upstream data from `impcredvol2/data/` is read in place, never copied. The `config.py` file defines paths to upstream data using relative paths from the project root.

**How to verify.** For every file in `markov_vol/data/`, you should be able to name the script that produces it. If you cannot, the file is a manual copy and should be deleted and regenerated.
