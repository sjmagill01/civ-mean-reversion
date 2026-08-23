# Code Audit -- markov_vol/

**Date:** 2026-06-26
**Scope:** Every .py file in the project. Checked: unit consistency, data integrity, algorithm correctness, strategy correctness, index/shape bugs, numerical issues, stale references.

---

## CRITICAL ISSUES (must fix before trusting results)

### 0. CIV Basis Points Are NOT Spread Basis Points
- **Files:** `strategies.py` (all functions), `run_pipeline.py`, paper, presentation
- **Problem:** Every strategy computes `pnl = position * delta_CIV * 10000` and calls the result "basis points." The paper then multiplied by notional x duration to get dollar P&L, assuming CIV bps = spread bps. This is wrong. CIV is asset volatility; the spread is a nonlinear function of CIV through Merton. The derivative (vega = ds/d_sigma_A) is 0.07 at the median portfolio, meaning 1 CIV bp = 0.07 spread bps. The dollar P&L was overstated by 14x.
- **Impact:** The $20.1M/yr CDX claim was wrong. Correct gross is $1.7M/yr, net is NEGATIVE after 2bp CDX costs. The breakeven of "9.7 bps" was in CIV bps, not spread bps -- the actual spread breakeven is 28 CIV bps, but the signal is only 9.7.
- **What's unaffected:** All Sharpe ratios (unit-free), hit rates, walk-forward results, lag tests, R2, regime identification, "CIV is not equity" finding.
- **Status:** FIXED (2026-06-27). Paper and presentation rewritten to use Sharpe-based sizing (unit-free). Vega problem explicitly documented. Dollar claims removed. Added `unit_analysis.py` for verification.

### 1. Kim Smoother May Use Wrong Transition Direction
- **File:** `hamilton_filter.py`, lines 107 vs 182
- **Problem:** The Hamilton filter prediction step uses `P.T @ pi_prev`, meaning P[i,j] = probability of transitioning from state i to state j (row-stochastic). But the Kim smoother uses `P[j, k]` in the backward pass, which would mean P[j,k] = transition from j to k -- this IS consistent with P being row-stochastic. The formula `smoothed[t,j] = filtered[t,j] * sum_k P[j,k] * smoothed[t+1,k] / predicted[t+1,k]` is the standard Kim (1994) recursion.
- **Status:** VERIFIED CORRECT (2026-06-26). Manual calculation on 2-state, 5-period example matches code output exactly. Noisy test confirms smoother changes filtered probs by up to 10%, probabilities sum to 1, direction is correct. P[j,k] = from j to k is consistent with P.T @ pi for prediction. Both follow Hamilton (1989) convention for row-stochastic P.

### 2. BIC Counts Total Cells, Not Valid Observations
- **File:** `run_pipeline.py`, lines 58-60
- **Problem:** BIC uses `n_obs = T * n_port` but some observations are NaN. Overstates the penalty. Should use `n_obs = np.sum(~np.isnan(y_all))`.
- **Impact:** BIC values are slightly wrong. Model selection (N=7 vs N=5) may not be affected since the NaN fraction is small (~4% daily), but it's incorrect in principle.
- **Fix:** Replace with `n_obs = np.sum(~np.isnan(y_all))`.
- **Status:** FIXED (2026-06-26).

---

## MEDIUM ISSUES (could affect results)

### 3. CIV Unit Detection Is Heuristic
- **File:** `data_loader.py`, lines 48-49
- **Problem:** Uses `np.nanmedian(y_all) > 1.0` to decide if data is percent or decimal. If a dataset has legitimate CIV values > 1.0 in decimal (100%+ asset vol), it gets divided by 100 incorrectly.
- **Impact:** Monthly portfolios from impcredvol2 are in percent (median ~23), so the heuristic works. But it's fragile.
- **Fix:** Check the actual range: if median is between 5 and 95, it's probably percent.

### 4. Leverage Unit Assumption
- **File:** `build_daily_civ.py`, lines 123-125
- **Problem:** `market_cap = abs(price) * shrout / 1000` assumes shrout is in thousands and total_debt is in millions (both from WRDS). This matches Compustat conventions but is not validated.
- **Impact:** If units mismatch, all leverage values are wrong → all CIV values are wrong.
- **Verification:** The sanity check (daily CIV median = 24.0% vs monthly 23.0%, ratio 1.04) confirms units are consistent. If leverage were off by 1000x, CIV would be completely different.

### 5. PnL Scaling Assumes Decimal CIV
- **File:** `strategies.py`, all strategy functions
- **Problem:** Multiplies CIV changes by 10,000 to get bps. If CIV is accidentally in percent, PnL is 100x too large.
- **Impact:** The sanity check in build_daily_civ.py catches this (ratio must be ~1.0). If it passes, PnL scaling is correct.

### 6. Merton Inversion Can Produce Extreme Values
- **File:** `merton.py`, lines 38-39
- **Problem:** `bond_val` is floored at 1e-300, so `log(1e-300) = -691`. For tau=0.25, spread = 2,764 -- nonsensical. These get filtered later (spread < 0.10 in build_daily_civ.py), but the inversion wastes iterations.
- **Impact:** Performance, not correctness. The filter catches extremes.

### 7. Date Detection Fallback in data_loader.py
- **File:** `data_loader.py`, lines 36-42
- **Problem:** If the parquet file's date column doesn't match expected names, the fallback creates synthetic integer indices. This would silently produce wrong time series alignment.
- **Impact:** Would only trigger if the impcredvol2 portfolio file format changes. Currently works correctly.

### 8. Forward-Fill Tolerance Is Arbitrary
- **File:** `build_daily_civ.py`, line 214
- **Problem:** 45-day backward tolerance for merge_asof means a daily observation can use a spread up to 45 days old. This is ~1.5 months. If a bond hasn't traded in over a month, maybe its spread shouldn't be used.
- **Impact:** About 2-3% of the daily CIV panel might use stale spreads. The effect on portfolio-level medians is small.

---

## LOW ISSUES (style/robustness, not affecting results)

### 9. Hardcoded Bins and Thresholds
- Leverage bins `[0, 0.30, 0.50, 0.70, 1.0]` in build_daily_civ.py
- Maturity bins `[0.25, 2, 4, 6, 8.5, 30]` in build_daily_civ.py
- CIV filter `0.01 < civ < 2.0` in build_daily_civ.py
- Stress state cutoff `n_vol - 2` in strategies.py
- Should all be in config.py but don't affect correctness.

### 10. No Type Hints
- No function signatures have type annotations. Makes it harder to catch shape mismatches at review time.

### 11. Seed Is Hardcoded
- `np.random.default_rng(42)` in estimation.py. Good for reproducibility but should be documented.

### 12. CRSP Path Hardcoded in run_pipeline.py
- Line 189: hardcodes path to crsp_daily.parquet. Should use config.

### 13. Portfolio Column Order Risk
- strategies.py parses column names to identify low-lev and high-lev portfolios. If column order changes between load and strategy, indices would be wrong. The current code is correct but fragile.

---

## VERIFIED CORRECT

- Hamilton filter predict/update: correct (P.T @ pi for row-stochastic P)
- Logit transform: invertible, correctly maps between bounded and unbounded
- Smart starts: anchored to `np.nanmean(y_all)`, verified on synthetic data
- Mean-reversion strategy: signal and PnL computation are correct
- OOS split: no look-ahead bias (cal ends 2023-12-31, OOS starts 2024-01-01)
- Intercepts: correctly separate cross-section from time series
- Grid sorting: always increasing because delta_log > 0 by bounds
- Merton spread formula: matches the standard Merton (1974) formula
- No markov_civ references anywhere
- No credentials in any file

---

## UNIT FLOW VERIFICATION

```
Yield spread (EOM yield minus matched treasury) → decimal (0.0150 = 150 bps) ✓
Compustat total_debt → millions ✓
CRSP price * shrout / 1000 → millions (matches Compustat) ✓
leverage = debt / (debt + mcap) → dimensionless [0, 1] ✓
Merton inversion → CIV in decimal (0.25 = 25% vol) ✓
Portfolio CIV → decimal ✓
Strategy PnL = position * CIV_change * 10000 → basis points ✓
Sharpe = mean(pnl) / std(pnl) * sqrt(freq) → annualized ✓
```

All units are consistent through the pipeline.

---

## RECOMMENDED FIXES BEFORE PUBLICATION

1. **Fix BIC calculation** -- use `np.sum(~np.isnan(y_all))` instead of `T * n_port`
2. **Verify Kim smoother** -- run forward-backward on a 3-state, 100-period synthetic example with known smoothed probabilities
3. **Add CIV range assertion** -- assert `np.nanmedian(y) < 2.0` after loading to catch unit errors early
4. **Move hardcoded bins to config.py** -- for maintainability, not correctness
