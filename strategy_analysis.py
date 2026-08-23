"""
strategy_analysis.py -- Careful, complete strategy analysis.

Every strategy:
  - Precisely defined (signal, position, P&L)
  - Tested at lags 0, 1, 2, 5
  - Autocorrelation of the underlying signal
  - In-sample (2018-2023) and OOS (2024)
  - Transaction cost breakeven
"""
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def sharpe(pnl, freq=252):
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(freq)


def autocorrelations(changes, max_lag=5):
    """Compute autocorrelations of a series of changes."""
    acs = {}
    for k in range(1, max_lag + 1):
        if len(changes) > k + 1:
            acs[k] = np.corrcoef(changes[k:], changes[:-k])[0, 1]
        else:
            acs[k] = np.nan
    return acs


def test_mean_reversion(series, name, freq=252):
    """
    Test mean-reversion of a series at multiple lags.

    Signal at end of day t: delta_t = series[t] - series[t-1]
    Position for day t+lag+1: -sign(delta_t)
    P&L: position * (series[t+lag+1] - series[t+lag])

    lag=0: signal today, trade today-to-tomorrow (standard)
    lag=1: signal today, trade tomorrow-to-day-after (skip-a-day)
    """
    changes = np.diff(series)
    n = len(changes)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Series length: {len(series)} days")
    print(f"  Daily change: mean={np.mean(changes)*10000:+.2f} bps, "
          f"std={np.std(changes)*10000:.1f} bps")

    # Autocorrelation
    acs = autocorrelations(changes)
    print(f"  Autocorrelation of daily changes:")
    for k, ac in acs.items():
        bar = '#' * int(abs(ac) * 40)
        print(f"    AC({k}): {ac:+.4f} {'|' + bar}")

    # Mean-reversion at each lag
    print(f"\n  {'Lag':<6} {'Sharpe':>8} {'Hit%':>8} {'DailyBps':>10} {'CumBps':>10} {'N':>6}")
    print(f"  {'-'*50}")

    results = {}
    for lag in [0, 1, 2, 5]:
        # Signal: change from t-1 to t
        # Position formed at end of day t, for execution starting at t+lag
        # P&L = position * change from t+lag to t+lag+1
        if lag == 0:
            signal = -changes[:-1]        # signal at t, from change t-1 to t
            realized = changes[1:]         # change from t to t+1
        else:
            if n <= lag + 1:
                continue
            signal = -changes[:-(lag + 1)]
            realized = changes[(lag + 1):]

        min_len = min(len(signal), len(realized))
        signal = signal[:min_len]
        realized = realized[:min_len]

        pos = np.sign(signal)
        pnl = pos * realized * 10000

        sr = sharpe(pnl, freq=freq)
        hit = np.mean(pos == np.sign(realized))
        daily_bps = np.mean(pnl)
        cum_bps = np.sum(pnl)

        results[lag] = {
            'sharpe': sr, 'hit': hit,
            'daily_bps': daily_bps, 'cum_bps': cum_bps,
            'n': len(pnl)
        }

        print(f"  {lag:<6} {sr:>+8.2f} {hit:>7.1%} {daily_bps:>+10.3f} {cum_bps:>+10.1f} {len(pnl):>6}")

    # Verdict
    lag0 = results.get(0, {}).get('sharpe', 0)
    lag1 = results.get(1, {}).get('sharpe', 0)

    print()
    if abs(lag0) < 0.5:
        print(f"  VERDICT: No signal at lag 0.")
    elif abs(lag1) < 0.3 * abs(lag0):
        print(f"  VERDICT: Signal does not survive 1-day lag "
              f"({lag0:+.2f} -> {lag1:+.2f}). Likely microstructure.")
    elif np.sign(lag0) != np.sign(lag1):
        print(f"  VERDICT: Signal FLIPS sign at lag 1 "
              f"({lag0:+.2f} -> {lag1:+.2f}). Alternating pattern.")
    else:
        retention = lag1 / lag0 if lag0 != 0 else 0
        print(f"  VERDICT: Signal survives lag 1 "
              f"({lag0:+.2f} -> {lag1:+.2f}, {retention:.0%} retained). "
              f"Potentially tradeable.")

    # Breakeven
    if results.get(0, {}).get('daily_bps', 0) > 0:
        print(f"  Breakeven round-trip cost: {results[0]['daily_bps']:.1f} bps")

    return results


def main():
    from markov_vol.src.data_loader import load_daily_portfolios

    y, dates, L, tau, labels = load_daily_portfolios()
    dates = pd.to_datetime(dates)

    # Parse leverage
    low_idx = []
    high_idx = []
    for i, c in enumerate(labels):
        m = re.search(r'L([\d.]+)', c)
        if m:
            lev = float(m.group(1))
            if lev <= 0.25:
                low_idx.append(i)
            elif lev >= 0.75:
                high_idx.append(i)

    # Split
    cal_mask = (dates >= '2018-01-01') & (dates <= '2023-12-31')
    oos_mask = (dates >= '2024-01-01') & (dates <= '2024-12-31')

    y_cal, y_oos = y[cal_mask], y[oos_mask]
    d_cal, d_oos = dates[cal_mask], dates[oos_mask]

    print("=" * 60)
    print("STRATEGY ANALYSIS")
    print("=" * 60)
    print(f"Calibration: {cal_mask.sum()} days ({d_cal[0].date()} to {d_cal[-1].date()})")
    print(f"OOS:         {oos_mask.sum()} days ({d_oos[0].date()} to {d_oos[-1].date()})")

    # Build series for each strategy
    for period_name, y_p, dates_p in [("CALIBRATION (2018-2023)", y_cal, d_cal),
                                        ("OUT-OF-SAMPLE (2024)", y_oos, d_oos)]:
        print(f"\n\n{'#'*60}")
        print(f"  {period_name}")
        print(f"{'#'*60}")

        # Series
        median_civ = np.nanmedian(y_p, axis=1)
        low_civ = np.nanmedian(y_p[:, low_idx], axis=1)
        high_civ = np.nanmedian(y_p[:, high_idx], axis=1)
        gap = low_civ - high_civ

        # Strategy 1: CIV Median Mean-Reversion
        test_mean_reversion(median_civ,
            "STRATEGY 1: CIV Median Mean-Reversion\n"
            "  Signal: median CIV change across 20 portfolios\n"
            "  Position: -sign(change), bet on reversal")

        # Strategy 2: Leverage Spread Mean-Reversion
        test_mean_reversion(gap,
            "STRATEGY 2: Leverage Spread Mean-Reversion\n"
            "  Signal: change in gap between low-lev and high-lev CIV\n"
            "  Position: -sign(gap change), bet on gap convergence\n"
            f"  Low-lev ports: {[labels[i] for i in low_idx[:2]]}...\n"
            f"  High-lev ports: {[labels[i] for i in high_idx[:2]]}...")

        # Strategy 3: Per-Portfolio Mean-Reversion (averaged)
        n_port = y_p.shape[1]
        T = len(y_p)
        all_pnl = {lag: [] for lag in [0, 1, 2, 5]}

        for lag in [0, 1, 2, 5]:
            port_pnls = []
            for p in range(n_port):
                series = y_p[:, p]
                valid = ~np.isnan(series)
                if valid.sum() < 20:
                    continue
                # Fill NaN with forward fill for continuity
                s = pd.Series(series).ffill().dropna().values  # ffill only: bfill would leak future values into the start
                changes = np.diff(s)

                if lag == 0:
                    sig = -changes[:-1]
                    real = changes[1:]
                else:
                    if len(changes) <= lag + 1:
                        continue
                    sig = -changes[:-(lag + 1)]
                    real = changes[(lag + 1):]

                min_len = min(len(sig), len(real))
                pos = np.sign(sig[:min_len])
                pnl = pos * real[:min_len] * 10000
                port_pnls.append(pnl)

            if port_pnls:
                # Average P&L across portfolios (equal weight)
                min_len = min(len(p) for p in port_pnls)
                avg_pnl = np.mean([p[:min_len] for p in port_pnls], axis=0)
                all_pnl[lag] = avg_pnl

        print(f"\n{'='*60}")
        print(f"  STRATEGY 3: Per-Portfolio Mean-Reversion (20 portfolios averaged)")
        print(f"  Signal: each portfolio's own CIV change")
        print(f"  Position: -sign(change) per portfolio, average P&L")
        print(f"{'='*60}")
        print(f"\n  {'Lag':<6} {'Sharpe':>8} {'Hit%':>8} {'DailyBps':>10} {'CumBps':>10}")
        print(f"  {'-'*44}")
        for lag in [0, 1, 2, 5]:
            pnl = all_pnl[lag]
            if len(pnl) > 5:
                sr = sharpe(pnl)
                hit = np.mean(pnl > 0)
                print(f"  {lag:<6} {sr:>+8.2f} {hit:>7.1%} {np.mean(pnl):>+10.3f} {np.sum(pnl):>+10.1f}")

        # Strategy 4: Weekly Mean-Reversion
        # Resample to weekly (Friday close)
        df_temp = pd.DataFrame({'civ': median_civ, 'gap': gap}, index=dates_p)
        weekly = df_temp.resample('W-FRI').last().dropna()

        if len(weekly) > 10:
            test_mean_reversion(weekly['civ'].values,
                "STRATEGY 4: Weekly CIV Mean-Reversion\n"
                "  Signal: Friday-to-Friday median CIV change\n"
                "  Position: -sign(change), hold for one week\n"
                "  Purpose: avoids daily microstructure entirely",
                freq=52)

            test_mean_reversion(weekly['gap'].values,
                "STRATEGY 5: Weekly Leverage Spread Mean-Reversion\n"
                "  Signal: Friday-to-Friday gap change\n"
                "  Position: -sign(gap change), hold for one week",
                freq=52)

        # Strategy 6: Equity Benchmark
        try:
            import os
            crsp = pd.read_parquet(os.environ.get(
                "CRSP_DAILY_PATH",
                str(Path(__file__).parent / "data" / "crsp_daily.parquet")))
            crsp['date'] = pd.to_datetime(crsp['date'])
            eq = crsp.groupby('date')['ret'].median()
            eq_vals = eq.reindex(dates_p).values

            # Equity mean-reversion
            eq_valid = ~np.isnan(eq_vals)
            eq_series = pd.Series(eq_vals).ffill().dropna().values  # ffill only: bfill would leak future values into the start
            eq_cumret = np.cumsum(eq_series)  # cumulative return as a "level"

            test_mean_reversion(eq_cumret,
                "STRATEGY 6: Equity Return Mean-Reversion (benchmark)\n"
                "  Signal: yesterday's equity return (median across firms)\n"
                "  Position: -sign(return), bet on reversal\n"
                "  Purpose: if CIV mean-rev = equity mean-rev, they should match")

        except Exception as e:
            print(f"\n  Equity benchmark skipped: {e}")


if __name__ == '__main__':
    main()
