"""
weekly_deep_dive.py -- Full development of weekly CIV mean-reversion strategy.
"""
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.src.data_loader import load_daily_portfolios


def sharpe(pnl, freq=52):
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(freq)


def main():
    y, dates, L, tau, labels = load_daily_portfolios()
    dates = pd.to_datetime(dates)

    df = pd.DataFrame(y, index=dates, columns=labels)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    weekly_median = weekly.median(axis=1)

    print("WEEKLY CIV MEAN-REVERSION: FULL DEVELOPMENT")
    print("=" * 70)
    print(f"Weekly data: {len(weekly)} weeks, {weekly.shape[1]} portfolios")
    print(f"Range: {weekly.index[0].date()} to {weekly.index[-1].date()}")
    print()
    print("DEFINITION:")
    print("  At Friday close, compute median CIV across 20 portfolios.")
    print("  Signal = -(CIV this Friday - CIV last Friday)")
    print("  Position = sign(signal): +1 if CIV fell, -1 if it rose.")
    print("  Hold one week. P&L = position * next week change * 10000 (bps)")

    # ================================================================
    # FULL SAMPLE
    # ================================================================
    wm = weekly_median.values
    wm_changes = np.diff(wm)
    n = len(wm_changes)

    print()
    print("=" * 70)
    print("PART 1: FULL SAMPLE (2002-2024)")
    print("=" * 70)
    print(f"Weeks: {n}")
    print(f"Weekly change: mean={np.mean(wm_changes)*10000:.1f} bps, "
          f"std={np.std(wm_changes)*10000:.1f} bps")

    print("Autocorrelation:")
    for k in range(1, 8):
        ac = np.corrcoef(wm_changes[k:], wm_changes[:-k])[0, 1]
        bar = "#" * int(abs(ac) * 40)
        print(f"  AC({k}): {ac:+.4f} {bar}")

    print()
    print(f"Lag   Sharpe    Hit     Wkly bps  Cum bps   N")
    print("-" * 52)
    for lag in [0, 1, 2, 4]:
        if lag == 0:
            sig = -wm_changes[:-1]
            real = wm_changes[1:]
        else:
            sig = -wm_changes[:-(lag + 1)]
            real = wm_changes[(lag + 1):]
        mn = min(len(sig), len(real))
        pos = np.sign(sig[:mn])
        pnl = pos * real[:mn] * 10000
        sr = sharpe(pnl)
        hit = np.mean(pos == np.sign(real[:mn]))
        print(f"{lag:<6}{sr:+.2f}     {hit:.1%}    {np.mean(pnl):+.2f}     {np.sum(pnl):+.0f}      {len(pnl)}")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    print()
    print("=" * 70)
    print("PART 2: WALK-FORWARD (260-wk cal, 52-wk test, 26-wk roll)")
    print("=" * 70)

    cal_wks = 260
    test_wks = 52
    windows = []

    for start in range(0, n - cal_wks - test_wks, 26):
        cal_end = start + cal_wks
        test_end = min(cal_end + test_wks, n)

        test_ch = wm_changes[cal_end:test_end]
        if len(test_ch) < 10:
            break

        sig = -test_ch[:-1]
        real = test_ch[1:]
        pos = np.sign(sig)
        pnl = pos * real * 10000
        sr = sharpe(pnl)
        hit = np.mean(pos == np.sign(real))

        sd = weekly.index[cal_end + 1].date()
        ed = weekly.index[min(test_end, len(weekly) - 1)].date()
        windows.append({"start": sd, "end": ed, "sharpe": sr,
                        "hit": hit, "cum": np.sum(pnl), "n": len(pnl)})

    print(f"Windows: {len(windows)}")
    print()
    for w in windows:
        print(f"  {w['start']} to {w['end']}: "
              f"Sharpe={w['sharpe']:+.2f}, hit={w['hit']:.1%}, cum={w['cum']:+.0f} bps")

    srs = [w["sharpe"] for w in windows]
    print()
    print(f"Mean Sharpe: {np.mean(srs):+.2f}")
    print(f"Median Sharpe: {np.median(srs):+.2f}")
    print(f"Positive in {np.mean(np.array(srs) > 0):.0%} of windows")
    print(f"Range: [{np.min(srs):+.2f}, {np.max(srs):+.2f}]")

    # ================================================================
    # FIXED SPLIT
    # ================================================================
    print()
    print("=" * 70)
    print("PART 3: FIXED SPLIT (cal 2002-2023, OOS 2024)")
    print("=" * 70)

    for name, mask in [("CAL 2002-2023", weekly.index < "2024-01-01"),
                        ("OOS 2024", weekly.index >= "2024-01-01")]:
        s = weekly_median[mask].values
        ch = np.diff(s)
        if len(ch) < 5:
            print(f"  {name}: too few weeks")
            continue
        print(f"\n  {name}: {len(s)} weeks")
        for lag in [0, 1, 2]:
            if lag == 0:
                sig = -ch[:-1]
                real = ch[1:]
            else:
                if len(ch) <= lag + 1:
                    continue
                sig = -ch[:-(lag + 1)]
                real = ch[(lag + 1):]
            mn = min(len(sig), len(real))
            pos = np.sign(sig[:mn])
            pnl = pos * real[:mn] * 10000
            sr = sharpe(pnl)
            hit = np.mean(pos == np.sign(real[:mn]))
            print(f"    Lag {lag}: Sharpe={sr:+.2f}, hit={hit:.1%}, "
                  f"cum={np.sum(pnl):+.0f} bps, n={len(pnl)}")

    # ================================================================
    # DOLLAR ECONOMICS
    # ================================================================
    print()
    print("=" * 70)
    print("PART 4: DOLLAR ECONOMICS (based on OOS 2024)")
    print("=" * 70)

    oos_wm = weekly_median[weekly.index >= "2024-01-01"].values
    oos_ch = np.diff(oos_wm)
    sig = -oos_ch[:-1]
    pos = np.sign(sig)
    pnl = pos * oos_ch[1:] * 10000
    mean_wk = np.mean(pnl)
    std_wk = np.std(pnl)
    ann = mean_wk * 52

    print(f"Weekly: mean={mean_wk:.1f} CIV bps, std={std_wk:.1f} CIV bps")
    print(f"Annual gross: {ann:.0f} CIV bps")
    print()
    print("IMPORTANT: these are CIV bps, NOT spread bps. Converting CIV to")
    print("tradeable spread P&L requires the Merton vega (~0.07 at median")
    print("leverage/maturity), a ~14x compression. In spread space the")
    print("strategy is roughly breakeven after 2bp CDX transaction costs.")
    print("See unit_analysis.py for the full vega-corrected dollar economics")
    print("and model-free Sharpe-based sizing.")

    # ================================================================
    # PER-PORTFOLIO
    # ================================================================
    print()
    print("=" * 70)
    print("PART 5: PER-PORTFOLIO WEEKLY (OOS 2024)")
    print("=" * 70)

    wk_oos = weekly[weekly.index >= "2024-01-01"]
    results = []
    for col in wk_oos.columns:
        s = pd.Series(wk_oos[col].values).ffill().dropna().values  # ffill only: bfill would leak future values into the start
        ch = np.diff(s)
        if len(ch) < 5:
            continue
        sig = -ch[:-1]
        real = ch[1:]
        pos = np.sign(sig)
        pnl = pos * real * 10000
        sr = sharpe(pnl)
        hit = np.mean(pos == np.sign(real))
        results.append({"name": col, "sharpe": sr, "hit": hit, "cum": np.sum(pnl)})

    print(f"{'Port':<15} {'Sharpe':<10} {'Hit':<10} {'Cum bps':<10}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: -x["sharpe"]):
        print(f"{r['name']:<15} {r['sharpe']:+.2f}      {r['hit']:.1%}     {r['cum']:+.0f}")

    # Diversified
    all_pnl = []
    for col in wk_oos.columns:
        s = pd.Series(wk_oos[col].values).ffill().dropna().values  # ffill only: bfill would leak future values into the start
        ch = np.diff(s)
        sig = -ch[:-1]
        pos = np.sign(sig)
        pnl = pos * ch[1:] * 10000
        all_pnl.append(pnl)

    ml = min(len(p) for p in all_pnl)
    div_pnl = np.mean([p[:ml] for p in all_pnl], axis=0)
    print(f"\nDiversified (20-port avg): Sharpe={sharpe(div_pnl):+.2f}")

    # Diversified lag test
    for lag in [0, 1]:
        port_pnls = []
        for col in wk_oos.columns:
            s = pd.Series(wk_oos[col].values).ffill().dropna().values  # ffill only: bfill would leak future values into the start
            ch = np.diff(s)
            if lag == 0:
                sig = -ch[:-1]
                real = ch[1:]
            else:
                if len(ch) <= lag + 1:
                    continue
                sig = -ch[:-(lag + 1)]
                real = ch[(lag + 1):]
            mn = min(len(sig), len(real))
            pos = np.sign(sig[:mn])
            pnl = pos * real[:mn] * 10000
            port_pnls.append(pnl)
        ml2 = min(len(p) for p in port_pnls)
        avg2 = np.mean([p[:ml2] for p in port_pnls], axis=0)
        print(f"  Diversified lag {lag}: Sharpe={sharpe(avg2):+.2f}")


if __name__ == "__main__":
    main()
