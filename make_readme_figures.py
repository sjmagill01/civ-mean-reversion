"""
make_readme_figures.py -- Figures embedded in README.md.

Left panel: cumulative P&L (CIV bps) of the weekly median-CIV
mean-reversion signal at lag 0 and lag 1, full sample.
Right panel: Sharpe by walk-forward window (260-week calibration,
52-week test, 26-week roll; windows overlap 50%).

Uses the identical signal construction as weekly_deep_dive.py.
Writes figures/fig_weekly_signal.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.src.data_loader import load_daily_portfolios

FIGDIR = Path(__file__).parent / "figures"


def weekly_pnl(wm_changes, lag):
    if lag == 0:
        sig = -wm_changes[:-1]
        real = wm_changes[1:]
    else:
        sig = -wm_changes[:-(lag + 1)]
        real = wm_changes[(lag + 1):]
    mn = min(len(sig), len(real))
    return np.sign(sig[:mn]) * real[:mn] * 10000


def sharpe(pnl, freq=52):
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(freq)


def main():
    FIGDIR.mkdir(exist_ok=True)
    y, dates, L, tau, labels = load_daily_portfolios()
    df = pd.DataFrame(y, index=pd.to_datetime(dates), columns=labels)
    weekly = df.resample("W-FRI").last().dropna(how="all")
    wm = weekly.median(axis=1).values
    wm_changes = np.diff(wm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- cumulative P&L ---
    for lag, color in [(0, "#1a3a5c"), (1, "#c0392b")]:
        pnl = weekly_pnl(wm_changes, lag)
        idx = weekly.index[-len(pnl):]
        ax1.plot(idx, np.cumsum(pnl), color=color, lw=1.5,
                 label=f"Lag {lag}: Sharpe {sharpe(pnl):.2f}")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylabel("Cumulative P&L (CIV bps)")
    ax1.set_title("Weekly CIV mean-reversion, 2002-2024")
    ax1.legend(frameon=False)

    # --- walk-forward Sharpes (identical loop to weekly_deep_dive.py) ---
    cal_wks, test_wks = 260, 52
    n = len(wm_changes)
    srs, mids = [], []
    for start in range(0, n - cal_wks - test_wks, 26):
        cal_end = start + cal_wks
        test_end = min(cal_end + test_wks, n)
        test_ch = wm_changes[cal_end:test_end]
        if len(test_ch) < 10:
            break
        pnl = np.sign(-test_ch[:-1]) * test_ch[1:] * 10000
        srs.append(sharpe(pnl))
        mids.append(weekly.index[cal_end + len(test_ch) // 2])
    colors = ["#1a3a5c" if s > 0 else "#c0392b" for s in srs]
    ax2.bar(mids, srs, width=150, color=colors)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("Window Sharpe (lag 0)")
    pos = sum(s > 0 for s in srs)
    ax2.set_title(f"Walk-forward: positive in {pos} of {len(srs)} windows")

    fig.tight_layout()
    out = FIGDIR / "fig_weekly_signal.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  ({len(srs)} windows, {pos} positive)")


if __name__ == "__main__":
    main()
