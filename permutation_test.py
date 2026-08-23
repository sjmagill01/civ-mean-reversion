"""
permutation_test.py -- Significance test for the weekly CIV mean-reversion signal.

Null hypothesis: weekly median-CIV changes have no serial dependence, so the
sign-of-last-change rule has no edge. We destroy the time ordering by randomly
permuting the sequence of weekly changes (which preserves the marginal
distribution exactly) and recompute the strategy Sharpe 10,000 times.

p-value = fraction of permuted Sharpes >= observed Sharpe (one-sided).

Tested at lag 0 (trade immediately) and lag 1 (one-week execution lag),
matching the headline claims (Sharpe 1.05 and 0.35 respectively).

Run: python permutation_test.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.src.data_loader import load_daily_portfolios

N_PERM = 10_000
SEED = 42


def strategy_sharpe(changes, lag):
    """Sharpe of -sign(last change) applied at the given execution lag."""
    if lag == 0:
        sig = -changes[:-1]
        real = changes[1:]
    else:
        sig = -changes[:-(lag + 1)]
        real = changes[(lag + 1):]
    mn = min(len(sig), len(real))
    pnl = np.sign(sig[:mn]) * real[:mn]
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(52)


def main():
    y, dates, L, tau, labels = load_daily_portfolios()
    dates = pd.to_datetime(dates)

    df = pd.DataFrame(y, index=dates, columns=labels)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    wm = weekly.median(axis=1).values
    changes = np.diff(wm)

    print("PERMUTATION TEST: WEEKLY CIV MEAN-REVERSION")
    print("=" * 60)
    print(f"Weeks: {len(wm)}  ({weekly.index[0].date()} to {weekly.index[-1].date()})")
    print(f"Permutations: {N_PERM:,}  (seed {SEED})")
    print()
    print("Null: weekly changes i.i.d. (shuffled order, same marginal dist)")
    print()

    rng = np.random.default_rng(SEED)
    for lag in [0, 1]:
        obs = strategy_sharpe(changes, lag)
        perm = np.empty(N_PERM)
        for i in range(N_PERM):
            perm[i] = strategy_sharpe(rng.permutation(changes), lag)
        # one-sided p with add-one correction
        p = (1 + np.sum(perm >= obs)) / (1 + N_PERM)
        print(f"Lag {lag}: observed Sharpe = {obs:+.3f}")
        print(f"        null Sharpe: mean {np.mean(perm):+.3f}, "
              f"sd {np.std(perm):.3f}, 95th pct {np.percentile(perm, 95):+.3f}, "
              f"99th pct {np.percentile(perm, 99):+.3f}")
        print(f"        one-sided p = {p:.4f}")
        print()


if __name__ == "__main__":
    main()
