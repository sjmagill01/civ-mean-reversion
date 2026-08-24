"""
cross_section_mr.py -- B1: cross-sectional weekly mean reversion in CIV space.

The headline weekly signal is a TIME-SERIES bet: sign of last week's
change in the cross-portfolio median. This script asks the orthogonal
question: do RELATIVE moves across the 20 portfolios revert? Each week,
rank last week's cross-sectionally demeaned CIV change; go long the
portfolios that cheapened relative to the surface and short the ones
that richened. Demeaning removes the common (median) move, so this bet
is distinct by construction from the time-series signal.

Strategies (all P&L in CIV bps of the weekly change, gross = 2):
  xsect_rank_change   linear rank weights on last week's demeaned dCIV
  xsect_sign_change   EW long bottom-5 / short top-5 by demeaned dCIV
  xsect_level_z       rank weights on demeaned LEVEL (analog of the
                      existing cross_sectional_dispersion strategy,
                      which sorts on level z-scores, not changes)

Checks: lag-1 execution delay, calendar-year walk-forward, within-week
permutation test (weights permuted across portfolios each week
independently; preserves each series' time structure, destroys the
cross-sectional alignment), correlation with the time-series baseline.

Units caveat: CIV bps are NOT spread bps; Merton vega ~0.07 compresses
economics roughly 14x. Sharpe is the meaningful number.

Run from the rsumplay root:  python markov_vol/cross_section_mr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.config import RESULTS_DIR
from markov_vol.src.data_loader import load_daily_portfolios

N_PERM = 2000
SEED = 42


def sharpe(pnl, freq=52):
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(freq))


def build_weekly_panel():
    y, dates, L, tau, labels = load_daily_portfolios()
    df = pd.DataFrame(y, index=pd.to_datetime(dates), columns=labels)
    weekly = df.resample("W-FRI").last().dropna(how="all")
    assert not weekly.isna().any().any(), "weekly panel has NaNs"
    return weekly


def rank_weights(x):
    """Linear weights from ranks of x, mean-zero, sum|w| = 2 (short high x)."""
    r = x.argsort().argsort().astype(float)          # 0..N-1
    w = -(r - r.mean())
    return w / np.abs(w).sum() * 2.0


def sign_weights(x, k=5):
    """EW long bottom-k, short top-k of x, gross 2."""
    order = np.argsort(x)
    w = np.zeros(len(x))
    w[order[:k]] = 1.0 / k
    w[order[-k:]] = -1.0 / k
    return w


def build_positions(weekly):
    """
    d[t] = CIV change over the week ending at panel index t+1.
    Position for the NEXT week is formed from the demeaned d[t]
    (lag 0: realized leg is d[t+1]; lag 1 delays one more week).
    Returns dict of (T-1, N) weight arrays plus the change matrix.
    """
    d = np.diff(weekly.values, axis=0)               # (T-1, N)
    dm = d - d.mean(axis=1, keepdims=True)           # remove common move
    lev = weekly.values[1:]                          # level at forming close
    lev_dm = lev - lev.mean(axis=1, keepdims=True)

    W = {"xsect_rank_change": np.array([rank_weights(row) for row in dm]),
         "xsect_sign_change": np.array([sign_weights(row) for row in dm]),
         "xsect_level_z": np.array([rank_weights(row) for row in lev_dm])}
    return W, d


def pnl_series(w, d, lag=0):
    """w[t] formed from info through change t; realized leg d[t+1+lag]."""
    k = lag + 1
    return np.einsum("ij,ij->i", w[:-k], d[k:]) * 1e4   # CIV bps


def yearly_walkforward(pnl, years):
    out = {}
    for yr in sorted(set(years)):
        m = years == yr
        if m.sum() >= 20:
            out[int(yr)] = round(sharpe(pnl[m]), 2)
    pos = sum(1 for v in out.values() if v > 0)
    return out, f"{pos}/{len(out)}"


def permutation_test(w, d, rng, lag=0):
    """Permute each week's weight vector across portfolios independently."""
    k = lag + 1
    obs = sharpe(pnl_series(w, d, lag=lag))
    n_t, n_p = w[:-k].shape
    perm_sharpes = np.empty(N_PERM)
    real = d[k:] * 1e4
    for p in range(N_PERM):
        idx = np.argsort(rng.random((n_t, n_p)), axis=1)
        wp = np.take_along_axis(w[:-k], idx, axis=1)
        perm_sharpes[p] = sharpe(np.einsum("ij,ij->i", wp, real))
    pval = float((np.sum(perm_sharpes >= obs) + 1) / (N_PERM + 1))
    return {"observed_sharpe": round(obs, 3), "p_one_sided": round(pval, 4),
            "n_perm": N_PERM}


def main():
    rng = np.random.default_rng(SEED)
    weekly = build_weekly_panel()
    print(f"Weekly panel: {len(weekly)} weeks x {weekly.shape[1]} portfolios, "
          f"{weekly.index[0].date()} -> {weekly.index[-1].date()}")

    W, d = build_positions(weekly)
    dates = weekly.index[1:]                          # forming close of d[t]

    # Time-series baseline on the same panel, for the correlation check
    wm = np.diff(weekly.median(axis=1).values)
    ts_pnl = -np.sign(wm[:-1]) * wm[1:] * 1e4

    results = {}
    for name, w in W.items():
        results[name] = {}
        for lag in (0, 1):
            pnl = pnl_series(w, d, lag=lag)
            yrs = dates[lag + 1:].year.values
            wf, wf_str = yearly_walkforward(pnl, yrs)
            entry = {"sharpe": round(sharpe(pnl), 3),
                     "mean_bps": round(float(np.mean(pnl)), 3),
                     "walkforward_positive": wf_str}
            if lag == 0:
                entry["walkforward_by_year"] = wf
                entry["corr_with_timeseries_pnl"] = round(
                    float(np.corrcoef(pnl, ts_pnl)[0, 1]), 3)
            entry["permutation"] = permutation_test(w, d, rng, lag=lag)
            results[name][f"lag{lag}"] = entry
            print(f"{name:20s} lag{lag}  Sharpe {entry['sharpe']:6.3f}  "
                  f"mean {entry['mean_bps']:7.3f} bps  WF {wf_str}"
                  f"  perm p={entry['permutation']['p_one_sided']:.4f}"
                  + (f"  corr_ts {entry['corr_with_timeseries_pnl']:+.3f}"
                     if lag == 0 else ""))
        results[name]["lag_decay_sharpe"] = {
            lag: round(sharpe(pnl_series(w, d, lag=lag)), 3)
            for lag in range(6)}
        print(f"{'':20s} decay 0-5: "
              + " ".join(f"{v:+.2f}" for v in
                         results[name]["lag_decay_sharpe"].values()))

    results["units_caveat"] = ("P&L in CIV bps, not spread bps; Merton vega "
                               "~0.07 compresses economics ~14x. Sharpe is "
                               "the meaningful statistic.")
    results["construction_caveat"] = (
        "The 20 portfolios are LOWESS-interpolated, so week-t estimation "
        "noise reverts mechanically at t+1; the lag-0 Sharpe is inflated by "
        "construction. Lag-1 is the honest headline.")
    out = RESULTS_DIR / "cross_section_mr.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
