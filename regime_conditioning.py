"""
regime_conditioning.py -- Does the Markov-switching model earn its place?

The weekly mean-reversion signal (sign of last week's change in median
CIV) is model-free. This script asks whether the fitted regime model
adds anything: is mean reversion stronger in some volatility regimes,
and does conditioning position size on the filtered regime improve the
strategy?

Causality protocol:
  1. The 7-state model (parameters AND portfolio intercepts) is
     estimated only on the calibration window (weeks through CAL_END).
  2. The Hamilton filter then runs forward over the full weekly sample
     with frozen parameters. Only FILTERED probabilities are used
     (never the Kim smoother, which is two-sided).
  3. The conditioning variable at week t (filtered expected vol) uses
     data through the same Friday close that forms the trading signal.
  4. Regime-bucket edges are computed from calibration weeks only.
  5. Headline comparisons are reported on the out-of-sample window.

Inference: a circular rotation test. The conditioning series is rotated
against the P&L series by every offset >= MIN_SHIFT weeks; this
preserves the autocorrelation of both series while destroying their
alignment. The p-value is the fraction of rotations in which the
conditioned strategy beats the unconditional baseline by at least the
observed margin.

Run from the rsumplay root:  python markov_vol/regime_conditioning.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.config import RESULTS_DIR
from markov_vol.src.data_loader import load_daily_portfolios
from markov_vol.src.estimation import estimate_markov_civ
from markov_vol.src.hamilton_filter import hamilton_filter

CAL_END = pd.Timestamp("2012-12-31")
N_VOL = 7
MIN_SHIFT = 8  # weeks; rotations closer than this could retain alignment


def sharpe(pnl, freq=52):
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 5 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(freq))


def build_weekly_panel():
    y, dates, L, tau, labels = load_daily_portfolios()
    df = pd.DataFrame(y, index=pd.to_datetime(dates), columns=labels)
    weekly = df.resample("W-FRI").last().dropna(how="all")
    return weekly, L, tau


def fit_and_filter(weekly, L, tau):
    """Estimate on calibration weeks only; filter the full sample."""
    cal = weekly.loc[:CAL_END]
    print(f"Calibration: {cal.index[0].date()} -> {cal.index[-1].date()}"
          f"  ({len(cal)} weeks)")
    print(f"Full sample: {len(weekly)} weeks; OOS starts "
          f"{weekly.loc[weekly.index > CAL_END].index[0].date()}")

    est = estimate_markov_civ(cal.values, L, tau, n_vol=N_VOL, verbose=True)

    # Frozen parameters + calibration intercepts, filtered forward over
    # the full sample. hamilton_filter would re-estimate intercepts from
    # the data it is given, so pass the calibration intercepts in.
    filt = hamilton_filter(
        weekly.values, est["sigma_grid"], est["P"], L, tau,
        est["params"]["sigma_eta"], port_intercepts=est["port_intercepts"],
    )
    evol = filt["filtered_probs"] @ est["sigma_grid"]        # E[sigma_t | t]
    pstay = filt["filtered_probs"] @ np.diag(est["P"])       # E[P(stay) | t]
    return est, evol, pstay


def aligned_series(weekly, evol, pstay, lag=0):
    """
    Align signal, conditioning, and realized P&L leg.

    ch[t] is the change over the week ending at index t+1 of the weekly
    panel (np.diff convention). For lag 0 the position for week t+1 is
    formed from ch[t] and conditioning known at the same close
    (evol[t+1] in panel indexing, = filter through that Friday);
    realized leg is ch[t+1]. Lag 1 delays execution one week.
    """
    wm = weekly.median(axis=1).values
    ch = np.diff(wm)                       # ch[t] spans close t -> close t+1
    cond_evol = evol[1:]                   # conditioning at the forming close
    cond_pstay = pstay[1:]
    dates = weekly.index[1:]               # date of the forming close

    k = lag + 1
    sig = -np.sign(ch[:-k])
    real = ch[k:]
    n = min(len(sig), len(real))
    return {
        "sig": sig[:n],
        "real": real[:n] * 10000.0,        # CIV bps
        "evol": cond_evol[:n],
        "pstay": cond_pstay[:n],
        "dates": dates[:n],                # forming date of each position
    }


def tercile_edges(x, dates):
    cal_mask = dates <= CAL_END
    return np.quantile(x[cal_mask], [1 / 3, 2 / 3])


def bucket(x, edges):
    return np.digitize(x, edges)           # 0=low, 1=mid, 2=high


def ar1_by_bucket(weekly, evol):
    """Regime-dependent AR(1): ch[t+1] on ch[t], bucketed by evol at t."""
    wm = weekly.median(axis=1).values
    ch = np.diff(wm)
    cond = evol[1:]                        # at the close ending ch[t]
    dates = weekly.index[1:]
    edges = tercile_edges(cond[:-1], dates[:-1])
    b = bucket(cond[:-1], edges)
    out = {}
    for j, name in enumerate(["low", "mid", "high"]):
        m = b == j
        x, ynext = ch[:-1][m], ch[1:][m]
        beta = float(np.cov(x, ynext)[0, 1] / np.var(x))
        r = float(np.corrcoef(x, ynext)[0, 1])
        out[name] = {"n_weeks": int(m.sum()), "ar1_beta": round(beta, 3),
                     "corr": round(r, 3)}
    return out, edges


def run_strategies(al, edges):
    """P&L of unconditional and regime-conditioned variants."""
    b = bucket(al["evol"], edges)
    oos = al["dates"] > CAL_END
    base = al["sig"] * al["real"]

    def scaled(w):
        w = np.asarray(w, dtype=float)
        cal_mean = w[~oos].mean()          # normalize on calibration only
        return base * (w / cal_mean if cal_mean > 0 else w)

    strategies = {"unconditional": base}
    for j, name in enumerate(["low", "mid", "high"]):
        strategies[f"only_{name}"] = base * (b == j)
    strategies["inv_vol_scaled"] = scaled(1.0 / al["evol"])
    strategies["vol_scaled"] = scaled(al["evol"])
    strategies["pstay_scaled"] = scaled(al["pstay"])

    rows = {}
    for name, pnl in strategies.items():
        rows[name] = {
            "sharpe_full": round(sharpe(pnl), 3),
            "sharpe_oos": round(sharpe(pnl[oos]), 3),
            "mean_bps_oos": round(float(np.mean(pnl[oos])), 3),
        }
    return rows, strategies, oos


def rotation_test(al, edges, variant, oos):
    """
    Rotate the conditioning series against (sig, real) by every offset
    in [MIN_SHIFT, n - MIN_SHIFT]; recompute the OOS Sharpe difference
    (conditioned minus unconditional). One-sided p-value.
    """
    base = al["sig"] * al["real"]
    n = len(base)
    obs = None
    diffs = []
    for shift in range(0, n):
        if 0 < shift < MIN_SHIFT or shift > n - MIN_SHIFT:
            continue
        evol_s = np.roll(al["evol"], shift)
        pstay_s = np.roll(al["pstay"], shift)
        if variant == "only_high":
            pnl = base * (bucket(evol_s, edges) == 2)
        elif variant == "inv_vol_scaled":
            w = 1.0 / evol_s
            pnl = base * (w / w[~oos].mean())
        elif variant == "pstay_scaled":
            pnl = base * (pstay_s / pstay_s[~oos].mean())
        else:
            raise ValueError(variant)
        d = sharpe(pnl[oos]) - sharpe(base[oos])
        if shift == 0:
            obs = d
        else:
            diffs.append(d)
    diffs = np.array(diffs)
    p = float((np.sum(diffs >= obs) + 1) / (len(diffs) + 1))
    return {"observed_oos_sharpe_diff": round(obs, 3),
            "n_rotations": int(len(diffs)),
            "p_one_sided": round(p, 4)}


def main():
    weekly, L, tau = build_weekly_panel()
    est, evol, pstay = fit_and_filter(weekly, L, tau)

    print("\nFrozen parameters (calibration 2002-2012):")
    for k, v in est["params"].items():
        print(f"  {k:14s} {v:.4f}")

    ar1, edges = ar1_by_bucket(weekly, evol)
    print("\nAR(1) of weekly median-CIV changes by filtered-vol tercile")
    print("(tercile edges from calibration weeks only):")
    for name, row in ar1.items():
        print(f"  {name:5s} n={row['n_weeks']:4d}  beta={row['ar1_beta']:+.3f}"
              f"  corr={row['corr']:+.3f}")

    results = {"params": {k: round(v, 5) for k, v in est["params"].items()},
               "cal_end": str(CAL_END.date()),
               "tercile_edges": [round(float(e), 4) for e in edges],
               "ar1_by_bucket": ar1, "lags": {}}

    for lag in (0, 1):
        al = aligned_series(weekly, evol, pstay, lag=lag)
        rows, strategies, oos = run_strategies(al, edges)
        print(f"\n=== Lag {lag} ===  (OOS = {int(oos.sum())} weeks after "
              f"{CAL_END.date()})")
        print(f"{'strategy':18s} {'Sharpe full':>11s} {'Sharpe OOS':>11s}"
              f" {'mean bps OOS':>13s}")
        for name, row in rows.items():
            print(f"{name:18s} {row['sharpe_full']:11.3f}"
                  f" {row['sharpe_oos']:11.3f} {row['mean_bps_oos']:13.3f}")
        results["lags"][lag] = {"strategies": rows}

        if lag == 0:
            print("\nRotation test (OOS Sharpe difference vs unconditional):")
            for variant in ("only_high", "inv_vol_scaled", "pstay_scaled"):
                rt = rotation_test(al, edges, variant, oos)
                print(f"  {variant:16s} diff={rt['observed_oos_sharpe_diff']:+.3f}"
                      f"  p={rt['p_one_sided']:.4f}"
                      f"  ({rt['n_rotations']} rotations)")
                results["lags"][lag].setdefault("rotation_test", {})[variant] = rt

    out = RESULTS_DIR / "regime_conditioning.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
