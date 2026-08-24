"""B3: does the regime model explain when the cross-sectional CIV
premium pays?

Bridge between Track A (7-state Markov-switching vol model, calibrated
2002-2012, Hamilton FILTERED probabilities only) and B2 (issuer-level
monthly CIV quintile long-shorts, carry-aware). For each B2 long-short
series, split months by the filtered expected-vol tercile at the signal
formation close (edges from calibration months only) and test whether
the premium differs across regimes with a circular-rotation test on the
high-minus-low mean difference.

Everything is causal: filtered (never smoothed) probabilities, monthly
conditioning = last weekly filtered value at or before the signal
month-end, tercile edges frozen on calibration data.

Run from rsumplay root: python markov_vol/regime_bridge.py
Output: markov_vol/data/results/regime_bridge.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from markov_vol.regime_conditioning import (build_weekly_panel,  # noqa: E402
                                            fit_and_filter, CAL_END)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data" / "results"
LS_CSV = RESULTS / "cross_section_monthly_ls.csv"

NW_LAGS = 6
MIN_SHIFT = 6      # months; rotations closer could retain alignment
N_KEYS = ["civ_t+1_ret_ex_xc_EW", "civ_t+2_skip_ret_ex_xc_EW",
          "civ_t+1_ret_EW", "dciv_t+2_skip_ret_ex_xc_EW",
          "civ_orth_t+1_ret_ex_xc_EW", "civ_orth_t+2_skip_ret_ex_xc_EW"]


def nw_mean(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 12:
        return {"mean_ann_pct": np.nan, "t_nw": np.nan,
                "n_months": int(len(s))}
    m = sm.OLS(s.values, np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"mean_ann_pct": round(float(s.mean() * 1200), 2),
            "t_nw": round(float(m.tvalues[0]), 2), "n_months": int(len(s))}


def monthly_filtered_evol() -> pd.Series:
    """Filtered E[sigma_t | t], last weekly value in each month."""
    weekly, L, tau = build_weekly_panel()
    est, evol, pstay = fit_and_filter(weekly, L, tau)
    s = pd.Series(evol, index=weekly.index)
    m = s.resample("ME").last()
    m.index = m.index.to_period("M")
    return m


def main() -> None:
    evol_m = monthly_filtered_evol()
    cal = evol_m[evol_m.index <= pd.Period(CAL_END, "M")]
    edges = np.quantile(cal.values, [1 / 3, 2 / 3])
    b = pd.Series(np.digitize(evol_m.values, edges), index=evol_m.index)
    print(f"Monthly conditioning: {len(evol_m)} months, tercile edges "
          f"(cal only) {edges.round(4).tolist()}, bucket counts "
          f"{np.bincount(b.values).tolist()}")

    ls = pd.read_csv(LS_CSV, index_col=0)
    ls.index = pd.PeriodIndex(ls.index, freq="M")

    results = {"tercile_edges": edges.round(5).tolist(),
               "conditioning": "filtered E[sigma|t], last weekly value at "
                               "or before signal month-end; edges from "
                               "calibration months (<=2012-12) only",
               "by_key": {}}
    rng = np.random.default_rng(42)

    for key in N_KEYS:
        s = ls[key].dropna()
        bb = b.reindex(s.index)
        ok = bb.notna()
        s, bb = s[ok], bb[ok].astype(int)
        entry = {"unconditional": nw_mean(s)}
        for name, code in [("low", 0), ("mid", 1), ("high", 2)]:
            entry[name] = nw_mean(s[bb == code])
        # Is the high bucket just the GFC (and COVID)? Drop both windows.
        crisis = ((s.index >= pd.Period("2008-07", "M"))
                  & (s.index <= pd.Period("2009-06", "M"))) \
            | ((s.index >= pd.Period("2020-02", "M"))
               & (s.index <= pd.Period("2020-12", "M")))
        entry["high_ex_crisis"] = nw_mean(s[(bb == 2) & ~crisis])
        hi = s[bb == 2].mean()
        lo = s[bb == 0].mean()
        obs = hi - lo

        # Circular-rotation test: rotate the bucket series against the
        # P&L by every admissible offset, recompute high-minus-low mean.
        n = len(s)
        bv, sv = bb.values, s.values
        diffs = []
        for k in range(MIN_SHIFT, n - MIN_SHIFT):
            br = np.roll(bv, k)
            m_hi = sv[br == 2].mean() if (br == 2).any() else np.nan
            m_lo = sv[br == 0].mean() if (br == 0).any() else np.nan
            diffs.append(m_hi - m_lo)
        diffs = np.array(diffs)
        p = float(np.mean(np.abs(diffs) >= abs(obs)))
        entry["high_minus_low"] = {
            "mean_ann_pct": round(float(obs * 1200), 2),
            "rotation_p_two_sided": round(p, 4),
            "n_rotations": int(len(diffs))}
        results["by_key"][key] = entry
        u = entry["unconditional"]
        print(f"\nB3 {key}  (uncond {u['mean_ann_pct']:+.2f}%/yr, "
              f"t {u['t_nw']:.2f})")
        for name in ["low", "mid", "high", "high_ex_crisis"]:
            e = entry[name]
            print(f"   {name:14s} {e['mean_ann_pct']:+7.2f}%/yr "
                  f"(t={e['t_nw']:5.2f}, n={e['n_months']})")
        print(f"   high-low {obs * 1200:+.2f}%/yr, rotation p = {p:.4f}")

    cond = pd.DataFrame({"evol": evol_m, "bucket": b})
    cond.index = cond.index.astype(str)
    cond.to_csv(RESULTS / "regime_bridge_conditioning.csv")
    with open(RESULTS / "regime_bridge.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {RESULTS / 'regime_bridge.json'} + conditioning csv")


if __name__ == "__main__":
    main()
