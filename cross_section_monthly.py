"""B2: issuer-level monthly CIV sorts against next-month bond returns,
carry-aware from the start.

A bond's credit return decomposes as

    excess return = carry (spread/12) - duration x (spread change) + small,

and carry is deterministic, so every sort here is reported on three
targets:

    ret        total return (ret_eom, amount-weighted bond->firm,
               winsorized at +/-50%)
    ret_ex_xc  rate-hedged ex-carry return:
               ret - rf/12 + D x d_rf - s/12, with D the amount-weighted
               bondret modified duration and d_rf the change in the
               firm's maturity-matched treasury yield over the return
               month (all conditioning info dated the month before the
               return month)
    ds         next-month spread change, bps/month (the only component a
               signal can genuinely forecast)

Signals at t: firm CIV level (Merton inversion of the yield-spread
panel, mean across maturity buckets) and dCIV (one-month change).
Quintile Q5-Q1 sorts, EW and VW, at t+1 and skip-a-month; double sorts
within IG/HY and maturity terciles; pre/post-2013 subperiods; 1,000-draw
within-month permutation test and walk-forward year count on the
headline.

Run from rsumplay root: python markov_vol/cross_section_monthly.py
Outputs: markov_vol/data/results/cross_section_monthly.json
         markov_vol/data/results/cross_section_monthly_ls.csv (for B3)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from markov_vol.src.merton import invert_spread_to_civ_vec  # noqa: E402

HERE = Path(__file__).resolve().parent
IMPC = HERE.parent / "impcredvol"
PANEL = Path(os.environ.get("CIV_PANEL_PATH",
                            IMPC / "data" / "wrds_panel_yieldspread.parquet"))
RET = IMPC / "wrds_data" / "raw" / "bondret_returns.parquet"
DUR = IMPC / "wrds_data" / "raw" / "bondret_duration.parquet"
RESULTS = HERE / "data" / "results"

N_Q = 5
MIN_FIRMS = 50
NW_LAGS = 6
WINSOR = 0.50
IG_MAX = 10          # rating_num 1-10 = IG, 11-16 = HY
N_PERM = 1000


def nw_mean(series: pd.Series, ann: float = 1200.0) -> dict:
    """Mean (annualized % by default) with Newey-West t."""
    s = series.dropna()
    if len(s) < 24:
        return {"mean": np.nan, "t_nw": np.nan, "n_months": int(len(s))}
    m = sm.OLS(s.values, np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"mean": round(float(s.mean() * ann), 2),
            "t_nw": round(float(m.tvalues[0]), 2), "n_months": int(len(s))}


def build_firm_civ() -> pd.DataFrame:
    """Firm-month CIV (mean across maturities), spread, rf, rating."""
    p = pd.read_parquet(PANEL, columns=["date", "firm_id", "rating",
                                        "spread", "maturity", "rf",
                                        "leverage"])
    p = p[(p["spread"] > 0) & (p["spread"] <= 0.10)].copy()
    p["civ"] = invert_spread_to_civ_vec(
        p["spread"].values, p["leverage"].values, p["maturity"].values)
    p = p[(p["civ"] > 0.01) & (p["civ"] < 2.0)]
    p["month"] = pd.to_datetime(p["date"]).dt.to_period("M")
    agg = (p.groupby(["month", "firm_id"])
           .agg(civ=("civ", "mean"), s=("spread", "mean"),
                rf=("rf", "mean"), rating=("rating", "first"),
                leverage=("leverage", "mean"))
           .reset_index())
    # CIV orthogonalized to leverage: monthly cross-sectional OLS
    # residual (the raw sort inherits a short-leverage tilt from the
    # smirk; the residual is the component leverage does not explain)
    def _orth(g):
        if len(g) < 10:
            return pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(len(g)), g["leverage"].values])
        beta, *_ = np.linalg.lstsq(X, g["civ"].values, rcond=None)
        return pd.Series(g["civ"].values - X @ beta, index=g.index)

    agg["civ_orth"] = (agg.groupby("month", group_keys=False)
                       .apply(_orth, include_groups=False))
    prev = agg[["month", "firm_id", "civ"]].rename(columns={"civ": "civ_prev"})
    prev["month"] = prev["month"] + 1
    agg = agg.merge(prev, on=["month", "firm_id"], how="left")
    agg["dciv"] = agg["civ"] - agg["civ_prev"]
    nxt = agg[["month", "firm_id", "s"]].rename(columns={"s": "s_next"})
    nxt["month"] = nxt["month"] - 1
    agg = agg.merge(nxt, on=["month", "firm_id"], how="left")
    agg["ds"] = agg["s_next"] - agg["s"]
    return agg


def build_firm_returns() -> pd.DataFrame:
    """Amount-weighted firm-month total returns, winsorized +/-50%,
    plus amount-weighted duration and maturity."""
    r = pd.read_parquet(RET, columns=["date", "issuer_cusip", "ret_eom",
                                      "price_eom", "amount_outstanding",
                                      "rating_num", "tmt"])
    r["date"] = pd.to_datetime(r["date"])
    r["month"] = r["date"].dt.to_period("M")
    r = r[(r["price_eom"] >= 5) & (r["price_eom"] <= 1000)].copy()
    r["ret_eom"] = pd.to_numeric(r["ret_eom"], errors="coerce") \
        .clip(-WINSOR, WINSOR)
    r = r.dropna(subset=["ret_eom"])
    r["w"] = pd.to_numeric(r["amount_outstanding"], errors="coerce") \
        .fillna(0.0).clip(lower=0.0)
    d = pd.read_parquet(DUR, columns=["date", "issuer_cusip", "cusip",
                                      "duration"])
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M")

    def wavg(g, col):
        w = g["w"].values
        if w.sum() <= 0:
            w = np.ones(len(g))
        return float(np.average(g[col].values, weights=w))

    out = (r.groupby(["month", "issuer_cusip"])
           .apply(lambda g: pd.Series(
               {"ret": wavg(g, "ret_eom"), "tmt": wavg(g, "tmt"),
                "amt": g["w"].sum(),
                "rating_num": g["rating_num"].median()}),
               include_groups=False)
           .reset_index().rename(columns={"issuer_cusip": "firm_id"}))
    dd = (d.dropna(subset=["duration"])
          .groupby(["month", "issuer_cusip"])["duration"].mean()
          .rename("D").reset_index()
          .rename(columns={"issuer_cusip": "firm_id"}))
    out = out.merge(dd, on=["month", "firm_id"], how="left")
    return out


def quintile_ls(df: pd.DataFrame, sig_col: str, ret_col: str,
                weight: str) -> pd.Series:
    """Monthly Q5-Q1 series; df needs month, firm_id, sig_col, ret_col,
    amt."""
    rows = {}
    d = df.dropna(subset=[sig_col, ret_col])
    for month, g in d.groupby("month"):
        if g["firm_id"].nunique() < MIN_FIRMS:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g[sig_col].rank(method="first"), N_Q, labels=False)
        if weight == "EW":
            m = g.groupby("q")[ret_col].mean()
        else:
            m = g.groupby("q").apply(
                lambda x: np.average(x[ret_col], weights=x["amt"])
                if x["amt"].sum() > 0 else x[ret_col].mean(),
                include_groups=False)
        if 0 in m.index and N_Q - 1 in m.index:
            rows[month] = float(m[N_Q - 1] - m[0])
    return pd.Series(rows).sort_index()


def main() -> None:
    sig = build_firm_civ()
    fr = build_firm_returns()
    print(f"Signal: {len(sig):,} firm-months, "
          f"{sig['firm_id'].nunique():,} firms; "
          f"returns: {len(fr):,} firm-months")

    # Carry-aware returns: conditioning info for return month m dated m-1
    cond = sig[["month", "firm_id", "s", "rf"]].copy()
    rf_next = sig[["month", "firm_id", "rf"]].rename(columns={"rf": "rf_next"})
    rf_next["month"] = rf_next["month"] - 1
    cond = cond.merge(rf_next, on=["month", "firm_id"], how="left")
    cond["drf"] = cond["rf_next"] - cond["rf"]
    cond["ret_month"] = cond["month"] + 1
    fr = fr.rename(columns={"month": "ret_month"}).merge(
        cond[["ret_month", "firm_id", "s", "rf", "drf"]],
        on=["ret_month", "firm_id"], how="left")
    fr["D"] = fr["D"].fillna(fr["tmt"].clip(upper=10) * 0.8)
    fr["ret_ex_xc"] = (fr["ret"] - fr["rf"] / 12.0
                       + fr["D"] * fr["drf"] - fr["s"] / 12.0)
    print(f"Rate-hedged coverage: {fr['ret_ex_xc'].notna().mean():.1%}")

    results = {"conventions": {
        "winsorize_ret": WINSOR, "n_quintiles": N_Q,
        "min_firms": MIN_FIRMS, "nw_lags": NW_LAGS,
        "ret_ex_xc": "ret - rf/12 + D*drf - s/12, conditioning dated "
                     "month before return month; D = bondret modified "
                     "duration (0.8*tmt fallback)",
        "ds_units": "bps/month, target dated t (change t to t+1)"}}

    # ------------------------------------------------------------------
    # Main grid: signal x lag x return x weight
    # ------------------------------------------------------------------
    ls_store = {}
    results["sorts"] = {}
    for sig_col in ["civ", "civ_orth", "dciv"]:
        for lag, lag_name in [(1, "t+1"), (2, "t+2_skip")]:
            s_l = sig.copy()
            s_l["ret_month"] = s_l["month"] + lag
            merged = s_l.merge(
                fr, on=["ret_month", "firm_id"], how="inner",
                suffixes=("_sig", ""))
            for ret_col in ["ret", "ret_ex_xc"]:
                for weight in ["EW", "VW"]:
                    ls = quintile_ls(merged, sig_col, ret_col, weight)
                    key = f"{sig_col}_{lag_name}_{ret_col}_{weight}"
                    results["sorts"][key] = nw_mean(ls)
                    ls_store[key] = ls
                    e = results["sorts"][key]
                    print(f"B2 {key:34s} Q5-Q1 {e['mean']:+6.2f}%/yr "
                          f"(t={e['t_nw']:5.2f}, n={e['n_months']})")

    # ------------------------------------------------------------------
    # Spread-change targets (no return data needed)
    # ------------------------------------------------------------------
    results["spread_change"] = {}
    sk = sig[["month", "firm_id", "ds"]].rename(columns={"ds": "ds_skip"})
    sk["month"] = sk["month"] - 1
    sigd = sig.merge(sk, on=["month", "firm_id"], how="left")
    sigd["amt"] = 1.0
    for sig_col in ["civ", "civ_orth", "dciv"]:
        for target, tname in [("ds", "t+1"), ("ds_skip", "skip")]:
            ls = quintile_ls(sigd, sig_col, target, "EW")
            results["spread_change"][f"{sig_col}_{tname}"] = \
                nw_mean(ls, ann=1e4)
            e = results["spread_change"][f"{sig_col}_{tname}"]
            print(f"B2 ds {sig_col:8s} {tname:5s} Q5-Q1 "
                  f"{e['mean']:+7.2f} bps/mo (t={e['t_nw']:5.2f})")

    # EIV guard: spread measurement noise at t mechanically reverts at
    # t+1 and loads on the spread level, so sort the signal within
    # spread quintiles with ds as the target; the within-quintile cell
    # is clean of the level-driven reversal.
    for sig_col in ["civ", "civ_orth", "dciv"]:
        for target, tname in [("ds", "t+1"), ("ds_skip", "skip")]:
            rows = {}
            d = sigd.dropna(subset=[sig_col, "s", target])
            for month, g in d.groupby("month"):
                if g["firm_id"].nunique() < MIN_FIRMS:
                    continue
                g = g.copy()
                g["sq"] = pd.qcut(g["s"].rank(method="first"), 5,
                                  labels=False)
                effects = []
                for _, gg in g.groupby("sq"):
                    if len(gg) < 10:
                        continue
                    gg = gg.copy()
                    gg["q"] = pd.qcut(gg[sig_col].rank(method="first"),
                                      N_Q, labels=False)
                    m = gg.groupby("q")[target].mean()
                    if 0 in m.index and N_Q - 1 in m.index:
                        effects.append(float(m[N_Q - 1] - m[0]))
                if effects:
                    rows[month] = float(np.mean(effects))
            e = nw_mean(pd.Series(rows).sort_index(), ann=1e4)
            results["spread_change"][
                f"{sig_col}_{tname}_within_spread_quintiles"] = e
            print(f"B2 ds {sig_col:8s} {tname:5s} within-spread Q5-Q1 "
                  f"{e['mean']:+7.2f} bps/mo (t={e['t_nw']:5.2f})")

    # ------------------------------------------------------------------
    # Double sorts: within IG/HY and within maturity terciles (headline:
    # civ, t+1, total EW and rate-hedged EW)
    # ------------------------------------------------------------------
    s_l = sig.copy()
    s_l["ret_month"] = s_l["month"] + 1
    m1 = s_l.merge(fr, on=["ret_month", "firm_id"], how="inner",
                   suffixes=("_sig", ""))
    results["double_sorts"] = {}
    for ret_col in ["ret", "ret_ex_xc"]:
        for bucket, bname in [
                (m1["rating_num"] <= IG_MAX, "IG"),
                (m1["rating_num"] > IG_MAX, "HY")]:
            ls = quintile_ls(m1[bucket], "civ", ret_col, "EW")
            results["double_sorts"][f"civ_t+1_{ret_col}_{bname}"] = \
                nw_mean(ls)
        rows = {}
        d = m1.dropna(subset=["civ", ret_col, "tmt"])
        for month, g in d.groupby("month"):
            if g["firm_id"].nunique() < MIN_FIRMS:
                continue
            g = g.copy()
            g["tb"] = pd.qcut(g["tmt"].rank(method="first"), 3, labels=False)
            effects = []
            for _, gg in g.groupby("tb"):
                if len(gg) < 15:
                    continue
                gg = gg.copy()
                gg["q"] = pd.qcut(gg["civ"].rank(method="first"), N_Q,
                                  labels=False)
                m = gg.groupby("q")[ret_col].mean()
                if 0 in m.index and N_Q - 1 in m.index:
                    effects.append(float(m[N_Q - 1] - m[0]))
            if effects:
                rows[month] = float(np.mean(effects))
        results["double_sorts"][f"civ_t+1_{ret_col}_within_tmt_terciles"] = \
            nw_mean(pd.Series(rows).sort_index())
    for k, e in results["double_sorts"].items():
        print(f"B2 double {k:40s} {e['mean']:+6.2f}%/yr (t={e['t_nw']:5.2f})")

    # ------------------------------------------------------------------
    # Subperiods, permutation, walk-forward on the headline
    # (civ, t+1, rate-hedged ex-carry, EW) and its skip version
    # ------------------------------------------------------------------
    results["robustness"] = {}
    for key in ["civ_t+1_ret_ex_xc_EW", "civ_t+2_skip_ret_ex_xc_EW",
                "civ_t+1_ret_EW", "civ_orth_t+1_ret_ex_xc_EW",
                "civ_orth_t+2_skip_ret_ex_xc_EW"]:
        ls = ls_store[key]
        pre = ls[ls.index < pd.Period("2013-01", "M")]
        post = ls[ls.index >= pd.Period("2013-01", "M")]
        yr = ls.groupby(ls.index.year).mean()
        results["robustness"][key] = {
            "pre2013": nw_mean(pre), "post2013": nw_mean(post),
            "walk_forward_positive_years":
                f"{int((yr > 0).sum())}/{len(yr)}"}

    rng = np.random.default_rng(42)
    s_l = sig.copy()
    s_l["ret_month"] = s_l["month"] + 1
    m1 = s_l.merge(fr, on=["ret_month", "firm_id"], how="inner",
                   suffixes=("_sig", ""))
    d = m1.dropna(subset=["civ", "ret_ex_xc"])
    obs = quintile_ls(d, "civ", "ret_ex_xc", "EW").mean()
    perm_means = []
    for _ in range(N_PERM):
        dp = d.copy()
        dp["civ"] = dp.groupby("month")["civ"].transform(
            lambda x: rng.permutation(x.values))
        perm_means.append(quintile_ls(dp, "civ", "ret_ex_xc", "EW").mean())
    p = float(np.mean(np.abs(perm_means) >= abs(obs)))
    results["robustness"]["permutation_civ_t+1_ret_ex_xc_EW"] = {
        "observed_ann_pct": round(float(obs * 1200), 2),
        "p_value": round(p, 4), "n_perm": N_PERM}
    print(f"Permutation p = {p:.4f} (obs {obs * 1200:+.2f}%/yr)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "cross_section_monthly.json", "w") as f:
        json.dump(results, f, indent=2)
    ls_df = pd.DataFrame({k: v for k, v in ls_store.items()})
    ls_df.index = ls_df.index.astype(str)
    ls_df.to_csv(RESULTS / "cross_section_monthly_ls.csv")
    print(f"Written: {RESULTS / 'cross_section_monthly.json'}")
    print(f"Written: {RESULTS / 'cross_section_monthly_ls.csv'} (for B3)")


if __name__ == "__main__":
    main()
