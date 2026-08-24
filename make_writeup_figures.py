"""Figures for the regime-conditioning and cross-section README sections.

All inputs are the results JSONs/CSV in data/results/ (no re-estimation):
  fig5_regime_conditioning.png  Track A: AR(1) by filtered-vol tercile +
                                OOS Sharpe by regime-conditioned strategy
  fig6_cross_section.png        B2: quintile long-short premia (total vs
                                rate-hedged ex-carry) + cumulative lines
  fig7_regime_bridge.png        B3: cross-sectional premium by filtered
                                regime tercile, with ex-crisis check

Run from rsumplay root: python markov_vol/make_writeup_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data" / "results"
FIGDIR = HERE / "figures"

NAVY = "#1a3a5c"
RED = "#c0392b"
GREEN = "#3a7d44"
GRAY = "#8a8a8a"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def fig5_regime_conditioning():
    d = load("regime_conditioning.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: AR(1) of weekly median-CIV changes by filtered-vol tercile
    buckets = ["low", "mid", "high"]
    betas = [d["ar1_by_bucket"][b]["ar1_beta"] for b in buckets]
    ns = [d["ar1_by_bucket"][b]["n_weeks"] for b in buckets]
    bars = ax1.bar(buckets, betas, color=[NAVY, GRAY, RED], width=0.55)
    for bar, beta, n in zip(bars, betas, ns):
        ax1.text(bar.get_x() + bar.get_width() / 2, beta - 0.012,
                 f"{beta:.3f}\n(n={n} wks)", ha="center", va="top",
                 fontsize=9)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylim(min(betas) * 1.45, 0.02)
    ax1.set_ylabel("AR(1) coefficient of weekly median-CIV changes")
    ax1.set_xlabel("Filtered expected-vol tercile (calibration edges)")
    ax1.set_title("Mean reversion is present in every regime")

    # right: OOS Sharpe of regime-conditioned strategy variants, lag 0
    strats = ["unconditional", "only_low", "only_mid", "only_high",
              "inv_vol_scaled", "vol_scaled", "pstay_scaled"]
    lab = ["uncond.", "only\nlow", "only\nmid", "only\nhigh",
           "inv-vol\nscaled", "vol\nscaled", "p-stay\nscaled"]
    sr = [d["lags"]["0"]["strategies"][s]["sharpe_oos"] for s in strats]
    colors = [NAVY] + [GRAY] * (len(strats) - 1)
    ax2.bar(lab, sr, color=colors, width=0.6)
    ax2.axhline(sr[0], color=NAVY, lw=1, ls="--",
                label=f"unconditional OOS Sharpe {sr[0]:.2f}")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("OOS Sharpe (2013-2024, lag 0)")
    ax2.set_title("No regime-conditioned variant significantly beats\n"
                  "unconditional (rotation test p = 0.39-0.89)")
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    out = FIGDIR / "fig5_regime_conditioning.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fig6_cross_section():
    d = load("cross_section_monthly.json")
    s = d["sorts"]
    ls = pd.read_csv(RESULTS / "cross_section_monthly_ls.csv", index_col=0)
    ls.index = pd.PeriodIndex(ls.index, freq="M").to_timestamp("M")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: long-short premium bars, civ vs civ_orth, EW
    specs = [("t+1 total", "{}_t+1_ret_EW"),
             ("t+1 rate-hedged\nex-carry", "{}_t+1_ret_ex_xc_EW"),
             ("skip-a-month\nrate-hedged", "{}_t+2_skip_ret_ex_xc_EW")]
    sigs = [("civ", "CIV", NAVY)]
    if "civ_orth_t+1_ret_ex_xc_EW" in s:
        sigs.append(("civ_orth", "CIV orth. to leverage", GREEN))
    x = np.arange(len(specs))
    w = 0.8 / len(sigs)
    for i, (sig, label, color) in enumerate(sigs):
        means = [s[k.format(sig)]["mean"] for _, k in specs]
        ts = [s[k.format(sig)]["t_nw"] for _, k in specs]
        bars = ax1.bar(x + (i - (len(sigs) - 1) / 2) * w, means, w,
                       color=color, label=label)
        for bar, m, t in zip(bars, means, ts):
            ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.15,
                     f"{m:+.1f}\n(t={t:.1f})", ha="center", fontsize=8)
    ax1.set_xticks(x, [n for n, _ in specs])
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylabel("Q5-Q1 long-short (%/yr, EW)")
    ax1.set_title("Monthly CIV quintile premium, 2002-2024")
    ax1.legend(frameon=False)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.18)

    # right: cumulative long-short P&L, rate-hedged ex-carry
    for key, label, color, style in [
            ("civ_t+1_ret_EW", "t+1, total return", GRAY, "-"),
            ("civ_t+1_ret_ex_xc_EW", "t+1, rate-hedged ex-carry",
             NAVY, "-"),
            ("civ_t+2_skip_ret_ex_xc_EW", "skip-a-month, hedged",
             RED, "--")]:
        v = ls[key].dropna()
        ax2.plot(v.index, v.cumsum() * 100, color=color, ls=style,
                 lw=1.4, label=label)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("Cumulative long-short return (%)")
    ax2.set_title("Ex-carry line keeps the premium and gains texture")
    ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out = FIGDIR / "fig6_cross_section.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fig7_regime_bridge():
    d = load("regime_bridge.json")
    keys = [("civ_t+1_ret_ex_xc_EW", "t+1, rate-hedged ex-carry"),
            ("civ_t+2_skip_ret_ex_xc_EW", "skip-a-month, rate-hedged")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (key, title) in zip(axes, keys):
        e = d["by_key"][key]
        buckets = ["low", "mid", "high"]
        means = [e[b]["mean_ann_pct"] for b in buckets]
        ts = [e[b]["t_nw"] for b in buckets]
        ns = [e[b]["n_months"] for b in buckets]
        bars = ax.bar(buckets, means, color=[NAVY, GRAY, RED], width=0.55)
        for bar, m, t, n in zip(bars, means, ts, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.25,
                    f"{m:+.2f}\n(t={t:.1f}, n={n})", ha="center",
                    fontsize=9)
        # ex-crisis check on the high bucket
        xc = e["high_ex_crisis"]
        ax.hlines(xc["mean_ann_pct"], 1.72, 2.28, color="black", lw=1.6,
                  ls=":",
                  label=f"high ex-GFC/COVID: {xc['mean_ann_pct']:+.2f} "
                        f"(t={xc['t_nw']:.1f})")
        hml = e["high_minus_low"]
        p, nr = hml["rotation_p_two_sided"], hml["n_rotations"]
        ptxt = f"p < {1 / nr:.3f}" if p == 0 else f"p = {p:.3f}"
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{title}\nhigh-low {hml['mean_ann_pct']:+.2f}%/yr, "
                     f"rotation {ptxt}")
        ax.set_xlabel("Filtered expected-vol tercile at formation")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("CIV quintile long-short (%/yr, EW)")
    top = axes[0].get_ylim()[1]
    axes[0].set_ylim(min(0, axes[0].get_ylim()[0]), top * 1.18)

    fig.suptitle("The regime model concentrates the cross-sectional "
                 "premium", y=1.0)
    fig.tight_layout()
    out = FIGDIR / "fig7_regime_bridge.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    fig5_regime_conditioning()
    fig6_cross_section()
    fig7_regime_bridge()
