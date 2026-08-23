"""
unit_analysis.py -- Careful unit verification for dollar economics.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markov_vol.src.merton import merton_spread, merton_vega
from markov_vol.src.data_loader import load_daily_portfolios


def main():
    y, dates, L, tau, labels = load_daily_portfolios()
    dates = pd.to_datetime(dates)
    oos = dates >= pd.Timestamp('2024-01-01')
    y_oos = y[oos]

    print("=" * 70)
    print("UNIT ANALYSIS")
    print("=" * 70)

    # Weekly CIV P&L
    weekly = pd.DataFrame(y_oos, index=dates[oos]).resample('W-FRI').last().dropna(how='all')
    wm = weekly.median(axis=1).values
    wm_changes = np.diff(wm)
    sig = -wm_changes[:-1]
    pos = np.sign(sig)
    pnl_civ_decimal = pos * wm_changes[1:]  # in CIV decimal
    pnl_civ_bps = pnl_civ_decimal * 10000

    mean_weekly_civ_bps = np.mean(pnl_civ_bps)
    std_weekly_civ_bps = np.std(pnl_civ_bps)
    sharpe = mean_weekly_civ_bps / std_weekly_civ_bps * np.sqrt(52)

    print(f"\nWeekly CIV P&L: mean = {mean_weekly_civ_bps:.2f} CIV bps/week")
    print(f"Weekly CIV P&L: std  = {std_weekly_civ_bps:.1f} CIV bps/week")
    print(f"Annualized Sharpe: {sharpe:.2f}")

    # Step 1: Compute Merton vega for each portfolio
    print("\n" + "=" * 70)
    print("MERTON VEGA BY PORTFOLIO")
    print("=" * 70)

    sigma_mid = np.nanmedian(y_oos)
    print(f"\nMedian CIV: {sigma_mid:.4f}")

    vegas = []
    durations = []
    for i, (l, t) in enumerate(zip(L, tau)):
        v = merton_vega(sigma_mid, l, t)
        vegas.append(v)
        durations.append(t)  # approximate duration ~ maturity for IG
        s = merton_spread(sigma_mid, l, t)
        if i < 5 or i >= 15:  # show first and last 5
            print(f"  {labels[i]}: spread={s*10000:.0f}bps, vega={v:.4f}")

    vegas = np.array(vegas)
    print(f"\n  Vega range: [{vegas.min():.4f}, {np.median(vegas):.4f}, {vegas.max():.4f}]")
    eff_vega = np.median(vegas)
    print(f"  Effective (median) vega: {eff_vega:.4f}")

    # Step 2: CIV bps -> spread bps
    print("\n" + "=" * 70)
    print("CIV TO SPREAD CONVERSION")
    print("=" * 70)

    weekly_spread_bps = mean_weekly_civ_bps * eff_vega
    print(f"\n  Weekly CIV P&L:    {mean_weekly_civ_bps:.2f} CIV bps")
    print(f"  x effective vega:  {eff_vega:.4f}")
    print(f"  = Weekly spread P&L: {weekly_spread_bps:.4f} spread bps")
    print(f"  Annual spread P&L: {weekly_spread_bps * 52:.2f} spread bps")

    # Step 3: Spread bps -> dollars
    print("\n" + "=" * 70)
    print("DOLLAR ECONOMICS")
    print("=" * 70)

    # For CDX IG 5yr: DV01 ~ $4,700 per $10M per 1bp
    # = $470 per $1M per 1bp = $47,000 per $100M per 1bp
    notional = 100  # $100M
    dv01_per_100M_per_bp = 47000  # dollars per 1bp spread change on $100M

    weekly_gross = dv01_per_100M_per_bp * weekly_spread_bps
    annual_gross = weekly_gross * 52

    print(f"\n  At $100M notional (CDX IG 5yr, DV01 = $47K/bp):")
    print(f"  Weekly gross: ${weekly_gross:,.0f}")
    print(f"  Annual gross: ${annual_gross:,.0f}")

    # Costs in spread bps
    for cost_name, cost_spread_bps in [("CDX IG (2bp)", 2), ("Credit ETF (5bp)", 5)]:
        weekly_cost = dv01_per_100M_per_bp * cost_spread_bps
        annual_cost = weekly_cost * 52
        annual_net = annual_gross - annual_cost
        print(f"\n  {cost_name}:")
        print(f"    Annual cost: ${annual_cost:,.0f}")
        print(f"    Annual net:  ${annual_net:,.0f}")

    # Step 4: Alternative -- Sharpe-based sizing
    print("\n" + "=" * 70)
    print("SHARPE-BASED SIZING (model-independent)")
    print("=" * 70)
    print(f"\n  Sharpe = {sharpe:.2f}")
    print(f"  Expected return = Sharpe x target vol")
    print(f"  (This is unit-free and doesn't depend on vega)")

    for target_vol_pct in [2, 5, 10]:
        ret_pct = sharpe * target_vol_pct
        dollar = 100e6 * ret_pct / 100
        print(f"\n  Target {target_vol_pct}% annual vol on $100M:")
        print(f"    Expected return: {ret_pct:.1f}% = ${dollar/1e6:.1f}M")

    # Step 5: Compare wrong vs correct
    print("\n" + "=" * 70)
    print("COMPARISON: WRONG vs CORRECT")
    print("=" * 70)

    wrong_annual = notional * 1e6 * mean_weekly_civ_bps * 52 / 10000 * 5
    print(f"\n  WRONG (CIV bps x duration x notional):")
    print(f"    ${wrong_annual/1e6:.1f}M/yr  <-- what the paper said")
    print(f"\n  CORRECT (CIV bps x vega x DV01 x 52):")
    print(f"    ${annual_gross/1e6:.1f}M/yr gross")
    print(f"\n  Ratio: {annual_gross / wrong_annual:.2f}x")

    # Breakeven in CIV bps
    # Cost per trade in spread bps (e.g., 2bp for CDX)
    # Need: weekly CIV P&L * vega > cost
    # Breakeven CIV bps = cost_spread_bps / vega
    breakeven_civ_bps = 2.0 / eff_vega
    print(f"\n  Breakeven CIV bps per trade (at 2bp CDX cost): {breakeven_civ_bps:.1f} CIV bps")
    print(f"  Weekly gross CIV P&L: {mean_weekly_civ_bps:.1f} CIV bps")
    print(f"  Ratio (gross/breakeven): {mean_weekly_civ_bps / breakeven_civ_bps:.2f}x")


if __name__ == "__main__":
    main()
