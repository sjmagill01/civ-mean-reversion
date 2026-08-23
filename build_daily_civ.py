"""
build_daily_civ.py -- Construct daily CIV portfolios.

Monthly yield spreads + daily CRSP leverage.

CRITICAL DATA DECISIONS (see PITFALLS.md):
- Spread: monthly credit spread = EOM yield minus maturity-matched treasury
  (wrds_panel_yieldspread.parquet). NOT bondret t_spread (that is a bid/ask
  transaction-cost measure, not a credit spread) and NOT TRACE avg_yield.
- Leverage: daily via permno bridge (bondcrsp_link -> ccm_link -> compustat debt + crsp daily mcap).
- CIV: Merton inversion at (monthly spread forward-filled to daily, daily leverage).

Output:
    markov_vol/data/daily_portfolios.parquet
    markov_vol/data/daily_civ_panel.parquet
"""
import sys
from pathlib import Path

# Add parent to path so markov_vol is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from markov_vol.src.merton import invert_spread_to_civ_vec

import os

# Raw WRDS pulls (bondcrsp_link, ccm_link, fundq, crsp daily, bondret) from
# the companion CIV pipeline. Not included in this repo (WRDS license).
DATA_RAW = Path(os.environ.get("CIV_RAW_DIR", Path(__file__).parent / "data" / "raw"))
# Corrected monthly spread panel (yield minus maturity-matched treasury).
PANEL_PATH = Path(os.environ.get(
    "CIV_PANEL_PATH", DATA_RAW / "wrds_panel_yieldspread.parquet"))
DATA_INT = Path(os.environ.get("CIV_INT_DIR", Path(__file__).parent / "data" / "intermediate"))
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_PORTFOLIOS = OUTPUT_DIR / "daily_portfolios.parquet"
OUTPUT_PANEL = OUTPUT_DIR / "daily_civ_panel.parquet"


def build_identifier_map():
    """
    Build issuer_cusip -> permno -> gvkey mapping.

    Uses bondcrsp_link (issuer_cusip -> permno, most common)
    and ccm_link (permno -> gvkey, latest linkdt).

    Returns
    -------
    id_map : pd.DataFrame
        Columns: issuer_cusip, permno (int), gvkey (str).
    """
    # Bond-equity link: issuer_cusip -> permno (most common mapping)
    bond_link = pd.read_parquet(DATA_RAW / "bondcrsp_link.parquet")
    cusip_permno = (
        bond_link.groupby("issuer_cusip")["permno"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    cusip_permno["permno"] = cusip_permno["permno"].astype(int)

    # CCM link: permno -> gvkey (latest linkdt wins)
    ccm_link = pd.read_parquet(DATA_RAW / "ccm_link.parquet")
    ccm = ccm_link.sort_values("linkdt").drop_duplicates("permno", keep="last")
    ccm = ccm[["permno", "gvkey"]].copy()
    ccm["permno"] = ccm["permno"].astype(int)
    ccm["gvkey"] = ccm["gvkey"].astype(str)

    # Merge
    id_map = cusip_permno.merge(ccm, on="permno", how="inner")
    print(f"Identifier map: {len(id_map)} issuer_cusips with permno+gvkey")
    return id_map


def build_daily_leverage(id_map):
    """
    Construct daily leverage = debt / (debt + market_cap).

    Debt: quarterly Compustat, forward-filled.
    Market cap: daily CRSP (abs(price) * shrout / 1000, in millions).
    Merged via merge_asof with 270-day backward tolerance.

    Parameters
    ----------
    id_map : pd.DataFrame
        From build_identifier_map().

    Returns
    -------
    daily_lev : pd.DataFrame
        Columns: permno, date, leverage. Keyed by permno (equity), not
        issuer_cusip: several issuer cusips can share one permno, and
        mapping permno back to a single cusip here would silently drop
        the rest (the pre-correction build lost ~2/3 of issuers this way).
    """
    # --- Compustat debt ---
    debt = pd.read_parquet(DATA_RAW / "compustat_debt.parquet")
    debt = debt[debt["total_debt"] > 0].copy()
    debt["gvkey"] = debt["gvkey"].astype(str)
    debt["date"] = pd.to_datetime(debt["datadate"])
    debt = debt.sort_values(["gvkey", "date"])

    # --- CRSP daily prices ---
    crsp = pd.read_parquet(DATA_RAW / "crsp_daily.parquet")
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["market_cap"] = abs(crsp["price"]) * crsp["shrout"] / 1000.0  # millions
    crsp = crsp[crsp["market_cap"] > 0].copy()

    # Map permno -> gvkey (permno -> gvkey is one-to-one after
    # drop_duplicates in build_identifier_map)
    permno_to_gvkey = (
        id_map.drop_duplicates("permno").set_index("permno")["gvkey"].to_dict()
    )
    crsp["permno"] = crsp["permno"].astype(int)
    crsp["gvkey"] = crsp["permno"].map(permno_to_gvkey)
    crsp = crsp.dropna(subset=["gvkey"])
    crsp = crsp.sort_values(["gvkey", "date"])

    # --- Merge: for each gvkey, merge_asof crsp onto debt ---
    all_lev = []
    for gvkey, crsp_g in crsp.groupby("gvkey"):
        debt_g = debt[debt["gvkey"] == gvkey].copy()
        if len(debt_g) == 0:
            continue

        merged = pd.merge_asof(
            crsp_g[["date", "permno", "market_cap"]].sort_values("date"),
            debt_g[["date", "total_debt"]].sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=270),
        )
        merged = merged.dropna(subset=["total_debt"])
        merged["leverage"] = merged["total_debt"] / (
            merged["total_debt"] + merged["market_cap"]
        )
        all_lev.append(merged[["permno", "date", "leverage"]])

    daily_lev = pd.concat(all_lev, ignore_index=True)

    # Filter reasonable leverage
    daily_lev = daily_lev[
        (daily_lev["leverage"] > 0.01) & (daily_lev["leverage"] < 0.99)
    ].copy()

    print(f"Daily leverage: {len(daily_lev):,} rows, "
          f"{daily_lev['permno'].nunique()} permnos, "
          f"{daily_lev['date'].nunique()} days")
    return daily_lev


def build_daily_spreads():
    """
    Load monthly credit spreads from wrds_panel_yieldspread.parquet.

    Spread = EOM bond yield minus maturity-matched treasury yield. This
    replaces the original t_spread source: bondret t_spread is a bid/ask
    transaction-cost measure, not a credit spread (2026-08 correction).
    Also NOT TRACE avg_yield (that is a YTM; see PITFALLS.md #1).

    Returns
    -------
    spreads : pd.DataFrame
        Columns: issuer_cusip, date, spread, maturity.
        Monthly frequency, firm-month median.
    """
    cols = ["date", "firm_id", "spread", "maturity"]
    panel = pd.read_parquet(PANEL_PATH, columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])

    # Filter valid spreads and maturities
    panel = panel[
        (panel["spread"] > 0) & (panel["spread"] < 0.10)
        & (panel["maturity"] >= 0.25) & (panel["maturity"] <= 30)
    ].copy()

    # Aggregate to firm-month median
    panel["month"] = panel["date"].dt.to_period("M")
    monthly = (
        panel.groupby(["firm_id", "month"])
        .agg(spread=("spread", "median"), maturity=("maturity", "median"))
        .reset_index()
    )
    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.drop(columns=["month"])
    monthly = monthly.rename(columns={"firm_id": "issuer_cusip"})

    print(f"Monthly spreads: {len(monthly):,} rows, "
          f"{monthly['issuer_cusip'].nunique()} issuers, "
          f"{monthly['date'].nunique()} months")
    return monthly


def compute_daily_civ(monthly_spreads, daily_leverage, id_map):
    """
    Compute daily CIV via Merton inversion.

    Forward-fills monthly spreads to daily using merge_asof (45-day tolerance).

    Parameters
    ----------
    monthly_spreads : pd.DataFrame
        From build_daily_spreads().
    daily_leverage : pd.DataFrame
        From build_daily_leverage().

    Returns
    -------
    civ_panel : pd.DataFrame
        Columns: issuer_cusip, date, spread, leverage, maturity, civ.
    """
    # Ensure consistent types
    monthly_spreads["issuer_cusip"] = monthly_spreads["issuer_cusip"].astype(str)

    # Map each spread cusip to its permno (unique per cusip), then expand
    # permno-level daily leverage to the cusips actually present in the
    # spread panel. Several cusips can share one permno; each inherits
    # the same equity leverage.
    cusip_to_permno = id_map.set_index("issuer_cusip")["permno"].to_dict()
    monthly_spreads["permno"] = monthly_spreads["issuer_cusip"].map(cusip_to_permno)
    monthly_spreads = monthly_spreads.dropna(subset=["permno"])
    monthly_spreads["permno"] = monthly_spreads["permno"].astype(int)

    pairs = monthly_spreads[["issuer_cusip", "permno"]].drop_duplicates()
    daily_leverage = daily_leverage.merge(pairs, on="permno", how="inner")
    print(f"Daily leverage expanded to cusips: {len(daily_leverage):,} rows, "
          f"{daily_leverage['issuer_cusip'].nunique()} issuers")

    # Sort by date for merge_asof (must be globally sorted on the 'on' key)
    monthly_spreads = monthly_spreads.sort_values("date")
    daily_leverage = daily_leverage.sort_values("date")

    # Merge: forward-fill monthly spreads to daily leverage dates
    merged = pd.merge_asof(
        daily_leverage,
        monthly_spreads[["issuer_cusip", "date", "spread", "maturity"]],
        on="date",
        by="issuer_cusip",
        direction="backward",
        tolerance=pd.Timedelta(days=45),
    )
    merged = merged.dropna(subset=["spread", "maturity"])

    print(f"Merged for inversion: {len(merged):,} rows")

    # Merton inversion (vectorized Newton-Raphson over the whole array)
    merged = merged.reset_index(drop=True)
    merged["civ"] = invert_spread_to_civ_vec(
        merged["spread"].to_numpy(),
        merged["leverage"].to_numpy(),
        merged["maturity"].to_numpy(),
    )

    # Filter reasonable CIV
    merged = merged[(merged["civ"] > 0.01) & (merged["civ"] < 2.0)].copy()

    print(f"Daily CIV panel: {len(merged):,} rows after filtering")
    return merged


def aggregate_to_portfolios(df):
    """
    Bin CIV into leverage x maturity portfolios.

    Leverage bins: [0, 0.30, 0.50, 0.70, 1.0] -> labels [0.20, 0.40, 0.60, 0.80]
    Maturity bins: [0.25, 2, 4, 6, 8.5, 30] -> labels [1, 3, 5, 7, 10]

    Parameters
    ----------
    df : pd.DataFrame
        Daily CIV panel with columns: date, leverage, maturity, civ.

    Returns
    -------
    wide : pd.DataFrame
        Pivoted: index=date, columns='L{lev}_T{mat}'.
        Only days with >= 10 portfolios populated.
    """
    df = df.copy()

    # Leverage bins
    df["lev_bin"] = pd.cut(
        df["leverage"],
        bins=[0, 0.30, 0.50, 0.70, 1.0],
        labels=[0.20, 0.40, 0.60, 0.80],
    )

    # Maturity bins
    df["mat_bin"] = pd.cut(
        df["maturity"],
        bins=[0.25, 2, 4, 6, 8.5, 30],
        labels=[1, 3, 5, 7, 10],
    )

    df = df.dropna(subset=["lev_bin", "mat_bin"])

    # Groupby median CIV
    grouped = (
        df.groupby(["date", "lev_bin", "mat_bin"])["civ"]
        .median()
        .reset_index()
    )

    # Create portfolio label
    grouped["portfolio"] = (
        "L" + grouped["lev_bin"].astype(str) + "_T" + grouped["mat_bin"].astype(str)
    )

    # Pivot to wide
    wide = grouped.pivot(index="date", columns="portfolio", values="civ")

    # Require >= 10 portfolios per day
    valid_days = wide.notna().sum(axis=1) >= 10
    wide = wide[valid_days]

    print(f"Daily portfolios: {len(wide)} days, {wide.shape[1]} portfolios")
    print(f"  Date range: {wide.index.min().date()} to {wide.index.max().date()}")
    print(f"  Portfolios: {list(wide.columns)}")
    return wide


def main():
    """
    Run full daily CIV construction pipeline.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: Build identifier map")
    print("=" * 60)
    id_map = build_identifier_map()

    print("\n" + "=" * 60)
    print("Step 2: Build daily leverage")
    print("=" * 60)
    daily_lev = build_daily_leverage(id_map)

    print("\n" + "=" * 60)
    print("Step 3: Build monthly yield spreads (corrected panel)")
    print("=" * 60)
    monthly_spreads = build_daily_spreads()

    print("\n" + "=" * 60)
    print("Step 4: Compute daily CIV via Merton inversion")
    print("=" * 60)
    civ_panel = compute_daily_civ(monthly_spreads, daily_lev, id_map)

    # Save full panel
    civ_panel.to_parquet(OUTPUT_PANEL, index=False)
    print(f"\nSaved CIV panel: {OUTPUT_PANEL}")

    print("\n" + "=" * 60)
    print("Step 5: Aggregate to leverage x maturity portfolios")
    print("=" * 60)
    wide = aggregate_to_portfolios(civ_panel)

    # Save with date as column (not index)
    wide_out = wide.reset_index()
    wide_out.to_parquet(OUTPUT_PORTFOLIOS, index=False)
    print(f"Saved portfolios: {OUTPUT_PORTFOLIOS}")

    # --- Sanity check: compare daily median to monthly median ---
    print("\n" + "=" * 60)
    print("Sanity check: daily vs monthly CIV median")
    print("=" * 60)
    try:
        monthly_path = OUTPUT_DIR / "portfolios_wide.parquet"
        monthly_port = pd.read_parquet(monthly_path)
        if "date" in monthly_port.columns:
            monthly_port = monthly_port.drop(columns=["date"])
        monthly_median = np.nanmedian(monthly_port.values)
        # Convert from pct if needed
        if monthly_median > 1.0:
            monthly_median /= 100.0
        daily_median = np.nanmedian(wide.values)
        ratio = daily_median / monthly_median
        print(f"  Monthly CIV median: {monthly_median:.4f}")
        print(f"  Daily CIV median:   {daily_median:.4f}")
        print(f"  Ratio (daily/monthly): {ratio:.3f}")
        if 0.7 < ratio < 1.5:
            print("  PASS: ratio is close to 1.0")
        else:
            print("  WARNING: ratio is far from 1.0 -- investigate data pipeline")
    except Exception as e:
        print(f"  Could not run sanity check: {e}")


if __name__ == "__main__":
    main()
