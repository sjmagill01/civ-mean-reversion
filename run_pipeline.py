"""
run_pipeline.py -- Complete Markov-switching CIV estimation + backtest pipeline.

Usage:
    python run_pipeline.py                   # full (monthly + daily + OOS)
    python run_pipeline.py --monthly-only    # skip daily
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path so markov_vol is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from markov_vol.config import (
    PORTFOLIOS_WIDE, DAILY_PORTFOLIOS, RESULTS_DIR, DATA_DIR,
    N_VOL_DEFAULT,
)
from markov_vol.src.data_loader import load_monthly_portfolios, load_daily_portfolios
from markov_vol.src.estimation import estimate_markov_civ, compute_bic
from markov_vol.src.hamilton_filter import hamilton_filter, estimate_intercepts
from markov_vol.src.markov_transition import build_vol_grid, build_transition_matrix
from markov_vol.src.diagnostics import r_squared
from markov_vol.src.strategies import run_oos_analysis


def step1_monthly():
    """
    Load monthly portfolios. Estimate N_vol in {3, 5, 7}. BIC select.
    Save JSONs + NPZs to data/results/.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Monthly estimation with BIC selection")
    print("=" * 70)

    y_all, dates, L_flat, tau_flat, labels = load_monthly_portfolios()
    T, n_port = y_all.shape
    print(f"Loaded monthly data: T={T}, n_port={n_port}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Grand mean CIV: {np.nanmean(y_all):.4f}")

    best_bic = np.inf
    best_nvol = None
    best_result = None
    all_results = {}

    for n_vol in [3, 5, 7]:
        print(f"\n--- Estimating n_vol = {n_vol} ---")
        t0 = time.time()
        result = estimate_markov_civ(y_all, L_flat, tau_flat, n_vol=n_vol, verbose=True)
        elapsed = time.time() - t0

        # BIC: 5 continuous params + n_vol*(n_vol-1) transition params + n_port intercepts
        n_params = 5 + n_vol * (n_vol - 1) + n_port
        n_obs = int(np.sum(~np.isnan(y_all)))
        bic = compute_bic(result["loglik"], n_params, n_obs)

        # R-squared
        r2 = r_squared(y_all, result["filter"]["fitted_civ"])

        print(f"  LogLik: {result['loglik']:.2f}")
        print(f"  BIC:    {bic:.2f}")
        print(f"  R2:     {r2:.4f}")
        print(f"  Time:   {elapsed:.1f}s")

        all_results[n_vol] = {
            "params": result["params"],
            "loglik": float(result["loglik"]),
            "bic": float(bic),
            "r2": float(r2),
            "n_vol": n_vol,
            "n_params": n_params,
            "elapsed": elapsed,
        }

        # Save NPZ for this n_vol
        npz_path = RESULTS_DIR / f"monthly_nvol{n_vol}.npz"
        np.savez(
            npz_path,
            sigma_grid=result["sigma_grid"],
            P=result["P"],
            port_intercepts=result["port_intercepts"],
            filtered_probs=result["filter"]["filtered_probs"],
            smoothed_probs=result["filter"]["smoothed_probs"],
            fitted_civ=result["filter"]["fitted_civ"],
        )
        print(f"  Saved: {npz_path.name}")

        if bic < best_bic:
            best_bic = bic
            best_nvol = n_vol
            best_result = result

    print(f"\n=== BIC selects n_vol = {best_nvol} (BIC = {best_bic:.2f}) ===")

    # Save summary JSON
    summary = {
        "best_nvol": best_nvol,
        "best_bic": float(best_bic),
        "results": {},
    }
    for nv, res in all_results.items():
        summary["results"][str(nv)] = {
            k: (v if not isinstance(v, dict) else
                {kk: float(vv) for kk, vv in v.items()})
            for k, v in res.items()
        }

    json_path = RESULTS_DIR / "monthly_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path.name}")

    return best_result, y_all, dates, L_flat, tau_flat, labels


def step2_daily_oos(monthly_result=None):
    """
    Load daily portfolios. Split cal/oos. Estimate on cal. OOS filter.
    Compare CIV mean-reversion Sharpe to equity benchmark.

    Parameters
    ----------
    monthly_result : dict, optional
        If provided, used for warm-starting. Otherwise estimate from scratch.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Daily out-of-sample evaluation")
    print("=" * 70)

    y_all, dates, L_flat, tau_flat, labels = load_daily_portfolios()
    T, n_port = y_all.shape
    dates_ts = pd.to_datetime(dates)

    print(f"Loaded daily data: T={T}, n_port={n_port}")
    print(f"  Date range: {dates_ts[0].date()} to {dates_ts[-1].date()}")

    # Split: calibration = 2018-2023, OOS = 2024
    cal_mask = (dates_ts >= "2018-01-01") & (dates_ts <= "2023-12-31")
    oos_mask = dates_ts >= "2024-01-01"

    y_cal = y_all[cal_mask]
    y_oos = y_all[oos_mask]
    dates_cal = dates_ts[cal_mask]
    dates_oos = dates_ts[oos_mask]

    print(f"  Calibration: {len(y_cal)} days ({dates_cal[0].date()} to {dates_cal[-1].date()})")
    print(f"  OOS:         {len(y_oos)} days ({dates_oos[0].date()} to {dates_oos[-1].date()})")

    # Estimate on calibration data (n_vol=7)
    n_vol = 7
    print(f"\n--- Estimating on calibration data (n_vol={n_vol}) ---")
    t0 = time.time()
    cal_result = estimate_markov_civ(y_cal, L_flat, tau_flat, n_vol=n_vol, verbose=True)
    cal_elapsed = time.time() - t0
    print(f"  Calibration time: {cal_elapsed:.1f}s")

    # Calibration R2
    r2_cal = r_squared(y_cal, cal_result["filter"]["fitted_civ"])
    print(f"  Calibration R2: {r2_cal:.4f}")

    # OOS filter with frozen parameters
    print("\n--- OOS filtering with frozen parameters ---")
    sigma_grid = cal_result["sigma_grid"]
    P = cal_result["P"]
    port_intercepts = cal_result["port_intercepts"]

    # Use last calibration filtered state as OOS initial distribution
    pi_init = cal_result["filter"]["filtered_probs"][-1]

    oos_filter = hamilton_filter(
        y_oos, sigma_grid, P, L_flat, tau_flat,
        cal_result["params"]["sigma_eta"],
        pi_0=pi_init,
        port_intercepts=port_intercepts,
    )

    r2_oos = r_squared(y_oos, oos_filter["fitted_civ"])
    print(f"  OOS R2: {r2_oos:.4f}")

    # Load CRSP daily for equity benchmark
    equity_oos = None
    try:
        import os
        crsp_path = Path(os.environ.get(
            "CRSP_DAILY_PATH",
            Path(__file__).parent / "data" / "crsp_daily.parquet"))
        if crsp_path.exists():
            crsp = pd.read_parquet(crsp_path)
            crsp["date"] = pd.to_datetime(crsp["date"])
            crsp_oos = crsp[crsp["date"] >= "2024-01-01"]
            if len(crsp_oos) > 0 and "ret" in crsp_oos.columns:
                # Average daily return across all stocks
                equity_daily = crsp_oos.groupby("date")["ret"].mean().sort_index()
                equity_oos = equity_daily.values[:len(y_oos)]
                if len(equity_oos) < len(y_oos):
                    pad = np.full(len(y_oos) - len(equity_oos), np.nan)
                    equity_oos = np.concatenate([equity_oos, pad])
                print(f"  Loaded equity benchmark: {len(equity_oos)} days")
    except Exception as e:
        print(f"  Could not load equity benchmark: {e}")

    # Run OOS strategy analysis
    strat_results = run_oos_analysis(
        y_cal, y_oos, labels,
        oos_filter["smoothed_probs"],
        sigma_grid, P,
        equity_oos=equity_oos if equity_oos is not None else None,
        verbose=True,
    )

    # Save results
    oos_summary = {
        "n_vol": n_vol,
        "cal_dates": [str(dates_cal[0].date()), str(dates_cal[-1].date())],
        "oos_dates": [str(dates_oos[0].date()), str(dates_oos[-1].date())],
        "cal_days": int(len(y_cal)),
        "oos_days": int(len(y_oos)),
        "r2_cal": float(r2_cal),
        "r2_oos": float(r2_oos),
        "params": {k: float(v) for k, v in cal_result["params"].items()},
        "strategies": strat_results,
    }

    json_path = RESULTS_DIR / "daily_oos.json"
    with open(json_path, "w") as f:
        json.dump(oos_summary, f, indent=2, default=float)
    print(f"\nSaved: {json_path.name}")

    npz_path = RESULTS_DIR / "daily_oos_filter.npz"
    np.savez(
        npz_path,
        sigma_grid=sigma_grid,
        P=P,
        port_intercepts=port_intercepts,
        filtered_probs_cal=cal_result["filter"]["filtered_probs"],
        smoothed_probs_cal=cal_result["filter"]["smoothed_probs"],
        filtered_probs_oos=oos_filter["filtered_probs"],
        smoothed_probs_oos=oos_filter["smoothed_probs"],
        fitted_civ_oos=oos_filter["fitted_civ"],
    )
    print(f"Saved: {npz_path.name}")

    return oos_summary


def main():
    parser = argparse.ArgumentParser(
        description="Markov-switching CIV estimation pipeline"
    )
    parser.add_argument(
        "--monthly-only", action="store_true",
        help="Run only monthly estimation (skip daily OOS)"
    )
    args = parser.parse_args()

    t_start = time.time()

    # Step 1: Monthly estimation
    monthly_result, y_all, dates, L_flat, tau_flat, labels = step1_monthly()

    # Step 2: Daily OOS (unless --monthly-only)
    if not args.monthly_only:
        if DAILY_PORTFOLIOS.exists():
            step2_daily_oos(monthly_result)
        else:
            print(f"\nWARNING: Daily portfolios not found at {DAILY_PORTFOLIOS}")
            print("Run build_daily_civ.py first, then rerun without --monthly-only")

    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total pipeline time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
