"""
Trading strategies for OOS evaluation of Markov-switching CIV model.
All strategies designed for out-of-sample use.
"""
import re
import numpy as np


def sharpe(pnl, freq=252):
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    pnl : ndarray
        Daily PnL series.
    freq : int
        Annualization factor.

    Returns
    -------
    sr : float
    """
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(freq))


def mean_reversion_strategy(y, hold=1):
    """
    Fade the move: mean-revert CIV over last `hold` days.

    Position = -(y[t] - y[t-hold]) (equal-weighted across portfolios).
    PnL[t+1] = position[t] * (y[t+1] - y[t]), averaged across portfolios.

    Parameters
    ----------
    y : ndarray of shape (T, N_port)
        CIV panel.
    hold : int
        Lookback for signal.

    Returns
    -------
    position : ndarray of shape (T,)
        Aggregate position signal.
    pnl : ndarray of shape (T,)
        PnL in bps (multiply by 1e4).
    """
    T, n_port = y.shape
    position = np.zeros(T)
    pnl = np.zeros(T)

    for t in range(hold, T - 1):
        # Signal: mean-revert the average CIV move
        delta = np.nanmean(y[t] - y[t - hold])
        position[t] = -delta
        # PnL next period
        realized = np.nanmean(y[t + 1] - y[t])
        pnl[t + 1] = position[t] * realized * 1e4  # bps

    return position, pnl


def equity_mean_reversion(equity_returns):
    """
    Benchmark mean-reversion strategy on raw equity returns.

    Parameters
    ----------
    equity_returns : ndarray of shape (T,) or (T, N)
        Equity return series.

    Returns
    -------
    position : ndarray of shape (T,)
    pnl : ndarray of shape (T,)
    """
    if equity_returns.ndim == 1:
        equity_returns = equity_returns[:, None]

    T = len(equity_returns)
    position = np.zeros(T)
    pnl = np.zeros(T)

    for t in range(1, T - 1):
        signal = -np.nanmean(equity_returns[t])
        position[t] = signal
        realized = np.nanmean(equity_returns[t + 1])
        pnl[t + 1] = position[t] * realized * 1e4

    return position, pnl


def cross_sectional_dispersion(y):
    """
    Short overshooting portfolios, long undershooting.

    At each t, compute z-score across portfolios.
    Short if z > 0.5, long if z < -0.5.

    Parameters
    ----------
    y : ndarray of shape (T, N_port)

    Returns
    -------
    pnl : ndarray of shape (T,)
        PnL in bps.
    """
    T, n_port = y.shape
    pnl = np.zeros(T)

    for t in range(T - 1):
        row = y[t]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        mu = np.nanmean(row)
        std = np.nanstd(row)
        if std < 1e-10:
            continue

        z = (row - mu) / std
        # Position: short overvalued, long undervalued
        pos = np.zeros(n_port)
        pos[z > 0.5] = -1.0
        pos[z < -0.5] = 1.0

        # PnL from next-period change
        delta = y[t + 1] - y[t]
        valid_next = ~np.isnan(delta)
        if valid_next.any():
            pnl[t + 1] = np.nansum(pos[valid_next] * delta[valid_next]) * 1e4 / n_port

    return pnl


def leverage_spread_mean_rev(y, cols):
    """
    Mean-revert the low-leverage vs high-leverage CIV gap.

    Parameters
    ----------
    y : ndarray of shape (T, N_port)
    cols : list of str
        Column names to parse for leverage.

    Returns
    -------
    pnl : ndarray of shape (T,)
        PnL in bps.
    """
    T, n_port = y.shape

    # Find low-lev and high-lev columns
    low_idx = []
    high_idx = []

    for i, col in enumerate(cols):
        # Try 'L0.20' format
        m = re.search(r'L([\d.]+)', col)
        if m:
            lev = float(m.group(1))
        else:
            # Try '20%' format
            m = re.search(r'(\d+)%', col)
            if m:
                lev = float(m.group(1)) / 100.0
            else:
                continue

        if lev <= 0.25:
            low_idx.append(i)
        elif lev >= 0.75:
            high_idx.append(i)

    if not low_idx or not high_idx:
        return np.zeros(T)

    pnl = np.zeros(T)
    for t in range(1, T - 1):
        low_mean = np.nanmean(y[t, low_idx])
        high_mean = np.nanmean(y[t, high_idx])
        gap = high_mean - low_mean

        # Mean-revert the gap: if gap is wide, expect it to narrow
        gap_mean = np.nanmean([np.nanmean(y[s, high_idx]) - np.nanmean(y[s, low_idx])
                               for s in range(max(0, t - 20), t + 1)])
        signal = -(gap - gap_mean)

        # PnL: realized change in gap
        low_next = np.nanmean(y[t + 1, low_idx])
        high_next = np.nanmean(y[t + 1, high_idx])
        gap_next = high_next - low_next
        realized = gap_next - gap
        pnl[t + 1] = signal * realized * 1e4

    return pnl


def stress_hedge(smoothed_probs, realized_change, lookback=10, threshold=0.10):
    """
    Buy protection when P(high-vol state) rises.

    Parameters
    ----------
    smoothed_probs : ndarray of shape (T, N_vol)
    realized_change : ndarray of shape (T,)
        Realized CIV change (for PnL calculation).
    lookback : int
        Window for detecting probability increase.
    threshold : float
        Minimum increase in P(state >= 5) to trigger signal.

    Returns
    -------
    pnl : ndarray of shape (T,)
        PnL in bps.
    signal : ndarray of shape (T,)
        Binary signal (1 = buy protection).
    """
    T, n_vol = smoothed_probs.shape

    # High-vol states: top ~30% of states
    high_cutoff = max(1, n_vol - 2)  # e.g., states 5,6 for n_vol=7
    p_high = smoothed_probs[:, high_cutoff:].sum(axis=1)

    pnl = np.zeros(T)
    signal = np.zeros(T)

    for t in range(lookback, T - 1):
        p_change = p_high[t] - p_high[t - lookback]
        if p_change > threshold:
            signal[t] = 1.0
            # Buy protection = profit from CIV increase
            pnl[t + 1] = realized_change[t + 1] * 1e4 if t + 1 < T else 0.0

    return pnl, signal


def run_oos_analysis(y_cal, y_oos, cols, smoothed_oos, sigma_grid, P,
                     equity_oos=None, verbose=True):
    """
    Run all strategies on OOS data and report results.

    Parameters
    ----------
    y_cal : ndarray of shape (T_cal, N_port)
        Calibration data (for parameter estimation context).
    y_oos : ndarray of shape (T_oos, N_port)
        Out-of-sample CIV data.
    cols : list of str
        Column names.
    smoothed_oos : ndarray of shape (T_oos, N_vol)
        Smoothed state probabilities for OOS period.
    sigma_grid : ndarray of shape (N_vol,)
    P : ndarray of shape (N_vol, N_vol)
    equity_oos : ndarray, optional
        Equity returns for benchmark.
    verbose : bool

    Returns
    -------
    results : dict
        Strategy name -> {sharpe, hit_rate, cum_bps}.
    """
    results = {}

    # 1. CIV mean reversion
    _, pnl_mr = mean_reversion_strategy(y_oos, hold=1)
    active = pnl_mr != 0
    results['civ_mean_rev'] = {
        'sharpe': sharpe(pnl_mr[active]),
        'hit_rate': float(np.mean(pnl_mr[active] > 0)) if active.any() else 0.0,
        'cum_bps': float(np.nansum(pnl_mr)),
    }

    # 2. Cross-sectional dispersion
    pnl_cs = cross_sectional_dispersion(y_oos)
    active = pnl_cs != 0
    results['cross_section'] = {
        'sharpe': sharpe(pnl_cs[active]),
        'hit_rate': float(np.mean(pnl_cs[active] > 0)) if active.any() else 0.0,
        'cum_bps': float(np.nansum(pnl_cs)),
    }

    # 3. Leverage spread mean reversion
    pnl_lev = leverage_spread_mean_rev(y_oos, cols)
    active = pnl_lev != 0
    results['lev_spread'] = {
        'sharpe': sharpe(pnl_lev[active]),
        'hit_rate': float(np.mean(pnl_lev[active] > 0)) if active.any() else 0.0,
        'cum_bps': float(np.nansum(pnl_lev)),
    }

    # 4. Stress hedge
    realized_change = np.nanmean(np.diff(y_oos, axis=0), axis=1)
    realized_change = np.concatenate([[0.0], realized_change])
    pnl_stress, sig_stress = stress_hedge(smoothed_oos, realized_change)
    active = pnl_stress != 0
    results['stress_hedge'] = {
        'sharpe': sharpe(pnl_stress[active]),
        'hit_rate': float(np.mean(pnl_stress[active] > 0)) if active.any() else 0.0,
        'cum_bps': float(np.nansum(pnl_stress)),
    }

    # 5. Equity benchmark (if available)
    if equity_oos is not None:
        _, pnl_eq = equity_mean_reversion(equity_oos)
        active = pnl_eq != 0
        results['equity_mr_bench'] = {
            'sharpe': sharpe(pnl_eq[active]),
            'hit_rate': float(np.mean(pnl_eq[active] > 0)) if active.any() else 0.0,
            'cum_bps': float(np.nansum(pnl_eq)),
        }

    if verbose:
        print("\n=== OOS Strategy Results ===")
        print(f"{'Strategy':<20s} {'Sharpe':>8s} {'Hit%':>8s} {'CumBps':>10s}")
        print("-" * 48)
        for name, res in results.items():
            print(f"{name:<20s} {res['sharpe']:>8.2f} "
                  f"{res['hit_rate']*100:>7.1f}% "
                  f"{res['cum_bps']:>10.1f}")

    return results
