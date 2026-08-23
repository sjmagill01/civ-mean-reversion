"""
Diagnostics for Markov-switching CIV model.
"""
import numpy as np


def r_squared(y_obs, y_fitted):
    """
    Overall R-squared, handling NaN.

    Parameters
    ----------
    y_obs : ndarray
    y_fitted : ndarray

    Returns
    -------
    r2 : float
    """
    mask = ~np.isnan(y_obs) & ~np.isnan(y_fitted)
    if mask.sum() == 0:
        return np.nan
    ss_res = np.sum((y_obs[mask] - y_fitted[mask]) ** 2)
    ss_tot = np.sum((y_obs[mask] - np.mean(y_obs[mask])) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def r_squared_by_portfolio(y_obs, y_fitted):
    """
    Per-column (portfolio) R-squared.

    Parameters
    ----------
    y_obs : ndarray of shape (T, N_port)
    y_fitted : ndarray of shape (T, N_port)

    Returns
    -------
    r2_vec : ndarray of shape (N_port,)
    """
    n_port = y_obs.shape[1]
    r2_vec = np.zeros(n_port)
    for p in range(n_port):
        r2_vec[p] = r_squared(y_obs[:, p], y_fitted[:, p])
    return r2_vec


def constant_merton_baseline(y_obs):
    """
    Baseline model: cross-sectional mean per time step.

    Parameters
    ----------
    y_obs : ndarray of shape (T, N_port)

    Returns
    -------
    y_baseline : ndarray of shape (T, N_port)
        Fitted values (constant across portfolios at each t).
    r2 : float
        R-squared of this baseline.
    """
    row_means = np.nanmean(y_obs, axis=1, keepdims=True)
    y_baseline = np.broadcast_to(row_means, y_obs.shape).copy()
    r2 = r_squared(y_obs, y_baseline)
    return y_baseline, r2


def regime_persistence(smoothed_probs):
    """
    Analyze regime persistence from smoothed probabilities.

    Parameters
    ----------
    smoothed_probs : ndarray of shape (T, N_vol)

    Returns
    -------
    result : dict
        modal_states : ndarray of shape (T,)
        durations : list of (state, duration) tuples
        state_fractions : ndarray of shape (N_vol,)
        transition_count : int
    """
    T, n_vol = smoothed_probs.shape
    modal_states = np.argmax(smoothed_probs, axis=1)

    # Duration analysis
    durations = []
    current_state = modal_states[0]
    current_dur = 1

    for t in range(1, T):
        if modal_states[t] == current_state:
            current_dur += 1
        else:
            durations.append((current_state, current_dur))
            current_state = modal_states[t]
            current_dur = 1
    durations.append((current_state, current_dur))

    # Fraction of time in each state
    state_fractions = np.zeros(n_vol)
    for j in range(n_vol):
        state_fractions[j] = np.mean(modal_states == j)

    # Transition count
    transition_count = int(np.sum(np.diff(modal_states) != 0))

    return {
        'modal_states': modal_states,
        'durations': durations,
        'state_fractions': state_fractions,
        'transition_count': transition_count,
    }


def smirk_slope(y_obs, L_grid_unique, tau_grid_unique):
    """
    Compute d(CIV)/d(leverage) at each maturity.

    Parameters
    ----------
    y_obs : ndarray of shape (T, N_port)
    L_grid_unique : ndarray
        Unique leverage values.
    tau_grid_unique : ndarray
        Unique maturity values.

    Returns
    -------
    slopes : dict
        Keys are maturity values, values are mean slopes.
    """
    T, n_port = y_obs.shape
    n_lev = len(L_grid_unique)
    n_mat = len(tau_grid_unique)

    # Assume portfolios are ordered: for each maturity, iterate over leverage
    slopes = {}
    for m_idx, tau in enumerate(tau_grid_unique):
        # Columns for this maturity
        col_start = m_idx * n_lev
        col_end = col_start + n_lev
        if col_end > n_port:
            break

        y_slice = y_obs[:, col_start:col_end]  # (T, n_lev)
        mean_civ = np.nanmean(y_slice, axis=0)  # (n_lev,)

        if len(L_grid_unique) >= 2:
            # Simple linear regression slope
            L_arr = np.array(L_grid_unique[:n_lev])
            valid = ~np.isnan(mean_civ)
            if valid.sum() >= 2:
                slope = np.polyfit(L_arr[valid], mean_civ[valid], 1)[0]
            else:
                slope = np.nan
        else:
            slope = np.nan

        slopes[float(tau)] = float(slope)

    return slopes
