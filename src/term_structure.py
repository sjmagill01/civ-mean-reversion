"""
Term structure analysis for Markov-switching CIV model.
"""
import numpy as np

from markov_vol.src.merton import merton_spread
from markov_vol.src.markov_transition import stationary_distribution


def expected_spread_term_structure(sigma_grid, P, pi_current, L, tau_values):
    """
    Expected credit spread term structure under the Markov chain.

    Eigendecomposes P for fast P^k computation, then computes
    E[spread(tau)] = sum_j pi_k(j) * spread(sigma_j, L, tau).

    Parameters
    ----------
    sigma_grid : ndarray of shape (N_vol,)
    P : ndarray of shape (N_vol, N_vol)
    pi_current : ndarray of shape (N_vol,)
        Current state distribution.
    L : float
        Leverage ratio.
    tau_values : array-like
        Maturity values (years) at which to compute expected spread.

    Returns
    -------
    expected_spreads : ndarray of shape (len(tau_values),)
    """
    tau_values = np.asarray(tau_values, dtype=float)
    n_vol = len(sigma_grid)

    # Eigendecompose for fast P^k
    eigenvalues, V = np.linalg.eig(P.T)
    V_inv = np.linalg.inv(V)

    expected_spreads = np.zeros(len(tau_values))

    for i, tau in enumerate(tau_values):
        # Approximate k as proportional to tau (e.g., monthly steps)
        k = max(1, int(round(tau * 12)))

        # P.T^k @ pi_current
        lam_k = eigenvalues ** k
        pi_k = np.real(V @ (lam_k * (V_inv @ pi_current)))
        pi_k = np.maximum(pi_k, 0.0)
        pi_k = pi_k / pi_k.sum()

        # Expected spread at this maturity
        spreads_j = merton_spread(sigma_grid, L, tau)
        expected_spreads[i] = np.dot(pi_k, spreads_j)

    return expected_spreads


def model_civ_by_leverage(sigma_grid, P, L_values, tau, pi=None):
    """
    CIV smirk: model CIV as a function of leverage from stationary distribution.

    Parameters
    ----------
    sigma_grid : ndarray of shape (N_vol,)
    P : ndarray of shape (N_vol, N_vol)
    L_values : array-like
        Leverage values at which to compute CIV.
    tau : float
        Maturity.
    pi : ndarray of shape (N_vol,), optional
        State distribution. Defaults to stationary.

    Returns
    -------
    civ_values : ndarray of shape (len(L_values),)
    """
    L_values = np.asarray(L_values, dtype=float)

    if pi is None:
        pi = stationary_distribution(P)

    civ_values = np.zeros(len(L_values))
    for i, L in enumerate(L_values):
        spreads_j = merton_spread(sigma_grid, L, tau)
        civ_values[i] = np.dot(pi, spreads_j)

    return civ_values


def decompose_current_vs_longrun(sigma_grid, P, pi_current, L, tau_grid):
    """
    Decompose spread into current-state and stationary-state components.

    Parameters
    ----------
    sigma_grid : ndarray of shape (N_vol,)
    P : ndarray of shape (N_vol, N_vol)
    pi_current : ndarray of shape (N_vol,)
    L : float
    tau_grid : array-like

    Returns
    -------
    result : dict
        current_spread : spread under current distribution
        longrun_spread : spread under stationary distribution
        excess_spread : current - longrun (positive = stressed)
        tau_grid : the maturity grid used
    """
    tau_grid = np.asarray(tau_grid, dtype=float)
    pi_stat = stationary_distribution(P)

    current_spread = np.zeros(len(tau_grid))
    longrun_spread = np.zeros(len(tau_grid))

    for i, tau in enumerate(tau_grid):
        s_j = merton_spread(sigma_grid, L, tau)
        current_spread[i] = np.dot(pi_current, s_j)
        longrun_spread[i] = np.dot(pi_stat, s_j)

    return {
        'current_spread': current_spread,
        'longrun_spread': longrun_spread,
        'excess_spread': current_spread - longrun_spread,
        'tau_grid': tau_grid,
    }
