"""
Pure Python/NumPy Merton model: spread, vega, and CIV inversion.
No C dependency.
"""
import numpy as np
from scipy.stats import norm


def merton_spread(sigma_A, L, tau):
    """
    Vectorized Merton credit spread (decimal).

    Parameters
    ----------
    sigma_A : float or array
        Asset volatility.
    L : float or array
        Leverage ratio (debt / assets).
    tau : float or array
        Time to maturity in years.

    Returns
    -------
    spread : float or array
        Credit spread in decimal.
    """
    sigma_A = np.asarray(sigma_A, dtype=float)
    L = np.asarray(L, dtype=float)
    tau = np.asarray(tau, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (-np.log(L)) / (sigma_A * sqrt_tau) + 0.5 * sigma_A * sqrt_tau
    d2 = d1 - sigma_A * sqrt_tau

    bond_val = norm.cdf(d2) + norm.cdf(-d1) / L
    bond_val = np.maximum(bond_val, 1e-300)

    spread = -(1.0 / tau) * np.log(bond_val)
    spread = np.maximum(spread, 0.0)
    return spread


def merton_vega(sigma_A, L, tau):
    """
    Analytical d(spread)/d(sigma_A).

    Parameters
    ----------
    sigma_A, L, tau : float or array
        Same as merton_spread.

    Returns
    -------
    vega : float or array
    """
    sigma_A = np.asarray(sigma_A, dtype=float)
    L = np.asarray(L, dtype=float)
    tau = np.asarray(tau, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (-np.log(L)) / (sigma_A * sqrt_tau) + 0.5 * sigma_A * sqrt_tau
    d2 = d1 - sigma_A * sqrt_tau

    # Partials of d1, d2 w.r.t. sigma_A
    dd1 = np.log(L) / (sigma_A**2 * sqrt_tau) + 0.5 * sqrt_tau
    dd2 = dd1 - sqrt_tau

    bond_val = norm.cdf(d2) + norm.cdf(-d1) / L
    bond_val = np.maximum(bond_val, 1e-300)

    # d(bond_val)/d(sigma_A)
    dbond = norm.pdf(d2) * dd2 - norm.pdf(d1) * dd1 / L

    # spread = -(1/tau) * ln(bond_val)
    # d(spread)/d(sigma_A) = -(1/tau) * dbond / bond_val
    vega = -(1.0 / tau) * dbond / bond_val
    return vega


def invert_spread_to_civ(spread, L, tau, max_iter=50, tol=1e-10):
    """
    Newton-Raphson inversion: given an observed spread, find the
    implied asset volatility (CIV).

    Parameters
    ----------
    spread : float
        Observed credit spread (decimal).
    L : float
        Leverage ratio.
    tau : float
        Time to maturity in years.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on spread residual.

    Returns
    -------
    sigma : float
        Credit-implied volatility, or NaN if inversion fails.
    """
    sigma = 0.25

    for _ in range(max_iter):
        s_model = merton_spread(sigma, L, tau)
        residual = float(s_model) - spread

        if abs(residual) < tol:
            break

        v = float(merton_vega(sigma, L, tau))
        if abs(v) < 1e-20:
            return np.nan

        delta = np.clip(residual / v, -0.5, 0.5)
        sigma = np.clip(sigma - delta, 1e-6, 10.0)

    # Final convergence check
    s_check = float(merton_spread(sigma, L, tau))
    if abs(s_check - spread) > 1e-5:
        return np.nan

    return sigma


def invert_spread_to_civ_vec(spread, L, tau, max_iter=50):
    """
    Vectorized Newton-Raphson inversion over whole arrays.

    Mirrors invert_spread_to_civ exactly: start 0.25, step clipped to
    +/-0.5, sigma clipped to [1e-6, 10], dead-vega and non-converged
    entries (|residual| > 1e-5) set to NaN.

    Parameters
    ----------
    spread, L, tau : array
        Observed spread (decimal), leverage, maturity (years).

    Returns
    -------
    sigma : array
        CIV, NaN where inversion fails.
    """
    spread = np.asarray(spread, dtype=float)
    L = np.asarray(L, dtype=float)
    tau = np.asarray(tau, dtype=float)

    sigma = np.full_like(spread, 0.25)
    for _ in range(max_iter):
        residual = merton_spread(sigma, L, tau) - spread
        v = merton_vega(sigma, L, tau)
        dead = np.abs(v) < 1e-20
        step = np.where(dead, 0.0, residual / np.where(dead, 1.0, v))
        sigma = np.clip(sigma - np.clip(step, -0.5, 0.5), 1e-6, 10.0)

    resid_final = np.abs(merton_spread(sigma, L, tau) - spread)
    sigma = np.where(resid_final > 1e-5, np.nan, sigma)
    return sigma
