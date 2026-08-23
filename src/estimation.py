"""
Estimation for Markov-switching CIV model.
Multi-start Nelder-Mead with smart starts anchored to data mean.
"""
import numpy as np
from scipy.optimize import minimize

from markov_vol.config import PARAM_NAMES, PARAM_BOUNDS, N_STARTS, MAXITER
from markov_vol.src.markov_transition import build_vol_grid, build_transition_matrix
from markov_vol.src.hamilton_filter import hamilton_filter, estimate_intercepts


# ── Logit transforms to enforce bounds ─────────────────────────────────

def _logit(x, lo, hi):
    """Map x in (lo, hi) to unconstrained real line."""
    p = (x - lo) / (hi - lo)
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def _inv_logit(z, lo, hi):
    """Map unconstrained z back to (lo, hi)."""
    p = 1.0 / (1.0 + np.exp(-z))
    return lo + p * (hi - lo)


def pack_params(params):
    """
    Transform natural parameters to unconstrained space.

    Parameters
    ----------
    params : dict or list/array
        If dict, keyed by PARAM_NAMES. If array, in PARAM_NAMES order.

    Returns
    -------
    x : ndarray of shape (5,)
    """
    if isinstance(params, dict):
        vals = [params[name] for name in PARAM_NAMES]
    else:
        vals = list(params)

    x = np.zeros(len(PARAM_NAMES))
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        x[i] = _logit(vals[i], lo, hi)
    return x


def unpack_params(x):
    """
    Transform unconstrained vector back to natural parameters.

    Parameters
    ----------
    x : ndarray of shape (5,)

    Returns
    -------
    params : dict keyed by PARAM_NAMES.
    """
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        params[name] = _inv_logit(x[i], lo, hi)
    return params


def negative_loglik(x, n_vol, y_all, L_grid, tau_grid, port_intercepts):
    """
    Negative log-likelihood for optimizer.

    Parameters
    ----------
    x : ndarray of shape (5,)
        Unconstrained parameter vector.
    n_vol : int
    y_all : ndarray of shape (T, N_port)
    L_grid : ndarray of shape (N_port,)
    tau_grid : ndarray of shape (N_port,)
    port_intercepts : ndarray of shape (N_port,)

    Returns
    -------
    neg_ll : float
    """
    try:
        params = unpack_params(x)
        sigma_grid = build_vol_grid(params['sigma_base'], params['delta_log'], n_vol)
        P = build_transition_matrix(n_vol, params['lambda_decay'], params['kappa_mr'])
        result = hamilton_filter(
            y_all, sigma_grid, P, L_grid, tau_grid,
            params['sigma_eta'], port_intercepts=port_intercepts
        )
        neg_ll = -result['loglik']
        if not np.isfinite(neg_ll):
            return 1e12
        return neg_ll
    except Exception:
        return 1e12


def _build_smart_starts(grand_mean, n_starts, rng):
    """
    Build smart starting points anchored to data mean.

    First 3 starts: sigma_base = grand_mean exactly, with varying
    delta_log / lambda_decay / kappa_mr.
    Remaining starts: perturb grand_mean by ~10%.

    This is the fix for Pitfall 3: without anchoring to the data mean,
    the optimizer gets stuck 20K loglik points from the optimum.

    Parameters
    ----------
    grand_mean : float
        Grand mean of observed CIV.
    n_starts : int
    rng : np.random.Generator

    Returns
    -------
    starts : list of dict
    """
    # Clamp grand_mean to bounds
    lo, hi = PARAM_BOUNDS['sigma_base']
    sb = np.clip(grand_mean, lo + 0.01, hi - 0.01)

    starts = []

    # Fixed anchored starts with different structure params
    anchored_configs = [
        {'sigma_base': sb, 'delta_log': 0.10, 'lambda_decay': 2.0,
         'kappa_mr': 0.5, 'sigma_eta': 0.03},
        {'sigma_base': sb, 'delta_log': 0.05, 'lambda_decay': 4.0,
         'kappa_mr': 1.0, 'sigma_eta': 0.05},
        {'sigma_base': sb, 'delta_log': 0.20, 'lambda_decay': 1.0,
         'kappa_mr': 0.2, 'sigma_eta': 0.02},
    ]

    for cfg in anchored_configs[:min(n_starts, 3)]:
        starts.append(cfg)

    # Remaining starts: perturb around grand_mean
    for _ in range(n_starts - len(starts)):
        sb_pert = sb * (1.0 + rng.uniform(-0.10, 0.10))
        sb_pert = np.clip(sb_pert, lo + 0.01, hi - 0.01)
        cfg = {
            'sigma_base': sb_pert,
            'delta_log': rng.uniform(0.03, 0.40),
            'lambda_decay': rng.uniform(0.8, 6.0),
            'kappa_mr': rng.uniform(0.0, 2.5),
            'sigma_eta': rng.uniform(0.01, 0.12),
        }
        starts.append(cfg)

    return starts


def estimate_markov_civ(y_all, L_grid, tau_grid, n_vol=None, n_starts=None,
                        maxiter=None, verbose=True):
    """
    Estimate Markov-switching CIV model via multi-start Nelder-Mead.

    Parameters
    ----------
    y_all : ndarray of shape (T, N_port)
    L_grid : ndarray of shape (N_port,)
    tau_grid : ndarray of shape (N_port,)
    n_vol : int, optional
    n_starts : int, optional
    maxiter : int, optional
    verbose : bool

    Returns
    -------
    result : dict
        Best parameters, loglik, filter output, port_intercepts, etc.
    """
    from markov_vol.config import N_VOL_DEFAULT

    if n_vol is None:
        n_vol = N_VOL_DEFAULT
    if n_starts is None:
        n_starts = N_STARTS
    if maxiter is None:
        maxiter = MAXITER

    # Compute port intercepts and grand mean from data
    port_intercepts = estimate_intercepts(y_all)
    grand_mean = float(np.nanmean(y_all))

    if verbose:
        print(f"Data grand mean: {grand_mean:.4f}")
        print(f"Port intercepts range: [{port_intercepts.min():.4f}, {port_intercepts.max():.4f}]")

    rng = np.random.default_rng(42)
    starts = _build_smart_starts(grand_mean, n_starts, rng)

    best_negll = np.inf
    best_x = None
    best_params = None

    for i, cfg in enumerate(starts):
        x0 = pack_params(cfg)

        res = minimize(
            negative_loglik, x0,
            args=(n_vol, y_all, L_grid, tau_grid, port_intercepts),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        if verbose:
            print(f"  Start {i+1}/{n_starts}: negLL = {res.fun:.2f}"
                  f"  (sigma_base={unpack_params(res.x)['sigma_base']:.4f})")

        if res.fun < best_negll:
            best_negll = res.fun
            best_x = res.x.copy()
            best_params = unpack_params(res.x)

    # Post-estimation: warn if sigma_base far from data mean
    if best_params is not None:
        sb = best_params['sigma_base']
        pct_diff = abs(sb - grand_mean) / grand_mean
        if pct_diff > 0.50:
            print(f"WARNING: sigma_base ({sb:.4f}) is {pct_diff*100:.0f}% away "
                  f"from data grand mean ({grand_mean:.4f}). "
                  f"Consider rerunning with more starts.")

    # Re-run filter at best params for full output
    sigma_grid = build_vol_grid(best_params['sigma_base'],
                                best_params['delta_log'], n_vol)
    P = build_transition_matrix(n_vol, best_params['lambda_decay'],
                                best_params['kappa_mr'])
    filter_out = hamilton_filter(
        y_all, sigma_grid, P, L_grid, tau_grid,
        best_params['sigma_eta'], port_intercepts=port_intercepts
    )

    return {
        'params': best_params,
        'loglik': filter_out['loglik'],
        'negll': best_negll,
        'n_vol': n_vol,
        'sigma_grid': sigma_grid,
        'P': P,
        'filter': filter_out,
        'port_intercepts': port_intercepts,
        'grand_mean': grand_mean,
    }


def compute_bic(loglik, n_params, n_obs):
    """
    Bayesian Information Criterion.

    BIC = -2 * loglik + n_params * ln(n_obs)

    Parameters
    ----------
    loglik : float
    n_params : int
    n_obs : int

    Returns
    -------
    bic : float
    """
    return -2.0 * loglik + n_params * np.log(n_obs)
