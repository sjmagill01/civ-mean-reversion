"""
Hamilton filter for Markov-switching CIV model.
Includes Kim smoother and simulation.
"""
import numpy as np
from markov_vol.src.markov_transition import stationary_distribution


def estimate_intercepts(y_obs):
    """
    Portfolio intercepts: port_means - grand_mean.

    Parameters
    ----------
    y_obs : ndarray of shape (T, N_port)

    Returns
    -------
    port_intercepts : ndarray of shape (N_port,)
    """
    port_means = np.nanmean(y_obs, axis=0)
    grand_mean = np.nanmean(port_means)
    return port_means - grand_mean


def precompute_model_civ(sigma_grid, L_grid, tau_grid, port_intercepts=None):
    """
    Precompute model CIV for each (state, portfolio) pair.

    model_civ[j, p] = sigma_grid[j] + port_intercepts[p]

    L_grid and tau_grid are parallel arrays of length N_portfolios
    (NOT a cross-product).

    Parameters
    ----------
    sigma_grid : ndarray of shape (N_vol,)
    L_grid : ndarray of shape (N_port,)
    tau_grid : ndarray of shape (N_port,)
    port_intercepts : ndarray of shape (N_port,), optional

    Returns
    -------
    model_civ : ndarray of shape (N_vol, N_port)
    """
    n_vol = len(sigma_grid)
    n_port = len(L_grid)

    if port_intercepts is None:
        port_intercepts = np.zeros(n_port)

    # model_civ[j, p] = sigma_grid[j] + port_intercepts[p]
    model_civ = sigma_grid[:, None] + port_intercepts[None, :]
    return model_civ


def hamilton_filter(y_obs, sigma_grid, P, L_grid, tau_grid, sigma_eta,
                    pi_0=None, port_intercepts=None):
    """
    Full Hamilton filter for Markov-switching CIV.

    Parameters
    ----------
    y_obs : ndarray of shape (T, N_port)
        Observed CIV panel.
    sigma_grid : ndarray of shape (N_vol,)
        Volatility states.
    P : ndarray of shape (N_vol, N_vol)
        Transition matrix.
    L_grid : ndarray of shape (N_port,)
        Leverage for each portfolio (parallel to tau_grid).
    tau_grid : ndarray of shape (N_port,)
        Maturity for each portfolio (parallel to L_grid).
    sigma_eta : float
        Observation noise standard deviation.
    pi_0 : ndarray of shape (N_vol,), optional
        Initial state distribution. Defaults to stationary.
    port_intercepts : ndarray of shape (N_port,), optional
        Portfolio intercepts. If None, estimated from y_obs.

    Returns
    -------
    result : dict
        loglik, filtered_probs, predicted_probs, smoothed_probs,
        fitted_civ, residuals, model_civ, sigma_grid.
    """
    T, n_port = y_obs.shape
    n_vol = len(sigma_grid)

    if port_intercepts is None:
        port_intercepts = estimate_intercepts(y_obs)

    if pi_0 is None:
        pi_0 = stationary_distribution(P)

    # Precompute model CIV: (N_vol, N_port)
    model_civ = precompute_model_civ(sigma_grid, L_grid, tau_grid, port_intercepts)

    filtered = np.zeros((T, n_vol))
    predicted = np.zeros((T, n_vol))
    loglik = 0.0

    pi_prev = pi_0.copy()

    for t in range(T):
        # Predict
        pi_pred = P.T @ pi_prev
        pi_pred = np.maximum(pi_pred, 1e-300)
        predicted[t] = pi_pred

        # Observation likelihood per state
        y_t = y_obs[t]
        valid = ~np.isnan(y_t)
        if not valid.any():
            filtered[t] = pi_pred
            pi_prev = pi_pred
            continue

        # Log-likelihood for each state j
        log_lik_j = np.zeros(n_vol)
        for j in range(n_vol):
            resid = y_t[valid] - model_civ[j, valid]
            log_lik_j[j] = -0.5 * np.sum(resid**2) / sigma_eta**2 \
                           - 0.5 * valid.sum() * np.log(2 * np.pi * sigma_eta**2)

        # Log-sum-exp for numerical stability
        log_joint = np.log(pi_pred) + log_lik_j
        max_lj = np.max(log_joint)
        log_marginal = max_lj + np.log(np.sum(np.exp(log_joint - max_lj)))
        loglik += log_marginal

        # Bayes update
        pi_filt = np.exp(log_joint - log_marginal)
        pi_filt = np.maximum(pi_filt, 1e-300)
        pi_filt = pi_filt / pi_filt.sum()
        filtered[t] = pi_filt
        pi_prev = pi_filt

    # Kim smoother
    smoothed = kim_smoother(filtered, predicted, P)

    # Fitted CIV: filtered probs @ model_civ
    fitted_civ = filtered @ model_civ  # (T, N_port)
    residuals = y_obs - fitted_civ

    return {
        'loglik': loglik,
        'filtered_probs': filtered,
        'predicted_probs': predicted,
        'smoothed_probs': smoothed,
        'fitted_civ': fitted_civ,
        'residuals': residuals,
        'model_civ': model_civ,
        'sigma_grid': sigma_grid,
    }


def kim_smoother(filtered, predicted, P):
    """
    Kim smoother (backward pass).

    smoothed[t, j] = filtered[t, j] * sum_k P[j,k] * smoothed[t+1, k] / predicted[t+1, k]

    Parameters
    ----------
    filtered : ndarray of shape (T, N_vol)
    predicted : ndarray of shape (T, N_vol)
    P : ndarray of shape (N_vol, N_vol)

    Returns
    -------
    smoothed : ndarray of shape (T, N_vol)
    """
    T, n_vol = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[T - 1] = filtered[T - 1]

    for t in range(T - 2, -1, -1):
        for j in range(n_vol):
            pred_next = np.maximum(predicted[t + 1], 1e-300)
            ratio = smoothed[t + 1] / pred_next
            smoothed[t, j] = filtered[t, j] * np.sum(P[j, :] * ratio)

        # Renormalize
        s = smoothed[t].sum()
        if s > 0:
            smoothed[t] /= s

    return smoothed


def simulate_markov_merton(sigma_grid, P, L_grid, tau_grid, sigma_eta,
                           T=270, seed=42):
    """
    Simulate from the Markov-switching CIV model.

    IMPORTANT: n_port = len(L_grid), NOT n_lev * n_mat.

    Parameters
    ----------
    sigma_grid : ndarray of shape (N_vol,)
    P : ndarray of shape (N_vol, N_vol)
    L_grid : ndarray of shape (N_port,)
        Parallel to tau_grid.
    tau_grid : ndarray of shape (N_port,)
        Parallel to L_grid.
    sigma_eta : float
    T : int
    seed : int

    Returns
    -------
    y_obs : ndarray of shape (T, n_port)
    true_states : ndarray of shape (T,)
    """
    rng = np.random.default_rng(seed)
    n_vol = len(sigma_grid)
    n_port = len(L_grid)

    # Stationary distribution for initial state
    pi_stat = stationary_distribution(P)
    state = rng.choice(n_vol, p=pi_stat)

    true_states = np.zeros(T, dtype=int)
    y_obs = np.zeros((T, n_port))

    for t in range(T):
        true_states[t] = state
        # Model CIV for this state
        mu_t = sigma_grid[state] * np.ones(n_port)
        y_obs[t] = mu_t + rng.normal(0, sigma_eta, size=n_port)
        # Transition
        state = rng.choice(n_vol, p=P[state])

    return y_obs, true_states
