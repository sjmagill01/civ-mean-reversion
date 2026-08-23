"""
Markov chain transition matrix construction and stationary distribution.
"""
import numpy as np


def build_vol_grid(sigma_base, delta_log, n_vol):
    """
    Log-even spacing of volatility states.

    sigma_j = sigma_base * exp((j - center) * delta_log)

    Parameters
    ----------
    sigma_base : float
        Center of the volatility grid.
    delta_log : float
        Log spacing between adjacent states.
    n_vol : int
        Number of volatility states.

    Returns
    -------
    sigma_grid : ndarray of shape (n_vol,)
    """
    center = (n_vol - 1) / 2.0
    j = np.arange(n_vol, dtype=float)
    sigma_grid = sigma_base * np.exp((j - center) * delta_log)
    return sigma_grid


def build_transition_matrix(n_vol, lambda_decay, kappa_mr):
    """
    Build transition matrix with distance decay and mean-reversion bias.

    P_ij proportional to exp(-lambda * |i-j|) * mr_bonus
    where mr_bonus = exp(kappa * step_toward_center / n_vol)
    and step_toward_center = dist_i * (dist_i - dist_j).

    Parameters
    ----------
    n_vol : int
        Number of volatility states.
    lambda_decay : float
        Controls sharpness of persistence (higher = more persistent).
    kappa_mr : float
        Mean-reversion strength (0 = none).

    Returns
    -------
    P : ndarray of shape (n_vol, n_vol)
        Row-stochastic transition matrix.
    """
    center = (n_vol - 1) / 2.0
    P = np.zeros((n_vol, n_vol))

    for i in range(n_vol):
        dist_i = i - center
        for j in range(n_vol):
            dist_j = j - center
            # Distance decay
            decay = np.exp(-lambda_decay * abs(i - j))
            # Mean-reversion bias: reward moves toward center
            step_toward_center = dist_i * (dist_i - dist_j)
            mr_bonus = np.exp(kappa_mr * step_toward_center / n_vol)
            P[i, j] = decay * mr_bonus

    # Row normalize
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    return P


def stationary_distribution(P):
    """
    Stationary distribution via eigendecomposition of P.T.

    Parameters
    ----------
    P : ndarray of shape (n, n)
        Row-stochastic transition matrix.

    Returns
    -------
    pi : ndarray of shape (n,)
        Stationary distribution (sums to 1).
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()
    return pi


def k_step_distribution(P, pi, k):
    """
    Iterate P.T @ pi for k steps.

    Parameters
    ----------
    P : ndarray of shape (n, n)
        Transition matrix.
    pi : ndarray of shape (n,)
        Initial distribution.
    k : int
        Number of steps.

    Returns
    -------
    pi_k : ndarray of shape (n,)
        Distribution after k steps.
    """
    pi_k = pi.copy()
    for _ in range(k):
        pi_k = P.T @ pi_k
    return pi_k


def k_step_distribution_eigen(P, pi, k):
    """
    Fast k-step distribution via eigendecomposition for large k.

    P^k = V * diag(lambda^k) * V^{-1}

    Parameters
    ----------
    P : ndarray of shape (n, n)
        Transition matrix.
    pi : ndarray of shape (n,)
        Initial distribution.
    k : int
        Number of steps.

    Returns
    -------
    pi_k : ndarray of shape (n,)
        Distribution after k steps.
    """
    eigenvalues, V = np.linalg.eig(P.T)
    V_inv = np.linalg.inv(V)
    # P.T^k @ pi = V @ diag(lam^k) @ V_inv @ pi
    lam_k = eigenvalues ** k
    pi_k = np.real(V @ (lam_k * (V_inv @ pi)))
    pi_k = np.maximum(pi_k, 0.0)
    pi_k = pi_k / pi_k.sum()
    return pi_k
