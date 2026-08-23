"""
Data loading for markov_vol project.
Handles both monthly (impcredvol2) and daily portfolio data.
"""
import re
import numpy as np
import pandas as pd

from markov_vol.config import PORTFOLIOS_WIDE, DAILY_PORTFOLIOS


def load_monthly_portfolios():
    """
    Load monthly CIV portfolios from impcredvol2.

    Reads PORTFOLIOS_WIDE parquet. Converts from pct to decimal if
    median > 1. Parses column names in monthly format: '1Y_20%' ->
    mat=1, lev=0.20.

    Returns
    -------
    y_all : ndarray of shape (T, N_port)
        CIV values in decimal.
    dates : ndarray
        Date index.
    L_flat : ndarray of shape (N_port,)
        Leverage for each portfolio.
    tau_flat : ndarray of shape (N_port,)
        Maturity in years for each portfolio.
    labels : list of str
        Column names.
    """
    df = pd.read_parquet(PORTFOLIOS_WIDE)

    # Identify the date column
    if 'date' in df.columns:
        dates = df['date'].values
        df = df.drop(columns=['date'])
    elif df.index.name == 'date' or df.index.name is not None:
        dates = df.index.values
    else:
        dates = np.arange(len(df))

    labels = list(df.columns)
    y_all = df.values.astype(float)

    # Convert from pct to decimal if needed
    if np.nanmedian(y_all) > 1.0:
        y_all = y_all / 100.0

    # Parse column names: monthly format '1Y_20%' or '3Y_40%'
    L_flat = np.zeros(len(labels))
    tau_flat = np.zeros(len(labels))

    for i, col in enumerate(labels):
        # Try '1Y_20%' format
        m = re.match(r'(\d+)Y_(\d+)%', col)
        if m:
            tau_flat[i] = float(m.group(1))
            L_flat[i] = float(m.group(2)) / 100.0
            continue

        # Try other formats
        m = re.match(r'(\d+)_(\d+)', col)
        if m:
            tau_flat[i] = float(m.group(1))
            L_flat[i] = float(m.group(2)) / 100.0
            continue

        # Fallback
        tau_flat[i] = 5.0
        L_flat[i] = 0.50

    return y_all, dates, L_flat, tau_flat, labels


def load_daily_portfolios():
    """
    Load daily CIV portfolios.

    Parses column names in daily format: 'L0.20_T1' -> lev=0.20, mat=1.

    Returns
    -------
    y_all : ndarray of shape (T, N_port)
    dates : ndarray
    L_flat : ndarray of shape (N_port,)
    tau_flat : ndarray of shape (N_port,)
    labels : list of str
    """
    df = pd.read_parquet(DAILY_PORTFOLIOS)

    # Identify the date column
    if 'date' in df.columns:
        dates = df['date'].values
        df = df.drop(columns=['date'])
    elif df.index.name == 'date' or df.index.name is not None:
        dates = df.index.values
    else:
        dates = np.arange(len(df))

    labels = list(df.columns)
    y_all = df.values.astype(float)

    # Convert from pct to decimal if needed
    if np.nanmedian(y_all) > 1.0:
        y_all = y_all / 100.0

    # Parse column names: daily format 'L0.20_T1'
    L_flat = np.zeros(len(labels))
    tau_flat = np.zeros(len(labels))

    for i, col in enumerate(labels):
        # Try 'L0.20_T1' format
        m = re.match(r'L([\d.]+)_T(\d+)', col)
        if m:
            L_flat[i] = float(m.group(1))
            tau_flat[i] = float(m.group(2))
            continue

        # Try '20%' style within column name
        m = re.match(r'(\d+)%.*?(\d+)', col)
        if m:
            L_flat[i] = float(m.group(1)) / 100.0
            tau_flat[i] = float(m.group(2))
            continue

        # Fallback
        L_flat[i] = 0.50
        tau_flat[i] = 5.0

    return y_all, dates, L_flat, tau_flat, labels
