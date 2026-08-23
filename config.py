"""
Configuration for markov_vol project.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_ROOT / "writeup" / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Data paths ─────────────────────────────────────────────────────────
# Monthly CIV portfolios are built by the companion impcredvol2 pipeline
# (WRDS bond spreads -> Merton-inverted CIV). Not included in this repo
# (WRDS license). Point MONTHLY_PORTFOLIOS_PATH at your own copy, or place
# it at data/portfolios_wide.parquet.
PORTFOLIOS_WIDE = Path(os.environ.get(
    "MONTHLY_PORTFOLIOS_PATH", DATA_DIR / "portfolios_wide.parquet"))
DAILY_PORTFOLIOS = DATA_DIR / "daily_portfolios.parquet"
DAILY_CIV_PANEL = DATA_DIR / "daily_civ_panel.parquet"
DAILY_TRACE = DATA_DIR / "daily_trace.parquet"

# ── Portfolio grids ────────────────────────────────────────────────────
LEVERAGE_GRID = [0.20, 0.40, 0.60, 0.80]
MATURITY_BUCKETS = [1, 3, 5, 7, 10]

# ── Model defaults ────────────────────────────────────────────────────
N_VOL_DEFAULT = 7

PARAM_NAMES = ['sigma_base', 'delta_log', 'lambda_decay', 'kappa_mr', 'sigma_eta']

PARAM_BOUNDS = {
    'sigma_base':   (0.10, 0.50),
    'delta_log':    (0.02, 0.50),
    'lambda_decay': (0.5,  8.0),
    'kappa_mr':     (0.0,  3.0),
    'sigma_eta':    (0.005, 0.15),
}

# ── Estimation ─────────────────────────────────────────────────────────
N_STARTS = 8
MAXITER = 400
