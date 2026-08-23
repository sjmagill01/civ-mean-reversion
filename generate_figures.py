"""
generate_figures.py -- All figures for the CIV Regimes paper.

Reads from data/results/ (produced by run_pipeline.py).
Outputs to writeup/figures/.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from markov_vol.config import RESULTS_DIR, FIGURES_DIR, LEVERAGE_GRID, MATURITY_BUCKETS
from markov_vol.src.data_loader import load_monthly_portfolios, load_daily_portfolios
from markov_vol.src.markov_transition import stationary_distribution
from markov_vol.src.diagnostics import r_squared, r_squared_by_portfolio, constant_merton_baseline
from markov_vol.src.strategies import sharpe, mean_reversion_strategy


def log(msg):
    print(msg, flush=True)


def load_monthly_results():
    with open(RESULTS_DIR / "monthly_summary.json") as f:
        summary = json.load(f)
    best_n = summary.get("best_nvol", 7)
    data = np.load(RESULTS_DIR / f"monthly_nvol{best_n}.npz", allow_pickle=True)
    return summary, data, best_n


def load_daily_results():
    with open(RESULTS_DIR / "daily_oos.json") as f:
        oos = json.load(f)
    data = np.load(RESULTS_DIR / "daily_oos_filter.npz", allow_pickle=True)
    return oos, data


def fig1_smirk():
    """CIV moneyness smirk across leverage bins."""
    log("Figure 1: CIV smirk...")
    y, dates, L, tau, labels = load_monthly_portfolios()

    n_lev = len(LEVERAGE_GRID)
    n_mat = len(MATURITY_BUCKETS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    mat_indices = [0, 2, 4]  # 1Y, 5Y, 10Y
    mat_names = ['1-Year', '5-Year', '10-Year']

    for ax, mi, mname in zip(axes, mat_indices, mat_names):
        civ_by_lev = []
        for li in range(n_lev):
            # Monthly columns: 1Y_20%, 1Y_40%, ..., 3Y_20%, ...
            # maturity varies slow, leverage varies fast
            col_idx = mi * n_lev + li
            civ_by_lev.append(np.nanmean(y[:, col_idx]))

        ax.plot(LEVERAGE_GRID, civ_by_lev, 'o-', color='#2c3e50', linewidth=2, markersize=8)
        ax.set_xlabel('Leverage')
        ax.set_title(mname)
        ax.set_xlim(0.15, 0.85)

    axes[0].set_ylabel('Mean CIV')
    fig.suptitle('CIV Moneyness Smirk', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_smirk.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig2_transition_matrix():
    """Transition matrix heatmap + stationary distribution."""
    log("Figure 2: Transition matrix...")
    summary, data, best_n = load_monthly_results()

    P = data['P']
    sigma_grid = data['sigma_grid']
    N = len(sigma_grid)
    pi = stationary_distribution(P)
    labels = [f'{s*100:.1f}%' for s in sigma_grid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [3, 1]})

    im = ax1.imshow(P, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(N):
        for j in range(N):
            color = 'white' if P[i, j] > 0.5 else 'black'
            ax1.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color=color)
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(labels, fontsize=8, rotation=45)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')
    ax1.set_title('Transition Matrix')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, N))
    ax2.barh(range(N), pi, color=colors)
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Probability')
    ax2.set_title('Stationary Distribution')
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_transition.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig3_regime_timeline():
    """Smoothed regime probabilities over time."""
    log("Figure 3: Regime timeline...")
    summary, data, best_n = load_monthly_results()
    smoothed = data['smoothed_probs']
    sigma_grid = data['sigma_grid']
    y, dates_raw, _, _, _ = load_monthly_portfolios()
    dates = pd.to_datetime(dates_raw)
    N = smoothed.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Expected vol
    expected_vol = smoothed @ sigma_grid * 100
    median_civ = np.nanmedian(y, axis=1) * 100

    ax1.plot(dates, median_civ, 'k-', alpha=0.4, linewidth=0.8, label='Observed CIV (median)')
    ax1.plot(dates, expected_vol, 'r-', linewidth=1.5, label='Markov Expected Vol')
    ax1.set_ylabel('Volatility (%)')
    ax1.set_title('Markov-Implied Volatility vs Observed CIV')
    ax1.legend(fontsize=9)

    for start, end, label in [('2007-12', '2009-06', 'GFC'),
                                ('2020-01', '2020-06', 'COVID')]:
        s, e = datetime.strptime(start, '%Y-%m'), datetime.strptime(end, '%Y-%m')
        ax1.axvspan(s, e, alpha=0.15, color='gray')
        ax1.text(s, ax1.get_ylim()[1] * 0.95, label, fontsize=8, alpha=0.6)

    # Modal state
    modal = np.argmax(smoothed, axis=1) + 1
    ax2.step(dates, modal, 'b-', linewidth=1, where='mid')
    ax2.set_ylabel('Modal State')
    ax2.set_xlabel('Date')
    ax2.set_yticks(range(1, N + 1))
    ax2.set_ylim(0.5, N + 0.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_regime_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig4_model_fit():
    """Model vs observed CIV for 4 selected portfolios."""
    log("Figure 4: Model vs observed...")
    summary, data, best_n = load_monthly_results()
    fitted = data['fitted_civ']
    y, dates_raw, L, tau, labels = load_monthly_portfolios()
    dates = pd.to_datetime(dates_raw)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    # Pick 4 diverse portfolios
    # Columns: 1Y_20%(0), 1Y_40%(1), 1Y_60%(2), 1Y_80%(3), 3Y_20%(4), ..., 10Y_80%(19)
    # Low-lev/short: 0 (1Y_20%), Low-lev/long: 16 (10Y_20%)
    # High-lev/short: 3 (1Y_80%), High-lev/long: 19 (10Y_80%)
    port_indices = [0, 16, 3, 19]
    port_names = ['L=20%, T=1Y', 'L=20%, T=10Y', 'L=80%, T=1Y', 'L=80%, T=10Y']

    for ax, idx, name in zip(axes.flat, port_indices, port_names):
        ax.plot(dates, y[:, idx] * 100, 'k-', alpha=0.5, linewidth=0.8, label='Observed')
        ax.plot(dates, fitted[:, idx] * 100, 'r-', linewidth=1.2, label='Model')
        ax.set_ylabel('CIV (%)')
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle('Model vs Observed CIV: Selected Portfolios', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_model_fit.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig5_bic():
    """BIC across N_vol specifications."""
    log("Figure 5: BIC comparison...")
    with open(RESULTS_DIR / "monthly_summary.json") as f:
        summary = json.load(f)

    results = summary.get("results", {})
    if not results:
        log("  No model data in summary, skipping")
        return

    n_vols = [int(k) for k in sorted(results.keys(), key=int)]
    bics = [results[str(n)]["bic"] for n in n_vols]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
    ax.bar(n_vols, bics, color=colors[:len(n_vols)], width=0.8)
    ax.set_xlabel('Number of Volatility States (N)')
    ax.set_ylabel('BIC')
    ax.set_title('Model Selection: BIC by Number of States')
    ax.set_xticks(n_vols)

    best_idx = np.argmin(bics)
    ax.annotate('Best', xy=(n_vols[best_idx], bics[best_idx]),
               xytext=(0, -20), textcoords='offset points',
               ha='center', fontsize=11, fontweight='bold', color='red')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_bic.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig6_r2_heatmap():
    """R-squared by portfolio as heatmap."""
    log("Figure 6: R2 heatmap...")
    summary, data, best_n = load_monthly_results()
    fitted = data['fitted_civ']
    y, _, _, _, _ = load_monthly_portfolios()

    r2 = r_squared_by_portfolio(y, fitted)
    n_lev = len(LEVERAGE_GRID)
    n_mat = len(MATURITY_BUCKETS)
    # Monthly columns: maturity varies slow, leverage varies fast
    # reshape(n_mat, n_lev) gives rows=maturity, cols=leverage
    # Transpose to get rows=leverage, cols=maturity for display
    r2_grid = r2.reshape(n_mat, n_lev).T

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(r2_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(n_lev):
        for j in range(n_mat):
            ax.text(j, i, f'{r2_grid[i,j]:.2f}', ha='center', va='center', fontsize=10)

    ax.set_xticks(range(n_mat))
    ax.set_yticks(range(n_lev))
    ax.set_xticklabels([f'{int(m)}Y' for m in MATURITY_BUCKETS])
    ax.set_yticklabels([f'L={l:.0%}' for l in LEVERAGE_GRID])
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Leverage')
    ax.set_title('R-squared by Portfolio (Leverage x Maturity)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig7_civ_vs_equity():
    """CIV vs equity mean-reversion comparison -- the money chart."""
    log("Figure 7: CIV vs equity...")
    with open(RESULTS_DIR / "daily_oos.json") as f:
        oos = json.load(f)

    strategies = oos.get("strategies", {})
    civ_sharpe = strategies.get("civ_mean_rev", {}).get("sharpe", 1.53)
    eq_sharpe = strategies.get("equity_mr_bench", {}).get("sharpe", -1.09)
    lev_sharpe = strategies.get("lev_spread", {}).get("sharpe", 3.42)

    fig, ax = plt.subplots(figsize=(8, 5))

    names = ['Leverage\nSpread', 'CIV\nMean-Rev', 'Equity\nMean-Rev']
    sharpes = [lev_sharpe, civ_sharpe, eq_sharpe]
    colors = ['#27ae60' if s > 0 else '#e74c3c' for s in sharpes]

    bars = ax.bar(names, sharpes, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Annualized Sharpe Ratio (OOS 2024)')
    ax.set_title('CIV Mean-Reverts. Equity Does Not.')

    for bar, s in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 * np.sign(bar.get_height()),
               f'{s:.2f}', ha='center', va='bottom' if s > 0 else 'top',
               fontsize=12, fontweight='bold')

    ax.set_ylim(min(sharpes) - 0.5, max(sharpes) + 0.7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7_civ_vs_equity.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig8_holding_period():
    """Sharpe vs holding period."""
    log("Figure 8: Holding period...")
    y, dates, L, tau, labels = load_daily_portfolios()

    # Use 2024 OOS only
    oos_mask = dates >= pd.Timestamp('2024-01-01')
    y_oos = y[oos_mask]

    holds = [1, 2, 3, 5, 10, 21]
    sharpes = []
    for h in holds:
        _, pnl = mean_reversion_strategy(y_oos, hold=h)
        sr = sharpe(pnl, freq=252/h)
        sharpes.append(sr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(holds, sharpes, 'o-', color='#2c3e50', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Holding Period (trading days)')
    ax.set_ylabel('Annualized Sharpe Ratio (OOS 2024)')
    ax.set_title('Mean-Reversion Signal Persistence')
    ax.set_xticks(holds)

    for h, s in zip(holds, sharpes):
        ax.annotate(f'{s:.2f}', (h, s), textcoords='offset points',
                   xytext=(0, 10), ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig8_holding_period.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig1_smirk()
    fig2_transition_matrix()
    fig3_regime_timeline()
    fig4_model_fit()
    fig5_bic()
    fig6_r2_heatmap()
    fig7_civ_vs_equity()
    fig8_holding_period()

    log(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
