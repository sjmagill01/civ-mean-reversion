# Next Project: Credit Vol as Equity Vol Signal

## Core Idea

The Markov CIV model extracts a volatility regime from bond markets -- a place where vol is hard to observe directly. The equity market has its own vol measures (VIX, realized vol, option-implied vol). If credit-derived vol contains information that equity vol doesn't -- or contains it earlier -- we can:

1. Use credit regimes to forecast equity vol
2. Trade equity vol instruments (VIX futures, straddles) on credit signals
3. Exploit credit-equity vol divergences

This solves the vega problem from markov_vol: instead of trading CIV in the bond market (where vega compresses the signal 14x), trade equity vol instruments where the signal maps directly to the instrument.

## Hypotheses

**H1: Level correlation.** Markov expected vol tracks VIX contemporaneously. Sanity check.

**H2: Credit vol leads equity vol.** Regime deterioration in credit predicts VIX increases. Lead time = weeks to months? Credit markets are slower (monthly spreads) but the regime filter processes the full CIV cross-section.

**H3: Credit vol lags equity vol.** VIX spikes propagate to credit with delay. Credit vol confirms, doesn't predict.

**H4: Divergences are tradeable.** When credit regime is high but VIX is low (or vice versa), buy/sell equity vol. The dislocation closes as markets converge.

## Data We Already Have

- Markov filtered/smoothed regime probabilities (monthly, 270 months)
- Markov expected vol = probability-weighted sigma across states
- VIX daily (used in miniproject1 robustness, available from FRED)
- CRSP daily equity returns (for realized vol)
- Daily CIV (5,665 days from markov_vol)

## Data We Need

- VIX futures term structure (for trading simulation) -- from CBOE or WRDS
- Equity option prices or implied vol surface -- for straddle strategies
- OR: just use VIX level as proxy (no futures data needed for the research question)

## Analysis Plan

### Phase 1: Correlation Structure
- Contemporaneous correlation: Markov expected vol vs VIX (monthly)
- Rolling correlation: is the relationship stable or time-varying?
- Scatter plot: regime expected vol vs VIX, colored by regime state

### Phase 2: Lead-Lag
- Granger causality: does credit regime predict VIX changes? Vice versa?
- Cross-correlation at lags 0, 1, 2, 3, 6, 12 months
- Event study: when Markov regime transitions to stress (state 5+), what happens to VIX in the next 1, 2, 4 weeks?

### Phase 3: Divergence Signal
- Construct divergence = Markov expected vol (standardized) - VIX (standardized)
- Does divergence predict: next-month VIX change? Next-month equity returns?
- Is divergence mean-reverting? (If so, trade the convergence)

### Phase 4: Trading Strategy
- Signal: credit-equity vol divergence or credit regime transition
- Instrument: VIX itself (conceptual) or equity straddles (practical)
- Skip-a-day test at every frequency
- Walk-forward validation
- Dollar economics: this time trace units correctly from signal through instrument P&L
  - VIX futures: P&L = notional * delta_VIX * multiplier ($1000 per point)
  - Straddles: P&L = vega * delta_IV * notional
  - NO vega compression problem because we're trading vol directly

### Phase 5: Paper
- Frame as: "The bond market's view of volatility predicts equity vol"
- Credit vol is a slow-moving, cross-sectionally derived measure
- Equity vol is fast-moving, options-derived
- The two converge but at different speeds -- the divergence is the signal

## Directory Structure

```
credit_equity_vol/
    README.md
    config.py
    fetch_vix.py          # Pull VIX from FRED
    build_signals.py      # Construct Markov expected vol, divergence, etc.
    run_analysis.py       # Correlation, lead-lag, Granger
    run_strategies.py     # Trading strategies with correct units
    src/
        __init__.py
        signals.py        # Signal construction
        granger.py        # Lead-lag tests
        strategies.py     # With proper unit tracing
        diagnostics.py
    writeup/
        main.tex
        figures/
    data/
        results/
```

## Key Lesson Carried Forward

EVERY dollar P&L must trace: signal units -> instrument vega/multiplier -> dollars.
For VIX futures: 1 VIX point = $1,000 per contract. No vega conversion needed.
For equity straddles: P&L = position_vega * delta_IV. Units are clean.
The CIV-to-spread vega problem does NOT apply when trading equity vol instruments.

## Connection to Existing Work

- Uses the Markov regime from markov_vol (reads the filtered probabilities)
- Connects to miniproject1's VIX robustness test (which found credit signal strengthens after VIX control)
- The CIV-equity bridge paper already documented that credit and equity see the same risk differently
- This project asks: can the TIMING difference between credit and equity vol be exploited?
