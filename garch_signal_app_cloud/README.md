# GARCH Regime Signal Engine v3

This version upgrades the app from historical risk-adjusted allocation to regime-aware allocation.

## What changed

Old:
- Allocates mostly toward historically stronger strategies.

New:
- Classifies historical market regimes using SPX/VIX open-close behavior.
- Measures each strategy's return by regime.
- Lets the user select today's expected regime.
- Allocates based on expected edge in that regime, adjusted by GARCH volatility, drawdown, and tail risk.

## Supported strategy files

- Range
- Greenday
- Weak
- Power Hour

The app accepts the actual strategy export format with columns such as:

- Date
- Strategy
- Daily_PL
- BuyingPower
- UnderlyingOpenQuote
- UnderlyingCloseQuote
- VIXOpenQuote
- VIXCloseQuote

## Deploy

Replace the existing GitHub files with:

- app.py
- requirements.txt
- README.md

Then redeploy Streamlit.
