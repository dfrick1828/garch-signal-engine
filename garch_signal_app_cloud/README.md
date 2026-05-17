# GARCH Probabilistic Regime Engine v4

This version replaces manual regime selection with probabilistic regime forecasting.

## What changed

Old v3:
- User manually selected one expected regime.

New v4:
- User enters current market inputs:
  - Current VIX
  - VIX change
  - SPX pre-market / overnight gap
  - Recent SPX realized volatility
- App estimates probabilities for:
  - Normal
  - Compression
  - Trend Up
  - Trend Down
  - Vol Expansion
- Strategy allocation is based on probability-weighted expected return by regime.

## Allocation concept

Allocation is now based on:

Probability-weighted expected regime edge
adjusted by:
- GARCH volatility
- drawdown
- tail risk

## Deploy

Replace existing GitHub files with:

- app.py
- requirements.txt
- README.md

Then reboot or redeploy the Streamlit app.
