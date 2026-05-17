# GARCH Signal Engine v5 — Exploration Capital

This version adds exploration capital to the probabilistic regime engine.

## What changed

- Splits recommended capital into:
  - Exploitation Capital
  - Exploration Capital
- Maintains small observation allocations for strategies that are not currently favored.
- Allows extreme-risk overrides where exploration capital is set to zero.

## Replace these files in GitHub

- app.py
- requirements.txt
- README.md

Then reboot/redeploy Streamlit.
