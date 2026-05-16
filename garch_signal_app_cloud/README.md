# GARCH Signal Engine — Cloud Version

Deploy on Streamlit Cloud:

1. Create a GitHub repository named `garch-signal-engine`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `sample_strategy_daily.csv`
3. Go to Streamlit Cloud.
4. Click New app.
5. Select the GitHub repo.
6. Set main file path to `app.py`.
7. Deploy.

Daily workflow:
- Open the Streamlit app link.
- Enter VIX, SPX 20-day realized volatility, and max capital.
- Upload Range, Weak, and Power Hour daily CSVs.
- Review the recommended deployment signal.

CSV format:

Date,Daily_PL,Capital_Used
2026-05-01,425,30000
2026-05-02,-310,30000
