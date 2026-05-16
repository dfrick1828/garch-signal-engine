
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="GARCH Signal Engine", page_icon="📈", layout="wide")
st.title("GARCH Signal Engine")
st.caption("Cloud dashboard for daily capital deployment signals.")

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\xa0", "").strip() for c in df.columns]
    return df

def fit_garch_11(returns):
    r = pd.Series(returns).dropna().astype(float).values * 100.0
    r = r[np.isfinite(r)]
    if len(r) < 30 or np.std(r) == 0:
        return None

    mu0 = np.mean(r)
    var0 = np.var(r)

    def nll(params):
        mu, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        eps = r - mu
        h = np.empty(len(r))
        h[0] = max(var0, 1e-8)
        for t in range(1, len(r)):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0 or not np.isfinite(h[t]):
                return 1e12
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + eps ** 2 / h)

    starts = [
        [mu0, max(var0 * 0.05, 1e-8), 0.05, 0.90],
        [mu0, max(var0 * 0.10, 1e-8), 0.10, 0.80],
        [mu0, max(var0 * 0.20, 1e-8), 0.15, 0.70],
        [mu0, max(var0 * 0.30, 1e-8), 0.20, 0.50],
    ]

    bounds = [(-50, 50), (1e-8, None), (0, 0.999), (0, 0.999)]
    constraints = {"type": "ineq", "fun": lambda p: 0.999 - p[2] - p[3]}

    best = None
    for s in starts:
        res = minimize(nll, s, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    mu, omega, alpha, beta = best.x
    eps = r - mu
    h = np.empty(len(r))
    h[0] = max(var0, 1e-8)
    for t in range(1, len(r)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]

    persistence = alpha + beta
    current_vol = np.sqrt(h[-1]) * np.sqrt(252)
    long_run_var = omega / (1 - persistence) if persistence < 1 else np.nan
    long_run_vol = np.sqrt(long_run_var) * np.sqrt(252) if np.isfinite(long_run_var) else np.nan
    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

    forecast = []
    next_var = h[-1]
    for _ in range(10):
        next_var = omega + persistence * next_var
        forecast.append(np.sqrt(next_var) * np.sqrt(252))

    return {
        "mu_daily_pct": mu,
        "alpha": alpha,
        "beta": beta,
        "alpha_beta": persistence,
        "current_ann_vol_pct": current_vol,
        "long_run_ann_vol_pct": long_run_vol,
        "half_life_days": half_life,
        "cond_vol_series_pct": np.sqrt(h) * np.sqrt(252),
        "forecast_vol_pct": np.array(forecast),
    }

def drawdown_from_pl(daily_pl):
    equity = pd.Series(daily_pl).cumsum()
    shifted = equity - equity.min() + 1
    return shifted / shifted.cummax() - 1

def parse_strategy_file(upload, fallback_strategy_name):
    if upload is None:
        return None
    if upload.name.lower().endswith(".xlsx"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)
    df = clean_columns(df)

    date_col = next((c for c in df.columns if c.lower() in ["date", "trade_date", "close_date"]), None)
    pl_col = next((c for c in df.columns if c.lower() in ["daily_pl", "daily p/l", "p/l", "pl", "net_pl", "net p/l", "profit_loss"]), None)
    cap_col = next((c for c in df.columns if c.lower() in ["capital_used", "capital used", "capital", "margin", "margin_req", "margin req.", "buying_power"]), None)
    strategy_col = next((c for c in df.columns if c.lower() == "strategy"), None)

    if date_col is None or pl_col is None:
        raise ValueError(f"{fallback_strategy_name}: file must include Date and Daily_PL or P/L.")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col])
    out["Strategy"] = df[strategy_col] if strategy_col else fallback_strategy_name
    out["Daily_PL"] = pd.to_numeric(df[pl_col], errors="coerce")
    out["Capital_Used"] = pd.to_numeric(df[cap_col], errors="coerce") if cap_col else np.nan
    return out.dropna(subset=["Date", "Daily_PL"])

def vix_regime_multiplier(vix):
    if pd.isna(vix): return 1.0
    if vix < 15: return 1.10
    if vix < 20: return 1.00
    if vix < 25: return 0.80
    return 0.60

def realized_vol_multiplier(rv):
    if pd.isna(rv): return 1.0
    if rv < 12: return 1.10
    if rv < 18: return 1.00
    if rv < 25: return 0.75
    return 0.55

def build_signals(strategy_df, max_capital, target_vol, min_mult, max_mult, current_vix=None, current_spx_rv=None):
    rows, charts = [], {}
    n_strats = strategy_df["Strategy"].nunique()

    for strategy, g in strategy_df.groupby("Strategy"):
        g = g.sort_values("Date").copy()
        if g["Capital_Used"].notna().sum() == 0:
            g["Capital_Used"] = max_capital / max(n_strats, 1)
        else:
            g["Capital_Used"] = g["Capital_Used"].ffill().bfill()

        g["Return"] = g["Daily_PL"] / g["Capital_Used"]
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["Return"])

        if len(g) < 30:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Recommended Capital": 0,
                         "Deployment %": 0, "Reason": "Need at least 30 observations."})
            continue

        model = fit_garch_11(g["Return"])
        if model is None:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Recommended Capital": 0,
                         "Deployment %": 0, "Reason": "Could not estimate GARCH model."})
            continue

        g["Drawdown"] = drawdown_from_pl(g["Daily_PL"])
        current_dd = float(g["Drawdown"].iloc[-1])
        kurt = float(g["Return"].kurt())
        p5 = float(g["Return"].quantile(0.05))
        win_rate = float((g["Daily_PL"] > 0).mean())
        avg_daily_return = float(g["Return"].mean())

        vol_mult = float(np.clip(target_vol / max(model["current_ann_vol_pct"], 1e-6), min_mult, max_mult))
        dd_mult = 1.00 if current_dd > -0.05 else 0.85 if current_dd > -0.10 else 0.65 if current_dd > -0.15 else 0.50 if current_dd > -0.20 else 0.35
        tail_mult = 1 / ((1 + max(kurt, 0) / 5) * (1 + abs(min(p5, 0)) * 10))
        regime_mult = vix_regime_multiplier(current_vix) * realized_vol_multiplier(current_spx_rv)
        edge_score = max(avg_daily_return, 0.00001) * max(win_rate, 0.01)
        raw_score = edge_score * vol_mult * dd_mult * tail_mult * regime_mult

        if model["current_ann_vol_pct"] > target_vol * 1.5 or current_dd < -0.15:
            signal = "Reduce"
        elif model["current_ann_vol_pct"] < target_vol and current_dd > -0.05:
            signal = "Deploy"
        else:
            signal = "Hold"

        rows.append({
            "Strategy": strategy, "Signal": signal, "Avg Daily Return %": avg_daily_return * 100,
            "Win Rate %": win_rate * 100, "GARCH Vol %": model["current_ann_vol_pct"],
            "Long-Run Vol %": model["long_run_ann_vol_pct"], "Alpha": model["alpha"],
            "Beta": model["beta"], "Alpha + Beta": model["alpha_beta"],
            "Vol Half-Life Days": model["half_life_days"], "Current Drawdown %": current_dd * 100,
            "Excess Kurtosis": kurt, "5th Percentile Return %": p5 * 100,
            "Vol Multiplier": vol_mult, "Drawdown Multiplier": dd_mult,
            "Tail Multiplier": tail_mult, "Regime Multiplier": regime_mult,
            "Raw Score": raw_score, "Reason": "Volatility, drawdown, tail-risk, and regime-adjusted allocation score."
        })
        charts[strategy] = {"dates": g["Date"], "vol": model["cond_vol_series_pct"], "forecast": model["forecast_vol_pct"]}

    signals = pd.DataFrame(rows)
    if "Raw Score" in signals.columns and signals["Raw Score"].fillna(0).sum() > 0:
        signals["Weight"] = signals["Raw Score"].fillna(0) / signals["Raw Score"].fillna(0).sum()
        signals["Recommended Capital"] = signals["Weight"] * max_capital
        signals["Deployment %"] = signals["Weight"] * 100
    else:
        signals["Weight"] = 0
        signals["Recommended Capital"] = 0
        signals["Deployment %"] = 0
    return signals.sort_values("Recommended Capital", ascending=False), charts

st.sidebar.header("Portfolio Controls")
max_capital = st.sidebar.number_input("Maximum Trading Capital", min_value=1000, value=100000, step=5000)
target_vol = st.sidebar.number_input("Target Annualized Strategy Volatility (%)", min_value=1.0, value=18.0, step=1.0)
min_mult = st.sidebar.slider("Minimum Strategy Multiplier", 0.0, 1.0, 0.40, 0.05)
max_mult = st.sidebar.slider("Maximum Strategy Multiplier", 0.5, 2.0, 1.25, 0.05)

st.sidebar.header("Market Regime Inputs")
current_vix = st.sidebar.number_input("Current VIX", min_value=0.0, value=18.0, step=0.5)
current_spx_rv = st.sidebar.number_input("SPX 20-Day Realized Vol (%)", min_value=0.0, value=12.0, step=0.5)
st.sidebar.write("Designed daily run time: **7:00 AM Mountain Time**")

st.header("1. Upload Strategy Daily Data")
st.markdown("""
Upload one file per strategy.

Required columns: `Date`, `Daily_PL` or `P/L`

Recommended columns: `Capital_Used` or `Margin`
""")

c1, c2, c3 = st.columns(3)
with c1: range_file = st.file_uploader("Range", type=["csv", "xlsx"])
with c2: weak_file = st.file_uploader("Weak", type=["csv", "xlsx"])
with c3: power_file = st.file_uploader("Power Hour", type=["csv", "xlsx"])

frames = []
for name, f in [("Range", range_file), ("Weak", weak_file), ("Power Hour", power_file)]:
    if f is not None:
        try:
            frames.append(parse_strategy_file(f, name))
        except Exception as e:
            st.error(str(e))

if not frames:
    st.info("Upload strategy-level daily files to generate signals.")
    st.stop()

strategy_df = pd.concat(frames, ignore_index=True)
st.subheader("Uploaded Data Preview")
st.dataframe(strategy_df.tail(25), use_container_width=True)

st.header("2. Today’s Capital Deployment Signal")
signals, charts = build_signals(strategy_df, max_capital, target_vol, min_mult, max_mult, current_vix, current_spx_rv)
st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

display_cols = ["Strategy", "Signal", "Recommended Capital", "Deployment %", "GARCH Vol %",
                "Current Drawdown %", "Excess Kurtosis", "Win Rate %", "Avg Daily Return %", "Alpha + Beta"]
display_cols = [c for c in display_cols if c in signals.columns]
st.dataframe(signals[display_cols], use_container_width=True)

total_deployed = float(signals["Recommended Capital"].sum())
cash_reserve = max_capital - total_deployed
m1, m2, m3 = st.columns(3)
m1.metric("Recommended Deployment", f"${total_deployed:,.0f}")
m2.metric("Cash Reserve", f"${cash_reserve:,.0f}")
m3.metric("Max Capital", f"${max_capital:,.0f}")

st.header("3. Charts")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(signals["Strategy"], signals["Recommended Capital"])
ax.set_title("Recommended Capital by Strategy")
ax.set_ylabel("Capital ($)")
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 5))
for strategy, ch in charts.items():
    ax2.plot(ch["dates"], ch["vol"], label=strategy)
ax2.set_title("Strategy-Level Conditional Volatility")
ax2.set_ylabel("Annualized Volatility (%)")
ax2.legend()
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 5))
days = np.arange(1, 11)
for strategy, ch in charts.items():
    ax3.plot(days, ch["forecast"], marker="o", label=strategy)
ax3.set_title("10-Day GARCH Volatility Forecast")
ax3.set_xlabel("Trading Days Forward")
ax3.set_ylabel("Annualized Volatility (%)")
ax3.legend()
st.pyplot(fig3)

st.header("4. Export Signal")
csv = signals.to_csv(index=False).encode("utf-8")
st.download_button("Download Today’s Signal CSV", csv, file_name=f"garch_signal_{datetime.now().date()}.csv", mime="text/csv")
st.warning("Research and decision-support only. This app does not place trades and should be validated before scaling capital.")
