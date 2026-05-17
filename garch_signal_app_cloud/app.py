
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="GARCH Probabilistic Regime Engine", page_icon="📈", layout="wide")
st.title("GARCH Probabilistic Regime Engine")
st.caption("Probability-weighted regime forecasting for intraday SPX premium-harvesting deployment.")

# -----------------------------
# Helpers
# -----------------------------
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
        res = minimize(
            nll, s, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
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
        "cond_vol_series_pct": np.sqrt(h) * np.sqrt(252),
        "forecast_vol_pct": np.array(forecast),
    }

def drawdown_from_pl(daily_pl):
    equity = pd.Series(daily_pl).cumsum()
    shifted = equity - equity.min() + 1
    return shifted / shifted.cummax() - 1

def classify_regime(row):
    vix_open = row.get("VIXOpenQuote", np.nan)
    vix_close = row.get("VIXCloseQuote", np.nan)
    spx_open = row.get("UnderlyingOpenQuote", np.nan)
    spx_close = row.get("UnderlyingCloseQuote", np.nan)

    vix_change = vix_close - vix_open if pd.notna(vix_open) and pd.notna(vix_close) else 0
    spx_move = (spx_close / spx_open - 1) if pd.notna(spx_open) and pd.notna(spx_close) and spx_open != 0 else 0
    abs_move = abs(spx_move)

    if vix_open >= 25 or vix_change >= 1.0 or abs_move >= 0.012:
        return "Vol Expansion"
    if spx_move >= 0.006:
        return "Trend Up"
    if spx_move <= -0.006:
        return "Trend Down"
    if vix_change <= -0.5 and abs_move <= 0.006:
        return "Compression"
    return "Normal"

def parse_trade_file(upload, fallback_strategy_name):
    if upload is None:
        return None

    if upload.name.lower().endswith(".xlsx"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    df = clean_columns(df)

    if "Date" not in df.columns:
        raise ValueError(f"{fallback_strategy_name}: file must include Date column.")

    if "Daily_PL" not in df.columns:
        alt = next((c for c in df.columns if c.lower() in ["daily p/l", "p/l", "pl", "net_pl", "net p/l"]), None)
        if alt:
            df["Daily_PL"] = df[alt]
        else:
            raise ValueError(f"{fallback_strategy_name}: file must include Daily_PL or P/L.")

    df["Strategy"] = fallback_strategy_name

    if "BuyingPower" not in df.columns:
        alt = next((c for c in df.columns if c.lower() in ["capital_used", "capital used", "margin", "margin req.", "buying_power"]), None)
        df["BuyingPower"] = df[alt] if alt else np.nan

    keep_cols = [
        "Date", "Strategy", "Daily_PL", "BuyingPower",
        "UnderlyingOpenQuote", "UnderlyingCloseQuote",
        "VIXOpenQuote", "VIXCloseQuote",
        "OpenTime", "FinalTradeClosedTime"
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    out = df[keep_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Daily_PL"] = pd.to_numeric(out["Daily_PL"], errors="coerce")
    out["BuyingPower"] = pd.to_numeric(out["BuyingPower"], errors="coerce")

    for c in ["UnderlyingOpenQuote", "UnderlyingCloseQuote", "VIXOpenQuote", "VIXCloseQuote"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.dropna(subset=["Date", "Daily_PL"])

def aggregate_daily(trades):
    daily = trades.groupby(["Date", "Strategy"]).agg(
        Daily_PL=("Daily_PL", "sum"),
        BuyingPower=("BuyingPower", "sum"),
        UnderlyingOpenQuote=("UnderlyingOpenQuote", "first"),
        UnderlyingCloseQuote=("UnderlyingCloseQuote", "last"),
        VIXOpenQuote=("VIXOpenQuote", "first"),
        VIXCloseQuote=("VIXCloseQuote", "last"),
        TradeCount=("Daily_PL", "count")
    ).reset_index()

    daily["Regime"] = daily.apply(classify_regime, axis=1)
    daily["Return"] = daily["Daily_PL"] / daily["BuyingPower"].replace(0, np.nan)
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily

def build_regime_performance(daily):
    perf = daily.groupby(["Strategy", "Regime"]).agg(
        Days=("Date", "count"),
        Avg_Return=("Return", "mean"),
        Median_Return=("Return", "median"),
        Avg_PL=("Daily_PL", "mean"),
        Win_Rate=("Daily_PL", lambda x: (x > 0).mean()),
        Vol=("Return", "std"),
        Worst_Day=("Daily_PL", "min"),
        Total_PL=("Daily_PL", "sum")
    ).reset_index()
    return perf

def infer_regime_probabilities(daily, current_vix, current_vix_change, current_spx_gap, recent_realized_vol):
    """
    Heuristic first-pass regime probability model.
    This replaces the manual regime selector.
    It uses current market inputs to produce a probability distribution across regimes.
    """
    regimes = ["Normal", "Compression", "Trend Up", "Trend Down", "Vol Expansion"]

    # Start with historical base rates
    hist = daily.drop_duplicates("Date")["Regime"].value_counts(normalize=True)
    probs = {r: float(hist.get(r, 0.05)) for r in regimes}

    # Ensure no zero probabilities
    for r in regimes:
        probs[r] = max(probs[r], 0.05)

    # Vol expansion drivers
    if current_vix >= 25:
        probs["Vol Expansion"] += 0.35
    elif current_vix >= 20:
        probs["Vol Expansion"] += 0.18
    elif current_vix <= 15:
        probs["Compression"] += 0.12
        probs["Normal"] += 0.08

    if current_vix_change >= 1.0:
        probs["Vol Expansion"] += 0.30
    elif current_vix_change >= 0.5:
        probs["Vol Expansion"] += 0.15
    elif current_vix_change <= -0.75:
        probs["Compression"] += 0.25
    elif current_vix_change <= -0.25:
        probs["Compression"] += 0.10

    # Gap / overnight directional risk
    if current_spx_gap >= 0.006:
        probs["Trend Up"] += 0.22
        probs["Vol Expansion"] += 0.08
    elif current_spx_gap <= -0.006:
        probs["Trend Down"] += 0.22
        probs["Vol Expansion"] += 0.10
    elif abs(current_spx_gap) <= 0.002:
        probs["Normal"] += 0.10
        probs["Compression"] += 0.08

    # Realized volatility state
    if recent_realized_vol >= 20:
        probs["Vol Expansion"] += 0.25
    elif recent_realized_vol >= 15:
        probs["Vol Expansion"] += 0.10
        probs["Normal"] += 0.05
    elif recent_realized_vol <= 10:
        probs["Compression"] += 0.15
        probs["Normal"] += 0.10

    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    return pd.DataFrame({
        "Regime": list(probs.keys()),
        "Probability": list(probs.values())
    }).sort_values("Probability", ascending=False)

def build_probabilistic_signals(daily, regime_probs, max_capital, target_vol, min_mult, max_mult):
    rows = []
    charts = {}
    perf = build_regime_performance(daily)

    prob_map = dict(zip(regime_probs["Regime"], regime_probs["Probability"]))

    for strategy, g in daily.groupby("Strategy"):
        g = g.sort_values("Date").copy()
        g["BuyingPower"] = g["BuyingPower"].ffill().bfill()
        g["Return"] = g["Daily_PL"] / g["BuyingPower"].replace(0, np.nan)
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["Return"])

        if len(g) < 30:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Recommended Capital": 0, "Deployment %": 0})
            continue

        model = fit_garch_11(g["Return"])
        if model is None:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Recommended Capital": 0, "Deployment %": 0})
            continue

        g["Drawdown"] = drawdown_from_pl(g["Daily_PL"])
        current_dd = float(g["Drawdown"].iloc[-1])
        kurt = float(g["Return"].kurt())
        p5 = float(g["Return"].quantile(0.05))
        avg_return_all = float(g["Return"].mean())
        win_rate_all = float((g["Daily_PL"] > 0).mean())

        expected_return = 0.0
        expected_win = 0.0
        weighted_days = 0.0

        for regime, prob in prob_map.items():
            rr = perf[(perf["Strategy"] == strategy) & (perf["Regime"] == regime)]
            if len(rr) > 0 and rr["Days"].iloc[0] >= 3:
                r_ret = float(rr["Avg_Return"].iloc[0])
                r_win = float(rr["Win_Rate"].iloc[0])
                r_days = int(rr["Days"].iloc[0])
            else:
                r_ret = avg_return_all
                r_win = win_rate_all
                r_days = 0

            expected_return += prob * r_ret
            expected_win += prob * r_win
            weighted_days += prob * r_days

        vol_mult = float(np.clip(target_vol / max(model["current_ann_vol_pct"], 1e-6), min_mult, max_mult))

        if current_dd > -0.05:
            dd_mult = 1.00
        elif current_dd > -0.10:
            dd_mult = 0.85
        elif current_dd > -0.15:
            dd_mult = 0.65
        elif current_dd > -0.20:
            dd_mult = 0.50
        else:
            dd_mult = 0.35

        tail_penalty = 1 + max(kurt, 0) / 5
        downside_penalty = 1 + abs(min(p5, 0)) * 10
        tail_mult = 1 / (tail_penalty * downside_penalty)

        edge_score = max(expected_return, 0.00001) * max(expected_win, 0.01)
        raw_score = edge_score * vol_mult * dd_mult * tail_mult

        if expected_return <= 0:
            signal = "Avoid"
        elif model["current_ann_vol_pct"] > target_vol * 1.5 or current_dd < -0.15:
            signal = "Reduce"
        elif expected_return > avg_return_all and model["current_ann_vol_pct"] <= target_vol * 1.25:
            signal = "Deploy"
        else:
            signal = "Hold"

        rows.append({
            "Strategy": strategy,
            "Signal": signal,
            "Probability-Weighted Expected Return %": expected_return * 100,
            "All-History Avg Return %": avg_return_all * 100,
            "Probability-Weighted Win Rate %": expected_win * 100,
            "Weighted Regime Sample Days": weighted_days,
            "GARCH Vol %": model["current_ann_vol_pct"],
            "Long-Run Vol %": model["long_run_ann_vol_pct"],
            "Alpha + Beta": model["alpha_beta"],
            "Current Drawdown %": current_dd * 100,
            "Excess Kurtosis": kurt,
            "5th Percentile Return %": p5 * 100,
            "Vol Multiplier": vol_mult,
            "Drawdown Multiplier": dd_mult,
            "Tail Multiplier": tail_mult,
            "Raw Score": raw_score,
        })

        charts[strategy] = {
            "dates": g["Date"],
            "vol": model["cond_vol_series_pct"],
            "forecast": model["forecast_vol_pct"],
        }

    signals = pd.DataFrame(rows)

    if "Raw Score" in signals.columns and signals["Raw Score"].fillna(0).sum() > 0:
        signals["Weight"] = signals["Raw Score"].fillna(0) / signals["Raw Score"].fillna(0).sum()
        signals["Recommended Capital"] = signals["Weight"] * max_capital
        signals["Deployment %"] = signals["Weight"] * 100
    else:
        signals["Weight"] = 0
        signals["Recommended Capital"] = 0
        signals["Deployment %"] = 0

    return signals.sort_values("Recommended Capital", ascending=False), charts, perf

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Portfolio Controls")
max_capital = st.sidebar.number_input("Maximum Trading Capital", min_value=1000, value=100000, step=5000)
target_vol = st.sidebar.number_input("Target Annualized Strategy Volatility (%)", min_value=1.0, value=18.0, step=1.0)
min_mult = st.sidebar.slider("Minimum Strategy Multiplier", 0.0, 1.0, 0.40, 0.05)
max_mult = st.sidebar.slider("Maximum Strategy Multiplier", 0.5, 2.0, 1.25, 0.05)

st.sidebar.header("Current Market Inputs")
current_vix = st.sidebar.number_input("Current VIX", min_value=0.0, value=18.0, step=0.5)
current_vix_change = st.sidebar.number_input("Current VIX Change", value=0.0, step=0.25)
current_spx_gap = st.sidebar.number_input("Overnight / Pre-Market SPX Gap (%)", value=0.0, step=0.1) / 100.0
recent_realized_vol = st.sidebar.number_input("Recent SPX Realized Vol (%)", min_value=0.0, value=12.0, step=0.5)

st.sidebar.caption("Version 4: Probabilistic regime forecasting")

# -----------------------------
# Uploads
# -----------------------------
st.header("1. Upload Strategy Export Files")
st.markdown("""
Upload actual strategy export files. The app aggregates them by date and strategy, classifies historical regimes, estimates regime probabilities, and produces probability-weighted deployment signals.
""")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

with c1:
    range_file = st.file_uploader("Range", type=["csv", "xlsx"])
with c2:
    greenday_file = st.file_uploader("Greenday", type=["csv", "xlsx"])
with c3:
    weak_file = st.file_uploader("Weak", type=["csv", "xlsx"])
with c4:
    power_file = st.file_uploader("Power Hour", type=["csv", "xlsx"])

uploads = [
    ("Range", range_file),
    ("Greenday", greenday_file),
    ("Weak", weak_file),
    ("Power Hour", power_file),
]

frames = []
for name, f in uploads:
    if f is not None:
        try:
            frames.append(parse_trade_file(f, name))
        except Exception as e:
            st.error(str(e))

if not frames:
    st.info("Upload strategy export files to generate signals.")
    st.stop()

trades = pd.concat(frames, ignore_index=True)
daily = aggregate_daily(trades)

st.subheader("Daily Aggregated Data Preview")
st.dataframe(daily.tail(25), use_container_width=True)

# -----------------------------
# Regime probabilities
# -----------------------------
st.header("2. Probabilistic Regime Forecast")

regime_probs = infer_regime_probabilities(
    daily,
    current_vix=current_vix,
    current_vix_change=current_vix_change,
    current_spx_gap=current_spx_gap,
    recent_realized_vol=recent_realized_vol
)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.dataframe(regime_probs.assign(Probability=lambda x: x["Probability"] * 100), use_container_width=True)

with col_b:
    figp, axp = plt.subplots(figsize=(6, 4))
    axp.bar(regime_probs["Regime"], regime_probs["Probability"] * 100)
    axp.set_title("Forecast Regime Probabilities")
    axp.set_ylabel("Probability (%)")
    axp.tick_params(axis="x", rotation=30)
    st.pyplot(figp)

# -----------------------------
# Regime matrix
# -----------------------------
st.header("3. Strategy Performance by Regime")

perf = build_regime_performance(daily)

pivot = perf.pivot_table(
    index="Strategy",
    columns="Regime",
    values="Avg_Return",
    aggfunc="mean"
) * 100

st.caption("Average daily return by strategy and regime.")
st.dataframe(pivot.round(3), use_container_width=True)

# -----------------------------
# Signals
# -----------------------------
st.header("4. Probability-Weighted Deployment Signal")

signals, charts, perf = build_probabilistic_signals(
    daily,
    regime_probs,
    max_capital,
    target_vol,
    min_mult,
    max_mult
)

st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

display_cols = [
    "Strategy", "Signal", "Recommended Capital", "Deployment %",
    "Probability-Weighted Expected Return %",
    "Probability-Weighted Win Rate %",
    "GARCH Vol %", "Current Drawdown %",
    "Excess Kurtosis", "Alpha + Beta"
]
display_cols = [c for c in display_cols if c in signals.columns]

st.dataframe(signals[display_cols], use_container_width=True)

total_deployed = float(signals["Recommended Capital"].sum())
cash_reserve = max_capital - total_deployed

m1, m2, m3 = st.columns(3)
m1.metric("Recommended Deployment", f"${total_deployed:,.0f}")
m2.metric("Cash Reserve", f"${cash_reserve:,.0f}")
m3.metric("Max Capital", f"${max_capital:,.0f}")

# -----------------------------
# Charts
# -----------------------------
st.header("5. Charts")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(signals["Strategy"], signals["Recommended Capital"])
ax.set_title("Recommended Capital by Strategy — Probability-Weighted")
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

# -----------------------------
# Downloads
# -----------------------------
st.header("6. Export")
st.download_button(
    "Download Today’s Signal CSV",
    signals.to_csv(index=False).encode("utf-8"),
    file_name=f"probabilistic_regime_garch_signal_{datetime.now().date()}.csv",
    mime="text/csv"
)

st.download_button(
    "Download Regime Probability CSV",
    regime_probs.to_csv(index=False).encode("utf-8"),
    file_name="regime_probabilities.csv",
    mime="text/csv"
)

st.download_button(
    "Download Regime Performance CSV",
    perf.to_csv(index=False).encode("utf-8"),
    file_name="strategy_regime_performance.csv",
    mime="text/csv"
)

st.warning("Research and decision-support only. This app does not place trades.")
