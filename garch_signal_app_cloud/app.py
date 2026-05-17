
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

st.set_page_config(page_title="GARCH Signal Engine v5", page_icon="📈", layout="wide")
st.title("GARCH Signal Engine v5")
st.caption("Probabilistic regime allocation with exploitation + exploration capital.")

REGIMES = ["Normal", "Compression", "Trend Up", "Trend Down", "Vol Expansion"]

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\xa0", "").strip() for c in df.columns]
    return df

def fit_garch_11(returns):
    r = pd.Series(returns).dropna().astype(float).values * 100.0
    r = r[np.isfinite(r)]
    if len(r) < 30 or np.std(r) == 0:
        return None

    mu0, var0 = np.mean(r), np.var(r)

    def nll(p):
        mu, omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        eps = r - mu
        h = np.empty(len(r))
        h[0] = max(var0, 1e-8)
        for t in range(1, len(r)):
            h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]
            if h[t] <= 0 or not np.isfinite(h[t]):
                return 1e12
        return 0.5 * np.sum(np.log(2*np.pi) + np.log(h) + eps**2 / h)

    starts = [
        [mu0, max(var0*0.05, 1e-8), 0.05, 0.90],
        [mu0, max(var0*0.10, 1e-8), 0.10, 0.80],
        [mu0, max(var0*0.20, 1e-8), 0.15, 0.70],
    ]
    bounds = [(-50, 50), (1e-8, None), (0, 0.999), (0, 0.999)]
    cons = {"type": "ineq", "fun": lambda p: 0.999 - p[2] - p[3]}

    best = None
    for s in starts:
        res = minimize(nll, s, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res

    mu, omega, alpha, beta = best.x
    eps = r - mu
    h = np.empty(len(r))
    h[0] = max(var0, 1e-8)
    for t in range(1, len(r)):
        h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]

    persistence = alpha + beta
    forecast = []
    next_var = h[-1]
    for _ in range(10):
        next_var = omega + persistence * next_var
        forecast.append(np.sqrt(next_var) * np.sqrt(252))

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_beta": persistence,
        "current_vol": np.sqrt(h[-1]) * np.sqrt(252),
        "vol_series": np.sqrt(h) * np.sqrt(252),
        "forecast": np.array(forecast),
    }

def drawdown_from_pl(pl):
    equity = pd.Series(pl).cumsum()
    shifted = equity - equity.min() + 1
    return shifted / shifted.cummax() - 1

def classify_regime(row):
    vo = row.get("VIXOpenQuote", np.nan)
    vc = row.get("VIXCloseQuote", np.nan)
    so = row.get("UnderlyingOpenQuote", np.nan)
    sc = row.get("UnderlyingCloseQuote", np.nan)

    vchg = vc - vo if pd.notna(vo) and pd.notna(vc) else 0
    smove = sc / so - 1 if pd.notna(so) and pd.notna(sc) and so != 0 else 0

    if vo >= 25 or vchg >= 1.0 or abs(smove) >= 0.012:
        return "Vol Expansion"
    if smove >= 0.006:
        return "Trend Up"
    if smove <= -0.006:
        return "Trend Down"
    if vchg <= -0.5 and abs(smove) <= 0.006:
        return "Compression"
    return "Normal"

def parse_file(upload, name):
    if upload is None:
        return None
    df = pd.read_excel(upload) if upload.name.lower().endswith(".xlsx") else pd.read_csv(upload)
    df = clean_columns(df)

    # Alternate export naming support
    if "Date" not in df.columns and "OpenDate" in df.columns:
        df["Date"] = df["OpenDate"]
    if "Daily_PL" not in df.columns and "TotalNetProfitLoss" in df.columns:
        df["Daily_PL"] = df["TotalNetProfitLoss"]

    if "Date" not in df.columns:
        raise ValueError(f"{name}: missing Date/OpenDate column.")
    if "Daily_PL" not in df.columns:
        alt = next((c for c in df.columns if c.lower() in ["daily p/l", "p/l", "pl", "net p/l", "net_pl"]), None)
        if alt:
            df["Daily_PL"] = df[alt]
        else:
            raise ValueError(f"{name}: missing Daily_PL/TotalNetProfitLoss/P&L column.")

    if "BuyingPower" not in df.columns:
        alt = next((c for c in df.columns if c.lower() in ["capital_used", "capital used", "margin", "margin req.", "buying_power"]), None)
        df["BuyingPower"] = df[alt] if alt else np.nan

    keep = ["Date","Daily_PL","BuyingPower","UnderlyingOpenQuote","UnderlyingCloseQuote","VIXOpenQuote","VIXCloseQuote"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    out = df[keep].copy()
    out["Strategy"] = name
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["Daily_PL","BuyingPower","UnderlyingOpenQuote","UnderlyingCloseQuote","VIXOpenQuote","VIXCloseQuote"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["Date","Daily_PL"])

def aggregate_daily(trades):
    daily = trades.groupby(["Date","Strategy"]).agg(
        Daily_PL=("Daily_PL","sum"),
        BuyingPower=("BuyingPower","sum"),
        UnderlyingOpenQuote=("UnderlyingOpenQuote","first"),
        UnderlyingCloseQuote=("UnderlyingCloseQuote","last"),
        VIXOpenQuote=("VIXOpenQuote","first"),
        VIXCloseQuote=("VIXCloseQuote","last"),
        TradeCount=("Daily_PL","count")
    ).reset_index()
    daily["Regime"] = daily.apply(classify_regime, axis=1)
    daily["Return"] = daily["Daily_PL"] / daily["BuyingPower"].replace(0, np.nan)
    return daily.replace([np.inf, -np.inf], np.nan)

def regime_performance(daily):
    return daily.groupby(["Strategy","Regime"]).agg(
        Days=("Date","count"),
        Avg_Return=("Return","mean"),
        Win_Rate=("Daily_PL", lambda x: (x > 0).mean()),
        Total_PL=("Daily_PL","sum"),
        Worst_Day=("Daily_PL","min")
    ).reset_index()

def infer_probs(daily, vix, vix_change, gap, rv):
    hist = daily.drop_duplicates("Date")["Regime"].value_counts(normalize=True)
    p = {r: max(float(hist.get(r, 0.05)), 0.05) for r in REGIMES}

    if vix >= 25:
        p["Vol Expansion"] += 0.35
    elif vix >= 20:
        p["Vol Expansion"] += 0.18
    elif vix <= 15:
        p["Compression"] += 0.12
        p["Normal"] += 0.08

    if vix_change >= 1.0:
        p["Vol Expansion"] += 0.30
    elif vix_change >= 0.5:
        p["Vol Expansion"] += 0.15
    elif vix_change <= -0.75:
        p["Compression"] += 0.25
    elif vix_change <= -0.25:
        p["Compression"] += 0.10

    if gap >= 0.006:
        p["Trend Up"] += 0.22
        p["Vol Expansion"] += 0.08
    elif gap <= -0.006:
        p["Trend Down"] += 0.22
        p["Vol Expansion"] += 0.10
    elif abs(gap) <= 0.002:
        p["Normal"] += 0.10
        p["Compression"] += 0.08

    if rv >= 20:
        p["Vol Expansion"] += 0.25
    elif rv >= 15:
        p["Vol Expansion"] += 0.10
    elif rv <= 10:
        p["Compression"] += 0.15
        p["Normal"] += 0.10

    total = sum(p.values())
    return pd.DataFrame({"Regime": list(p.keys()), "Probability": [v/total for v in p.values()]}).sort_values("Probability", ascending=False)

def build_signals(daily, probs, max_capital, target_vol, min_mult, max_mult, exploration_pct, kurt_cut, dd_cut, vol_multiple):
    perf = regime_performance(daily)
    pmap = dict(zip(probs["Regime"], probs["Probability"]))
    rows, charts = [], {}

    for strategy, g in daily.groupby("Strategy"):
        g = g.sort_values("Date").copy()
        g["BuyingPower"] = g["BuyingPower"].ffill().bfill()
        g["Return"] = g["Daily_PL"] / g["BuyingPower"].replace(0, np.nan)
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["Return"])

        if len(g) < 30:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Raw Score": 0, "Extreme Risk Override": False})
            continue

        model = fit_garch_11(g["Return"])
        if model is None:
            rows.append({"Strategy": strategy, "Signal": "Insufficient data", "Raw Score": 0, "Extreme Risk Override": False})
            continue

        g["Drawdown"] = drawdown_from_pl(g["Daily_PL"])
        dd = float(g["Drawdown"].iloc[-1])
        kurt = float(g["Return"].kurt())
        p5 = float(g["Return"].quantile(0.05))
        avg_all = float(g["Return"].mean())
        win_all = float((g["Daily_PL"] > 0).mean())

        exp_ret, exp_win = 0.0, 0.0
        for reg, pr in pmap.items():
            rr = perf[(perf["Strategy"] == strategy) & (perf["Regime"] == reg)]
            if len(rr) and rr["Days"].iloc[0] >= 3:
                rret = float(rr["Avg_Return"].iloc[0])
                rwin = float(rr["Win_Rate"].iloc[0])
            else:
                rret, rwin = avg_all, win_all
            exp_ret += pr * rret
            exp_win += pr * rwin

        vol_mult = float(np.clip(target_vol / max(model["current_vol"], 1e-6), min_mult, max_mult))
        dd_mult = 1.0 if dd > -0.05 else 0.85 if dd > -0.10 else 0.65 if dd > -0.15 else 0.50 if dd > -0.20 else 0.35
        tail_mult = 1 / ((1 + max(kurt, 0)/5) * (1 + abs(min(p5, 0))*10))

        extreme = (kurt >= kurt_cut) or (dd*100 <= -abs(dd_cut)) or (model["current_vol"] >= target_vol * vol_multiple)
        raw = max(exp_ret, 0.00001) * max(exp_win, 0.01) * vol_mult * dd_mult * tail_mult
        if extreme:
            raw = 0

        if extreme:
            signal = "Avoid / No Exploration"
        elif exp_ret <= 0:
            signal = "Explore Only"
        elif model["current_vol"] > target_vol * 1.5 or dd < -0.15:
            signal = "Reduce + Explore"
        elif exp_ret > avg_all and model["current_vol"] <= target_vol * 1.25:
            signal = "Deploy"
        else:
            signal = "Hold + Explore"

        rows.append({
            "Strategy": strategy,
            "Signal": signal,
            "Extreme Risk Override": extreme,
            "Prob-Weighted Expected Return %": exp_ret * 100,
            "Prob-Weighted Win Rate %": exp_win * 100,
            "All-History Avg Return %": avg_all * 100,
            "GARCH Vol %": model["current_vol"],
            "Alpha + Beta": model["alpha_beta"],
            "Current Drawdown %": dd * 100,
            "Excess Kurtosis": kurt,
            "5th Percentile Return %": p5 * 100,
            "Vol Multiplier": vol_mult,
            "Drawdown Multiplier": dd_mult,
            "Tail Multiplier": tail_mult,
            "Raw Score": raw,
        })
        charts[strategy] = {"dates": g["Date"], "vol": model["vol_series"], "forecast": model["forecast"]}

    sig = pd.DataFrame(rows)

    eligible = ~sig["Extreme Risk Override"].fillna(False)
    n_eligible = int(eligible.sum())
    exploration_pool = max_capital * exploration_pct if n_eligible > 0 else 0
    exploration_each = exploration_pool / n_eligible if n_eligible > 0 else 0
    exploitation_pool = max_capital - exploration_pool

    total_score = sig["Raw Score"].fillna(0).sum()
    sig["Exploitation Weight"] = sig["Raw Score"].fillna(0) / total_score if total_score > 0 else 0
    sig["Exploitation Capital"] = sig["Exploitation Weight"] * exploitation_pool
    sig["Exploration Capital"] = np.where(eligible, exploration_each, 0)
    sig["Recommended Capital"] = sig["Exploitation Capital"] + sig["Exploration Capital"]
    sig["Deployment %"] = sig["Recommended Capital"] / max_capital * 100

    return sig.sort_values("Recommended Capital", ascending=False), charts, perf

# Sidebar
st.sidebar.header("Portfolio Controls")
max_capital = st.sidebar.number_input("Maximum Trading Capital", min_value=1000, value=100000, step=5000)
target_vol = st.sidebar.number_input("Target Annualized Strategy Volatility (%)", min_value=1.0, value=18.0, step=1.0)
min_mult = st.sidebar.slider("Minimum Strategy Multiplier", 0.0, 1.0, 0.40, 0.05)
max_mult = st.sidebar.slider("Maximum Strategy Multiplier", 0.5, 2.0, 1.25, 0.05)

st.sidebar.header("Exploration Capital")
exploration_pct = st.sidebar.slider("Exploration Pool (% of max capital)", 0.0, 0.30, 0.10, 0.01)

st.sidebar.header("Extreme Risk Overrides")
kurt_cut = st.sidebar.number_input("No-exploration kurtosis threshold", min_value=1.0, value=25.0, step=1.0)
dd_cut = st.sidebar.number_input("No-exploration drawdown threshold (%)", min_value=1.0, value=25.0, step=1.0)
vol_multiple = st.sidebar.number_input("No-exploration vol multiple vs target", min_value=1.0, value=3.0, step=0.25)

st.sidebar.header("Current Market Inputs")
current_vix = st.sidebar.number_input("Current VIX", min_value=0.0, value=18.0, step=0.5)
current_vix_change = st.sidebar.number_input("Current VIX Change", value=0.0, step=0.25)
current_spx_gap = st.sidebar.number_input("Overnight / Pre-Market SPX Gap (%)", value=0.0, step=0.1) / 100
recent_rv = st.sidebar.number_input("Recent SPX Realized Vol (%)", min_value=0.0, value=12.0, step=0.5)

# Uploads
st.header("1. Upload Strategy Export Files")
st.caption("Supports Date/Daily_PL or OpenDate/TotalNetProfitLoss exports.")

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

frames = []
for name, f in [("Range", range_file), ("Greenday", greenday_file), ("Weak", weak_file), ("Power Hour", power_file)]:
    if f is not None:
        try:
            frames.append(parse_file(f, name))
        except Exception as e:
            st.error(str(e))

if not frames:
    st.info("Upload strategy export files to generate signals.")
    st.stop()

trades = pd.concat(frames, ignore_index=True)
daily = aggregate_daily(trades)

st.subheader("Daily Aggregated Data Preview")
st.dataframe(daily.tail(25), use_container_width=True)

# Probabilities
st.header("2. Probabilistic Regime Forecast")
probs = infer_probs(daily, current_vix, current_vix_change, current_spx_gap, recent_rv)

col_a, col_b = st.columns(2)
with col_a:
    st.dataframe(probs.assign(Probability=lambda x: x["Probability"] * 100), use_container_width=True)
with col_b:
    figp, axp = plt.subplots(figsize=(6,4))
    axp.bar(probs["Regime"], probs["Probability"]*100)
    axp.set_title("Forecast Regime Probabilities")
    axp.set_ylabel("Probability (%)")
    axp.tick_params(axis="x", rotation=30)
    st.pyplot(figp)

st.header("3. Strategy Performance by Regime")
perf = regime_performance(daily)
pivot = perf.pivot_table(index="Strategy", columns="Regime", values="Avg_Return", aggfunc="mean") * 100
st.dataframe(pivot.round(3), use_container_width=True)

# Signals
st.header("4. Deployment Signal with Exploration Capital")
signals, charts, perf = build_signals(
    daily, probs, max_capital, target_vol, min_mult, max_mult,
    exploration_pct, kurt_cut, dd_cut, vol_multiple
)

display_cols = [
    "Strategy", "Signal", "Recommended Capital", "Exploitation Capital", "Exploration Capital", "Deployment %",
    "Prob-Weighted Expected Return %", "Prob-Weighted Win Rate %",
    "GARCH Vol %", "Current Drawdown %", "Excess Kurtosis", "Extreme Risk Override"
]
st.dataframe(signals[[c for c in display_cols if c in signals.columns]], use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommended Deployment", f"${signals['Recommended Capital'].sum():,.0f}")
m2.metric("Exploitation", f"${signals['Exploitation Capital'].sum():,.0f}")
m3.metric("Exploration", f"${signals['Exploration Capital'].sum():,.0f}")
m4.metric("Cash Reserve", f"${max_capital - signals['Recommended Capital'].sum():,.0f}")

st.header("5. Charts")
fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(signals))
ax.bar(x, signals["Exploitation Capital"], label="Exploitation")
ax.bar(x, signals["Exploration Capital"], bottom=signals["Exploitation Capital"], label="Exploration")
ax.set_xticks(x)
ax.set_xticklabels(signals["Strategy"])
ax.set_title("Recommended Capital by Strategy")
ax.set_ylabel("Capital ($)")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10,5))
for strategy, ch in charts.items():
    ax2.plot(ch["dates"], ch["vol"], label=strategy)
ax2.set_title("Strategy-Level Conditional Volatility")
ax2.set_ylabel("Annualized Volatility (%)")
ax2.legend()
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10,5))
days = np.arange(1, 11)
for strategy, ch in charts.items():
    ax3.plot(days, ch["forecast"], marker="o", label=strategy)
ax3.set_title("10-Day GARCH Volatility Forecast")
ax3.set_xlabel("Trading Days Forward")
ax3.set_ylabel("Annualized Volatility (%)")
ax3.legend()
st.pyplot(fig3)

st.header("6. Export")
st.download_button(
    "Download Today’s Signal CSV",
    signals.to_csv(index=False).encode("utf-8"),
    file_name=f"v5_exploration_signal_{datetime.now().date()}.csv",
    mime="text/csv"
)
st.download_button(
    "Download Regime Probabilities CSV",
    probs.to_csv(index=False).encode("utf-8"),
    file_name="regime_probabilities.csv",
    mime="text/csv"
)
st.download_button(
    "Download Regime Performance CSV",
    perf.to_csv(index=False).encode("utf-8"),
    file_name="strategy_regime_performance.csv",
    mime="text/csv"
)

st.warning("Research and decision-support only. Exploration capital preserves learning but does not override extreme risk controls.")
