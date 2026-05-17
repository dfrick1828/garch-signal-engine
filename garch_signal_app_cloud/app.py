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
    ]

    bounds = [(-50, 50), (1e-8, None), (0, 0.999), (0, 0.999)]
    cons = {"type": "ineq", "fun": lambda p: 0.999 - p[2] - p[3]}

    best = None
    for s in starts:
        res = minimize(nll, s, method="SLSQP", bounds=bounds, constraints=cons)
        if best is None or res.fun < best.fun:
            best = res

    mu, omega, alpha, beta = best.x
    eps = r - mu
    h = np.empty(len(r))
    h[0] = max(var0, 1e-8)

    for t in range(1, len(r)):
        h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]

    persistence = alpha + beta

    return {
        "current_vol": np.sqrt(h[-1]) * np.sqrt(252),
        "forecast": np.array([np.sqrt(omega + persistence * h[-1]) * np.sqrt(252)] * 10),
        "alpha_beta": persistence,
        "vol_series": np.sqrt(h) * np.sqrt(252),
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
        raise ValueError(f"{name}: missing Daily_PL/TotalNetProfitLoss column.")

    if "BuyingPower" not in df.columns:
        df["BuyingPower"] = np.nan

    keep = [
        "Date",
        "Daily_PL",
        "BuyingPower",
        "UnderlyingOpenQuote",
        "UnderlyingCloseQuote",
        "VIXOpenQuote",
        "VIXCloseQuote"
    ]

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    out = df[keep].copy()
    out["Strategy"] = name

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    for c in keep[1:]:
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
    ).reset_index()

    daily["Regime"] = daily.apply(classify_regime, axis=1)
    daily["Return"] = daily["Daily_PL"] / daily["BuyingPower"].replace(0, np.nan)

    return daily.replace([np.inf, -np.inf], np.nan)

st.sidebar.header("Portfolio Controls")
max_capital = st.sidebar.number_input("Maximum Trading Capital", value=100000)
target_vol = st.sidebar.number_input("Target Volatility", value=18.0)
exploration_pct = st.sidebar.slider("Exploration Pool %", 0.0, 0.3, 0.1)

st.header("Upload Strategy Files")

range_file = st.file_uploader("Range")
greenday_file = st.file_uploader("Greenday")
weak_file = st.file_uploader("Weak")
power_file = st.file_uploader("Power Hour")

frames = []

for name, file in [
    ("Range", range_file),
    ("Greenday", greenday_file),
    ("Weak", weak_file),
    ("Power Hour", power_file),
]:
    if file is not None:
        frames.append(parse_file(file, name))

if not frames:
    st.stop()

trades = pd.concat(frames, ignore_index=True)
daily = aggregate_daily(trades)

st.subheader("Daily Data")
st.dataframe(daily.tail(20), use_container_width=True)

st.success("App loaded successfully with OpenDate and TotalNetProfitLoss support.")
