import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Optional: LLM-assisted thesis generation (OpenAI)
# Requires: pip install openai, and env var OPENAI_API_KEY set.
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

st.set_page_config(page_title="Speculation Stock Agent", page_icon="📈", layout="wide")

DEFAULT_UNIVERSE = [
    "PLTR","SOFI","RIVN","ROKU","U","COIN","NET","SNOW","ARM","TSLA",
    "SHOP","DDOG","MDB","CRSP","BE","QS","DKNG","ASTS","IONQ","RKLB"
]

@dataclass
class ScoreConfig:
    lookback_days: int = 252   # ~1 trading year
    momentum_window: int = 126 # ~6 months
    vol_window: int = 63       # ~3 months
    vol_spike_window: int = 20 # ~1 month
    min_price: float = 2.0
    max_drawdown_cap: float = 0.60

def to_series(x):
    """Coerce yfinance columns to a 1D Series safely."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype="float64")
        return x.iloc[:, 0]
    return x

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) else np.nan

def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    peak = close.cummax()
    dd = (close / peak) - 1.0
    return float(dd.min())

def zscore_latest(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) < window + 2:
        return float("nan")
    w = s.iloc[-window:]
    mu = float(w.mean())
    sigma = float(w.std(ddof=0))
    if sigma == 0.0:
        return 0.0
    return float((w.iloc[-1] - mu) / sigma)

def normalize_0_100(value, vmin: float, vmax: float):
    """
    Normalize a scalar OR pandas Series to a 0–100 scale.
    Returns same type as input (float for scalar, Series for Series).
    """
    if vmax == vmin or pd.isna(vmin) or pd.isna(vmax):
        if isinstance(value, pd.Series):
            return pd.Series(50.0, index=value.index)
        return 50.0

    if isinstance(value, pd.Series):
        return 100.0 * (value - vmin) / (vmax - vmin)

    if pd.isna(value):
        return float("nan")
    return float(100.0 * (value - vmin) / (vmax - vmin))

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_history(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, auto_adjust=True, progress=False)
            if df is not None and len(df) > 30:
                df = df.rename(columns=str.title)
                data[t] = df
        except Exception:
            continue
    return data

def score_ticker(df: pd.DataFrame, cfg: ScoreConfig) -> Dict[str, float]:
    close = to_series(df["Close"]).dropna()
    vol = to_series(df["Volume"]).dropna()
    if close.empty:
        return {}

    last_price = float(close.iloc[-1])
    if last_price < cfg.min_price:
        return {"eligible": 0}

    mom = float(close.pct_change(cfg.momentum_window).iloc[-1]) if len(close) > cfg.momentum_window else np.nan
    ret = close.pct_change().dropna()
    vol_ann = float(ret.iloc[-cfg.vol_window:].std() * math.sqrt(252)) if len(ret) >= cfg.vol_window else np.nan
    vol_z = zscore_latest(vol, cfg.vol_spike_window)
    rsi = compute_rsi(close, 14)
    mdd = max_drawdown(close)

    dd_penalty = 0.0
    if not np.isnan(mdd) and abs(mdd) > cfg.max_drawdown_cap:
        dd_penalty = -15.0

    return {
        "eligible": 1,
        "price": last_price,
        "momentum_6m": mom,
        "vol_ann": vol_ann,
        "vol_z": vol_z,
        "rsi": rsi,
        "max_drawdown": mdd,
        "dd_penalty": dd_penalty,
    }

def build_scores(history: Dict[str, pd.DataFrame], cfg: ScoreConfig) -> pd.DataFrame:
    rows = []
    for t, df in history.items():
        s = score_ticker(df, cfg)
        if not s:
            continue
        s["ticker"] = t
        rows.append(s)
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("ticker")

    out["mom_capped"] = out["momentum_6m"].clip(lower=-0.6, upper=2.0)
    out["vol_capped"] = out["vol_ann"].clip(lower=0.15, upper=1.50)
    out["volz_capped"] = out["vol_z"].clip(lower=-2.0, upper=6.0)

    out["rsi_score"] = 100 - (out["rsi"] - 50).abs() * 2.2
    out["rsi_score"] = out["rsi_score"].clip(lower=0, upper=100)

    out["mom_score"] = normalize_0_100(out["mom_capped"], out["mom_capped"].min(), out["mom_capped"].max())
    out["vol_score"] = normalize_0_100(out["vol_capped"], out["vol_capped"].min(), out["vol_capped"].max())
    out["volz_score"] = normalize_0_100(out["volz_capped"], out["volz_capped"].min(), out["volz_capped"].max())

    out["mom_score"] = out["mom_score"].clip(0, 100)
    out["vol_score"] = out["vol_score"].clip(0, 100)
    out["volz_score"] = out["volz_score"].clip(0, 100)

    w_mom, w_vol, w_volz, w_rsi = 0.40, 0.20, 0.25, 0.15
    out["spec_score"] = (
        w_mom * out["mom_score"] +
        w_vol * out["vol_score"] +
        w_volz * out["volz_score"] +
        w_rsi * out["rsi_score"] +
        out["dd_penalty"].fillna(0.0)
    )

    return out.sort_values("spec_score", ascending=False)

def suggest_allocation(top: pd.DataFrame, monthly_budget: float, mode: str) -> pd.DataFrame:
    if top.empty:
        return top

    if mode == "1 pick (100%)":
        alloc = [monthly_budget] + [0]*(len(top)-1)
    elif mode == "2 picks (70/30)":
        alloc = [0.70*monthly_budget, 0.30*monthly_budget] + [0]*(len(top)-2)
    elif mode == "3 picks (60/25/15)":
        w = [0.60, 0.25, 0.15]
        alloc = [w[i]*monthly_budget if i < 3 else 0 for i in range(len(top))]
    else:
        scores = top["spec_score"].clip(lower=0)
        alloc = (monthly_budget * scores / scores.sum()).tolist() if scores.sum() else [monthly_budget/len(top)] * len(top)

    out = top.copy()
    out["allocation_$"] = pd.Series(alloc, index=out.index).round(2)
    return out

def rationale_row(row: pd.Series) -> str:
    parts = []
    for label, key, fmt in [
        ("6M momentum", "momentum_6m", lambda v: f"{v*100:.1f}%"),
        ("volatility (ann.)", "vol_ann", lambda v: f"{v*100:.0f}%"),
        ("volume z-score", "vol_z", lambda v: f"{v:.2f}"),
        ("RSI", "rsi", lambda v: f"{v:.0f}"),
        ("max drawdown (period)", "max_drawdown", lambda v: f"{v*100:.0f}%"),
    ]:
        v = row.get(key, np.nan)
        if not pd.isna(v):
            parts.append(f"{label}: {fmt(float(v))}")
    return " • ".join(parts)

def summarize_metrics_for_llm(ticker: str, row: pd.Series) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"Spec score: {row.get('spec_score', np.nan):.1f}\n"
        f"Price: {row.get('price', np.nan):.2f}\n"
        f"6M momentum: {row.get('momentum_6m', np.nan):.3f}\n"
        f"Ann vol: {row.get('vol_ann', np.nan):.3f}\n"
        f"Volume z: {row.get('vol_z', np.nan):.3f}\n"
        f"RSI: {row.get('rsi', np.nan):.1f}\n"
        f"Max drawdown: {row.get('max_drawdown', np.nan):.3f}\n"
    )

def llm_thesis_and_invalidation(ticker: str, metrics_text: str, model: str, temperature: float = 0.3) -> dict:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Add `openai` to requirements and install it.")

    client = OpenAI()  # reads OPENAI_API_KEY from env
    prompt = f"""
You are an investing assistant helping a user build a speculative ($100/month) high-risk growth watchlist.
Generate a concise, decision-useful thesis and explicit invalidation rules for the ticker.

Use ONLY the provided metrics as quantitative evidence. You may reference common qualitative drivers (product, competition, dilution, regulation) but do not invent specific factual claims (e.g., exact revenue, customers, partnerships).
If you are uncertain about a claim, phrase it as a hypothesis.

Return STRICT JSON with keys:
- thesis_bullets: array of 3-5 bullets
- key_risks: array of 2-4 bullets
- invalidation_triggers: array of 3-6 triggers written as "IF ... THEN ..." statements
- monitoring_questions: array of 3-5 short questions

Metrics:
{metrics_text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )

    text = getattr(resp, "output_text", None)
    if not text and hasattr(resp, "output"):
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = ""
    text = (text or "").strip()

    import json, re
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Controls")
universe_text = st.sidebar.text_area(
    "Ticker universe (comma or newline separated)",
    value=", ".join(DEFAULT_UNIVERSE),
    height=140
)
tickers = [t.strip().upper() for t in universe_text.replace("\n", ",").split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))

monthly_budget = st.sidebar.number_input("Monthly budget ($)", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)

alloc_mode = st.sidebar.selectbox(
    "Allocation strategy",
    ["3 picks (60/25/15)", "2 picks (70/30)", "1 pick (100%)", "Score-weighted (top N)"],
    index=0
)

top_n = st.sidebar.slider("How many top tickers to show", 3, 15, 10)
min_price = st.sidebar.number_input("Min price filter ($)", min_value=0.5, max_value=100.0, value=2.0, step=0.5)
period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y"], index=1)
refresh = st.sidebar.button("🔄 Refresh data")

st.sidebar.divider()
st.sidebar.subheader("🤖 LLM thesis (optional)")
llm_enabled = st.sidebar.checkbox("Enable LLM-assisted thesis", value=False)
llm_model = st.sidebar.text_input("Model name", value="gpt-4o-mini")
llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

cfg = ScoreConfig(min_price=min_price)

# -----------------------------
# Main UI
# -----------------------------
st.title("📈 Speculation Stock Agent (Streamlit Prototype)")
st.caption("Scans a ticker universe, ranks by a transparent 'Speculation Score', and suggests a $100/month allocation. For education & fun—**not financial advice**.")

if refresh:
    fetch_history.clear()

with st.spinner("Fetching market data..."):
    history = fetch_history(tickers, period=period)

if not history:
    st.error("No data returned. Check tickers and try again.")
    st.stop()

scores = build_scores(history, cfg)
eligible = scores[scores["eligible"] == 1].copy()
if eligible.empty:
    st.warning("No eligible tickers after filters. Try lowering Min price filter or adjusting universe.")
    st.stop()

top = eligible.head(top_n)
rec = suggest_allocation(top, monthly_budget, alloc_mode)

st.subheader("✅ This Month's Suggested Buys")
rec_display = rec[[
    "spec_score","allocation_$","price","momentum_6m","vol_ann","vol_z","rsi","max_drawdown"
]].copy()

rec_display = rec_display.rename(columns={
    "spec_score":"Spec Score",
    "allocation_$":"Allocation ($)",
    "price":"Price",
    "momentum_6m":"6M Momentum",
    "vol_ann":"Vol (ann.)",
    "vol_z":"Volume Z",
    "rsi":"RSI",
    "max_drawdown":"Max DD (period)"
})

st.dataframe(
    rec_display.style.format({
        "Spec Score":"{:.1f}",
        "Allocation ($)":"${:.2f}",
        "Price":"${:.2f}",
        "6M Momentum":"{:.1%}",
        "Vol (ann.)":"{:.0%}",
        "Volume Z":"{:.2f}",
        "RSI":"{:.0f}",
        "Max DD (period)":"{:.0%}",
    }),
    use_container_width=True,
    height=380
)

st.subheader("🧾 Why these picks (signal-based)")
for t, row_iter in rec.iterrows():
    if row_iter.get("allocation_$", 0) <= 0:
        continue
    with st.expander(f"{t} — Allocate ${row_iter['allocation_$']:.2f}"):
        st.write(rationale_row(row_iter))
        st.write("**Watch next:** earnings date, guidance changes, dilution/cash runway (if early-stage), and whether volume remains elevated.")

st.divider()

st.subheader("🔎 Explore a Ticker")
ticker_sel = st.selectbox("Select ticker", options=list(top.index), index=0)
df = history[ticker_sel].copy().rename(columns=str.title)

left, right = st.columns([1.4, 1])
with left:
    st.write("**Price chart (adjusted)**")
    st.line_chart(df["Close"], height=280)
    st.write("**Volume**")
    st.bar_chart(df["Volume"], height=180)

with right:
    row = eligible.loc[ticker_sel]
    st.metric("Speculation Score", f"{row['spec_score']:.1f}")
    st.metric("Price", f"${row['price']:.2f}")
    st.metric("6M Momentum", f"{row['momentum_6m']*100:.1f}%" if not pd.isna(row["momentum_6m"]) else "n/a")
    st.metric("Volatility (ann.)", f"{row['vol_ann']*100:.0f}%" if not pd.isna(row["vol_ann"]) else "n/a")
    st.metric("Volume z-score", f"{row['vol_z']:.2f}" if not pd.isna(row["vol_z"]) else "n/a")
    st.metric("RSI", f"{row['rsi']:.0f}" if not pd.isna(row["rsi"]) else "n/a")
    st.metric("Max drawdown", f"{row['max_drawdown']*100:.0f}%" if not pd.isna(row["max_drawdown"]) else "n/a")

if llm_enabled:
    st.subheader("🤖 Thesis + Invalidation Rules (LLM-assisted)")
    if not _OPENAI_AVAILABLE:
        st.warning("OpenAI SDK not installed. Add `openai` to requirements.txt and reinstall.")
    else:
        if "thesis_cache" not in st.session_state:
            st.session_state["thesis_cache"] = {}

        metrics_text = summarize_metrics_for_llm(ticker_sel, row)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.code(metrics_text, language="text")
        with col2:
            st.info("This generates **hypotheses**, not facts. It uses only the metrics shown as quantitative evidence.")

        gen = st.button("Generate / Refresh thesis for selected ticker")
        cache_key = f"{ticker_sel}:{llm_model}:{llm_temp:.2f}"

        if gen or (cache_key not in st.session_state["thesis_cache"]):
            try:
                with st.spinner("Calling OpenAI…"):
                    st.session_state["thesis_cache"][cache_key] = llm_thesis_and_invalidation(
                        ticker_sel, metrics_text, model=llm_model, temperature=float(llm_temp)
                    )
            except Exception as e:
                st.error(f"LLM call failed: {e}")

        result = st.session_state["thesis_cache"].get(cache_key)
        if isinstance(result, dict):
            cA, cB = st.columns([1, 1])
            with cA:
                st.markdown("### Thesis (why it could run)")
                for b in result.get("thesis_bullets", []):
                    st.write(f"• {b}")
                st.markdown("### Key risks")
                for b in result.get("key_risks", []):
                    st.write(f"• {b}")
            with cB:
                st.markdown("### Invalidation triggers (explicit)")
                for b in result.get("invalidation_triggers", []):
                    st.write(f"• {b}")
                st.markdown("### Monitoring questions")
                for b in result.get("monitoring_questions", []):
                    st.write(f"• {b}")

st.divider()
st.subheader("⬇️ Export")
report = rec.copy()
report["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
csv = report.reset_index().to_csv(index=False).encode("utf-8")
st.download_button("Download recommendations CSV", data=csv, file_name="speculation_agent_recommendations.csv", mime="text/csv")
st.caption("Next upgrades: earnings/news catalysts, watchlists, backtesting, and automated alerts.")
