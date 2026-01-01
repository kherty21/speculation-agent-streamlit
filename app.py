import os
import math
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests

# Optional: LLM-assisted thesis generation (OpenAI)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# =========================
# App config
# =========================
st.set_page_config(page_title="Speculation Stock Agent", page_icon="📈", layout="wide")

DEFAULT_UNIVERSE = [
    "PLTR", "SOFI", "RIVN", "ROKU", "U", "COIN", "NET", "SNOW", "ARM", "TSLA",
    "SHOP", "DDOG", "MDB", "CRSP", "BE", "QS", "DKNG", "ASTS", "IONQ", "RKLB"
]

JOURNAL_FILE = "decision_journal.csv"
BENCHMARK_TICKER = "QQQ"


# =========================
# Data structures
# =========================
@dataclass
class ScoreConfig:
    momentum_window: int = 126
    vol_window: int = 63
    vol_spike_window: int = 20
    min_price: float = 2.0
    max_drawdown_cap: float = 0.60


# =========================
# Helpers
# =========================
def to_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype="float64")
        return x.iloc[:, 0]
    return x


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    close = close.dropna()
    if len(close) < period + 2:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) else float("nan")


def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return float("nan")
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
    if vmax == vmin or pd.isna(vmin) or pd.isna(vmax):
        if isinstance(value, pd.Series):
            return pd.Series(50.0, index=value.index)
        return 50.0
    if isinstance(value, pd.Series):
        return 100.0 * (value - vmin) / (vmax - vmin)
    if pd.isna(value):
        return float("nan")
    return float(100.0 * (value - vmin) / (vmax - vmin))


def days_until(date_iso: Optional[str]) -> Optional[int]:
    if not date_iso:
        return None
    try:
        d = dt.date.fromisoformat(date_iso)
        return (d - dt.date.today()).days
    except Exception:
        return None

def _get_finnhub_key() -> Optional[str]:
    try:
        k = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        k = None
    return k or os.getenv("FINNHUB_API_KEY")


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def finnhub_next_earnings_date(ticker: str) -> Optional[str]:
    """
    Returns next earnings date (ISO string) using Finnhub earnings calendar.
    """
    token = _get_finnhub_key()
    if not token:
        return None

    # Query a 6-month window ahead (you can widen this if you want)
    today = dt.date.today()
    to_date = today + dt.timedelta(days=180)

    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {
        "from": today.isoformat(),
        "to": to_date.isoformat(),
        "symbol": ticker,
        "token": token,
    }

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        items = data.get("earningsCalendar", []) or []
        if not items:
            return None

        # Pick earliest report date >= today
        dates = []
        for it in items:
            d = it.get("date")
            if d:
                try:
                    dd = dt.date.fromisoformat(d)
                    if dd >= today:
                        dates.append(dd)
                except Exception:
                    pass

        if not dates:
            return None
        return min(dates).isoformat()
    except Exception:
        return None


@st.cache_data(ttl=30 * 60, show_spinner=False)
def finnhub_company_news(ticker: str, limit: int = 10) -> List[dict]:
    """
    Returns recent company news items via Finnhub.
    Normalizes to: title, publisher, url, published_at, age_hours
    """
    token = _get_finnhub_key()
    if not token:
        return []

    today = dt.date.today()
    frm = today - dt.timedelta(days=14)  # lookback window
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": frm.isoformat(),
        "to": today.isoformat(),
        "token": token,
    }

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        raw = r.json() or []
        if not isinstance(raw, list) or not raw:
            return []

        now = dt.datetime.now(dt.timezone.utc)
        out = []
        for n in raw[:limit]:
            ts = n.get("datetime")  # unix seconds
            published = None
            if ts:
                try:
                    published = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
                except Exception:
                    published = None

            age_hours = (now - published).total_seconds() / 3600.0 if published else np.nan

            out.append({
                "title": n.get("headline", "") or "",
                "publisher": n.get("source", "") or "",
                "url": n.get("url", "") or "",
                "published_at": published.isoformat().replace("+00:00", "Z") if published else "",
                "age_hours": age_hours,
            })
        return out
    except Exception:
        return []

def _get_finnhub_key() -> Optional[str]:
    try:
        k = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        k = None
    return k or os.getenv("FINNHUB_API_KEY")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def finnhub_earnings_calendar_bulk(from_iso: str, to_iso: str, cache_day: str) -> pd.DataFrame:
    """
    One bulk call to Finnhub earnings calendar.
    cache_day is a dummy parameter (e.g., today's date) to force daily cache refresh.
    Returns DataFrame with columns: symbol, date, hour (may be blank), etc.
    """
    token = _get_finnhub_key()
    if not token:
        return pd.DataFrame()

    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"from": from_iso, "to": to_iso, "token": token}

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json() or {}
        items = data.get("earningsCalendar", []) or []
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        # Normalize minimal columns we need
        if "symbol" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def build_next_earnings_map(df_calendar: pd.DataFrame) -> Dict[str, str]:
    """
    Builds {SYMBOL: next_earnings_iso} from a bulk earnings calendar DataFrame.
    Picks earliest earnings date >= today per symbol.
    """
    if df_calendar is None or df_calendar.empty:
        return {}

    today = dt.date.today()
    out: Dict[str, str] = {}

    # ensure date parsing
    tmp = df_calendar.copy()
    tmp["date_parsed"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
    tmp = tmp.dropna(subset=["date_parsed", "symbol"])

    # keep future dates only
    tmp = tmp[tmp["date_parsed"] >= today]

    if tmp.empty:
        return {}

    tmp = tmp.sort_values(["symbol", "date_parsed"])
    firsts = tmp.groupby("symbol", as_index=False).first()

    for _, row in firsts.iterrows():
        sym = str(row["symbol"]).upper().strip()
        d = row["date_parsed"]
        if isinstance(d, dt.date):
            out[sym] = d.isoformat()

    return out


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def finnhub_company_news_daily(ticker: str, limit: int, cache_day: str) -> List[dict]:
    """
    Company news via Finnhub, cached per ticker per day.
    cache_day is today's date (forces daily refresh).
    """
    token = _get_finnhub_key()
    if not token:
        return []

    today = dt.date.today()
    frm = today - dt.timedelta(days=14)

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": frm.isoformat(), "to": today.isoformat(), "token": token}

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        raw = r.json() or []
        if not isinstance(raw, list) or not raw:
            return []

        now = dt.datetime.now(dt.timezone.utc)
        out = []

        for n in raw[:limit]:
            ts = n.get("datetime")
            published = None
            if ts:
                try:
                    published = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
                except Exception:
                    published = None

            age_hours = (now - published).total_seconds() / 3600.0 if published else np.nan

            out.append({
                "title": n.get("headline", "") or "",
                "publisher": n.get("source", "") or "",
                "url": n.get("url", "") or "",
                "published_at": published.isoformat().replace("+00:00", "Z") if published else "",
                "age_hours": age_hours,
            })

        return out
    except Exception:
        return []


# =========================
# yfinance fetching
# =========================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_history(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, auto_adjust=True, progress=False)
            if df is not None and len(df) > 30:
                df = df.rename(columns=str.title)
                data[t] = df
        except Exception:
            continue
    return data


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_last_close(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="7d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return float("nan")
        df = df.rename(columns=str.title)
        c = to_series(df["Close"]).dropna()
        return float(c.iloc[-1]) if len(c) else float("nan")
    except Exception:
        return float("nan")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_close_on_or_after(ticker: str, date_iso: str) -> float:
    try:
        d0 = dt.date.fromisoformat(date_iso)
        d1 = d0 + dt.timedelta(days=7)
        df = yf.download(
            ticker,
            start=d0.isoformat(),
            end=d1.isoformat(),
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            return float("nan")
        df = df.rename(columns=str.title)
        c = to_series(df["Close"]).dropna()
        return float(c.iloc[0]) if len(c) else float("nan")
    except Exception:
        return float("nan")

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_next_earnings_date(ticker: str) -> Optional[str]:
    # Finnhub primary
    d = finnhub_next_earnings_date(ticker)
    if d:
        return d

    # Optional fallback to yfinance (keeps behavior if Finnhub misses)
    try:
        tk = yf.Ticker(ticker)
        try:
            ed = tk.get_earnings_dates(limit=8)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                dates = [x.date() for x in ed.index.to_pydatetime()]
                today = dt.date.today()
                future = [x for x in dates if x >= today]
                if future:
                    return min(future).isoformat()
        except Exception:
            pass

        cal = getattr(tk, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                vals = cal.loc["Earnings Date"].dropna().tolist()
                if vals:
                    return pd.to_datetime(vals[0]).date().isoformat()
    except Exception:
        pass

    return None


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_news(ticker: str, limit: int = 10) -> List[dict]:
    cache_day = dt.date.today().isoformat()
    items = finnhub_company_news_daily(ticker, limit=int(limit), cache_day=cache_day)
    return items

    # Optional fallback to yfinance
    items2: List[dict] = []
    try:
        tk = yf.Ticker(ticker)
        raw = []
        if hasattr(tk, "get_news"):
            try:
                raw = tk.get_news(count=limit) or []
            except Exception:
                raw = []
        if not raw:
            raw = getattr(tk, "news", None) or []

        if not raw:
            return []

        now = dt.datetime.now(dt.timezone.utc)
        for n in raw[:limit]:
            title = n.get("title") or ""
            publisher = n.get("publisher") or n.get("source") or ""
            link = n.get("link") or n.get("url") or ""
            ts = n.get("providerPublishTime") or n.get("provider_publish_time")

            published = None
            if ts:
                try:
                    published = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
                except Exception:
                    published = None

            age_hours = (now - published).total_seconds() / 3600.0 if published else np.nan
            items2.append({
                "title": title,
                "publisher": publisher,
                "url": link,
                "published_at": published.isoformat().replace("+00:00", "Z") if published else "",
                "age_hours": age_hours,
            })
        return items2
    except Exception:
        return []

def catalyst_score(next_earnings_iso: Optional[str], news_items: List[dict], news_recency_hours: int = 72) -> float:
    """
    Simple heuristic catalyst score (0-100):
    - Earnings within 0-21 days contributes up to 60 points
    - Recent news within recency window contributes up to 40 points
    """
    score = 0.0
    dte = days_until(next_earnings_iso)

    if dte is not None:
        if dte <= 0:
            score += 60.0
        elif dte <= 21:
            score += 60.0 * (1.0 - (dte / 21.0))

    recent = 0
    for it in news_items:
        ah = it.get("age_hours")
        if ah is not None and not pd.isna(ah) and ah <= news_recency_hours:
            recent += 1
    score += min(40.0, recent * 10.0)

    return float(min(100.0, max(0.0, score)))


# =========================
# Scoring
# =========================
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
        alloc = [monthly_budget] + [0] * (len(top) - 1)
    elif mode == "2 picks (70/30)":
        alloc = [0.70 * monthly_budget, 0.30 * monthly_budget] + [0] * (len(top) - 2)
    elif mode == "3 picks (60/25/15)":
        weights = [0.60, 0.25, 0.15]
        alloc = [weights[i] * monthly_budget if i < 3 else 0 for i in range(len(top))]
    else:
        scores = top["spec_score"].clip(lower=0)
        if scores.sum():
            alloc = (monthly_budget * scores / scores.sum()).tolist()
        else:
            alloc = [monthly_budget / len(top)] * len(top)

    out = top.copy()
    out["allocation_$"] = pd.Series(alloc, index=out.index).round(2)
    return out


# =========================
# LLM (OpenAI)
# =========================
def get_openai_client() -> Optional["OpenAI"]:
    """
    Creates an OpenAI client if possible.
    Reads API key from:
      - st.secrets["OPENAI_API_KEY"]
      - env var OPENAI_API_KEY
    """
    if not _OPENAI_AVAILABLE:
        return None

    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None

    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    return OpenAI(api_key=key)


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


def llm_thesis_and_invalidation(
    client: "OpenAI",
    ticker: str,
    metrics_text: str,
    model: str,
    temperature: float = 0.3
) -> dict:
    prompt = f"""
You are an investing assistant helping a user build a speculative ($100/month) high-risk growth watchlist.
Generate a concise, decision-useful thesis and explicit invalidation rules for the ticker.

Use ONLY the provided metrics as quantitative evidence. You may reference common qualitative drivers (product, competition, dilution, regulation)
but do not invent specific factual claims (e.g., exact revenue, customers, partnerships).
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

    text = getattr(resp, "output_text", "") or ""
    text = text.strip()

    # Attempt strict JSON parse; fallback to extracting JSON object
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# =========================
# Journal + performance
# =========================
def journal_path() -> str:
    return os.path.join(os.getcwd(), JOURNAL_FILE)


def load_journal() -> pd.DataFrame:
    path = journal_path()
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_to_journal(rows: List[dict]) -> None:
    path = journal_path()
    df_new = pd.DataFrame(rows)
    df_old = load_journal()
    df_all = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
    df_all.to_csv(path, index=False)


def compute_performance(journal: pd.DataFrame) -> pd.DataFrame:
    if journal.empty:
        return journal

    j = journal.copy()

    for c in ["date", "ticker", "buy_price", "allocation_$", "spec_score"]:
        if c not in j.columns:
            j[c] = np.nan

    j["current_price"] = j["ticker"].apply(fetch_last_close)
    j["return_%"] = (j["current_price"] / j["buy_price"] - 1.0) * 100.0

    def bench_return(row):
        d = str(row.get("date", ""))[:10]
        if not d or d == "nan":
            return float("nan")
        b0 = fetch_close_on_or_after(BENCHMARK_TICKER, d)
        b1 = fetch_last_close(BENCHMARK_TICKER)
        if pd.isna(b0) or pd.isna(b1) or b0 == 0:
            return float("nan")
        return (b1 / b0 - 1.0) * 100.0

    j["bench_return_%"] = j.apply(bench_return, axis=1)
    j["alpha_vs_bench_%"] = j["return_%"] - j["bench_return_%"]
    return j


# =========================
# Sidebar controls
# =========================
st.sidebar.title("⚙️ Controls")

universe_text = st.sidebar.text_area(
    "Ticker universe (comma/newline separated)",
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
st.sidebar.subheader("🗞️ Catalysts")
news_limit = st.sidebar.slider("News items per ticker", 3, 20, 10)
news_recency_hours = st.sidebar.slider("News recency window (hours)", 24, 240, 72, 24)
scan_days = st.sidebar.slider("Earnings scan window (days)", 7, 120, 45, 7)

st.sidebar.divider()
st.sidebar.subheader("🧭 Top N scan display")
sort_by = st.sidebar.selectbox(
    "Sort Top-N catalyst table by",
    ["Earnings soonest", "Catalyst score", "Spec score"],
    index=0
)
filter_to_earnings_window = st.sidebar.checkbox(
    "Only show tickers with earnings in the next X days",
    value=False
)
earnings_window_days = st.sidebar.slider(
    "X (days)",
    7, 120, int(scan_days), 7,
    disabled=not filter_to_earnings_window
)
scan_pause = st.sidebar.slider("Scan throttle (seconds per ticker)", 0.0, 1.0, 0.15, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🤖 LLM thesis (optional)")
llm_enabled = st.sidebar.checkbox("Enable LLM-assisted thesis", value=False)
llm_model = st.sidebar.text_input("Model name", value="gpt-4o-mini")
llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

cfg = ScoreConfig(min_price=min_price)

# =========================
# Main UI
# =========================
st.title("📈 Speculation Stock Agent")
st.caption("Recommendations + catalysts + LLM thesis + decision logging + performance tracking (vs QQQ). Educational use only.")

tab1, tab2 = st.tabs(["Recommendations", "Decision Journal"])

if refresh:
    fetch_history.clear()
    fetch_last_close.clear()
    fetch_close_on_or_after.clear()
    fetch_next_earnings_date.clear()
    fetch_news.clear()

with st.spinner("Fetching market data..."):
    history = fetch_history(tickers, period=period)

if not history:
    st.error("No data returned. Check tickers and try again.")
    st.stop()

scores = build_scores(history, cfg)
eligible = scores[scores["eligible"] == 1].copy()

if eligible.empty:
    st.warning("No eligible tickers after filters.")
    st.stop()

top = eligible.head(top_n)
rec = suggest_allocation(top, monthly_budget, alloc_mode)

if "thesis_cache" not in st.session_state:
    st.session_state["thesis_cache"] = {}

if "catalyst_table" not in st.session_state:
    st.session_state["catalyst_table"] = None


def build_topn_catalyst_table() -> pd.DataFrame:
    rows = []
    today = dt.date.today()
    to_day = today + dt.timedelta(days=int(scan_days))  # scan window
    cache_day = today.isoformat()

    # ✅ ONE bulk earnings call
    cal_df = finnhub_earnings_calendar_bulk(
        from_iso=today.isoformat(),
        to_iso=to_day.isoformat(),
        cache_day=cache_day
    )
    earnings_map = build_next_earnings_map(cal_df)  # {SYM: "YYYY-MM-DD"}

    for t in list(top.index):
        sym = t.upper().strip()

        # ✅ earnings lookup is now O(1), no API call
        e = earnings_map.get(sym, None)
        dte = days_until(e)

        # ✅ news is cached per ticker per day
        news = fetch_news(sym, limit=int(news_limit))

        recent_news = 0
        for it in news:
            ah = it.get("age_hours")
            if ah is not None and not pd.isna(ah) and ah <= news_recency_hours:
                recent_news += 1

        cat = catalyst_score(e, news, news_recency_hours=news_recency_hours)

        rows.append({
            "Ticker": sym,
            "Spec Score": round(float(top.loc[sym, "spec_score"]), 1),
            "Next earnings (est.)": e or "",
            "Days to earnings": dte if dte is not None else np.nan,
            f"Recent news (<= {news_recency_hours}h)": int(recent_news),
            "Total news (fetched)": int(len(news)),
            "Catalyst Score": round(cat, 1),
        })

        # Optional throttle still helps if you have lots of tickers, but now only news calls matter
        if scan_pause > 0:
            time.sleep(float(scan_pause))

    return pd.DataFrame(rows)


def sort_catalyst_table(tbl: pd.DataFrame) -> pd.DataFrame:
    if tbl is None or tbl.empty:
        return tbl

    if sort_by == "Earnings soonest":
        if "Days to earnings" in tbl.columns:
            return tbl.sort_values(
                ["Days to earnings", "Catalyst Score", "Spec Score"],
                ascending=[True, False, False],
                na_position="last"
            )
        return tbl.sort_values(["Catalyst Score", "Spec Score"], ascending=[False, False], na_position="last")

    if sort_by == "Catalyst score":
        return tbl.sort_values(
            ["Catalyst Score", "Days to earnings", "Spec Score"],
            ascending=[False, True, False],
            na_position="last"
        )

    # Spec score
    return tbl.sort_values(
        ["Spec Score", "Catalyst Score", "Days to earnings"],
        ascending=[False, False, True],
        na_position="last"
    )


def filter_catalyst_table(tbl: pd.DataFrame) -> pd.DataFrame:
    if tbl is None or tbl.empty:
        return tbl
    if not filter_to_earnings_window:
        return tbl

    if "Days to earnings" not in tbl.columns:
        return tbl

    # Strict filter: only tickers with known earnings date and within window
    view_tbl = tbl.copy()
    view_tbl = view_tbl[
        view_tbl["Days to earnings"].notna() &
        (view_tbl["Days to earnings"] >= 0) &
        (view_tbl["Days to earnings"] <= earnings_window_days)
    ]
    return view_tbl


with tab1:
    st.subheader("✅ This Month's Suggested Buys")
    rec_display = rec[["spec_score", "allocation_$", "price", "momentum_6m", "vol_ann", "vol_z", "rsi", "max_drawdown"]].copy()
    rec_display = rec_display.rename(columns={
        "spec_score": "Spec Score",
        "allocation_$": "Allocation ($)",
        "price": "Price",
        "momentum_6m": "6M Momentum",
        "vol_ann": "Vol (ann.)",
        "vol_z": "Volume Z",
        "rsi": "RSI",
        "max_drawdown": "Max DD (period)"
    })

    st.dataframe(
        rec_display.style.format({
            "Spec Score": "{:.1f}",
            "Allocation ($)": "${:.2f}",
            "Price": "${:.2f}",
            "6M Momentum": "{:.1%}",
            "Vol (ann.)": "{:.0%}",
            "Volume Z": "{:.2f}",
            "RSI": "{:.0f}",
            "Max DD (period)": "{:.0%}",
        }),
        use_container_width=True,
        height=360
    )

    st.divider()
    st.subheader("🧭 Upcoming catalysts across Top N")
    st.caption("Scans next earnings dates + counts recent news items for every ticker in Top N (one view).")

    colA, colB = st.columns([0.2, 0.8])
    with colA:
        scan_btn = st.button("Scan Top N catalysts")
    with colB:
        st.write(f"Sort: **{sort_by}**  •  News recency: **{news_recency_hours}h**  •  Earnings window: **{scan_days}d**")

    if scan_btn or st.session_state["catalyst_table"] is None:
        with st.spinner("Scanning earnings + news for Top N…"):
            st.session_state["catalyst_table"] = build_topn_catalyst_table()

    tbl = st.session_state["catalyst_table"]
    tbl_sorted = sort_catalyst_table(tbl) if isinstance(tbl, pd.DataFrame) else tbl
    tbl_view = filter_catalyst_table(tbl_sorted) if isinstance(tbl_sorted, pd.DataFrame) else tbl_sorted

    if isinstance(tbl_view, pd.DataFrame) and not tbl_view.empty:
        st.dataframe(tbl_view, use_container_width=True, height=320)
        csv_bytes = tbl_view.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top N catalyst scan CSV", data=csv_bytes, file_name="catalyst_scan_topN.csv", mime="text/csv")
    else:
        # Show a helpful message rather than a blank area
        if filter_to_earnings_window:
            st.warning("No tickers matched the earnings window filter (or earnings dates were unavailable from yfinance). Try increasing X days or disabling the filter.")
        else:
            st.info("Click **Scan Top N catalysts** to populate the table. If it stays empty, yfinance may be returning no earnings/news coverage for these tickers.")

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
        st.metric("6M Momentum", f"{row['momentum_6m'] * 100:.1f}%" if not pd.isna(row["momentum_6m"]) else "n/a")
        st.metric("Volatility (ann.)", f"{row['vol_ann'] * 100:.0f}%" if not pd.isna(row["vol_ann"]) else "n/a")
        st.metric("Volume z-score", f"{row['vol_z']:.2f}" if not pd.isna(row["vol_z"]) else "n/a")
        st.metric("RSI", f"{row['rsi']:.0f}" if not pd.isna(row["rsi"]) else "n/a")
        st.metric("Max drawdown", f"{row['max_drawdown'] * 100:.0f}%" if not pd.isna(row["max_drawdown"]) else "n/a")

    st.divider()
    st.subheader("🗓️ Catalysts (selected ticker): Earnings + News")

    next_earnings = fetch_next_earnings_date(ticker_sel)
    dte = days_until(next_earnings)
    news = fetch_news(ticker_sel, limit=int(news_limit))
    cat = catalyst_score(next_earnings, news, news_recency_hours=news_recency_hours)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Catalyst Score", f"{cat:.0f}/100")
    with c2:
        st.metric("Next earnings (est.)", next_earnings or "n/a")
    with c3:
        st.metric("Days to earnings", f"{dte}" if dte is not None else "n/a")

    st.caption("Earnings dates/news are best-effort via public feeds through yfinance; verify before trading.")

    if news:
        for it in news:
            title = it.get("title", "")
            pub = it.get("publisher", "")
            url = it.get("url", "")
            age = it.get("age_hours", np.nan)
            recent_badge = "🟢 Recent" if (age is not None and not pd.isna(age) and age <= news_recency_hours) else ""
            when = it.get("published_at", "")

            with st.container():
                cols = st.columns([0.82, 0.18])
                with cols[0]:
                    st.markdown(f"**{title}**  \n{pub} • {when}")
                    if url:
                        st.link_button("Open article", url)
                with cols[1]:
                    if recent_badge:
                        st.write(recent_badge)
    else:
        st.info("No news items found for this ticker via yfinance.")

    # Optional: debug payload visibility (helps when yfinance returns blanks on Cloud)
    with st.expander("🔧 Debug (yfinance raw coverage)"):
        st.write("Tip: If these are empty on Streamlit Cloud, yfinance may be blocked/rate-limited or missing coverage.")
        try:
            tk = yf.Ticker(ticker_sel)
            try:
                st.write("calendar:", tk.calendar)
            except Exception as e:
                st.write("calendar error:", e)
            try:
                st.write("news length:", len(getattr(tk, "news", []) or []))
            except Exception as e:
                st.write("news error:", e)
            if hasattr(tk, "get_news"):
                try:
                    gn = tk.get_news(count=5) or []
                    st.write("get_news length:", len(gn))
                except Exception as e:
                    st.write("get_news error:", e)
        except Exception as e:
            st.write("Ticker init error:", e)

    # LLM thesis section
    if llm_enabled:
        st.divider()
        st.subheader("🤖 Thesis + Invalidation Rules (LLM)")
        client = get_openai_client()
        if client is None:
            st.warning("OpenAI client not available. Add OPENAI_API_KEY to Streamlit Secrets or environment, and ensure openai is installed.")
        else:
            metrics_text = summarize_metrics_for_llm(ticker_sel, row)
            st.code(metrics_text, language="text")
            gen = st.button("Generate / Refresh thesis for selected ticker")
            cache_key = f"{ticker_sel}:{llm_model}:{llm_temp:.2f}"

            if gen or (cache_key not in st.session_state["thesis_cache"]):
                try:
                    with st.spinner("Calling OpenAI…"):
                        st.session_state["thesis_cache"][cache_key] = llm_thesis_and_invalidation(
                            client=client,
                            ticker=ticker_sel,
                            metrics_text=metrics_text,
                            model=llm_model,
                            temperature=float(llm_temp),
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
                    st.markdown("### Invalidation triggers")
                    for b in result.get("invalidation_triggers", []):
                        st.write(f"• {b}")
                    st.markdown("### Monitoring questions")
                    for b in result.get("monitoring_questions", []):
                        st.write(f"• {b}")

    st.divider()
    st.subheader("📝 Log this month's buys")
    decision_date = st.date_input("Decision date", value=dt.date.today())
    note = st.text_input("Note (optional)", value="")
    st.caption(f"Will log rows with Allocation > $0 to `{JOURNAL_FILE}`.")

    if st.button("Log allocation to journal"):
        rows_to_log = []
        client = get_openai_client()  # optional, only for caching thesis_json that may exist
        for t, r in rec.iterrows():
            alloc = float(r.get("allocation_$", 0) or 0)
            if alloc <= 0:
                continue

            thesis_obj = None
            ck = f"{t}:{llm_model}:{llm_temp:.2f}"
            if ck in st.session_state["thesis_cache"]:
                thesis_obj = st.session_state["thesis_cache"][ck]

            e = fetch_next_earnings_date(t) or ""
            n = fetch_news(t, limit=int(news_limit))

            rows_to_log.append({
                "date": decision_date.isoformat(),
                "ticker": t,
                "allocation_$": float(alloc),
                "buy_price": float(r.get("price", np.nan)),
                "spec_score": float(r.get("spec_score", np.nan)),
                "alloc_mode": alloc_mode,
                "universe_size": len(tickers),
                "note": note,
                "next_earnings_est": e,
                "news_count": len(n),
                "thesis_json": json.dumps(thesis_obj) if thesis_obj else "",
                "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
            })

        if not rows_to_log:
            st.warning("No rows to log (all allocations are $0).")
        else:
            append_to_journal(rows_to_log)
            st.success(f"Logged {len(rows_to_log)} rows to {JOURNAL_FILE}.")


with tab2:
    st.subheader("📒 Decision Journal + Performance")
    journal = load_journal()

    if journal.empty:
        st.info("No journal entries yet. Go to Recommendations and click **Log allocation to journal**.")
    else:
        st.write("**Raw journal entries**")
        st.dataframe(journal, use_container_width=True, height=240)

        st.divider()
        st.write(f"**Performance vs {BENCHMARK_TICKER}**")
        with st.spinner("Calculating performance (fetching latest prices)…"):
            perf = compute_performance(journal)

        cols = [
            "date", "ticker", "allocation_$", "buy_price", "current_price",
            "return_%", "bench_return_%", "alpha_vs_bench_%", "spec_score",
            "next_earnings_est", "news_count", "note"
        ]
        cols = [c for c in cols if c in perf.columns]
        view = perf[cols].copy()

        st.dataframe(
            view.style.format({
                "allocation_$": "${:.2f}",
                "buy_price": "${:.2f}",
                "current_price": "${:.2f}",
                "return_%": "{:.2f}%",
                "bench_return_%": "{:.2f}%",
                "alpha_vs_bench_%": "{:.2f}%",
                "spec_score": "{:.1f}",
            }),
            use_container_width=True,
            height=350
        )

        total_alloc = float(perf["allocation_$"].sum()) if "allocation_$" in perf.columns else 0.0
        weighted = (
            (perf["return_%"] * (perf["allocation_$"] / total_alloc)).sum()
            if total_alloc else perf["return_%"].mean()
        )
        st.metric("Allocation-weighted return (journal)", f"{weighted:.2f}%")

        st.caption("Streamlit Community Cloud file storage can be ephemeral. Use the download/upload below to back up and restore your journal.")
        csv_bytes = journal.to_csv(index=False).encode("utf-8")
        st.download_button("Download journal CSV", data=csv_bytes, file_name=JOURNAL_FILE, mime="text/csv")

        st.write("**Restore journal from CSV**")
        up = st.file_uploader("Upload a previously downloaded journal CSV", type=["csv"])
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                df_up.to_csv(journal_path(), index=False)
                st.success("Journal restored. Refresh the page to see updates.")
            except Exception as e:
                st.error(f"Upload failed: {e}")
