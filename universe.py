# universe.py
"""
Universe utilities:
- Pull Finnhub US symbol list (cached weekly)
- Build a dynamic daily-cached universe (subset) from the US list
- Validate US common stock (best-effort) using Finnhub profile2 (cached per ticker per day)
- Provide an audit table for UI
"""

import os
import datetime as dt
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
import requests


# -------------------------
# Finnhub key
# -------------------------
def get_finnhub_key() -> Optional[str]:
    try:
        k = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        k = None
    k = (k or os.getenv("FINNHUB_API_KEY") or "").strip()
    return k or None


# -------------------------
# Shared HTTP session
# -------------------------
@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "speculation-agent/1.0"})
    return s


# -------------------------
# Finnhub US symbol list (weekly cached)
# -------------------------
@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def finnhub_symbols_us(cache_week: str) -> pd.DataFrame:
    """
    Finnhub US symbol list, cached weekly.
    Returns columns if present: symbol, description, type, mic, currency, figi
    """
    token = get_finnhub_key()
    if not token:
        return pd.DataFrame()

    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {"exchange": "US", "token": token}

    try:
        r = get_http_session().get(url, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json() or []
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        for col in ["symbol", "description", "type", "mic", "currency", "figi"]:
            if col not in df.columns:
                df[col] = ""

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        df["description"] = df["description"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


# -------------------------
# Daily-cached dynamic universe builder
# -------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_dynamic_universe_daily(
    cache_day: str,
    max_symbols: int = 300,
    allowed_symbol_pattern: str = r"^[A-Z\.\-]{1,6}$",
    include_types: Tuple[str, ...] = ("COMMON STOCK", "EQUITY"),
) -> List[str]:
    """
    Builds a dynamic universe from Finnhub's US symbol list and caches it daily.
    This is a *candidate* list; you'll still validate tickers via profile2 in filter_universe_us_stocks.

    include_types:
      Finnhub 'type' is not perfectly standardized; include a few common equity-like strings.

    Note: We do not do expensive per-symbol calls here. This is intentionally cheap.
    """
    token = get_finnhub_key()
    if not token:
        return []

    cache_week = dt.date.today().strftime("%G-W%V")
    df = finnhub_symbols_us(cache_week=cache_week)
    if df.empty:
        return []

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["type"] = df["type"].astype(str).str.upper().str.strip()

    # keep equity-like types (best-effort)
    inc = set([t.upper() for t in include_types])
    df = df[df["type"].isin(inc)]

    # remove weird symbols
    df = df[df["symbol"].str.match(allowed_symbol_pattern, na=False)]

    # de-dup, stable-ish ordering
    tickers = df["symbol"].dropna().unique().tolist()

    # cap
    return tickers[: int(max_symbols)]


# -------------------------
# Profile2 (daily cached per ticker)
# -------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def finnhub_company_profile(ticker: str, cache_day: str) -> dict:
    """
    Finnhub profile2 cached per ticker per day.
    """
    token = get_finnhub_key()
    if not token:
        return {}

    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": ticker, "token": token}

    try:
        r = get_http_session().get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json() or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}
_ETF_NAME_KEYWORDS = (
    " etf", " fund", " trust", " index", " etn", " ucits",
    " spdr", " ishares", " vanguard"
)


def is_valid_us_common_stock(profile: dict) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason).
    """
    if not profile:
        return False, "Missing profile (bad symbol / API limit / no coverage)"

    exchange = (profile.get("exchange") or "").upper().strip()
    country = (profile.get("country") or "").upper().strip()
    currency = (profile.get("currency") or "").upper().strip()
    name = (profile.get("name") or "").lower().strip()

    if exchange not in _ALLOWED_EXCHANGES:
        return False, f"Exchange not allowed ({exchange or 'blank'})"
    if country != "US":
        return False, f"Country not US ({country or 'blank'})"
    if currency != "USD":
        return False, f"Currency not USD ({currency or 'blank'})"

    if name:
        padded = f" {name} "
        if any(k in padded for k in _ETF_NAME_KEYWORDS):
            return False, "Looks like ETF/fund/trust/ETN (name heuristic)"

    return True, "OK"


def filter_universe_us_stocks(
    tickers: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns (valid_tickers, audit_df).
    Uses Finnhub profile2 per ticker (cached daily).
    """
    cache_day = dt.date.today().isoformat()

    # normalize & dedupe
    clean: List[str] = []
    seen = set()
    for t in tickers:
        t = (t or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        clean.append(t)

    rows = []
    valid = []

    for t in clean:
        profile = finnhub_company_profile(t, cache_day=cache_day)
        ok, reason = is_valid_us_common_stock(profile)

        rows.append({
            "Ticker": t,
            "Name": profile.get("name", ""),
            "Exchange": profile.get("exchange", ""),
            "Country": profile.get("country", ""),
            "Currency": profile.get("currency", ""),
            "Valid US Stock": bool(ok),
            "Reason": reason,
        })

        if ok:
            valid.append(t)

    audit_df = pd.DataFrame(rows)
    return valid, audit_df
