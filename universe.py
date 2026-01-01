# universe.py
"""
Universe utilities:
- Pull Finnhub US symbol list (cached weekly)
- Build a dynamic daily-cached universe (subset) from the US list
- Validate US common stocks (best-effort) using Finnhub profile2 (cached per ticker per day)
- Provide an audit table for UI

Key improvements:
- Normalize exchange strings like "NASDAQ NMS - GLOBAL MARKET" and "NEW YORK STOCK EXCHANGE, INC."
- Optional explicit block of NYSE ARCA (ETF-heavy)
- Clear Reason column for audit
- Efficient caching: symbol list weekly, universe daily, profiles daily per ticker
"""

import os
import datetime as dt
from typing import Optional, Tuple, List

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

        # ensure columns exist
        for col in ["symbol", "description", "type", "mic", "currency", "figi"]:
            if col not in df.columns:
                df[col] = ""

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        df["description"] = df["description"].astype(str).str.strip()
        df["currency"] = df["currency"].astype(str).str.upper().str.strip()
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
    """
    token = get_finnhub_key()
    if not token:
        return []

    cache_week = dt.date.today().strftime("%G-W%V")
    df = finnhub_symbols_us(cache_week=cache_week)
    if df.empty:
        return []

    df = df.copy()

    # keep equity-like types (best-effort)
    inc = {t.upper() for t in include_types}
    df = df[df["type"].isin(inc)]

    # keep sane-looking symbols
    df = df[df["symbol"].str.match(allowed_symbol_pattern, na=False)]

    # (optional) keep USD rows if provided
    if "currency" in df.columns:
        df = df[df["currency"].isin(["USD", ""])]

    tickers = df["symbol"].dropna().unique().tolist()
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


# -------------------------
# Validation helpers
# -------------------------
_ETF_NAME_KEYWORDS = (
    " etf", " fund", " trust", " index", " etn", " ucits",
    " spdr", " ishares", " vanguard"
)

def normalize_exchange(exchange_raw: str) -> str:
    """
    Normalize Finnhub exchange strings to one of: NASDAQ, NYSE, AMEX, NYSE ARCA, or raw.
    Examples:
      - "NASDAQ NMS - GLOBAL MARKET" -> "NASDAQ"
      - "NEW YORK STOCK EXCHANGE, INC." -> "NYSE"
    """
    ex = (exchange_raw or "").upper()
    if "NASDAQ" in ex:
        return "NASDAQ"
    if "NEW YORK STOCK EXCHANGE" in ex or "NYSE" in ex:
        if "ARCA" in ex:
            return "NYSE ARCA"
        return "NYSE"
    if "AMEX" in ex or "NYSE AMERICAN" in ex:
        return "AMEX"
    return ex.strip()


def is_valid_us_common_stock(profile: dict, block_arca: bool = True) -> Tuple[bool, str]:
    """
    US-listed stock, not an ETF/fund/trust (best-effort).
    Returns (is_valid, reason).
    """
    if not profile:
        return False, "Missing profile (bad symbol / API limit / no coverage)"

    exchange_raw = profile.get("exchange") or ""
    exchange_norm = normalize_exchange(exchange_raw)

    country = (profile.get("country") or "").upper().strip()
    currency = (profile.get("currency") or "").upper().strip()
    name = (profile.get("name") or "").lower().strip()

    if block_arca and exchange_norm == "NYSE ARCA":
        return False, f"NYSE ARCA blocked (ETF-heavy): {exchange_raw}"

    allowed = {"NYSE", "NASDAQ", "AMEX"}
    if exchange_norm not in allowed:
        return False, f"Exchange not allowed ({exchange_raw})"

    if country != "US":
        return False, f"Country not US ({country or 'blank'})"
    if currency != "USD":
        return False, f"Currency not USD ({currency or 'blank'})"

    if name:
        padded = f" {name} "
        if any(k in padded for k in _ETF_NAME_KEYWORDS):
            return False, "Looks like ETF/fund/trust/ETN (name heuristic)"

    return True, "OK"


def filter_universe_us_stocks(tickers: List[str]) -> Tuple[List[str], pd.DataFrame]:
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
        ok, reason = is_valid_us_common_stock(profile, block_arca=True)

        rows.append({
            "Ticker": t,
            "Name": profile.get("name", ""),
            "Exchange": profile.get("exchange", ""),
            "Exchange (normalized)": normalize_exchange(profile.get("exchange", "")),
            "Country": profile.get("country", ""),
            "Currency": profile.get("currency", ""),
            "Valid US Stock": bool(ok),
            "Reason": reason,
        })

        if ok:
            valid.append(t)

    audit_df = pd.DataFrame(rows)
    return valid, audit_df
