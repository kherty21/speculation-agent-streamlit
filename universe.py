# universe.py
"""
Universe validation helpers:
- US stocks only (NYSE/NASDAQ/AMEX)
- Exclude ETFs/funds/trusts/ETNs (best-effort)
- Cached Finnhub profile2 calls (per ticker per day)
- More efficient: optional bulk lookup of the full US symbol list (cached) to pre-filter
"""

import os
import datetime as dt
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
import requests


# -------------------------
# Keys + HTTP
# -------------------------
def get_finnhub_key() -> Optional[str]:
    """
    Reads Finnhub key from Streamlit secrets or env var.
    """
    try:
        k = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        k = None
    return (k or os.getenv("FINNHUB_API_KEY") or "").strip() or None


def _session() -> requests.Session:
    """
    Reuse HTTP connections for speed (especially on cloud).
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "speculation-agent/1.0"})
    return s


# -------------------------
# Optional: bulk symbol list (fast local filter)
# -------------------------
@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def finnhub_symbols_us(cache_week: str) -> pd.DataFrame:
    """
    Downloads Finnhub US symbol list once per week.
    Used to pre-filter obvious non-US symbols and reduce profile2 calls.
    Requires FINNHUB_API_KEY.

    Returns a DataFrame with at least:
      - symbol
      - description
      - type
      - mic (may exist)
    """
    token = get_finnhub_key()
    if not token:
        return pd.DataFrame()

    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {"exchange": "US", "token": token}

    try:
        r = _session().get(url, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json() or []
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        # Standardize expected columns if present
        for col in ["symbol", "description", "type", "mic", "currency", "figi"]:
            if col not in df.columns:
                df[col] = ""

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _prefilter_using_symbol_list(tickers: List[str], sym_df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns (candidates, reason_map) after prefiltering with Finnhub symbol list.
    This does not guarantee "common stock", but it reduces calls.
    """
    reason: Dict[str, str] = {}
    if sym_df is None or sym_df.empty or "symbol" not in sym_df.columns:
        return tickers, reason

    symbols = set(sym_df["symbol"].dropna().astype(str).str.upper().str.strip().tolist())
    candidates: List[str] = []
    for t in tickers:
        if t in symbols:
            candidates.append(t)
        else:
            reason[t] = "Not in Finnhub US symbol list"
    return candidates, reason


# -------------------------
# Per-ticker profile lookup
# -------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def finnhub_company_profile(ticker: str, cache_day: str) -> dict:
    """
    Finnhub profile2 cached per ticker per day.
    cache_day is today's date string to force daily refresh.
    """
    token = get_finnhub_key()
    if not token:
        return {}

    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": ticker, "token": token}

    try:
        r = _session().get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json() or {}
        # Finnhub sometimes returns {} for unknowns; normalize to dict
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# -------------------------
# Validation rules
# -------------------------
_ETF_NAME_KEYWORDS = (
    " etf", " fund", " trust", " index", " etn", " ucits", " spdr", " ishares", " vanguard"
)

_ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}


def is_valid_us_common_stock(profile: dict) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Best-effort validation: US-listed on major exchanges + exclude ETF-like names.
    """
    if not profile:
        return False, "Missing profile (bad symbol or API limit)"

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

    # Heuristic ETF/fund detection via name keywords
    if name:
        padded = f" {name} "
        if any(k in padded for k in _ETF_NAME_KEYWORDS):
            return False, "Looks like ETF/fund/trust/ETN (name heuristic)"

    return True, "OK"


# -------------------------
# Public API
# -------------------------
def filter_universe_us_stocks(
    tickers: List[str],
    use_symbol_list_prefilter: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filters tickers to (US stocks only, no ETFs) and returns (valid_tickers, audit_df).

    Efficiency:
    - Optional weekly-cached Finnhub US symbol list prefilter to reduce profile2 calls
    - Daily-cached Finnhub profile2 per ticker
    """
    # Normalize / dedupe
    clean = []
    seen = set()
    for t in tickers:
        t = (t or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        clean.append(t)

    cache_day = dt.date.today().isoformat()

    # Prefilter using symbol list (reduces profile calls)
    prefilter_reason: Dict[str, str] = {}
    candidates = clean
    if use_symbol_list_prefilter:
        cache_week = dt.date.today().strftime("%G-W%V")  # ISO week key
        sym_df = finnhub_symbols_us(cache_week=cache_week)
        candidates, prefilter_reason = _prefilter_using_symbol_list(clean, sym_df)

    rows = []
    valid: List[str] = []

    for t in clean:
        if t not in candidates:
            rows.append({
                "Ticker": t,
                "Name": "",
                "Exchange": "",
                "Country": "",
                "Currency": "",
                "Valid US Stock": False,
                "Reason": prefilter_reason.get(t, "Filtered by precheck"),
            })
            continue

        profile = finnhub_company_profile(t, cache_day=cache_day)
        ok, reason = is_valid_us_common_stock(profile)

        rows.append({
            "Ticker": t,
            "Name": profile.get("name", "") if isinstance(profile, dict) else "",
            "Exchange": profile.get("exchange", "") if isinstance(profile, dict) else "",
            "Country": profile.get("country", "") if isinstance(profile, dict) else "",
            "Currency": profile.get("currency", "") if isinstance(profile, dict) else "",
            "Valid US Stock": bool(ok),
            "Reason": reason,
        })

        if ok:
            valid.append(t)

    audit_df = pd.DataFrame(rows)
    return valid, audit_df

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_dynamic_universe_from_finnhub(
    cache_day: str,
    exchanges: tuple = ("NYSE", "NASDAQ", "AMEX"),
    max_symbols: int = 400,
    include_types: tuple = ("COMMON STOCK", "EQUITY"),
) -> List[str]:
    """
    Returns a candidate ticker universe from Finnhub's US symbol list.
    """
    cache_week = dt.date.today().strftime("%G-W%V")
    df = finnhub_symbols_us(cache_week=cache_week)
    if df.empty:
        return []

    # Finnhub symbol list provides "type" but may not map perfectly
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # Keep only plausible equity/common stock-like entries
    df = df[df["type"].isin([t.upper() for t in include_types])]

    # Basic cleanup: remove weird symbols (optional)
    df = df[df["symbol"].str.match(r"^[A-Z\.\-]{1,6}$", na=False)]

    tickers = df["symbol"].dropna().unique().tolist()
    return tickers[:max_symbols]

