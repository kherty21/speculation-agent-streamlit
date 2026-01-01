# universe.py
import os
import datetime as dt
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import requests


def get_finnhub_key() -> Optional[str]:
    try:
        k = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        k = None
    return k or os.getenv("FINNHUB_API_KEY")


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
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def is_valid_us_common_stock(profile: dict) -> bool:
    """
    US-listed stock, not an ETF/fund/trust (best-effort).
    """
    if not profile:
        return False

    exchange = (profile.get("exchange") or "").upper().strip()
    country = (profile.get("country") or "").upper().strip()
    currency = (profile.get("currency") or "").upper().strip()
    name = (profile.get("name") or "").lower().strip()
    ipo = profile.get("ipo", "")

    allowed_exchanges = {"NYSE", "NASDAQ", "AMEX"}
    if exchange not in allowed_exchanges:
        return False
    if country != "US" or currency != "USD":
        return False

    # Heuristic ETF/fund detection
    etf_keywords = [" etf", " fund", " trust", " index", " etn"]
    if any(k in f" {name}" for k in etf_keywords):
        return False

    # Some ETFs include "Trust" or "Fund" but the above catches most
    # If needed, you can add a hard blocklist too.

    # IPO missing sometimes; don’t fail on it.
    _ = ipo
    return True


def filter_universe_us_stocks(tickers: List[str]) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns (valid_tickers, audit_df)
    """
    cache_day = dt.date.today().isoformat()
    rows = []
    valid = []

    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue

        profile = finnhub_company_profile(t, cache_day=cache_day)
        ok = is_valid_us_common_stock(profile)

        rows.append({
            "Ticker": t,
            "Name": profile.get("name", ""),
            "Exchange": profile.get("exchange", ""),
            "Country": profile.get("country", ""),
            "Currency": profile.get("currency", ""),
            "Valid US Stock": ok,
        })

        if ok:
            valid.append(t)

    audit_df = pd.DataFrame(rows)
    return valid, audit_df
