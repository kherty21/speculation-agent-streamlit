import os
import math
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Optional: LLM-assisted thesis generation (OpenAI)
try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# =========================
# App Config
# =========================
st.set_page_config(page_title="Speculation Stock Agent", page_icon="📈", layout="wide")

DEFAULT_UNIVERSE = [
    "PLTR",
    "SOFI",
    "RIVN",
    "ROKU",
    "U",
    "COIN",
    "NET",
    "SNOW",
    "ARM",
    "TSLA",
    "SHOP",
    "DDOG",
    "MDB",
    "CRSP",
    "BE",
    "QS",
    "DKNG",
    "ASTS",
    "IONQ",
    "RKLB",
]

JOURNAL_FILE = "decision_journal.csv"
BENCHMARK_TICKER = "QQQ"


# =========================
# Data Models
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
    if len(close) < period + 5:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.il
