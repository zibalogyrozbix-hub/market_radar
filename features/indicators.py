"""Causal technical indicators (each value at t uses data up to and including t only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / (dn + EPS)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


GAP_HOURS = 6  # a break longer than this between two candles = weekend / session gap


def true_range(df: pd.DataFrame, gap_hours: float = GAP_HOURS) -> pd.Series:
    """True range. The first candle after a weekend/session gap uses only high-low: the gap to the previous close is
    not volatility of that hour, and with an EMA of 14 it would otherwise inflate ATR (and every 'volatility
    expansion' signal) for the next ~14 candles after each weekly open."""
    pc = df["close"].shift(1)
    hl = df["high"] - df["low"]
    tr = pd.concat([hl, (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    if isinstance(df.index, pd.DatetimeIndex) and gap_hours and gap_hours > 0:
        gap = df.index.to_series().diff() > pd.Timedelta(hours=gap_hours)
        tr = tr.where(~gap.to_numpy(), hl)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr_s = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (tr_s + EPS)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (tr_s + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std()
    return (2 * k * s) / (m + EPS)


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    """z-score of x_t versus the *previous* `window` values (current value excluded)."""
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean().shift(1)
    sd = x.rolling(window, min_periods=max(20, window // 4)).std().shift(1)
    return (x - mu) / (sd + EPS)


def rolling_pctile(x: pd.Series, window: int = 500) -> pd.Series:
    return x.rolling(window, min_periods=max(50, window // 5)).rank(pct=True)
