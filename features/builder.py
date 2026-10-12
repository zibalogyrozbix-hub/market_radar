"""Feature engine: candles -> feature frame. Also builds (separate, future-looking) targets.

Features use only past/current data. Targets use the future and must NEVER be fed to models as
inputs; they are produced by `make_targets` and only used for training/evaluation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from features import indicators as ind

# Columns fed to ML models and similarity search
FEATURE_COLUMNS = [
    "ret_1", "ret_4", "ret_12", "ret_24",
    "realized_vol", "atr_pct", "vol_pctile", "atr_ratio", "bb_width", "bb_width_pctile",
    "rsi", "macd_hist_atr", "roc_10", "velocity", "ema_accel", "divergence",
    "ema20_dist", "ema50_dist", "ema100_dist", "ema200_dist", "ema20_slope",
    "adx", "di_diff", "structure",
    "tick_vol_z", "spread_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

PRETTY = {
    "ret_1": "1-bar return", "ret_4": "4-bar return", "ret_12": "12-bar return", "ret_24": "24-bar return",
    "realized_vol": "Realized volatility", "atr_pct": "ATR %", "vol_pctile": "Volatility percentile",
    "atr_ratio": "ATR expansion", "bb_width": "Bollinger width", "bb_width_pctile": "Bollinger width percentile",
    "rsi": "RSI", "macd_hist_atr": "MACD histogram", "roc_10": "Rate of change", "velocity": "Price velocity",
    "ema_accel": "EMA acceleration", "divergence": "Momentum divergence",
    "ema20_dist": "Distance from EMA20", "ema50_dist": "Distance from EMA50",
    "ema100_dist": "Distance from EMA100", "ema200_dist": "Distance from EMA200",
    "ema20_slope": "EMA20 slope", "adx": "ADX", "di_diff": "DI+/DI- balance", "structure": "Market structure",
    "tick_vol_z": "Tick-volume deviation", "spread_z": "Spread deviation",
    "hour_sin": "Time of day", "hour_cos": "Time of day", "dow_sin": "Day of week", "dow_cos": "Day of week",
}

Z_COLUMNS = ["z_ret", "z_vol", "z_tickvol", "z_spread", "z_velocity"]


def build_features(candles: pd.DataFrame, z_window: int = 100, donchian_window: int = 20) -> pd.DataFrame:
    """Return frame indexed by timestamp with base columns, helper columns and FEATURE_COLUMNS."""
    df = candles.copy()
    df = df.set_index("timestamp") if "timestamp" in df.columns else df
    c, h, l = df["close"], df["high"], df["low"]
    out = df[["open", "high", "low", "close", "tick_volume", "spread"]].copy()

    ret1 = np.log(c).diff()
    out["ret_1"] = ret1 * 100
    out["ret_4"] = np.log(c).diff(4) * 100
    out["ret_12"] = np.log(c).diff(12) * 100
    out["ret_24"] = np.log(c).diff(24) * 100

    out["atr"] = ind.atr(df, 14)
    out["atr_pct"] = out["atr"] / c * 100
    out["realized_vol"] = ret1.rolling(20, min_periods=20).std() * 100
    out["vol_pctile"] = ind.rolling_pctile(out["atr_pct"], 500)
    out["atr_ratio"] = out["atr"] / (out["atr"].rolling(50, min_periods=50).mean() + ind.EPS)
    out["bb_width"] = ind.bollinger_width(c, 20) * 100
    out["bb_width_pctile"] = ind.rolling_pctile(out["bb_width"], 500)

    out["rsi"] = ind.rsi(c, 14)
    _, _, hist = ind.macd(c)
    out["macd_hist_atr"] = hist / (out["atr"] + ind.EPS)
    out["roc_10"] = c.pct_change(10) * 100
    out["velocity"] = ret1.rolling(3, min_periods=3).mean() * 100 / (out["atr_pct"] + ind.EPS)

    for n in (20, 50, 100, 200):
        e = ind.ema(c, n)
        out[f"ema{n}"] = e
        out[f"ema{n}_dist"] = (c - e) / e * 100
    out["sma50"] = ind.sma(c, 50)
    out["ema20_slope"] = out["ema20"].pct_change(5) * 100
    out["ema_accel"] = out["ema20_slope"].diff(3) / (out["atr_pct"] + ind.EPS)

    adx_v, pdi, mdi = ind.adx(df, 14)
    out["adx"], out["plus_di"], out["minus_di"] = adx_v, pdi, mdi
    out["di_diff"] = (pdi - mdi) / (pdi + mdi + ind.EPS)

    hi20, lo20 = h.rolling(20, min_periods=20).max(), l.rolling(20, min_periods=20).min()
    hh = np.sign(hi20 - hi20.shift(20))  # +1 higher high, -1 lower high
    hl = np.sign(lo20 - lo20.shift(20))  # +1 higher low, -1 lower low
    out["structure"] = (hh + hl) / 2.0
    dw = max(5, int(donchian_window))  # regime.donchian_window in config.yaml
    out["donch_up"] = (c > h.rolling(dw, min_periods=dw).max().shift(1)).astype(float)
    out["donch_dn"] = (c < l.rolling(dw, min_periods=dw).min().shift(1)).astype(float)

    # RSI divergence: price higher-high vs RSI lower; price lower-low vs RSI higher (20-bar compare)
    px_hi = (c >= c.rolling(20, min_periods=20).max()) & (out["rsi"] < out["rsi"].shift(20) - 3)
    px_lo = (c <= c.rolling(20, min_periods=20).min()) & (out["rsi"] > out["rsi"].shift(20) + 3)
    out["divergence"] = px_lo.astype(float) - px_hi.astype(float)  # +1 bullish, -1 bearish

    tv = np.log1p(out["tick_volume"])
    out["tick_vol_z"] = ind.rolling_z(tv, z_window)
    out["spread_z"] = ind.rolling_z(out["spread"].astype(float), z_window)

    idx = out.index
    hour = idx.hour + idx.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # anomaly z-scores (vs previous window, current excluded)
    out["z_ret"] = ind.rolling_z(out["ret_1"], z_window)
    out["z_vol"] = ind.rolling_z(out["atr_pct"], z_window)
    out["z_tickvol"] = out["tick_vol_z"]
    out["z_spread"] = out["spread_z"]
    out["z_velocity"] = ind.rolling_z(out["velocity"], z_window)

    return out.replace([np.inf, -np.inf], np.nan)


def make_targets(close: pd.Series, horizon: int, threshold: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """Future-looking targets (never use as features).

    future_return: close[t+h]/close[t] - 1 ; direction: 1 if future_return > threshold else 0.
    Last `horizon` rows are NaN (unknown at those timestamps).
    """
    fr = close.shift(-horizon) / close - 1.0
    direction = (fr > threshold).astype(float).where(fr.notna())
    return direction.rename("direction"), fr.rename("future_return")


def model_frame(feat: pd.DataFrame) -> pd.DataFrame:
    return feat[FEATURE_COLUMNS]
