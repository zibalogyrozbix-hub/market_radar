"""Engine A - Trend."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_series(feat: pd.DataFrame) -> pd.DataFrame:
    """Vectorised trend bias (-1..1) and strength (0..100) for every row."""
    c = feat["close"]
    align = (np.sign(c - feat["ema20"]) + np.sign(feat["ema20"] - feat["ema50"])
             + np.sign(feat["ema50"] - feat["ema100"]) + np.sign(feat["ema100"] - feat["ema200"])) / 4.0
    slope = np.tanh(feat["ema20_slope"] / (feat["atr_pct"] + 1e-9))
    bias = 0.35 * align + 0.25 * slope + 0.20 * feat["di_diff"] + 0.20 * feat["structure"]
    strength = (0.6 * (feat["adx"] / 50.0).clip(0, 1) + 0.4 * align.abs()) * 100
    return pd.DataFrame({"trend_bias": bias.clip(-1, 1), "trend_strength": strength.clip(0, 100)}, index=feat.index)


def analyze(feat: pd.DataFrame) -> dict:
    ts = trend_series(feat).iloc[-1]
    bias, strength = float(ts["trend_bias"]), float(ts["trend_strength"])
    if np.isnan(bias) or np.isnan(strength):
        return {"trend_score": None, "trend_direction": "NEUTRAL", "trend_strength": None, "trend_bias": 0.0}
    direction = "UP" if bias > 0.15 else "DOWN" if bias < -0.15 else "NEUTRAL"
    last = feat.iloc[-1]
    return {
        "trend_score": round(strength, 1),
        "trend_direction": direction,
        "trend_strength": round(strength, 1),
        "trend_bias": round(bias, 4),
        "adx": round(float(last["adx"]), 1),
    }
