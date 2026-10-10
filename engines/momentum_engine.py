"""Engine B - Momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_series(feat: pd.DataFrame) -> pd.Series:
    rsi_n = (feat["rsi"] - 50.0) / 50.0
    macd_n = np.tanh(feat["macd_hist_atr"] * 2.0)
    roc_n = np.tanh(feat["roc_10"] / (feat["atr_pct"] * 3.0 + 1e-9))
    vel_n = np.tanh(feat["velocity"])
    acc_n = np.tanh(feat["ema_accel"])
    bias = 0.25 * rsi_n + 0.25 * macd_n + 0.20 * roc_n + 0.15 * vel_n + 0.10 * acc_n + 0.05 * feat["divergence"]
    return bias.clip(-1, 1).rename("momentum_bias")


def analyze(feat: pd.DataFrame) -> dict:
    bias = float(momentum_series(feat).iloc[-1])
    if np.isnan(bias):
        return {"momentum_score": None, "momentum_direction": "NEUTRAL", "momentum_bias": 0.0}
    score = min(100.0, abs(bias) * 130.0)
    direction = "UP" if bias > 0.1 else "DOWN" if bias < -0.1 else "NEUTRAL"
    last = feat.iloc[-1]
    return {
        "momentum_score": round(score, 1),
        "momentum_direction": direction,
        "momentum_bias": round(bias, 4),
        "rsi": round(float(last["rsi"]), 1),
        "divergence": int(last["divergence"]) if not np.isnan(last["divergence"]) else 0,
    }
