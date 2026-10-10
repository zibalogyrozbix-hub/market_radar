"""Engine C - Volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def state_from_row(vol_pctile: float, atr_ratio: float, bbw_pctile: float) -> str:
    if atr_ratio > 1.3:
        return "EXPANSION"
    if atr_ratio < 0.8 and bbw_pctile < 0.2:
        return "COMPRESSION"
    if vol_pctile > 0.85:
        return "HIGH"
    if vol_pctile < 0.15:
        return "LOW"
    return "NORMAL"


def analyze(feat: pd.DataFrame) -> dict:
    last = feat.iloc[-1]
    pct, ratio, bbp = last["vol_pctile"], last["atr_ratio"], last["bb_width_pctile"]
    if any(np.isnan(v) for v in (pct, ratio, bbp)):
        return {"volatility_score": None, "volatility_state": "NORMAL"}
    return {
        "volatility_score": round(float(pct) * 100, 1),
        "volatility_state": state_from_row(float(pct), float(ratio), float(bbp)),
        "atr_pct": round(float(last["atr_pct"]), 4),
        "realized_vol": round(float(last["realized_vol"]), 4),
        "atr_ratio": round(float(ratio), 3),
        "bb_width": round(float(last["bb_width"]), 4),
    }
