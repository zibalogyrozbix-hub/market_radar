"""Market regime detector (rule-based; swap the class for an HMM/clustering model later)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engines.trend_engine import trend_series

REGIMES = ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY",
           "BREAKOUT", "REVERSAL", "COMPRESSION", "EXPANSION"]


class RegimeDetector:
    """Interface: `detect_series(feat) -> pd.Series[str]`. Subclass to plug in an HMM or k-means."""

    def detect_series(self, feat: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def detect(self, feat: pd.DataFrame) -> str:
        return str(self.detect_series(feat.tail(5)).iloc[-1])


class RuleBasedRegime(RegimeDetector):
    def detect_series(self, feat: pd.DataFrame) -> pd.Series:
        ts = trend_series(feat)
        bias, strength = ts["trend_bias"], ts["trend_strength"]
        pct, ratio, bbp = feat["vol_pctile"], feat["atr_ratio"], feat["bb_width_pctile"]
        breakout = ((feat["donch_up"] > 0) | (feat["donch_dn"] > 0)) & (ratio > 1.1)
        # reversal: short-term slope flips against the prevailing EMA50 side while RSI is stretched
        against = (np.sign(feat["ema20_slope"]) != np.sign(feat["ema50_dist"])) & (feat["ema50_dist"].abs() > feat["atr_pct"] * 1.5)
        stretched = (feat["rsi"] > 68) | (feat["rsi"] < 32)
        reversal = against & stretched & (feat["divergence"] != 0)

        out = pd.Series("RANGE", index=feat.index, dtype=object)
        out[(strength < 35) & (pct < 0.15)] = "LOW_VOLATILITY"
        out[(ratio < 0.8) & (bbp < 0.2)] = "COMPRESSION"
        out[(strength >= 40) & (bias > 0.25)] = "TREND_UP"
        out[(strength >= 40) & (bias < -0.25)] = "TREND_DOWN"
        out[pct > 0.85] = "HIGH_VOLATILITY"
        out[ratio > 1.3] = "EXPANSION"
        out[breakout] = "BREAKOUT"
        out[reversal] = "REVERSAL"
        out[feat[["vol_pctile", "atr_ratio", "adx"]].isna().any(axis=1)] = "UNKNOWN"
        return out


def analyze(feat: pd.DataFrame, detector: RegimeDetector | None = None) -> dict:
    det = detector or RuleBasedRegime()
    s = det.detect_series(feat.tail(5))
    cur, prev = str(s.iloc[-1]), str(s.iloc[-2]) if len(s) > 1 else "UNKNOWN"
    return {"regime": cur, "previous_regime": prev, "regime_changed": cur != prev and prev != "UNKNOWN"}
