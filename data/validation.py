"""Candle validation and cleaning. Invalid data => no prediction."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.timeframes import tf_delta

REQUIRED = ["timestamp", "open", "high", "low", "close"]
OPTIONAL_DEFAULTS = {"tick_volume": 0.0, "spread": 0.0}


@dataclass
class ValidationReport:
    valid: bool = True
    reasons: List[str] = field(default_factory=list)
    n_input: int = 0
    n_output: int = 0
    n_duplicates: int = 0
    n_nan_dropped: int = 0
    n_unrealistic: int = 0
    n_missing_candles: int = 0
    n_abnormal_gaps: int = 0
    n_recent_abnormal_gaps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _is_weekend_gap(prev_ts: pd.Timestamp, ts: pd.Timestamp) -> bool:
    gap_h = (ts - prev_ts).total_seconds() / 3600
    return prev_ts.weekday() in (4, 5) and ts.weekday() in (6, 0) and gap_h <= 80


def validate_candles(
    df: pd.DataFrame,
    timeframe: str,
    min_bars: int = 600,
    max_gap_multiplier: float = 3.0,
    recent_window: int = 300,
    max_bar_return: float = 0.25,
) -> Tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(n_input=0 if df is None else len(df))
    if df is None or len(df) == 0:
        rep.valid = False
        rep.reasons.append("empty dataframe")
        return pd.DataFrame(columns=REQUIRED), rep

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        rep.valid = False
        rep.reasons.append(f"missing columns: {missing}")
        return df, rep

    out = df.copy()
    # timezone: standardise on UTC (naive timestamps are assumed UTC)
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["timestamp"] = ts
    out = out.dropna(subset=["timestamp"])

    for col in ["open", "high", "low", "close", "tick_volume", "spread"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = OPTIONAL_DEFAULTS[col]
    out = out.replace([np.inf, -np.inf], np.nan)

    before = len(out)
    out = out.drop_duplicates(subset="timestamp", keep="last")
    rep.n_duplicates = before - len(out)
    out = out.sort_values("timestamp")

    before = len(out)
    out = out.dropna(subset=["open", "high", "low", "close"])
    rep.n_nan_dropped = before - len(out)
    out["tick_volume"] = out["tick_volume"].fillna(0.0)
    out["spread"] = out["spread"].ffill().fillna(0.0)

    # unrealistic candles
    bad = (
        (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (out["high"] < out["low"])
        | (out["high"] < out[["open", "close"]].max(axis=1) - 1e-12)
        | (out["low"] > out[["open", "close"]].min(axis=1) + 1e-12)
        | (out["close"].pct_change().abs() > max_bar_return)
    )
    rep.n_unrealistic = int(bad.sum())
    out = out[~bad]
    out = out.reset_index(drop=True)

    # gaps / missing candles
    delta = tf_delta(timeframe)
    if len(out) > 1:
        diffs = out["timestamp"].diff()
        gap_idx = np.where(diffs > delta * 1.5)[0]
        n_missing = 0
        abnormal = []
        for i in gap_idx:
            prev_ts, cur_ts = out["timestamp"].iloc[i - 1], out["timestamp"].iloc[i]
            n_missing += int(round((cur_ts - prev_ts) / delta)) - 1
            if (cur_ts - prev_ts) > delta * max_gap_multiplier and not _is_weekend_gap(prev_ts, cur_ts):
                abnormal.append(i)
        rep.n_missing_candles = n_missing
        rep.n_abnormal_gaps = len(abnormal)
        cutoff = len(out) - recent_window
        rep.n_recent_abnormal_gaps = sum(1 for i in abnormal if i >= cutoff)

    rep.n_output = len(out)
    if rep.n_output < min_bars:
        rep.valid = False
        rep.reasons.append(f"only {rep.n_output} candles (< min_bars={min_bars})")
    if rep.n_recent_abnormal_gaps > 0:
        rep.valid = False
        rep.reasons.append(f"{rep.n_recent_abnormal_gaps} abnormal gap(s) in the last {recent_window} candles")
    if rep.n_input and rep.n_unrealistic / max(rep.n_input, 1) > 0.01:
        rep.valid = False
        rep.reasons.append(f"{rep.n_unrealistic} unrealistic candles (>1%)")
    return out, rep
