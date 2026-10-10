"""Timeframe helpers."""
from __future__ import annotations

import pandas as pd

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def tf_minutes(tf: str) -> int:
    try:
        return TF_MINUTES[tf.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {tf}") from exc


def tf_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=tf_minutes(tf))
