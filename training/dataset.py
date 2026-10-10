"""Training dataset construction (features at t -> targets from t+1..t+horizon)."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from features.builder import FEATURE_COLUMNS, build_features, make_targets


def build_dataset(candles: pd.DataFrame, horizon: int, threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (feature_frame, dataset). `dataset` has FEATURE_COLUMNS + direction + future_return,
    with rows whose targets are not yet known (last `horizon` bars) or whose features are NaN removed.
    """
    feat = build_features(candles)
    direction, fut = make_targets(feat["close"], horizon, threshold)
    ds = feat[FEATURE_COLUMNS].join(direction).join(fut).dropna()
    return feat, ds
