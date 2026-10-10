"""Prediction tracking: resolve outcomes, build performance data, fit ensemble calibrators."""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from database import repository as repo
from database.models import Candle
from evaluation.metrics import bucket_table, classification_metrics, regression_metrics
from models.calibration import Calibrator
from utils.logging_setup import get_logger
from utils.timeframes import tf_delta

log = get_logger("tracker")


def resolve_predictions(s) -> int:
    """For every unresolved prediction whose horizon has elapsed and whose target candle is stored,
    record the actual outcome (in prediction_results; the prediction itself is never modified)."""
    n = 0
    for pred in repo.unresolved_predictions(s):
        target = repo.aware(pred.timestamp) + tf_delta(pred.timeframe) * pred.horizon
        close = s.scalar(select(Candle.close).where(Candle.symbol == pred.symbol, Candle.timeframe == pred.timeframe,
                                                    Candle.timestamp == target))
        if close is None:
            continue
        repo.resolve_prediction(s, pred, float(close))
        n += 1
    if n:
        log.info("Resolved %d prediction(s)", n)
    return n


def _group(df: pd.DataFrame, col: str, p_col: str = "probability_up") -> Dict[str, Any]:
    out = {}
    for key, g in df.groupby(col):
        out[str(key)] = classification_metrics(g["y"], g[p_col])
    return out


def performance_report(s, last_n: Optional[int] = 1000, persist: bool = True) -> Dict[str, Any]:
    df = repo.resolved_predictions_df(s, limit=last_n)
    df = df.dropna(subset=["probability_up"]) if len(df) else df
    if df.empty:
        return {"n": 0, "message": "no resolved predictions yet"}
    p = df["probability_up"]
    report = {
        "n": int(len(df)),
        "overall": classification_metrics(df["y"], p),
        "regression": regression_metrics(df["actual_return"], df["expected_return"]),
        "buckets": bucket_table(df["y"], p),
        "by_symbol": _group(df, "symbol"),
        "by_timeframe": _group(df, "timeframe"),
        "by_regime": _group(df, "regime"),
        "with_clear_edge": classification_metrics(df.loc[df["edge_label"] == "CLEAR EDGE", "y"],
                                                  df.loc[df["edge_label"] == "CLEAR EDGE", "probability_up"])
        if (df["edge_label"] == "CLEAR EDGE").any() else {},
    }
    pred_up = p >= 0.5
    report["false_positives"] = int((pred_up & (df["y"] == 0)).sum())
    report["false_negatives"] = int((~pred_up & (df["y"] == 1)).sum())
    if persist:
        repo.save_metric(s, "overall", "live_performance", {k: v for k, v in report.items() if k != "buckets"})
        repo.save_metric(s, "overall", "probability_buckets", {"buckets": report["buckets"]})
        for scope in ("by_symbol", "by_timeframe", "by_regime"):
            for k, m in report[scope].items():
                repo.save_metric(s, f"{scope[3:]}:{k}", "live_performance", m)
    return report


def fit_ensemble_calibrator(s, symbol: str, model_version: Optional[str], cfg: dict) -> Optional[Calibrator]:
    """Calibrate the *ensemble* probability using this system's own resolved predictions
    (same model version only). Returns None until enough history exists."""
    df = repo.resolved_predictions_df(s, symbol=symbol)
    if df.empty:
        return None
    df = df[(df["model_version"] == model_version) & df["probability_raw"].notna()]
    need = int(cfg.get("models", {}).get("min_calibration_rows", 150))
    if len(df) < need or df["y"].nunique() < 2:
        return None
    try:
        return Calibrator(cfg.get("models", {}).get("calibration_method", "isotonic")).fit(df["probability_raw"], df["y"])
    except Exception as exc:
        log.warning("ensemble calibration failed for %s: %s", symbol, exc)
        return None
