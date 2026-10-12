"""Training pipeline: dataset -> walk-forward evaluation -> final fit + calibration -> versioned save.

A new model only becomes active if its out-of-sample walk-forward quality beats the active one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from evaluation.backtest import walk_forward
from features.builder import FEATURE_COLUMNS
from models.calibration import Calibrator
from models.ml_models import HAS_XGB, MLModelSet
from training.dataset import build_dataset
from training.model_registry import ModelRegistry
from utils.logging_setup import get_logger

log = get_logger("train")


def selection_summary(wf: dict) -> Dict[str, Any]:
    m = wf.get("metrics", {})
    return {
        "brier": m.get("brier"), "brier_skill": m.get("brier_skill"), "ece": m.get("ece"),
        "roc_auc": m.get("roc_auc"), "accuracy": m.get("accuracy"),
        "directional_accuracy": wf.get("regression", {}).get("directional_accuracy"),
        "return_mae": wf.get("regression", {}).get("mae"),
        "stability_brier_std": wf.get("stability_brier_std"), "calibrated": wf.get("calibrated"),
        "n_oos": m.get("n"),
    }


def is_better(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> bool:
    """Model selection: Brier first, then calibration error; guard return error and stability.
    Accuracy alone is never used."""
    if old is None or old.get("brier") is None:
        return True
    if new.get("brier") is None:
        return False
    d = new["brier"] - old["brier"]
    if d > 1e-4:
        return False
    if abs(d) <= 1e-4 and (new.get("ece") or 1) > (old.get("ece") or 1):
        return False
    if new.get("return_mae") and old.get("return_mae") and new["return_mae"] > 1.2 * old["return_mae"]:
        return False
    if new.get("stability_brier_std") and old.get("stability_brier_std") and new["stability_brier_std"] > 2 * old["stability_brier_std"]:
        return False
    return True


def train_symbol(symbol: str, timeframe: str, candles: pd.DataFrame, cfg: dict,
                 registry: ModelRegistry, force_activate: bool = False) -> Dict[str, Any]:
    horizon = int(cfg["prediction"]["horizon"])
    thr = float(cfg["prediction"].get("direction_threshold", 0.0))
    mcfg = cfg.get("models", {})
    _, ds = build_dataset(candles, horizon, thr, int(cfg.get("regime", {}).get("donchian_window", 20)))
    if len(ds) < max(int(mcfg.get("min_train_rows", 400)) * 2, 600):
        log.warning("%s %s: only %d training rows - skipped", symbol, timeframe, len(ds))
        return {"symbol": symbol, "timeframe": timeframe, "trained": False, "reason": f"only {len(ds)} rows"}

    log.info("%s %s: walk-forward evaluation on %d rows", symbol, timeframe, len(ds))
    wf = walk_forward(ds, cfg, horizon)
    summary = selection_summary(wf) if wf.get("ok") else {}

    # final fit: older 80% fit, newest 20% (after purge gap) calibrates
    cut = int(len(ds) * 0.8)
    fit_df, cal_df = ds.iloc[:cut], ds.iloc[cut + horizon:]
    ms = MLModelSet(mcfg).fit(fit_df[FEATURE_COLUMNS], fit_df["direction"], fit_df["future_return"])
    calibrator = None
    method = mcfg.get("calibration_method", "isotonic")
    if len(cal_df) >= int(mcfg.get("min_calibration_rows", 150)) and cal_df["direction"].nunique() == 2:
        calibrator = Calibrator(method).fit(ms.predict_ensemble_raw(cal_df[FEATURE_COLUMNS]), cal_df["direction"])
    else:
        log.warning("%s %s: not enough data to calibrate -> probabilities stay 'raw'", symbol, timeframe)

    version = registry.next_version(symbol, timeframe)
    previous_v = registry.active_version(symbol, timeframe)
    previous = registry.meta(symbol, timeframe, previous_v)["summary"] if previous_v and registry.meta(symbol, timeframe, previous_v) else None
    better = is_better(summary, previous)
    activate = force_activate or better or previous is None

    meta = {
        "symbol": symbol, "timeframe": timeframe, "horizon": horizon, "n_rows": int(len(ds)),
        "trained_until": str(ds.index[-1]), "summary": summary, "calibrated": calibrator is not None,
        "calibration_method": method if calibrator else None, "models": list(ms.classifiers),
        "model_failures": ms.failed, "xgboost": HAS_XGB, "feature_importance": ms.feature_importance,
        "top_drivers": ms.top_drivers(), "buckets": wf.get("buckets"), "regression": wf.get("regression"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    registry.save(symbol, timeframe, version, {"model_set": ms, "calibrator": calibrator, "meta": meta}, meta)
    if activate:
        registry.activate(symbol, timeframe, version)
    log.info("%s %s: %s trained (brier=%s skill=%s) activated=%s", symbol, timeframe, version,
             summary.get("brier"), summary.get("brier_skill"), activate)
    return {"symbol": symbol, "timeframe": timeframe, "trained": True, "version": version, "activated": bool(activate),
            "previous_version": previous_v, "summary": summary, "meta": meta, "walk_forward": wf}


def load_or_train(symbol: str, timeframe: str, candles: pd.DataFrame, cfg: dict, registry: ModelRegistry) -> Optional[Dict[str, Any]]:
    """Return the active bundle; train a first model automatically when none exists."""
    bundle = registry.load_active(symbol, timeframe)
    if bundle is not None:
        return bundle
    rep = train_symbol(symbol, timeframe, candles, cfg, registry)
    return registry.load_active(symbol, timeframe) if rep.get("trained") else None
