"""Walk-forward backtest of the ML layer (no look-ahead, scalers/models/calibrators fit on past only)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from evaluation.metrics import bucket_table, classification_metrics, regression_metrics
from features.builder import FEATURE_COLUMNS
from models.calibration import Calibrator
from models.ml_models import MLModelSet
from training.validation import assert_chronological, walk_forward_splits
from utils.logging_setup import get_logger

log = get_logger("backtest")


def walk_forward(ds: pd.DataFrame, cfg: dict, horizon: int, n_splits: int = 5, min_train: Optional[int] = None) -> Dict:
    """ds: dataset from `build_dataset`. Returns OOS predictions and metrics.

    In each fold: fit on past rows; calibrate on the LAST 20% of the training window (a model fit on
    the first 80%); predict the unseen future block.
    """
    mcfg = cfg.get("models", {})
    min_train = min_train or max(int(mcfg.get("min_train_rows", 400)), 800)
    method = mcfg.get("calibration_method", "isotonic")
    rows, fold_metrics = [], []
    for k, (tr, te) in enumerate(walk_forward_splits(len(ds), n_splits, min_train, gap=horizon)):
        assert_chronological(tr, te)
        cut = int(len(tr) * 0.8)
        fit_idx, cal_idx = tr[:cut], tr[cut + horizon:]
        a, b = ds.iloc[fit_idx], ds.iloc[cal_idx]
        t = ds.iloc[te]
        try:
            ms = MLModelSet(mcfg).fit(a[FEATURE_COLUMNS], a["direction"], a["future_return"], with_quantiles=False)
            p_cal_in = ms.predict_ensemble_raw(b[FEATURE_COLUMNS])
            calib: Optional[Calibrator] = None
            if len(b) >= int(mcfg.get("min_calibration_rows", 150)) and b["direction"].nunique() == 2:
                calib = Calibrator(method).fit(p_cal_in, b["direction"])
            p_raw = ms.predict_ensemble_raw(t[FEATURE_COLUMNS])
            p_cal = calib.transform(p_raw) if calib else np.full_like(p_raw, np.nan)
            pr = ms.predict_return(t[FEATURE_COLUMNS])
        except Exception as exc:
            log.warning("fold %d failed: %s", k, exc)
            continue
        fold = pd.DataFrame({"p_raw": p_raw, "p_cal": p_cal, "pred_return": pr, "y": t["direction"].to_numpy(),
                             "future_return": t["future_return"].to_numpy(), "fold": k}, index=t.index)
        rows.append(fold)
        fm = classification_metrics(fold["y"], fold["p_cal"] if calib else fold["p_raw"])
        fm["fold"] = k
        fold_metrics.append(fm)
    if not rows:
        return {"ok": False, "reason": "not enough data for walk-forward", "oos": pd.DataFrame()}
    oos = pd.concat(rows)
    has_cal = oos["p_cal"].notna().all()
    p_used = oos["p_cal"] if has_cal else oos["p_raw"]
    briers = [f["brier"] for f in fold_metrics if f.get("brier") is not None]
    return {
        "ok": True,
        "oos": oos,
        "calibrated": bool(has_cal),
        "metrics_raw": classification_metrics(oos["y"], oos["p_raw"]),
        "metrics": classification_metrics(oos["y"], p_used),
        "regression": regression_metrics(oos["future_return"], oos["pred_return"]),
        "buckets": bucket_table(oos["y"], p_used),
        "fold_metrics": fold_metrics,
        "stability_brier_std": float(np.std(briers)) if briers else None,
    }
