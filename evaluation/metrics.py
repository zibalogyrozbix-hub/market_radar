"""Evaluation metrics for classification, calibration and expected-return error."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn import metrics as skm

from models.calibration import brier_score, calibration_curve, expected_calibration_error


def classification_metrics(y, p, threshold: float = 0.5) -> Dict[str, float | None]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return {}
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    base = float(y.mean())
    brier = brier_score(y, p)
    brier_base = brier_score(y, np.full_like(y, base))
    return {
        "n": int(len(y)),
        "accuracy": float(skm.accuracy_score(y, pred)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y, p)) if both else None,
        "brier": brier,
        "brier_baseline": brier_base,
        "brier_skill": float(1 - brier / brier_base) if brier_base > 0 else None,
        "ece": expected_calibration_error(y, p),
        "avg_predicted_probability": float(p.mean()),
        "actual_hit_rate": base,
    }


def regression_metrics(true_ret, pred_ret) -> Dict[str, float]:
    t, p = np.asarray(true_ret, dtype=float), np.asarray(pred_ret, dtype=float)
    m = ~(np.isnan(t) | np.isnan(p))
    t, p = t[m], p[m]
    if len(t) == 0:
        return {}
    return {
        "mae": float(np.mean(np.abs(t - p))),
        "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
        "directional_accuracy": float(np.mean(np.sign(t) == np.sign(p))),
    }


def bucket_table(y, p, edges: List[float] | None = None) -> List[dict]:
    """e.g. predictions 70-75% -> actual accuracy of the predicted side."""
    edges = edges or [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0001]
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        out.append({"bucket": f"{lo * 100:.0f}-{min(hi, 1.0) * 100:.0f}%", "n": int(m.sum()),
                    "mean_predicted_up": float(p[m].mean()), "actual_up_rate": float(y[m].mean())})
    return out


__all__ = ["classification_metrics", "regression_metrics", "bucket_table", "calibration_curve"]
