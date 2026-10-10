"""Weighted ensemble. Weights come from config (`weights:`); a meta-model can replace `combine` later.

Directional sources (trend, momentum, similarity, ML, cross-market) are converted to P(up) and
averaged with renormalised weights. Volatility and anomaly are *non-directional*: their weights
shrink the final probability toward 0.5 when conditions are unstable.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from models.calibration import Calibrator


def model_agreement(probs: Dict[str, Optional[float]]) -> str:
    vals = [v for v in probs.values() if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return "N/A"
    spread = max(vals) - min(vals)
    same_side = all(v >= 0.5 for v in vals) or all(v <= 0.5 for v in vals)
    majority = max(sum(v > 0.5 for v in vals), sum(v < 0.5 for v in vals)) / len(vals)
    if same_side and spread <= 0.12:
        return "HIGH"
    if (same_side and spread <= 0.25) or (majority >= 0.75 and spread < 0.2):
        return "MEDIUM"
    return "LOW"


def combine(
    parts: dict,
    weights: dict,
    ensemble_calibrator: Optional[Calibrator] = None,
) -> dict:
    """parts keys: trend_bias, momentum_bias, cross_bias, similarity_up_rate, ml_prob_up,
    ml_expected_return, similarity_median_return, volatility_score, volatility_state, anomaly_score,
    model_probs (dict of per-model P(up))."""
    w = lambda k: float(weights.get(k, 0.0))

    src: Dict[str, Optional[float]] = {
        "trend": None if parts.get("trend_bias") is None else 0.5 + 0.12 * parts["trend_bias"],
        "momentum": None if parts.get("momentum_bias") is None else 0.5 + 0.12 * parts["momentum_bias"],
        "cross_market": None if parts.get("cross_bias") is None else 0.5 + 0.2 * parts["cross_bias"],
        "similarity": parts.get("similarity_up_rate"),
        "ml": parts.get("ml_prob_up"),
    }
    wt = {"trend": w("trend_weight"), "momentum": w("momentum_weight"), "cross_market": w("cross_market_weight"),
          "similarity": w("similarity_weight"), "ml": w("ml_weight")}
    avail = {k: v for k, v in src.items() if v is not None and wt[k] > 0}
    if not avail:
        return {"probability_up": None, "probability_down": None, "expected_return": None, "confidence": None,
                "probability_kind": "N/A", "sources": src, "model_agreement": "N/A"}
    tot = sum(wt[k] for k in avail)
    p_dir = sum(wt[k] * avail[k] for k in avail) / tot

    # non-directional shrinkage
    vol_ext = 0.0
    if parts.get("volatility_state") in ("HIGH", "EXPANSION") and parts.get("volatility_score") is not None:
        vol_ext = parts["volatility_score"] / 100.0
    anom = (parts.get("anomaly_score") or 0.0) / 100.0
    shrink = min(0.5, w("volatility_weight") * vol_ext + w("anomaly_weight") * anom)
    p = 0.5 + (p_dir - 0.5) * (1 - shrink)

    p_raw_ensemble = float(np.clip(p, 0.01, 0.99))
    kind = "raw model probability"
    if ensemble_calibrator is not None:
        p = float(ensemble_calibrator.transform([p])[0])
        kind = "calibrated"
    p = float(np.clip(p, 0.01, 0.99))

    probs = dict(parts.get("model_probs") or {})
    probs["similarity"] = parts.get("similarity_up_rate")
    agreement = model_agreement(probs)

    # expected return: blend ML regression and similarity median (weights renormalised)
    er_parts = [(parts.get("ml_expected_return"), w("ml_weight")), (parts.get("similarity_median_return"), w("similarity_weight"))]
    er_parts = [(v, x) for v, x in er_parts if v is not None and not np.isnan(v) and x > 0]
    exp_ret = float(sum(v * x for v, x in er_parts) / sum(x for _, x in er_parts)) if er_parts else None

    factor = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, "N/A": None}[agreement]
    confidence = None if factor is None else round(float(np.clip(abs(p - 0.5) / 0.25, 0, 1) * factor * 100), 1)

    return {
        "probability_up": round(p, 4),
        "probability_down": round(1 - p, 4),
        "expected_return": exp_ret,
        "confidence": confidence,
        "probability_kind": kind,
        "probability_raw": round(p_raw_ensemble, 4),
        "sources": {k: (None if v is None else round(float(v), 4)) for k, v in src.items()},
        "model_agreement": agreement,
        "shrinkage": round(shrink, 4),
    }
