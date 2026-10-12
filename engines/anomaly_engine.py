"""Engine D - Anomaly detection (z-scores + optional IsolationForest)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features.builder import FEATURE_COLUMNS
from utils.logging_setup import get_logger

log = get_logger("anomaly")

# z column -> (anomaly type, label)
Z_MAP = {
    "z_ret": "RETURN_ANOMALY",
    "z_vol": "VOLATILITY_SPIKE",
    "z_tickvol": "VOLUME_ANOMALY",
    "z_spread": "SPREAD_ANOMALY",
    "z_velocity": "VELOCITY_ANOMALY",
}


def z_to_score(z: float, lo: float = 1.0, hi: float = 4.0) -> float:
    """Map |z| to 0..100: <= lo sigma => 0, >= hi sigma => 100."""
    return float(np.clip((abs(z) - lo) / (hi - lo), 0, 1) * 100)


def _iforest_score(feat: pd.DataFrame, max_train: int = 2000, seed: int = 0) -> Optional[float]:
    """0..100 anomaly share from IsolationForest fit ONLY on rows before the current one."""
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
    except Exception:  # pragma: no cover
        return None
    X = feat[FEATURE_COLUMNS].dropna()
    if len(X) < 300:
        return None
    train, query = X.iloc[:-1].tail(max_train), X.iloc[[-1]]
    if X.index[-1] != feat.index[-1]:
        return None  # latest row has NaN features
    sc = RobustScaler().fit(train)
    Xt = np.clip(sc.transform(train), -10, 10)
    Xq = np.clip(sc.transform(query), -10, 10)
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=1).fit(Xt)
    train_scores = iso.score_samples(Xt)
    q = float(iso.score_samples(Xq)[0])
    share_lower = float((train_scores <= q).mean())
    return float(np.clip((0.10 - share_lower) / 0.10, 0, 1) * 100)


def analyze(feat: pd.DataFrame, cfg: dict | None = None, cross: Optional[dict] = None) -> dict:
    acfg = (cfg or {}).get("anomaly", {})
    z_thr = float(acfg.get("z_threshold", 2.5))
    last = feat.iloc[-1]

    scores: Dict[str, float] = {}
    deviations: Dict[str, float] = {}
    types: List[str] = []
    for col, label in Z_MAP.items():
        z = last.get(col, np.nan)
        if z is None or np.isnan(z):
            continue
        deviations[col] = round(float(z), 2)
        scores[label] = z_to_score(float(z))
        if abs(z) >= z_thr and not (col == "z_vol" and z < 0):
            types.append(label)

    corr_score = 0.0
    if cross and cross.get("correlation_break"):
        corr_score = float(cross.get("cross_market_score") or 0)
        types.append("CORRELATION_BREAK")
    scores["CORRELATION_BREAK"] = corr_score

    # z_ret / z_velocity / z_vol move together; the second term must come from a different family, otherwise
    # one event is counted twice.
    cluster = {"RETURN_ANOMALY": "price", "VELOCITY_ANOMALY": "price", "VOLATILITY_SPIKE": "vol",
               "VOLUME_ANOMALY": "liquidity", "SPREAD_ANOMALY": "liquidity", "CORRELATION_BREAK": "cross"}
    best_by_cluster: Dict[str, float] = {}
    for k, v in scores.items():
        c = cluster.get(k, k)
        best_by_cluster[c] = max(best_by_cluster.get(c, 0.0), v)
    ranked = sorted(best_by_cluster.values(), reverse=True)
    stat = (0.7 * ranked[0] + 0.3 * ranked[1]) if len(ranked) > 1 else (ranked[0] if ranked else 0.0)

    iso = None
    if acfg.get("use_isolation_forest", True):
        try:
            iso = _iforest_score(feat)
        except Exception as exc:  # never break the scan
            log.warning("IsolationForest skipped: %s", exc)
    score = stat if iso is None else 0.75 * stat + 0.25 * iso

    top_col = max(deviations, key=lambda k: abs(deviations[k])) if deviations else None
    return {
        "anomaly_score": round(float(score), 1),
        "anomaly_detected": bool(types) or score >= float(acfg.get("alert_threshold", 80)),
        "anomaly_types": sorted(set(types)),
        "anomaly_component_scores": {k: round(v, 1) for k, v in scores.items()},
        "anomaly_deviations": deviations,
        "anomaly_top_deviation_sigma": deviations.get(top_col) if top_col else None,
        "anomaly_isolation_forest": None if iso is None else round(iso, 1),
    }
