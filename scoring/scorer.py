"""Edge detection, risk score and Market Intelligence Score."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def baseline_up_rate(direction: pd.Series, window: int = 1000) -> Optional[float]:
    d = direction.dropna().tail(window)
    return float(d.mean()) if len(d) >= 100 else None


def compute_edge(prob_up: Optional[float], baseline: Optional[float], min_edge: float) -> dict:
    """Edge = probability of the favoured side minus that side's historical base rate."""
    if prob_up is None or baseline is None:
        return {"edge": None, "edge_side": None, "baseline_up": baseline, "edge_label": "N/A"}
    if prob_up >= 0.5:
        side, edge = "UP", prob_up - baseline
    else:
        side, edge = "DOWN", (1 - prob_up) - (1 - baseline)
    label = "CLEAR EDGE" if edge >= min_edge else "NO CLEAR EDGE"
    return {"edge": round(float(edge), 4), "edge_side": side, "baseline_up": round(baseline, 4), "edge_label": label}


def risk_score(vol_score, vol_state, spread_z, tick_vol_z, spread, atr, anomaly_score, prob_up,
               agreement: str) -> dict:
    """Risk is independent from direction/probability."""
    vol_c = 0.5 if vol_score is None else vol_score / 100.0
    if vol_state in ("HIGH", "EXPANSION"):
        vol_c = min(1.0, vol_c + 0.15)
    spread_c = 0.0 if spread_z is None or np.isnan(spread_z) else float(np.clip(spread_z / 4.0, 0, 1))
    liq_c = 0.0 if tick_vol_z is None or np.isnan(tick_vol_z) else float(np.clip(-tick_vol_z / 3.0, 0, 1))
    spread_liq = max(spread_c, liq_c)
    anom_c = (anomaly_score or 0.0) / 100.0
    unc_c = 0.5 if prob_up is None else 1 - float(np.clip(abs(prob_up - 0.5) / 0.25, 0, 1))
    dis_c = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0, "N/A": 0.5}.get(agreement, 0.5)
    score = 100 * (0.30 * vol_c + 0.15 * spread_liq + 0.20 * anom_c + 0.20 * unc_c + 0.15 * dis_c)
    label = "LOW" if score < 30 else "MEDIUM" if score < 50 else "HIGH" if score < 70 else "EXTREME"
    return {"risk_score": round(float(score), 1), "risk": label}


def market_score(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> dict:
    """Weighted mean of available component scores (0-100). Measures how notable/informative the
    current state is - it is NOT a direction and NOT a probability."""
    avail = {k: v for k, v in components.items() if v is not None and not np.isnan(v) and weights.get(k, 0) > 0}
    if not avail:
        return {"market_score": None, "market_score_components": {}}
    tot = sum(weights[k] for k in avail)
    score = sum(weights[k] * avail[k] for k in avail) / tot
    return {"market_score": round(float(score), 1),
            "market_score_components": {k: round(float(v), 1) for k, v in avail.items()}}


def ml_component(prob_up: Optional[float]) -> Optional[float]:
    if prob_up is None:
        return None
    return float(np.clip((max(prob_up, 1 - prob_up) - 0.5) / 0.25, 0, 1) * 100)
