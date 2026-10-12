"""Event detection, priority scoring, duplicate prevention and cooldown logic (pure functions)."""
from __future__ import annotations

import hashlib
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from publishing.consistency import edge_trusted

NOTABLE_REGIMES = {"BREAKOUT", "REVERSAL", "EXPANSION", "TREND_UP", "TREND_DOWN", "HIGH_VOLATILITY", "COMPRESSION"}


def event_hash(symbol: str, timeframe: str, event_type: str, event_time) -> str:
    key = f"{symbol}|{timeframe}|{event_type}|{pd.Timestamp(event_time).isoformat()}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def signature(result: Dict[str, Any]) -> Dict[str, float]:
    """Numeric fingerprint of the situation; used to decide whether anything really changed."""
    def g(k, d=0.0):
        v = result.get(k)
        return d if v is None else float(v)
    return {
        "anomaly_score": g("anomaly_score"), "market_score": g("market_score"),
        "probability_up": g("probability_up", 0.5), "similarity_score": g("similarity_score"),
        "cross_market_score": g("cross_market_score"), "volatility_score": g("volatility_score"),
        "regime_code": float(zlib.crc32(str(result.get("regime", "")).encode()) % 997),
    }


_SCALE = {"anomaly_score": 100, "market_score": 100, "probability_up": 0.25, "similarity_score": 100,
          "cross_market_score": 100, "volatility_score": 100, "regime_code": 1}


def novelty_score(prev_sig: Optional[Dict[str, float]], sig: Dict[str, float]) -> float:
    """1.0 for a never-seen event; otherwise the largest scaled change of the *numeric* indicators vs the last
    published one. A different regime counts as 0.6 (it has its own priority component); it no longer forces 1.0."""
    if not prev_sig:
        return 1.0
    ch = []
    for k, scale in _SCALE.items():
        if k == "regime_code" or k not in prev_sig or k not in sig:
            continue
        ch.append(min(1.0, abs(sig[k] - prev_sig[k]) / scale / 0.5))
    nov = float(max(ch)) if ch else 0.0
    if "regime_code" in prev_sig and "regime_code" in sig and sig["regime_code"] != prev_sig["regime_code"]:
        nov = max(nov, 0.6)
    return nov


def escalated(prev_sig: Optional[Dict[str, float]], sig: Dict[str, float]) -> bool:
    """Only a real intensification may break through a cooldown (not just any change)."""
    if not prev_sig:
        return True
    rise = lambda k, d: sig.get(k, 0.0) - prev_sig.get(k, 0.0) >= d  # noqa: E731
    return rise("anomaly_score", 15) or rise("volatility_score", 15) or rise("cross_market_score", 15) or rise("market_score", 20)


# Different event types are worth posting for different reasons, so they weigh the components differently.
# Override in config.yaml with `priority_weights_by_type: {REGIME_CHANGE: {...}}`.
DEFAULT_TYPE_WEIGHTS = {
    "REGIME_CHANGE": {"regime_change": 0.35, "novelty": 0.25, "anomaly": 0.15, "similarity": 0.10, "cross_market": 0.15},
    "VOLATILITY_ALERT": {"volatility": 0.45, "novelty": 0.25, "anomaly": 0.15, "similarity": 0.10, "cross_market": 0.05},
}


def weights_for(event_type: str, cfg: dict) -> Dict[str, float]:
    return ((cfg.get("priority_weights_by_type") or {}).get(event_type) or DEFAULT_TYPE_WEIGHTS.get(event_type)
            or cfg.get("priority_weights", {}))


def priority_score(result: Dict[str, Any], novelty: float, weights: Dict[str, float]) -> Dict[str, float]:
    comp = {
        "anomaly": (result.get("anomaly_score") or 0) / 100,
        "novelty": float(novelty),
        # an edge only earns priority when the probability is calibrated and the model has proven skill
        "probability_edge": float(np.clip((result.get("edge") or 0) / 0.15, 0, 1))
        if (result.get("edge_label") == "CLEAR EDGE" and edge_trusted(result)) else 0.0,
        "similarity": (result.get("similarity_score") or 0) / 100,
        "regime_change": 1.0 if result.get("regime_changed") else 0.0,
        "cross_market": (result.get("cross_market_score") or 0) / 100 if result.get("correlation_break") else 0.0,
        "volatility": (result.get("volatility_score") or 0) / 100 if result.get("volatility_state") in ("EXPANSION", "HIGH") else 0.0,
    }
    total = sum(weights.get(k, 0.0) * v for k, v in comp.items())
    return {"priority": float(total), **comp}


def detect_candidate_events(result: Dict[str, Any], cfg: dict) -> List[str]:
    """Which event types does this scan qualify for (before priority/cooldown filtering)?"""
    th = cfg.get("thresholds", {})
    types = []
    if result.get("anomaly_detected") and (result.get("anomaly_score") or 0) >= th.get("anomaly_alert", 80):
        types.append("ANOMALY_ALERT")
    if (result.get("similarity_score") or 0) >= th.get("similarity_alert", 85):  # similarity does not depend on the ML models
        types.append("HISTORICAL_MATCH")
    if result.get("regime_changed") and result.get("regime") in NOTABLE_REGIMES:
        types.append("REGIME_CHANGE")
    if result.get("correlation_break"):
        types.append("CROSS_MARKET_BREAK")
    if result.get("volatility_state") in ("EXPANSION", "HIGH") and (result.get("volatility_score") or 0) >= th.get("volatility_alert", 90):
        types.append("VOLATILITY_ALERT")
    if (result.get("market_score") or 0) >= th.get("scan_score", 60) or (result.get("edge_label") == "CLEAR EDGE" and edge_trusted(result)):
        types.append("MARKET_SCAN")
    return types


def in_cooldown(last_time: Optional[datetime], now: datetime, minutes: float) -> bool:
    if last_time is None:
        return False
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    return (now - last_time) < timedelta(minutes=minutes)


def evaluate_event(result: Dict[str, Any], event_type: str, cfg: dict, last_event: Optional[dict],
                   already_published_hash: bool, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Decide whether to publish one candidate event.

    last_event: {"time": datetime, "signature": dict} of the last *published* event of the same
    symbol/type (or None). Returns {"publish": bool, "reason": str, "priority": float, ...}.
    """
    now = now or datetime.now(timezone.utc)
    th = cfg.get("thresholds", {})
    sig = signature(result)
    nov = novelty_score(last_event["signature"] if last_event else None, sig)
    pr = priority_score(result, nov, weights_for(event_type, cfg))
    threshold = ((cfg.get("thresholds_by_type") or {}).get(event_type)
                 or (th.get("regime_change_priority", 0.5) if event_type == "REGIME_CHANGE" else th.get("post_priority", 0.45)))
    out = {"type": event_type, "priority": pr["priority"], "components": pr, "signature": sig, "novelty": nov,
           "threshold": threshold, "publish": False, "reason": ""}
    if already_published_hash:
        out["reason"] = "duplicate event (same hash)"
        return out
    cd = float(cfg.get("publishing", {}).get("cooldown_minutes", {}).get(event_type, 60))
    if last_event and in_cooldown(last_event["time"], now, cd) and not escalated(last_event["signature"], sig):
        out["reason"] = f"cooldown ({cd:.0f} min), no real escalation"
        return out
    if pr["priority"] <= threshold:
        out["reason"] = f"priority {pr['priority']:.2f} <= {threshold:.2f}"
        return out
    out["publish"], out["reason"] = True, "ok"
    return out
