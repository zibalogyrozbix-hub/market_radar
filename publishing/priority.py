"""Event detection, priority scoring, duplicate prevention and cooldown logic (pure functions)."""
from __future__ import annotations

import hashlib
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
    """1.0 for a never-seen event; otherwise the largest scaled change vs the last published one."""
    if not prev_sig:
        return 1.0
    ch = []
    for k, scale in _SCALE.items():
        if k in prev_sig and k in sig:
            d = abs(sig[k] - prev_sig[k])
            ch.append(1.0 if k == "regime_code" and d > 0 else (0.0 if k == "regime_code" else min(1.0, d / scale / 0.5)))
    return float(max(ch)) if ch else 1.0


def priority_score(result: Dict[str, Any], novelty: float, weights: Dict[str, float]) -> Dict[str, float]:
    comp = {
        "anomaly": (result.get("anomaly_score") or 0) / 100,
        "novelty": float(novelty),
        "probability_edge": float(np.clip((result.get("edge") or 0) / 0.15, 0, 1)) if result.get("edge_label") == "CLEAR EDGE" else 0.0,
        "similarity": (result.get("similarity_score") or 0) / 100,
        "regime_change": 1.0 if result.get("regime_changed") else 0.0,
        "cross_market": (result.get("cross_market_score") or 0) / 100 if result.get("correlation_break") else 0.0,
    }
    total = sum(weights.get(k, 0.0) * v for k, v in comp.items())
    return {"priority": float(total), **comp}


def detect_candidate_events(result: Dict[str, Any], cfg: dict) -> List[str]:
    """Which event types does this scan qualify for (before priority/cooldown filtering)?"""
    th = cfg.get("thresholds", {})
    types = []
    if result.get("anomaly_detected") and (result.get("anomaly_score") or 0) >= th.get("anomaly_alert", 80):
        types.append("ANOMALY_ALERT")
    if (result.get("similarity_score") or 0) >= th.get("similarity_alert", 85) and result.get("edge_label") == "CLEAR EDGE":
        types.append("HISTORICAL_MATCH")
    if result.get("regime_changed") and result.get("regime") in NOTABLE_REGIMES:
        types.append("REGIME_CHANGE")
    if result.get("correlation_break"):
        types.append("CROSS_MARKET_BREAK")
    if result.get("volatility_state") in ("EXPANSION", "HIGH") and (result.get("volatility_score") or 0) >= th.get("volatility_alert", 90):
        types.append("VOLATILITY_ALERT")
    if (result.get("market_score") or 0) >= th.get("scan_score", 60) or result.get("edge_label") == "CLEAR EDGE":
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
    pr = priority_score(result, nov, cfg.get("priority_weights", {}))
    threshold = th.get("regime_change_priority", 0.5) if event_type == "REGIME_CHANGE" else th.get("post_priority", 0.45)
    out = {"type": event_type, "priority": pr["priority"], "components": pr, "signature": sig, "novelty": nov,
           "threshold": threshold, "publish": False, "reason": ""}
    if already_published_hash:
        out["reason"] = "duplicate event (same hash)"
        return out
    cd = float(cfg.get("publishing", {}).get("cooldown_minutes", {}).get(event_type, 60))
    if last_event and in_cooldown(last_event["time"], now, cd) and nov < 0.5:
        out["reason"] = f"cooldown ({cd:.0f} min) and situation unchanged"
        return out
    if pr["priority"] <= threshold:
        out["reason"] = f"priority {pr['priority']:.2f} <= {threshold:.2f}"
        return out
    out["publish"], out["reason"] = True, "ok"
    return out
