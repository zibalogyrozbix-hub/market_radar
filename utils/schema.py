"""Standard analysis result - the contract between engines, storage and publishing."""
from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np

REQUIRED_KEYS = [
    "symbol", "timeframe", "timestamp", "price", "horizon",
    "trend_score", "momentum_score", "volatility_score", "anomaly_score", "similarity_score",
    "probability_up", "probability_down", "probability_kind", "expected_return",
    "regime", "risk", "risk_score", "model_agreement", "market_score", "edge", "edge_label",
    "model_version", "why",
]


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def finalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy (NaN -> None); raises if the contract is violated."""
    out = _clean(result)
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise ValueError(f"result violates schema, missing: {missing}")
    for k in ("trend_score", "momentum_score", "volatility_score", "anomaly_score", "similarity_score", "market_score"):
        v = out[k]
        if v is not None and not (0 <= v <= 100):
            raise ValueError(f"{k} out of range: {v}")
    for k in ("probability_up", "probability_down"):
        v = out[k]
        if v is not None and not (0 <= v <= 1):
            raise ValueError(f"{k} out of range: {v}")
    return out


def to_json(result: Dict[str, Any]) -> str:
    return json.dumps(_clean(result), ensure_ascii=False, indent=2)
