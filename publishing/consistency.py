"""Consistency guard: one place that decides what a post may claim, so text and image never contradict each other.

`consistency_guard(result, cfg)` returns a *copy* of the result with extra fields:
  lean        {"side": "UP"|"DOWN"|None, "p_side": float|None, "neutral": bool}   direction the models lean to
  alignment   "aligned" | "mixed" | "n/a"       trend/momentum versus the models' lean
  mixed_note  e.g. "trend ↘ · momentum ↘, models lean ↑" (only when mixed)
  edge_ok     False => the post must say "no clear edge" (mixed signals or edge not trustworthy)
  move        {"show": bool, "text": str}       expected move, or an honest "≈0 within normal range"
  history     {"show": bool, "coin_flip": bool} similar-situations line
  watch       one conditional "what to watch" sentence ("" => omit the line)
  has_volume  False for providers without real volume (Twelve Data FX): volume rows are hidden
Nothing here changes stored predictions; it only governs what is printed.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

NEUTRAL_BAND = 0.03          # |p - 0.5| below this => "no directional lean"
MIN_SIMILARITY_LINE = 60     # similarity score needed to print the "similar situations" line


def _sign(direction) -> int:
    d = str(direction or "").upper()
    return 1 if d in ("UP", "BULLISH", "LONG") else -1 if d in ("DOWN", "BEARISH", "SHORT") else 0


def _arrow(s: int) -> str:
    return "↗" if s > 0 else "↘" if s < 0 else "→"


def lean_of(r: Dict[str, Any]) -> Dict[str, Any]:
    p = r.get("probability_up")
    if p is None:
        return {"side": None, "p_side": None, "neutral": True}
    if abs(p - 0.5) < NEUTRAL_BAND:
        return {"side": None, "p_side": None, "neutral": True}
    return {"side": "UP" if p > 0.5 else "DOWN", "p_side": max(p, 1 - p), "neutral": False}


def edge_trusted(r: Dict[str, Any]) -> bool:
    """An edge is only worth highlighting/ranking when the probability is calibrated AND the model beat the baseline."""
    skill = r.get("model_skill")
    return r.get("probability_kind") == "calibrated" and skill is not None and skill > 0


def expected_move(r: Dict[str, Any], lean: Dict[str, Any], horizon_txt: str) -> Dict[str, Any]:
    er = r.get("expected_return")
    if er is None:
        return {"show": False, "text": "N/A"}
    sim = r.get("similarity_details") or {}
    p25, p75 = sim.get("p25"), sim.get("p75")
    if p25 is None or p75 is None:
        return {"show": True, "text": f"{er * 100:+.2f}%" if abs(er) * 100 >= 0.005 else "≈0.00%"}
    half = abs(p75 - p25) / 2
    same_side = lean["side"] is not None and ((er > 0) == (lean["side"] == "UP"))
    if half > 0 and abs(er) >= 0.25 * half and same_side:
        return {"show": True, "text": f"{er * 100:+.2f}%"}
    return {"show": False, "text": f"≈0 (within normal ±{half * 100:.2f}%) · {horizon_txt}"}


def _watch(r: Dict[str, Any]) -> str:
    reg = r.get("regime")
    t_score, m_score = r.get("trend_score"), r.get("momentum_score")
    t_dir, m_dir = _sign(r.get("trend_direction")), _sign(r.get("momentum_direction") or r.get("momentum_bias"))
    if reg in ("TREND_UP", "TREND_DOWN"):
        trend_sign = 1 if reg == "TREND_UP" else -1
        if m_dir and m_dir != trend_sign:
            return "Momentum is turning against the trend."
        if m_score is not None and t_score is not None and m_score < t_score - 10:
            return "Momentum is weaker than the trend (fading)."
        if (r.get("volatility_score") or 0) >= 60:
            return "Volatility rising; swings against the trend may be larger."
        return ""
    return {
        "COMPRESSION": "A break of the recent range with rising volatility.",
        "EXPANSION": "Whether volatility keeps rising or fades back.",
        "BREAKOUT": "Follow-through on the next candles, or a fall back into the range.",
        "REVERSAL": "Whether momentum confirms the turn.",
        "RANGE": "A decisive move out of the range.",
        "HIGH_VOLATILITY": "Whether volatility cools down.",
        "LOW_VOLATILITY": "A sudden pick-up in volatility.",
    }.get(reg, "")


def consistency_guard(result: Dict[str, Any], cfg: Optional[dict] = None) -> Dict[str, Any]:
    r = copy.deepcopy(result)
    hz = f"{r['horizon']}h" if r.get("horizon") and str(r.get("timeframe")) == "H1" else \
        (f"{r['horizon']} candles" if r.get("horizon") else "next candles")
    lean = lean_of(r)
    t_sign, m_sign = _sign(r.get("trend_direction")), _sign(r.get("momentum_direction") or r.get("momentum_bias"))
    lean_sign = 0 if lean["side"] is None else (1 if lean["side"] == "UP" else -1)
    alignment, note = "n/a", ""
    if lean_sign:
        opposing = [(n, s) for n, s in (("trend", t_sign), ("momentum", m_sign)) if s and s != lean_sign]
        alignment = "mixed" if opposing else "aligned"
        if opposing:
            tm = " · ".join(f"{n} {_arrow(s)}" for n, s in (("trend", t_sign), ("momentum", m_sign)) if s)
            note = f"{tm}, models lean {'↑' if lean_sign > 0 else '↓'}"
    edge_ok = alignment != "mixed" and r.get("edge_label") == "CLEAR EDGE" and not lean["neutral"]

    n = r.get("similar_states") or 0
    up = r.get("historical_up_rate")
    show_hist = bool(n and up is not None and (r.get("similarity_score") or 0) >= MIN_SIMILARITY_LINE)
    coin = bool(n and up is not None and abs(up - 0.5) < 1 / math.sqrt(max(n, 1)))
    has_volume = bool(r.get("has_volume", True))

    r.update({
        "lean": lean, "alignment": alignment, "mixed_note": note, "edge_ok": edge_ok,
        "move": expected_move(r, lean, hz), "history": {"show": show_hist, "coin_flip": coin},
        "watch": _watch(r), "has_volume": has_volume, "horizon_txt": hz,
    })
    return r
