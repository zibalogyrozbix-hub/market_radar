"""Engine G - Cross-market analysis (rolling correlation, correlation break, divergence)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def peers_for(cfg: dict, symbol: str) -> List[str]:
    cm = cfg.get("cross_market", {})
    return list(cm.get(symbol, cm.get("default", [])))


def _log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"]).diff()


def analyze(symbol: str, own: pd.DataFrame, peers: Dict[str, pd.DataFrame], cfg: dict | None = None) -> dict:
    ccfg = (cfg or {}).get("cross_market", {})
    window = int(ccfg.get("window", 100))
    base_w = int(ccfg.get("baseline_window", 500))
    brk_thr = float(ccfg.get("break_threshold", 0.45))

    own_r = _log_returns(own.set_index("timestamp") if "timestamp" in own.columns else own).rename(symbol)
    pairs, breaks, divs = [], [], []
    rs_list = []
    for name, pdf in peers.items():
        pr = _log_returns(pdf.set_index("timestamp") if "timestamp" in pdf.columns else pdf).rename(name)
        j = pd.concat([own_r, pr], axis=1, join="inner").dropna()
        if len(j) < window + 50:
            continue
        cur = float(j.iloc[-window:].corr().iloc[0, 1])
        normal = float(j.iloc[-base_w:-window].corr().iloc[0, 1]) if len(j) >= base_w else float(j.iloc[:-window].corr().iloc[0, 1])
        if np.isnan(cur) or np.isnan(normal):
            continue
        change = cur - normal
        broke = abs(change) >= brk_thr
        # divergence: own cumulative move vs. beta-implied move over the last `window` bars
        w = j.iloc[-window:]
        beta = float(np.cov(j.iloc[:-window].iloc[:, 0], j.iloc[:-window].iloc[:, 1])[0, 1]
                     / (j.iloc[:-window].iloc[:, 1].var() + 1e-12)) if len(j) > window + 50 else 0.0
        resid = float(w.iloc[:, 0].sum() - beta * w.iloc[:, 1].sum())
        resid_sd = float((j.iloc[:, 0] - beta * j.iloc[:, 1]).std() * np.sqrt(window) + 1e-12)
        div_z = resid / resid_sd
        rel_strength = float(w.iloc[:, 0].sum() - w.iloc[:, 1].sum())
        pairs.append({"peer": name, "normal_corr": round(normal, 3), "current_corr": round(cur, 3),
                      "corr_change": round(change, 3), "break": bool(broke),
                      "divergence_z": round(div_z, 2), "relative_strength": round(rel_strength, 5)})
        if broke:
            breaks.append(abs(change))
        divs.append(abs(div_z))
        rs_list.append(rel_strength)

    if not pairs:
        return {"cross_market_score": None, "correlation_break": False, "divergence_score": None,
                "cross_market_bias": 0.0, "cross_pairs": [], "cross_market_reason": "no peer data"}

    max_change = max(abs(p["corr_change"]) for p in pairs)
    score = float(np.clip(max_change / (2 * brk_thr), 0, 1) * 100)
    div_score = float(np.clip((max(divs) - 1.0) / 3.0, 0, 1) * 100)
    # signed, weak bias: own market lagging its peers (negative relative strength) -> mild mean reversion up
    bias = float(np.clip(-np.tanh(np.mean(rs_list) * 20) * 0.3, -0.3, 0.3))
    return {
        "cross_market_score": round(score, 1),
        "correlation_break": bool(breaks),
        "divergence_score": round(div_score, 1),
        "cross_market_bias": round(bias, 4),
        "cross_pairs": sorted(pairs, key=lambda p: -abs(p["corr_change"])),
    }
