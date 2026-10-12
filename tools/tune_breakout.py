"""Backtest of the BREAKOUT definition: which Donchian window (N previous candles) gives breakouts that continue?

For every window N a *new* breakout is a bar that closes beyond the high/low of the previous N candles while
ATR ratio > 1.1 (and the previous bar was not already a breakout). We then look `horizon` candles ahead and
measure how often price continued in the breakout direction, compared with how often it moved that way on an
ordinary bar (baseline). Usage:  python main.py --tune-breakout [--symbol EURUSD]
(or the `tune-breakout` job in GitHub Actions). Costs ~1 Twelve Data credit per symbol.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from features.builder import build_features

WINDOWS = (20, 48, 72)
ATR_RATIO_MIN = 1.1
MIN_EVENTS = 150  # pooled events needed before a window may be recommended


def breakout_events(df: pd.DataFrame, window: int, horizon: int = 4) -> pd.DataFrame:
    """One row per new breakout: direction (+1/-1), forward return, forward move in ATR units."""
    feat = build_features(df, donchian_window=window)
    up, dn = feat["donch_up"] > 0, feat["donch_dn"] > 0
    brk = (up | dn) & (feat["atr_ratio"] > ATR_RATIO_MIN)
    new = brk & ~brk.shift(1, fill_value=False)
    direction = pd.Series(np.where(up, 1, -1), index=feat.index)
    fwd = np.log(feat["close"]).shift(-horizon) - np.log(feat["close"])
    atr = feat["atr_pct"] / 100.0  # atr_pct is in percent; fwd is a log-return fraction
    out = pd.DataFrame({"dir": direction, "fwd": fwd, "atr": atr, "base_up": (fwd > 0).astype(float)})
    out = out[new].dropna()
    out["cont"] = (np.sign(out["fwd"]) == out["dir"]).astype(float)
    out["move_atr"] = out["dir"] * out["fwd"] / out["atr"].replace(0, np.nan)
    return out.assign(n_bars=len(feat.dropna(subset=["atr_ratio"])))


def summarize_windows(per_symbol: Dict[str, pd.DataFrame], windows: Iterable[int] = WINDOWS, horizon: int = 4) -> List[dict]:
    rows = []
    for w in windows:
        ev = [breakout_events(df, w, horizon).assign(sym=s) for s, df in per_symbol.items()]
        ev = pd.concat(ev) if ev else pd.DataFrame()
        n = len(ev)
        if n == 0:
            rows.append({"window": w, "events": 0})
            continue
        bars = int(ev.groupby("sym")["n_bars"].first().sum())
        # baseline: probability that an ordinary bar moves in the same direction as the breakout did
        base = float(np.where(ev["dir"] > 0, ev["base_up"].mean(), 1 - ev["base_up"].mean()).mean())
        cont = float(ev["cont"].mean())
        se = math.sqrt(max(base * (1 - base), 1e-9) / n)
        rows.append({"window": w, "events": n, "per_1000_bars": round(1000 * n / max(bars, 1), 1),
                     "continued_pct": round(100 * cont, 1), "baseline_pct": round(100 * base, 1),
                     "excess_pp": round(100 * (cont - base), 1), "z": round((cont - base) / se, 2),
                     "mean_move_atr": round(float(ev["move_atr"].mean()), 3)})
    return rows


def recommend(rows: List[dict]) -> dict:
    """Highest z-score among windows with enough events; None if nothing is convincing."""
    ok = [r for r in rows if r.get("events", 0) >= MIN_EVENTS]
    best = max(ok, key=lambda r: r["z"], default=None)
    if best is None:
        return {"window": None, "reason": f"fewer than {MIN_EVENTS} events for every window"}
    if best["z"] < 1.64:  # ~95% one-sided
        return {"window": None, "reason": f"no window beats the baseline significantly (best z={best['z']} at N={best['window']})"}
    return {"window": best["window"], "reason": f"best z={best['z']}, excess {best['excess_pp']} pp"}


def run(cfg: dict, pipe, symbols=None) -> int:
    horizon = int(cfg["prediction"]["horizon"])
    tf = (cfg["timeframes"] or ["H1"])[0]
    peer_only = set(cfg.get("data", {}).get("peer_only", []))
    data = {}
    for sym in symbols or [s for s in cfg["symbols"] if s not in peer_only]:
        df, rep = pipe.loader.get(sym, tf)
        if df is None:
            print(f"skip {sym}: {'; '.join(rep.reasons)[:100]}")
            continue
        data[sym] = df
    if not data:
        print("no data")
        return 1
    rows = summarize_windows(data, WINDOWS, horizon)
    print(f"BREAKOUT window test | {len(data)} symbols | {tf} | look-ahead {horizon} candles | ATR ratio > {ATR_RATIO_MIN}")
    print(f"{'N':>4} {'events':>7} {'/1000bars':>9} {'continued%':>10} {'baseline%':>9} {'excess pp':>9} {'z':>6} {'move(ATR)':>9}")
    for r in rows:
        if r.get("events"):
            print(f"{r['window']:>4} {r['events']:>7} {r['per_1000_bars']:>9} {r['continued_pct']:>10} {r['baseline_pct']:>9} "
                  f"{r['excess_pp']:>9} {r['z']:>6} {r['mean_move_atr']:>9}")
        else:
            print(f"{r['window']:>4} no events")
    rec = recommend(rows)
    print("RECOMMENDATION:", json.dumps(rec))
    if rec["window"]:
        print(f"-> set regime.donchian_window: {rec['window']} in config.yaml, then run the `train` job")
    else:
        print("-> keep the current window (or choose manually by taste); no statistical reason to change")
    return 0
