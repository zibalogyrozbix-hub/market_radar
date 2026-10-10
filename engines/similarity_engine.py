"""Engine E - Historical similarity (nearest neighbours in scaled feature space).

Leakage control: the neighbour pool only contains states whose future outcome was fully known at
the query time (rows older than `embargo` bars, with a non-NaN future return), and the scaler is
fit on that pool only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from features.builder import FEATURE_COLUMNS

NA_RESULT = {
    "similarity_score": None, "n_similar": 0, "history_size": 0, "up_rate": None,
    "mean_future_return": None, "median_future_return": None,
    "p10": None, "p25": None, "p50": None, "p75": None, "p90": None,
    "similarity_reason": "insufficient history",
}


def analyze(feat: pd.DataFrame, future_return: pd.Series, cfg: dict | None = None,
            direction_threshold: float = 0.0, n_reference: int = 300) -> dict:
    scfg = (cfg or {}).get("similarity", {})
    k = int(scfg.get("k", 500))
    min_hist = int(scfg.get("min_history", 800))
    min_nb = int(scfg.get("min_neighbors", 50))
    embargo = int(scfg.get("embargo", 8))

    X_all = feat[FEATURE_COLUMNS]
    q = X_all.iloc[[-1]]
    if q.isna().any(axis=1).iloc[0]:
        return {**NA_RESULT, "similarity_reason": "current state has missing features"}

    pool = X_all.iloc[: max(0, len(X_all) - 1 - embargo)]
    fr = future_return.reindex(pool.index)
    mask = pool.notna().all(axis=1) & fr.notna()
    pool, fr = pool[mask], fr[mask]
    if len(pool) < max(min_hist, min_nb):
        return {**NA_RESULT, "history_size": int(len(pool))}

    k_eff = int(min(k, len(pool)))
    scaler = RobustScaler().fit(pool)
    Xp = np.clip(scaler.transform(pool), -8, 8)
    Xq = np.clip(scaler.transform(q), -8, 8)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xp)

    dist, idx = nn.kneighbors(Xq, n_neighbors=k_eff)
    q_dist = float(dist.mean())

    # reference: how tight is a *typical* historical state's neighbourhood?
    ref_pos = np.linspace(0, len(Xp) - 1, min(n_reference, len(Xp))).astype(int)
    rd, _ = nn.kneighbors(Xp[ref_pos], n_neighbors=k_eff + 1)
    ref = rd[:, 1:].mean(axis=1)  # drop self
    share_tighter = float((ref < q_dist).mean())  # fraction of history with a *tighter* neighbourhood
    score = float((1 - share_tighter) * 100)

    nb_ret = fr.to_numpy()[idx[0]]
    pct = np.percentile(nb_ret, [10, 25, 50, 75, 90])
    return {
        "similarity_score": round(score, 1),
        "n_similar": k_eff,
        "history_size": int(len(pool)),
        "up_rate": round(float((nb_ret > direction_threshold).mean()), 4),
        "mean_future_return": float(nb_ret.mean()),
        "median_future_return": float(np.median(nb_ret)),
        "p10": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
        "p75": float(pct[3]), "p90": float(pct[4]),
        "mean_neighbor_distance": round(q_dist, 4),
        "similarity_reason": "ok" if k_eff >= min_nb else "few neighbours",
    }
