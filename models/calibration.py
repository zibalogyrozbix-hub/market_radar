"""Probability calibration (isotonic / Platt) and calibration diagnostics."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class Calibrator:
    """Maps raw probabilities to calibrated probabilities. Fit on held-out, chronologically later data."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be isotonic or platt")
        self.method = method
        self.model = None
        self.n_fit = 0

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw, y) -> "Calibrator":
        p_raw, y = np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration needs both classes")
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p_raw, y)
        else:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p_raw), y)
        self.n_fit = len(y)
        return self

    def transform(self, p_raw):
        p_raw = np.asarray(p_raw, dtype=float)
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        if self.method == "isotonic":
            return np.clip(self.model.predict(p_raw), 0.01, 0.99)
        return np.clip(self.model.predict_proba(self._logit(p_raw))[:, 1], 0.01, 0.99)


def brier_score(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_curve(y, p, n_bins: int = 10) -> list[dict]:
    """Reliability table: per probability bin -> mean predicted vs actual frequency."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        hi_inc = i == n_bins - 1
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if hi_inc else (p < edges[i + 1]))
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                    "mean_predicted": float(p[m].mean()), "actual_frequency": float(y[m].mean())})
    return out


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    curve = calibration_curve(y, p, n_bins)
    n = sum(c["n"] for c in curve)
    return float(sum(c["n"] / n * abs(c["mean_predicted"] - c["actual_frequency"]) for c in curve)) if n else float("nan")
