"""Phase 3 - ML model set: Logistic, RandomForest, XGBoost (optional) + return regression/quantiles."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from features.builder import FEATURE_COLUMNS
from utils.logging_setup import get_logger

log = get_logger("models")

try:  # graceful fallback when xgboost is missing
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover - depends on environment
    XGBClassifier = None
    HAS_XGB = False


class MLModelSet:
    """Independent classifiers (P(up)) + regressors (expected return, quantiles).

    The scaler lives inside each sklearn pipeline, so it is fit on the training rows only.
    A model that fails to fit is dropped; the set keeps working with the remaining ones.
    """

    def __init__(self, params: dict | None = None, random_state: int = 42):
        self.params = params or {}
        self.rs = random_state
        self.classifiers: Dict[str, object] = {}
        self.mean_regressors: Dict[str, object] = {}
        self.quantile_models: Dict[float, object] = {}
        self.quantiles: List[float] = list(self.params.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]))
        self.feature_columns = list(FEATURE_COLUMNS)
        self.feature_importance: Dict[str, float] = {}
        self.failed: Dict[str, str] = {}

    # ---- training -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_dir: pd.Series, y_ret: pd.Series, with_quantiles: bool = True) -> "MLModelSet":
        Xv = X[self.feature_columns].to_numpy(dtype=float)
        yd, yr = np.asarray(y_dir, dtype=float), np.asarray(y_ret, dtype=float)
        p = self.params
        lp, rp, xp = p.get("logistic", {}), p.get("random_forest", {}), p.get("xgboost", {})

        builders = {
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=lp.get("C", 0.5), max_iter=lp.get("max_iter", 500))),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=rp.get("n_estimators", 200), max_depth=rp.get("max_depth", 6),
                min_samples_leaf=rp.get("min_samples_leaf", 30), n_jobs=1, random_state=self.rs),
        }
        if HAS_XGB:
            builders["xgboost"] = lambda: XGBClassifier(
                n_estimators=xp.get("n_estimators", 200), max_depth=xp.get("max_depth", 3),
                learning_rate=xp.get("learning_rate", 0.05), subsample=xp.get("subsample", 0.8),
                eval_metric="logloss", n_jobs=1, random_state=self.rs, verbosity=0)
        else:
            self.failed["xgboost"] = "xgboost not installed"
            log.warning("xgboost unavailable: continuing with Logistic + RandomForest")

        for name, build in builders.items():
            try:
                self.classifiers[name] = build().fit(Xv, yd)
            except Exception as exc:
                self.failed[name] = str(exc)
                log.warning("Model %s failed to fit: %s", name, exc)
        if not self.classifiers:
            raise RuntimeError("no classifier could be trained")

        # expected return (average of a linear and a tree model => shrunk estimate)
        try:
            self.mean_regressors["ridge"] = make_pipeline(RobustScaler(), Ridge(alpha=50.0)).fit(Xv, yr)
            self.mean_regressors["rf_reg"] = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=40, n_jobs=1, random_state=self.rs).fit(Xv, yr)
        except Exception as exc:
            self.failed["regression"] = str(exc)
            log.warning("Regression failed: %s", exc)

        # quantile estimation (gradient boosting with quantile loss)
        for q in (self.quantiles if with_quantiles else []):
            try:
                self.quantile_models[q] = GradientBoostingRegressor(
                    loss="quantile", alpha=q, n_estimators=60, max_depth=2, learning_rate=0.05,
                    subsample=0.8, random_state=self.rs).fit(Xv, yr)
            except Exception as exc:
                self.failed[f"quantile_{q}"] = str(exc)

        self._compute_importance()
        return self

    def _compute_importance(self) -> None:
        imps = []
        for name in ("random_forest", "xgboost"):
            m = self.classifiers.get(name)
            if m is not None and hasattr(m, "feature_importances_"):
                v = np.asarray(m.feature_importances_, dtype=float)
                imps.append(v / (v.sum() + 1e-12))
        if imps:
            avg = np.mean(imps, axis=0)
            self.feature_importance = {c: float(v) for c, v in zip(self.feature_columns, avg)}

    # ---- inference ----------------------------------------------------------------
    def predict_proba_up(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        Xv = X[self.feature_columns].to_numpy(dtype=float)
        out = {}
        for name, m in self.classifiers.items():
            try:
                classes = list(m.classes_) if hasattr(m, "classes_") else list(m[-1].classes_)
                out[name] = m.predict_proba(Xv)[:, classes.index(1.0)]
            except Exception as exc:
                log.warning("Model %s failed at inference: %s", name, exc)
        return out

    def predict_ensemble_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) average of classifier probabilities."""
        probs = self.predict_proba_up(X)
        if not probs:
            raise RuntimeError("no model produced a prediction")
        return np.mean(np.vstack(list(probs.values())), axis=0)

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.feature_columns].to_numpy(dtype=float)
        preds = [m.predict(Xv) for m in self.mean_regressors.values()]
        return np.mean(np.vstack(preds), axis=0) if preds else np.full(len(Xv), np.nan)

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[str, Optional[float]]:
        Xv = X[self.feature_columns].to_numpy(dtype=float)
        res = {}
        for q, m in self.quantile_models.items():
            res[f"p{int(round(q * 100))}"] = float(m.predict(Xv)[0])
        if res:  # enforce monotonic quantiles
            keys = sorted(res, key=lambda k: int(k[1:]))
            vals = np.maximum.accumulate([res[k] for k in keys])
            res = dict(zip(keys, map(float, vals)))
        return res

    def top_drivers(self, n: int = 4) -> List[str]:
        from features.builder import PRETTY

        seen, out = set(), []
        for col, _ in sorted(self.feature_importance.items(), key=lambda kv: -kv[1]):
            label = PRETTY.get(col, col)
            if label not in seen:
                seen.add(label)
                out.append(label)
            if len(out) >= n:
                break
        return out
