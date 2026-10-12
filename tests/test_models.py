import tempfile
from pathlib import Path

import numpy as np

from evaluation.backtest import walk_forward
from evaluation.metrics import classification_metrics, bucket_table
from features.builder import FEATURE_COLUMNS
from models import ensemble as ens
from models.calibration import Calibrator, brier_score, calibration_curve
from models.ml_models import MLModelSet
from scoring import scorer
from training.dataset import build_dataset
from training.model_registry import ModelRegistry
from training.train import is_better, train_symbol
from training.validation import assert_chronological, chronological_split, walk_forward_splits
from tests.helpers import cfg, candles


def _ds():
    return build_dataset(candles(), 4)[1]


def test_chronological_split_no_overlap():
    tr, ca, te = chronological_split(1000, 0.7, 0.15, gap=4)
    assert tr.max() < ca.min() < ca.max() < te.min()
    assert ca.min() - tr.max() > 4 and te.min() - ca.max() > 4


def test_walk_forward_splits_purge_and_order():
    splits = list(walk_forward_splits(3000, 5, 1000, gap=4))
    assert len(splits) == 5
    for tr, te in splits:
        assert_chronological(tr, te)
        assert te.min() - tr.max() > 4
    assert splits[1][1].min() > splits[0][1].max()


def test_ml_models_predict_valid_probabilities():
    ds = _ds().iloc[:1500]
    ms = MLModelSet(cfg()["models"]).fit(ds[FEATURE_COLUMNS], ds["direction"], ds["future_return"])
    assert "logistic" in ms.classifiers and "random_forest" in ms.classifiers
    X = ds[FEATURE_COLUMNS].iloc[-5:]
    for p in ms.predict_proba_up(X).values():
        assert ((p >= 0) & (p <= 1)).all()
    q = ms.predict_quantiles(X.iloc[[-1]])
    assert list(q) == ["p10", "p25", "p50", "p75", "p90"] and q["p10"] <= q["p50"] <= q["p90"]
    assert np.isfinite(ms.predict_return(X)).all()
    assert ms.top_drivers(3)


def test_ml_survives_failed_model(monkeypatch=None):
    ds = _ds().iloc[:1200]
    ms = MLModelSet({"logistic": {"C": -1.0}})  # invalid C => logistic fails, RF still works
    ms.fit(ds[FEATURE_COLUMNS], ds["direction"], ds["future_return"], with_quantiles=False)
    assert "logistic" in ms.failed and "random_forest" in ms.classifiers
    assert ms.predict_ensemble_raw(ds[FEATURE_COLUMNS].iloc[-3:]).shape == (3,)


def test_calibration_improves_overconfident_probs():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.3, 0.7, 4000)
    y = (rng.uniform(size=4000) < p_true).astype(float)
    p_over = np.clip(0.5 + (p_true - 0.5) * 3, 0.01, 0.99)     # overconfident
    for method in ("isotonic", "platt"):
        cal = Calibrator(method).fit(p_over[:2000], y[:2000])
        out = cal.transform(p_over[2000:])
        assert brier_score(y[2000:], out) < brier_score(y[2000:], p_over[2000:])
        assert ((out > 0) & (out < 1)).all()
    curve = calibration_curve(y, p_over)
    assert curve and all(0 <= c["actual_frequency"] <= 1 for c in curve)


def test_metrics_basic():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    p = np.array([.9, .2, .8, .6, .4, .3, .7, .1])
    m = classification_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["brier"] < 0.1 and m["brier_skill"] > 0
    assert bucket_table(y, p)


def test_walk_forward_has_no_lookahead_and_oos_only():
    ds = _ds()
    wf = walk_forward(ds, cfg(), 4, n_splits=3, min_train=1000)
    assert wf["ok"]
    oos = wf["oos"]
    assert oos.index.is_monotonic_increasing and oos.index.is_unique
    assert oos.index.min() > ds.index[1000]            # nothing before first training window ends
    assert wf["metrics"]["n"] == len(oos)


def test_training_registry_and_activation_rule():
    reg = ModelRegistry(Path(tempfile.mkdtemp()))
    rep = train_symbol("XAUUSD", "H1", candles(), cfg(), reg)
    assert rep["trained"] and rep["activated"] and rep["version"] == "ML_v001"
    bundle = reg.load_active("XAUUSD", "H1")
    assert bundle["version"] == "ML_v001" and "model_set" in bundle
    assert reg.next_version("XAUUSD", "H1") == "ML_v002"
    assert is_better({"brier": 0.24}, {"brier": 0.25})
    assert not is_better({"brier": 0.26}, {"brier": 0.25})
    assert is_better({"brier": 0.25}, None)
    assert not is_better({"brier": 0.25, "return_mae": 2.0}, {"brier": 0.25, "return_mae": 1.0})


def test_training_skips_when_data_too_short():
    reg = ModelRegistry(Path(tempfile.mkdtemp()))
    out = train_symbol("XAUUSD", "H1", candles().iloc[:700].reset_index(drop=True), cfg(), reg)
    assert out["trained"] is False


def test_ensemble_weights_from_config_and_bounds():
    parts = {"trend_bias": 0.8, "momentum_bias": 0.6, "cross_bias": 0.0, "similarity_up_rate": 0.7,
             "ml_prob_up": 0.7, "ml_expected_return": 0.002, "similarity_median_return": 0.003,
             "volatility_score": 50, "volatility_state": "NORMAL", "anomaly_score": 0,
             "model_probs": {"logistic": 0.68, "random_forest": 0.75}}
    out = ens.combine(parts, cfg()["weights"])
    assert 0.5 < out["probability_up"] < 0.8 and abs(out["probability_up"] + out["probability_down"] - 1) < 1e-9
    assert out["probability_kind"] == "raw model probability" and out["model_agreement"] == "MEDIUM"  # trend/momentum (<=0.60) now count too
    assert 0.002 <= out["expected_return"] <= 0.003
    # heavier ML weight must move the output toward the ML probability
    w2 = dict(cfg()["weights"], ml_weight=5.0)
    assert ens.combine(parts, w2)["probability_up"] > out["probability_up"] - 1e-9
    # anomaly/volatility shrink toward 0.5
    hot = dict(parts, anomaly_score=100, volatility_state="EXPANSION", volatility_score=100)
    assert ens.combine(hot, cfg()["weights"])["probability_up"] < out["probability_up"]


def test_ensemble_calibrated_label_and_no_inputs():
    cal = Calibrator("platt").fit(np.linspace(.2, .8, 200), (np.linspace(.2, .8, 200) > .5).astype(float))
    out = ens.combine({"ml_prob_up": 0.7, "model_probs": {}}, cfg()["weights"], cal)
    assert out["probability_kind"] == "calibrated"
    none = ens.combine({"model_probs": {}}, cfg()["weights"])
    assert none["probability_up"] is None and none["probability_kind"] == "N/A"


def test_model_agreement_levels():
    assert ens.model_agreement({"a": .68, "b": .75, "c": .77, "d": .71}) == "HIGH"
    assert ens.model_agreement({"a": .52, "b": .81, "c": .49}) == "LOW"
    assert ens.model_agreement({"a": .6}) == "N/A"


def test_edge_and_no_clear_edge():
    e = scorer.compute_edge(0.74, 0.51, 0.05)
    assert abs(e["edge"] - 0.23) < 1e-9 and e["edge_label"] == "CLEAR EDGE"
    assert scorer.compute_edge(0.52, 0.51, 0.05)["edge_label"] == "NO CLEAR EDGE"
    d = scorer.compute_edge(0.30, 0.50, 0.05)
    assert d["edge_side"] == "DOWN" and d["edge_label"] == "CLEAR EDGE"
    assert scorer.compute_edge(None, 0.5, 0.05)["edge_label"] == "N/A"


def test_risk_independent_of_probability_direction():
    calm = scorer.risk_score(30, "NORMAL", 0, 0, 10, 1, 5, 0.74, "HIGH")
    wild = scorer.risk_score(98, "EXPANSION", 3.5, -2.5, 10, 1, 95, 0.74, "LOW")
    assert calm["risk"] in ("LOW", "MEDIUM") and wild["risk"] in ("HIGH", "EXTREME")
    assert wild["risk_score"] > calm["risk_score"]


def test_market_score_weighted_and_handles_missing():
    comps = {"trend": 81, "momentum": 87, "volatility": 72, "anomaly": 91, "similarity": 94, "ml": 74, "cross_market": None}
    s = scorer.market_score(comps, cfg()["score_weights"])
    assert 70 < s["market_score"] < 95 and "cross_market" not in s["market_score_components"]
    assert scorer.market_score({"trend": None}, cfg()["score_weights"])["market_score"] is None
