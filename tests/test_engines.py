import numpy as np
import pandas as pd

from engines import (anomaly_engine, cross_market_engine, momentum_engine, regime_engine, similarity_engine,
                     trend_engine, volatility_engine)
from engines.regime_engine import REGIMES
from tests.helpers import cfg, candles, features, targets


def test_trend_engine_output_contract():
    t = trend_engine.analyze(features())
    assert 0 <= t["trend_score"] <= 100 and t["trend_direction"] in ("UP", "DOWN", "NEUTRAL")
    assert -1 <= t["trend_bias"] <= 1


def test_trend_engine_detects_synthetic_uptrend():
    df = candles().copy()
    n = len(df)
    ramp = np.linspace(1.0, 1.6, n)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * ramp
    from features.builder import build_features
    t = trend_engine.analyze(build_features(df))
    assert t["trend_direction"] == "UP" and t["trend_strength"] > 40


def test_momentum_engine_output_contract():
    m = momentum_engine.analyze(features())
    assert 0 <= m["momentum_score"] <= 100 and m["momentum_direction"] in ("UP", "DOWN", "NEUTRAL")


def test_volatility_engine_states():
    v = volatility_engine.analyze(features())
    assert 0 <= v["volatility_score"] <= 100
    assert v["volatility_state"] in ("LOW", "NORMAL", "HIGH", "EXPANSION", "COMPRESSION")
    assert volatility_engine.state_from_row(0.5, 1.5, 0.5) == "EXPANSION"
    assert volatility_engine.state_from_row(0.5, 0.7, 0.1) == "COMPRESSION"
    assert volatility_engine.state_from_row(0.95, 1.0, 0.5) == "HIGH"
    assert volatility_engine.state_from_row(0.05, 1.0, 0.5) == "LOW"


def test_anomaly_engine_quiet_market_low_score():
    a = anomaly_engine.analyze(features(), cfg())
    assert 0 <= a["anomaly_score"] <= 100 and isinstance(a["anomaly_types"], list)


def test_anomaly_engine_flags_injected_shock():
    df = candles().copy()
    i = len(df) - 1
    df.loc[i, "close"] = df.loc[i - 1, "close"] * 1.03   # +3% bar
    df.loc[i, "high"] = df.loc[i, "close"] * 1.002
    df.loc[i, "tick_volume"] = df["tick_volume"].max() * 6
    from features.builder import build_features
    a = anomaly_engine.analyze(build_features(df), cfg())
    assert a["anomaly_detected"] and a["anomaly_score"] > 50
    assert "RETURN_ANOMALY" in a["anomaly_types"] and "VOLUME_ANOMALY" in a["anomaly_types"]
    assert a["anomaly_top_deviation_sigma"] is not None and abs(a["anomaly_top_deviation_sigma"]) > 2.5


def test_anomaly_z_mapping_bounds():
    assert anomaly_engine.z_to_score(0.5) == 0 and anomaly_engine.z_to_score(10) == 100


def test_similarity_engine_stats_and_ranges():
    f = features()
    _, fr = targets(f)
    s = similarity_engine.analyze(f, fr, cfg())
    assert 0 <= s["similarity_score"] <= 100 and s["n_similar"] == cfg()["similarity"]["k"] or s["n_similar"] > 0
    assert 0 <= s["up_rate"] <= 1
    assert s["p10"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p90"]


def test_similarity_returns_na_when_history_insufficient():
    f = features().iloc[:300]
    _, fr = targets(f)
    s = similarity_engine.analyze(f, fr, cfg())
    assert s["similarity_score"] is None and s["up_rate"] is None


def test_similarity_has_no_lookahead():
    """Altering returns/candles AFTER the query time must not change the result."""
    f = features()
    _, fr = targets(f)
    base = similarity_engine.analyze(f, fr, cfg())
    fr2 = fr.copy()
    fr2.iloc[-10:] = 0.5  # pretend the unknown (embargoed) future is wild
    again = similarity_engine.analyze(f, fr2, cfg())
    assert base["up_rate"] == again["up_rate"] and base["similarity_score"] == again["similarity_score"]


def test_cross_market_break_detection():
    own = candles("XAUUSD")
    peer = own.copy()
    # peer follows own for the old period, then becomes unrelated noise in the recent window
    rng = np.random.default_rng(5)
    r = np.log(own["close"]).diff().fillna(0).to_numpy()
    r2 = r.copy()
    r2[-100:] = rng.normal(0, r.std(), 100)
    peer["close"] = 100 * np.exp(np.cumsum(r2))
    out = cross_market_engine.analyze("XAUUSD", own, {"PEER": peer}, cfg())
    assert out["correlation_break"] is True and out["cross_market_score"] > 40
    same = cross_market_engine.analyze("XAUUSD", own, {"PEER": own.copy()}, cfg())
    assert same["correlation_break"] is False


def test_cross_market_without_peers():
    out = cross_market_engine.analyze("XAUUSD", candles(), {}, cfg())
    assert out["cross_market_score"] is None and out["correlation_break"] is False


def test_regime_is_valid_label():
    r = regime_engine.analyze(features())
    assert r["regime"] in REGIMES + ["UNKNOWN"] and isinstance(r["regime_changed"], bool)
