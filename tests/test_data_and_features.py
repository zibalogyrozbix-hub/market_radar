import numpy as np
import pandas as pd

from data.validation import validate_candles
from features.builder import FEATURE_COLUMNS, build_features, make_targets
from tests.helpers import candles, features


def _raw(n=800):
    ts = pd.date_range("2026-01-05", periods=n, freq="60min", tz="UTC")
    c = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.1, "low": c - 0.1, "close": c,
                         "tick_volume": 100, "spread": 10})


def test_validation_clean_data_is_valid():
    df, rep = validate_candles(_raw(), "H1")
    assert rep.valid and rep.n_output == 800


def test_validation_removes_duplicates_and_sorts():
    raw = _raw()
    shuffled = pd.concat([raw, raw.iloc[10:15]]).sample(frac=1, random_state=3)
    df, rep = validate_candles(shuffled, "H1")
    assert rep.n_duplicates == 5
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique


def test_validation_flags_nan_and_unrealistic():
    raw = _raw()
    raw.loc[5, "close"] = np.nan
    raw.loc[9, "high"] = raw.loc[9, "low"] - 1  # high < low
    df, rep = validate_candles(raw, "H1")
    assert rep.n_nan_dropped == 1 and rep.n_unrealistic >= 1
    assert not df["close"].isna().any()


def test_validation_rejects_too_few_candles():
    _, rep = validate_candles(_raw(100), "H1")
    assert not rep.valid and any("min_bars" in r for r in rep.reasons)


def test_validation_detects_recent_gap_but_tolerates_weekend():
    raw = _raw()
    gap = raw.drop(index=range(700, 720)).reset_index(drop=True)  # 20h hole near the end
    _, rep = validate_candles(gap, "H1")
    assert rep.n_abnormal_gaps == 1 and not rep.valid
    # a Friday->Monday gap must NOT be abnormal
    ts = pd.date_range("2026-01-05", periods=900, freq="60min", tz="UTC")
    ts = ts[~((ts.weekday == 5) | ((ts.weekday == 4) & (ts.hour >= 22)) | ((ts.weekday == 6) & (ts.hour < 22)))]
    c = 100 + np.cumsum(np.random.default_rng(2).normal(0, 0.1, len(ts)))
    wk = pd.DataFrame({"timestamp": ts, "open": c, "high": c + .1, "low": c - .1, "close": c, "tick_volume": 1, "spread": 1})
    _, rep2 = validate_candles(wk, "H1", min_bars=500)
    assert rep2.n_abnormal_gaps == 0


def test_features_have_all_columns_and_are_finite_at_end():
    f = features()
    assert all(c in f.columns for c in FEATURE_COLUMNS)
    assert np.isfinite(f[FEATURE_COLUMNS].iloc[-1].to_numpy(dtype=float)).all()


def test_features_are_causal():
    """Changing FUTURE candles must not change feature values of earlier rows."""
    df = candles()
    full = build_features(df)
    cut = len(df) - 300
    mutated = df.copy()
    mutated.loc[cut:, ["open", "high", "low", "close"]] *= 1.5
    alt = build_features(mutated)
    a, b = full[FEATURE_COLUMNS].iloc[:cut], alt[FEATURE_COLUMNS].iloc[:cut]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_targets_look_forward_and_last_rows_nan():
    f = features()
    d, fr = make_targets(f["close"], 4)
    assert fr.iloc[-4:].isna().all() and fr.iloc[:-4].notna().all()
    i = 100
    assert abs(fr.iloc[i] - (f["close"].iloc[i + 4] / f["close"].iloc[i] - 1)) < 1e-12
    assert set(d.dropna().unique()) <= {0.0, 1.0}


def test_donchian_window_is_configurable_and_breakout_tool():
    from tools.tune_breakout import recommend, summarize_windows
    from tests.helpers import candles as _c
    df = _c("EURUSD", 1500)
    a, b = build_features(df, donchian_window=20), build_features(df, donchian_window=72)
    assert a["donch_up"].sum() > b["donch_up"].sum()  # a longer lookback is a stricter breakout
    rows = summarize_windows({"EURUSD": df}, (20, 48, 72), 4)
    assert [r["window"] for r in rows] == [20, 48, 72] and rows[0]["events"] > 0
    assert 0 <= rows[0]["continued_pct"] <= 100
    assert recommend([{"window": 20, "events": 10, "z": 5.0}])["window"] is None  # too few events
    assert recommend([{"window": 48, "events": 500, "z": 2.5, "excess_pp": 3.0}])["window"] == 48
