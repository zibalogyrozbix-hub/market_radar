"""Shared test helpers (plain functions - no pytest fixtures needed)."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import load_config  # noqa: E402
from data.providers import SyntheticProvider  # noqa: E402
from data.validation import validate_candles  # noqa: E402
from features.builder import build_features, make_targets  # noqa: E402

END = pd.Timestamp("2026-01-05 12:00", tz="UTC")


@functools.lru_cache(maxsize=None)
def cfg():
    c = load_config()
    c["models"]["xgboost"]["n_estimators"] = 30
    return c


@functools.lru_cache(maxsize=None)
def candles(symbol="XAUUSD", bars=2500, seed=7):
    df = SyntheticProvider(seed=seed, end=END).fetch(symbol, "H1", bars)
    clean, rep = validate_candles(df, "H1")
    assert rep.valid, rep.reasons
    return clean


@functools.lru_cache(maxsize=None)
def features(symbol="XAUUSD", bars=2500):
    return build_features(candles(symbol, bars))


def targets(feat, horizon=4):
    return make_targets(feat["close"], horizon)
