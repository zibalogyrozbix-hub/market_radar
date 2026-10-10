"""Database / tracking tests. Skipped automatically when SQLAlchemy is not installed."""
import copy
from datetime import timedelta

try:
    import sqlalchemy  # noqa: F401
    HAVE_SA = True
except ImportError:
    HAVE_SA = False

from tests.helpers import cfg
from tests.test_publishing import BASE


def _skip():
    try:
        import pytest
        pytest.skip("sqlalchemy not installed")
    except ImportError:
        return True


def _db():
    c = copy.deepcopy(cfg())
    c["database"]["url"] = "sqlite:///:memory:"
    from database.session import Database
    return Database(c)


def test_prediction_saved_resolved_and_immutable():
    if not HAVE_SA:
        return _skip()
    import pandas as pd
    from database import repository as repo
    from database.models import Candle, Prediction
    from evaluation.tracker import performance_report, resolve_predictions
    db = _db()
    ts = pd.Timestamp(BASE["timestamp"])
    with db.session() as s:
        assert repo.save_prediction(s, BASE) is not None
        assert repo.save_prediction(s, BASE) is None            # same candle+model => no duplicate
        target = ts + pd.Timedelta(hours=4)
        s.add(Candle(symbol="XAUUSD", timeframe="H1", timestamp=target.to_pydatetime(), open=1, high=1, low=1,
                     close=BASE["price"] * 1.01, tick_volume=1, spread=1))
    with db.session() as s:
        assert resolve_predictions(s) == 1 and resolve_predictions(s) == 0
        df = repo.resolved_predictions_df(s)
        assert df.iloc[0]["actual_direction"] == "UP" and bool(df.iloc[0]["correct"])
        rep = performance_report(s)
        assert rep["n"] == 1
    try:
        with db.session() as s:
            p = s.query(Prediction).first()
            p.probability_up = 0.1
    except RuntimeError:
        pass
    else:
        raise AssertionError("predictions must be immutable")


def test_posts_stay_pending_until_sent():
    if not HAVE_SA:
        return _skip()
    from database import repository as repo
    db = _db()
    with db.session() as s:
        row = repo.create_post(s, "MARKET_SCAN", "XAUUSD", "text", None)
        repo.mark_post(s, row, False, "telegram down")
    with db.session() as s:
        pend = repo.pending_posts(s)
        assert len(pend) == 1 and pend[0].last_error == "telegram down"
        repo.mark_post(s, pend[0], True)
    with db.session() as s:
        assert repo.pending_posts(s) == []


def test_event_dedupe_storage():
    if not HAVE_SA:
        return _skip()
    from database import repository as repo
    from publishing import priority as prio
    db = _db()
    h = prio.event_hash("XAUUSD", "H1", "ANOMALY_ALERT", BASE["timestamp"])
    ev = {"hash": h, "symbol": "XAUUSD", "timeframe": "H1", "type": "ANOMALY_ALERT", "event_time": BASE["timestamp"],
          "priority": 0.8, "signature": prio.signature(BASE), "why": ["x"]}
    with db.session() as s:
        assert not repo.event_exists(s, h)
        repo.create_event(s, ev, True)
    with db.session() as s:
        assert repo.event_exists(s, h)
        assert repo.last_published_event(s, "XAUUSD", "ANOMALY_ALERT") is not None
