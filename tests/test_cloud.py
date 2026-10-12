"""Twelve Data provider, market hours, DB-backed model registry and libSQL (Turso) dialect smoke tests."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tests.helpers import cfg  # noqa: F401  (also puts the project on sys.path)

import pandas as pd

from data.providers import TwelveDataError, TwelveDataProvider
from utils.market_hours import is_market_open


class FakeResp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b


class FakeHTTP:
    def __init__(self, bodies):
        self.bodies, self.calls = list(bodies), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        b = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return b if isinstance(b, FakeResp) else FakeResp(b)


def _values(n=50, forming=True):
    end = pd.Timestamp.now(tz="UTC").floor("1h")  # open time of the still-forming candle
    ts = [end - pd.Timedelta(hours=i) for i in range(0 if forming else 1, n)]
    return {"status": "ok", "values": [{"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "open": "1.1", "high": "1.2",
                                        "low": "1.0", "close": "1.15"} for t in ts]}


def _prov(http, **kw):
    t = {"now": 0.0}
    sleeps = []

    def sleep(x):
        sleeps.append(x)
        t["now"] += x

    p = TwelveDataProvider("KEY", http=http, sleep=sleep, clock=lambda: t["now"], **kw)
    p._sleeps = sleeps
    return p


def test_twelvedata_maps_symbols_and_drops_forming_candle():
    http = FakeHTTP([_values(50, forming=True)])
    p = _prov(http, symbol_map={"US500": "SPY"})
    df = p.fetch("EURGBP", "H1", 40)
    assert http.calls[0]["symbol"] == "EUR/GBP" and http.calls[0]["interval"] == "1h"
    assert http.calls[0]["timezone"] == "UTC" and http.calls[0]["apikey"] == "KEY"
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[-1] + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=30)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]
    assert p.td_symbol("US500") == "SPY" and p.td_symbol("XAUUSD") == "XAU/USD"


def test_twelvedata_rate_limit_window():
    http = FakeHTTP([_values(10)])
    p = _prov(http, requests_per_minute=8)
    for _ in range(9):
        p.fetch("EURUSD", "H1", 5)
    assert len(p._sleeps) == 1 and 55 <= p._sleeps[0] <= 61, p._sleeps  # 9th call waits for the window


def test_twelvedata_429_retries_then_succeeds_and_errors_raise():
    http = FakeHTTP([{"status": "error", "code": 429, "message": "run out of API credits for the current minute"},
                     _values(10)])
    p = _prov(http)
    assert len(p.fetch("EURUSD", "H1", 5)) > 0 and 61 in p._sleeps
    bad = _prov(FakeHTTP([{"status": "error", "code": 400, "message": "symbol not found / plan"}]))
    try:
        bad.fetch("US500", "H1", 5)
        raise AssertionError("expected TwelveDataError")
    except TwelveDataError as exc:
        assert "400" in str(exc)
    day = _prov(FakeHTTP([{"status": "error", "code": 429, "message": "You have run out of API credits for the day"}]))
    try:
        day.fetch("EURUSD", "H1", 5)
        raise AssertionError("daily limit must not retry forever")
    except TwelveDataError:
        pass


def test_twelvedata_stale_guard_and_session_exemption():
    old = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=30)
    body = {"status": "ok", "values": [{"datetime": (old - pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                                        "open": "1", "high": "1", "low": "1", "close": "1"} for i in range(5)]}
    try:
        _prov(FakeHTTP([body]), max_stale_minutes=240).fetch("EURUSD", "H1", 5)
        raise AssertionError("stale data must be rejected")
    except TwelveDataError as exc:
        assert "stale" in str(exc)
    ok = _prov(FakeHTTP([body]), max_stale_minutes=240, session_symbols=["US500"]).fetch("US500", "H1", 5)
    assert len(ok) == 5
    closed = _prov(FakeHTTP([body]), max_stale_minutes=240, market_open=lambda s: False).fetch("EURUSD", "H1", 5)
    assert len(closed) == 5  # market closed (weekend): old candles are expected, e.g. for the weekly train job


def test_twelvedata_bad_key_fails_fast_and_key_is_cleaned():
    http = FakeHTTP([{"status": "error", "code": 401, "message": "**apikey** parameter is incorrect or not specified."}])
    p = _prov(http)
    p.api_key = " \"abc123\"\n".strip().strip('"')
    for _ in range(3):
        try:
            p.fetch("EURUSD", "H1", 5)
            raise AssertionError("expected TwelveDataError")
        except TwelveDataError as exc:
            assert "TWELVEDATA_API_KEY" in str(exc)
    assert len(http.calls) == 1, "after the first 401 no further requests may be sent"
    assert TwelveDataProvider(' "k1"\n', http=http).api_key == "k1"
    from config.settings import _clean
    assert _clean("  'tok'\n") == "tok" and _clean("  \n") is None


def test_publish_test_sends_every_post_type_without_db_writes():
    from data.loader import MarketDataLoader
    from data.providers import SyntheticProvider
    from pipeline import RadarPipeline
    from publishing.post_generator import _BUILDERS
    from publishing.telegram import TelegramPublisher
    from tests.helpers import END, cfg as _cfg
    from training.model_registry import ModelRegistry

    c = dict(_cfg())
    c["publishing"] = {**c["publishing"], "send_images": False}
    sent = []

    class Fake(TelegramPublisher):
        def publish(self, text, image_path=None):
            sent.append(text)
            return True, ""

    db = _db()
    pipe = RadarPipeline(c, db=db, loader=MarketDataLoader(c, SyntheticProvider(seed=7, end=END)),
                         registry=ModelRegistry(Path(tempfile.mkdtemp())), publisher=Fake("t", "c"))
    for t in sorted(_BUILDERS):
        assert pipe.publish_test(t, "EURUSD"), t
    assert len(sent) == len(_BUILDERS) and all(s.startswith("🧪 TEST POST") for s in sent)
    from sqlalchemy import func, select
    from database.models import Prediction, PostRow
    with db.session() as s:
        assert s.scalar(select(func.count(Prediction.id))) == 0 and s.scalar(select(func.count(PostRow.id))) == 0
    try:
        pipe.publish_test("NOPE")
        raise AssertionError("unknown type must fail")
    except ValueError:
        pass


def test_market_hours():
    c = {"market_hours": {"always_open": ["BTCUSD"], "session_only": ["US500"]}}
    utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)  # noqa: E731
    assert is_market_open("EURUSD", c, utc("2026-01-07 12:00"))        # Wednesday
    assert is_market_open("EURUSD", c, utc("2026-01-09 21:30"))        # Friday before 22:00
    assert not is_market_open("EURUSD", c, utc("2026-01-09 22:00"))    # Friday close
    assert not is_market_open("XAUUSD", c, utc("2026-01-10 12:00"))    # Saturday
    assert not is_market_open("EURUSD", c, utc("2026-01-11 21:59"))    # Sunday before open
    assert is_market_open("EURUSD", c, utc("2026-01-11 22:00"))        # Sunday open
    assert is_market_open("BTCUSD", c, utc("2026-01-10 12:00"))        # crypto always
    assert is_market_open("US500", c, utc("2026-01-07 03:00")) and not is_market_open("US500", c, utc("2026-01-10 12:00"))


def _db(url=None):
    from database.session import Database

    d = tempfile.mkdtemp()
    return Database({"database": {"url": url or f"sqlite:///{Path(d) / 'm.db'}"}, "_root": d})


def test_db_model_registry_roundtrip_with_chunks():
    from training.db_registry import CHUNK, DBModelRegistry

    reg = DBModelRegistry(_db())
    assert reg.next_version("EURGBP", "H1") == "ML_v001" and reg.load_active("EURGBP", "H1") is None
    blob = os.urandom(CHUNK * 2 + 123).hex()  # incompressible-ish payload => several chunks
    reg.save("EURGBP", "H1", "ML_v001", {"payload": blob, "k": 1}, {"summary": {"brier": 0.2}})
    reg.activate("EURGBP", "H1", "ML_v001")
    reg.save("EURGBP", "H1", "ML_v002", {"payload": "x"}, {"summary": {"brier": 0.1}})
    assert reg.next_version("EURGBP", "H1") == "ML_v003"
    assert reg.active_version("EURGBP", "H1") == "ML_v001"
    fresh = DBModelRegistry(reg.db)  # new process: nothing cached
    b = fresh.load_active("EURGBP", "H1")
    assert b["payload"] == blob and b["version"] == "ML_v001"
    assert fresh.meta("EURGBP", "H1", "ML_v002")["summary"]["brier"] == 0.1
    reg.activate("EURGBP", "H1", "ML_v002")
    assert fresh.load_active("EURGBP", "H1")["payload"] == "x"


def test_libsql_dialect_local_smoke():
    """The same ORM must work through the libSQL dialect that Turso uses (local file, no network)."""
    try:
        import sqlalchemy_libsql  # noqa: F401
    except Exception:
        return True  # skipped: optional dependency
    d = tempfile.mkdtemp()
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from database import repository as repo
    from database.models import Base, EventRow

    eng = create_engine(f"sqlite+libsql:///{Path(d) / 'l.db'}", future=True)
    Base.metadata.create_all(eng)
    S = sessionmaker(eng, expire_on_commit=False, future=True)
    ev = {"hash": "abc", "symbol": "EURGBP", "timeframe": "H1", "type": "ANOMALY_ALERT",
          "event_time": pd.Timestamp("2026-10-07 12:00", tz="UTC"), "priority": 0.7, "signature": {"a": 1}, "why": ["x"]}
    with S() as s:
        repo.create_event(s, ev, published=True)
        s.commit()
    with S() as s:
        assert repo.event_exists(s, "abc")
        assert s.scalar(select(EventRow).where(EventRow.symbol == "EURGBP")).signature == {"a": 1}


def test_db_model_registry_through_libsql_dialect():
    try:
        import sqlalchemy_libsql  # noqa: F401
    except Exception:
        return True
    from training.db_registry import CHUNK, DBModelRegistry

    d = tempfile.mkdtemp()
    reg = DBModelRegistry(_db(f"sqlite+libsql:///{Path(d) / 'r.db'}"))
    blob = os.urandom(CHUNK + 500).hex()
    reg.save("EURGBP", "H1", "ML_v001", {"payload": blob}, {"summary": {}})
    reg.activate("EURGBP", "H1", "ML_v001")
    assert DBModelRegistry(reg.db).load_active("EURGBP", "H1")["payload"] == blob


def test_turso_url_and_missing_token():
    from database.session import make_engine, turso_url

    assert turso_url("libsql://radar-user.turso.io") == "sqlite+libsql://radar-user.turso.io?secure=true"
    old = {k: os.environ.pop(k, None) for k in ("TURSO_AUTH_TOKEN", "DATABASE_URL")}
    os.environ["TURSO_DATABASE_URL"] = "libsql://radar-user.turso.io"
    try:
        make_engine({"database": {"url": "sqlite:///x.db"}})
        raise AssertionError("must require TURSO_AUTH_TOKEN")
    except RuntimeError as exc:
        assert "TURSO_AUTH_TOKEN" in str(exc)
    finally:
        os.environ.pop("TURSO_DATABASE_URL", None)
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def test_pipeline_scan_skips_closed_and_peer_only():
    from pipeline import RadarPipeline
    from tests.helpers import cfg as _cfg

    c = dict(_cfg())
    pipe = RadarPipeline.__new__(RadarPipeline)
    pipe.cfg = c
    got = pipe.scannable_symbols()
    assert "US500" not in got and "NAS100" not in got and "EURGBP" in got or not is_market_open("EURGBP", c)
