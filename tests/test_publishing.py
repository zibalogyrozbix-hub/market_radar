import copy
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline import summarize
from publishing import priority as prio
from publishing.image_generator import render_card, render_summary
from publishing.post_generator import FORBIDDEN, POST_TYPES, generate_post
from publishing.telegram import TelegramPublisher, build_message_payload, build_photo_payload
from scheduler.runner import next_run, run_daemon
from tests.helpers import cfg
from utils.schema import REQUIRED_KEYS, finalize_result

BASE = {
    "symbol": "XAUUSD", "timeframe": "H1", "timestamp": "2026-01-05T12:00:00+00:00", "price": 2300.0, "horizon": 4,
    "trend_score": 81.0, "trend_direction": "UP", "momentum_score": 87.0, "volatility_score": 72.0,
    "volatility_state": "EXPANSION", "atr_ratio": 1.4, "anomaly_score": 91.0, "anomaly_detected": True,
    "anomaly_types": ["VOLATILITY_SPIKE"], "anomaly_deviations": {"z_vol": 3.1, "z_velocity": 2.7, "z_ret": 1.0},
    "anomaly_component_scores": {"VOLUME_ANOMALY": 91.0, "CORRELATION_BREAK": 78.0},
    "similarity_score": 94.0, "similar_states": 500, "historical_up_rate": 0.71,
    "similarity_details": {"history_size": 4000, "median_future_return": 0.0034, "p25": -0.0008, "p75": 0.0051},
    "cross_market_score": 78.0, "correlation_break": True, "divergence_score": 60.0,
    "cross_pairs": [{"peer": "EURUSD", "normal_corr": -0.72, "current_corr": -0.21, "corr_change": 0.51, "break": True}],
    "probability_up": 0.74, "probability_down": 0.26, "probability_kind": "calibrated", "expected_return": 0.0028,
    "regime": "EXPANSION", "previous_regime": "RANGE", "regime_changed": True, "risk": "MEDIUM", "risk_score": 40.0,
    "model_agreement": "HIGH", "model_skill": 0.05, "market_score": 86.0, "edge": 0.23, "edge_label": "CLEAR EDGE", "model_version": "ML_v001",
    "why": ["Volatility expansion", "Historical similarity"],
}


def r(**kw):
    d = copy.deepcopy(BASE)
    d.update(kw)
    return d


def test_all_post_types_generate_without_signal_language():
    summary = summarize([BASE])
    for t in POST_TYPES:
        data = summary if t == "DAILY_MARKET_SUMMARY" else (
            {"symbol": "XAUUSD", "timeframe": "H1", "version": "ML_v002", "activated": True, "summary": {"brier": 0.24}}
            if t == "MODEL_UPDATE" else BASE)
        text = generate_post(t, data, cfg())
        assert text and not any(w in text.upper().replace("NOT A GUARANTEED", "") for w in FORBIDDEN)
        assert "BUY" not in text.split() and "SELL" not in text.split()


def test_market_scan_content_and_no_clear_edge_wording():
    from publishing.post_generator import strip_html, visible_length
    text = generate_post("MARKET_SCAN", BASE, cfg())
    plain = strip_html(text)
    for s in ("XAU/USD", "Volatility expansion", "TREND", "▰", "MKT SCORE", "74%", "+0.28%", "Risk MEDIUM",
              "Model Agreement HIGH", "500", "71%", "not financial advice", "#XAUUSD #MarketScan #Expansion", "candle 12:00 UTC"):
        assert s in plain, s
    assert "↗" in plain and "baseline 🔥" in plain  # calibrated 23 pp edge is highlighted
    assert 'href="https://telegra.ph/FXSCAN--User-Guide-10-10#Section-Probability"' in text
    assert visible_length(text) <= 1000
    ne = generate_post("MARKET_SCAN", r(edge_label="NO CLEAR EDGE", edge=0.01), cfg())
    assert "none clear" in ne and "74%" in ne  # values stay visible, no hard gating


def test_raw_probability_is_labelled_and_na_is_shown():
    raw = generate_post("MARKET_SCAN", r(probability_kind="raw model probability"), cfg())
    assert "(raw model)" in raw and "baseline 🔥" not in raw and "✅" not in raw
    na = generate_post("MARKET_SCAN", r(similarity_score=None, similar_states=None, historical_up_rate=None), cfg())
    assert "▱▱▱▱▱▱▱▱▱▱ N/A" in na and "Similar past situations: N/A" in na


def test_track_record_and_methods_lines_and_html_escape():
    from publishing.post_generator import strip_html
    t = strip_html(generate_post("MARKET_SCAN", r(track_record={"n": 37, "brier": 0.2}), cfg()))
    assert "building (37/100)" in t
    t = strip_html(generate_post("MARKET_SCAN", r(track_record={"n": 150, "brier": 0.231},
                                                   methods_info={"models": ["logistic", "xgboost"], "calibration": "isotonic"}), cfg()))
    assert "Brier 0.231" in t and "Logistic + XGBoost" in t and "isotonic calibration" in t
    assert "&lt;" in generate_post("ANOMALY_ALERT", r(why=["a<b"]), cfg())


def test_every_type_fits_caption_and_has_hashtags():
    from publishing.post_generator import visible_length
    for t in ("MARKET_SCAN", "ANOMALY_ALERT", "HISTORICAL_MATCH", "REGIME_CHANGE", "CROSS_MARKET_BREAK", "VOLATILITY_ALERT"):
        x = generate_post(t, BASE, cfg())
        assert visible_length(x) <= 1024 and "#XAUUSD" in x, t


def test_telegram_html_button_and_plain_fallback():
    from publishing.telegram import _plain
    pl = build_message_payload("@c", "<b>x</b>", "HTML", ("🧭 How do I use this?", "https://t.me/x"))
    assert pl["parse_mode"] == "HTML" and "inline_keyboard" in pl["reply_markup"]
    assert _plain("<b>a &amp; b</b>") == "a & b"
    calls = []

    class H:
        def post(self, url, data=None, files=None, timeout=None):
            calls.append(data)
            return FakeResp(400, {"description": "Bad Request: can't parse entities"}) if len(calls) == 1 else FakeResp(200, {})
    ok, _ = TelegramPublisher("T", "@c", http=H(), sleep=lambda s: None, parse_mode="HTML").publish("<b>x</b>")
    assert ok and calls[1].get("parse_mode") is None and calls[1]["text"] == "x"


def test_priority_threshold_dedupe_and_cooldown():
    c = cfg()
    now = datetime(2026, 1, 5, 13, tzinfo=timezone.utc)
    ok = prio.evaluate_event(BASE, "ANOMALY_ALERT", c, None, False, now)
    assert ok["publish"] and ok["priority"] > c["thresholds"]["post_priority"]
    quiet = prio.evaluate_event(r(anomaly_score=5, anomaly_detected=False, similarity_score=10, edge=0, edge_label="NO CLEAR EDGE",
                                  regime_changed=False, correlation_break=False), "MARKET_SCAN", c, None, False, now)
    assert not quiet["publish"] and "priority" in quiet["reason"]
    dup = prio.evaluate_event(BASE, "ANOMALY_ALERT", c, None, True, now)
    assert not dup["publish"] and "duplicate" in dup["reason"]
    last = {"time": now - timedelta(minutes=20), "signature": prio.signature(BASE)}
    cool = prio.evaluate_event(BASE, "ANOMALY_ALERT", c, last, False, now)
    assert not cool["publish"] and "cooldown" in cool["reason"]
    # a materially different situation overrides the cooldown
    changed = prio.evaluate_event(r(anomaly_score=99.0, regime="BREAKOUT"), "ANOMALY_ALERT", c, last, False, now)
    assert changed["publish"]


def test_event_hash_stable_and_distinct():
    a = prio.event_hash("XAUUSD", "H1", "ANOMALY_ALERT", "2026-01-05T12:00:00+00:00")
    assert a == prio.event_hash("XAUUSD", "H1", "ANOMALY_ALERT", "2026-01-05T12:00:00+00:00")
    assert a != prio.event_hash("XAUUSD", "H1", "MARKET_SCAN", "2026-01-05T12:00:00+00:00")
    assert prio.signature(BASE) == prio.signature(copy.deepcopy(BASE))  # deterministic across runs


def test_candidate_event_detection():
    types = prio.detect_candidate_events(r(volatility_score=95.0), cfg())
    assert "VOLATILITY_ALERT" not in prio.detect_candidate_events(BASE, cfg())  # 72 < threshold 90
    assert {"ANOMALY_ALERT", "HISTORICAL_MATCH", "REGIME_CHANGE", "CROSS_MARKET_BREAK", "VOLATILITY_ALERT", "MARKET_SCAN"} <= set(types)
    assert prio.detect_candidate_events(r(anomaly_score=10, anomaly_detected=False, similarity_score=20, regime_changed=False,
                                          correlation_break=False, volatility_state="NORMAL", market_score=30,
                                          edge_label="NO CLEAR EDGE"), cfg()) == []


class FakeResp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.text = code, body or {}, str(body)

    def json(self):
        return self._b


class FakeHttp:
    def __init__(self, codes):
        self.codes, self.calls = list(codes), []

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append((url, data, files))
        c = self.codes.pop(0)
        return FakeResp(c, {"description": "x", "parameters": {"retry_after": 1}})


def test_telegram_payloads():
    m = build_message_payload("@chan", "hi")
    assert m["chat_id"] == "@chan" and m["text"] == "hi"
    assert len(build_message_payload("c", "x" * 9000)["text"]) == 4096
    assert len(build_photo_payload("c", "y" * 2000)["caption"]) == 1024
    assert "caption" not in build_photo_payload("c")


def test_telegram_retry_then_success_and_failure_paths():
    sleeps = []
    http = FakeHttp([500, 429, 200])
    t = TelegramPublisher("TOK", "@c", retries=3, http=http, sleep=sleeps.append)
    ok, err = t.publish("hello")
    assert ok and len(http.calls) == 3 and len(sleeps) == 2
    assert http.calls[0][0].endswith("/botTOK/sendMessage")
    bad = TelegramPublisher("TOK", "@c", retries=3, http=FakeHttp([400]), sleep=lambda s: None)
    ok, err = bad.publish("hello")
    assert not ok and "400" in err           # 4xx (non-429) is not retried
    down = TelegramPublisher("TOK", "@c", retries=2, http=FakeHttp([500, 500]), sleep=lambda s: None)
    assert not down.publish("hello")[0]
    assert TelegramPublisher(None, None).publish("x") == (False, "telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL_ID)")


def test_telegram_photo_with_caption_or_followup_message():
    d = Path(tempfile.mkdtemp())
    img = render_card(BASE, "MARKET_SCAN", cfg(), d / "a.png")
    http = FakeHttp([200])
    ok, _ = TelegramPublisher("T", "@c", http=http, sleep=lambda s: None).publish("short", str(img))
    assert ok and http.calls[0][0].endswith("/sendPhoto") and http.calls[0][1]["caption"] == "short"
    http2 = FakeHttp([200, 200])
    ok, _ = TelegramPublisher("T", "@c", http=http2, sleep=lambda s: None).publish("L" * 1500, str(img))
    assert ok and [c[0].rsplit("/", 1)[1] for c in http2.calls] == ["sendPhoto", "sendMessage"]


def test_images_render_with_expected_size():
    from PIL import Image
    d = Path(tempfile.mkdtemp())
    for t in ("MARKET_SCAN", "ANOMALY_ALERT"):
        assert Image.open(render_card(BASE, t, cfg(), d / f"{t}.png")).size == (1200, 675)
    assert Image.open(render_summary(summarize([BASE]), cfg(), d / "s.png")).size == (1200, 675)
    render_card(r(market_score=None, similarity_score=None, probability_up=None), "MARKET_SCAN", cfg(), d / "na.png")


def test_schema_contract():
    out = finalize_result(BASE)
    assert all(k in out for k in REQUIRED_KEYS)
    bad = copy.deepcopy(BASE)
    bad["trend_score"] = 140
    try:
        finalize_result(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range score must be rejected")
    nan = copy.deepcopy(BASE)
    nan["expected_return"] = float("nan")
    assert finalize_result(nan)["expected_return"] is None


def test_scheduler_alignment_and_daemon_loop():
    now = datetime(2026, 1, 5, 12, 17, 33, tzinfo=timezone.utc)
    assert next_run(now, 60) == datetime(2026, 1, 5, 13, 0, 20, tzinfo=timezone.utc)

    class P:
        n = 0
        def scan(self):
            P.n += 1
            if P.n == 1:
                raise RuntimeError("boom")   # a failing cycle must not kill the daemon
        def publish_daily_summary(self):
            return True
    sleeps = []
    assert run_daemon(P(), cfg(), max_cycles=3, sleep=sleeps.append, clock=lambda: now) == 3 and len(sleeps) == 2


def test_edge_direction_skill_label_and_zero_move():
    from publishing.post_generator import strip_html
    down = strip_html(generate_post("MARKET_SCAN", r(probability_up=0.40, edge=0.051, expected_return=-0.000001), cfg()))
    assert "toward ↓" in down and "≈0.00%" in down and "-0.00%" not in down
    ns = strip_html(generate_post("MARKET_SCAN", r(model_skill=-0.03), cfg()))
    assert "no proven skill" in ns and "baseline 🔥" not in ns
    assert "no proven skill" not in strip_html(generate_post("MARKET_SCAN", BASE, cfg()))


def test_weekly_open_quiet_window_and_dst_hours():
    from utils.market_hours import is_market_open, just_reopened
    c = cfg()
    u = lambda *a: datetime(*a, tzinfo=timezone.utc)
    # October 2026 = US summer time: FX opens Sunday 21:00 UTC
    assert is_market_open("EURUSD", c, u(2026, 10, 11, 21, 5)) and not is_market_open("EURUSD", c, u(2026, 10, 11, 20, 55))
    assert just_reopened("EURUSD", c, u(2026, 10, 11, 21), 3) and just_reopened("EURUSD", c, u(2026, 10, 11, 23), 3)
    assert not just_reopened("EURUSD", c, u(2026, 10, 12, 0), 3) and not just_reopened("BTCUSD", c, u(2026, 10, 11, 21), 3)
    # January = winter time: opens 22:00 UTC
    assert not is_market_open("EURUSD", c, u(2026, 1, 11, 21, 30)) and is_market_open("EURUSD", c, u(2026, 1, 11, 22, 5))
