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
    assert "▱▱▱▱▱▱▱▱▱▱ N/A" in na and "similar" not in na.lower().replace("similarity", "")  # weak/absent similarity: no history line


def test_track_record_and_methods_lines_and_html_escape():
    from publishing.post_generator import strip_html
    t = strip_html(generate_post("MARKET_SCAN", r(track_record={"symbol": "XAUUSD", "n": 150, "n_eff": 37, "show": False}), cfg()))
    assert "Brier" not in t and "Track record" not in t  # hidden until meaningful
    t = strip_html(generate_post("MARKET_SCAN", r(track_record={"symbol": "XAUUSD", "n": 500, "n_eff": 125, "brier": 0.231,
                                                                 "base_brier": 0.246, "show": True},
                                                   methods_info={"models": ["logistic", "xgboost"], "calibration": "isotonic"}), cfg()))
    assert "Brier 0.231 vs baseline 0.246" in t and "≈125 independent" in t
    assert "Logistic + XGBoost" in t and "isotonic calibration" in t
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
    # a different regime / small change is NOT enough to break a cooldown ...
    small = prio.evaluate_event(r(anomaly_score=99.0, regime="BREAKOUT"), "ANOMALY_ALERT", c, last, False, now)
    assert not small["publish"] and "cooldown" in small["reason"]
    # ... a real intensification (+15 anomaly points or more) is
    last70 = {"time": now - timedelta(minutes=20), "signature": prio.signature(r(anomaly_score=70.0))}
    assert prio.evaluate_event(r(anomaly_score=99.0), "ANOMALY_ALERT", c, last70, False, now)["publish"]


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
    down = strip_html(generate_post("MARKET_SCAN", r(probability_up=0.40, edge=0.071, expected_return=-0.000001,
                                                     trend_direction="DOWN", momentum_direction="DOWN"), cfg()))
    assert "toward ↓" in down and "≈0 (within normal" in down and "-0.00%" not in down
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


# ---- consistency guard (list items 15-22) --------------------------------------------------------------
AUD = dict(symbol="AUDUSD", regime="TREND_DOWN", trend_score=57.0, trend_direction="DOWN", momentum_score=39.0,
           momentum_direction="DOWN", volatility_score=23.0, anomaly_score=0.0, similarity_score=43.0, market_score=32.0,
           probability_up=0.53, probability_kind="raw model probability", model_skill=-0.02, edge=0.03, edge_label="NO CLEAR EDGE",
           expected_return=0.0002, similar_states=100, historical_up_rate=0.55, risk="MEDIUM", model_agreement="MEDIUM",
           similarity_details={"history_size": 4000, "p25": -0.0008, "p75": 0.0009}, regime_changed=False)


def test_regression_aud_usd_downtrend_with_up_probability():
    from publishing.post_generator import strip_html
    t = strip_html(generate_post("MARKET_SCAN", r(**AUD), cfg()))
    assert "↑ 53% · ↓ 47%" in t and "no directional lean" not in t  # 3 pp from 50% is a (weak) lean, shown on both sides
    assert "Mixed signals: trend ↘ · momentum ↘, models lean ↑" in t
    assert "Statistical edge: none clear (mixed signals)" in t
    assert "≈0 (within normal ±0.09%)" in t and "+0.02%" not in t   # expected move is below the noise band
    assert "Similar past situations" not in t and "similar past situations" not in t  # similarity 43 < 60: no history line
    assert "weaker than the trend" in t  # momentum 39 < trend 57: the "fading" watch line is true here


def test_regression_gbp_usd_uptrend_with_down_lean_and_conditional_watch():
    from publishing.post_generator import strip_html
    g = dict(AUD, symbol="GBPUSD", regime="TREND_UP", trend_direction="UP", momentum_direction="UP", momentum_score=50.0,
             trend_score=45.0, probability_up=0.37)
    t = strip_html(generate_post("MARKET_SCAN", r(**g), cfg()))
    assert "↑ 37% · ↓ 63%" in t and "Mixed signals: trend ↗ · momentum ↗, models lean ↓" in t
    assert "fading" not in t  # momentum (50) is stronger than the trend (45) and in the same direction


def test_neutral_band_and_aligned_signals():
    from publishing.consistency import consistency_guard
    n = consistency_guard(r(probability_up=0.51), cfg())
    assert n["lean"]["neutral"] and n["alignment"] == "n/a"
    a = consistency_guard(r(**dict(AUD, probability_up=0.35)), cfg())
    assert a["lean"]["side"] == "DOWN" and a["alignment"] == "aligned"


def test_expected_move_shown_only_when_meaningful_and_same_side():
    from publishing.consistency import consistency_guard
    big = consistency_guard(r(**dict(AUD, probability_up=0.70, expected_return=0.0020)), cfg())
    assert big["move"]["show"] and big["move"]["text"] == "+0.20%"
    wrong_side = consistency_guard(r(**dict(AUD, probability_up=0.70, expected_return=-0.0020)), cfg())
    assert not wrong_side["move"]["show"]


def test_history_line_gated_and_coin_flip_label():
    from publishing.post_generator import strip_html
    t = strip_html(generate_post("MARKET_SCAN", r(similarity_score=80.0, historical_up_rate=0.51, similar_states=100), cfg()))
    assert "coin flip" in t
    t2 = strip_html(generate_post("MARKET_SCAN", r(similarity_score=80.0, historical_up_rate=0.70, similar_states=100), cfg()))
    assert "70% were higher" in t2 and "coin flip" not in t2


def test_volume_row_hidden_without_real_volume():
    from publishing.post_generator import strip_html
    assert "Volume" in strip_html(generate_post("ANOMALY_ALERT", BASE, cfg()))
    assert "Volume" not in strip_html(generate_post("ANOMALY_ALERT", r(has_volume=False), cfg()))


def test_image_card_uses_same_logic_as_text():
    from publishing.consistency import consistency_guard
    g = consistency_guard(r(**AUD), cfg())
    assert g["alignment"] == "mixed" and not g["move"]["show"]
    with tempfile.TemporaryDirectory() as d:
        assert render_card(r(**AUD), "MARKET_SCAN", cfg(), Path(d) / "a.png").exists()


# ---- priority / edge / regime (list items 1-6, 18) ------------------------------------------------------
def test_untrusted_edge_gets_no_priority_and_cannot_trigger_scan():
    raw_edge = r(edge_label="CLEAR EDGE", edge=0.12, market_score=30.0, model_skill=-0.02)
    assert prio.priority_score(raw_edge, 0.0, {"probability_edge": 1.0})["probability_edge"] == 0.0
    assert "MARKET_SCAN" not in prio.detect_candidate_events(raw_edge, cfg())
    good = r(edge_label="CLEAR EDGE", edge=0.12, market_score=30.0, model_skill=0.04, probability_kind="calibrated")
    assert prio.priority_score(good, 0.0, {"probability_edge": 1.0})["probability_edge"] == 0.8
    assert "MARKET_SCAN" in prio.detect_candidate_events(good, cfg())


def test_novelty_ignores_regime_code_but_scores_it_moderately():
    s1 = prio.signature(BASE)
    s2 = dict(s1, regime_code=s1["regime_code"] + 1)
    assert prio.novelty_score(s1, s2) == 0.6 and prio.novelty_score(None, s1) == 1.0 and prio.novelty_score(s1, s1) == 0.0


def test_regime_and_volatility_events_have_their_own_weights():
    c = cfg()
    reg = r(regime_changed=True, edge_label="NO CLEAR EDGE", anomaly_score=10.0, similarity_score=20.0, correlation_break=False)
    out = prio.evaluate_event(reg, "REGIME_CHANGE", c, None, False, datetime(2026, 1, 5, 13, tzinfo=timezone.utc))
    assert out["publish"]  # first-seen regime change reaches the threshold even without an ML edge
    vol = r(volatility_state="EXPANSION", volatility_score=96.0, anomaly_score=5.0, similarity_score=10.0, regime_changed=False,
            correlation_break=False)
    assert prio.evaluate_event(vol, "VOLATILITY_ALERT", c, None, False, datetime(2026, 1, 5, 13, tzinfo=timezone.utc))["publish"]


def test_edge_uses_half_as_floor_and_needs_distance_from_50():
    from scoring.scorer import compute_edge
    drift = compute_edge(0.53, 0.442, 0.07)  # asset drifted down 44%; a 53% up-probability is NOT an 8.8 pp edge
    assert drift["edge"] == 0.03 and drift["edge_label"] == "NO CLEAR EDGE"
    assert compute_edge(0.62, 0.50, 0.07)["edge_label"] == "CLEAR EDGE"
    assert compute_edge(0.30, 0.50, 0.07)["edge"] == 0.2  # down side measured against 50%, not the up base rate


def test_regime_change_needs_confirmation():
    import pandas as pd
    from engines import regime_engine

    class Seq(regime_engine.RegimeDetector):
        def __init__(self, seq):
            self.seq = seq

        def detect_series(self, feat):
            return pd.Series(self.seq[-len(feat):], index=feat.index, dtype=object)

    feat = pd.DataFrame({"x": range(10)})
    flicker = regime_engine.analyze(feat, Seq(["RANGE"] * 8 + ["BREAKOUT", "RANGE"]), confirm=2)
    assert not flicker["regime_changed"]
    one = regime_engine.analyze(feat, Seq(["RANGE"] * 9 + ["BREAKOUT"]), confirm=2)
    assert not one["regime_changed"]  # one candle is not confirmed yet
    held = regime_engine.analyze(feat, Seq(["RANGE"] * 8 + ["BREAKOUT", "BREAKOUT"]), confirm=2)
    assert held["regime_changed"] and held["previous_regime"] == "RANGE" and held["regime"] == "BREAKOUT"
    assert regime_engine.analyze(feat, Seq(["RANGE"] * 9 + ["BREAKOUT"]), confirm=1)["regime_changed"]  # old behaviour


def test_anomaly_second_term_comes_from_another_cluster():
    import numpy as np
    import pandas as pd
    from engines import anomaly_engine
    idx = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    base = {c: 0.0 for c in ("z_ret", "z_vol", "z_tickvol", "z_spread", "z_velocity")}
    feat = pd.DataFrame(base, index=idx)
    for c in anomaly_engine.FEATURE_COLUMNS:
        if c not in feat:
            feat[c] = np.random.default_rng(0).normal(size=len(idx))
    one_cluster = feat.copy()
    one_cluster.loc[idx[-1], ["z_ret", "z_velocity"]] = 4.0  # same family twice: must not be double counted
    two_cluster = feat.copy()
    two_cluster.loc[idx[-1], ["z_ret", "z_vol"]] = 4.0
    c = {"anomaly": {"use_isolation_forest": False}}
    assert anomaly_engine.analyze(one_cluster, c)["anomaly_score"] == 70.0
    assert anomaly_engine.analyze(two_cluster, c)["anomaly_score"] == 100.0


def test_hours_since_gap_detects_weekend_open_from_data():
    import pandas as pd
    from utils.market_hours import hours_since_gap
    # Friday 18:00..21:00 then Sunday 20:00.. (provider convention differs from our DST clock: data decides)
    idx = list(pd.date_range("2026-10-16 18:00", periods=4, freq="h", tz="UTC")) + \
        list(pd.date_range("2026-10-18 20:00", periods=6, freq="h", tz="UTC"))
    assert hours_since_gap(idx) == 5.0
    assert hours_since_gap(pd.date_range("2026-10-18 00:00", periods=50, freq="h", tz="UTC")) is None  # BTC-like: no gap


def test_true_range_ignores_weekend_gap_but_not_normal_candles():
    import pandas as pd
    from features import indicators as ind
    idx = list(pd.date_range("2026-10-16 20:00", periods=2, freq="h", tz="UTC")) + \
        list(pd.date_range("2026-10-18 21:00", periods=2, freq="h", tz="UTC"))
    df = pd.DataFrame({"high": [1.1010, 1.1012, 1.1060, 1.1062], "low": [1.1000, 1.1002, 1.1050, 1.1052],
                       "close": [1.1005, 1.1008, 1.1055, 1.1058]}, index=idx)
    tr = ind.true_range(df)
    assert abs(tr.iloc[2] - 0.0010) < 1e-9            # gap candle: only high-low, not the 0.0050 jump
    assert abs(tr.iloc[3] - 0.0010) < 1e-9
    assert ind.true_range(df, gap_hours=0).iloc[2] > 0.004  # old behaviour is still available


def test_rolling_and_type_caps_limit_bursts():
    """A burst of volatility alerts must not use the whole day budget; same-type alerts per run are limited."""
    import copy
    from datetime import datetime, timezone
    from pipeline import RadarPipeline
    from tests.helpers import cfg as _cfg

    c = copy.deepcopy(_cfg())
    c["publishing"].update({"max_posts_per_run": 5, "max_posts_per_day": 20, "rolling_window_hours": 6,
                            "max_posts_per_rolling_window": 6, "max_event_age_minutes": 0,
                            "max_posts_per_type_per_run": {"VOLATILITY_ALERT": 2}})

    class P(RadarPipeline):
        def __init__(self):  # no DB / loader needed
            self.cfg, self.rejected, self.recent = c, [], 5

        def _last_event(self, *a):
            return None

        def _event_known(self, h):
            return False

        def _caps_state(self, now, since=None):
            return {"_total": self.recent if since is not None else 5}

        def _log_rejected(self, ev, reason):
            self.rejected.append(reason)

    def res(sym):
        return {"symbol": sym, "timeframe": "H1", "timestamp": datetime.now(timezone.utc).isoformat(), "why": [],
                "volatility_state": "EXPANSION", "volatility_score": 99, "anomaly_score": 30, "anomaly_detected": False,
                "similarity_score": 10, "cross_market_score": 0, "market_score": 30, "regime": "EXPANSION",
                "regime_changed": False, "edge_label": "NO CLEAR EDGE", "probability_up": 0.5}

    p = P()
    out = p.rank_events([res(s) for s in ("EURUSD", "GBPUSD", "EURGBP")])
    assert len(out) == 1                                   # rolling window: 6 - 5 already sent = 1 free slot
    assert any("rolling cap" in r for r in p.rejected)
    p.recent = 0
    p.rejected.clear()
    out = p.rank_events([res(s) for s in ("EURUSD", "GBPUSD", "EURGBP")])
    assert len(out) == 2 and any("cap per run" in r for r in p.rejected)  # type cap: 2 VOLATILITY_ALERT per run
