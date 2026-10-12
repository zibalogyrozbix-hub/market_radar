"""Telegram post text generation (deterministic templates, no LLM, no signal language).

Output is Telegram-HTML (parse_mode=HTML): bold headline, <code> for aligned bars, and hyperlinks on *labels*
(never on numbers) that point to sections of the public reader's guide.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from config.settings import display_name

POST_TYPES = ["MARKET_SCAN", "ANOMALY_ALERT", "HISTORICAL_MATCH", "REGIME_CHANGE", "CROSS_MARKET_BREAK",
              "VOLATILITY_ALERT", "DAILY_MARKET_SUMMARY", "MODEL_UPDATE"]
FORBIDDEN = ["BUY NOW", "SELL NOW", "GUARANTEED", "PROFIT SIGNAL", "100% ACCURATE"]

DEFAULT_GUIDE_URL = "https://telegra.ph/FXSCAN--User-Guide-10-10"
# Telegraph builds heading anchors from the heading text (spaces -> "-"); keep the guide headings in sync.
# Each target heading is preceded by a small "Section ..." marker heading so the scroll lands just above the real title.
ANCHORS = {"probability": "Section-Probability", "edge": "Section-Statistical-edge", "risk": "Section-Risk",
           "agreement": "Section-Model-agreement", "similarity": "Section-Similarity", "state": "Section-Market-states",
           "usage": "Section-Using-it-in-your-trading"}
DEFAULT_DISCLAIMER = "Statistical analysis · not financial advice"
BUTTON_LABEL = "🧭 How do I use this?"

# regime -> (emoji, title, one-sentence meaning, what to watch)
REGIMES = {
    "COMPRESSION": ("🧊", "Volatility squeeze",
                    "Very low volatility; a larger move often follows, but its direction is not predicted.",
                    "A break of the recent range with rising volatility."),
    "EXPANSION": ("🔥", "Volatility expansion",
                  "Price is moving faster than usual; swings are larger than normal.",
                  "Whether volatility keeps rising or fades back."),
    "BREAKOUT": ("⚡", "Breakout",
                 "Price has left its recent range with unusual force.",
                 "Follow-through on the next candles, or a fall back into the range."),
    "REVERSAL": ("🔄", "Possible reversal",
                 "The prevailing direction is losing strength and the opposite pressure is building.",
                 "Whether momentum confirms the turn."),
    "TREND_UP": ("📈", "Uptrend",
                 "Price is making sustained progress upward.", "Momentum fading, or volatility rising against the trend."),
    "TREND_DOWN": ("📉", "Downtrend",
                   "Price is making sustained progress downward.", "Momentum fading, or volatility rising against the trend."),
    "RANGE": ("↔️", "Range-bound",
              "Price is moving sideways without a clear direction.", "A decisive move out of the range."),
    "HIGH_VOLATILITY": ("🌪️", "High volatility",
                        "Large, fast swings; conditions are unstable and risk is elevated.", "Whether volatility cools down."),
    "LOW_VOLATILITY": ("😴", "Quiet market",
                       "Small, slow moves; little is happening right now.", "A sudden pick-up in volatility."),
}
REGIME_ICON = {k: v[0] for k, v in REGIMES.items()}
_LIGHT = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}
_AGREE = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠"}


# ---------------------------------------------------------------- helpers
def _e(x) -> str:
    return html.escape(str(x), quote=False)


def _n(v, fmt="{:.0f}", suffix="", na="N/A"):
    return na if v is None else fmt.format(v) + suffix


def _pct(v, signed=True, digits=2, na="N/A"):
    if v is None:
        return na
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def _move(v) -> str:
    if v is None:
        return "N/A"
    return "≈0.00%" if abs(v) * 100 < 0.005 else f"{v * 100:+.2f}%"


def _score(v) -> str:
    return "N/A" if v is None else f"{v:.0f}/100"


def _bar(v, arrow: str = "") -> str:
    if v is None:
        return "▱" * 10 + " N/A"
    k = max(0, min(10, int(round(v / 10.0))))
    return "▰" * k + "▱" * (10 - k) + f" {v:.0f}" + (f" {arrow}" if arrow else "")


def _arrow(direction) -> str:
    d = str(direction or "").upper()
    return "↗" if d in ("UP", "BULLISH", "LONG") else "↘" if d in ("DOWN", "BEARISH", "SHORT") else "→" if d else ""


def _links(cfg: dict):
    pc = cfg.get("publishing", {})
    base = pc.get("guide_url", DEFAULT_GUIDE_URL)
    anchors = {**ANCHORS, **(pc.get("guide_anchors") or {})}

    def link(key: str, text: str) -> str:
        if not base or not pc.get("link_labels", True):
            return _e(text)
        return f'<a href="{_e(base)}#{anchors[key]}">{_e(text)}</a>'
    return link


def guide_usage_url(cfg: dict) -> str:
    pc = cfg.get("publishing", {})
    anchors = {**ANCHORS, **(pc.get("guide_anchors") or {})}
    return f"{pc.get('guide_url', DEFAULT_GUIDE_URL)}#{anchors['usage']}"


def _disclaimer(cfg: dict) -> str:
    return "⚠️ " + _e(cfg.get("publishing", {}).get("disclaimer") or DEFAULT_DISCLAIMER)


def _price(r: Dict[str, Any]) -> str:
    p, sym = r.get("price"), str(r.get("symbol", ""))
    if p is None:
        return "N/A"
    if "JPY" in sym:
        d = 3
    elif len(sym) == 6 and sym.isalpha() and sym[:3] not in ("XAU", "XAG", "BTC", "ETH"):
        d = 5
    else:
        d = 2
    return f"{p:,.{d}f}"


def _clock(ts) -> str:
    m = re.search(r"T(\d\d:\d\d)", str(ts or ""))
    return f"{m.group(1)} UTC" if m else ""


def _meta(r: Dict[str, Any]) -> str:
    parts = [str(r.get("timeframe", ""))]
    if _clock(r.get("timestamp")):
        parts.append(f"candle {_clock(r['timestamp'])}")
    parts.append(_price(r))
    if r.get("horizon"):
        parts.append(f"horizon {r['horizon']}h" if str(r.get("timeframe")) == "H1" else f"horizon {r['horizon']} candles")
    return _e(" · ".join(p for p in parts if p))


def _hashtags(r: Dict[str, Any], kind: str, state: Optional[str] = None) -> str:
    tags = ["#" + re.sub(r"[^A-Za-z0-9]", "", str(r.get("symbol", "")))[:12], "#" + kind]
    st = state if state is not None else r.get("regime")
    if st:
        tags.append("#" + "".join(w.capitalize() for w in str(st).split("_")))
    return " ".join(tags)


def _h(r, cfg, head: str) -> str:
    return f"<b>{_e(head)}</b>\n{_meta(r)}"


def _horizon(r) -> str:
    return f"{r['horizon']}h" if r.get("horizon") and str(r.get("timeframe")) == "H1" else \
        (f"{r['horizon']} candles" if r.get("horizon") else "next candles")


def _regime_headline(r, cfg) -> str:
    sym = display_name(cfg, r["symbol"])
    reg = r.get("regime")
    ic, title, _m, _w = REGIMES.get(reg, ("🔹", str(reg or "N/A").replace("_", " ").title(), "", ""))
    extra = ""
    vs = r.get("volatility_score")
    if reg == "COMPRESSION" and vs is not None:
        extra = f"quietest {max(1, round(vs))}% of recent history"
    elif reg in ("EXPANSION", "HIGH_VOLATILITY") and vs is not None:
        extra = f"more volatile than {round(vs)}% of recent history"
    elif reg == "LOW_VOLATILITY" and vs is not None:
        extra = f"quieter than {round(100 - vs)}% of recent history"
    return f"{ic} {sym} · {title}" + (f" — {extra}" if extra else "")


def _state_block(r, cfg, link) -> str:
    reg = r.get("regime")
    ic, _t, meaning, _w = REGIMES.get(reg, ("🔹", "", "", ""))
    out = f"{ic} <b>{link('state', str(reg or 'N/A').replace('_', ' '))}</b>"
    return out + (f"\n<i>{_e(meaning)}</i>" if meaning else "")


def _edge_lines(r, link) -> List[str]:
    """Probability / edge / expected move. Always shown; the edge is only highlighted when it is calibrated
    and the model has shown measurable skill."""
    p = r.get("probability_up")
    raw = r.get("probability_kind") != "calibrated"
    skill = r.get("model_skill")
    noskill = skill is None or skill <= 0
    hz = _horizon(r)
    if p is None:
        return [f"{link('probability', 'Probability')} ↑ over {hz}: N/A"]
    tags = (["raw model"] if raw else []) + (["no proven skill"] if noskill else [])
    lines = [f"{link('probability', 'Probability')} ↑ over {hz}: <b>{p * 100:.0f}%</b>" + (f" ({' · '.join(tags)})" if tags else "")]
    edge = r.get("edge")
    side = "↓" if p < 0.5 else "↑"
    if r.get("edge_label") == "NO CLEAR EDGE":
        lines.append(f"{link('edge', 'Statistical edge')}: none clear" + (f" ({edge * 100:+.1f} pp)" if edge is not None else ""))
    elif edge is None or r.get("edge_label") == "N/A":
        lines.append(f"{link('edge', 'Statistical edge')}: N/A")
    else:
        mark = "" if (raw or noskill) else (" 🔥" if edge * 100 >= 10 else " ✅" if edge * 100 >= 5 else "")
        lines.append(f"{link('edge', 'Statistical edge')}: {edge * 100:+.1f} pp toward {side} vs baseline{mark}")
    lines.append(f"<blockquote>Expected move: {_move(r.get('expected_return'))}</blockquote>")  # quoted for visibility
    return lines


def _risk_line(r, link) -> str:
    rk, ag = r.get("risk"), r.get("model_agreement")
    return (f"{_LIGHT.get(rk, '⚪')} {link('risk', 'Risk')} {_e(rk or 'N/A')}\n"
            f"{_AGREE.get(ag, '⚪')} {link('agreement', 'Model Agreement')} {_e(ag or 'N/A')}")


def _history_line(r, link) -> str:
    n, up = r.get("similar_states"), r.get("historical_up_rate")
    if not n or up is None:
        return "🧬 Similar past situations: N/A"
    sim = r.get("similarity_details") or {}
    s = f"🧬 Of {n:,} {link('similarity', 'similar')} past situations, {up * 100:.0f}% were higher {_horizon(r)} later"
    if sim.get("p25") is not None and sim.get("p75") is not None:
        s += f"\n    range of outcomes (P25–P75): {_pct(sim['p25'])} … {_pct(sim['p75'])}"
    return s


def _track_line(r) -> str:
    t = r.get("track_record") or {}
    n = int(t.get("n") or 0)
    if n >= 100 and t.get("brier") is not None:
        return f"📈 Last {n} resolved: Brier {t['brier']:.3f} (0.250 = coin flip)"
    return f"📈 Track record: building ({n}/100)"


_MODEL_NAMES = {"logistic": "Logistic", "random_forest": "Random Forest", "rf": "Random Forest", "xgboost": "XGBoost",
                "xgb": "XGBoost"}


def _methods(r, kinds: List[str]) -> str:
    info = r.get("methods_info") or {}
    parts = []
    if "ml" in kinds and info.get("models"):
        parts.append(" + ".join(_MODEL_NAMES.get(str(m).lower(), str(m)) for m in info["models"]))
        if info.get("calibration"):
            parts.append(f"{info['calibration']} calibration")
    if "anomaly" in kinds:
        parts.append("z-scores" + (" + Isolation Forest" if info.get("isolation_forest", True) else ""))
    if "similarity" in kinds:
        parts.append("nearest-neighbour history search")
    if "cross" in kinds:
        parts.append("rolling correlation (short vs long window)")
    if "walkforward" in kinds:
        parts.append("walk-forward validation + calibration")
    if "rules" in kinds:
        parts.append("rule-based regime classifier")
    return ("🔬 " + _e(" · ".join(parts))) if parts else ""


def _join(*blocks: str) -> str:
    return "\n\n".join(b for b in blocks if b)


def _bars(r) -> str:
    rows = [("TREND", _bar(r.get("trend_score"), _arrow(r.get("trend_direction")))),
            ("MOMENTUM", _bar(r.get("momentum_score"), _arrow(r.get("momentum_direction") or r.get("momentum_bias")))),
            ("VOLATILITY", _bar(r.get("volatility_score"))),
            ("ANOMALY", _bar(r.get("anomaly_score"))),
            ("SIMILARITY", _bar(r.get("similarity_score"))),
            ("MKT SCORE", _bar(r.get("market_score")))]
    return "<code>" + "\n".join(f"{k:<10} {v}" for k, v in rows) + "</code>"


# ---------------------------------------------------------------- post types
def market_scan(r, cfg) -> str:
    link = _links(cfg)
    reg = r.get("regime")
    watch = REGIMES.get(reg, ("", "", "", ""))[3]
    return _join(
        _h(r, cfg, _regime_headline(r, cfg)),
        _bars(r),
        _state_block(r, cfg, link),
        "\n".join(_edge_lines(r, link)),
        _risk_line(r, link),
        _history_line(r, link),
        f"👀 Watch: {_e(watch)}" if watch else "",
        "\n".join(x for x in (_track_line(r), _methods(r, ["ml", "similarity"])) if x),
        _disclaimer(cfg) + "\n" + _hashtags(r, "MarketScan"),
    )


def anomaly_alert(r, cfg) -> str:
    link = _links(cfg)
    d = r.get("anomaly_deviations", {}) or {}
    comp = r.get("anomaly_component_scores", {}) or {}

    def sig(k):
        return "N/A" if d.get(k) is None else f"{d[k]:+.1f}σ"

    rows = [("Volatility", sig("z_vol")), ("Velocity", sig("z_velocity")), ("Return", sig("z_ret")),
            ("Volume", _score(comp.get("VOLUME_ANOMALY")))]
    if r.get("correlation_break"):
        rows.append(("Corr.break", _score(comp.get("CORRELATION_BREAK"))))
    body = "<code>" + "\n".join([f"{'ANOMALY':<10} {_bar(r.get('anomaly_score'))}"] +
                                [f"{k:<10} {v}" for k, v in rows]) + "</code>"
    why = ""
    if r.get("why"):
        why = "<b>Why flagged</b>\n" + "\n".join(f"{i}. {_e(w)}" for i, w in enumerate(r["why"][:4], 1))
    return _join(
        _h(r, cfg, f"🚨 {display_name(cfg, r['symbol'])} · Unusual activity — score {_n(r.get('anomaly_score'))}/100"),
        body,
        "⚠️ <b>UNUSUAL ACTIVITY</b>\n<i>Price behaviour is far from its recent normal. This flags unusualness, not direction.</i>",
        why,
        "\n".join(_edge_lines(r, link)) if r.get("probability_up") is not None else "",
        _risk_line(r, link),
        "\n".join(x for x in (_track_line(r), _methods(r, ["anomaly", "ml"])) if x),
        _disclaimer(cfg) + "\n" + _hashtags(r, "Anomaly", "Unusual"),
    )


def historical_match(r, cfg) -> str:
    link = _links(cfg)
    sim = r.get("similarity_details", {}) or {}
    rows = [("Similarity", _score(r.get("similarity_score"))),
            ("Hist. UP", _pct(r.get("historical_up_rate"), signed=False, digits=0)),
            ("Median", _pct(sim.get("median_future_return"))), ("P25", _pct(sim.get("p25"))), ("P75", _pct(sim.get("p75")))]
    return _join(
        _h(r, cfg, f"🧬 {display_name(cfg, r['symbol'])} · DNA match — similarity {_n(r.get('similarity_score'))}/100"),
        f"The current market state was compared with its {r.get('similar_states', 0):,} nearest "
        f"{link('similarity', 'historical states')} (of {sim.get('history_size', 0):,} usable).",
        "<code>" + "\n".join(f"{k:<10} {v}" for k, v in rows) + "</code>",
        _state_block(r, cfg, link),
        _history_line(r, link),
        "<i>Historical similarity does not guarantee future results.</i>",
        _methods(r, ["similarity"]),
        _disclaimer(cfg) + "\n" + _hashtags(r, "DNAMatch"),
    )


def regime_change(r, cfg) -> str:
    link = _links(cfg)
    prev = str(r.get("previous_regime", "N/A")).replace("_", " ")
    cur = str(r.get("regime", "N/A")).replace("_", " ")
    bars = "<code>" + "\n".join(f"{k:<10} {v}" for k, v in (
        ("TREND", _bar(r.get("trend_score"), _arrow(r.get("trend_direction")))),
        ("MOMENTUM", _bar(r.get("momentum_score"), _arrow(r.get("momentum_direction") or r.get("momentum_bias")))),
        ("VOLATILITY", _bar(r.get("volatility_score"))), ("ANOMALY", _bar(r.get("anomaly_score"))))) + "</code>"
    return _join(
        _h(r, cfg, f"⚡ {display_name(cfg, r['symbol'])} · Regime change: {prev} → {cur}"),
        bars,
        _state_block(r, cfg, link),
        "\n".join(_edge_lines(r, link)),
        _risk_line(r, link),
        _methods(r, ["rules", "ml"]),
        _disclaimer(cfg) + "\n" + _hashtags(r, "RegimeChange"),
    )


def cross_market_break(r, cfg) -> str:
    link = _links(cfg)
    pairs = (r.get("cross_pairs") or [])[:3]
    sym = display_name(cfg, r["symbol"])
    lines = []
    for p in pairs:
        flag = "⚠️ BREAK" if p["break"] else "ok"
        lines.append(f"{_e(sym)} vs {_e(display_name(cfg, p['peer']))}\n"
                     f"<code>normal {p['normal_corr']:+.2f} → now {p['current_corr']:+.2f}  {flag}</code>")
    head = f"🔗 {sym} · Correlation break" + (f" vs {display_name(cfg, pairs[0]['peer'])}" if pairs else "")
    return _join(
        _h(r, cfg, head),
        "\n".join(lines),
        "<code>" + f"{'DIVERGENCE':<10} {_bar(r.get('divergence_score'))}\n{'CROSS-MKT':<10} {_bar(r.get('cross_market_score'))}" + "</code>",
        "<i>Markets that usually move together are behaving differently. This flags a change, not a direction.</i>",
        _risk_line(r, link),
        _methods(r, ["cross"]),
        _disclaimer(cfg) + "\n" + _hashtags(r, "CrossMarket", "Divergence"),
    )


def volatility_alert(r, cfg) -> str:
    link = _links(cfg)
    st = r.get("volatility_state", "N/A")
    return _join(
        _h(r, cfg, f"🌪️ {display_name(cfg, r['symbol'])} · Volatility {str(st).lower()}"),
        "<code>" + f"{'VOLATILITY':<10} {_bar(r.get('volatility_score'))}\n{'ATR RATIO':<10} {_n(r.get('atr_ratio'), '{:.2f}', 'x')}" + "</code>",
        "<i>Swings are larger than usual: expect wider moves and be careful with position size.</i>",
        _risk_line(r, link),
        _methods(r, ["rules"]),
        _disclaimer(cfg) + "\n" + _hashtags(r, "Volatility", st),
    )


def daily_summary(s: Dict[str, Any], cfg: dict) -> str:
    medals = ["🥇", "🥈", "🥉"]
    rows = [("Markets scanned", s.get("markets_scanned", 0)), ("Anomalies", s.get("anomalies", 0)),
            ("Regime changes", s.get("regime_changes", 0)), ("Strong matches", s.get("strong_matches", 0)),
            ("High-vol assets", s.get("high_vol", 0))]
    top = [f"{medals[i]} {_e(display_name(cfg, t['symbol']))} · score {t['score']:.0f}"
           for i, t in enumerate(s.get("top", [])[:3])]
    unusual = (f"🚨 Most unusual: {_e(display_name(cfg, s['top_unusual']))}" if s.get("top_unusual") else "")
    t = s.get("track_record") or {}
    track = (f"📈 Last {t['n']} resolved: Brier {t['brier']:.3f} · hit rate {t['accuracy'] * 100:.0f}% (Brier 0.250 = coin flip)"
             if t.get("n") and t.get("brier") is not None and t.get("accuracy") is not None else "")
    return _join(
        "<b>📊 Daily market summary</b>",
        "<code>" + "\n".join(f"{k:<16} {v}" for k, v in rows) + "</code>",
        ("<b>Top markets</b>\n" + "\n".join(top)) if top else "", unusual, track,
        "<i>System-generated quantitative analysis.</i>",
        _disclaimer(cfg) + "\n#DailySummary #MarketRadar",
    )


def model_update(u: Dict[str, Any], cfg: dict) -> str:
    sm = u.get("summary", {})
    skill = sm.get("brier_skill")
    verdict = "no measurable skill over baseline" if skill is None or skill <= 0 else f"Brier skill {skill * 100:+.1f}% vs baseline"
    rows = [("Version", u.get("version")), ("Status", "ACTIVATED" if u.get("activated") else "KEPT PREVIOUS"),
            ("Brier (OOS)", _n(sm.get("brier"), "{:.4f}")), ("ROC-AUC", _n(sm.get("roc_auc"), "{:.3f}")),
            ("ECE", _n(sm.get("ece"), "{:.3f}"))]
    return _join(
        f"<b>🔧 {_e(display_name(cfg, u['symbol']))} · Model update</b>\n{_e(u['timeframe'])}",
        "<code>" + "\n".join(f"{k:<12} {_e(v)}" for k, v in rows) + "</code>",
        f"Result: {_e(verdict)}",
        _methods(u, ["walkforward"]),
        _disclaimer(cfg) + "\n" + _hashtags(u, "ModelUpdate", "") ,
    )


_BUILDERS = {"MARKET_SCAN": market_scan, "ANOMALY_ALERT": anomaly_alert, "HISTORICAL_MATCH": historical_match,
             "REGIME_CHANGE": regime_change, "CROSS_MARKET_BREAK": cross_market_break,
             "VOLATILITY_ALERT": volatility_alert, "DAILY_MARKET_SUMMARY": daily_summary, "MODEL_UPDATE": model_update}


def visible_length(text: str) -> int:
    """Length Telegram counts against the caption limit (tags removed, entities decoded)."""
    return len(html.unescape(re.sub(r"<[^>]+>", "", text)))


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def generate_post(post_type: str, data: Dict[str, Any], cfg: dict) -> str:
    if post_type not in _BUILDERS:
        raise ValueError(f"unknown post type {post_type}")
    text = _BUILDERS[post_type](data, cfg)
    up = strip_html(text).upper()
    for negated in ("NOT A GUARANTEED", "DOES NOT GUARANTEE", "NO GUARANTEE"):  # honest disclaimers are allowed
        up = up.replace(negated, "")
    bad = [w for w in FORBIDDEN if w in up]
    if bad:  # defensive guard against signal-selling language
        raise ValueError(f"forbidden wording in post: {bad}")
    return text
