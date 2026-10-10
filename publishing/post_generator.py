"""Telegram post text generation (deterministic templates, no LLM, no signal language)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import display_name

LINE = "━━━━━━━━━━━━"
POST_TYPES = ["MARKET_SCAN", "ANOMALY_ALERT", "HISTORICAL_MATCH", "REGIME_CHANGE", "CROSS_MARKET_BREAK",
              "VOLATILITY_ALERT", "DAILY_MARKET_SUMMARY", "MODEL_UPDATE"]
REGIME_ICON = {"EXPANSION": "🔥", "BREAKOUT": "⚡", "REVERSAL": "🔄", "COMPRESSION": "🧊", "TREND_UP": "📈",
               "TREND_DOWN": "📉", "RANGE": "↔️", "HIGH_VOLATILITY": "🌪️", "LOW_VOLATILITY": "😴"}
FORBIDDEN = ["BUY NOW", "SELL NOW", "GUARANTEED", "PROFIT SIGNAL", "100% ACCURATE"]


def _n(v, fmt="{:.0f}", suffix="", na="N/A"):
    return na if v is None else fmt.format(v) + suffix


def _pct(v, signed=True, digits=2, na="N/A"):
    if v is None:
        return na
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def _score(v) -> str:
    return "N/A" if v is None else f"{v:.0f}/100"


def _prob_line(r: Dict[str, Any]) -> str:
    p = r.get("probability_up")
    if p is None:
        return "Probability ↑\nN/A"
    tag = "" if r.get("probability_kind") == "calibrated" else " (raw model probability)"
    return f"Probability ↑\n{p * 100:.0f}%{tag}"


def _disclaimer(cfg: dict, extra: str = "") -> str:
    base = cfg.get("publishing", {}).get("disclaimer", "Statistical market analysis. Not a trading signal.")
    return f"⚠️ {extra or base}"


def _header(r: Dict[str, Any], cfg: dict) -> str:
    return f"{display_name(cfg, r['symbol'])} · {r['timeframe']}"


def _edge_block(r: Dict[str, Any]) -> str:
    if r.get("edge_label") == "NO CLEAR EDGE":
        return "Statistical Edge\n🚫 NO CLEAR EDGE"
    if r.get("edge_label") == "N/A" or r.get("edge") is None:
        return "Statistical Edge\nN/A"
    return (f"{_prob_line(r)}\n\nExpected Move\n{_pct(r.get('expected_return'))}\n\n"
            f"Statistical Edge\n{r['edge'] * 100:+.1f} pp vs baseline")


def market_scan(r, cfg) -> str:
    icon = REGIME_ICON.get(r.get("regime"), "🔹")
    sim_states = "N/A" if not r.get("similar_states") else f"{r['similar_states']:,}"
    hist_up = _pct(r.get("historical_up_rate"), signed=False, digits=0)
    return "\n".join([
        "🧠 AI MARKET SCAN", "", _header(r, cfg), "", LINE, "",
        f"TREND        {_score(r.get('trend_score'))}",
        f"MOMENTUM     {_score(r.get('momentum_score'))}",
        f"VOLATILITY   {_score(r.get('volatility_score'))}",
        f"ANOMALY      {_score(r.get('anomaly_score'))}",
        f"SIMILARITY   {_score(r.get('similarity_score'))}", "", LINE, "",
        f"MARKET STATE\n{icon} {r.get('regime', 'N/A')}", "",
        _edge_block(r), "",
        f"Risk\n{r.get('risk', 'N/A')}", "",
        f"Model Agreement\n{r.get('model_agreement', 'N/A')}", "", LINE, "",
        f"Similar Historical States\n{sim_states}", "", f"Historical UP Rate\n{hist_up}", "",
        f"Market Intelligence Score\n{_score(r.get('market_score'))}", "", LINE, "",
        _disclaimer(cfg, "Statistical market analysis\nNot a guaranteed trading signal."),
    ])


def anomaly_alert(r, cfg) -> str:
    d = r.get("anomaly_deviations", {}) or {}
    comp = r.get("anomaly_component_scores", {}) or {}

    def sig(k):
        return "N/A" if d.get(k) is None else f"{d[k]:+.1f}σ"

    lines = ["🚨 MARKET ANOMALY DETECTED", "", display_name(cfg, r["symbol"]) + f" · {r['timeframe']}", "",
             f"Anomaly Score\n{_score(r.get('anomaly_score'))}", "", LINE, "",
             f"Volatility deviation\n{sig('z_vol')}", "", f"Price velocity\n{sig('z_velocity')}", "",
             f"Return deviation\n{sig('z_ret')}", "",
             f"Volume anomaly\n{_score(comp.get('VOLUME_ANOMALY'))}"]
    if r.get("correlation_break"):
        lines += ["", f"Correlation break\n{_score(comp.get('CORRELATION_BREAK'))}"]
    lines += ["", LINE, "", "Market State\n⚠️ UNUSUAL ACTIVITY"]
    if r.get("why"):
        lines += ["", "Why flagged"] + [f"{i}. {w}" for i, w in enumerate(r["why"][:4], 1)]
    lines += ["", _disclaimer(cfg)]
    return "\n".join(lines)


def historical_match(r, cfg) -> str:
    sim = r.get("similarity_details", {}) or {}
    return "\n".join([
        "🧬 MARKET DNA MATCH", "", _header(r, cfg), "",
        f"The current market state was compared with {r.get('similar_states', 0):,} nearest historical states "
        f"(out of {sim.get('history_size', 0):,} usable).", "", LINE, "",
        f"Similarity\n{_score(r.get('similarity_score'))}", "",
        f"Historical UP\n{_pct(r.get('historical_up_rate'), signed=False, digits=0)}", "",
        f"Median Future Move\n{_pct(sim.get('median_future_return'))}", "",
        f"P25\n{_pct(sim.get('p25'))}", "", f"P75\n{_pct(sim.get('p75'))}", "", LINE, "",
        f"Market State\n{REGIME_ICON.get(r.get('regime'), '🔹')} {r.get('regime', 'N/A')}", "",
        _disclaimer(cfg, "Historical similarity does not guarantee future results."),
    ])


def regime_change(r, cfg) -> str:
    return "\n".join([
        "⚡ MARKET REGIME CHANGE", "", display_name(cfg, r["symbol"]) + f" · {r['timeframe']}", "",
        f"Previous\n{r.get('previous_regime', 'N/A')}", "", f"Current\n{r.get('regime', 'N/A')}", "", LINE, "",
        f"Trend Strength\n{_score(r.get('trend_score'))}", "", f"Volatility\n{_score(r.get('volatility_score'))}", "",
        f"Momentum\n{_score(r.get('momentum_score'))}", "", f"Anomaly\n{_score(r.get('anomaly_score'))}", "", LINE, "",
        _edge_block(r), "", _disclaimer(cfg),
    ])


def cross_market_break(r, cfg) -> str:
    lines = ["🔗 CROSS-MARKET CORRELATION BREAK", "", display_name(cfg, r["symbol"]) + f" · {r['timeframe']}", "", LINE]
    for p in (r.get("cross_pairs") or [])[:3]:
        flag = "⚠️ BREAK" if p["break"] else "ok"
        lines += ["", f"{display_name(cfg, r['symbol'])} vs {display_name(cfg, p['peer'])}",
                  f"Normal {p['normal_corr']:+.2f}  →  Current {p['current_corr']:+.2f}   {flag}"]
    lines += ["", LINE, "", f"Divergence Score\n{_score(r.get('divergence_score'))}", "",
              f"Cross-market Score\n{_score(r.get('cross_market_score'))}", "", _disclaimer(cfg)]
    return "\n".join(lines)


def volatility_alert(r, cfg) -> str:
    return "\n".join([
        "🌪️ VOLATILITY ALERT", "", _header(r, cfg), "", LINE, "",
        f"Volatility Percentile\n{_score(r.get('volatility_score'))}", "",
        f"State\n{r.get('volatility_state', 'N/A')}", "", f"ATR Expansion\n{_n(r.get('atr_ratio'), '{:.2f}', 'x')}", "",
        f"Risk\n{r.get('risk', 'N/A')}", "", LINE, "", _disclaimer(cfg),
    ])


def daily_summary(s: Dict[str, Any], cfg: dict) -> str:
    medals = ["🥇", "🥈", "🥉"]
    lines = ["📊 AI MARKET RADAR", "DAILY SUMMARY", "", LINE, "",
             f"Markets scanned\n{s.get('markets_scanned', 0)}", "", f"Anomalies\n{s.get('anomalies', 0)}", "",
             f"Major regime changes\n{s.get('regime_changes', 0)}", "",
             f"Strong historical matches\n{s.get('strong_matches', 0)}", "",
             f"High-volatility assets\n{s.get('high_vol', 0)}", "", LINE, "", "TOP MARKETS"]
    for i, t in enumerate(s.get("top", [])[:3]):
        lines += ["", f"{medals[i]} {display_name(cfg, t['symbol'])}", f"Score: {t['score']:.0f}"]
    if s.get("top_unusual"):
        lines += ["", LINE, "", f"Most unusual market\n{display_name(cfg, s['top_unusual'])}"]
    track = s.get("track_record")
    if track and track.get("n"):
        lines += ["", LINE, "", f"Track record (last {track['n']} resolved)",
                  f"Brier {track['brier']:.3f} · hit rate {track['accuracy'] * 100:.0f}%"]
    lines += ["", LINE, "", "System-generated quantitative analysis.", _disclaimer(cfg)]
    return "\n".join(lines)


def model_update(u: Dict[str, Any], cfg: dict) -> str:
    sm = u.get("summary", {})
    skill = sm.get("brier_skill")
    verdict = "no measurable skill over baseline" if skill is None or skill <= 0 else f"Brier skill {skill * 100:+.1f}% vs baseline"
    return "\n".join([
        "🔧 MODEL UPDATE", "", f"{display_name(cfg, u['symbol'])} · {u['timeframe']}", "", LINE, "",
        f"Version\n{u.get('version')}", "", f"Status\n{'ACTIVATED' if u.get('activated') else 'KEPT PREVIOUS MODEL'}", "",
        f"Out-of-sample Brier\n{_n(sm.get('brier'), '{:.4f}')}", "", f"ROC-AUC\n{_n(sm.get('roc_auc'), '{:.3f}')}", "",
        f"Calibration error (ECE)\n{_n(sm.get('ece'), '{:.3f}')}", "", f"Result\n{verdict}", "", LINE, "",
        _disclaimer(cfg),
    ])


_BUILDERS = {"MARKET_SCAN": market_scan, "ANOMALY_ALERT": anomaly_alert, "HISTORICAL_MATCH": historical_match,
             "REGIME_CHANGE": regime_change, "CROSS_MARKET_BREAK": cross_market_break,
             "VOLATILITY_ALERT": volatility_alert, "DAILY_MARKET_SUMMARY": daily_summary, "MODEL_UPDATE": model_update}


def generate_post(post_type: str, data: Dict[str, Any], cfg: dict) -> str:
    if post_type not in _BUILDERS:
        raise ValueError(f"unknown post type {post_type}")
    text = _BUILDERS[post_type](data, cfg)
    up = text.upper()
    for negated in ("NOT A GUARANTEED", "DOES NOT GUARANTEE", "NO GUARANTEE"):  # honest disclaimers are allowed
        up = up.replace(negated, "")
    bad = [w for w in FORBIDDEN if w in up]
    if bad:  # defensive guard against signal-selling language
        raise ValueError(f"forbidden wording in post: {bad}")
    return text
