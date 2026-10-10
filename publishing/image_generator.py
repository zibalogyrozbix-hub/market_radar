"""1200x675 Pillow cards: minimal, dark, mobile-readable, branded from config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

from config.settings import display_name, resolve_path

W, H = 1200, 675
BG, PANEL, TXT, MUTED, GRID = (9, 13, 21), (17, 24, 36), (234, 240, 248), (128, 143, 165), (34, 46, 66)
RISK_COLOR = {"LOW": (80, 200, 140), "MEDIUM": (240, 190, 70), "HIGH": (240, 120, 60), "EXTREME": (235, 70, 80)}
TITLES = {"MARKET_SCAN": "AI MARKET SCAN", "ANOMALY_ALERT": "MARKET ANOMALY", "HISTORICAL_MATCH": "MARKET DNA MATCH",
          "REGIME_CHANGE": "REGIME CHANGE", "CROSS_MARKET_BREAK": "CORRELATION BREAK",
          "VOLATILITY_ALERT": "VOLATILITY ALERT", "DAILY_MARKET_SUMMARY": "DAILY SUMMARY", "MODEL_UPDATE": "MODEL UPDATE"}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _hex(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _bar(d: ImageDraw.ImageDraw, x, y, w, label, value, accent, f_lbl, f_val):
    d.text((x, y), label, font=f_lbl, fill=MUTED, anchor="lm")
    d.text((x + w, y), "N/A" if value is None else f"{value:.0f}", font=f_val, fill=TXT, anchor="rm")
    by = y + 22
    d.rounded_rectangle((x, by, x + w, by + 12), 6, fill=GRID)
    if value is not None:
        d.rounded_rectangle((x, by, x + max(12, w * min(value, 100) / 100), by + 12), 6, fill=accent)


def _fit(d, text: str, max_w: int, start: int = 44, bold: bool = True) -> ImageFont.FreeTypeFont:
    size = start
    while size > 16:
        f = _font(size, bold)
        if d.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return _font(16, bold)


def _tile(d, x, y, w, h, label, value, color, f_lbl, f_val=None):
    d.rounded_rectangle((x, y, x + w, y + h), 14, fill=PANEL)
    d.text((x + 18, y + 24), label, font=f_lbl, fill=MUTED, anchor="lm")
    d.text((x + 18, y + h - 34), value, font=_fit(d, value, w - 36), fill=color, anchor="lm")


def _frame(cfg: dict, title: str, subtitle: str):
    accent = _hex(cfg.get("branding", {}).get("accent_color", "#00E5A8"))
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 8, H), fill=accent)
    d.text((48, 44), cfg.get("branding", {}).get("logo_text", "AI MARKET RADAR"), font=_font(24, True), fill=accent, anchor="lm")
    d.text((W - 48, 44), title, font=_font(24, True), fill=MUTED, anchor="rm")
    return img, d, accent


def _footer(d, cfg):
    txt = cfg.get("publishing", {}).get("disclaimer", "Statistical market analysis. Not a trading signal.")
    d.text((48, H - 28), txt, font=_font(16), fill=MUTED, anchor="lm")


def render_card(result: Dict[str, Any], post_type: str, cfg: dict, out_path: Optional[Path] = None) -> Path:
    img, d, accent = _frame(cfg, TITLES.get(post_type, "AI MARKET SCAN"), "")
    f12, f16, f20, f28, f40, f72 = _font(18), _font(20), _font(24), _font(30, True), _font(44, True), _font(84, True)

    d.text((48, 118), display_name(cfg, result["symbol"]), font=_font(58, True), fill=TXT, anchor="lm")
    ts = str(result.get("timestamp", ""))[:16].replace("T", " ")
    d.text((48, 168), f"{result['timeframe']}  ·  {ts} UTC", font=f16, fill=MUTED, anchor="lm")

    # score ring
    cx, cy, r = 190, 340, 105
    d.arc((cx - r, cy - r, cx + r, cy + r), 0, 360, fill=GRID, width=20)
    ms = result.get("market_score")
    if ms is not None:
        d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * min(ms, 100) / 100, fill=accent, width=20)
    d.text((cx, cy - 14), "N/A" if ms is None else f"{ms:.0f}", font=f72, fill=TXT, anchor="mm")
    d.text((cx, cy + 46), "MARKET SCORE", font=_font(15), fill=MUTED, anchor="mm")

    # bars
    x0, bw = 400, 700
    rows = [("TREND", result.get("trend_score")), ("MOMENTUM", result.get("momentum_score")),
            ("VOLATILITY", result.get("volatility_score")), ("ANOMALY", result.get("anomaly_score")),
            ("SIMILARITY", result.get("similarity_score"))]
    for i, (lbl, v) in enumerate(rows):
        _bar(d, x0, 230 + i * 56, bw, lbl, v, accent, f16, f20)

    # bottom tiles
    ty, th, gap = 505, 112, 20
    tw = (W - 96 - 3 * gap) // 4
    p = result.get("probability_up")
    if result.get("edge_label") == "NO CLEAR EDGE" or p is None:
        t1, v1, c1 = "DIRECTIONAL EDGE", "NO CLEAR EDGE", MUTED
        t2, v2 = "EXPECTED MOVE", "—"
    else:
        raw = "" if result.get("probability_kind") == "calibrated" else " raw"
        t1, v1, c1 = "PROBABILITY ↑" + raw.upper(), f"{p * 100:.0f}%", TXT
        er = result.get("expected_return")
        t2, v2 = "EXPECTED MOVE", "N/A" if er is None else f"{er * 100:+.2f}%"
    risk = result.get("risk", "N/A")
    tiles = [(t1, v1, c1), (t2, v2, TXT), ("REGIME", str(result.get("regime", "N/A")).replace("_", " "), accent),
             ("RISK", risk, RISK_COLOR.get(risk, MUTED))]
    for i, (l, v, c) in enumerate(tiles):
        _tile(d, 48 + i * (tw + gap), ty, tw, th, l, v, c, f12)
    _footer(d, cfg)
    return _save(img, cfg, f"{result['symbol']}_{result['timeframe']}_{post_type}", out_path)


def render_summary(summary: Dict[str, Any], cfg: dict, out_path: Optional[Path] = None) -> Path:
    img, d, accent = _frame(cfg, "DAILY SUMMARY", "")
    f16, f20, f40, f58 = _font(20), _font(24), _font(44, True), _font(58, True)
    d.text((48, 118), "DAILY MARKET RADAR", font=f58, fill=TXT, anchor="lm")
    stats = [("MARKETS", summary.get("markets_scanned", 0)), ("ANOMALIES", summary.get("anomalies", 0)),
             ("REGIME CHANGES", summary.get("regime_changes", 0)), ("HIST. MATCHES", summary.get("strong_matches", 0))]
    for i, (l, v) in enumerate(stats):
        _tile(d, 48 + i * 281, 190, 261, 112, l, str(v), accent, _font(18))
    d.text((48, 345), "TOP MARKETS", font=_font(20, True), fill=MUTED, anchor="lm")
    for i, t in enumerate(summary.get("top", [])[:3]):
        y = 400 + i * 62
        d.text((48, y), f"{i + 1}", font=f40, fill=accent, anchor="lm")
        d.text((110, y), display_name(cfg, t["symbol"]), font=_font(34, True), fill=TXT, anchor="lm")
        d.rounded_rectangle((420, y - 9, 420 + 620, y + 9), 9, fill=GRID)
        d.rounded_rectangle((420, y - 9, 420 + max(18, 620 * t["score"] / 100), y + 9), 9, fill=accent)
        d.text((1150, y), f"{t['score']:.0f}", font=f40, fill=TXT, anchor="rm")
    _footer(d, cfg)
    return _save(img, cfg, "DAILY_SUMMARY", out_path)


def _save(img: Image.Image, cfg: dict, stem: str, out_path: Optional[Path]) -> Path:
    if out_path is None:
        out_dir = resolve_path(cfg, "data/images")
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        out_path = out_dir / f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=True)
    return out_path
