"""Which markets are open? Used to skip scans (and save API credits) when a market is closed.

Config (all optional):
  market_hours:
    always_open: [BTCUSD]        # 24/7 markets
    session_only: [US500, NAS100]  # US-session proxies: open on weekdays only (nightly gaps are normal)
    fx_close: {weekday: 4, hour: 22}   # Friday 22:00 UTC
    fx_open:  {weekday: 6, hour: 22}   # Sunday 22:00 UTC
Everything else (FX, metals) follows the FX week: closed from Friday 22:00 to Sunday 22:00 UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional


def _fx_week_closed(now: datetime, close_wd=4, close_h=22, open_wd=6, open_h=22) -> bool:
    """Minutes into the week (Mon 00:00 = 0) compared with the weekly close/open instants."""
    pos = now.weekday() * 1440 + now.hour * 60 + now.minute
    return close_wd * 1440 + close_h * 60 <= pos < open_wd * 1440 + open_h * 60


def _nth_sunday(year: int, month: int, n: int) -> int:
    first = datetime(year, month, 1).weekday()  # Mon=0
    return 1 + (6 - first) % 7 + 7 * (n - 1)


def us_dst(now: datetime) -> bool:
    """US daylight saving (2nd Sunday of March .. 1st Sunday of November) - FX rolls with New York 17:00."""
    y = now.year
    start = datetime(y, 3, _nth_sunday(y, 3, 2), tzinfo=timezone.utc)
    end = datetime(y, 11, _nth_sunday(y, 11, 1), tzinfo=timezone.utc)
    return start <= now < end


def _fx_hours(now: datetime, mh: dict):
    c, o = mh.get("fx_close") or {}, mh.get("fx_open") or {}
    auto = 21 if us_dst(now) else 22  # 17:00 New York = 21:00 UTC in summer, 22:00 UTC in winter
    return (int(c.get("weekday", 4)), int(c.get("hour", auto)), int(o.get("weekday", 6)), int(o.get("hour", auto)))


def just_reopened(symbol: str, cfg: dict, ts: datetime, hours: float = 3) -> bool:
    """True for the first `hours` after the weekly FX/metals open: gaps and wide spreads make those candles
    look like events when they are not."""
    mh = cfg.get("market_hours", {}) or {}
    if hours <= 0 or symbol in (mh.get("always_open") or ["BTCUSD"]) or symbol in (mh.get("session_only") or []):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    _cw, _ch, ow, oh = _fx_hours(ts, mh)
    days_back = (ts.weekday() - ow) % 7
    open_at = (ts - timedelta(days=days_back)).replace(hour=oh, minute=0, second=0, microsecond=0)
    return timedelta(0) <= ts - open_at < timedelta(hours=hours)


def hours_since_gap(index, min_gap_hours: float = 6, lookback: int = 200) -> Optional[float]:
    """Hours between the first candle after the latest gap (> `min_gap_hours`) and the newest candle, or None when
    there was no gap in the last `lookback` candles. Data-driven, so it does not depend on the provider's weekly-open
    convention or on DST rules (unlike `just_reopened`)."""
    try:
        import pandas as pd
        idx = pd.DatetimeIndex(index)[-lookback:]
        if len(idx) < 2:
            return None
        gaps = (idx[1:] - idx[:-1]) > pd.Timedelta(hours=min_gap_hours)
        if not gaps.any():
            return None
        first_after = idx[1:][gaps][-1]
        return float((idx[-1] - first_after) / pd.Timedelta(hours=1))
    except Exception:
        return None


def is_market_open(symbol: str, cfg: dict, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    mh = cfg.get("market_hours", {}) or {}
    if symbol in (mh.get("always_open") or ["BTCUSD"]):
        return True
    if symbol in (mh.get("session_only") or []):
        return now.weekday() < 5  # US-session instruments: no data on weekends (nightly gaps are handled by the provider)
    cw, ch, ow, oh = _fx_hours(now, mh)
    return not _fx_week_closed(now, cw, ch, ow, oh)
