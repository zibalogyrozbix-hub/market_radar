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

from datetime import datetime, timezone
from typing import Optional


def _fx_week_closed(now: datetime, close_wd=4, close_h=22, open_wd=6, open_h=22) -> bool:
    """Minutes into the week (Mon 00:00 = 0) compared with the weekly close/open instants."""
    pos = now.weekday() * 1440 + now.hour * 60 + now.minute
    return close_wd * 1440 + close_h * 60 <= pos < open_wd * 1440 + open_h * 60


def is_market_open(symbol: str, cfg: dict, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    mh = cfg.get("market_hours", {}) or {}
    if symbol in (mh.get("always_open") or ["BTCUSD"]):
        return True
    if symbol in (mh.get("session_only") or []):
        return now.weekday() < 5  # US-session instruments: no data on weekends (nightly gaps are handled by the provider)
    c, o = mh.get("fx_close", {}), mh.get("fx_open", {})
    return not _fx_week_closed(now, int(c.get("weekday", 4)), int(c.get("hour", 22)),
                               int(o.get("weekday", 6)), int(o.get("hour", 22)))
