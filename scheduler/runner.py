"""Daemon loop: scheduled scan + daily summary. Also usable one-shot from cron/Docker/Actions."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from utils.logging_setup import get_logger

log = get_logger("scheduler")


def next_run(now: datetime, every_minutes: int, buffer_seconds: int = 20) -> datetime:
    """Next multiple of `every_minutes` (aligned to the clock) plus a small buffer so the candle has closed."""
    base = now.replace(second=0, microsecond=0)
    minutes_since_midnight = base.hour * 60 + base.minute
    step = every_minutes - (minutes_since_midnight % every_minutes)
    return base + timedelta(minutes=step, seconds=buffer_seconds)


def run_daemon(pipeline, cfg: dict, max_cycles: Optional[int] = None,
               sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    pcfg = cfg.get("publishing", {})
    every = int(pcfg.get("scheduled_scan_every_minutes", 60))
    summary_hour = int(pcfg.get("daily_summary_hour_utc", 20))
    last_summary_day = None
    cycles = 0
    log.info("Daemon started: scan every %d min, daily summary at %02d:00 UTC", every, summary_hour)
    while max_cycles is None or cycles < max_cycles:
        try:
            pipeline.scan()
            now = clock()
            if now.hour >= summary_hour and last_summary_day != now.date():
                if pipeline.publish_daily_summary():
                    last_summary_day = now.date()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # the daemon must survive anything
            log.exception("cycle failed: %s", exc)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        wake = next_run(clock(), every)
        delay = max(1.0, (wake - clock()).total_seconds())
        log.info("Next cycle at %s UTC (in %.0fs)", wake.strftime("%H:%M:%S"), delay)
        sleep(delay)
    return cycles
