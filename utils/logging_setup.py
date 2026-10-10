"""Logging: console + rotating file (logs/market_radar.log)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(cfg: dict | None = None, level: str | None = None) -> logging.Logger:
    global _CONFIGURED
    log = logging.getLogger("market_radar")
    if _CONFIGURED:
        if level:
            log.setLevel(level.upper())
        return log
    lcfg = (cfg or {}).get("logging", {})
    lvl = (level or lcfg.get("level") or "INFO").upper()
    log.setLevel(lvl)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    try:
        root = Path((cfg or {}).get("_root", "."))
        fpath = root / lcfg.get("file", "logs/market_radar.log")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(fpath, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as exc:  # read-only FS etc.: keep console logging
        log.warning("File logging disabled: %s", exc)
    log.propagate = False
    _CONFIGURED = True
    return log


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"market_radar.{name}")
