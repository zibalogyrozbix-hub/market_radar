"""Provider selection + cached multi-symbol loading with validation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import get_secrets, resolve_path
from data.providers import CSVProvider, DataProvider, MT5Provider, SyntheticProvider, TwelveDataProvider
from data.validation import ValidationReport, validate_candles
from utils.logging_setup import get_logger
from utils.market_hours import is_market_open

log = get_logger("loader")


def make_provider(cfg: dict) -> DataProvider:
    choice = cfg.get("data", {}).get("provider", "auto").lower()
    secrets = get_secrets()
    if choice == "twelvedata" or (choice == "auto" and secrets.get("twelvedata_api_key")):
        try:
            return make_twelvedata(cfg)
        except Exception as exc:
            if choice == "twelvedata":
                raise
            log.info("Twelve Data unavailable (%s); falling back", exc)
    if choice in ("mt5", "auto"):
        try:
            prov = MT5Provider(secrets)
            log.info("Data provider: MetaTrader5")
            return prov
        except Exception as exc:
            if choice == "mt5":
                raise
            log.info("MT5 unavailable (%s); falling back", exc)
    if choice in ("csv", "auto"):
        csv_dir = resolve_path(cfg, cfg.get("data", {}).get("csv_dir", "data/csv"))
        if csv_dir.exists() and any(csv_dir.glob("*.csv")):
            log.info("Data provider: CSV (%s)", csv_dir)
            return CSVProvider(csv_dir)
        if choice == "csv":
            raise FileNotFoundError(f"No CSV files in {csv_dir}")
    log.warning("Data provider: SYNTHETIC data (demo only; not real market data)")
    return SyntheticProvider()


def make_twelvedata(cfg: dict) -> TwelveDataProvider:
    tcfg = cfg.get("twelvedata", {}) or {}
    prov = TwelveDataProvider(
        get_secrets().get("twelvedata_api_key") or "", symbol_map=tcfg.get("symbol_map"),
        requests_per_minute=int(tcfg.get("requests_per_minute", 8)), max_retries=int(tcfg.get("max_retries", 3)),
        max_stale_minutes=int(tcfg.get("max_stale_minutes", 0)),
        session_symbols=cfg.get("data", {}).get("session_symbols", []),
        market_open=lambda s: is_market_open(s, cfg))
    log.info("Data provider: Twelve Data (%d mapped symbols, %d req/min)", len(prov.symbol_map), prov.rpm)
    return prov


class MarketDataLoader:
    """Fetches and validates candles once per (symbol, timeframe) per run."""

    def __init__(self, cfg: dict, provider: Optional[DataProvider] = None):
        self.cfg = cfg
        self.provider = provider or make_provider(cfg)
        self._cache: Dict[Tuple[str, str], Tuple[Optional[pd.DataFrame], ValidationReport]] = {}

    def get(self, symbol: str, timeframe: str) -> Tuple[Optional[pd.DataFrame], ValidationReport]:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key]
        dcfg = self.cfg.get("data", {})
        try:
            raw = self.provider.fetch(symbol, timeframe, int(dcfg.get("bars", 5000)))
            log.info("Data fetched: %s %s rows=%d", symbol, timeframe, len(raw))
        except Exception as exc:
            log.warning("Data fetch failed for %s %s: %s", symbol, timeframe, exc)
            rep = ValidationReport(valid=False, reasons=[f"fetch failed: {exc}"])
            self._cache[key] = (None, rep)
            return self._cache[key]
        # US-session-only instruments (ETF proxies) have a gap every night by design
        gap_mult = 1e6 if symbol in dcfg.get("session_symbols", []) else float(dcfg.get("max_gap_multiplier", 3.0))
        df, rep = validate_candles(raw, timeframe, int(dcfg.get("min_bars", 600)), gap_mult)
        if not rep.valid:
            log.warning("Validation failed for %s %s: %s", symbol, timeframe, "; ".join(rep.reasons))
        else:
            log.info("Validated %s %s rows=%d (dups=%d, missing=%d, flagged=%d)", symbol, timeframe,
                     rep.n_output, rep.n_duplicates, rep.n_missing_candles, rep.n_unrealistic)
        self._cache[key] = (df if rep.valid else None, rep)
        return self._cache[key]

    def peers(self, symbol: str, timeframe: str, names: list[str]) -> Dict[str, pd.DataFrame]:
        out = {}
        for n in names:
            if n == symbol:
                continue
            if not is_market_open(n, self.cfg):  # stale weekend peers would mislead (and cost API credits)
                continue
            df, rep = self.get(n, timeframe)
            if df is not None:
                out[n] = df
        return out
