"""Market data providers: Twelve Data (free API), MetaTrader5 (live), CSV (offline), synthetic (tests/demo)."""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.logging_setup import get_logger
from utils.timeframes import tf_minutes

log = get_logger("data")
COLUMNS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]


class DataProvider:
    name = "base"

    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class MT5Provider(DataProvider):
    """MetaTrader5 (Windows only; the MT5 terminal must be installed/running)."""

    name = "mt5"
    _TF = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
           "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}

    def __init__(self, secrets: dict):
        import MetaTrader5 as mt5  # raises ImportError off-Windows

        self.mt5 = mt5
        self.suffix = secrets.get("mt5_symbol_suffix") or ""
        kwargs = {}
        if secrets.get("mt5_path"):
            kwargs["path"] = secrets["mt5_path"]
        if secrets.get("mt5_login"):
            kwargs.update(login=int(secrets["mt5_login"]), password=secrets.get("mt5_password") or "",
                          server=secrets.get("mt5_server") or "")
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        mt5 = self.mt5
        broker_symbol = symbol + self.suffix
        mt5.symbol_select(broker_symbol, True)
        tf = getattr(mt5, self._TF[timeframe.upper()])
        # start_pos=1 skips the still-forming candle
        rates = mt5.copy_rates_from_pos(broker_symbol, tf, 1, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No MT5 data for {broker_symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df[COLUMNS]

    def fetch_ticks(self, symbol: str, n: int = 5000) -> pd.DataFrame:
        """Optional tick data."""
        mt5 = self.mt5
        ticks = mt5.copy_ticks_from(symbol + self.suffix, pd.Timestamp.utcnow().to_pydatetime(), n, mt5.COPY_TICKS_ALL)
        if ticks is None:
            return pd.DataFrame()
        df = pd.DataFrame(ticks)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df


class CSVProvider(DataProvider):
    """Reads <csv_dir>/<SYMBOL>_<TF>.csv with columns timestamp,open,high,low,close[,tick_volume,spread]."""

    name = "csv"

    def __init__(self, csv_dir: Path):
        self.dir = Path(csv_dir)

    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        path = self.dir / f"{symbol}_{timeframe}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        return df.tail(bars).reset_index(drop=True)


class TwelveDataError(RuntimeError):
    """Raised for Twelve Data API errors. `retryable` is False for plan/symbol/daily-limit errors."""

    def __init__(self, msg: str, retryable: bool = False):
        super().__init__(msg)
        self.retryable = retryable


class TwelveDataProvider(DataProvider):
    """Twelve Data REST API (https://twelvedata.com) - works from any runner (GitHub Actions, cron, ...).

    * Free "Basic" plan: 8 API credits/minute and 800/day, 1 credit per symbol per request. A sliding-window
      limiter keeps requests under `requests_per_minute`; a 429 waits for the next minute and retries.
    * Twelve Data labels each bar with its OPEN time (same as MT5). The newest bar is dropped when it is
      still forming, so the system only ever sees closed candles.
    * FX/metals/crypto have no tick volume or spread; volume is kept when the API returns it, else 0.
    * `symbol_map` translates internal names (EURUSD, XAUUSD, US500 ...) to Twelve Data symbols.
    * `max_stale_minutes` rejects data whose newest closed bar is too old (holiday / API hiccup).
      `session_symbols` (e.g. ETF proxies that only trade during US hours) are exempt.
    """

    name = "twelvedata"
    BASE_URL = "https://api.twelvedata.com/time_series"
    _INTERVAL = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}
    MAX_OUTPUTSIZE = 5000

    def __init__(self, api_key: str, symbol_map: Optional[dict] = None, requests_per_minute: int = 8,
                 max_retries: int = 3, timeout: int = 30, max_stale_minutes: int = 0,
                 session_symbols: Optional[list] = None, market_open=None, http=None, sleep=None, clock=None):
        import time as _time

        if not api_key:
            raise RuntimeError("TWELVEDATA_API_KEY is not set")
        self.api_key = api_key
        self.symbol_map = dict(symbol_map or {})
        self.rpm = max(1, int(requests_per_minute))
        self.max_retries = max(1, int(max_retries))
        self.timeout = timeout
        self.max_stale = pd.Timedelta(minutes=int(max_stale_minutes)) if max_stale_minutes else None
        self.session_symbols = set(session_symbols or [])
        self.market_open = market_open  # optional callable(symbol) -> bool; stale check only while open
        if http is None:
            import requests as http  # noqa: PLW0127
        self.http = http
        self._sleep = sleep or _time.sleep
        self._clock = clock or _time.monotonic
        self._calls: list = []  # monotonic times of recent calls (sliding 60 s window)
        self.credits_used = 0

    # -- helpers ---------------------------------------------------------------------------
    def td_symbol(self, symbol: str) -> str:
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        if len(symbol) == 6 and symbol.isalpha():  # EURUSD -> EUR/USD, XAUUSD -> XAU/USD
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol

    def _throttle(self) -> None:
        window = 61.0  # a little over a minute: Twelve Data counts per calendar minute
        now = self._clock()
        self._calls = [t for t in self._calls if now - t < window]
        if len(self._calls) >= self.rpm:
            wait = window - (now - self._calls[0])
            if wait > 0:
                log.info("Twelve Data rate limit: waiting %.0fs", wait)
                self._sleep(wait)
            now = self._clock()
            self._calls = [t for t in self._calls if now - t < window]
        self._calls.append(self._clock())

    def _request(self, params: dict) -> dict:
        last = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.http.get(self.BASE_URL, params=params, timeout=self.timeout)
                self.credits_used += 1
                body = resp.json()
            except Exception as exc:  # network / JSON error
                last = f"request failed: {exc}"
                self._sleep(min(2 ** attempt, 10))
                continue
            if isinstance(body, dict) and body.get("status") == "error":
                code = int(body.get("code") or resp.status_code or 0)
                msg = str(body.get("message", ""))[:300]
                if code == 429 and "day" not in msg.lower():
                    last = f"rate limited ({msg})"
                    log.warning("Twelve Data 429 for %s: waiting for the next minute", params.get("symbol"))
                    self._sleep(61)
                    self._calls.clear()
                    continue
                raise TwelveDataError(f"Twelve Data {code}: {msg}", retryable=False)
            if resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                self._sleep(min(2 ** attempt, 10))
                continue
            return body
        raise TwelveDataError(f"Twelve Data failed after {self.max_retries} attempts: {last}", retryable=True)

    # -- API -------------------------------------------------------------------------------
    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        tf = timeframe.upper()
        if tf not in self._INTERVAL:
            raise ValueError(f"Unsupported timeframe for Twelve Data: {timeframe}")
        params = {"symbol": self.td_symbol(symbol), "interval": self._INTERVAL[tf],
                  "outputsize": min(int(bars) + 1, self.MAX_OUTPUTSIZE), "timezone": "UTC",
                  "order": "DESC", "format": "JSON", "apikey": self.api_key}
        body = self._request(params)
        values = body.get("values") if isinstance(body, dict) else None
        if not values:
            raise TwelveDataError(f"Twelve Data returned no candles for {params['symbol']}")
        df = pd.DataFrame(values)
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["tick_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df.columns else 0.0
        df["spread"] = 0.0
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        step = tf_minutes(tf)
        now = pd.Timestamp.now(tz="UTC")
        # drop the still-forming candle (its open time + timeframe is in the future)
        df = df[df["timestamp"] + pd.Timedelta(minutes=step) <= now + pd.Timedelta(seconds=30)]
        if df.empty:
            raise TwelveDataError(f"no closed candles for {params['symbol']}")
        if (self.max_stale is not None and symbol not in self.session_symbols
                and (self.market_open is None or self.market_open(symbol))):
            age = now - (df["timestamp"].iloc[-1] + pd.Timedelta(minutes=step))
            if age > self.max_stale:
                raise TwelveDataError(f"stale data for {symbol}: newest closed candle is {age} old")
        return df[COLUMNS].tail(bars).reset_index(drop=True)


_BASE_PRICE = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150.0, "AUDUSD": 0.66, "USDCAD": 1.36,
               "USDCHF": 0.89, "XAUUSD": 2300.0, "XAGUSD": 28.0, "BTCUSD": 60000.0, "US500": 5200.0,
               "NAS100": 18000.0}
_VOL_SCALE = {"XAUUSD": 0.0016, "XAGUSD": 0.0028, "BTCUSD": 0.004, "US500": 0.0012, "NAS100": 0.0016}
_FACTOR_LOAD = {"EURUSD": 0.4, "GBPUSD": 0.4, "USDJPY": -0.3, "AUDUSD": 0.5, "USDCAD": -0.3, "USDCHF": -0.4,
                "XAUUSD": 0.3, "XAGUSD": 0.4, "BTCUSD": 0.5, "US500": 0.8, "NAS100": 0.9}


class SyntheticProvider(DataProvider):
    """Deterministic synthetic OHLC with volatility clustering and a shared market factor.

    NOT real market data: it exists so the pipeline and tests run anywhere. Predictions made on
    it carry no meaning; the system labels the data source in logs.
    """

    name = "synthetic"

    def __init__(self, seed: int = 7, end: Optional[pd.Timestamp] = None):
        self.seed = seed
        self.end = end

    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        step = tf_minutes(timeframe)
        end = self.end or pd.Timestamp.now(tz="UTC")
        end = end.floor(f"{step}min") - pd.Timedelta(minutes=step)
        idx = pd.date_range(end=end, periods=bars, freq=f"{step}min", tz="UTC")

        common = np.random.default_rng(self.seed)
        own = np.random.default_rng(self.seed * 1000 + zlib.crc32(symbol.encode()) % 100000)
        n = bars
        factor = common.standard_normal(n)
        vol_log = np.zeros(n)
        for i in range(1, n):  # log-vol AR(1) => volatility clustering
            vol_log[i] = 0.97 * vol_log[i - 1] + 0.18 * own.standard_normal()
        sigma = _VOL_SCALE.get(symbol, 0.0007) * np.exp(vol_log)
        eps = 0.8 * own.standard_normal(n) + _FACTOR_LOAD.get(symbol, 0.3) * factor
        # occasional shock bars
        shocks = (own.random(n) < 0.004) * own.choice([-1, 1], n) * own.uniform(3, 6, n)
        ret = np.zeros(n)
        for i in range(1, n):  # weak momentum
            ret[i] = 0.03 * ret[i - 1] + sigma[i] * (eps[i] + shocks[i])
        close = _BASE_PRICE.get(symbol, 100.0) * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(own.standard_normal((n, 2))) * sigma[:, None] * 0.6
        high = np.maximum(open_, close) * (1 + wick[:, 0])
        low = np.minimum(open_, close) * (1 - wick[:, 1])
        tick_vol = np.maximum(1, (200 * (sigma / sigma.mean()) * own.lognormal(0, 0.4, n))).round()
        spread = np.maximum(1, (12 * (1 + 0.5 * (sigma / sigma.mean() - 1)) * own.lognormal(0, 0.1, n))).round()
        return pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close,
                             "tick_volume": tick_vol, "spread": spread})
