"""Configuration loading. YAML for settings, environment variables (.env) for secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except Exception:  # python-dotenv is optional
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load config.yaml (or MARKET_RADAR_CONFIG / explicit path)."""
    _load_dotenv()
    cfg_path = Path(path or os.getenv("MARKET_RADAR_CONFIG") or ROOT / "config" / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_root"] = str(ROOT)
    if os.getenv("DATABASE_URL"):
        cfg.setdefault("database", {})["url"] = os.environ["DATABASE_URL"]
    if os.getenv("DATA_PROVIDER"):
        cfg.setdefault("data", {})["provider"] = os.environ["DATA_PROVIDER"]
    return cfg


def get_secrets() -> Dict[str, str | None]:
    """Secrets come only from the environment, never from config.yaml."""
    return {
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_channel": os.getenv("TELEGRAM_CHANNEL_ID"),
        "twelvedata_api_key": os.getenv("TWELVEDATA_API_KEY"),
        "turso_database_url": os.getenv("TURSO_DATABASE_URL"),
        "turso_auth_token": os.getenv("TURSO_AUTH_TOKEN"),
        "mt5_login": os.getenv("MT5_LOGIN"),
        "mt5_password": os.getenv("MT5_PASSWORD"),
        "mt5_server": os.getenv("MT5_SERVER"),
        "mt5_path": os.getenv("MT5_PATH"),
        "mt5_symbol_suffix": os.getenv("MT5_SYMBOL_SUFFIX", ""),
    }


def resolve_path(cfg: Dict[str, Any], rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = Path(cfg.get("_root", ROOT)) / p
    return p


def display_name(cfg: Dict[str, Any], symbol: str) -> str:
    return cfg.get("display_names", {}).get(symbol, symbol)
