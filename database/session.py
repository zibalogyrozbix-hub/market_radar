"""Engine/session management."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base
from config.settings import resolve_path


def turso_url(raw: str) -> str:
    """libsql://db-org.turso.io  ->  sqlite+libsql://db-org.turso.io?secure=true (sqlalchemy-libsql dialect)."""
    host = raw.split("://", 1)[1] if "://" in raw else raw
    host = host.split("?", 1)[0].rstrip("/")
    return f"sqlite+libsql://{host}?secure=true"


def make_engine(cfg: dict):
    import os

    url = cfg.get("database", {}).get("url", "sqlite:///data/market_radar.db")
    turso_raw = os.getenv("TURSO_DATABASE_URL") or (url if url.startswith(("libsql://", "wss://")) else "")
    if turso_raw and not os.getenv("DATABASE_URL"):
        token = os.getenv("TURSO_AUTH_TOKEN")
        if not token:
            raise RuntimeError("TURSO_DATABASE_URL is set but TURSO_AUTH_TOKEN is missing")
        # Turso (libSQL) over HTTPS/WebSocket. Needs `pip install sqlalchemy-libsql`.
        return create_engine(turso_url(turso_raw), connect_args={"auth_token": token}, future=True,
                             pool_pre_ping=True, pool_recycle=300)
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
        rel = url[len("sqlite:///"):]
        path = resolve_path(cfg, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    return create_engine(url, future=True)


class Database:
    def __init__(self, cfg: dict):
        self.engine = make_engine(cfg)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Session:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
