"""Model registry stored in the database (works with SQLite or Turso/libSQL).

Same interface as ModelRegistry. Bundles are joblib-compressed, base64-encoded (libSQL has no BLOB parameters) and split into
~256 KB rows so a single remote request stays small. Loaded bundles are cached in memory for the lifetime of the process.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
from sqlalchemy import delete, func, select

from database.models import ModelArtifactRow

CHUNK = 192 * 1024  # raw bytes per row (-> 256 KB of base64 text)


class DBModelRegistry:
    def __init__(self, db):
        self.db = db
        self._cache: Dict[tuple, Dict[str, Any]] = {}

    # -- low-level blob helpers ---------------------------------------------------------------
    def _put(self, s, symbol, tf, version, kind, payload: bytes) -> None:
        s.execute(delete(ModelArtifactRow).where(
            ModelArtifactRow.symbol == symbol, ModelArtifactRow.timeframe == tf,
            ModelArtifactRow.version == version, ModelArtifactRow.kind == kind))
        s.flush()
        for i in range(0, max(len(payload), 1), CHUNK):
            s.add(ModelArtifactRow(symbol=symbol, timeframe=tf, version=version, kind=kind, part=i // CHUNK,
                                   data=base64.b64encode(payload[i:i + CHUNK]).decode("ascii")))

    def _get(self, s, symbol, tf, version, kind) -> Optional[bytes]:
        rows = s.scalars(select(ModelArtifactRow).where(
            ModelArtifactRow.symbol == symbol, ModelArtifactRow.timeframe == tf,
            ModelArtifactRow.version == version, ModelArtifactRow.kind == kind).order_by(ModelArtifactRow.part)).all()
        return b"".join(base64.b64decode(r.data) for r in rows) if rows else None

    # -- registry interface -------------------------------------------------------------------
    def next_version(self, symbol: str, tf: str) -> str:
        with self.db.session() as s:
            versions = s.scalars(select(ModelArtifactRow.version).where(
                ModelArtifactRow.symbol == symbol, ModelArtifactRow.timeframe == tf,
                ModelArtifactRow.kind == "meta").distinct()).all()
        nums = [int(v.split("_v")[-1]) for v in versions if "_v" in v]
        return f"ML_v{(max(nums) + 1 if nums else 1):03d}"

    def save(self, symbol: str, tf: str, version: str, bundle: Dict[str, Any], meta: Dict[str, Any]) -> None:
        buf = io.BytesIO()
        joblib.dump(bundle, buf, compress=3)
        meta = {**meta, "version": version, "saved_at": datetime.now(timezone.utc).isoformat()}
        with self.db.session() as s:
            self._put(s, symbol, tf, version, "bundle", buf.getvalue())
            self._put(s, symbol, tf, version, "meta", json.dumps(meta, default=str).encode("utf-8"))

    def activate(self, symbol: str, tf: str, version: str) -> None:
        with self.db.session() as s:
            self._put(s, symbol, tf, "-", "active", json.dumps({"version": version}).encode("utf-8"))
        self._cache.pop((symbol, tf), None)

    def active_version(self, symbol: str, tf: str) -> Optional[str]:
        with self.db.session() as s:
            raw = self._get(s, symbol, tf, "-", "active")
        return json.loads(raw).get("version") if raw else None

    def meta(self, symbol: str, tf: str, version: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as s:
            raw = self._get(s, symbol, tf, version, "meta")
        return json.loads(raw) if raw else None

    def load_active(self, symbol: str, tf: str) -> Optional[Dict[str, Any]]:
        v = self.active_version(symbol, tf)
        if not v:
            return None
        key = (symbol, tf, v)
        if key not in self._cache:
            with self.db.session() as s:
                raw = self._get(s, symbol, tf, v, "bundle")
            if raw is None:
                return None
            self._cache[key] = joblib.load(io.BytesIO(raw))
        bundle = dict(self._cache[key])
        bundle["version"] = v
        return bundle
