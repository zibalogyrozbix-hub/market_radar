"""Versioned model storage. One directory per symbol/timeframe, `active.json` points to the live version."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, symbol: str, tf: str) -> Path:
        d = self.root / f"{symbol}_{tf}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def next_version(self, symbol: str, tf: str) -> str:
        d = self._dir(symbol, tf)
        nums = [int(p.stem.split("_v")[-1]) for p in d.glob("ML_v*.joblib")]
        return f"ML_v{(max(nums) + 1 if nums else 1):03d}"

    def save(self, symbol: str, tf: str, version: str, bundle: Dict[str, Any], meta: Dict[str, Any]) -> None:
        d = self._dir(symbol, tf)
        joblib.dump(bundle, d / f"{version}.joblib")
        meta = {**meta, "version": version, "saved_at": datetime.now(timezone.utc).isoformat()}
        (d / f"{version}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def activate(self, symbol: str, tf: str, version: str) -> None:
        (self._dir(symbol, tf) / "active.json").write_text(json.dumps({"version": version}), encoding="utf-8")

    def active_version(self, symbol: str, tf: str) -> Optional[str]:
        f = self._dir(symbol, tf) / "active.json"
        if not f.exists():
            return None
        return json.loads(f.read_text(encoding="utf-8")).get("version")

    def meta(self, symbol: str, tf: str, version: str) -> Optional[Dict[str, Any]]:
        f = self._dir(symbol, tf) / f"{version}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def load_active(self, symbol: str, tf: str) -> Optional[Dict[str, Any]]:
        v = self.active_version(symbol, tf)
        if not v:
            return None
        f = self._dir(symbol, tf) / f"{v}.joblib"
        if not f.exists():
            return None
        bundle = joblib.load(f)
        bundle["version"] = v
        return bundle
