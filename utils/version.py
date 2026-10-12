"""Commit hash of the running code (GitHub Actions sets GITHUB_SHA) - lets you verify which version is deployed."""
from __future__ import annotations

import os
import subprocess


def code_version() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            sha = ""
    return sha[:10] or "unknown"
