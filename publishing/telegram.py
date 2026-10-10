"""Telegram Bot API publisher (sendMessage / sendPhoto) with retry and graceful failure."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests

from utils.logging_setup import get_logger

log = get_logger("telegram")
API = "https://api.telegram.org/bot{token}/{method}"
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4096


def build_message_payload(chat_id: str, text: str) -> dict:
    return {"chat_id": chat_id, "text": text[:MESSAGE_LIMIT], "disable_web_page_preview": True}


def build_photo_payload(chat_id: str, caption: Optional[str] = None) -> dict:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption[:CAPTION_LIMIT]
    return data


class TelegramPublisher:
    def __init__(self, token: Optional[str], chat_id: Optional[str], retries: int = 3, timeout: int = 20,
                 http=None, sleep: Callable[[float], None] = time.sleep):
        self.token, self.chat_id = token, chat_id
        self.retries, self.timeout = max(1, retries), timeout
        self.http = http or requests
        self.sleep = sleep

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _call(self, method: str, data: dict, files: Optional[dict] = None) -> Tuple[bool, str]:
        url = API.format(token=self.token, method=method)
        last_err = ""
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.http.post(url, data=data, files=files, timeout=self.timeout)
                if resp.status_code == 200:
                    return True, ""
                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                last_err = f"HTTP {resp.status_code}: {body.get('description', resp.text[:200])}"
                if resp.status_code == 429:
                    wait = float(body.get("parameters", {}).get("retry_after", 2 ** attempt))
                elif resp.status_code >= 500:
                    wait = 2 ** attempt
                else:  # 4xx other than 429 will not succeed on retry
                    return False, last_err
            except requests.RequestException as exc:
                last_err, wait = f"network error: {exc}", 2 ** attempt
            log.warning("Telegram %s attempt %d/%d failed: %s", method, attempt, self.retries, last_err)
            if attempt < self.retries:
                self.sleep(min(wait, 60))
        return False, last_err

    def send_message(self, text: str) -> Tuple[bool, str]:
        return self._call("sendMessage", build_message_payload(self.chat_id, text))

    def send_photo(self, image_path: str, caption: Optional[str] = None) -> Tuple[bool, str]:
        with open(image_path, "rb") as fh:
            return self._call("sendPhoto", build_photo_payload(self.chat_id, caption), files={"photo": fh})

    def publish(self, text: str, image_path: Optional[str] = None) -> Tuple[bool, str]:
        """Photo + caption when it fits, otherwise photo then message. Never raises."""
        if not self.configured:
            return False, "telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL_ID)"
        try:
            if image_path and Path(image_path).exists():
                if len(text) <= CAPTION_LIMIT:
                    return self.send_photo(image_path, text)
                ok, err = self.send_photo(image_path)
                if not ok:
                    return False, err
            return self.send_message(text)
        except Exception as exc:
            return False, f"unexpected error: {exc}"
