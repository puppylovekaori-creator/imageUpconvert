from __future__ import annotations

from dataclasses import asdict, fields
import json
from typing import Any

from .app_config import AppSettings, CONFIG_PATH

def load_settings() -> AppSettings:
    if not CONFIG_PATH.exists():
        return AppSettings()

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return AppSettings()
    if not isinstance(raw, dict):
        return AppSettings()

    allowed = {field.name for field in fields(AppSettings)}
    data = {key: value for key, value in raw.items() if key in allowed}
    return AppSettings(**data)


def save_settings(data: AppSettings | dict[str, Any]) -> None:
    payload: dict[str, Any]
    if isinstance(data, AppSettings):
        payload = asdict(data)
    else:
        payload = asdict(AppSettings(**data))
    CONFIG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
