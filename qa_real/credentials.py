"""凭证本地存储：页面输入 → 落盘 → 所有探针/测试自动读取。

安全约定
--------
- 落盘位置 `<repo>/.secrets/credentials.json`，已加入 .gitignore；
- 文件权限尽量收紧到 0600（Windows 上为 best-effort ACL）；
- 任何返回值/日志只给 `mask_key()` 后的掩码，**明文只在内存里出现一次**；
- 环境变量优先级高于文件，便于 CI 注入而不落盘。
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

STORE_DIR = Path(__file__).resolve().parents[1] / ".secrets"
STORE_FILE = STORE_DIR / "credentials.json"

DEFAULT_WS_URL = "wss://api.mosi.cn/v1/realtime"


def mask_key(key: str | None) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 8}{key[-4:]}"


def _ensure_dir() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = STORE_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")


def save(ws_url: str, api_key: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_dir()
    payload = {
        "ws_url": ws_url.strip(),
        "api_key": api_key.strip(),
        "extra": extra or {},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STORE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(STORE_FILE, 0o600)
    return status()


def clear() -> dict[str, Any]:
    with contextlib.suppress(FileNotFoundError):
        STORE_FILE.unlink()
    return status()


def _from_file() -> dict[str, Any]:
    if not STORE_FILE.exists():
        return {}
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load() -> dict[str, Any]:
    """合并读取：环境变量优先，其次落盘文件。"""
    data = _from_file()
    url = os.getenv("MOSS_LIVE_WS_URL") or data.get("ws_url") or DEFAULT_WS_URL
    key = os.getenv("MOSS_LIVE_API_KEY") or data.get("api_key") or ""
    source = "env" if os.getenv("MOSS_LIVE_API_KEY") else ("file" if data.get("api_key") else "none")
    return {
        "ws_url": url,
        "api_key": key,
        "extra": data.get("extra", {}),
        "source": source,
        "updated_at": data.get("updated_at"),
    }


def status() -> dict[str, Any]:
    """对外安全视图：只暴露掩码。"""
    cur = load()
    return {
        "configured": bool(cur.get("api_key")),
        "ws_url": cur.get("ws_url"),
        "api_key_masked": mask_key(cur.get("api_key")),
        "source": cur.get("source"),
        "updated_at": cur.get("updated_at"),
        "store_path": str(STORE_FILE),
    }
