"""本地凭证配置台 + 连通性自检（stdlib only，不引入 Web 框架）。

用法
----
    python -m qa_real.web.server            # http://127.0.0.1:8787

页面里填 ws_url 与 API key → 保存 → 自动写入 `<repo>/.secrets/credentials.json`
（已 .gitignore）。之后所有探针/评测命令通过 `qa_real.credentials.load()` 自动读取，
环境变量 `MOSS_LIVE_API_KEY` 优先级更高（便于 CI 注入而不落盘）。

安全
----
- 默认只绑 127.0.0.1，不对外暴露；
- 页面与 API 只回显掩码 key；
- 落盘文件 chmod 0600。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from qa_real import credentials
from qa_real.ws_probe import probe

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceQA/1.0"

    # ---------- 基础设施 ----------
    def log_message(self, format: str, *args: Any) -> None:
        # 不记录 query/headers，避免任何凭证进入日志
        print(f"[web] {self.address_string()} {format % args}")

    def _send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # ---------- 路由 ----------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif self.path == "/api/status":
            self._send_json({"ok": True, **credentials.status()})
        elif self.path == "/api/health":
            self._send_json({"ok": True, "service": "voice-qa-credential-console"})
        else:
            self._send_json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/credentials":
            body = self._read_body()
            ws_url = str(body.get("ws_url") or "").strip()
            api_key = str(body.get("api_key") or "").strip()
            if not ws_url.startswith(("ws://", "wss://")):
                self._send_json({"ok": False, "error": "ws_url 必须以 ws:// 或 wss:// 开头"}, 400)
                return
            if len(api_key) < 8:
                self._send_json({"ok": False, "error": "api_key 看起来太短"}, 400)
                return
            self._send_json({"ok": True, **credentials.save(ws_url, api_key)})
            return

        if self.path == "/api/probe":
            body = self._read_body()
            cur = credentials.load()
            ws_url = str(body.get("ws_url") or cur["ws_url"])
            api_key = str(body.get("api_key") or cur["api_key"])
            if not api_key:
                self._send_json({"ok": False, "error": "尚未配置 API key，请先保存"}, 400)
                return
            try:
                result = asyncio.run(
                    probe(
                        ws_url,
                        api_key,
                        tone_ms=float(body.get("tone_ms") or 1200),
                        tail_ms=float(body.get("tail_ms") or 2500),
                        send_tone=bool(body.get("send_tone", True)),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 探针异常必须如实回传
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._send_json({"ok": bool(result.get("ok")), "result": result})
            return

        self._send_json({"ok": False, "error": "not found"}, 404)

    def do_DELETE(self) -> None:
        if self.path == "/api/credentials":
            self._send_json({"ok": True, **credentials.clear()})
            return
        self._send_json({"ok": False, "error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser(description="本地凭证配置台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"凭证配置台: http://{args.host}:{args.port}")
    print(f"凭证落盘  : {credentials.STORE_FILE}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
