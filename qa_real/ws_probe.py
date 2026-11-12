"""真实 WebSocket 连通性探针：验证 key/endpoint 是否可用，并采第一手协议事件。

为什么单独做一个探针
--------------------
接真实模型前必须先回答"这条路通不通"。探针只回答连通性与协议形态：
握手是否被接受、服务端回什么事件类型、首个下行音频何时到达。
它不替代完整评测，只用于把 `protocol_observed` 从 false 变 true 的第一步。

安全
----
API key 只进入 WebSocket 握手头，**绝不**进入返回值、日志或异常信息。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import numpy as np

from qa_real.credentials import mask_key
from voice_qa_engine.clients.moss_protocol_adapter import MossLiveProtocolAdapter

SAMPLE_RATE = 16000


def _pcm16_tone(duration_ms: float, freq: float = 440.0, amplitude: float = 0.25) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000.0)
    t = np.arange(n) / SAMPLE_RATE
    wave = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return wave.tobytes()


def _pcm16_silence(duration_ms: float) -> bytes:
    return np.zeros(int(SAMPLE_RATE * duration_ms / 1000.0), dtype=np.int16).tobytes()


async def probe(
    ws_url: str,
    api_key: str,
    tone_ms: float = 1200.0,
    tail_ms: float = 2500.0,
    send_tone: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """连一次真实服务端，返回可判读的探测结果（不含明文 key）。"""
    try:
        import websockets
    except ImportError:
        return {
            "ok": False,
            "protocol_observed": False,
            "error": "websockets 未安装：pip install websockets",
            "ws_url": ws_url,
            "api_key_masked": mask_key(api_key),
        }

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    adapter = MossLiveProtocolAdapter(target_sample_rate=SAMPLE_RATE)
    session_id = f"probe_{int(time.time())}"
    events: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "ok": False,
        "protocol_observed": False,
        "ws_url": ws_url,
        "api_key_masked": mask_key(api_key),
        "auth_header_present": bool(api_key),
        "sample_rate": SAMPLE_RATE,
        "send_tone": send_tone,
    }

    t_connect_start = time.monotonic_ns()
    try:
        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            result["connect_ms"] = round((time.monotonic_ns() - t_connect_start) / 1e6, 2)

            handshake = adapter.build_handshake({"session_id": session_id})
            await ws.send(__import__("json").dumps(handshake))
            result["handshake_sent"] = handshake.get("type")

            async def recv_loop(deadline_ns: int) -> None:
                while time.monotonic_ns() < deadline_ns:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except TimeoutError:
                        continue
                    t_ns = time.monotonic_ns()
                    if isinstance(raw, (bytes, bytearray)):
                        events.append({"t_ms": round(t_ns / 1e6, 2), "type": "binary", "bytes": len(raw)})
                        continue
                    with contextlib.suppress(Exception):
                        data = __import__("json").loads(raw)
                        events.append(
                            {
                                "t_ms": round(t_ns / 1e6, 2),
                                "type": str(data.get("type", "unknown")),
                                "keys": sorted(data.keys())[:12],
                            }
                        )

            deadline = time.monotonic_ns() + int((tone_ms + tail_ms) * 1e6)
            recv_task = asyncio.create_task(recv_loop(deadline))

            if send_tone:
                # 按真实 20ms 节奏推流，避免把对端缓冲一次灌爆
                tone = _pcm16_tone(tone_ms)
                frame_bytes = int(SAMPLE_RATE * 0.02) * 2
                for i in range(0, len(tone) - frame_bytes + 1, frame_bytes):
                    chunk = tone[i : i + frame_bytes]
                    await ws.send(
                        __import__("json").dumps(
                            adapter.build_audio_message(
                                _frame(chunk, sequence_id=i // frame_bytes)
                            )
                        )
                    )
                    await asyncio.sleep(0.02)
                # 继续补静音，保持长连接与上下文（文档 10.2 要求）
                silence = _pcm16_silence(min(tail_ms, 800.0))
                for i in range(0, len(silence) - frame_bytes + 1, frame_bytes):
                    await ws.send(
                        __import__("json").dumps(
                            adapter.build_audio_message(
                                _frame(silence[i : i + frame_bytes], sequence_id=10_000 + i // frame_bytes)
                            )
                        )
                    )
                    await asyncio.sleep(0.02)

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(recv_task, timeout=(tone_ms + tail_ms) / 1000.0 + 1.0)
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task

        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - 探针必须把任何失败如实带回
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["error_hint"] = _hint(exc)
        result["connect_failed_ms"] = round((time.monotonic_ns() - t_connect_start) / 1e6, 2)

    types: dict[str, int] = {}
    for e in events:
        types[e["type"]] = types.get(e["type"], 0) + 1
    result["event_count"] = len(events)
    result["event_types"] = types
    result["first_events"] = events[:20]
    result["protocol_observed"] = bool(result.get("ok") and len(events) > 0)
    if result["protocol_observed"]:
        result["ok"] = True
    return result


def _frame(pcm: bytes, sequence_id: int) -> Any:
    from voice_qa_engine.audio.models import AudioFormat, AudioFrame, SampleFormat

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return AudioFrame(
        samples=samples,
        format=AudioFormat(SAMPLE_RATE, 1, SampleFormat.PCM_F32LE),
        timestamp_ns=time.monotonic_ns(),
        sequence_id=sequence_id,
        duration_ms=len(samples) / SAMPLE_RATE * 1000.0,
    )


def _hint(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "鉴权被拒：检查 API key 是否正确/过期，或是否需要额外的 header（如 X-Api-Key）"
    if "dns" in msg or "name or service not known" in msg or "getaddrinfo" in msg:
        return "域名解析失败：检查 ws_url 拼写与网络/DNS"
    if "timeout" in msg or "timed out" in msg:
        return "连接超时：检查网络、防火墙，或 endpoint 是否仍在服务"
    if "certificate" in msg or "ssl" in msg:
        return "TLS 失败：检查证书链，或代理是否拦截"
    if "404" in msg:
        return "路径不存在：ws_url 的路径部分可能不是 /v1/realtime"
    return "未归类错误：把 error 原文发给维护者"
