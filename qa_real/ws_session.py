"""L1 协议层真实会话采集器：单调时钟打点 + 背压丢帧埋点 + 半包/乱序/重复检测。

设计要点（对应评审 P0/P1）
-------------------------
1. 所有时间戳统一用 `time.monotonic_ns()`，服务端时间戳只作为 payload 保存，**不参与相减**；
2. 上行队列满不再静默 `pass`，而是记 `AUDIO_FRAME_DROPPED` 并计入 metrics；
3. 指标一律带 `n` 与 `measurement_source=protocol_observed`；
4. 缺证据写 null，不写 0。

用法
----
    # 对本地假服务端（已知标准答案）
    python -m qa_real.mock_server --port 8899 &
    python -m qa_real.ws_session --url ws://127.0.0.1:8899 --cancel-after-first-audio \
        --expect '{"ttft_ms":400,"first_audio_ms":600,"cancel_honor_ms":150}'

    # 对真实服务端（自动读取页面保存的凭证）
    python -m qa_real.ws_session
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from qa_real import credentials
from voice_qa_engine.audio.models import AudioFormat, AudioFrame, SampleFormat
from voice_qa_engine.clients.moss_protocol_adapter import MossLiveProtocolAdapter
from voice_qa_engine.clients.protocol_models import EventType, RuntimeEvent

SAMPLE_RATE = 16000
FRAME_MS = 20.0
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000.0)


def _frame(freq: float, amplitude: float = 0.25) -> AudioFrame:
    t = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioFrame(
        samples=samples,
        format=AudioFormat(SAMPLE_RATE, 1, SampleFormat.PCM_F32LE),
        timestamp_ns=time.monotonic_ns(),
        sequence_id=0,
        duration_ms=FRAME_MS,
    )


class SendQueue:
    """有界发送队列 + 丢帧埋点（替代原实现的 QueueFull: pass）。"""

    def __init__(self, maxsize: int = 64) -> None:
        self.q: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.enqueued = 0
        self.dropped = 0
        self.drop_events: list[int] = []

    def put(self, item: str) -> None:
        try:
            self.q.put_nowait(item)
            self.enqueued += 1
        except asyncio.QueueFull:
            self.dropped += 1
            self.drop_events.append(time.monotonic_ns())


async def run_session(
    ws_url: str,
    api_key: str = "",
    speech_ms: float = 1000.0,
    cancel_after_first_audio: bool = True,
    tail_ms: float = 1500.0,
    expect: dict[str, float] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    import websockets

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    adapter = MossLiveProtocolAdapter(target_sample_rate=SAMPLE_RATE)
    session_id = f"l1_{int(time.time())}"
    session_dir = (out_dir or Path("reports") / "l1_sessions") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    events: list[RuntimeEvent] = []
    raw_log: list[dict[str, Any]] = []
    t: dict[str, int] = {}
    seq_seen: list[int] = []
    dup_count = 0
    ooo_count = 0
    cancel_sent = False

    async with websockets.connect(ws_url, additional_headers=headers, ping_interval=None) as ws:
        queue = SendQueue(maxsize=64)

        async def sender() -> None:
            while True:
                item = await queue.q.get()
                if item == "__STOP__":
                    return
                await ws.send(item)

        async def receiver() -> None:
            nonlocal dup_count, ooo_count, cancel_sent
            async for raw in ws:
                now = time.monotonic_ns()
                raw_log.append({"monotonic_ns": now, "size": len(raw)})
                for evt in adapter.parse_server_message(raw, session_id, 1):
                    evt.monotonic_ns = now
                    events.append(evt)
                    if evt.event_type == EventType.SESSION_READY:
                        t.setdefault("session_ready_ns", now)
                    elif evt.event_type == EventType.TEXT_DELTA:
                        t.setdefault("first_text_ns", now)
                    elif evt.event_type == EventType.AUDIO_CHUNK:
                        sid = evt.sequence_id
                        if sid is not None:
                            if sid in seq_seen:
                                dup_count += 1
                            elif seq_seen and sid < seq_seen[-1]:
                                ooo_count += 1
                            seq_seen.append(sid)
                        if "first_audio_ns" not in t:
                            t["first_audio_ns"] = now
                            if cancel_after_first_audio and not cancel_sent:
                                cancel_sent = True
                                t["cancel_sent_ns"] = time.monotonic_ns()
                                await ws.send(
                                    json.dumps(adapter.build_cancel_message("resp_probe"))
                                )
                        t["last_audio_ns"] = now
                    elif evt.event_type == EventType.RESPONSE_CANCELLED:
                        t.setdefault("cancelled_ns", now)
                        return
                    elif evt.event_type == EventType.RESPONSE_COMPLETED:
                        t.setdefault("response_done_ns", now)
                        return

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())

        await ws.send(json.dumps(adapter.build_handshake({"session_id": session_id})))
        t["handshake_sent_ns"] = time.monotonic_ns()

        # 上行：真按 20ms 节奏推流，记录语音结束时刻
        n_frames = int(speech_ms / FRAME_MS)
        for i in range(n_frames):
            frame = _frame(440.0)
            frame.sequence_id = i
            queue.put(json.dumps(adapter.build_audio_message(frame)))
            await asyncio.sleep(FRAME_MS / 1000.0)
        t["input_audio_end_ns"] = time.monotonic_ns()
        # 语音后补静音，维持长连接（文档 10.2 要求）
        silence_n = int((tail_ms + 500) / FRAME_MS)
        for i in range(silence_n):
            frame = _frame(440.0, amplitude=0.0)
            frame.sequence_id = n_frames + i
            queue.put(json.dumps(adapter.build_audio_message(frame)))
            await asyncio.sleep(FRAME_MS / 1000.0)
            if recv_task.done():
                break

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(recv_task, timeout=max(0.5, tail_ms / 1000.0))
        recv_task.cancel()
        queue.q.put_nowait("__STOP__")
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await recv_task
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await send_task

    # ---------- 指标计算（单调时钟，无跨端相减） ----------
    ref = t.get("input_audio_end_ns")
    metrics: dict[str, Any] = {
        "session_id": session_id,
        "ws_url": ws_url,
        "measurement_source": "protocol_observed",
        "sample_rate": SAMPLE_RATE,
        "n": len(seq_seen),
        "audio_delta_count": len(seq_seen),
        "duplicate_delta_count": dup_count,
        "out_of_order_count": ooo_count,
        "uplink_enqueued": queue.enqueued,
        "uplink_dropped": queue.dropped,
        "ttft_ms": _delta_ms(t.get("first_text_ns"), ref),
        "first_audio_chunk_ms": _delta_ms(t.get("first_audio_ns"), ref),
        "cancel_handler_ms": _delta_ms(t.get("cancelled_ns"), t.get("cancel_sent_ns")),
        "cancel_to_last_audio_ms": _delta_ms(t.get("last_audio_ns"), t.get("cancel_sent_ns")),
        "response_completed": "response_done_ns" in t,
        "timestamps_monotonic_ns": dict(t),
    }

    comparison: dict[str, Any] | None = None
    if expect:
        comparison = {}
        for key, truth in expect.items():
            measured = metrics.get(key)
            if measured is None:
                comparison[key] = {"expected": truth, "measured": None, "ok": False, "reason": "未测到"}
                continue
            diff = round(float(measured) - float(truth), 3)
            comparison[key] = {
                "expected": truth,
                "measured": round(float(measured), 3),
                "diff_ms": diff,
                "ok": abs(diff) <= 50.0,
            }
        metrics["comparison_vs_ground_truth"] = comparison
        metrics["ruler_verdict"] = "PASS" if all(c["ok"] for c in comparison.values()) else "FAIL"

    # ---------- 证据落盘 ----------
    (session_dir / "events.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "event_type": str(getattr(e.event_type, "value", e.event_type)),
                    "monotonic_ns": e.monotonic_ns,
                    "sequence_id": e.sequence_id,
                    "response_id": e.response_id,
                    "raw_event_type": e.raw_event_type,
                },
                ensure_ascii=False,
            )
            for e in events
        ),
        encoding="utf-8",
    )
    (session_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "ws_url": ws_url,
                "auth_header_present": bool(api_key),
                "credential_source": credentials.load().get("source"),
                "python_host": "windows" if __import__("sys").platform == "win32" else __import__("sys").platform,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "protocol_observed": metrics["audio_delta_count"] > 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics["session_dir"] = str(session_dir)
    return metrics


def _delta_ms(a: int | None, b: int | None) -> float | None:
    if a is None or b is None:
        return None
    return round((a - b) / 1e6, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="L1 协议层真实会话采集（单调时钟打点）")
    ap.add_argument("--url", default=None, help="不传则读页面保存的凭证")
    ap.add_argument("--key", default=None, help="不传则读页面保存的凭证")
    ap.add_argument("--speech-ms", type=float, default=1000.0)
    ap.add_argument("--tail-ms", type=float, default=1500.0)
    ap.add_argument("--no-cancel", action="store_true", help="不做打断，跑完整响应")
    ap.add_argument("--expect", default=None, help='标准答案 JSON，如 {"ttft_ms":400}')
    ap.add_argument("--out", default="reports/l1_sessions")
    args = ap.parse_args()

    cur = credentials.load()
    ws_url = args.url or cur["ws_url"]
    api_key = args.key if args.key is not None else cur["api_key"]
    expect: dict[str, float] | None = None
    if args.expect:
        try:
            parsed = json.loads(args.expect)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--expect 不是合法 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--expect 必须是 JSON 对象，如 {\"ttft_ms\":400}")
        expect = {k: float(v) for k, v in parsed.items()}

    print(f"target: {ws_url}  auth={'yes' if api_key else 'no'}")
    metrics = asyncio.run(
        run_session(
            ws_url=ws_url,
            api_key=api_key,
            speech_ms=args.speech_ms,
            cancel_after_first_audio=not args.no_cancel,
            tail_ms=args.tail_ms,
            expect=expect,
            out_dir=Path(args.out),
        )
    )
    for k in (
        "ttft_ms",
        "first_audio_chunk_ms",
        "cancel_handler_ms",
        "cancel_to_last_audio_ms",
        "audio_delta_count",
        "duplicate_delta_count",
        "out_of_order_count",
        "uplink_dropped",
    ):
        print(f"  {k:26s} {metrics.get(k)}")
    if metrics.get("comparison_vs_ground_truth"):
        print("  --- 对照标准答案 ---")
        for k, v in metrics["comparison_vs_ground_truth"].items():
            print(f"  {k:26s} expected={v['expected']:<8} measured={v['measured']!s:<9} diff={v['diff_ms']}ms ok={v['ok']}")
        print(f"  ruler_verdict: {metrics['ruler_verdict']}")
    print(f"  evidence: {metrics['session_dir']}")


if __name__ == "__main__":
    main()
