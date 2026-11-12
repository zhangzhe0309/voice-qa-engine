"""可注入故障的本地假服务端（L1 测量链的"已知量发生器"）。

它的唯一用途：**给测量链一个标准答案**。
我们事先知道 TTFT=400ms、首音频=600ms、CANCEL 兑现=150ms，
然后用 `qa_real.ws_session` 去测。测不准 = 尺子坏；测准了 = 这把尺子才可以去量真模型。

协议对齐 `MossLiveProtocolAdapter`（OpenAI Realtime 风格）：
  client → session.update / input_audio_buffer.append / response.cancel
  server → session.created / response.created / response.text.delta
           / response.audio.delta / response.cancelled / response.done

故障注入旋钮（全部有确定性种子）
--------------------------------
--ttft-ms            语音结束 → 首个 text.delta
--first-audio-ms     语音结束 → 首个 audio.delta
--cancel-honor-ms    response.cancel 收到 → 停止推送 + cancelled
--dup-delta-rate     重复发送音频分片的比例（测客户端去重）
--out-of-order       相邻音频分片乱序发送（测 seq 处理）
--silent-audio       下发全零音频（测 totalAudioEnergy 类断言）
--leak-after-cancel  取消后仍继续推 N 帧（冒充"服务端没停机"的缺陷，用于验证漏音检测）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000


def _b64_tone(ms: float, freq: float = 660.0, amp: float = 0.2, silent: bool = False) -> str:
    n = int(SAMPLE_RATE * ms / 1000.0)
    if silent:
        pcm = np.zeros(n, dtype=np.int16)
    else:
        t = np.arange(n) / SAMPLE_RATE
        pcm = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


class MockConfig:
    def __init__(self, **kw: Any) -> None:
        self.ttft_ms = float(kw.get("ttft_ms", 400.0))
        self.first_audio_ms = float(kw.get("first_audio_ms", 600.0))
        self.chunk_ms = float(kw.get("chunk_ms", 20.0))
        self.total_audio_ms = float(kw.get("total_audio_ms", 3000.0))
        self.cancel_honor_ms = float(kw.get("cancel_honor_ms", 150.0))
        self.dup_delta_rate = float(kw.get("dup_delta_rate", 0.0))
        self.out_of_order = bool(kw.get("out_of_order", False))
        self.silent_audio = bool(kw.get("silent_audio", False))
        self.leak_after_cancel = int(kw.get("leak_after_cancel", 0))
        self.seed = int(kw.get("seed", 7))

    def ground_truth(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "first_audio_ms": self.first_audio_ms,
            "chunk_ms": self.chunk_ms,
            "total_audio_ms": self.total_audio_ms,
            "cancel_honor_ms": self.cancel_honor_ms,
            "dup_delta_rate": self.dup_delta_rate,
            "out_of_order": self.out_of_order,
            "silent_audio": self.silent_audio,
            "leak_after_cancel": self.leak_after_cancel,
        }


class Session:
    """一条连接的会话状态机（单请求串行，够用且好推理）。"""

    def __init__(self, cfg: MockConfig, ws: Any) -> None:
        self.cfg = cfg
        self.ws = ws
        self.rng = random.Random(cfg.seed)
        self.last_input_ns = 0
        self.last_speech_ns = 0
        self.speech_seen = False
        self.speech_started = False
        self.silence_ns = 100_000_000  # 服务端 VAD：连续 100ms 静音判定话毕
        self.energy_threshold = 120.0
        self.speech_ended = asyncio.Event()
        self.cancel_event = asyncio.Event()
        self.cancelled = False
        self.response_id: str | None = None
        self.seq = 0
        self.ground_truth_log: dict[str, Any] = {}

    async def send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    async def on_message(self, raw: str | bytes) -> None:
        if isinstance(raw, (bytes, bytearray)):
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        mtype = data.get("type", "")

        if mtype == "session.update":
            await self.send({"type": "session.created", "session": {"id": "mock_session"}})
            await self.send({"type": "session.updated"})
            return

        if mtype == "input_audio_buffer.append":
            now = time.monotonic_ns()
            self.last_input_ns = now
            pcm = base64.b64decode(data.get("audio") or "")
            arr = np.frombuffer(pcm, dtype=np.int16)
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) if arr.size else 0.0
            if rms > self.energy_threshold:
                self.last_speech_ns = now
                if not self.speech_seen:
                    self.speech_seen = True
                    await self.send({"type": "input_audio_buffer.speech_started"})
            elif (
                self.speech_seen
                and self.last_speech_ns
                and now - self.last_speech_ns >= self.silence_ns
                and not self.speech_ended.is_set()
            ):
                self.speech_ended.set()
            return

        if mtype == "response.cancel":
            self.cancel_event.set()
            return

    async def cancel_watcher(self) -> None:
        await self.cancel_event.wait()
        self.cancelled = True
        self.ground_truth_log["cancel_received_ns"] = time.monotonic_ns()

    async def run_response(self) -> None:
        """等语音结束 → 按已知时间表推送响应。"""
        await self.speech_ended.wait()
        t_end = self.last_speech_ns
        self.ground_truth_log["speech_end_ns"] = t_end

        # 首个文本增量
        await asyncio.sleep(max(0.0, (t_end + self.cfg.ttft_ms * 1e6 - time.monotonic_ns()) / 1e9))
        if self.cancelled:
            return
        self.response_id = f"resp_{int(time.time() * 1000)}"
        await self.send({"type": "response.created", "response": {"id": self.response_id}})
        self.ground_truth_log["response_created_ns"] = time.monotonic_ns()
        await self.send({"type": "response.text.delta", "response_id": self.response_id, "delta": "好"})
        self.ground_truth_log["first_text_ns"] = time.monotonic_ns()

        # 音频
        await asyncio.sleep(
            max(0.0, (t_end + self.cfg.first_audio_ms * 1e6 - time.monotonic_ns()) / 1e9)
        )
        payload = _b64_tone(self.cfg.chunk_ms, silent=self.cfg.silent_audio)
        sent = 0
        n_chunks = int(self.cfg.total_audio_ms / self.cfg.chunk_ms)
        pending_swap: dict[str, Any] | None = None

        leaked = 0
        while sent < n_chunks:
            if self.cancelled and leaked < self.cfg.leak_after_cancel:
                # 缺陷注入：取消后继续推帧（模拟服务端未停机）
                await self.send(
                    {
                        "type": "response.audio.delta",
                        "response_id": self.response_id,
                        "sequence_id": self.seq,
                        "delta": payload,
                    }
                )
                self.seq += 1
                leaked += 1
                await asyncio.sleep(self.cfg.chunk_ms / 1000.0)
                continue

            if self.cancelled:
                # 兑现取消：等待 cancel_honor_ms 后停推
                honor_ns = self.ground_truth_log.get("cancel_received_ns", 0) + int(
                    self.cfg.cancel_honor_ms * 1e6
                )
                await asyncio.sleep(max(0.0, (honor_ns - time.monotonic_ns()) / 1e9))
                await self.send({"type": "response.cancelled", "response_id": self.response_id})
                self.ground_truth_log["cancelled_ns"] = time.monotonic_ns()
                return

            msg = {
                "type": "response.audio.delta",
                "response_id": self.response_id,
                "sequence_id": self.seq,
                "delta": payload,
            }
            if sent == 0:
                self.ground_truth_log["first_audio_ns"] = time.monotonic_ns()

            if self.cfg.out_of_order and pending_swap is not None:
                await self.send(pending_swap)
                pending_swap = None
                await self.send(msg)
            elif self.cfg.out_of_order and self.rng.random() < 0.3:
                pending_swap = msg  # 押后，下一轮先发
            else:
                await self.send(msg)

            if self.cfg.dup_delta_rate and self.rng.random() < self.cfg.dup_delta_rate:
                await self.send(msg)  # 重复分片

            self.seq += 1
            sent += 1
            await asyncio.sleep(self.cfg.chunk_ms / 1000.0)

        await self.send({"type": "response.done", "response": {"id": self.response_id}})
        self.ground_truth_log["response_done_ns"] = time.monotonic_ns()


async def serve(cfg: MockConfig, host: str, port: int, truth_path: Path) -> None:
    import websockets

    async def handler(ws: Any) -> None:
        sess = Session(cfg, ws)
        watcher = asyncio.create_task(sess.cancel_watcher())
        responder = asyncio.create_task(sess.run_response())
        try:
            async for raw in ws:
                await sess.on_message(raw)
                if sess.ground_truth_log.get("response_done_ns") or sess.ground_truth_log.get(
                    "cancelled_ns"
                ):
                    break
        except Exception as exc:  # noqa: BLE001 - 连接断开对 mock 是正常事件
            print(f"[mock] connection closed: {type(exc).__name__}: {exc}")
        finally:
            responder.cancel()
            watcher.cancel()
            truth_path.parent.mkdir(parents=True, exist_ok=True)
            base = sess.ground_truth_log.get("speech_end_ns") or 0
            truth = {
                "config": cfg.ground_truth(),
                "observed_by_server": {
                    k: round((v - base) / 1e6, 3) for k, v in sess.ground_truth_log.items() if v and base
                },
            }
            truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    async with websockets.serve(handler, host, port, ping_interval=None):
        print(f"mock server: ws://{host}:{port}")
        print(f"ground truth: {json.dumps(cfg.ground_truth(), ensure_ascii=False)}")
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser(description="可注入故障的本地假服务端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--ttft-ms", type=float, default=400.0)
    ap.add_argument("--first-audio-ms", type=float, default=600.0)
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--total-audio-ms", type=float, default=3000.0)
    ap.add_argument("--cancel-honor-ms", type=float, default=150.0)
    ap.add_argument("--dup-delta-rate", type=float, default=0.0)
    ap.add_argument("--out-of-order", action="store_true")
    ap.add_argument("--silent-audio", action="store_true")
    ap.add_argument("--leak-after-cancel", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--truth", default="reports/mock_server/ground_truth.json")
    args = ap.parse_args()

    cfg = MockConfig(
        ttft_ms=args.ttft_ms,
        first_audio_ms=args.first_audio_ms,
        chunk_ms=args.chunk_ms,
        total_audio_ms=args.total_audio_ms,
        cancel_honor_ms=args.cancel_honor_ms,
        dup_delta_rate=args.dup_delta_rate,
        out_of_order=args.out_of_order,
        silent_audio=args.silent_audio,
        leak_after_cancel=args.leak_after_cancel,
        seed=args.seed,
    )
    asyncio.run(serve(cfg, args.host, args.port, Path(args.truth)))


if __name__ == "__main__":
    main()
