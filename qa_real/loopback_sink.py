"""真实物理播放 Sink + 物理刹车时延/残音测量。

替代 MockPlaybackSink：把音频真正送进 WASAPI，同时用 loopback 回采，
从而把"扬声器到底还在响多久"变成一段可以测量的波形，而不是一个布尔标志。

真值链
------
app 写帧  ->  WASAPI 混音  ->  loopback 回采(20ms RMS 包络)
                                    |
                        t_stop 调用时刻与之比较
                                    |
                physical_stop_latency_ms = t_last_audible - t_stop

`t_last_audible` 取包络中最后一个 > -45 dBFS 的帧末尾，
与文档 4.2「残音 ≤ 1 帧 (<-45dBFS)」口径一致。
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qa_real._paudio import pa_module
from qa_real.calibrate import CallbackCapture
from qa_real.devices import find_loopback, find_mic, find_output
from qa_real.probe_signal import envelope_db, last_audible_frame_index, summarize_ms


def _float_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0).__mul__(32767.0).astype(np.int16)


@dataclass
class PhysicalTrace:
    """一次真实播放的物理观测结果。"""

    t_stop_ns: int | None
    capture_start_ns: int | None
    envelope_db: list[float]
    frame_ms: float
    device_latency_ms: float
    started_ns: int = 0
    played_chunks: int = 0
    stop_calls: list[int] = field(default_factory=list)

    def physical_stop_latency_ms(self) -> float | None:
        if self.t_stop_ns is None or self.capture_start_ns is None or not self.envelope_db:
            return None
        db = np.asarray(self.envelope_db, dtype=np.float64)
        last = last_audible_frame_index(db, -45.0)
        if last < 0:
            return 0.0
        t_last_ns = self.capture_start_ns + int((last + 1) * self.frame_ms * 1e6)
        return round((t_last_ns - self.t_stop_ns) / 1e6, 3)

    def to_dict(self) -> dict[str, Any]:
        lat = self.physical_stop_latency_ms()
        db = np.asarray(self.envelope_db, dtype=np.float64) if self.envelope_db else np.zeros(0)
        audio_start = int(np.argmax(db > -45.0)) if db.size and (db > -45.0).any() else -1
        return {
            "measurement_source": "physical_loopback",
            "played_chunks": self.played_chunks,
            "physical_stop_latency_ms": lat,
            "residual_frames_le_45db": (round(lat / self.frame_ms, 2) if lat is not None else None),
            "audible_frames": int((db > -45.0).sum()) if db.size else 0,
            "first_audible_frame": audio_start,
            "device_latency_ms": round(self.device_latency_ms, 3),
            "stop_calls": len(self.stop_calls),
        }


class WasapiLoopbackSink:
    """把 PlaybackChunk 真正播到扬声器，并用 loopback 回采。"""

    def __init__(
        self,
        sample_rate: int = 48000,
        device_latency_ms: float = 0.0,
        frame_ms: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.device_latency_ms = device_latency_ms
        self.frame_ms = frame_ms
        p = pa_module().PyAudio()
        self._p = p
        self.lb = find_loopback()
        self.out = find_output(self.lb)
        self.mic = find_mic()
        self._out_stream = p.open(
            format=pa_module().paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=self.out.index,
            frames_per_buffer=480,
        )
        self._capture = CallbackCapture(p, self.lb.index, sample_rate, min(2, max(1, self.lb.in_ch)))
        self._capture.start()
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._played = 0
        self._stop_calls: list[int] = []

    def _write_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                self._out_stream.write(item)
            except (OSError, RuntimeError):
                break

    async def play_chunk(self, samples: np.ndarray) -> None:
        pcm = _float_to_int16(samples)
        stereo = np.repeat(pcm, 2)
        self._q.put(stereo.tobytes())
        self._played += 1

    async def stop_and_flush(self) -> None:
        """记录停止时刻并立刻停流（不清空 PC 侧缓冲，让 loopback 去测真实残音）。"""
        self._stop_calls.append(time.monotonic_ns())
        await asyncio.to_thread(self._hard_stop)

    def snapshot_tail(self, seconds: float = 0.4) -> np.ndarray:
        """当前采集缓冲中最近 seconds 秒的回采样点（不停流）。"""
        return self._capture.snapshot(seconds)

    def resume(self) -> None:
        """模拟"新话轮重新开始播放"：重开输出流（采集端保持不动）。"""
        if not self._out_stream.is_active():
            self._out_stream.start_stream()

    def _hard_stop(self) -> None:
        with contextlib.suppress(Exception):
            self._out_stream.stop_stream()

    async def finish(self, tail_ms: float = 400.0) -> PhysicalTrace:
        """停止采集并给出物理观测。"""
        await asyncio.sleep(tail_ms / 1000.0)
        pcm = self._capture.stop()
        db, frame_ms = envelope_db(pcm, self.sample_rate, self.frame_ms)
        trace = PhysicalTrace(
            t_stop_ns=self._stop_calls[0] if self._stop_calls else None,
            capture_start_ns=self._capture.first_sample_ns,
            envelope_db=[round(float(x), 2) for x in db],
            frame_ms=frame_ms,
            device_latency_ms=self.device_latency_ms,
            played_chunks=self._played,
            stop_calls=list(self._stop_calls),
        )
        return trace

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._q.put_nowait(None)
        with contextlib.suppress(Exception):
            self._out_stream.stop_stream()
            self._out_stream.close()
        with contextlib.suppress(Exception):
            self._writer.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._p.terminate()


def summarize_traces(traces: list[PhysicalTrace]) -> dict[str, Any]:
    lats = [t.physical_stop_latency_ms() for t in traces]
    vals = [v for v in lats if v is not None]
    frames = [round(v / traces[0].frame_ms, 2) for v in vals] if vals else []
    return {
        "n": len(vals),
        "physical_stop_latency_ms": summarize_ms(vals),
        "residual_frames": summarize_ms(frames),
        "assertions": {
            "residual_le_1_frame": (all(f <= 1.0 for f in frames) if frames else None),
            "residual_le_20ms": (all(v <= 20.0 for v in vals) if vals else None),
        },
        "per_trial_ms": lats,
    }
