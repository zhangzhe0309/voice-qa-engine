"""变异测试 1 号：幽灵音频（stale generation chunk）。

为什么这个变异体重要
--------------------
现有仓库只用"事件时间轴里有没有 PLAYBACK_STARTED"来断言无幽灵音频，
那是**逻辑自证**：只要记录事件的那段代码被绕过（或本身就是 mock），断言永远通过。
本变异体把判据换成 **loopback 波形的 3kHz 带能量** —— 幽灵音频哪怕只漏一帧，
也必须被物理回采抓到。

三个变体（缺一不可，否则是自证）
------------------------------
1. positive_control : 绕过引擎直接播 3kHz → 检测器必须报告"有幽灵音频"（证明检测器有牙）
2. guarded          : 真实 PlaybackEngine + 真代次守卫 → 应报告"无幽灵音频"
3. mutant           : 关掉代次守卫（模拟代码退化）→ 检测器必须报告"有幽灵音频"（变异体被杀死）

mutation_score = 被杀死数 / 变异体总数。
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

from qa_real.loopback_sink import WasapiLoopbackSink
from voice_qa_engine.audio.models import AudioFormat, AudioFrame, SampleFormat
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.queue import PlaybackChunk
from voice_qa_engine.playback.sink import PlaybackSink

GHOST_FREQ_HZ = 3000.0
GHOST_BAND_HZ = 200.0
GHOST_ENERGY_RATIO_THRESHOLD = 0.05  # 3kHz 带能量占全带比例阈值


class EngineSinkAdapter(PlaybackSink):
    """把 PlaybackEngine 的 PlaybackSink 接口接到真实 WASAPI 播放。"""

    def __init__(self, sink: WasapiLoopbackSink) -> None:
        self.sink = sink
        self.played: list[int] = []

    async def play(self, chunk: PlaybackChunk) -> None:
        self.played.append(chunk.sequence_id)
        await self.sink.play_chunk(chunk.audio.samples)

    async def stop_and_flush(self) -> None:
        await self.sink.stop_and_flush()

    async def close(self) -> None:
        return None


def make_chunk(
    generation_id: int,
    sequence_id: int,
    freq: float,
    sample_rate: int = 48000,
    duration_ms: float = 20.0,
    amplitude: float = 0.35,
    response_id: str = "resp_probe",
) -> PlaybackChunk:
    n = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(n) / sample_rate
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    frame = AudioFrame(
        samples=samples,
        format=AudioFormat(sample_rate, 1, SampleFormat.PCM_F32LE),
        timestamp_ns=time.monotonic_ns(),
        sequence_id=sequence_id,
        duration_ms=duration_ms,
    )
    return PlaybackChunk(
        session_id="probe_session",
        response_id=response_id,
        generation_id=generation_id,
        sequence_id=sequence_id,
        audio=frame,
        received_monotonic_ns=time.monotonic_ns(),
        expected_duration_ms=duration_ms,
        cancel_token_id=f"tok_{generation_id}",
        connection_epoch=1,
    )


def ghost_energy_ratio(pcm: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """回采波形中 3kHz 带能量占比 —— 幽灵音频的物理判据。"""
    x = np.asarray(pcm, dtype=np.float64).ravel()
    if x.size < 512:
        return {"ghost_detected": False, "note": "capture_too_short", "ghost_band_ratio": 0.0}
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    band = (freqs >= GHOST_FREQ_HZ - GHOST_BAND_HZ) & (freqs <= GHOST_FREQ_HZ + GHOST_BAND_HZ)
    total = float(np.sum(spec**2)) + 1e-12
    band_e = float(np.sum(spec[band] ** 2))
    ratio = band_e / total
    return {
        "ghost_detected": bool(ratio >= GHOST_ENERGY_RATIO_THRESHOLD),
        "ghost_band_ratio": round(ratio, 4),
        "rms": round(float(np.sqrt(np.mean(x**2))), 2),
        "samples": int(x.size),
    }


async def run_variant(variant: str, chunks: int) -> dict[str, Any]:
    sink = WasapiLoopbackSink(device_latency_ms=0.0)
    adapter = EngineSinkAdapter(sink)
    engine = PlaybackEngine(sink=adapter, timeline=None)

    if variant in ("mutant", "mutant_push_only"):
        # 关闭 push 侧代次守卫（模拟回归缺陷）
        async def _no_guard(chunk: PlaybackChunk) -> bool:
            with contextlib.suppress(asyncio.QueueFull):
                engine.queue.put_nowait(chunk)
                return True
            return False

        engine.push_chunk = _no_guard  # type: ignore[method-assign]

    accepted = False
    try:
        await engine.start()
        gen1 = await engine.begin_response("resp_probe")
        for i in range(chunks):
            await engine.push_chunk(make_chunk(gen1, i, 1000.0))
            await asyncio.sleep(0.02)

        await engine.interrupt(reason="user_barge_in")
        await sink.stop_and_flush()
        await asyncio.sleep(0.12)
        sink.resume()

        if variant == "mutant":
            # 再关掉 Actor 侧的二次守卫（模拟"打断自增代次"的原子性丢失）
            engine.generation_id = gen1
            engine.cancel_token = None

        stale = make_chunk(gen1, 999, GHOST_FREQ_HZ)
        if variant == "positive_control":
            # 绕过引擎，直接播：证明检测器看得见幽灵音频
            await sink.play_chunk(stale.audio.samples)
            accepted = True
            await asyncio.sleep(0.25)
            pcm = await _collect_tail(sink)
        else:
            accepted = await engine.push_chunk(stale)
            await asyncio.sleep(0.25)
            pcm = await _collect_tail(sink)
    finally:
        with contextlib.suppress(Exception):
            await engine.close()
        sink.close()

    analysis = ghost_energy_ratio(pcm, sink.sample_rate)
    return {
        "variant": variant,
        "stale_chunk_accepted": bool(accepted),
        "played_sequences": adapter.played,
        **analysis,
    }


async def _collect_tail(sink: WasapiLoopbackSink) -> np.ndarray:
    """再采一小段，用于频谱判定。"""
    await asyncio.sleep(0.05)
    return sink.snapshot_tail(seconds=0.4)


def judge(results: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["variant"]: r for r in results}
    detector_ok = bool(by["positive_control"]["ghost_detected"])
    guarded_ok = not by["guarded"]["ghost_detected"]
    mutant_killed = bool(by["mutant"]["ghost_detected"])
    push_only_survived = "mutant_push_only" in by and not by["mutant_push_only"]["ghost_detected"]
    return {
        "detector_has_teeth": detector_ok,
        "guarded_engine_clean": guarded_ok,
        "mutant_killed": mutant_killed,
        "equivalent_mutant_survived": push_only_survived,
        "mutation_score": round((1.0 if mutant_killed else 0.0) / 1.0, 3),
        "verdict": ("PASS" if (detector_ok and guarded_ok and mutant_killed) else "FAIL"),
        "why": {
            "detector_has_teeth": "positive_control 直接播 3kHz 必须被判为幽灵音频（检测器有牙）",
            "guarded_engine_clean": "真实双层代次守卫下 stale chunk 必须被拒且无声",
            "mutant_killed": "关掉 push+Actor 两层守卫后，必须被物理回采抓到",
            "equivalent_mutant_survived": (
                "只关 push 侧守卫时未漏音 —— 说明 Actor 二次校验是真防线，该变异体等价存活"
            ),
        },
    }


async def main_async(chunks: int) -> dict[str, Any]:
    results = []
    for variant in ("positive_control", "guarded", "mutant_push_only", "mutant"):
        r = await run_variant(variant, chunks)
        results.append(r)
        print(
            f"  {variant:16s} accepted={r['stale_chunk_accepted']!s:5s} "
            f"ghost={r['ghost_detected']!s:5s} ratio={r['ghost_band_ratio']}"
        )
    data = {
        "kind": "mutation_test",
        "mutation_id": "MUT-005-GHOST-AUDIO",
        "detection_source": "physical_loopback_3khz_band_energy",
        "results": results,
        "judgement": judge(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="幽灵音频变异测试（物理回采判据）")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--out", type=str, default="reports/mutations/ghost_audio.json")
    args = ap.parse_args()
    data = asyncio.run(main_async(args.chunks))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\njudgement: {json.dumps(data['judgement'], ensure_ascii=False)}")
    print(f"saved    : {out}")


if __name__ == "__main__":
    main()
