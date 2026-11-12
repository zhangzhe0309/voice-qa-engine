"""用真实扬声器 + loopback 回采，测量"打断刹车"的物理残音。

这个脚本回答一个此前只能靠 mock 声称的问题：
**调用 stop 之后，扬声器到底还在响多久？**

方法
----
1. 以 20ms 节奏把 1kHz 语音样音频块真正推过 PlaybackEngine/WasapiLoopbackSink；
2. 播到第 k 块时调用 stop_and_flush()，记录 t_stop（单调时钟）；
3. 继续回采 400ms，取 20ms RMS 包络，找最后一个 > -45dBFS 的帧；
4. physical_stop_latency_ms = t_last_audible - t_stop - device_latency_const。

判据（文档 4.2 口径）：残音 ≤ 1 帧（20ms）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from qa_real.loopback_sink import PhysicalTrace, WasapiLoopbackSink, summarize_traces


def tone_chunk(n: int, sample_rate: int, freq: float = 1000.0, amplitude: float = 0.35) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


async def run_trial(
    sink: WasapiLoopbackSink, sample_rate: int, chunks: int, chunk_ms: float
) -> PhysicalTrace:
    n = int(sample_rate * chunk_ms / 1000.0)
    chunk = tone_chunk(n, sample_rate)
    for _ in range(chunks):
        await sink.play_chunk(chunk)
        await asyncio.sleep(chunk_ms / 1000.0)
    await sink.stop_and_flush()
    return await sink.finish(tail_ms=400.0)


async def main_async(trials: int, chunks: int, chunk_ms: float, device_const_ms: float) -> dict:
    sink = WasapiLoopbackSink(device_latency_ms=device_const_ms)
    traces: list[PhysicalTrace] = []
    try:
        for i in range(trials):
            trace = await run_trial(sink, sink.sample_rate, chunks, chunk_ms)
            traces.append(trace)
            print(
                f"  trial {i}: chunks={trace.played_chunks} "
                f"stop_calls={len(trace.stop_calls)} "
                f"physical_stop={trace.physical_stop_latency_ms()} ms"
            )
            # 每轮重建流，避免 stop 之后的状态影响下一轮
            if i != trials - 1:
                sink.close()
                sink = WasapiLoopbackSink(device_latency_ms=device_const_ms)
    finally:
        sink.close()
    summary = summarize_traces(traces)
    summary["measurement_source"] = "physical_loopback"
    summary["sample_rate"] = 48000
    summary["chunk_ms"] = chunk_ms
    summary["chunks_before_stop"] = chunks
    summary["device_latency_const_ms"] = device_const_ms
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["evidence"] = [t.to_dict() for t in traces]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="物理打断刹车时延与残音测量")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=25, help="停止前播放的 20ms 块数")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument(
        "--calibration",
        type=str,
        default="reports/calibration/calibration.json",
        help="标定结果，用于扣除设备常数",
    )
    ap.add_argument("--out", type=str, default="reports/physical_bargein/probe.json")
    args = ap.parse_args()

    const = 0.0
    cal_path = Path(args.calibration)
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        const = float(cal.get("device_latency_const_ms") or 0.0)
        print(f"calibration: device_latency_const = {const} ms")
    else:
        print("calibration: 未找到，device_latency_const = 0ms（结果会内含驱动缓冲）")

    result = asyncio.run(main_async(args.trials, args.chunks, args.chunk_ms, const))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lat = result["physical_stop_latency_ms"]
    print(f"\nphysical_stop_latency_ms : {json.dumps(lat, ensure_ascii=False)}")
    print(f"residual_frames          : {json.dumps(result['residual_frames'], ensure_ascii=False)}")
    print(f"assertions               : {json.dumps(result['assertions'], ensure_ascii=False)}")
    print(f"saved                    : {out}")


if __name__ == "__main__":
    main()
