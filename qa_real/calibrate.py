"""物理延迟标定：先验证尺子，再测量被测系统。

测量什么
--------
- `render_offset_ms`：应用 write() 调用 → WASAPI loopback 采到该样点。
  这是"客户端播放管线常数"（驱动缓冲 + WASAPI 混音 + 采集回调），
  后续所有播放侧时延都必须扣掉它才能谈物理量。
- `air_offset_ms`：同一 write() → 麦克风采到。差值即换能器 + 空气 + ADC
  的额外代价。只有这个量才能支撑文档里"扬声器真实震动出声"的说法。

为什么必须标定
--------------
不标定的话 `P95 <= 300ms` 里到底有多少是模型、多少是本机声卡缓冲，永远分不清。
所以 session.json 里必须落 `device_latency_const_ms`，否则跨机器数字不可比。

工程要点（踩过的坑）
--------------------
1. 必须用 **callback 流**而不是阻塞 read()：阻塞读在 Windows 上会因缓冲滞后
   静默丢样点，绝对样点下标与时间不再线性对应，offset 会震荡几百 ms。
2. 必须用 **整体互相关求全局 lag**，再逐 click 在 ±50ms 窗口内细化；
   把所有命中点都减同一个 t0 会得到锯齿序列，看起来像巨大抖动。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qa_real import devices
from qa_real._paudio import pa_module
from qa_real.probe_signal import ClickTrain, make_click_train, summarize_ms


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0).__mul__(32767.0).astype(np.int16)


class CallbackCapture:
    """callback 采集：无丢样、无阻塞滞后，并记录第一块数据的时刻锚。"""

    def __init__(self, p: Any, device_index: int, sample_rate: int, channels: int, block_n: int = 480) -> None:
        self.sample_rate = sample_rate
        self.channels = max(1, channels)
        self.block_n = block_n
        self._blocks: list[np.ndarray] = []
        self._block_ns: list[int] = []
        self._lock = threading.Lock()
        self.first_sample_ns: int | None = None
        self.reported_latency_ms: float | None = None
        self.status_flags = 0
        self.stream = p.open(
            format=pa_module().paInt16,
            channels=self.channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=block_n,
            stream_callback=self._cb,
        )

    def _cb(self, in_data: Any, frame_count: int, time_info: Any, status_flags: int) -> tuple[None, int]:
        t_ns = time.monotonic_ns()
        frame = np.frombuffer(in_data, dtype=np.int16)
        if self.channels > 1:
            frame = frame.reshape(-1, self.channels)[:, 0]
        with self._lock:
            if self.first_sample_ns is None:
                # 本块的首样点在回调前 frame_count 个样点被采集
                self.first_sample_ns = t_ns - int(frame_count / self.sample_rate * 1e9)
            self._blocks.append(frame.copy())
            self._block_ns.append(t_ns)
            if status_flags:
                self.status_flags |= int(status_flags)
        return (None, pa_module().paContinue)

    def start(self) -> None:
        with contextlib.suppress(Exception):
            self.reported_latency_ms = round(float(self.stream.get_input_latency()) * 1000.0, 2)
        self.stream.start_stream()

    def snapshot(self, seconds: float) -> np.ndarray:
        """不停止采集，返回最近 seconds 秒的样点（用于在线频谱判定）。"""
        with self._lock:
            if not self._blocks:
                return np.zeros(0, dtype=np.int16)
            want = int(self.sample_rate * seconds)
            out: list[np.ndarray] = []
            total = 0
            for blk in reversed(self._blocks):
                out.append(blk)
                total += blk.size
                if total >= want:
                    break
            if not out:
                return np.zeros(0, dtype=np.int16)
            merged = np.concatenate(list(reversed(out)))
            return merged[-want:] if merged.size > want else merged

    def stop(self) -> np.ndarray:
        with contextlib.suppress(Exception):
            self.stream.stop_stream()
            self.stream.close()
        with self._lock:
            if not self._blocks:
                return np.zeros(0, dtype=np.int16)
            return np.concatenate(self._blocks)


def estimate_global_lag(reference: np.ndarray, capture: np.ndarray) -> int:
    """整体互相关：返回 reference[0] 在 capture 中的样点下标（可为负）。"""
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(capture, dtype=np.float64)
    if r.size == 0 or c.size == 0 or c.size < r.size:
        return 0
    n = 1 << int(np.ceil(np.log2(c.size + r.size)))
    corr = np.fft.irfft(np.fft.rfft(c, n) * np.conj(np.fft.rfft(r, n)), n)
    peak = int(np.argmax(corr))
    if peak > n // 2:
        peak -= n
    return peak


def refine_onset(capture: np.ndarray, expected_idx: int, search_n: int) -> int:
    """在期望位置 ±search_n 内取最大绝对样点作为该 click 的落点。"""
    lo = max(0, expected_idx - search_n)
    hi = min(capture.size, expected_idx + search_n)
    if hi <= lo:
        return expected_idx
    seg = np.abs(np.asarray(capture[lo:hi], dtype=np.float64))
    return lo + int(np.argmax(seg))


@dataclass
class CalibrationResult:
    sample_rate: int
    trials: int
    render_offsets_ms: list[float]
    air_offsets_ms: list[float]
    device: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        render = summarize_ms(self.render_offsets_ms)
        air = summarize_ms(self.air_offsets_ms)
        thresh = (render["p95"] + 25.0) if render["n"] else 60.0
        return {
            "kind": "physical_latency_calibration",
            "measurement_source": "physical_loopback_and_microphone",
            "sample_rate": self.sample_rate,
            "trials": self.trials,
            "render_path_offset_ms": render,
            "acoustic_path_offset_ms": air,
            "device_latency_const_ms": render["p50"],
            "air_path_extra_ms": (round(air["p50"] - render["p50"], 3) if air["n"] and render["n"] else None),
            "render_jitter_threshold_ms": round(thresh, 3),
            "device": self.device,
            "raw": self.raw,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": (
                "render_path_offset_ms 是应用 write() 到 loopback 采到的物理偏移，"
                "已包含驱动缓冲；所有播放侧时延测量须扣除该常数。"
                "acoustic_path_offset_ms 额外包含扬声器+空气+麦克风AD，"
                "为 null 表示麦克风未采到探针（房间静音或未接麦）。"
            ),
        }


def measure_path(
    cap: CallbackCapture,
    pcm: np.ndarray,
    tag: str,
    reference: np.ndarray,
    click_offsets: list[int],
    sample_rate: int,
    t0_ns: int,
    trial_info: dict[str, Any],
) -> list[float]:
    """测量一条物理路径的偏移（毫秒）。返回单次 offset 列表。"""
    if pcm.size < reference.size or cap.first_sample_ns is None:
        trial_info[f"{tag}_reason"] = "capture_too_short"
        return []
    lag = estimate_global_lag(reference, pcm)
    base_ns = cap.first_sample_ns + int(lag / sample_rate * 1e9)
    base_ms = (base_ns - t0_ns) / 1e6
    search = int(sample_rate * 0.05)
    deltas: list[float] = []
    for off in click_offsets:
        exp = lag + off
        got = refine_onset(pcm, exp, search)
        deltas.append(round((got - exp) / sample_rate * 1000.0, 3))
    trial_info[f"{tag}_global_lag_ms"] = round(base_ms, 3)
    trial_info[f"{tag}_click_delta_ms"] = deltas
    # 全局偏移 + 首 click 细化误差 = 该路径的单次 offset
    return [round(base_ms + deltas[0], 3)]


def calibrate(
    trials: int = 5,
    loopback_contains: str | None = None,
    click_count: int = 6,
) -> CalibrationResult:
    p = pa_module().PyAudio()
    lb = devices.find_loopback(loopback_contains)
    out = devices.find_output(lb)
    mic = devices.find_mic()
    sample_rate = 48000

    train: ClickTrain = make_click_train(sample_rate=sample_rate, click_count=click_count, gap_ms=200.0)
    ref = train.samples.astype(np.float64)
    lead_n = train.click_offsets_samples[0] if train.click_offsets_samples else 0

    render_offsets: list[float] = []
    air_offsets: list[float] = []
    raw_log: dict[str, Any] = {
        "output_stream_latency_ms": None,
        "loopback_input_latency_ms": None,
        "mic_input_latency_ms": None,
        "per_trial": [],
    }
    raw_dir = Path("reports") / "calibration" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        out_stream = p.open(
            format=pa_module().paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=out.index,
            frames_per_buffer=480,
        )
        with contextlib.suppress(Exception):
            raw_log["output_stream_latency_ms"] = round(float(out_stream.get_output_latency()) * 1000.0, 2)

        pcm_out = _to_int16(ref)
        stereo = np.repeat(pcm_out, 2)

        for t in range(trials):
            lb_cap = CallbackCapture(p, lb.index, sample_rate, min(2, max(1, lb.in_ch)))
            mic_cap = (
                CallbackCapture(p, mic.index, sample_rate, min(2, max(1, mic.in_ch))) if mic else None
            )
            lb_cap.start()
            if mic_cap is not None:
                mic_cap.start()
            time.sleep(0.25)
            t0_ns = time.monotonic_ns()
            step = 4800
            for i in range(0, stereo.size, step):
                out_stream.write(stereo[i : i + step].tobytes())
            time.sleep(0.35)
            lb_pcm = lb_cap.stop()
            mic_pcm = mic_cap.stop() if mic_cap is not None else np.zeros(0, dtype=np.int16)
            raw_log["loopback_input_latency_ms"] = lb_cap.reported_latency_ms
            if mic_cap is not None:
                raw_log["mic_input_latency_ms"] = mic_cap.reported_latency_ms

            trial_info: dict[str, Any] = {"trial": t, "loopback_samples": int(lb_pcm.size)}

            render_offsets.extend(
                measure_path(lb_cap, lb_pcm, "loopback", ref, train.click_offsets_samples, sample_rate, t0_ns, trial_info)
            )
            if mic_cap is not None:
                air_offsets.extend(
                    measure_path(mic_cap, mic_pcm, "mic", ref, train.click_offsets_samples, sample_rate, t0_ns, trial_info)
                )
            raw_log["per_trial"].append(trial_info)

            np.save(raw_dir / f"trial{t}_loopback.npy", lb_pcm)
            if mic_cap is not None:
                np.save(raw_dir / f"trial{t}_mic.npy", mic_pcm)
    finally:
        with contextlib.suppress(Exception):
            p.terminate()

    if render_offsets:
        med = statistics.median(render_offsets)
        render_offsets = [v for v in render_offsets if abs(v - med) <= 2 * (abs(med) * 0.5 + 20.0)]
    raw_log["lead_in_samples"] = lead_n
    return CalibrationResult(
        sample_rate=sample_rate,
        trials=trials,
        render_offsets_ms=render_offsets,
        air_offsets_ms=air_offsets,
        device={
            "loopback": lb.to_dict(),
            "output": out.to_dict(),
            "microphone": mic.to_dict() if mic else None,
        },
        raw=raw_log,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="物理延迟标定（先验证尺子）")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--loopback", type=str, default=None, help="loopback 设备名子串，如 Speaker")
    ap.add_argument("--out", type=str, default="reports/calibration/calibration.json")
    args = ap.parse_args()

    res = calibrate(trials=args.trials, loopback_contains=args.loopback)
    data = res.to_dict()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"loopback device : {data['device']['loopback']['name']}")
    print(f"output device   : {data['device']['output']['name']}")
    mic = data["device"]["microphone"]
    print(f"microphone      : {mic['name'] if mic else 'N/A'}")
    print(f"render path     : {json.dumps(data['render_path_offset_ms'], ensure_ascii=False)}")
    print(f"acoustic path   : {json.dumps(data['acoustic_path_offset_ms'], ensure_ascii=False)}")
    print(f"device const    : {data['device_latency_const_ms']} ms")
    print(f"air path extra  : {data['air_path_extra_ms']} ms")
    print(f"saved           : {out_path}")


if __name__ == "__main__":
    main()
