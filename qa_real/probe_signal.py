"""探测信号与到达时刻检测。

用 1kHz Hann 窗脉冲串做物理探针：
- 播放端知道每个 click 的"写入时刻"
- 回采端用匹配滤波定位每个 click 的"到达样点"
两者相减即为该路径的真实物理偏移（含驱动/缓冲/换能器/空气）。

为什么不用互相关整段：脉冲间隔固定，逐 click 匹配滤波可以同时给出
偏移的 P50/P95 与漂移（jitter），避免单次测量掩盖抖动。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ClickTrain:
    samples: np.ndarray          # float32 mono, [-1, 1]
    sample_rate: int
    click_offsets_samples: list[int]
    click_template: np.ndarray   # 单个 click 的波形（用于匹配滤波）


def make_click_train(
    sample_rate: int = 48000,
    click_count: int = 10,
    click_ms: float = 5.0,
    gap_ms: float = 300.0,
    freq_hz: float = 1000.0,
    amplitude: float = 0.5,
    lead_in_ms: float = 200.0,
) -> ClickTrain:
    """生成 click 串。每个 click 是 Hann 窗调制的 1kHz 正弦。"""
    click_n = int(sample_rate * click_ms / 1000.0)
    gap_n = int(sample_rate * gap_ms / 1000.0)
    lead_n = int(sample_rate * lead_in_ms / 1000.0)
    total = lead_n + gap_n * click_count + click_n

    template = np.zeros(click_n, dtype=np.float64)
    t = np.arange(click_n) / sample_rate
    template[:] = amplitude * np.sin(2 * np.pi * freq_hz * t)
    template *= np.hanning(click_n)

    buf = np.zeros(total, dtype=np.float64)
    offsets: list[int] = []
    for i in range(click_count):
        start = lead_n + i * gap_n
        buf[start:start + click_n] += template
        offsets.append(start)

    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = buf / peak * min(1.0, amplitude)

    return ClickTrain(
        samples=buf.astype(np.float32),
        sample_rate=sample_rate,
        click_offsets_samples=offsets,
        click_template=template.astype(np.float32),
    )


def detect_click_onsets(
    captured: np.ndarray,
    template: np.ndarray,
    sample_rate: int,
    expected_count: int,
    min_gap_ms: float = 100.0,
    rel_threshold: float = 0.25,
) -> list[int]:
    """匹配滤波定位 click 起始样点。

    返回 [起点下标...]。用归一化互相关，阈值相对于最强峰，
    并对相邻峰施加最小间隔约束（click 间隔 300ms，最小间隔 100ms 足够）。
    """
    x = np.asarray(captured, dtype=np.float64).ravel()
    h = np.asarray(template, dtype=np.float64).ravel()
    if x.size < h.size or h.size == 0:
        return []

    # FFT 互相关（非归一化的峰位置已足够；阈值按相对能量判定）
    n = 1 << int(np.ceil(np.log2(x.size + h.size)))
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(h, n)), n)[: x.size - h.size + 1]
    energy = float(np.dot(h, h))
    score = corr / (energy + 1e-12)   # 归一化到"若完全对齐则≈1"

    # 局部极大 + 阈值
    abs_score = np.abs(score)
    peak = float(abs_score.max()) if abs_score.size else 0.0
    if peak <= 0:
        return []
    thresh = peak * rel_threshold

    min_gap_n = int(sample_rate * min_gap_ms / 1000.0)
    candidates = np.where(abs_score >= thresh)[0]

    # 按分数降序贪心取峰，保证最小间隔
    order = candidates[np.argsort(abs_score[candidates])[::-1]]
    picked: list[int] = []
    for idx in order:
        if all(abs(int(idx) - p) >= min_gap_n for p in picked):
            picked.append(int(idx))
        if len(picked) >= expected_count:
            break
    return sorted(picked)


def envelope_db(
    pcm: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    ref: float = 32768.0,
) -> tuple[np.ndarray, float]:
    """20ms 帧 RMS 包络（dBFS），与文档 -45dBFS 口径一致。

    作为打断残音/静默期判定的物理真值源。
    """
    x = np.asarray(pcm, dtype=np.float64).ravel()
    frame_n = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = x.size // frame_n
    if n_frames == 0:
        return np.zeros(0), frame_ms
    frames = x[: n_frames * frame_n].reshape(n_frames, frame_n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / ref)
    return db, frame_ms


def last_audible_frame_index(db: np.ndarray, threshold_db: float = -45.0) -> int:
    """最后一个超过阈值的帧下标；无则 -1。"""
    idx = np.where(db > threshold_db)[0]
    return int(idx[-1]) if idx.size else -1


def first_audible_frame_index(db: np.ndarray, threshold_db: float = -45.0) -> int:
    idx = np.where(db > threshold_db)[0]
    return int(idx[0]) if idx.size else -1


def summarize_ms(values: Sequence[float]) -> dict:
    """P50/P95/P99/最小/最大/N —— 所有概率指标必须带 N 与分位数。"""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
    }
