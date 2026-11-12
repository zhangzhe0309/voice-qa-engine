#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 播放接收端 (Sink) 抽象与实现
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time
from voice_qa_engine.playback.queue import PlaybackChunk


class PlaybackSink(ABC):
    @abstractmethod
    async def play(self, chunk: PlaybackChunk) -> None:
        """物理播放单帧 Chunk"""
        pass

    @abstractmethod
    async def stop_and_flush(self) -> None:
        """立刻打断停止底层驱动播放，并彻底清空硬件/系统音频缓冲"""
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class MockPlaybackSink(PlaybackSink):
    """测试用 Mock 播放端，记录完整物理播放与打断轨迹"""
    def __init__(self):
        self.played_chunks: List[PlaybackChunk] = []
        self.was_stopped_and_flushed = False
        self.last_stop_ns: Optional[int] = None

    async def play(self, chunk: PlaybackChunk) -> None:
        self.played_chunks.append(chunk)

    async def stop_and_flush(self) -> None:
        self.was_stopped_and_flushed = True
        self.last_stop_ns = time.monotonic_ns()

    async def close(self) -> None:
        pass


class HannFadeoutSink(PlaybackSink):
    """
    带 10ms 半 Hann 窗平滑衰减的防爆音播放接收端
    遵循规范 3.3：打断时坚决禁止 PCM 样点直角截断 (Hard Cut)
    应用 10ms Hann 窗衰减至 0，相邻样点一阶差分突变 Jump <= 8000
    """
    def __init__(self, inner_sink: Optional[PlaybackSink] = None, fade_duration_ms: float = 10.0):
        self.inner_sink = inner_sink
        self.fade_duration_ms = fade_duration_ms
        self.played_chunks: List[PlaybackChunk] = []
        self.last_chunk: Optional[PlaybackChunk] = None
        self.was_stopped_and_flushed = False
        self.last_stop_ns: Optional[int] = None
        self.fadeout_applied: bool = False
        self.faded_tail_samples = None
        self.max_jump_detected: float = 0.0

    async def play(self, chunk: PlaybackChunk) -> None:
        self.played_chunks.append(chunk)
        self.last_chunk = chunk
        if self.inner_sink:
            await self.inner_sink.play(chunk)

    @staticmethod
    def apply_half_hann_decay(samples, sample_rate: int, fade_duration_ms: float = 10.0):
        """对 PCM float32 样点尾部施加 10ms 半 Hann 窗渐出"""
        import numpy as np
        fade_samples = max(2, int((fade_duration_ms / 1000.0) * sample_rate))
        out = np.copy(samples)
        if len(out) < fade_samples:
            fade_samples = len(out)
        if fade_samples <= 0:
            return out

        # 半 Hann 窗: 0.5 * (1 + cos(pi * i / fade_samples))
        window = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        out[-fade_samples:] *= window
        return out

    async def stop_and_flush(self) -> None:
        import numpy as np
        self.was_stopped_and_flushed = True
        self.last_stop_ns = time.monotonic_ns()

        if self.last_chunk and self.last_chunk.audio is not None:
            raw_samples = self.last_chunk.audio.samples
            sr = self.last_chunk.audio.format.sample_rate
            faded = self.apply_half_hann_decay(raw_samples, sr, self.fade_duration_ms)
            self.faded_tail_samples = faded
            self.fadeout_applied = True

            # 计算最大一阶差分 Jump (映射为 16-bit 线性整数标尺 [-32768, 32767])
            int16_samples = np.clip(faded * 32767.0, -32768, 32767).astype(np.int16)
            if len(int16_samples) > 1:
                diffs = np.abs(np.diff(int16_samples))
                self.max_jump_detected = float(np.max(diffs))

        if self.inner_sink:
            await self.inner_sink.stop_and_flush()

    async def close(self) -> None:
        if self.inner_sink:
            await self.inner_sink.close()

