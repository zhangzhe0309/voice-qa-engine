#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 流式音频输入源 (Streaming Audio Input)
支持运行时动态注入 PCM 片段（打断/附和/背景争吵定时注入），空队列时持续产出静音帧，
满足"全时上行推流 (Continuous Audio Ingestion)"契约与 30 分钟长时会话的内存约束。
"""

import asyncio
import time
from collections import deque
from typing import Optional

import numpy as np

from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.audio.input.base import AudioInput


class StreamingAudioInput(AudioInput):
    """
    动态注入式音频输入源。

    - push_segment(): 将 PCM 片段 (float32 [-1,1]) 排入待发队列
    - read(): 队列有数据则按 20ms 帧切分产出；队列空则产出静音帧维持长连接
    - close(): 终止产出循环（读完队列内残留后返回 None）
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 20.0,
        silence_after_drain: bool = True,
        real_time_pace: bool = True,
    ):
        self._format = AudioFormat(
            sample_rate=sample_rate,
            channels=1,
            sample_format=SampleFormat.PCM_F32LE,
        )
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = int(round(sample_rate * frame_duration_ms / 1000.0))
        self.silence_after_drain = silence_after_drain
        self.real_time_pace = real_time_pace

        self._pending: deque[np.ndarray] = deque()
        self._carry: Optional[np.ndarray] = None
        self._sequence_id = 0
        self._is_open = False
        self._closed = False
        self._data_available = asyncio.Event()
        self.total_pushed_samples = 0
        self.total_silence_frames = 0

    async def open(self) -> None:
        self._is_open = True
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        self._data_available.set()
        self._is_open = False

    def push_segment(self, samples: np.ndarray) -> None:
        """运行时注入 PCM 片段（float32 归一化单声道）"""
        if not self._is_open:
            raise RuntimeError("StreamingAudioInput is not open")
        arr = np.asarray(samples, dtype=np.float32).ravel()
        if arr.size == 0:
            return
        self._pending.append(arr)
        self.total_pushed_samples += arr.size
        self._data_available.set()

    async def read(self) -> Optional[AudioFrame]:
        if not self._is_open:
            raise RuntimeError("StreamingAudioInput is not open")

        while True:
            chunk = self._next_chunk()
            if chunk is not None:
                return await self._pace(self._make_frame(chunk))

            if self._closed:
                return None
            if not self.silence_after_drain:
                await self._data_available.wait()
                self._data_available.clear()
                continue

            # 全时推流契约：空队列产出静音帧
            self.total_silence_frames += 1
            return await self._pace(self._make_frame(np.zeros(self.samples_per_frame, dtype=np.float32)))

    async def _pace(self, frame: AudioFrame) -> AudioFrame:
        if self.real_time_pace:
            # 实时节流："全时推流"的真实语义 = 20ms 帧按 20ms 速率产出。
            # 无节流的 sleep(0) 热轮询会以数百万帧/秒伪造推流并淹没流式落盘与事件链。
            await asyncio.sleep(self.frame_duration_ms / 1000.0)
        else:
            await asyncio.sleep(0)  # 显式 fast 模式：让出事件循环
        return frame

    def _next_chunk(self) -> Optional[np.ndarray]:
        if self._carry is not None and self._carry.size >= self.samples_per_frame:
            chunk = self._carry[: self.samples_per_frame]
            self._carry = self._carry[self.samples_per_frame:]
            if self._carry.size == 0:
                self._carry = None
            return chunk

        while self._pending:
            data = self._pending.popleft()
            combined = data if self._carry is None else np.concatenate([self._carry, data])
            if combined.size >= self.samples_per_frame:
                chunk = combined[: self.samples_per_frame]
                self._carry = combined[self.samples_per_frame:]
                return chunk
            self._carry = combined
        return None

    def _make_frame(self, chunk: np.ndarray) -> AudioFrame:
        frame = AudioFrame(
            samples=chunk.astype(np.float32),
            format=self._format,
            timestamp_ns=time.monotonic_ns(),
            sequence_id=self._sequence_id,
            duration_ms=self.frame_duration_ms,
        )
        self._sequence_id += 1
        return frame
