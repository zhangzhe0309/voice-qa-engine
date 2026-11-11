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
