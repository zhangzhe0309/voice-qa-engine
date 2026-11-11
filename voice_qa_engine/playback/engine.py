#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 防幽灵音频播放引擎 (PlaybackEngine)
单一 Actor 模式，四位一体 (response_id, generation_id, sequence_id, cancel_token)
彻底消除打断延迟与并发竞态幽灵音频
"""

import asyncio
import time
from typing import Optional

from voice_qa_engine.playback.state import PlaybackState
from voice_qa_engine.playback.queue import PlaybackChunk
from voice_qa_engine.playback.sink import PlaybackSink
from voice_qa_engine.runtime.cancellation import CancelToken
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType


class PlaybackEngine:
    def __init__(self, sink: PlaybackSink, timeline: Optional[EventTimeline] = None):
        self.sink = sink
        self.timeline = timeline
        self.state = PlaybackState.IDLE

        # 核心防幽灵音频四要素
        self.generation_id: int = 0
        self.active_response_id: Optional[str] = None
        self.expected_sequence_id: int = 0
        self.cancel_token: Optional[CancelToken] = None

        # 内部有界控制与播放队列
        self.queue = asyncio.Queue(maxsize=100)
        self._lock = asyncio.Lock()
        self._actor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """拉起单一播放 Actor 任务，串行消费队列"""
        if self._actor_task is None or self._actor_task.done():
            self._stop_event.clear()
            self._actor_task = asyncio.create_task(self._playback_actor())

    async def begin_response(self, response_id: str) -> int:
        """开始新的 Response，原子自增本地播放代次 generation_id 并刷新 CancelToken"""
        async with self._lock:
            self.generation_id += 1
            self.active_response_id = response_id
            self.expected_sequence_id = 0
            self.cancel_token = CancelToken(self.generation_id)
            self.state = PlaybackState.BUFFERING
            return self.generation_id

    async def push_chunk(self, chunk: PlaybackChunk) -> bool:
        """
        向播放引擎推入待播 Chunk
        硬断言：代次落后（stale generation）一票否决丢弃，绝不上屏进队！
        """
        async with self._lock:
            # 1. 严格代次硬校验 (防幽灵音频核心逻辑)
            if chunk.generation_id < self.generation_id:
                if self.timeline:
                    await self.timeline.record(RuntimeEvent(
                        event_type=EventType.AUDIO_CHUNK_REJECTED,
                        session_id=chunk.session_id,
                        response_id=chunk.response_id,
                        generation_id=chunk.generation_id,
                        sequence_id=chunk.sequence_id,
                        source="playback_engine",
                        payload={"reason": "stale_generation", "current_generation": self.generation_id}
                    ))
                return False

            # 2. 检查当前 CancelToken 是否已取消
            if self.cancel_token and self.cancel_token.cancelled:
                return False

            try:
                self.queue.put_nowait(chunk)
                return True
            except asyncio.QueueFull:
                return False

    async def interrupt(self, reason: str = "barge_in"):
        """
        用户插话打断处理（先失效代次、清空队列，再停底层硬件 Sink）
        """
        async with self._lock:
            old_gen = self.generation_id
            self.generation_id += 1
            self.state = PlaybackState.INTERRUPTING

            # 1. 取消当前代次令牌
            if self.cancel_token:
                self.cancel_token.cancel(reason=reason)

            # 2. 物理清空本地待播队列
            flushed_count = 0
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    flushed_count += 1
                except (asyncio.QueueEmpty, ValueError):
                    break

            # 3. 记录事件
            if self.timeline:
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
                    session_id="",
                    response_id=self.active_response_id,
                    generation_id=old_gen,
                    source="playback_engine",
                    payload={"old_generation_id": old_gen, "new_generation_id": self.generation_id, "reason": reason}
                ))
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.PLAYBACK_QUEUE_FLUSHED,
                    session_id="",
                    response_id=self.active_response_id,
                    generation_id=old_gen,
                    source="playback_engine",
                    payload={"flushed_count": flushed_count}
                ))

            # 4. 物理停止底层 Sink 驱动
            await self.sink.stop_and_flush()

            if self.timeline:
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.PLAYBACK_SINK_STOPPED,
                    session_id="",
                    response_id=self.active_response_id,
                    generation_id=old_gen,
                    source="playback_engine",
                    payload={}
                ))

            self.state = PlaybackState.IDLE

    async def _playback_actor(self):
        """单一播放消费 Actor，保证播放与打断在时序上绝对串行无竞争"""
        while not self._stop_event.is_set():
            try:
                chunk: PlaybackChunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # 二次校验代次（防止在队列等待期间被外部打断）
            async with self._lock:
                if chunk.generation_id != self.generation_id:
                    self.queue.task_done()
                    continue
                if self.cancel_token and self.cancel_token.cancelled:
                    self.queue.task_done()
                    continue
                self.state = PlaybackState.PLAYING

            # 真实物理调用 Sink 播放
            await self.sink.play(chunk)
            self.queue.task_done()

            if self.timeline:
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.PLAYBACK_STARTED,
                    session_id=chunk.session_id,
                    response_id=chunk.response_id,
                    generation_id=chunk.generation_id,
                    sequence_id=chunk.sequence_id,
                    source="playback_actor",
                    payload={"duration_ms": chunk.expected_duration_ms}
                ))

    async def close(self):
        self._stop_event.set()
        if self._actor_task and not self._actor_task.done():
            self._actor_task.cancel()
            try:
                await self._actor_task
            except asyncio.CancelledError:
                pass
        await self.sink.close()
        self.state = PlaybackState.IDLE
