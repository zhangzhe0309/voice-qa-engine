#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 统一会话协调层 (SessionRuntime)
协调 AudioInput、WebSocket Client、PlaybackEngine、EventTimeline 与 EvidenceRecorder
"""

import asyncio
from typing import Optional, Dict, Any

from voice_qa_engine.runtime.session_state import SessionState
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.runtime.response_registry import ResponseRegistry
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.runtime.context_stack import ContextCallStack
from voice_qa_engine.runtime.tool_runtime import ToolAbortManager


class SessionRuntime:
    def __init__(
        self,
        session_id: str,
        audio_input,
        playback_engine: PlaybackEngine,
        timeline: EventTimeline,
        recorder: EvidenceRecorder,
        client=None,
        context_stack: Optional[ContextCallStack] = None,
        tool_manager: Optional[ToolAbortManager] = None
    ):
        self.session_id = session_id
        self.audio_input = audio_input
        self.playback_engine = playback_engine
        self.timeline = timeline
        self.recorder = recorder
        self.client = client
        self.context_stack = context_stack or ContextCallStack()
        self.tool_manager = tool_manager or ToolAbortManager()

        self.state = SessionState.CREATED
        self.registry = ResponseRegistry()
        self._input_task: Optional[asyncio.Task] = None

    async def connect(self):
        self.state = SessionState.CONNECTING
        if self.client and hasattr(self.client, 'connect'):
            await self.client.connect()
        self.state = SessionState.READY
        await self.timeline.record(RuntimeEvent(
            event_type=EventType.SESSION_READY,
            session_id=self.session_id,
            source="session_runtime"
        ))

    async def start(self):
        if self.state != SessionState.READY:
            raise RuntimeError(f"Session cannot start in state: {self.state}")
        self.state = SessionState.RUNNING
        await self.playback_engine.start()

    async def feed_audio(self):
        """流式读取音频帧并推送上行"""
        if not self.audio_input:
            return
        await self.audio_input.open()
        try:
            while self.state == SessionState.RUNNING:
                frame = await self.audio_input.read()
                if frame is None:
                    break
                await self.recorder.write_input_audio(frame)
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.INPUT_AUDIO_SENT,
                    session_id=self.session_id,
                    sequence_id=frame.sequence_id,
                    source="audio_input",
                    payload={"duration_ms": frame.duration_ms}
                ))
                if self.client and hasattr(self.client, 'send_audio'):
                    await self.client.send_audio(frame)
        finally:
            await self.timeline.record(RuntimeEvent(
                event_type=EventType.INPUT_AUDIO_END,
                session_id=self.session_id,
                source="audio_input"
            ))

    async def close(self):
        self.state = SessionState.CLOSING
        if self.audio_input:
            await self.audio_input.close()
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()
        await self.playback_engine.close()
        self.state = SessionState.CLOSED
