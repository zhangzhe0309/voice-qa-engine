#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 协议与归一化事件模型
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


class EventType(str, Enum):
    SESSION_READY = "session.ready"
    INPUT_AUDIO_SENT = "input.audio.sent"
    INPUT_AUDIO_END = "input.audio.end"
    RESPONSE_STARTED = "response.started"
    TEXT_DELTA = "response.text.delta"
    AUDIO_CHUNK = "response.audio.chunk"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_CANCELLED = "response.cancelled"
    RESPONSE_CANCEL_REQUESTED = "response.cancel.requested"
    INPUT_SPEECH_STARTED = "input.speech.started"
    INPUT_SPEECH_STOPPED = "input.speech.stopped"
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_INTERRUPT_REQUESTED = "playback.interrupt.requested"
    PLAYBACK_QUEUE_FLUSHED = "playback.queue.flushed"
    PLAYBACK_SINK_STOPPED = "playback.sink.stopped"
    PLAYBACK_SINK_UNDERRUN = "playback.sink.underrun"
    AUDIO_CHUNK_REJECTED = "response.audio.chunk.rejected"
    PLAYBACK_DRAINED = "playback.drained"
    CRISIS_EMERGENCY_STOP = "crisis.emergency.stop"
    ERROR = "error"


@dataclass
class RuntimeEvent:
    event_type: str
    session_id: str
    connection_epoch: int = 1
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    response_id: Optional[str] = None
    generation_id: Optional[int] = None
    sequence_id: Optional[int] = None
    monotonic_ns: int = 0
    wall_time_utc: Optional[str] = None
    direction: str = "server_to_client"
    source: str = "websocket"
    payload: dict[str, Any] = field(default_factory=dict)
    raw_event_type: Optional[str] = None
    raw_message_ref: Optional[str] = None
    field_origin: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.monotonic_ns <= 0:
            self.monotonic_ns = time.monotonic_ns()
        if not self.wall_time_utc:
            self.wall_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
