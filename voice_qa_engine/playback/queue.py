#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 播放分片数据模型
"""

from dataclasses import dataclass
from typing import Optional
from voice_qa_engine.audio.models import AudioFrame


@dataclass
class PlaybackChunk:
    session_id: str
    response_id: str
    generation_id: int
    sequence_id: int
    audio: AudioFrame
    received_monotonic_ns: int
    expected_duration_ms: float
    cancel_token_id: str
    connection_epoch: int = 1
    source_event_id: Optional[str] = None
