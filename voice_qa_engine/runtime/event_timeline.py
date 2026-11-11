#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 事件时间轴记录器
"""

import time
import asyncio
from typing import List, Optional
from voice_qa_engine.clients.protocol_models import RuntimeEvent


class EventTimeline:
    def __init__(self, recorder=None):
        self.recorder = recorder
        self.events: List[RuntimeEvent] = []
        self._lock = asyncio.Lock()

    async def record(self, event: RuntimeEvent):
        if event.monotonic_ns <= 0:
            event.monotonic_ns = time.monotonic_ns()
        async with self._lock:
            self.events.append(event)
        if self.recorder and hasattr(self.recorder, 'write_event'):
            await self.recorder.write_event(event)

    def snapshot(self) -> List[RuntimeEvent]:
        return list(self.events)

    def find_all(self, event_type: str) -> List[RuntimeEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def first(self, event_type: str) -> Optional[RuntimeEvent]:
        for e in self.events:
            if e.event_type == event_type:
                return e
        return None
