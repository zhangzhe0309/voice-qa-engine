#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 取消令牌 (CancelToken)
"""

import asyncio
import uuid
from typing import Optional


class CancelToken:
    def __init__(self, generation_id: int):
        self.id = str(uuid.uuid4())
        self.generation_id = generation_id
        self._event = asyncio.Event()
        self.reason: Optional[str] = None

    def cancel(self, reason: str = "cancelled"):
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self):
        await self._event.wait()
