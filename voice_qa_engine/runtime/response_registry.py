#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 响应代次注册表
将服务端的 response_id 与客户端本地单调递增的 generation_id 严格绑定并维护生命周期
"""

from typing import Dict, Optional


class ResponseRegistry:
    def __init__(self):
        self._resp_to_gen: Dict[str, int] = {}
        self._active_status: Dict[str, bool] = {}

    def bind(self, response_id: str, generation_id: int):
        self._resp_to_gen[response_id] = generation_id
        self._active_status[response_id] = True

    def get_generation(self, response_id: str) -> Optional[int]:
        return self._resp_to_gen.get(response_id)

    def invalidate(self, response_id: str):
        if response_id in self._active_status:
            self._active_status[response_id] = False

    def is_active(self, response_id: str, generation_id: int) -> bool:
        expected_gen = self._resp_to_gen.get(response_id)
        if expected_gen is None or expected_gen != generation_id:
            return False
        return self._active_status.get(response_id, False)
