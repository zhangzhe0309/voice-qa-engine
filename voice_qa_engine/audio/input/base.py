#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 音频输入源基类
统一抽象输出 AudioFrame，上层无论接入 WAV、麦克风还是 Loopback 均保持一致
"""

from abc import ABC, abstractmethod
from typing import Optional
from voice_qa_engine.audio.models import AudioFrame


class AudioInput(ABC):
    @abstractmethod
    async def open(self) -> None:
        """打开并初始化输入流"""
        pass

    @abstractmethod
    async def read(self) -> Optional[AudioFrame]:
        """读取下一个单帧 AudioFrame，文件或流结束时返回 None"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """安全关闭流"""
        pass
