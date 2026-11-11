#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - WAV 文件音频输入源
从物理 WAV 文件或 BytesIO 逐帧分片读取，按真实时钟或快速模式流式吐出 AudioFrame
"""

import wave
import time
import io
from typing import Optional, Union
import numpy as np

from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.audio.input.base import AudioInput
from voice_qa_engine.audio.format_adapter import PCMFormatAdapter


class WavAudioInput(AudioInput):
    def __init__(
        self,
        file_path_or_obj: Union[str, io.BytesIO] = None,
        file_obj: io.BytesIO = None,
        frame_duration_ms: float = 20.0,
        real_time_pace: bool = False
    ):
        self.target = file_obj if file_obj is not None else file_path_or_obj
        self.frame_duration_ms = frame_duration_ms
        self.real_time_pace = real_time_pace

        self._wf = None
        self._format = None
        self._samples_per_frame = 0
        self._sequence_id = 0
        self._adapter = PCMFormatAdapter()
        self._is_open = False

    async def open(self) -> None:
        if isinstance(self.target, str):
            self._wf = wave.open(self.target, 'rb')
        else:
            self._wf = wave.open(self.target, 'rb')

        channels = self._wf.getnchannels()
        sample_rate = self._wf.getframerate()
        sampwidth = self._wf.getsampwidth()

        if sampwidth == 2:
            s_fmt = SampleFormat.PCM_S16LE
        elif sampwidth == 4:
            s_fmt = SampleFormat.PCM_S32LE
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8} bits")

        self._format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            sample_format=s_fmt
        )
        self._samples_per_frame = int(round(sample_rate * (self.frame_duration_ms / 1000.0)))
        self._sequence_id = 0
        self._is_open = True

    async def read(self) -> Optional[AudioFrame]:
        if not self._is_open or self._wf is None:
            raise RuntimeError("WavAudioInput is not open")

        raw_bytes = self._wf.readframes(self._samples_per_frame)
        if not raw_bytes:
            return None

        # 解码为内部归一化 float32 AudioFrame
        frame = self._adapter.decode(raw_bytes, self._format, sequence_id=self._sequence_id)
        frame.sequence_id = self._sequence_id
        self._sequence_id += 1

        if self.real_time_pace:
            # 真实时间模拟（按照帧时长异步休眠）
            import asyncio
            await asyncio.sleep(self.frame_duration_ms / 1000.0)

        return frame

    async def close(self) -> None:
        if self._wf is not None:
            self._wf.close()
            self._wf = None
        self._is_open = False
