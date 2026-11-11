#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 音频格式与编码适配器
外部字节编码 ⇄ 内部归一化 float32 PCM 帧
"""

from abc import ABC, abstractmethod
import time
import numpy as np
from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.audio.resampler import AudioResampler


class AudioFormatAdapter(ABC):
    @abstractmethod
    def decode(self, payload: bytes, source_format: AudioFormat, sequence_id: int = 0) -> AudioFrame:
        """外部字节流 -> 内部 float32 AudioFrame"""
        pass

    @abstractmethod
    def encode(self, frame: AudioFrame, target_format: AudioFormat) -> bytes:
        """内部 float32 AudioFrame -> 外部字节流"""
        pass


class PCMFormatAdapter(AudioFormatAdapter):
    """标准线性 PCM 编解码适配器"""

    def __init__(self):
        self.resampler = AudioResampler()

    def decode(self, payload: bytes, source_format: AudioFormat, sequence_id: int = 0) -> AudioFrame:
        now_ns = time.monotonic_ns()

        if source_format.sample_format == SampleFormat.PCM_S16LE:
            int16_arr = np.frombuffer(payload, dtype=np.int16)
            # 归一化至 [-1.0, 1.0]
            float_samples = (int16_arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        elif source_format.sample_format == SampleFormat.PCM_F32LE:
            float_samples = np.frombuffer(payload, dtype=np.float32).copy()
        elif source_format.sample_format == SampleFormat.PCM_S32LE:
            int32_arr = np.frombuffer(payload, dtype=np.int32)
            float_samples = (int32_arr.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
        else:
            raise NotImplementedError(f"Unsupported decode format: {source_format.sample_format}")

        duration_ms = (len(float_samples) / source_format.sample_rate) * 1000.0
        internal_format = AudioFormat(
            sample_rate=source_format.sample_rate,
            channels=source_format.channels,
            sample_format=SampleFormat.PCM_F32LE,
            channel_layout=source_format.channel_layout
        )

        return AudioFrame(
            samples=float_samples,
            format=internal_format,
            timestamp_ns=now_ns,
            sequence_id=sequence_id,
            duration_ms=duration_ms
        )

    def encode(self, frame: AudioFrame, target_format: AudioFormat) -> bytes:
        samples = frame.samples

        # 1. 采样率变换 (必要时重采样)
        if frame.format.sample_rate != target_format.sample_rate:
            samples = self.resampler.resample(
                samples,
                orig_sr=frame.format.sample_rate,
                target_sr=target_format.sample_rate
            )

        # 2. 位深转换
        if target_format.sample_format == SampleFormat.PCM_S16LE:
            # 放大到 int16 范围 [-32768, 32767]
            clipped = np.clip(samples, -1.0, 1.0)
            int16_samples = (clipped * 32767.0).astype(np.int16)
            return int16_samples.tobytes()
        elif target_format.sample_format == SampleFormat.PCM_F32LE:
            return samples.astype(np.float32).tobytes()
        elif target_format.sample_format == SampleFormat.PCM_S32LE:
            clipped = np.clip(samples, -1.0, 1.0)
            int32_samples = (clipped * 2147483647.0).astype(np.int32)
            return int32_samples.tobytes()
        else:
            raise NotImplementedError(f"Unsupported encode format: {target_format.sample_format}")
