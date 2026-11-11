#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 核心音频数据模型
内部音频统一采用归一化线性 float32 PCM 表示，解耦具体硬件/协议采样率与声道
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
import numpy as np


class SampleFormat(str, Enum):
    PCM_S16LE = "pcm_s16le"
    PCM_S24LE = "pcm_s24le"
    PCM_S32LE = "pcm_s32le"
    PCM_F32LE = "pcm_f32le"


@dataclass(frozen=True)
class AudioFormat:
    """音频流格式元数据"""
    sample_rate: int
    channels: int
    sample_format: SampleFormat = SampleFormat.PCM_F32LE
    channel_layout: str = "mono"

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive integer")
        if self.channels <= 0:
            raise ValueError("channels must be positive integer")

    @property
    def bytes_per_sample(self) -> int:
        if self.sample_format == SampleFormat.PCM_S16LE:
            return 2
        elif self.sample_format == SampleFormat.PCM_S24LE:
            return 3
        elif self.sample_format in (SampleFormat.PCM_S32LE, SampleFormat.PCM_F32LE):
            return 4
        return 4


@dataclass
class AudioFrame:
    """
    单段音频帧数据
    samples: 内部统一使用 float32 格式，范围归一化在 [-1.0, 1.0]
    """
    samples: np.ndarray
    format: AudioFormat
    timestamp_ns: int
    sequence_id: int
    duration_ms: float

    def __post_init__(self):
        if not isinstance(self.samples, np.ndarray):
            self.samples = np.asarray(self.samples, dtype=np.float32)

        # 校验时钟
        if self.timestamp_ns <= 0:
            self.timestamp_ns = time.monotonic_ns()

        # 根据真实样本数与采样率校验或纠偏 duration_ms
        calculated_ms = (len(self.samples) / self.format.sample_rate) * 1000.0
        if abs(self.duration_ms - calculated_ms) > 1.0:
            self.duration_ms = calculated_ms
