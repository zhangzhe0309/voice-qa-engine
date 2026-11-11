#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 高性能音频重采样器
支持 16kHz, 24kHz, 48kHz 等不同采样率之间高精度重采样转换
"""

import numpy as np
try:
    from scipy import signal
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class AudioResampler:
    def __init__(self):
        pass

    def resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        重采样浮点音频数组
        samples: float32, 1D 数组
        """
        if orig_sr == target_sr:
            return samples.astype(np.float32, copy=False)

        if len(samples) == 0:
            return np.empty(0, dtype=np.float32)

        target_length = int(round(len(samples) * float(target_sr) / orig_sr))

        if _HAS_SCIPY:
            # 使用 scipy.signal.resample 进行多相/FFT重采样
            resampled = signal.resample(samples, target_length)
            return resampled.astype(np.float32)
        else:
            # 降级使用高精度线性插值
            orig_indices = np.linspace(0, len(samples) - 1, len(samples))
            target_indices = np.linspace(0, len(samples) - 1, target_length)
            resampled = np.interp(target_indices, orig_indices, samples)
            return resampled.astype(np.float32)
