#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 重叠语音/争吵合成器 (Overlap Speech Synthesizer)
纯合成多说话人重叠语音波形（多基频谐波 + 交替包络冲突），无外部素材依赖。
用于声纹排他/多人讨论拒识场景：按目标 SNR 注入背景争吵音频。
"""

import random
from typing import List, Optional

import numpy as np


class OverlapSpeechSynthesizer:
    """多人重叠语音合成器（确定性种子）"""

    DEFAULT_BASE_FREQS = [110.0, 190.0, 145.0]  # 男低/女高/中音基频

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def generate_argument(
        self,
        duration_ms: float,
        n_speakers: int = 2,
        snr_db: float = 5.0,
        reference_rms: float = 0.05,
        base_freqs: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        生成 n 人争吵重叠语音（float32 [-1,1]），相对参考电平按目标 SNR 定标。
        reference_rms: 主用户语音的 RMS 电平基准（默认 -26dBFS 量级）。
        """
        if n_speakers < 2:
            raise ValueError("争吵场景至少需要 2 个说话人")
        n_total = int(self.sample_rate * duration_ms / 1000.0)
        freqs = (base_freqs or self.DEFAULT_BASE_FREQS)[:n_speakers]

        mix = np.zeros(n_total, dtype=np.float64)
        for idx, f0 in enumerate(freqs):
            mix += self._speaker_track(f0, n_total, speaker_seed=idx)

        # 目标 SNR 定标：noise_rms = ref_rms * 10^(-snr/20)
        target_rms = reference_rms * (10 ** (-snr_db / 20.0))
        current_rms = float(np.sqrt(np.mean(mix ** 2))) or 1e-9
        mix *= target_rms / current_rms

        peak = float(np.max(np.abs(mix))) or 1.0
        if peak > 0.99:  # 防削波
            mix *= 0.99 / peak
        return mix.astype(np.float32)

    def _speaker_track(self, f0: float, n_total: int, speaker_seed: int) -> np.ndarray:
        """单说话人轨：谐波叠声门音源 + 交替发声包络（激发/抢话冲突感）"""
        rng = random.Random(self.rng.randint(0, 1 << 30) + speaker_seed)
        t = np.arange(n_total) / self.sample_rate
        phase_jitter = 1.0 + (rng.random() - 0.5) * 0.04
        harmonics = np.zeros(n_total, dtype=np.float64)
        for k in range(1, 9):
            harmonics += (1.0 / k) * np.sin(
                2 * np.pi * f0 * phase_jitter * k * t + rng.random() * 2 * np.pi
            )

        # 交替发声包络：150~400ms 发声 / 60~200ms 停顿
        envelope = np.zeros(n_total, dtype=np.float64)
        pos = rng.randint(0, 200)
        while pos < n_total:
            on_len = int(self.sample_rate * (rng.uniform(0.15, 0.40)))
            off_len = int(self.sample_rate * (rng.uniform(0.06, 0.20)))
            end = min(pos + on_len, n_total)
            # 10ms 余弦起落沿，防爆音
            ramp = int(self.sample_rate * 0.01)
            seg = np.ones(end - pos, dtype=np.float64)
            if ramp * 2 < seg.size:
                seg[:ramp] = np.linspace(0, 1, ramp)
                seg[-ramp:] = np.linspace(1, 0, ramp)
            envelope[pos:end] = seg
            pos = end + off_len
        return harmonics * envelope

    def generate_inhalation(self, duration_ms: float = 80.0, amp: float = 0.35) -> np.ndarray:
        """
        急促倒吸气声（宽带摩擦噪声 + 上行包络），供口吃/不流利对抗场景前置注入。
        """
        n_total = int(self.sample_rate * duration_ms / 1000.0)
        rng = np.random.default_rng(self.rng.randint(0, 1 << 30))
        noise = rng.standard_normal(n_total)
        # 带通感：一阶差分强调高频摩擦
        noise = np.diff(noise, prepend=0.0)
        envelope = np.linspace(0.2, 1.0, n_total)  # 上行包络（吸气渐强）
        wave = noise * envelope
        current_rms = float(np.sqrt(np.mean(wave ** 2))) or 1e-9
        wave *= amp / current_rms
        peak = float(np.max(np.abs(wave))) or 1.0
        if peak > 0.99:  # 防削波
            wave *= 0.99 / peak
        return wave.astype(np.float32)
