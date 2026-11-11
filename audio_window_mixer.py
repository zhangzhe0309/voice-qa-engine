"""Audio Window Mixer: 短时音频加窗混音与打断对抗注入器
负责计算短时能量、起振点检测与 Hann 窗平滑交叉淡化注入。
注意：本模块专注于声学加窗混音与离散样点切片注入，不依赖外部 MFA 深度学习对齐器。
"""

import math
from typing import List, Tuple, Dict, Any, Optional


class AudioWindowMixer:
    def __init__(self, frame_duration_ms: int = 20, sample_rate: int = 24000):
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.frame_size = int((frame_duration_ms / 1000.0) * sample_rate)

    def calculate_short_time_energy(self, pcm_samples: List[int]) -> List[float]:
        """按帧计算短时能量 (RMS)"""
        energies = []
        for i in range(0, len(pcm_samples), self.frame_size):
            frame = pcm_samples[i:i + self.frame_size]
            if not frame:
                continue
            rms = math.sqrt(sum(s * s for s in frame) / len(frame))
            energies.append(rms)
        return energies

    def detect_onset_sample_index(self, pcm_samples: List[int], energy_threshold: float = 500.0) -> int:
        """检测有效语音的起振样点下标"""
        for i in range(0, len(pcm_samples), self.frame_size):
            frame = pcm_samples[i:i + self.frame_size]
            if not frame:
                break
            rms = math.sqrt(sum(s * s for s in frame) / len(frame))
            if rms >= energy_threshold:
                return i
        return 0

    def mix_bargein_at_sample(
        self,
        base_samples: List[int],
        bargein_samples: List[int],
        inject_sample_idx: int,
        duck_ratio: float = 0.2,
        fade_ms: int = 10
    ) -> List[int]:
        """在指定样点注入打断音频，并对底音频实施加窗淡出与压麦 (Ducking)"""
        out = list(base_samples)
        fade_samples = int((fade_ms / 1000.0) * self.sample_rate)

        # 确保输出长度足够
        req_len = inject_sample_idx + len(bargein_samples)
        if len(out) < req_len:
            out.extend([0] * (req_len - len(out)))

        # 注入并混音
        for j, b_val in enumerate(bargein_samples):
            idx = inject_sample_idx + j
            # 计算加窗淡出比例
            if j < fade_samples:
                w = 0.5 * (1.0 + math.cos(math.pi * j / fade_samples))
                cur_duck = duck_ratio + (1.0 - duck_ratio) * w
            else:
                cur_duck = duck_ratio

            base_val = out[idx] * cur_duck
            mixed = int(round(base_val + b_val))
            clamped = max(-32768, min(32767, mixed))
            out[idx] = clamped

        return out
