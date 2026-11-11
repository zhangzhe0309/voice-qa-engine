"""Noise Evaluator: 环境噪声注入与 ASR/VAD 抗扰度度量引擎
支持指定 SNR (信噪比) 噪声线性叠加合成，以及中文字错率 (CER) 动态规划算法度量。
严格修复全局随机种子污染问题，采用局部独立 Random 实例。
"""

import math
import struct
import random
from typing import List, Dict, Any, Optional


class NoiseEvaluator:
    def __init__(self, default_seed: int = 42):
        self.default_seed = default_seed
        self.rng = random.Random(default_seed)

    def set_seed(self, seed: int):
        self.rng = random.Random(seed)

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
        """计算中文字符错误率 (CER = (Sub + Ins + Del) / TotalRef)."""
        ref_chars = list(reference.strip())
        hyp_chars = list(hypothesis.strip())
        r_len = len(ref_chars)
        h_len = len(hyp_chars)

        if r_len == 0:
            return {"cer": 1.0 if h_len > 0 else 0.0, "distance": h_len, "ref_len": 0, "hyp_len": h_len}

        dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
        for i in range(r_len + 1):
            dp[i][0] = i
        for j in range(h_len + 1):
            dp[0][j] = j

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,      # Deletion
                        dp[i][j - 1] + 1,      # Insertion
                        dp[i - 1][j - 1] + 1   # Substitution
                    )

        distance = dp[r_len][h_len]
        cer = distance / r_len
        accuracy = max(0.0, 1.0 - cer)

        return {
            "cer": round(cer, 4),
            "accuracy": round(accuracy, 4),
            "distance": distance,
            "ref_len": r_len,
            "hyp_len": h_len
        }

    def inject_noise_to_pcm(
        self,
        pcm_samples: List[int],
        snr_db: float,
        noise_type: str = "gaussian",
        seed: Optional[int] = None
    ) -> List[int]:
        """将纯净语音样点以指定 SNR (dB) 叠加环境噪声。
        SNR_dB = 10 * log10(P_signal / P_noise)
        使用独立 RNG 实例，绝不污染全局 random 状态。
        """
        n = len(pcm_samples)
        if n == 0:
            return []

        # 仅对有效振幅区间计信号功率，防止全静音主导
        active_samples = [s for s in pcm_samples if abs(s) > 100]
        calc_pool = active_samples if len(active_samples) > (n * 0.1) else pcm_samples
        signal_power = sum(s * s for s in calc_pool) / max(1, len(calc_pool))
        if signal_power == 0:
            return pcm_samples

        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        target_noise_std = math.sqrt(target_noise_power)

        local_rng = random.Random(seed) if seed is not None else self.rng

        noisy_samples = []
        for val in pcm_samples:
            if noise_type == "gaussian":
                noise = local_rng.gauss(0, target_noise_std)
            elif noise_type == "hallway_murmur":
                noise = local_rng.gauss(0, target_noise_std) * 0.7 + local_rng.uniform(-1, 1) * target_noise_std * 0.3
            else:
                noise = local_rng.uniform(-target_noise_std * 1.732, target_noise_std * 1.732)

            mixed = int(round(val + noise))
            clamped = max(-32768, min(32767, mixed))
            noisy_samples.append(clamped)

        return noisy_samples

    @staticmethod
    def pcm_to_wav_bytes(samples: List[int], sample_rate: int = 24000) -> bytes:
        """将 PCM 采样点列表打包为标准 16-bit 单声道 WAV 文件字节流。"""
        num_samples = len(samples)
        data_bytes = struct.pack(f"<{num_samples}h", *samples)
        data_size = len(data_bytes)
        riff_size = data_size + 36

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            riff_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            data_size
        )
        return header + data_bytes
