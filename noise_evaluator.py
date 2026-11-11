"""Noise Evaluator: 环境噪声注入与 ASR/VAD 抗扰度度量引擎
支持指定 SNR (信噪比) 噪声线性叠加合成，以及中文字错率 (CER) 动态规划算法度量。
"""

import math
import struct
import random
from typing import List, Tuple, Dict, Any


class NoiseEvaluator:
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
        """计算中文字符错误率 (CER = (Sub + Ins + Del) / TotalRef)."""
        ref_chars = list(reference.strip())
        hyp_chars = list(hypothesis.strip())
        r_len = len(ref_chars)
        h_len = len(hyp_chars)

        if r_len == 0:
            return {"cer": 1.0 if h_len > 0 else 0.0, "distance": h_len, "ref_len": 0, "hyp_len": h_len}

        # 动态规划矩阵 (Levenshtein)
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

    @staticmethod
    def inject_noise_to_pcm(pcm_samples: List[int], snr_db: float, noise_type: str = "gaussian") -> List[int]:
        """将纯净语音样点以指定 SNR (dB) 叠加环境噪声。
        SNR_dB = 10 * log10(P_signal / P_noise)
        """
        n = len(pcm_samples)
        if n == 0:
            return []

        # 计算信号功率
        signal_power = sum(s * s for s in pcm_samples) / n
        if signal_power == 0:
            return pcm_samples

        # 计算目标噪声功率: P_noise = P_signal / (10^(SNR/10))
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        target_noise_std = math.sqrt(target_noise_power)

        noisy_samples = []
        random.seed(42)  # 固定随机种子确保测试可重复性

        for val in pcm_samples:
            if noise_type == "gaussian":
                noise = random.gauss(0, target_noise_std)
            elif noise_type == "hallway_murmur":
                # 模拟走廊低频环境人声咕哝
                noise = random.gauss(0, target_noise_std) * 0.7 + random.uniform(-1, 1) * target_noise_std * 0.3
            else:
                noise = random.uniform(-target_noise_std * 1.732, target_noise_std * 1.732)

            mixed = int(round(val + noise))
            # 限制在 16-bit 范围 (-32768 ~ 32767)
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
            16,               # Subchunk1Size
            1,                # AudioFormat (PCM)
            1,                # NumChannels (Mono)
            sample_rate,      # SampleRate
            sample_rate * 2,  # ByteRate
            2,                # BlockAlign
            16,               # BitsPerSample
            b"data",
            data_size
        )
        return header + data_bytes
