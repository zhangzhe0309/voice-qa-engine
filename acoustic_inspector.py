"""Acoustic Inspector: 物理声学质量与爆音扫描引擎
对 PCM 16-bit WAV/MP3 进行一阶差分突变、削波截断、直流偏置及容器头合法性深度质检。
"""

import struct
import math
from typing import Dict, Any, List


class AcousticInspector:
    def __init__(self, jump_threshold: int = 8000, clipping_threshold: int = 32700):
        self.jump_threshold = jump_threshold
        self.clipping_threshold = clipping_threshold

    def inspect_wav_bytes(self, audio_bytes: bytes) -> Dict[str, Any]:
        """对原始音频字节流执行全量声学物理质检。"""
        total_len = len(audio_bytes)
        if total_len < 44:
            return {"valid": False, "error": "文件长度小于 44 字节，非合法音频"}

        # 1. 容器头合法性与 4GB 假长度探测
        riff_tag = audio_bytes[:4]
        wave_tag = audio_bytes[8:12]

        is_wav = (riff_tag == b'RIFF' and wave_tag == b'WAVE')
        header_anomalies = []

        riff_size = 0
        data_size = 0
        sample_rate = 24000
        channels = 1
        bits_per_sample = 16
        data_offset = 44

        if is_wav:
            riff_size = struct.unpack("<I", audio_bytes[4:8])[0]
            fmt_size = struct.unpack("<I", audio_bytes[16:20])[0]
            audio_format = struct.unpack("<H", audio_bytes[20:22])[0]
            channels = struct.unpack("<H", audio_bytes[22:24])[0]
            sample_rate = struct.unpack("<I", audio_bytes[24:28])[0]
            bits_per_sample = struct.unpack("<H", audio_bytes[34:36])[0]

            # 寻找 data chunk
            offset = 12 + fmt_size + 8
            while offset + 8 <= total_len:
                sub_tag = audio_bytes[offset:offset+4]
                sub_size = struct.unpack("<I", audio_bytes[offset+4:offset+8])[0]
                if sub_tag == b'data':
                    data_offset = offset + 8
                    data_size = sub_size
                    break
                offset += 8 + sub_size

            # 4GB 假长度与未闭合检测
            if riff_size == 0xFFFFFFFF or data_size == 0xFFFFFFFF:
                header_anomalies.append({
                    "type": "UNCLOSED_STREAM_HEADER",
                    "severity": "CRITICAL",
                    "message": "WAV 头部标记为 0xFFFFFFFF (4GB 假长度)，在移动端/小程序会触发 Unexpected EOF 强制硬件复位爆音"
                })
            elif riff_size + 8 != total_len or data_size + data_offset > total_len:
                header_anomalies.append({
                    "type": "HEADER_SIZE_MISMATCH",
                    "severity": "WARNING",
                    "message": f"头文件声明大小 (RIFF={riff_size}) 与实际物理大小 ({total_len}) 不一致"
                })
        else:
            if audio_bytes[:3] == b'ID3':
                header_anomalies.append({
                    "type": "MP3_ID3_METADATA_PRESENT",
                    "severity": "WARNING",
                    "message": "音频以 ID3v2 标签起始，若直接与 PCM 或分片拼接会导致解码器破帧杂音"
                })

        # 2. 提取 PCM 16-bit 样点并进行信号分析
        pcm_bytes = audio_bytes[data_offset:]
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0:
            return {"valid": False, "error": "有效 PCM 样点数为 0"}

        samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
        duration = num_samples / max(1, sample_rate)

        # 3. 声学物理特征计算
        max_jump = 0
        max_jump_idx = 0
        jump_events = []
        clipped_samples = 0
        sum_val = 0
        sum_sq = 0

        for i in range(num_samples):
            val = samples[i]
            sum_val += val
            sum_sq += val * val

            if abs(val) >= self.clipping_threshold:
                clipped_samples += 1

            if i > 0:
                diff = abs(val - samples[i - 1])
                if diff > max_jump:
                    max_jump = diff
                    max_jump_idx = i
                if diff >= self.jump_threshold:
                    jump_events.append({
                        "sample_idx": i,
                        "time_s": round(i / sample_rate, 4),
                        "jump_delta": diff,
                        "from_amp": samples[i - 1],
                        "to_amp": val
                    })

        mean_dc = sum_val / num_samples
        rms_energy = math.sqrt(sum_sq / num_samples)
        clipping_rate = clipped_samples / num_samples

        # 4. 开头起振与结尾截断分析
        head_window = samples[:min(50, num_samples)]
        tail_window = samples[max(0, num_samples - 50):]
        head_step = abs(head_window[0])
        tail_step = abs(tail_window[-1])

        transient_anomalies = []
        if head_step > 500:
            transient_anomalies.append({
                "type": "HARD_HEAD_START",
                "severity": "WARNING",
                "message": f"开头起振首个样点绝对值达 {head_step} (非零阶跃)，无软淡入缓冲，容易在喇叭开启瞬间产生轻微弹片声"
            })
        if tail_step > 300:
            transient_anomalies.append({
                "type": "HARD_TAIL_CUTOFF",
                "severity": "WARNING",
                "message": f"结尾最后样点绝对值达 {tail_step} (非零截断)，容易在音频播放完毕瞬间产生硬件回零爆音"
            })

        risk_score = 0
        if max_jump >= 12000:
            risk_score += 45
        elif max_jump >= self.jump_threshold:
            risk_score += 25

        if any(h["severity"] == "CRITICAL" for h in header_anomalies):
            risk_score += 50

        if clipping_rate > 0.001:
            risk_score += 20

        if len(transient_anomalies) > 0:
            risk_score += 15

        risk_score = min(100, risk_score)

        return {
            "valid": True,
            "is_wav": is_wav,
            "format_info": {
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "bit_depth": bits_per_sample,
                "duration_seconds": round(duration, 3),
                "total_samples": num_samples,
                "file_size_bytes": total_len
            },
            "metrics": {
                "max_sample_jump": max_jump,
                "max_jump_time_s": round(max_jump_idx / sample_rate, 4),
                "jumps_exceeding_threshold_count": len(jump_events),
                "dc_offset": round(mean_dc, 2),
                "rms_energy": round(rms_energy, 1),
                "clipping_sample_count": clipped_samples,
                "clipping_rate_pct": round(clipping_rate * 100, 4),
                "head_step_level": head_step,
                "tail_step_level": tail_step,
                "pop_risk_score": risk_score
            },
            "header_anomalies": header_anomalies,
            "transient_anomalies": transient_anomalies,
            "top_jump_events": jump_events[:5]
        }
