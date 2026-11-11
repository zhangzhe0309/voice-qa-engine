"""Acoustic Inspector: 物理声学质量与爆音扫描引擎
对 PCM 16-bit WAV 进行一阶差分突变、削波截断、直流偏置及容器头合法性深度质检。
对非 PCM/非 WAV 音频实施严格的格式兼容判定与明确拒绝，避免产生虚假统计。
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
            return {
                "valid": False,
                "status": "FAIL",
                "error": "文件长度小于 44 字节，非合法音频容器"
            }

        riff_tag = audio_bytes[:4]
        wave_tag = audio_bytes[8:12]

        # 格式识别与非 PCM 拒绝
        if riff_tag != b'RIFF' or wave_tag != b'WAVE':
            if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:2] == b'\xff\xf3':
                return {
                    "valid": False,
                    "status": "UNSUPPORTED_FORMAT",
                    "format": "MP3",
                    "error": "检测到 MP3 压缩编码流。本声学物理分析器仅支持未压缩的线性 PCM 信号，请先转码为 WAV PCM 后检测。"
                }
            return {
                "valid": False,
                "status": "UNSUPPORTED_FORMAT",
                "error": "非标准 RIFF/WAVE 容器格式"
            }

        header_anomalies = []
        riff_size = struct.unpack("<I", audio_bytes[4:8])[0]

        # 解析 fmt subchunk
        audio_format = 1
        channels = 1
        sample_rate = 24000
        bits_per_sample = 16
        fmt_found = False

        data_offset = -1
        data_size = 0

        offset = 12
        while offset + 8 <= total_len:
            sub_tag = audio_bytes[offset:offset+4]
            sub_size = struct.unpack("<I", audio_bytes[offset+4:offset+8])[0]
            if sub_tag == b'fmt ':
                fmt_found = True
                if sub_size >= 16:
                    audio_format = struct.unpack("<H", audio_bytes[offset+8:offset+10])[0]
                    channels = struct.unpack("<H", audio_bytes[offset+10:offset+12])[0]
                    sample_rate = struct.unpack("<I", audio_bytes[offset+12:offset+16])[0]
                    bits_per_sample = struct.unpack("<H", audio_bytes[offset+22:offset+24])[0]
            elif sub_tag == b'data':
                data_offset = offset + 8
                data_size = sub_size
                break
            
            # 步进（处理奇数字节对齐）
            offset += 8 + sub_size + (sub_size % 2)

        if not fmt_found:
            return {"valid": False, "status": "FAIL", "error": "WAV 文件中未发现合法的 'fmt ' 描述块"}

        if audio_format != 1:
            return {
                "valid": False,
                "status": "UNSUPPORTED_FORMAT",
                "format_code": audio_format,
                "error": f"暂不支持非线性 PCM 编码 (AudioFormat={audio_format}，如 IEEE Float=3 或 A-law=6)"
            }

        if bits_per_sample != 16:
            return {
                "valid": False,
                "status": "UNSUPPORTED_FORMAT",
                "bit_depth": bits_per_sample,
                "error": f"当前采样深度为 {bits_per_sample}-bit，仅支持 16-bit 有符号整数 PCM 分析"
            }

        if data_offset == -1:
            return {"valid": False, "status": "FAIL", "error": "WAV 容器中未发现合法的 'data' 数据块"}

        # 4GB 假长度与大小溢出检测
        if riff_size == 0xFFFFFFFF or data_size == 0xFFFFFFFF:
            header_anomalies.append({
                "type": "UNCLOSED_STREAM_HEADER",
                "severity": "CRITICAL",
                "message": "WAV 头部标记为 0xFFFFFFFF (4GB 假长度)，在移动端/小程序会触发 Unexpected EOF 强制硬件复位爆音"
            })
        elif (riff_size + 8 > total_len + 100) or (data_size + data_offset > total_len):
            header_anomalies.append({
                "type": "HEADER_SIZE_MISMATCH",
                "severity": "WARNING",
                "message": f"头文件声明大小 (RIFF={riff_size}, DATA={data_size}) 与实际物理大小 ({total_len}) 不闭合"
            })

        # 提取 PCM 16-bit 样点
        pcm_bytes = audio_bytes[data_offset:data_offset + data_size] if data_size > 0 else audio_bytes[data_offset:]
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0:
            return {"valid": False, "status": "FAIL", "error": "有效 PCM 样点数为 0"}

        samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
        duration = num_samples / max(1, sample_rate * max(1, channels))

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
                        "time_s": round(i / (sample_rate * channels), 4),
                        "jump_delta": diff,
                        "from_amp": samples[i - 1],
                        "to_amp": val
                    })

        mean_dc = sum_val / num_samples
        rms_energy = math.sqrt(sum_sq / num_samples)
        clipping_rate = clipped_samples / num_samples

        head_window = samples[:min(50, num_samples)]
        tail_window = samples[max(0, num_samples - 50):]
        head_step = abs(head_window[0])
        tail_step = abs(tail_window[-1])

        transient_anomalies = []
        if head_step > 500:
            transient_anomalies.append({
                "type": "HARD_HEAD_START",
                "severity": "WARNING",
                "message": f"开头起振首个样点绝值达 {head_step} (非零阶跃)，无软淡入缓冲，容易在扬声器开启瞬间产生轻微弹片声"
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
            "status": "PASS" if risk_score < 40 else "FAIL",
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
                "max_jump_time_s": round(max_jump_idx / max(1, sample_rate * channels), 4),
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
