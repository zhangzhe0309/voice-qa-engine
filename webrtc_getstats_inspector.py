#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webrtc_getstats_inspector.py — WebRTC getStats 实时统计与音轨时序解析套件
严格按照 W3C WebRTC 统计规范实现：
1. 增量差分计算：正确处理 packetsReceived / packetsLost 单调递增计数器与回绕重置；
2. 规范转换：统一遵循 RTCInboundRtpStreamStats 规范将 jitter (秒) 转换为毫秒；
3. 严格分层断言：解耦 audioLevel (瞬时能量) 与 totalAudioEnergy (累计能量)；
4. 状态隔离：将仿真流与真实流结果显式打上 measurement_source 标记。
"""

import time
import math
from typing import Dict, Any, List, Optional


class WebRTCStatsInspector:
    def __init__(self, jitter_p95_threshold_ms: float = 12.0, jitter_max_threshold_ms: float = 15.0):
        self.jitter_p95_limit = jitter_p95_threshold_ms
        self.jitter_max_limit = jitter_max_threshold_ms
        self.loss_limit_pct = 0.0
        self.rtt_limit_ms = 80.0

    @staticmethod
    def _compute_percentile(data: List[float], percentile: float) -> float:
        """计算百分位数 (兼容无 numpy 纯 Python 环境)"""
        if not data:
            return 0.0
        sorted_d = sorted(data)
        n = len(sorted_d)
        if n == 1:
            return sorted_d[0]
        k = (n - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_d[int(k)]
        d0 = sorted_d[int(f)] * (c - k)
        d1 = sorted_d[int(c)] * (k - f)
        return d0 + d1

    def parse_inbound_rtp_report(
        self,
        stats_samples: List[Dict[str, Any]],
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """解析时序 getStats 样本点并计算时延、抖动分布与丢包率。
        正确处理单调递增计数器（通过差分计算各周期增量）。
        """
        if not stats_samples:
            return {
                "status": "INCONCLUSIVE",
                "execution_mode": "empty",
                "error": "未提供有效 getStats 样本集"
            }

        jitters_ms = []
        audio_levels = []
        rtts_ms = []
        concealed_events = 0

        # 处理累计包数与丢包数 (差分增量累加)
        total_delta_received = 0
        total_delta_lost = 0

        prev_received = None
        prev_lost = None

        for s in stats_samples:
            # 1. Jitter: W3C 规范中单位严格为秒 (seconds)
            raw_jitter_sec = float(s.get("jitter", 0.0))
            j_ms = raw_jitter_sec * 1000.0
            jitters_ms.append(j_ms)

            # 2. 瞬时音量 audioLevel: 范围 0.0 ~ 1.0
            level = float(s.get("audioLevel", 0.0))
            audio_levels.append(level)

            # 3. RTT: roundTripTime 在部分实现为秒，在此统一按秒转毫秒
            if "roundTripTime" in s:
                rtts_ms.append(float(s["roundTripTime"]) * 1000.0)

            # 4. 计数器增量差分计算
            cur_received = s.get("packetsReceived")
            cur_lost = s.get("packetsLost")

            if cur_received is not None:
                if prev_received is not None and cur_received >= prev_received:
                    total_delta_received += (cur_received - prev_received)
                elif prev_received is None:
                    # 首个采样点的基准增量
                    total_delta_received += cur_received
                prev_received = cur_received

            if cur_lost is not None:
                if prev_lost is not None and cur_lost >= prev_lost:
                    total_delta_lost += (cur_lost - prev_lost)
                elif prev_lost is None:
                    total_delta_lost += cur_lost
                prev_lost = cur_lost

            if s.get("concealedSamples", 0) > 100 or s.get("bufferUnderrun", False):
                concealed_events += 1

        p50_jitter = self._compute_percentile(jitters_ms, 50.0)
        p95_jitter = self._compute_percentile(jitters_ms, 95.0)
        max_jitter = max(jitters_ms) if jitters_ms else 0.0

        total_stream_packets = total_delta_received + total_delta_lost
        loss_rate_pct = (total_delta_lost / max(1, total_stream_packets)) * 100.0
        avg_rtt = (sum(rtts_ms) / len(rtts_ms)) if rtts_ms else 0.0
        max_level = max(audio_levels) if audio_levels else 0.0

        # 断言逻辑
        passed_jitter = (p95_jitter <= self.jitter_p95_limit and max_jitter <= self.jitter_max_limit)
        passed_loss = (loss_rate_pct <= self.loss_limit_pct)
        passed_sound = (max_level > 0.001)
        passed_underrun = (concealed_events == 0)

        all_passed = (passed_jitter and passed_loss and passed_sound and passed_underrun)

        status = "SIMULATED_PASS" if (all_passed and is_simulated) else ("PASS" if all_passed else "FAIL")

        return {
            "status": status,
            "measurement_source": "mock_samples" if is_simulated else "webrtc_getstats_stream",
            "samples_count": len(stats_samples),
            "metrics": {
                "jitter_p50_ms": round(p50_jitter, 2),
                "jitter_p95_ms": round(p95_jitter, 2),
                "jitter_max_ms": round(max_jitter, 2),
                "packet_loss_rate_pct": round(loss_rate_pct, 4),
                "total_packets_received": total_delta_received,
                "total_packets_lost": total_delta_lost,
                "max_audio_level": round(max_level, 4),
                "avg_rtt_ms": round(avg_rtt, 2),
                "audio_concealment_events": concealed_events
            },
            "gate_assertions": {
                "jitter_p95_within_sla": passed_jitter,
                "zero_packet_loss": passed_loss,
                "audio_track_active": passed_sound,
                "zero_concealment_jitter": passed_underrun
            }
        }

    def generate_full_duplex_session_mock(self, duration_sec: int = 10, inject_tool_at_sec: int = 5) -> List[Dict[str, Any]]:
        """仿真 10s 全双工 WebRTC 指标流（带单调递增计数器）"""
        import random
        samples = []
        cumulative_packets = 0
        cumulative_lost = 0

        for sec in range(duration_sec * 2):  # 每 500ms 一次采样
            t = sec * 0.5
            is_tool_running = (t >= inject_tool_at_sec and t <= inject_tool_at_sec + 1.5)
            jitter_sec = (random.uniform(8.0, 13.5) if is_tool_running else random.uniform(3.0, 6.5)) / 1000.0
            energy = random.uniform(0.35, 0.85) if (t < 7.0 or t > 8.0) else 0.0

            # 递增计数器
            cumulative_packets += 25
            samples.append({
                "timestamp": time.time() + t,
                "jitter": jitter_sec,
                "packetsReceived": cumulative_packets,
                "packetsLost": cumulative_lost,
                "audioLevel": energy,
                "roundTripTime": 0.035,
                "concealedSamples": 0
            })
        return samples
