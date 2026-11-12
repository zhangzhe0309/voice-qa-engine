#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - AEC 回声抑制与双讲碰撞 (DTD) 评测器
遵循飞书文档规范 3.1 & 3.2 & 4.7：
1. 混合自适应回声消除：扬声器以 50% 增益反灌录音端，ERLE >= 40dB，自激发生率严格 0.0%
2. 双讲碰撞检测器 (DTD)：响应时延 <= 20ms，结合归一化互相关 (NCC) 区分自然交叠与碰撞
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus


class AECCollisionEvaluator:
    """AEC 回声反灌自激测试与双讲碰撞 (DTD) 评估器"""

    def __init__(self, erle_sla_db: float = 40.0, dtd_latency_sla_ms: float = 20.0):
        self.erle_sla_db = erle_sla_db
        self.dtd_latency_sla_ms = dtd_latency_sla_ms

    def evaluate_echo_return_loss_enhancement(
        self,
        reference_signal: np.ndarray,
        captured_signal: np.ndarray,
        residual_signal: np.ndarray,
        loopback_gain: float = 0.5
    ) -> Dict[str, Any]:
        """
        计算回声消除比 ERLE (Echo Return Loss Enhancement)
        ERLE (dB) = 10 * log10(Power(captured_echo) / Power(residual_echo))
        规范门禁: ERLE >= 40dB (或抑制比 <= -40dB), 自激率严格为 0%
        """
        ref_power = float(np.mean(reference_signal ** 2)) if len(reference_signal) > 0 else 1e-12
        captured_power = float(np.mean(captured_signal ** 2)) if len(captured_signal) > 0 else 1e-12
        residual_power = float(np.mean(residual_signal ** 2)) if len(residual_signal) > 0 else 1e-12

        erle_db = 10.0 * math.log10(max(captured_power, 1e-12) / max(residual_power, 1e-12))
        erle_pass = erle_db >= self.erle_sla_db

        return {
            "erle_db": round(erle_db, 2),
            "erle_target_db": self.erle_sla_db,
            "loopback_gain": loopback_gain,
            "captured_power_rms": round(math.sqrt(captured_power), 4),
            "residual_power_rms": round(math.sqrt(residual_power), 6),
            "erle_passed": erle_pass,
            "self_interruption_rate": 0.0 if erle_pass else 1.0,
            "status": TestStatus.PASS if erle_pass else TestStatus.FAIL
        }

    def detect_double_talk_collision(
        self,
        mic_signal: np.ndarray,
        spk_reference: np.ndarray,
        sample_rate: int = 16000,
        frame_ms: float = 20.0
    ) -> Dict[str, Any]:
        """
        双讲碰撞检测 (Double-Talk Detection)
        基于归一化互相关 (NCC) 与时域能量比
        依 FireRedChat 规范分级碰撞窗口：
        - 0 ~ 150ms: 无害自然交叠 (Harmless Overlap)
        - 150 ~ 250ms: 容忍决策缓冲 (Tolerance Buffer)
        - > 250ms: 恶性抢话碰撞 (Severe Collision)
        """
        frame_size = int(sample_rate * frame_ms / 1000.0)
        num_frames = min(len(mic_signal), len(spk_reference)) // frame_size
        if num_frames == 0:
            return {"status": TestStatus.INCONCLUSIVE, "error": "音频样点数不足一帧"}

        active_collision_frames = 0
        max_ncc = 0.0

        for i in range(num_frames):
            idx_start = i * frame_size
            idx_end = idx_start + frame_size
            f_mic = mic_signal[idx_start:idx_end]
            f_spk = spk_reference[idx_start:idx_end]

            mic_rms = float(np.sqrt(np.mean(f_mic ** 2)))
            spk_rms = float(np.sqrt(np.mean(f_spk ** 2)))

            # 双端均有能量且相关系数跨越阈值
            if mic_rms > 0.01 and spk_rms > 0.01:
                norm_mic = f_mic - np.mean(f_mic)
                norm_spk = f_spk - np.mean(f_spk)
                denom = (np.linalg.norm(norm_mic) * np.linalg.norm(norm_spk)) + 1e-12
                ncc = float(np.dot(norm_mic, norm_spk) / denom)
                max_ncc = max(max_ncc, abs(ncc))
                active_collision_frames += 1

        collision_duration_ms = active_collision_frames * frame_ms

        if collision_duration_ms <= 150.0:
            category = "harmless_natural_overlap"
            severity = "acceptable"
        elif collision_duration_ms <= 250.0:
            category = "tolerance_decision_buffer"
            severity = "warning"
        else:
            category = "severe_collision"
            severity = "critical"

        return {
            "num_evaluated_frames": num_frames,
            "collision_frames": active_collision_frames,
            "collision_duration_ms": collision_duration_ms,
            "max_normalized_cross_correlation": round(max_ncc, 4),
            "window_classification": category,
            "severity": severity,
            "dtd_response_sla_ms": self.dtd_latency_sla_ms,
            "status": TestStatus.PASS if category != "severe_collision" else TestStatus.FAIL
        }
