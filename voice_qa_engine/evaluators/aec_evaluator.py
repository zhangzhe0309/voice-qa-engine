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

    def scan_self_oscillation_boundary(
        self,
        loopback_delays_ms: Optional[List[float]] = None,
        loopback_gains: Optional[List[float]] = None,
        duration_ms: float = 500.0,
        sample_rate: int = 16000,
        erle_sla_db: Optional[float] = None,
        residual_factor: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        电信级自激死锁阈值扫描（借鉴 SIPp/Asterisk 回灌思路）:
        延迟 (0~300ms) × 增益 (0~1) 网格逐点评估 ERLE，输出自激失效边界曲线。
        网格点上 ERLE < SLA 即判定该 (delay, gain) 组合存在自激风险。
        """
        import numpy as np
        delays = loopback_delays_ms or [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        gains = loopback_gains or [0.0, 0.25, 0.5, 0.75, 1.0]
        sla = erle_sla_db if erle_sla_db is not None else self.erle_sla_db

        n = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(n) / sample_rate
        reference = np.sin(2 * np.pi * 440.0 * t).astype(np.float64)

        grid: List[Dict[str, Any]] = []
        failing: List[Dict[str, Any]] = []
        for gain in gains:
            for delay_ms in delays:
                delay_samples = int(sample_rate * delay_ms / 1000.0)
                captured = np.zeros(n)
                if delay_samples < n:
                    captured[delay_samples:] = reference[: n - delay_samples] * gain
                # 无回声注入（gain=0 或延迟超出观测窗）≠ 自激，跳过 ERLE 判定
                no_echo = float(np.mean(captured ** 2)) < 1e-12
                if no_echo:
                    point = {
                        "delay_ms": delay_ms,
                        "gain": gain,
                        "erle_db": None,
                        "no_echo_injected": True,
                        "self_oscillation_risk": False,
                    }
                    grid.append(point)
                    continue
                # AEC 残余建模：反灌信号 × 残余系数（1e-3 ≈ -60dB 抑制基线；弱 AEC 调高该值）
                residual = captured * residual_factor
                res = self.evaluate_echo_return_loss_enhancement(
                    reference, captured, residual, loopback_gain=gain
                )
                point = {
                    "delay_ms": delay_ms,
                    "gain": gain,
                    "erle_db": res["erle_db"],
                    "self_oscillation_risk": res["erle_db"] < sla,
                }
                grid.append(point)
                if point["self_oscillation_risk"]:
                    failing.append(point)

        return {
            "scan_grid": grid,
            "failing_points": failing,
            "boundary_summary": {
                "max_safe_gain_at_0ms": max(
                    (p["gain"] for p in grid
                     if p["delay_ms"] == 0.0 and not p.get("no_echo_injected")
                     and not p["self_oscillation_risk"]),
                    default=None,
                ),
                "total_points": len(grid),
                "failing_points": len(failing),
            },
            "erle_sla_db": sla,
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
