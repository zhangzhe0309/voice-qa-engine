#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 双通道话权仲裁决策引擎 (Dual-Channel Turn-Taking Arbiter)
遵循飞书文档规范 第 4 节：
1. 声学快通道 (20ms): 能量与过零率检测，瞬间压低模型音量至 20% (Duck-down)
2. 语义慢通道 (60~150ms): 分类仲裁
   - A. 附和词 ("嗯/对/好的") -> 恢复音量 100%，继续流式下发
   - B. 思考留白 ("呃...2s") -> 进入 Holding 倾听态，模型静音但不交权
   - C. 有效打断 ("停一下/改要求") -> 发出 CANCEL 信令，全链路彻底截断
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np


class TurnTakingDecision(str, Enum):
    BACKCHANNEL_RESTORE = "BACKCHANNEL_RESTORE"
    HOLDING_LISTEN = "HOLDING_LISTEN"
    EFFECTIVE_BARGE_IN = "EFFECTIVE_BARGE_IN"
    SILENCE_CONTINUE = "SILENCE_CONTINUE"


class DualChannelTurnTakingArbiter:
    """双通道话权仲裁中枢"""

    DEFAULT_BACKCHANNELS = {"嗯", "对", "好的", "行", "对的", "是的", "aha", "yeah", "mhm"}
    DEFAULT_HESITATION_FILLERS = {"呃", "那个", "就是", "然后", "这个", "umm", "uh"}

    def __init__(self, energy_threshold: float = 0.02, duck_volume_ratio: float = 0.2):
        self.energy_threshold = energy_threshold
        self.duck_volume_ratio = duck_volume_ratio
        self.current_volume = 1.0
        self.is_holding = False

    def fast_acoustic_track(self, frame_samples: np.ndarray) -> Dict[str, Any]:
        """
        ① 声学快通道 (20ms): 纯物理能量与过零率探测
        第一时间压麦 (Duck-down 20%)，提供极佳拟人体感
        """
        if len(frame_samples) == 0:
            return {"speech_detected": False, "volume_ratio": 1.0}

        rms = float(np.sqrt(np.mean(frame_samples ** 2)))
        zero_crossings = int(np.sum(np.abs(np.diff(np.sign(frame_samples)))) / 2)
        speech_detected = rms >= self.energy_threshold

        if speech_detected:
            self.current_volume = self.duck_volume_ratio
        else:
            self.current_volume = 1.0

        return {
            "speech_detected": speech_detected,
            "rms_energy": round(rms, 4),
            "zero_crossings": zero_crossings,
            "volume_ratio": self.current_volume,
            "latency_ms": 20.0
        }

    # FireRedChat 碰撞窗口分级阈值 (ms)
    OVERLAP_HARMLESS_MAX_MS = 150.0
    OVERLAP_TOLERANCE_MAX_MS = 250.0

    @classmethod
    def classify_collision_window(cls, overlap_ms: float) -> Dict[str, Any]:
        """
        话权交叠窗口分级（FireRedChat 模型）:
        0~150ms 无害自然交叠 | 150~250ms 容忍决策缓冲 | >250ms 恶性抢话碰撞
        """
        if overlap_ms <= cls.OVERLAP_HARMLESS_MAX_MS:
            grade = "harmless_overlap"
        elif overlap_ms <= cls.OVERLAP_TOLERANCE_MAX_MS:
            grade = "tolerance_buffer"
        else:
            grade = "malignant_collision"
        return {"overlap_ms": overlap_ms, "grade": grade}

    @classmethod
    def floor_yield_metrics(cls, collision_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        主动让步率 (Floor Yielding Ratio):
        collision_events: [{"overlap_ms": float, "yielded": bool}]
        SLA: >= 0.98（恶性碰撞中模型必须主动让步）
        """
        malignant = [e for e in collision_events
                     if cls.classify_collision_window(e.get("overlap_ms", 0))["grade"] == "malignant_collision"]
        yielded = [e for e in malignant if e.get("yielded")]
        rate = (len(yielded) / len(malignant)) if malignant else None
        return {
            "total_collisions": len(collision_events),
            "malignant_collisions": len(malignant),
            "malignant_yielded": len(yielded),
            "floor_yielding_rate": round(rate, 4) if rate is not None else None,
            "sla": 0.98,
        }

    def slow_semantic_track(
        self,
        text_content: str,
        pause_duration_s: float = 0.0,
        voice_active: bool = True
    ) -> Dict[str, Any]:
        """
        ② 语义慢通道 (60~150ms): 上下文前瞻与意图分类
        """
        stripped = text_content.strip()

        # 1. 附和词识别
        if stripped in self.DEFAULT_BACKCHANNELS:
            self.current_volume = 1.0
            self.is_holding = False
            return {
                "decision": TurnTakingDecision.BACKCHANNEL_RESTORE,
                "action": "restore_volume_100",
                "volume_ratio": 1.0,
                "emit_cancel": False,
                "reason": f"命中弱附和词 '{stripped}'，恢复音量继续流式播报"
            }

        # 2. 思考迟疑与填充词
        is_filler = any(f in stripped for f in self.DEFAULT_HESITATION_FILLERS) or (not voice_active and pause_duration_s > 0)
        if is_filler and pause_duration_s <= 2.5:
            self.is_holding = True
            return {
                "decision": TurnTakingDecision.HOLDING_LISTEN,
                "action": "maintain_holding_listening",
                "volume_ratio": 0.0,
                "emit_cancel": False,
                "reason": f"识别为人类正常思考留白 ({pause_duration_s}s)，保持 Holding 态倾听，不提前抢话"
            }

        # 3. 超时交权 (>= 3.0s)
        if not voice_active and pause_duration_s >= 3.0:
            self.is_holding = False
            return {
                "decision": TurnTakingDecision.EFFECTIVE_BARGE_IN,
                "action": "timeout_turn_shift",
                "volume_ratio": 1.0,
                "emit_cancel": False,
                "reason": f"思考留白已超时 ({pause_duration_s}s >= 3.0s)，系统礼貌发起交权"
            }

        # 4. 有效打断
        if stripped:
            self.current_volume = 0.0
            self.is_holding = False
            return {
                "decision": TurnTakingDecision.EFFECTIVE_BARGE_IN,
                "action": "emit_cancel_and_flush",
                "volume_ratio": 0.0,
                "emit_cancel": True,
                "reason": f"识别为有效打断与新需求 '{stripped}'，发起 CANCEL 并立即 Flush 本地队列"
            }

        return {
            "decision": TurnTakingDecision.SILENCE_CONTINUE,
            "action": "continue_current_state",
            "volume_ratio": self.current_volume,
            "emit_cancel": False,
            "reason": "正常流式状态延续"
        }
