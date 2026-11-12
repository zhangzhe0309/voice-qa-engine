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
