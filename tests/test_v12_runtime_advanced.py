#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 进阶运行时与 AEC 回声评测单元测试
覆盖：
1. AECCollisionEvaluator: ERLE 回声消除比计算与双讲碰撞 (DTD) 分级窗口
2. DualChannelTurnTakingArbiter: 声学快通道压麦 (Duck-down 20%) 与语义慢通道决策 (附和恢复、留白Holding、打断CANCEL)
3. ToolAbortManager: 异步工具全局 AbortController 熔断 (<= 10ms) 与防脏数据污染
"""

import unittest
import numpy as np

from voice_qa_engine.evaluators.aec_evaluator import AECCollisionEvaluator
from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.runtime.turn_taking import (
    DualChannelTurnTakingArbiter,
    TurnTakingDecision,
)
from voice_qa_engine.runtime.tool_runtime import (
    ToolAbortManager,
    ToolCallState,
)


class TestRuntimeAdvanced(unittest.TestCase):
    def test_aec_evaluator_erle_pass(self):
        evaluator = AECCollisionEvaluator(erle_sla_db=40.0)
        # 产生高能量扬声器参考信号与极低残余回声信号 (抑制比达到 > 40dB)
        ref = np.ones(1600, dtype=np.float32) * 0.5
        captured = ref * 0.5  # 50% 增益反灌
        residual = np.ones(1600, dtype=np.float32) * 0.002  # 经 AEC 消除后微弱残存

        res = evaluator.evaluate_echo_return_loss_enhancement(ref, captured, residual, loopback_gain=0.5)
        self.assertEqual(res["status"], TestStatus.PASS)
        self.assertGreaterEqual(res["erle_db"], 40.0)
        self.assertEqual(res["self_interruption_rate"], 0.0)

    def test_dtd_collision_window_classification(self):
        evaluator = AECCollisionEvaluator()
        # 16kHz, 5 帧 (100ms) 交叠 -> 自然无害交叠
        mic = np.ones(16000 // 10, dtype=np.float32) * 0.2
        spk = np.ones(16000 // 10, dtype=np.float32) * 0.3
        res = evaluator.detect_double_talk_collision(mic, spk, sample_rate=16000, frame_ms=20.0)

        self.assertEqual(res["status"], TestStatus.PASS)
        self.assertEqual(res["window_classification"], "harmless_natural_overlap")

    def test_dual_channel_arbiter_fast_ducking_and_slow_backchannel(self):
        arbiter = DualChannelTurnTakingArbiter()

        # 1. 快通道：检测到高能量输入，瞬间压低音量至 20%
        loud_samples = np.ones(320, dtype=np.float32) * 0.3
        fast_res = arbiter.fast_acoustic_track(loud_samples)
        self.assertTrue(fast_res["speech_detected"])
        self.assertEqual(fast_res["volume_ratio"], 0.2)

        # 2. 慢通道：识别为微弱附和 ("嗯") -> 恢复音量 100% 且不发 CANCEL
        slow_res = arbiter.slow_semantic_track("嗯")
        self.assertEqual(slow_res["decision"], TurnTakingDecision.BACKCHANNEL_RESTORE)
        self.assertEqual(slow_res["volume_ratio"], 1.0)
        self.assertFalse(slow_res["emit_cancel"])

    def test_dual_channel_arbiter_hesitation_holding(self):
        arbiter = DualChannelTurnTakingArbiter()
        # 识别为留白填充词 ("呃...") 且停顿 1.5s -> 保持 Holding 倾听
        res = arbiter.slow_semantic_track("呃...", pause_duration_s=1.5, voice_active=False)
        self.assertEqual(res["decision"], TurnTakingDecision.HOLDING_LISTEN)
        self.assertFalse(res["emit_cancel"])
        self.assertTrue(arbiter.is_holding)

    def test_dual_channel_arbiter_effective_barge_in(self):
        arbiter = DualChannelTurnTakingArbiter()
        # 识别为有效打断新指令 ("停一下别说了") -> 发出 CANCEL
        res = arbiter.slow_semantic_track("停一下别说了")
        self.assertEqual(res["decision"], TurnTakingDecision.EFFECTIVE_BARGE_IN)
        self.assertTrue(res["emit_cancel"])

    def test_tool_abort_manager_prevents_stale_data_pollution(self):
        manager = ToolAbortManager(abort_sla_ms=10.0)

        # 代次 1 注册异步工具
        manager.register_tool_call("tool_weather_01", "query_weather", generation_id=1)

        # 触发打断，熔断代次 1 的所有在途工具
        aborted = manager.abort_for_generation(generation_id=1)
        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0].state, ToolCallState.ABORTED)
        self.assertLessEqual(aborted[0].abort_latency_ms, 10.0)

        # 模拟外部异步任务延迟返回结果，提交给代次 1
        accepted = manager.submit_tool_result("tool_weather_01", {"weather": "rain"}, generation_id=1)
        self.assertFalse(accepted)  # 严格一票否决丢弃，防止脏数据上屏

        summary = manager.get_summary_metrics()
        self.assertTrue(summary["zero_stale_contamination"])


if __name__ == "__main__":
    unittest.main()
