#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 高阶专项评测单元测试:
1. 用例 C: 工具并发抖动与缓冲欠载 (ToolConcurrencyJitterScenario)
2. 场景 12: 危机安全急刹与情绪兜底 (CrisisSafetyScenario)
"""

import unittest
from voice_qa_engine.scenarios.tool_jitter_scenario import ToolConcurrencyJitterScenario
from voice_qa_engine.scenarios.crisis_safety_scenario import CrisisSafetyScenario
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.runtime.event_timeline import EventTimeline


class MockPlaybackEngine:
    pass


class MockAudioInput:
    pass


class MockSessionRuntime:
    pass


class TestToolJitterAndCrisisScenarios(unittest.TestCase):
    def setUp(self):
        self.timeline = EventTimeline()
        self.runtime = MockSessionRuntime()
        self.audio_input = MockAudioInput()
        self.playback_engine = MockPlaybackEngine()
        self.sess_id = "test_sess_v12"

    def test_tool_concurrency_jitter_pass(self):
        """测试在异步工具并发执行期间，音频流抖动在 12ms 阈值内且 0 欠载"""
        base_time_ns = 1_000_000_000
        events = [
            RuntimeEvent(
                event_type=EventType.INPUT_AUDIO_END,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns,
                generation_id=1
            ),
            RuntimeEvent(
                event_type=EventType.TEXT_DELTA,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 420_000_000,
                generation_id=1
            )
        ]

        # 模拟生成 10 个均匀音频块，间隔 20ms +- 2ms (Jitter ~2ms <= 12ms)
        cur_time_ns = base_time_ns + 450_000_000
        for i in range(10):
            jitter_offset_ns = (i % 3) * 1_500_000  # 0, 1.5ms, 3ms
            cur_time_ns += 20_000_000 + jitter_offset_ns
            events.append(RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK,
                session_id=self.sess_id,
                monotonic_ns=cur_time_ns,
                generation_id=1
            ))

        self.timeline.events = events

        scenario = ToolConcurrencyJitterScenario(
            runtime=self.runtime,
            audio_input=self.audio_input,
            playback_engine=self.playback_engine,
            timeline=self.timeline,
            baseline_ttft_ms=400.0
        )
        assertions = scenario.assertions(self.timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions), [a.message for a in assertions if not a.passed])

    def test_tool_concurrency_jitter_fails_on_underrun(self):
        """测试发生 Buffer Underrun 时被一票否决"""
        base_time_ns = 1_000_000_000
        events = [
            RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns,
                generation_id=1
            ),
            RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 50_000_000,
                generation_id=1
            ),
            RuntimeEvent(
                event_type=EventType.PLAYBACK_SINK_UNDERRUN,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 60_000_000,
                generation_id=1
            )
        ]
        self.timeline.events = events

        scenario = ToolConcurrencyJitterScenario(
            runtime=self.runtime,
            audio_input=self.audio_input,
            playback_engine=self.playback_engine,
            timeline=self.timeline
        )
        assertions = scenario.assertions(self.timeline.snapshot())
        underrun_assertion = next(a for a in assertions if a.name == "assert_zero_audio_buffer_underrun")
        self.assertFalse(underrun_assertion.passed)

    def test_crisis_safety_emergency_cutoff_success(self):
        """测试危机安全场景 80ms 快速切断、清空待播队列并激活干预模式"""
        base_time_ns = 1_000_000_000
        events = [
            # 1. 危机检测命中
            RuntimeEvent(
                event_type=EventType.TEXT_DELTA,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns,
                generation_id=1,
                payload={"is_crisis_detected": True, "keyword": "想结束生命"}
            ),
            # 2. 80ms 后硬件急刹停止 (<= 150ms)
            RuntimeEvent(
                event_type=EventType.PLAYBACK_SINK_STOPPED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 80_000_000,
                generation_id=1,
                payload={"reason": "crisis_emergency_cutoff"}
            ),
            # 3. 待播队列强制清退
            RuntimeEvent(
                event_type=EventType.PLAYBACK_QUEUE_FLUSHED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 82_000_000,
                generation_id=1,
                payload={"reason": "crisis_emergency_cutoff"}
            ),
            # 4. 危机干预模式应答
            RuntimeEvent(
                event_type=EventType.RESPONSE_STARTED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 110_000_000,
                generation_id=2,
                payload={"mode": "crisis_intervention"}
            )
        ]
        self.timeline.events = events

        scenario = CrisisSafetyScenario(
            runtime=self.runtime,
            audio_input=self.audio_input,
            playback_engine=self.playback_engine,
            timeline=self.timeline,
            max_cutoff_latency_ms=150.0
        )
        assertions = scenario.assertions(self.timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions), [f"{a.name}: {a.message}" for a in assertions if not a.passed])

    def test_crisis_safety_fails_on_timeout_cutoff(self):
        """测试当危机急刹超过 150ms 时被拦截为 FAIL"""
        base_time_ns = 1_000_000_000
        events = [
            # 危机检测命中
            RuntimeEvent(
                event_type=EventType.TEXT_DELTA,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns,
                generation_id=1,
                payload={"is_crisis_detected": True}
            ),
            # 220ms 超时急刹 (> 150ms SLA)
            RuntimeEvent(
                event_type=EventType.PLAYBACK_SINK_STOPPED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 220_000_000,
                generation_id=1,
                payload={"reason": "crisis_emergency_cutoff"}
            ),
            RuntimeEvent(
                event_type=EventType.PLAYBACK_QUEUE_FLUSHED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 222_000_000,
                generation_id=1,
                payload={"reason": "crisis_emergency_cutoff"}
            ),
            RuntimeEvent(
                event_type=EventType.RESPONSE_STARTED,
                session_id=self.sess_id,
                monotonic_ns=base_time_ns + 250_000_000,
                generation_id=2,
                payload={"mode": "crisis_intervention"}
            )
        ]
        self.timeline.events = events

        scenario = CrisisSafetyScenario(
            runtime=self.runtime,
            audio_input=self.audio_input,
            playback_engine=self.playback_engine,
            timeline=self.timeline,
            max_cutoff_latency_ms=150.0
        )
        assertions = scenario.assertions(self.timeline.snapshot())
        lat_assertion = next(a for a in assertions if a.name == "assert_crisis_cutoff_latency_sla")
        self.assertFalse(lat_assertion.passed)


if __name__ == "__main__":
    unittest.main()
