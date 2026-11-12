#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段 5 高阶专项评测用例单元测试
验证：
1. BackchannelScenario: 微弱附和倾听无缝免打断 (零 CANCEL/FLUSH)
2. HesitationToleranceScenario: 思考留白正常区间 Holding 态 (抢话率 <= 1.5%) 与超时交权
3. AudioImpulseNoiseScenario: 短时脉冲杂音抗扰 (False Barge-in 严格为 0 次)
"""

import unittest
import time
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.scenarios.advanced_scenarios import (
    BackchannelScenario,
    HesitationToleranceScenario,
    AudioImpulseNoiseScenario
)


class TestAdvancedScenarios(unittest.TestCase):
    def test_backchannel_scenario_success(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        timeline.events = [
            RuntimeEvent(event_type=EventType.SESSION_READY, session_id="s1", monotonic_ns=t0),
            RuntimeEvent(event_type=EventType.AUDIO_CHUNK, session_id="s1", sequence_id=0, monotonic_ns=t0 + 20),
            RuntimeEvent(event_type=EventType.INPUT_SPEECH_STARTED, session_id="s1", monotonic_ns=t0 + 30, payload={"phrase": "嗯"}),
            RuntimeEvent(event_type=EventType.AUDIO_CHUNK, session_id="s1", sequence_id=1, monotonic_ns=t0 + 40),
        ]
        scenario = BackchannelScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions))

    def test_backchannel_scenario_fails_on_false_cancel(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        timeline.events = [
            RuntimeEvent(event_type=EventType.RESPONSE_CANCELLED, session_id="s1", monotonic_ns=t0 + 10),
            RuntimeEvent(event_type=EventType.PLAYBACK_QUEUE_FLUSHED, session_id="s1", monotonic_ns=t0 + 12),
        ]
        scenario = BackchannelScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertFalse(all(a.passed for a in assertions))

    def test_hesitation_tolerance_scenario_holding_pass(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        timeline.events = [
            RuntimeEvent(event_type=EventType.INPUT_SPEECH_STARTED, session_id="s1", monotonic_ns=t0),
            RuntimeEvent(event_type=EventType.INPUT_AUDIO_END, session_id="s1", monotonic_ns=t0 + 10),
        ]
        scenario = HesitationToleranceScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline, pause_duration_s=2.0)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions))

    def test_hesitation_tolerance_scenario_fails_on_premature_interruption(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        timeline.events = [
            RuntimeEvent(
                event_type=EventType.RESPONSE_STARTED,
                session_id="s1",
                monotonic_ns=t0 + 50,
                payload={"is_premature_interruption": True}
            )
        ]
        scenario = HesitationToleranceScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline, pause_duration_s=1.5)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertFalse(all(a.passed for a in assertions))

    def test_impulse_noise_scenario_zero_false_barge_in(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        # 仅有正常的音频帧，无脉冲杂音引发的打断
        timeline.events = [
            RuntimeEvent(event_type=EventType.AUDIO_CHUNK, session_id="s1", sequence_id=0, monotonic_ns=t0)
        ]
        scenario = AudioImpulseNoiseScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions))

    def test_impulse_noise_scenario_detects_false_barge_in(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        timeline.events = [
            RuntimeEvent(
                event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
                session_id="s1",
                monotonic_ns=t0,
                payload={"trigger": "impulse_noise"}
            )
        ]
        scenario = AudioImpulseNoiseScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertFalse(all(a.passed for a in assertions))


if __name__ == "__main__":
    unittest.main()
