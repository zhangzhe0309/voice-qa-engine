#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 黄金基线 soak 域测试（压缩时长跑通六阶段编排）"""

import time
import tempfile
import unittest

from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext
from voice_qa_engine.scenarios.golden_baseline_scenario import GoldenBaselineSoakScenario
from voice_qa_engine.runtime.session_runtime import SessionRuntime
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.runtime.context_stack import ContextCallStack
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.sink import HannFadeoutSink
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput


def _make_ctx_factory(base: str):
    async def factory(scenario_id: str) -> ScenarioContext:
        sid = f"{scenario_id}_{int(time.time() * 1e6)}"
        timeline = EventTimeline()
        engine = PlaybackEngine(sink=HannFadeoutSink(), timeline=timeline)
        audio_input = StreamingAudioInput()
        runtime = SessionRuntime(
            session_id=sid, audio_input=audio_input, playback_engine=engine,
            timeline=timeline, recorder=EvidenceRecorder(base_dir=base, session_id=sid),
            context_stack=ContextCallStack(default_ttl=10),
        )
        await runtime.connect()
        await runtime.start()
        return ScenarioContext(runtime, audio_input, engine, timeline, runtime.recorder, {})
    return factory


class TestGoldenBaselineSoak(unittest.IsolatedAsyncioTestCase):
    async def test_six_phase_soak_compressed_run(self):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
            return GoldenBaselineSoakScenario(
                runtime, audio_input, playback_engine, timeline,
                total_duration_ms=6000.0, time_compression=100.0,
                n_interrupts=15, n_subtasks=7, n_backchannels=20,
            )

        registry.register(build)
        report = await SuiteRunner("soak_test", _make_ctx_factory(base)).run(registry)
        entry = report["scenarios"][0]
        self.assertEqual(entry["status"], TestStatus.SIMULATED_PASS, msg=str(entry["errors"]))
        passed = {a["name"] for a in entry["assertions"] if a["passed"]}
        self.assertIn("assert_six_phases_completed", passed)
        self.assertIn("assert_fatigue_injection_counts", passed)
        self.assertIn("assert_memory_drift_within_sla", passed)
        self.assertIn("assert_interruption_invariance", passed)
        self.assertIn("assert_chatter_silence_no_downlink", passed)
        self.assertEqual(entry["metrics"]["injections"]["interrupts"], 16)  # 15 疲劳 + 1 阶段 2
        self.assertEqual(entry["metrics"]["injections"]["backchannels"], 20)
        self.assertTrue(entry["evidence_valid"])

    async def test_memory_drift_assertion_rejects_null(self):
        scenario = GoldenBaselineSoakScenario(None, None, None, None, time_compression=1.0)
        results = scenario.assertions([], {"memory_drift_ratio": None, "phases": {}, "injections": {}})
        drift = next(a for a in results if a.name == "assert_memory_drift_within_sla")
        self.assertFalse(drift.passed)  # 无证据不虚标
        self.assertIn("psutil", drift.message)


if __name__ == "__main__":
    unittest.main()
