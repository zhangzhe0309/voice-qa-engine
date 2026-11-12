#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 双 Agent 对弈域测试: ToolExecutionLedger / VirtualUserAgent / SparringScenario"""

import os
import time
import tempfile
import unittest

from voice_qa_engine.agents.tool_ledger import ToolExecutionLedger
from voice_qa_engine.agents.virtual_user import ScriptedVUA, LLMPoweredVUA
from voice_qa_engine.agents.sparring_scenario import SparringScenario, MockDialogBackend
from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext
from voice_qa_engine.runtime.session_runtime import SessionRuntime
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.runtime.context_stack import ContextCallStack
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.sink import HannFadeoutSink
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput


class TestToolExecutionLedger(unittest.TestCase):
    def test_consistent_claims_pass(self):
        ledger = ToolExecutionLedger()
        ledger.record_claim("c1", "query_weather", {"city": "北京"})
        ledger.record_execution("c1", "query_weather", True, {"city": "北京"})
        report = ledger.verify()
        self.assertTrue(report["consistent"])
        self.assertTrue(all(a.passed for a in ledger.assertions()))

    def test_fake_completion_one_vote_veto(self):
        ledger = ToolExecutionLedger()
        ledger.record_claim("c1", "write_schedule")
        ledger.record_execution("c1", "write_schedule", False)  # 声称完成但未落盘
        report = ledger.verify()
        self.assertFalse(report["consistent"])
        self.assertEqual(report["fake_completions"], ["c1"])

    def test_missing_execution_detected(self):
        ledger = ToolExecutionLedger()
        ledger.record_claim("c1", "query_weather")
        report = ledger.verify()
        self.assertFalse(report["consistent"])
        self.assertEqual(report["missing_executions"], ["c1"])

    def test_ghost_execution_detected(self):
        ledger = ToolExecutionLedger()
        ledger.record_execution("c9", "mystery_tool", True)
        report = ledger.verify()
        self.assertFalse(report["consistent"])
        self.assertEqual(report["unclaimed_executions"], ["c9"])


class TestVirtualUserAgent(unittest.IsolatedAsyncioTestCase):
    async def test_scripted_vua_sequence(self):
        vua = ScriptedVUA(["帮我规划五天方案", "换李老师负责", "接着说"])
        u1 = await vua.next_utterance([], "goal")
        u2 = await vua.next_utterance([], "goal")
        u3 = await vua.next_utterance([], "goal")
        u4 = await vua.next_utterance([], "goal")  # 耗尽后复用队尾
        self.assertEqual(u2, "换李老师负责")
        self.assertEqual(u4, u3)

    async def test_llm_vua_unavailable_without_key(self):
        vua = LLMPoweredVUA(api_key="")
        self.assertFalse(vua.is_available())
        with self.assertRaises(RuntimeError):
            await vua.next_utterance([], "goal")


def _make_ctx_factory(base: str, fake_completion: bool = False):
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
        return ScenarioContext(runtime, audio_input, engine, timeline, runtime.recorder, {})
    return factory


class TestSparringScenario(unittest.IsolatedAsyncioTestCase):
    async def _run(self, fake_completion=False):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def register():
            vua = ScriptedVUA([f"推进第{i}步方案" for i in range(10)])
            backend = MockDialogBackend(fake_completion=fake_completion, total_rounds=10)

            def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
                return SparringScenario(runtime, audio_input, playback_engine, timeline,
                                        vua=vua, backend=backend, min_rounds=10)

            registry.register(build)
        register()

        report = await SuiteRunner("sparring_test", _make_ctx_factory(base)).run(registry)
        return report, base

    async def test_sparring_simulated_pass_with_real_ledger(self):
        report, _ = await self._run(fake_completion=False)
        entry = report["scenarios"][0]
        self.assertEqual(entry["status"], TestStatus.SIMULATED_PASS)
        self.assertTrue(entry["metrics"]["tool_ledger"]["consistent"])
        self.assertTrue(all(a["passed"] for a in entry["assertions"]))

    async def test_sparring_fake_completion_fails(self):
        report, _ = await self._run(fake_completion=True)
        entry = report["scenarios"][0]
        self.assertEqual(entry["status"], TestStatus.FAIL)
        self.assertFalse(entry["metrics"]["tool_ledger"]["consistent"])
        failed_names = [a["name"] for a in entry["assertions"] if not a["passed"]]
        self.assertIn("assert_tool_claims_physically_executed", failed_names)

    async def test_sparring_not_run_without_llm_key(self):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
            return SparringScenario(runtime, audio_input, playback_engine, timeline,
                                    vua=LLMPoweredVUA(api_key=""), min_rounds=3)

        registry.register(build)
        report = await SuiteRunner("sparring_nokey", _make_ctx_factory(base)).run(registry)
        self.assertEqual(report["scenarios"][0]["status"], TestStatus.NOT_RUN)

    async def test_interrupt_params_merged(self):
        base = tempfile.mkdtemp()
        vua = ScriptedVUA(["开始吧", "继续", "继续", "继续", "继续", "继续"])
        backend = MockDialogBackend(total_rounds=6)
        timeline = EventTimeline()
        engine = PlaybackEngine(sink=HannFadeoutSink(), timeline=timeline)
        audio_input = StreamingAudioInput()
        runtime = SessionRuntime(
            session_id="spar_int", audio_input=audio_input, playback_engine=engine,
            timeline=timeline, recorder=EvidenceRecorder(base_dir=tempfile.mkdtemp(), session_id="spar_int"),
            context_stack=ContextCallStack(default_ttl=10),
        )
        scenario = SparringScenario(runtime, audio_input, engine, timeline,
                                    vua=vua, backend=backend, min_rounds=6, interrupt_round=3)
        result = await scenario.execute()
        self.assertEqual(result.metrics["interrupted_round"], 3)
        self.assertEqual(result.status, TestStatus.SIMULATED_PASS)


if __name__ == "__main__":
    unittest.main()
