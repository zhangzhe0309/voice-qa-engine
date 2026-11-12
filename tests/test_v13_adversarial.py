#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 对抗增强包测试: 统计置信 / 不流利耐受 / 谐音误唤醒 / 碰撞窗口 / 停算熔断 / 语料矩阵"""

import time
import tempfile
import unittest

from voice_qa_engine.evaluators.statistics import ratio_verdict, wilson_interval, assertion_from_verdict
from voice_qa_engine.evaluators.abort_evaluator import inference_abort_assertions, extract_abort_metrics
from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.runtime.turn_taking import DualChannelTurnTakingArbiter
from voice_qa_engine.audio.corpus_matrix import CorpusMatrix, HOMOPHONE_TRAPS
from voice_qa_engine.scenarios.disfluency_scenario import DisfluencyScenario
from voice_qa_engine.scenarios.homophone_wakeup_scenario import HomophoneWakeupScenario
from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext
from voice_qa_engine.runtime.session_runtime import SessionRuntime
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.sink import HannFadeoutSink
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput


class TestStatistics(unittest.TestCase):
    def test_wilson_zero_failures_upper_bound(self):
        low, high = wilson_interval(0, 200)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 0.02)

    def test_small_sample_inconclusive(self):
        v = ratio_verdict("t", 199, 200, sla=0.985, min_samples=500)
        self.assertTrue(v.inconclusive)
        self.assertFalse(v.passed)
        self.assertIn("无法统计判定", v.message)

    def test_large_sample_pass_and_fail(self):
        # 95% CI 下证明 ≥98.5% 需要足够样本：n=2000, 10 例失败 → 下界 ≈ 0.9915
        ok = ratio_verdict("ok", 1990, 2000, sla=0.985, min_samples=200)
        self.assertTrue(ok.passed)
        bad = ratio_verdict("bad", 1950, 2000, sla=0.985, min_samples=200)
        self.assertFalse(bad.passed)

    def test_lower_is_better_uses_upper_bound(self):
        v = ratio_verdict("err", 15, 2000, sla=0.015, higher_is_better=False, min_samples=200)
        self.assertTrue(v.passed)
        v2 = ratio_verdict("err2", 60, 2000, sla=0.015, higher_is_better=False, min_samples=200)
        self.assertFalse(v2.passed)


class TestCollisionWindow(unittest.TestCase):
    def test_grade_boundaries(self):
        arb = DualChannelTurnTakingArbiter
        self.assertEqual(arb.classify_collision_window(100)["grade"], "harmless_overlap")
        self.assertEqual(arb.classify_collision_window(200)["grade"], "tolerance_buffer")
        self.assertEqual(arb.classify_collision_window(300)["grade"], "malignant_collision")

    def test_floor_yield_rate(self):
        events = [{"overlap_ms": 300, "yielded": True},
                  {"overlap_ms": 400, "yielded": True},
                  {"overlap_ms": 100, "yielded": False}]  # 无害交叠不计入
        m = DualChannelTurnTakingArbiter.floor_yield_metrics(events)
        self.assertEqual(m["malignant_collisions"], 2)
        self.assertEqual(m["floor_yielding_rate"], 1.0)

    def test_no_malignant_collisions_rate_null(self):
        m = DualChannelTurnTakingArbiter.floor_yield_metrics([{"overlap_ms": 50, "yielded": False}])
        self.assertIsNone(m["floor_yielding_rate"])


class TestInferenceAbortEvaluator(unittest.TestCase):
    def test_pass_within_sla(self):
        results = inference_abort_assertions(abort_ms=12.0, discarded_tokens=2)
        self.assertTrue(all(a.passed for a in results))

    def test_fail_on_violation(self):
        results = inference_abort_assertions(abort_ms=25.0, discarded_tokens=5)
        self.assertFalse(all(a.passed for a in results))

    def test_null_evidence_rejected(self):
        results = inference_abort_assertions(None, None)
        self.assertFalse(all(a.passed for a in results))
        self.assertIn("不确定", results[0].message)

    def test_extract_from_ground_truth(self):
        abort_ms, tokens = extract_abort_metrics({"inference_abort_ms": 10.0, "discarded_tokens": 1})
        self.assertEqual((abort_ms, tokens), (10.0, 1))


class TestCorpusMatrix(unittest.TestCase):
    def test_24_case_matrix(self):
        cases = CorpusMatrix().build_cases()
        self.assertEqual(len(cases), 27)  # 20 方言×声线 + 6 谐音 + 1 基准
        traps = [c for c in cases if c.category == "homophone_trap"]
        self.assertEqual(len(traps), len(HOMOPHONE_TRAPS))
        self.assertTrue(all(not c.expected_wake for c in traps))

    def test_health_report_blocks_bad_cer(self):
        cases = CorpusMatrix().build_cases()
        asr = {c.case_id: 0.0 for c in cases}
        asr[cases[0].case_id] = 0.05  # 超门禁
        report = CorpusMatrix.corpus_health_report(cases, asr)
        self.assertFalse(report["healthy"])
        self.assertEqual(len(report["blocked"]), 1)

    def test_health_report_blocks_missing(self):
        cases = CorpusMatrix().build_cases()
        report = CorpusMatrix.corpus_health_report(cases, {})
        self.assertEqual(report["blocked"][0]["reason"], "missing_asr_result")


def _make_ctx_factory(base: str):
    async def factory(scenario_id: str) -> ScenarioContext:
        sid = f"{scenario_id}_{int(time.time() * 1e6)}"
        timeline = EventTimeline()
        engine = PlaybackEngine(sink=HannFadeoutSink(), timeline=timeline)
        audio_input = StreamingAudioInput()
        runtime = SessionRuntime(
            session_id=sid, audio_input=audio_input, playback_engine=engine,
            timeline=timeline, recorder=EvidenceRecorder(base_dir=base, session_id=sid),
        )
        await runtime.connect()
        await runtime.start()
        return ScenarioContext(runtime, audio_input, engine, timeline, runtime.recorder, {})
    return factory


class TestAdversarialScenarios(unittest.IsolatedAsyncioTestCase):
    async def test_disfluency_large_sample_simulated_pass(self):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
            return DisfluencyScenario(runtime, audio_input, playback_engine, timeline,
                                      trials=300, min_samples=200)

        registry.register(build)
        report = await SuiteRunner("disfluency", _make_ctx_factory(base)).run(registry)
        entry = report["scenarios"][0]
        self.assertEqual(entry["status"], TestStatus.SIMULATED_PASS, msg=str(entry["errors"]))

    async def test_disfluency_small_sample_fails_statistics_gate(self):
        scenario = DisfluencyScenario(None, None, None, None, trials=30, min_samples=200)
        results = scenario.assertions([], {"trials": 30, "premature_interruptions": 0,
                                           "preemptive_duckdown_hits": 30, "negation_leaks": 0})
        gate = next(a for a in results if a.name == "assert_premature_interruption_rate_sla")
        self.assertFalse(gate.passed)
        self.assertIn("无法统计判定", gate.message)

    async def test_homophone_zero_false_wakeup(self):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
            return HomophoneWakeupScenario(runtime, audio_input, playback_engine, timeline,
                                           repetitions_per_trap=40)  # 6×40=240 >= 200

        registry.register(build)
        report = await SuiteRunner("homophone", _make_ctx_factory(base)).run(registry)
        entry = report["scenarios"][0]
        self.assertEqual(entry["status"], TestStatus.SIMULATED_PASS, msg=str(entry["errors"]))
        self.assertEqual(entry["metrics"]["false_wakeups"], 0)
        self.assertEqual(entry["metrics"]["total_injections"], 240)


if __name__ == "__main__":
    unittest.main()
