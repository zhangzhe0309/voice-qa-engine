#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 声纹排他/多人讨论拒识域测试: OverlapSpeechSynthesizer / SpeakerExclusionScenario"""

import os
import time
import tempfile
import unittest

import numpy as np

from voice_qa_engine.audio.overlap_speech import OverlapSpeechSynthesizer
from voice_qa_engine.evaluators.assertion_models import TestStatus
from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext
from voice_qa_engine.scenarios.speaker_exclusion_scenario import SpeakerExclusionScenario
from voice_qa_engine.runtime.session_runtime import SessionRuntime
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.sink import HannFadeoutSink
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput


class TestOverlapSpeechSynthesizer(unittest.TestCase):
    def test_argument_waveform_properties(self):
        synth = OverlapSpeechSynthesizer(seed=7)
        pcm = synth.generate_argument(1000.0, n_speakers=2, snr_db=5.0)
        self.assertEqual(pcm.dtype, np.float32)
        self.assertEqual(pcm.size, 16000)
        self.assertLessEqual(float(np.max(np.abs(pcm))), 1.0)
        self.assertGreater(float(np.sqrt(np.mean(pcm ** 2))), 0.0)

    def test_snr_scaling(self):
        synth = OverlapSpeechSynthesizer(seed=7)
        loud = synth.generate_argument(500.0, snr_db=20.0)
        quiet = synth.generate_argument(500.0, snr_db=0.0)
        rms_loud = float(np.sqrt(np.mean(loud ** 2)))
        rms_quiet = float(np.sqrt(np.mean(quiet ** 2)))
        # SNR 越高噪声越弱：20dB 背景电平应远低于 0dB
        self.assertLess(rms_loud, rms_quiet / 5)

    def test_requires_two_speakers(self):
        synth = OverlapSpeechSynthesizer()
        with self.assertRaises(ValueError):
            synth.generate_argument(500.0, n_speakers=1)

    def test_inhalation_waveform(self):
        synth = OverlapSpeechSynthesizer()
        pcm = synth.generate_inhalation(80.0)
        self.assertEqual(pcm.size, 1280)
        self.assertLessEqual(float(np.max(np.abs(pcm))), 1.0)


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


class TestSpeakerExclusionScenario(unittest.IsolatedAsyncioTestCase):
    async def test_background_rejection_simulated_inconclusive_no_wakeup(self):
        base = tempfile.mkdtemp()
        registry = ScenarioRegistry()

        def build(runtime, audio_input, playback_engine, timeline, evaluator=None):
            return SpeakerExclusionScenario(
                runtime, audio_input, playback_engine, timeline,
                background_duration_ms=300.0, n_speakers=2, snr_db=5.0,
            )

        registry.register(build)
        report = await SuiteRunner("speaker_exclusion", _make_ctx_factory(base)).run(registry)
        entry = report["scenarios"][0]
        # 无真实唤醒证据 → INCONCLUSIVE（一票否决不虚标），但核心拒识断言必须全过
        self.assertEqual(entry["status"], TestStatus.INCONCLUSIVE)
        passed_names = [a["name"] for a in entry["assertions"] if a["passed"]]
        self.assertIn("assert_zero_downlink_during_chatter", passed_names)
        self.assertIn("assert_zero_false_bargein", passed_names)
        self.assertIn("assert_directed_wakeup_evidence_present",
                      [a["name"] for a in entry["assertions"] if not a["passed"]])
        self.assertTrue(entry["evidence_valid"])

    async def test_downlink_leak_fails_core_assertion(self):
        """争吵窗口内若出现下行分片 → 核心断言 FAIL"""
        scenario = SpeakerExclusionScenario(None, None, None, None)
        metrics = {
            "background_window_ms": 7000.0, "n_speakers": 2, "snr_db": 5.0,
            "audio_chunks_in_window": 3, "downlink_bytes_in_window": 960,
            "wakeup_observed": True, "wakeup_latency_ms": 500.0, "recall_hit": True,
        }
        results = scenario.assertions([], metrics)
        by_name = {a.name: a for a in results}
        self.assertFalse(by_name["assert_zero_downlink_during_chatter"].passed)
        self.assertFalse(by_name["assert_zero_false_bargein"].passed)
        status = SpeakerExclusionScenario._decide_status(results, metrics)
        self.assertEqual(status, TestStatus.FAIL)

    async def test_wakeup_latency_sla(self):
        scenario = SpeakerExclusionScenario(None, None, None, None, wakeup_latency_sla_ms=600.0)
        good = {"background_window_ms": 1000.0, "audio_chunks_in_window": 0,
                "downlink_bytes_in_window": 0, "wakeup_observed": True,
                "wakeup_latency_ms": 550.0, "recall_hit": True}
        bad = dict(good, wakeup_latency_ms=900.0)
        self.assertTrue(all(a.passed for a in scenario.assertions([], good)))
        bad_results = scenario.assertions([], bad)
        self.assertFalse(next(a for a in bad_results if a.name == "assert_directed_wakeup_latency").passed)


if __name__ == "__main__":
    unittest.main()
