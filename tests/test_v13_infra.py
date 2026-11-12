#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 基础设施测试: StreamingAudioInput / EvidenceRecorder 流式模式 / SuiteRunner"""

import os
import tempfile
import time
import unittest
import numpy as np

from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.evaluators.assertion_models import AssertionResult, TestStatus
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext


class TestStreamingAudioInput(unittest.IsolatedAsyncioTestCase):
    async def test_push_segment_frame_splitting(self):
        src = StreamingAudioInput(sample_rate=16000, frame_duration_ms=20.0)
        await src.open()
        # 50ms 片段 = 800 样本 → 2 整帧 + 1 残帧
        src.push_segment(np.ones(800, dtype=np.float32))
        frames = []
        while len(frames) < 2:
            frames.append(await src.read())
        self.assertEqual(frames[0].samples.size, 320)
        self.assertTrue(np.allclose(frames[0].samples, 1.0))
        # 残帧 160 样本在下一帧补齐: 读第三帧应为 320 样本(残帧+静音? 实现为残帧不足时与后续合并或静音混合)
        third = await src.read()
        self.assertEqual(third.samples.size, 320)
        await src.close()
        with self.assertRaises(RuntimeError):
            await src.read()

    async def test_silence_after_drain_and_close(self):
        src = StreamingAudioInput()
        await src.open()
        frame = await src.read()
        self.assertEqual(frame.samples.size, 320)
        self.assertTrue(np.all(frame.samples == 0.0))
        await src.close()
        with self.assertRaises(RuntimeError):
            await src.read()

    async def test_push_before_open_raises(self):
        src = StreamingAudioInput()
        with self.assertRaises(RuntimeError):
            src.push_segment(np.zeros(320, dtype=np.float32))


class TestStreamingRecorder(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_mode_produces_valid_wav(self):
        from voice_qa_engine.audio.models import AudioFormat, SampleFormat, AudioFrame
        base = tempfile.mkdtemp()
        rec = EvidenceRecorder(base_dir=base, session_id="sess_stream", streaming=True)
        await rec.initialize()
        fmt = AudioFormat(16000, 1, SampleFormat.PCM_F32LE)
        frame = AudioFrame(np.ones(320, dtype=np.float32) * 0.5, fmt, time.monotonic_ns(), 0, 20.0)
        await rec.write_input_audio(frame)
        await rec.write_output_audio(frame)
        await rec.finalize(metrics={"k": 1}, session_meta={"sid": "sess_stream"})

        import wave
        for name in ("input_audio.wav", "output_audio.wav"):
            path = os.path.join(base, "sess_stream", name)
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getnframes(), 320)
        manifest_path = os.path.join(base, "sess_stream", "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

    async def test_streaming_mode_empty_writes_empty_wav(self):
        base = tempfile.mkdtemp()
        rec = EvidenceRecorder(base_dir=base, session_id="sess_empty", streaming=True)
        await rec.initialize()
        await rec.finalize(metrics={}, session_meta={})
        import wave
        path = os.path.join(base, "sess_empty", "output_audio.wav")
        with wave.open(path, "rb") as wf:
            self.assertEqual(wf.getnframes(), 0)


class _DummyScenario(BaseVoiceScenario):
    async def execute(self) -> ScenarioResult:
        events = self.timeline.snapshot()
        assertions = [AssertionResult(name="always_ok", passed=True)]
        return ScenarioResult(
            scenario_id="dummy",
            status=TestStatus.PASS if all(a.passed for a in assertions) else TestStatus.FAIL,
            started_ns=time.monotonic_ns(),
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
        )

    def assertions(self, events):
        return [AssertionResult(name="always_ok", passed=True)]


class _FailScenario(_DummyScenario):
    async def execute(self) -> ScenarioResult:
        result = await super().execute()
        result.scenario_id = "fail_case"
        result.assertions = [AssertionResult(name="must_fail", passed=False)]
        result.status = TestStatus.FAIL
        return result


class TestSuiteRunner(unittest.IsolatedAsyncioTestCase):
    async def test_runner_aggregates_and_validates_evidence(self):
        from voice_qa_engine.runtime.session_runtime import SessionRuntime
        from voice_qa_engine.runtime.event_timeline import EventTimeline
        from voice_qa_engine.playback.engine import PlaybackEngine
        from voice_qa_engine.playback.sink import HannFadeoutSink
        from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput as SAI

        base = tempfile.mkdtemp()
        counter = {"n": 0}

        async def factory(scenario_id: str) -> ScenarioContext:
            counter["n"] += 1
            sid = f"{scenario_id}_{counter['n']}"
            timeline = EventTimeline()
            engine = PlaybackEngine(sink=HannFadeoutSink(), timeline=timeline)
            audio_input = SAI()
            runtime = SessionRuntime(
                session_id=sid, audio_input=audio_input, playback_engine=engine,
                timeline=timeline, recorder=EvidenceRecorder(base_dir=base, session_id=sid),
            )
            return ScenarioContext(runtime, audio_input, engine, timeline, runtime.recorder, {})

        registry = ScenarioRegistry()
        registry.register(_DummyScenario)
        registry.register(_FailScenario)

        runner = SuiteRunner("v13_smoke", factory)
        report = await runner.run(registry)

        self.assertEqual(report["overall_status"], TestStatus.FAIL)
        self.assertEqual(len(report["scenarios"]), 2)
        self.assertTrue(report["scenarios"][0]["evidence_valid"])
        self.assertEqual(report["scenarios"][1]["status"], TestStatus.FAIL)
        # 每场景证据目录均已生成（每场景一个会话目录）
        session_dirs = [d for d in os.listdir(base) if d.startswith(("dummy", "fail"))]
        self.assertEqual(len(session_dirs), 2)


if __name__ == "__main__":
    unittest.main()
