#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段 4 & 阶段 5 TDD 单元测试
测试场景执行、断言六态判定、五大必备物理证据生成与 SHA-256 完整性
"""

import unittest
import asyncio
import os
import json
import shutil
import tempfile
import time
import numpy as np

from voice_qa_engine.evaluators.assertion_models import TestStatus, AssertionResult, AssertionSeverity
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.scenarios.barge_in import BargeInScenario
from voice_qa_engine.scenarios.rapid_barge_in import RapidDoubleBargeInScenario


class TestScenariosAndEvidence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="voice_qa_test_")
        self.session_id = "session_test_20261111"
        self.recorder = EvidenceRecorder(base_dir=self.test_dir, session_id=self.session_id)
        await self.recorder.initialize()

    async def asyncTearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_evidence_recorder_five_artifacts_and_manifest(self):
        # 1. 写入 input frame
        fmt = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_F32LE)
        frame = AudioFrame(
            samples=np.zeros(320, dtype=np.float32),
            format=fmt,
            timestamp_ns=1000,
            sequence_id=0,
            duration_ms=20.0
        )
        await self.recorder.write_input_audio(frame)
        await self.recorder.write_output_audio(frame)

        # 2. 写入事件
        evt = RuntimeEvent(
            event_type=EventType.SESSION_READY,
            session_id=self.session_id,
            monotonic_ns=1000
        )
        await self.recorder.write_event(evt)

        # 3. 写入 metrics 与 session 元数据
        metrics = {"first_audio_chunk_ms": 320.5, "local_stop_latency_ms": 110.2}
        session_meta = {"scenario": "barge_in", "status": TestStatus.SIMULATED_PASS}
        await self.recorder.finalize(metrics=metrics, session_meta=session_meta)

        # 4. 验证五个必备产物是否真实落盘
        session_folder = os.path.join(self.test_dir, self.session_id)
        self.assertTrue(os.path.exists(os.path.join(session_folder, "input_audio.wav")))
        self.assertTrue(os.path.exists(os.path.join(session_folder, "output_audio.wav")))
        self.assertTrue(os.path.exists(os.path.join(session_folder, "events.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(session_folder, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(session_folder, "session.json")))

        # 5. 验证 manifest.json SHA-256 哈希校验
        manifest_path = os.path.join(session_folder, "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertIn("input_audio.wav", manifest["artifacts"])
        self.assertIn("sha256", manifest["artifacts"]["input_audio.wav"])
        self.assertGreater(len(manifest["artifacts"]["input_audio.wav"]["sha256"]), 20)


class TestScenariosAssertions(unittest.TestCase):
    def test_barge_in_assertion_success(self):
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        # 模拟打断事件链
        evt1 = RuntimeEvent(
            event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
            session_id="s1",
            monotonic_ns=t0 + 10,
            payload={"old_generation_id": 1, "new_generation_id": 2, "reason": "user_speech"}
        )
        evt2 = RuntimeEvent(
            event_type=EventType.PLAYBACK_QUEUE_FLUSHED,
            session_id="s1",
            monotonic_ns=t0 + 15,
            payload={"flushed_count": 3}
        )
        evt3 = RuntimeEvent(
            event_type=EventType.PLAYBACK_SINK_STOPPED,
            session_id="s1",
            monotonic_ns=t0 + 20,
            payload={}
        )
        timeline.events = [evt1, evt2, evt3]

        scenario = BargeInScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        self.assertTrue(all(a.passed for a in assertions))

    def test_rapid_barge_in_catches_stale_resumption(self):
        """测试快速双打断门禁能精准拦截'旧代次意外恢复播放'的恶性缺陷"""
        timeline = EventTimeline()
        t0 = time.monotonic_ns()
        evt_int1 = RuntimeEvent(
            event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
            session_id="s1",
            monotonic_ns=t0 + 10,
            payload={"old_generation_id": 1, "new_generation_id": 2}
        )
        evt_int2 = RuntimeEvent(
            event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
            session_id="s1",
            monotonic_ns=t0 + 50,
            payload={"old_generation_id": 2, "new_generation_id": 3}
        )
        # 恶性缺陷：在第二次打后，旧代次 2 的 Chunk 居然又触发了播放！
        evt_stale_play = RuntimeEvent(
            event_type=EventType.PLAYBACK_STARTED,
            session_id="s1",
            generation_id=2,  # <= second_old
            monotonic_ns=t0 + 60
        )
        timeline.events = [evt_int1, evt_int2, evt_stale_play]

        scenario = RapidDoubleBargeInScenario(runtime=None, audio_input=None, playback_engine=None, timeline=timeline)
        assertions = scenario.assertions(timeline.snapshot())
        failed = [a for a in assertions if not a.passed]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].name, "assert_stale_generation_never_resumes")


if __name__ == '__main__':
    unittest.main()
