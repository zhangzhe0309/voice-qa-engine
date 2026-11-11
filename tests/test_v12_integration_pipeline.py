#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段集成测试
验证 SessionRuntime 贯通 WavAudioInput -> Timeline -> PlaybackEngine -> EvidenceRecorder
"""

import unittest
import asyncio
import io
import wave
import tempfile
import shutil
import os
import numpy as np

from voice_qa_engine.runtime.session_runtime import SessionRuntime
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.playback.engine import PlaybackEngine
from voice_qa_engine.playback.sink import MockPlaybackSink
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.audio.input.wav_input import WavAudioInput
from voice_qa_engine.evaluators.assertion_models import TestStatus


class TestIntegrationPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="v12_integration_")
        self.session_id = "session_integ_001"

    async def asyncTearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_full_pipeline_flow_and_artifact_generation(self):
        # 1. 构造一个 60ms 的内存测试 WAV (3 个 20ms 帧)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((np.zeros(960, dtype=np.int16)).tobytes())
        buf.seek(0)

        audio_input = WavAudioInput(file_obj=buf, frame_duration_ms=20.0)
        timeline = EventTimeline()
        sink = MockPlaybackSink()
        engine = PlaybackEngine(sink=sink, timeline=timeline)
        recorder = EvidenceRecorder(base_dir=self.test_dir, session_id=self.session_id)
        await recorder.initialize()

        runtime = SessionRuntime(
            session_id=self.session_id,
            audio_input=audio_input,
            playback_engine=engine,
            timeline=timeline,
            recorder=recorder
        )

        # 2. 运行流程
        await runtime.connect()
        await runtime.start()
        await runtime.feed_audio()

        # 3. 产生回答代次并打断
        gen = await engine.begin_response("resp_abc")
        await engine.interrupt(reason="test_barge_in")

        # 4. 关闭并生成最终证据产物
        await runtime.close()
        await recorder.finalize(
            metrics={"test_metric": 100},
            session_meta={"status": TestStatus.SIMULATED_PASS}
        )

        # 5. 校验物理落盘
        sess_dir = os.path.join(self.test_dir, self.session_id)
        self.assertTrue(os.path.exists(os.path.join(sess_dir, "input_audio.wav")))
        self.assertTrue(os.path.exists(os.path.join(sess_dir, "events.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(sess_dir, "manifest.json")))

        # 校验事件数量
        events = timeline.snapshot()
        event_types = [e.event_type for e in events]
        self.assertIn("session.ready", event_types)
        self.assertIn("input.audio.sent", event_types)
        self.assertIn("input.audio.end", event_types)
        self.assertIn("playback.interrupt.requested", event_types)


if __name__ == '__main__':
    unittest.main()
