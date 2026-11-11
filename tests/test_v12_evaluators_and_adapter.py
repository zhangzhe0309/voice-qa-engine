#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段 6 单元测试
覆盖：
1. MossLiveProtocolAdapter 握手、音频帧与打断取消信令映射
2. LatencyEvaluator 单调时钟物理时延计算
3. EvidenceValidator 产物哈希比对与防伪拦截
"""

import unittest
import os
import json
import tempfile
import shutil
import time

from voice_qa_engine.clients.moss_protocol_adapter import MossLiveProtocolAdapter
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.evaluators.latency import LatencyEvaluator
from voice_qa_engine.evaluators.evidence_validator import EvidenceValidator
from voice_qa_engine.evidence.recorder import EvidenceRecorder
import numpy as np


class TestEvaluatorsAndAdapter(unittest.IsolatedAsyncioTestCase):
    def test_protocol_adapter_handshake_and_cancel(self):
        adapter = MossLiveProtocolAdapter(target_sample_rate=16000)
        handshake = adapter.build_handshake({"voice": "moss_female_1"})
        self.assertEqual(handshake["type"], "session.update")
        self.assertEqual(handshake["session"]["input_audio_sample_rate"], 16000)

        cancel_msg = adapter.build_cancel_message("resp_999")
        self.assertEqual(cancel_msg["type"], "response.cancel")
        self.assertEqual(cancel_msg["response_id"], "resp_999")

    def test_latency_evaluator_ttft_and_local_stop(self):
        t0 = time.monotonic_ns()
        events = [
            RuntimeEvent(
                event_type=EventType.INPUT_AUDIO_END,
                session_id="s1",
                monotonic_ns=t0
            ),
            RuntimeEvent(
                event_type=EventType.TEXT_DELTA,
                session_id="s1",
                monotonic_ns=t0 + 200_000_000  # +200ms
            ),
            RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK,
                session_id="s1",
                monotonic_ns=t0 + 350_000_000  # +350ms
            ),
            RuntimeEvent(
                event_type=EventType.PLAYBACK_INTERRUPT_REQUESTED,
                session_id="s1",
                monotonic_ns=t0 + 500_000_000
            ),
            RuntimeEvent(
                event_type=EventType.PLAYBACK_QUEUE_FLUSHED,
                session_id="s1",
                monotonic_ns=t0 + 505_000_000  # +5ms
            ),
            RuntimeEvent(
                event_type=EventType.PLAYBACK_SINK_STOPPED,
                session_id="s1",
                monotonic_ns=t0 + 515_000_000  # +15ms
            ),
            RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK_REJECTED,
                session_id="s1",
                monotonic_ns=t0 + 520_000_000
            )
        ]

        metrics = LatencyEvaluator.calculate_latencies(events)
        self.assertAlmostEqual(metrics["ttft_ms"], 200.0, places=1)
        self.assertAlmostEqual(metrics["first_audio_chunk_ms"], 350.0, places=1)
        self.assertAlmostEqual(metrics["queue_flush_latency_ms"], 5.0, places=1)
        self.assertAlmostEqual(metrics["local_stop_latency_ms"], 15.0, places=1)
        self.assertEqual(metrics["post_interrupt_old_chunk_rejected_count"], 1)
        self.assertEqual(metrics["old_chunk_rejection_rate"], 1.0)

    async def test_evidence_validator_detects_tampering(self):
        test_dir = tempfile.mkdtemp(prefix="val_test_")
        session_id = "sess_val_01"
        try:
            recorder = EvidenceRecorder(base_dir=test_dir, session_id=session_id)
            await recorder.initialize()
            fmt = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_F32LE)
            frame = AudioFrame(
                samples=np.zeros(160, dtype=np.float32),
                format=fmt,
                timestamp_ns=100,
                sequence_id=0,
                duration_ms=10.0
            )
            await recorder.write_input_audio(frame)
            await recorder.write_output_audio(frame)
            await recorder.write_event(RuntimeEvent(event_type=EventType.SESSION_READY, session_id=session_id))
            await recorder.finalize(metrics={}, session_meta={})

            sess_dir = os.path.join(test_dir, session_id)
            # 正常校验通过
            valid, msg, _ = EvidenceValidator.validate_session_dir(sess_dir)
            self.assertTrue(valid)

            # 恶意篡改文件内容
            with open(os.path.join(sess_dir, "metrics.json"), "a") as f:
                f.write(" ")

            # 篡改后哈希比对必须被一票否决
            tampered_valid, tampered_msg, _ = EvidenceValidator.validate_session_dir(sess_dir)
            self.assertFalse(tampered_valid)
            self.assertIn("文件哈希校验失败", tampered_msg)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
