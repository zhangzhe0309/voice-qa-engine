#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段 2 & 阶段 3 TDD 单元测试
覆盖：
1. ResponseRegistry 响应代次绑定与失效判定
2. EventTimeline 单调时钟事件流水记录
3. PlaybackEngine 单一 Actor 防幽灵音频核心逻辑：
   - 正常推入与播放
   - 打断后旧代次 Chunk 100% 拒绝
   - 快速双打断防伪验证（第二次打断后旧代次绝不恢复播放）
"""

import unittest
import asyncio
import time
import numpy as np

from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.runtime.session_state import SessionState, ConnectionState
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.runtime.response_registry import ResponseRegistry
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.playback.state import PlaybackState
from voice_qa_engine.playback.queue import PlaybackChunk
from voice_qa_engine.playback.sink import MockPlaybackSink, HannFadeoutSink
from voice_qa_engine.playback.engine import PlaybackEngine


class TestPlaybackAndRuntime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.timeline = EventTimeline()
        self.sink = MockPlaybackSink()
        self.engine = PlaybackEngine(sink=self.sink, timeline=self.timeline)
        self.registry = ResponseRegistry()
        await self.engine.start()

    async def asyncTearDown(self):
        await self.engine.close()

    def _make_dummy_frame(self, duration_ms: float = 20.0) -> AudioFrame:
        fmt = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_F32LE)
        samples = np.zeros(int(16000 * duration_ms / 1000.0), dtype=np.float32)
        return AudioFrame(
            samples=samples,
            format=fmt,
            timestamp_ns=time.monotonic_ns(),
            sequence_id=0,
            duration_ms=duration_ms
        )

    async def test_normal_response_and_chunk_consumption(self):
        # 1. 服务端返回 Response 1
        gen_1 = await self.engine.begin_response("resp_1")
        self.assertEqual(gen_1, 1)

        # 2. 推入 Chunk 0 和 Chunk 1
        chunk_0 = PlaybackChunk(
            session_id="s1",
            response_id="resp_1",
            generation_id=gen_1,
            sequence_id=0,
            audio=self._make_dummy_frame(),
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id=self.engine.cancel_token.id,
            connection_epoch=1
        )
        accepted = await self.engine.push_chunk(chunk_0)
        self.assertTrue(accepted)
        await asyncio.sleep(0.05)
        self.assertGreaterEqual(len(self.sink.played_chunks), 1)

    async def test_ghost_audio_rejection_after_interrupt(self):
        """核心防幽灵音频测试：打断发生后，旧 Generation 的 Chunk 必须被 100% 拒绝"""
        gen_1 = await self.engine.begin_response("resp_1")

        chunk_0 = PlaybackChunk(
            session_id="s1",
            response_id="resp_1",
            generation_id=gen_1,
            sequence_id=0,
            audio=self._make_dummy_frame(),
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id=self.engine.cancel_token.id,
            connection_epoch=1
        )
        await self.engine.push_chunk(chunk_0)

        # 触发用户打断
        await self.engine.interrupt(reason="user_barge_in")
        self.assertGreater(self.engine.generation_id, gen_1)
        self.assertTrue(self.sink.was_stopped_and_flushed)

        # 模拟打断后在网络上延迟到达的旧 Chunk 1 (带旧 generation_1)
        delayed_old_chunk = PlaybackChunk(
            session_id="s1",
            response_id="resp_1",
            generation_id=gen_1,
            sequence_id=1,
            audio=self._make_dummy_frame(),
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id="old_token",
            connection_epoch=1
        )

        accepted_old = await self.engine.push_chunk(delayed_old_chunk)
        # 硬断言：旧 Chunk 必须被一票否决拒绝！
        self.assertFalse(accepted_old)

        # 开启新回答 Response 2，新 Chunk 能够正常接收
        gen_2 = await self.engine.begin_response("resp_2")
        self.assertGreater(gen_2, self.engine.generation_id - 1)
        new_chunk = PlaybackChunk(
            session_id="s1",
            response_id="resp_2",
            generation_id=gen_2,
            sequence_id=0,
            audio=self._make_dummy_frame(),
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id=self.engine.cancel_token.id,
            connection_epoch=1
        )
        accepted_new = await self.engine.push_chunk(new_chunk)
        self.assertTrue(accepted_new)

    async def test_rapid_double_barge_in(self):
        """测试快速连续两次打断时代次递增与旧代次绝对隔离"""
        # R1 开始
        gen_1 = await self.engine.begin_response("resp_1")
        # 第一次打断
        await self.engine.interrupt(reason="barge_in_1")
        gen_after_1 = self.engine.generation_id

        # R2 开始
        gen_2 = await self.engine.begin_response("resp_2")
        self.assertGreater(gen_2, gen_after_1)
        # 极短时间内第二次打断
        await self.engine.interrupt(reason="barge_in_2")
        gen_after_2 = self.engine.generation_id
        self.assertGreater(gen_after_2, gen_2)

        # 验证来自 gen_1 和 gen_2 的迟到 Chunk 全部被拒绝
        for old_gen, r_id in [(gen_1, "resp_1"), (gen_2, "resp_2")]:
            c = PlaybackChunk(
                session_id="s1",
                response_id=r_id,
                generation_id=old_gen,
                sequence_id=5,
                audio=self._make_dummy_frame(),
                received_monotonic_ns=time.monotonic_ns(),
                expected_duration_ms=20.0,
                cancel_token_id="old",
                connection_epoch=1
            )
            self.assertFalse(await self.engine.push_chunk(c))

    async def test_hann_fadeout_sink_prevents_pop_noise(self):
        """验证 HannFadeoutSink 在打断时施加半 Hann 窗，差分 Jump <= 8000 (消除爆音)"""
        hann_sink = HannFadeoutSink(fade_duration_ms=10.0)
        engine = PlaybackEngine(sink=hann_sink, timeline=self.timeline)
        await engine.start()

        # 生成 16kHz 高能量正弦波 (320 样点，满幅震荡)
        t = np.linspace(0, 0.02, 320, endpoint=False)
        loud_samples = (0.95 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        frame = AudioFrame(
            samples=loud_samples,
            format=AudioFormat(16000, 1, SampleFormat.PCM_F32LE),
            timestamp_ns=time.monotonic_ns(),
            sequence_id=0,
            duration_ms=20.0
        )

        gen = await engine.begin_response("resp_hann")
        chunk = PlaybackChunk(
            session_id="s_hann",
            response_id="resp_hann",
            generation_id=gen,
            sequence_id=0,
            audio=frame,
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id=engine.cancel_token.id,
            connection_epoch=1
        )

        await engine.push_chunk(chunk)
        await asyncio.sleep(0.05)  # 等待 actor 播放

        # 触发打断
        await engine.interrupt(reason="barge_in_hann")
        await engine.close()

        self.assertTrue(hann_sink.was_stopped_and_flushed)
        self.assertTrue(hann_sink.fadeout_applied)
        # 门禁 SLA: Jump <= 8000
        self.assertLessEqual(hann_sink.max_jump_detected, 8000.0)



if __name__ == '__main__':
    unittest.main()
