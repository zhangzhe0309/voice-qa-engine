#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 阶段 1 TDD 单元测试
测试音频数据模型、PCM格式适配、重采样与WAV分帧输入
"""

import unittest
import numpy as np
import time
import io
import wave
import asyncio

from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.audio.format_adapter import PCMFormatAdapter
from voice_qa_engine.audio.resampler import AudioResampler
from voice_qa_engine.audio.input.wav_input import WavAudioInput


class TestAudioModels(unittest.TestCase):
    def test_audio_format_validation(self):
        # 正常创建
        fmt = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_S16LE)
        self.assertEqual(fmt.sample_rate, 16000)
        self.assertEqual(fmt.channels, 1)

        # 异常校验：采样率 <= 0
        with self.assertRaises(ValueError):
            AudioFormat(sample_rate=0, channels=1, sample_format=SampleFormat.PCM_S16LE)

        # 异常校验：声道数 <= 0
        with self.assertRaises(ValueError):
            AudioFormat(sample_rate=16000, channels=-1, sample_format=SampleFormat.PCM_S16LE)

    def test_audio_frame_duration_calculation(self):
        fmt = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_F32LE)
        samples = np.zeros(320, dtype=np.float32)  # 320 samples at 16kHz = 20ms
        now_ns = time.monotonic_ns()
        frame = AudioFrame(
            samples=samples,
            format=fmt,
            timestamp_ns=now_ns,
            sequence_id=1,
            duration_ms=20.0
        )
        self.assertAlmostEqual(frame.duration_ms, 20.0, places=2)
        self.assertEqual(frame.samples.shape[0], 320)


class TestPCMFormatAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = PCMFormatAdapter()

    def test_encode_decode_roundtrip_s16le(self):
        fmt_s16 = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_S16LE)
        fmt_f32 = AudioFormat(sample_rate=16000, channels=1, sample_format=SampleFormat.PCM_F32LE)

        # 创建一段 16-bit 线性测试音频
        original_int16 = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        raw_bytes = original_int16.tobytes()

        # 解码为内部 float32 Frame
        frame = self.adapter.decode(raw_bytes, fmt_s16)
        self.assertEqual(frame.format.sample_format, SampleFormat.PCM_F32LE)
        self.assertEqual(frame.samples.dtype, np.float32)

        # 验证数值范围在 [-1.0, 1.0] 归一化区间
        self.assertAlmostEqual(frame.samples[0], -1.0, delta=0.001)
        self.assertAlmostEqual(frame.samples[2], 0.0, delta=0.001)
        self.assertAlmostEqual(frame.samples[4], 1.0, delta=0.001)

        # 重新编码回 s16le 字节
        encoded_bytes = self.adapter.encode(frame, fmt_s16)
        restored_int16 = np.frombuffer(encoded_bytes, dtype=np.int16)
        np.testing.assert_array_almost_equal(original_int16, restored_int16, decimal=0)

    def test_resample_16k_to_24k(self):
        # 16kHz -> 24kHz
        resampler = AudioResampler()
        samples_16k = np.sin(np.linspace(0, 2 * np.pi * 440, 1600, endpoint=False, dtype=np.float32))  # 100ms
        samples_24k = resampler.resample(samples_16k, orig_sr=16000, target_sr=24000)

        # 100ms 在 24kHz 下应该是 2400 个点
        self.assertEqual(samples_24k.shape[0], 2400)
        self.assertEqual(samples_24k.dtype, np.float32)


class TestWavAudioInput(unittest.IsolatedAsyncioTestCase):
    async def test_wav_streaming_frames(self):
        # 动态生成一个 100ms 16kHz mono 的内存 WAV 文件 (5 个 20ms 分帧)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            data_100ms = (np.sin(np.linspace(0, 10, 1600)) * 10000).astype(np.int16)
            wf.writeframes(data_100ms.tobytes())
        wav_buffer.seek(0)

        # 通过 WavAudioInput 读取
        audio_input = WavAudioInput(file_obj=wav_buffer, frame_duration_ms=20.0)
        await audio_input.open()

        frames = []
        while True:
            frame = await audio_input.read()
            if frame is None:
                break
            frames.append(frame)

        await audio_input.close()

        # 100ms / 20ms = 5 帧
        self.assertEqual(len(frames), 5)
        # 序列号递增
        for idx, f in enumerate(frames):
            self.assertEqual(f.sequence_id, idx)
            self.assertEqual(f.samples.shape[0], 320)  # 20ms * 16000 / 1000 = 320
            self.assertAlmostEqual(f.duration_ms, 20.0, places=1)


if __name__ == '__main__':
    unittest.main()
