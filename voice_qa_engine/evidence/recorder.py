#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 物理证据记录器 (Evidence Recorder)
落盘五个核心产物与 Manifest SHA-256，保障证据链完整独立可溯源
"""

import os
import json
import wave
import numpy as np
from typing import Dict, Any, Optional

from voice_qa_engine.audio.models import AudioFrame
from voice_qa_engine.clients.protocol_models import RuntimeEvent
from voice_qa_engine.evidence.manifest import generate_manifest


class EvidenceRecorder:
    """
    streaming=False: 内存缓冲模式（默认，兼容既有单测用法，finalize 时统一落盘）
    streaming=True : 流式落盘模式（长时 soak 会话必需，write_* 即时追加写 WAV，杜绝内存累积）
    """

    def __init__(self, base_dir: str, session_id: str, streaming: bool = False):
        self.base_dir = base_dir
        self.session_id = session_id
        self.session_dir = os.path.join(base_dir, session_id)
        self.streaming = streaming
        self.input_frames: list[AudioFrame] = []
        self.output_frames: list[AudioFrame] = []
        self._input_writer = None
        self._output_writer = None
        self._input_wav_path = os.path.join(self.session_dir, "input_audio.wav")
        self._output_wav_path = os.path.join(self.session_dir, "output_audio.wav")
        self.events_file_path = os.path.join(self.session_dir, "events.jsonl")

    async def initialize(self):
        os.makedirs(self.session_dir, exist_ok=True)
        # 清空或初始化 events 文件
        with open(self.events_file_path, "w", encoding="utf-8"):
            pass
        if self.streaming:
            self._input_writer = self._open_stream_writer(self._input_wav_path)
            self._output_writer = self._open_stream_writer(self._output_wav_path)

    @staticmethod
    def _open_stream_writer(filepath: str):
        wf = wave.open(filepath, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        return wf

    def _stream_write(self, writer, frame: AudioFrame):
        clipped = np.clip(frame.samples, -1.0, 1.0)
        writer.writeframes((clipped * 32767.0).astype(np.int16).tobytes())

    async def write_input_audio(self, frame: AudioFrame):
        if self.streaming:
            self._stream_write(self._input_writer, frame)
        else:
            self.input_frames.append(frame)

    async def write_output_audio(self, frame: AudioFrame):
        if self.streaming:
            self._stream_write(self._output_writer, frame)
        else:
            self.output_frames.append(frame)

    async def write_event(self, event: RuntimeEvent):
        # 序列化单条事件为 JSON Line
        event_dict = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "session_id": event.session_id,
            "connection_epoch": event.connection_epoch,
            "response_id": event.response_id,
            "generation_id": event.generation_id,
            "sequence_id": event.sequence_id,
            "monotonic_ns": event.monotonic_ns,
            "wall_time_utc": event.wall_time_utc,
            "source": event.source,
            "payload": event.payload,
            "raw_event_type": event.raw_event_type,
            "field_origin": event.field_origin
        }
        with open(self.events_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")

    def _save_wav(self, frames: list[AudioFrame], filename: str):
        filepath = os.path.join(self.session_dir, filename)
        if not frames:
            # 写入 0 样本的空标准 WAV 头
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"")
            return

        sample_rate = frames[0].format.sample_rate
        channels = frames[0].format.channels

        # 合并所有样本并转换为 16-bit PCM
        all_samples = np.concatenate([f.samples for f in frames])
        clipped = np.clip(all_samples, -1.0, 1.0)
        pcm_bytes = (clipped * 32767.0).astype(np.int16).tobytes()

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

    async def finalize(self, metrics: Dict[str, Any], session_meta: Dict[str, Any]):
        if self.streaming:
            # 流式模式：关闭 WAV 写头完成文件收尾（未写入任何帧时即为 0 样本空 WAV）
            if self._input_writer is not None:
                self._input_writer.close()
                self._input_writer = None
            if self._output_writer is not None:
                self._output_writer.close()
                self._output_writer = None
        else:
            # 1. 物理导出 input_audio.wav 与 output_audio.wav
            self._save_wav(self.input_frames, "input_audio.wav")
            self._save_wav(self.output_frames, "output_audio.wav")

        # 2. 导出 metrics.json
        metrics_path = os.path.join(self.session_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 3. 导出 session.json
        session_path = os.path.join(self.session_dir, "session.json")
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_meta, f, indent=2, ensure_ascii=False)

        # 4. 生成 manifest.json
        manifest = generate_manifest(
            self.session_dir,
            ["input_audio.wav", "output_audio.wav", "events.jsonl", "metrics.json", "session.json"]
        )
        manifest_path = os.path.join(self.session_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
