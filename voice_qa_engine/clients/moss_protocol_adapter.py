#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - MOSS-Live 协议双向映射适配器
将目标服务官方 WebSocket 协议字段与内部归一化 RuntimeEvent 双向转换
"""

import json
import base64
import time
from typing import List, Dict, Any, Optional
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.models import SampleFormat, AudioFormat, AudioFrame
from voice_qa_engine.audio.format_adapter import PCMFormatAdapter


class MossProtocolAdapter:
    """抽象协议基类"""
    def build_handshake(self, config: Dict[str, Any]) -> dict:
        raise NotImplementedError

    def build_audio_message(self, frame: AudioFrame) -> Any:
        raise NotImplementedError

    def build_cancel_message(self, response_id: str) -> Any:
        raise NotImplementedError

    def parse_server_message(self, raw_message: Any, session_id: str, connection_epoch: int) -> List[RuntimeEvent]:
        raise NotImplementedError


class MossLiveProtocolAdapter(MossProtocolAdapter):
    """
    MOSS-Live / Realtime 全双工协议标准适配器
    支持 JSON/Base64 与二进制 PCM 混合帧解析
    """
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.audio_adapter = PCMFormatAdapter()

    def build_handshake(self, config: Dict[str, Any]) -> dict:
        return {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "voice": config.get("voice", "moss_live_default"),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_sample_rate": self.target_sample_rate,
                "output_audio_sample_rate": self.target_sample_rate,
                "instructions": config.get("instructions", "You are a helpful voice assistant.")
            }
        }

    def build_audio_message(self, frame: AudioFrame) -> dict:
        target_fmt = AudioFormat(
            sample_rate=self.target_sample_rate,
            channels=1,
            sample_format=SampleFormat.PCM_S16LE
        )
        raw_pcm16 = self.audio_adapter.encode(frame, target_fmt)
        b64_audio = base64.b64encode(raw_pcm16).decode("utf-8")
        return {
            "type": "input_audio_buffer.append",
            "audio": b64_audio,
            "sequence_id": frame.sequence_id
        }

    def build_cancel_message(self, response_id: str) -> dict:
        return {
            "type": "response.cancel",
            "response_id": response_id
        }

    def parse_server_message(self, raw_message: Any, session_id: str, connection_epoch: int) -> List[RuntimeEvent]:
        now_ns = time.monotonic_ns()
        events = []

        if isinstance(raw_message, bytes):
            # 纯二进制音频流
            return events

        try:
            data = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
        except Exception:
            return events

        msg_type = data.get("type", "")

        if msg_type == "session.created" or msg_type == "session.updated":
            events.append(RuntimeEvent(
                event_type=EventType.SESSION_READY,
                session_id=session_id,
                connection_epoch=connection_epoch,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type,
                field_origin={"event_type": "server", "session_id": "client"}
            ))

        elif msg_type == "response.created":
            resp_id = data.get("response", {}).get("id", data.get("response_id"))
            events.append(RuntimeEvent(
                event_type=EventType.RESPONSE_STARTED,
                session_id=session_id,
                connection_epoch=connection_epoch,
                response_id=resp_id,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type,
                field_origin={"response_id": "server"}
            ))

        elif msg_type in ("response.audio.delta", "response.output_audio.delta"):
            resp_id = data.get("response_id")
            delta_b64 = data.get("delta", "")
            seq_id = data.get("sequence_id", 0)
            events.append(RuntimeEvent(
                event_type=EventType.AUDIO_CHUNK,
                session_id=session_id,
                connection_epoch=connection_epoch,
                response_id=resp_id,
                sequence_id=seq_id,
                monotonic_ns=now_ns,
                source="server_event",
                payload={"delta_b64": delta_b64, "bytes_length": len(delta_b64)},
                raw_event_type=msg_type,
                field_origin={"response_id": "server", "audio": "server"}
            ))

        elif msg_type == "response.text.delta":
            resp_id = data.get("response_id")
            text = data.get("delta", "")
            events.append(RuntimeEvent(
                event_type=EventType.TEXT_DELTA,
                session_id=session_id,
                connection_epoch=connection_epoch,
                response_id=resp_id,
                monotonic_ns=now_ns,
                source="server_event",
                payload={"delta": text},
                raw_event_type=msg_type
            ))

        elif msg_type == "response.done" or msg_type == "response.completed":
            resp_id = data.get("response", {}).get("id", data.get("response_id"))
            events.append(RuntimeEvent(
                event_type=EventType.RESPONSE_COMPLETED,
                session_id=session_id,
                connection_epoch=connection_epoch,
                response_id=resp_id,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type
            ))

        elif msg_type == "response.cancelled":
            resp_id = data.get("response_id")
            events.append(RuntimeEvent(
                event_type=EventType.RESPONSE_CANCELLED,
                session_id=session_id,
                connection_epoch=connection_epoch,
                response_id=resp_id,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type
            ))

        elif msg_type == "input_audio_buffer.speech_started":
            events.append(RuntimeEvent(
                event_type=EventType.INPUT_SPEECH_STARTED,
                session_id=session_id,
                connection_epoch=connection_epoch,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type
            ))

        elif msg_type == "error":
            events.append(RuntimeEvent(
                event_type=EventType.ERROR,
                session_id=session_id,
                connection_epoch=connection_epoch,
                monotonic_ns=now_ns,
                source="server_event",
                payload=data,
                raw_event_type=msg_type
            ))

        return events
