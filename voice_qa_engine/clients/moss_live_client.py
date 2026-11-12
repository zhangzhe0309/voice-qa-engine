"""
Voice QA Engine V1.2 - MOSS-Live WebSocket 双向流式客户端
负责网络传输、会话代次管理、上行持续推流与下行事件监听并发分离
"""

import asyncio
import json
import time
from typing import Optional, AsyncIterator, Dict, Any

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    _HAS_WEBSOCKETS = False

from voice_qa_engine.runtime.session_state import ConnectionState
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.clients.moss_protocol_adapter import MossProtocolAdapter
from voice_qa_engine.runtime.event_timeline import EventTimeline
from voice_qa_engine.audio.models import AudioFrame


class MossLiveClient:
    def __init__(
        self,
        protocol: MossProtocolAdapter,
        timeline: Optional[EventTimeline] = None,
        session_id: str = "default_session"
    ):
        self.protocol = protocol
        self.timeline = timeline
        self.session_id = session_id

        self.state = ConnectionState.DISCONNECTED
        self.connection_epoch = 0
        self.ws = None

        self._send_queue = asyncio.Queue(maxsize=150)
        self._event_queue = asyncio.Queue(maxsize=500)
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def connect(self, endpoint: str = "ws://localhost:8080/v1/realtime", headers: Optional[Dict[str, str]] = None):
        """建立 WebSocket 连接并执行协议握手"""
        if not _HAS_WEBSOCKETS or websockets is None:
            raise RuntimeError("websockets package is not installed")

        self.state = ConnectionState.CONNECTING
        self.connection_epoch += 1

        try:
            self.ws = await websockets.connect(
                endpoint,
                additional_headers=headers or {},
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self.state = ConnectionState.CONNECTED

            # 发送初始握手消息
            handshake = self.protocol.build_handshake({"session_id": self.session_id})
            await self.ws.send(json.dumps(handshake))
            self.state = ConnectionState.AUTHENTICATED

            # 拉起并发发送与接收协程
            self._stop_event.clear()
            self._sender_task = asyncio.create_task(self._send_loop())
            self._receiver_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            self.state = ConnectionState.FAILED
            if self.timeline:
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.ERROR,
                    session_id=self.session_id,
                    connection_epoch=self.connection_epoch,
                    source="websocket_client",
                    payload={"error_type": "connect_failed", "details": str(e)}
                ))
            raise

    async def send_audio(self, frame: AudioFrame):
        """将音频帧封装为上行协议消息推入有界发送队列"""
        if self.state != ConnectionState.AUTHENTICATED:
            return
        msg = self.protocol.build_audio_message(frame)
        try:
            self._send_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    async def send_control(self, payload: dict):
        """发送控制指令 (如打断取消)"""
        if self.ws:
            text = json.dumps(payload) if isinstance(payload, dict) else payload
            await self.ws.send(text)

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        """对外异步暴露下行事件生成器"""
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
                self._event_queue.task_done()
            except asyncio.TimeoutError:
                continue

    async def _send_loop(self):
        while not self._stop_event.is_set() and self.ws:
            try:
                msg = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                payload_str = json.dumps(msg) if isinstance(msg, dict) else msg
                await self.ws.send(payload_str)
                self._send_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def _receive_loop(self):
        while not self._stop_event.is_set() and self.ws:
            try:
                raw_msg = await self.ws.recv()
                events = self.protocol.parse_server_message(
                    raw_msg,
                    self.session_id,
                    self.connection_epoch
                )
                for evt in events:
                    if self.timeline:
                        await self.timeline.record(evt)
                    await self._event_queue.put(evt)
            except Exception:
                break

    async def close(self):
        self._stop_event.set()
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()

        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        self.state = ConnectionState.CLOSED
