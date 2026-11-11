#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 协议适配器抽象与实现
将外部私有 WebSocket 协议消息与内部归一化 RuntimeEvent 双向映射
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType


class MossProtocolAdapter(ABC):
    @abstractmethod
    def build_handshake(self, config: Dict[str, Any]) -> dict:
        """生成握手鉴权消息体"""
        pass

    @abstractmethod
    def build_audio_message(self, payload: bytes, sequence_id: int) -> Any:
        """生成上行音频帧消息"""
        pass

    @abstractmethod
    def build_cancel_message(self, response_id: str) -> Any:
        """生成客户端打断/取消请求消息"""
        pass

    @abstractmethod
    def parse_server_message(self, raw_message: Any, session_id: str) -> List[RuntimeEvent]:
        """将服务端原始下行消息解析为内部归一化 RuntimeEvent 列表"""
        pass


class MockMossProtocolAdapter(MossProtocolAdapter):
    """测试与协议骨架适配器"""
    def build_handshake(self, config: Dict[str, Any]) -> dict:
        return {"action": "session.init", "model": config.get("model", "moss-live-v1")}

    def build_audio_message(self, payload: bytes, sequence_id: int) -> dict:
        return {"action": "input.audio", "seq": sequence_id, "size": len(payload)}

    def build_cancel_message(self, response_id: str) -> dict:
        return {"action": "response.cancel", "response_id": response_id}

    def parse_server_message(self, raw_message: Any, session_id: str) -> List[RuntimeEvent]:
        # 内部映射，严禁把本地生成字段伪造成服务端字段
        return []
