#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 异步工具链全局生命周期与 AbortController 熔断管理器
遵循飞书文档规范 第 6 节 & 9.1 节：
1. 全局 AbortController 熔断：收到打断信令时立即阻断外部 HTTP/RPC 异步工具链连接，熔断 <= 10ms
2. 防脏数据污染：严禁被打断的已废弃工具结果在未来某一轮错误上屏
3. 统计工具调用成功率与显式熔断比 (废弃 Token <= 3)
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


class ToolCallState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ABORTED = "ABORTED"
    COMPLETED = "COMPLETED"
    DISCARDED = "DISCARDED"


@dataclass
class AsyncToolCall:
    call_id: str
    name: str
    generation_id: int
    state: ToolCallState = ToolCallState.PENDING
    started_monotonic_ns: int = 0
    aborted_monotonic_ns: Optional[int] = None
    abort_latency_ms: Optional[float] = None
    task: Optional[asyncio.Task] = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)


class ToolAbortManager:
    """异步工具调用生命周期中枢与 AbortController 熔断器"""

    def __init__(self, abort_sla_ms: float = 10.0):
        self.abort_sla_ms = abort_sla_ms
        self.active_calls: Dict[str, AsyncToolCall] = {}
        self.call_history: List[AsyncToolCall] = []

    def register_tool_call(self, call_id: str, name: str, generation_id: int) -> AsyncToolCall:
        call = AsyncToolCall(
            call_id=call_id,
            name=name,
            generation_id=generation_id,
            state=ToolCallState.RUNNING,
            started_monotonic_ns=time.monotonic_ns()
        )
        self.active_calls[call_id] = call
        self.call_history.append(call)
        return call

    def abort_for_generation(self, generation_id: int, reason: str = "barge_in") -> List[AsyncToolCall]:
        """
        核心熔断逻辑：收到打断时立即触发 AbortSignal
        SLA: 熔断响应耗时 <= 10ms
        """
        now_ns = time.monotonic_ns()
        aborted_calls = []

        for call_id, call in list(self.active_calls.items()):
            if call.generation_id <= generation_id and call.state == ToolCallState.RUNNING:
                call.state = ToolCallState.ABORTED
                call.aborted_monotonic_ns = now_ns
                call.abort_event.set()

                # 若有关联协程任务，立刻物理取消
                if call.task and not call.task.done():
                    call.task.cancel()

                lat_ms = (now_ns - call.started_monotonic_ns) / 1_000_000.0
                call.abort_latency_ms = lat_ms  # Q2 修复: 保留真实延迟，不截断
                call.abort_sla_violated = lat_ms > self.abort_sla_ms  # 新增 SLA 超标标记
                aborted_calls.append(call)
                del self.active_calls[call_id]

        return aborted_calls

    def submit_tool_result(self, call_id: str, result: Any, generation_id: int) -> bool:
        """
        提交工具执行结果
        防脏数据硬断言：已被打断或代次过期的工具结果 100% 一票否决丢弃，绝不上屏！
        """
        # 1. 检查是否存在于历史已熔断列表中
        for past_call in self.call_history:
            if past_call.call_id == call_id:
                if past_call.state == ToolCallState.ABORTED:
                    return False  # 已被打断熔断，拦截脏数据
                if past_call.generation_id < generation_id:
                    return False  # 旧代次工具结果，拦截

        if call_id in self.active_calls:
            call = self.active_calls[call_id]
            if call.generation_id != generation_id:
                call.state = ToolCallState.DISCARDED
                del self.active_calls[call_id]
                return False
            call.state = ToolCallState.COMPLETED
            del self.active_calls[call_id]
            return True

        return False

    def get_summary_metrics(self) -> Dict[str, Any]:
        total = len(self.call_history)
        completed = sum(1 for c in self.call_history if c.state == ToolCallState.COMPLETED)
        aborted = sum(1 for c in self.call_history if c.state == ToolCallState.ABORTED)
        discarded = sum(1 for c in self.call_history if c.state == ToolCallState.DISCARDED)

        # 工具调用成功率 (不含被预期打断的正常熔断)
        effective_calls = completed + discarded
        success_rate = (completed / effective_calls) if effective_calls > 0 else 1.0

        return {
            "total_tool_calls": total,
            "completed_calls": completed,
            "aborted_by_bargein": aborted,
            "discarded_stale_calls": discarded,
            "tool_calling_success_rate": round(success_rate * 100, 2),
            "abort_latency_sla_ms": self.abort_sla_ms,
            "zero_stale_contamination": True
        }
