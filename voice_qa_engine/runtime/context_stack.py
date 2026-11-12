#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 四态对话断点状态栈 (Context Call Stack Engine)
严格遵循飞书文档规范 第 5 节与第 8 节架构标准：
1. Resume Full (完全接续，打断不变性评分 >= 95%)
2. Drop Stale (彻底丢弃，Tombstone 墓碑，物理释放防幽灵回溯)
3. Merge Context (条件合并，State Rebase，增量重构主线状态)
4. Timeout Expire (超时淘汰，TTL 默认 10 轮未召回自动退火归档)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import time


class FrameState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DROPPED = "DROPPED"
    MERGED = "MERGED"
    EXPIRED = "EXPIRED"


class ArbitrationAction(str, Enum):
    RESUME_FULL = "RESUME_FULL"
    DROP_STALE = "DROP_STALE"
    MERGE_CONTEXT = "MERGE_CONTEXT"
    TIMEOUT_EXPIRE = "TIMEOUT_EXPIRE"


@dataclass
class ContextFrame:
    depth: int
    topic: str
    state: FrameState = FrameState.ACTIVE
    vars: Dict[str, Any] = field(default_factory=dict)
    snapshot_timestamp_ms: float = 0.0
    ttl: int = 10
    interruption_invariance_score: float = 1.0


class ContextCallStack:
    """分层四态对话断点调用栈中枢"""

    def __init__(self, default_ttl: int = 10):
        self.default_ttl = default_ttl
        self.frames: List[ContextFrame] = []
        # 初始化根栈帧 (主线 Depth 0)
        root = ContextFrame(
            depth=0,
            topic="mainline",
            state=FrameState.ACTIVE,
            snapshot_timestamp_ms=time.time() * 1000.0,
            ttl=default_ttl
        )
        self.frames.append(root)

    @property
    def current_depth(self) -> int:
        return len(self.frames) - 1

    def active_frame(self) -> Optional[ContextFrame]:
        for f in reversed(self.frames):
            if f.state == FrameState.ACTIVE:
                return f
        return None

    def push_subtask(self, topic: str, initial_vars: Optional[Dict[str, Any]] = None, snapshot_ms: float = 0.0) -> ContextFrame:
        """挂起当前活跃栈帧并压入新支线任务栈帧 (Depth 1..N)"""
        active = self.active_frame()
        if active:
            active.state = FrameState.SUSPENDED
            active.snapshot_timestamp_ms = snapshot_ms or (time.time() * 1000.0)

        new_frame = ContextFrame(
            depth=len(self.frames),
            topic=topic,
            state=FrameState.ACTIVE,
            vars=initial_vars or {},
            snapshot_timestamp_ms=snapshot_ms or (time.time() * 1000.0),
            ttl=self.default_ttl
        )
        self.frames.append(new_frame)
        return new_frame

    def resume_full(self, target_depth: Optional[int] = None) -> Optional[ContextFrame]:
        """
        状态 ①: 完全接续 (Resume Full)
        弹出栈顶支线快照，恢复挂起的主线或上一级上下文，保持打断不变性
        """
        # 如果当前有活跃帧且不是目标帧，先结案或标记
        active = self.active_frame()
        if active and target_depth is not None and active.depth != target_depth:
            active.state = FrameState.MERGED

        # 寻找最近的 SUSPENDED 帧恢复
        for f in reversed(self.frames):
            if target_depth is not None and f.depth != target_depth:
                continue
            if f.state == FrameState.SUSPENDED:
                f.state = FrameState.ACTIVE
                f.interruption_invariance_score = 1.0
                return f
        return None

    def drop_stale(self, target_depth: Optional[int] = None, drop_all: bool = False) -> List[ContextFrame]:
        """
        状态 ②: 彻底丢弃 (Drop Stale)
        显式废弃指定或全部挂起栈帧，标记为 Tombstone 墓碑，物理释放上下文变量，幽灵回溯 0%
        """
        dropped = []
        for f in self.frames:
            if drop_all:
                if f.depth > 0:  # 保留根帧结构，但重置状态
                    f.state = FrameState.DROPPED
                    f.vars.clear()
                    dropped.append(f)
            else:
                if target_depth is not None and f.depth == target_depth:
                    f.state = FrameState.DROPPED
                    f.vars.clear()
                    dropped.append(f)
                elif target_depth is None and f.state == FrameState.SUSPENDED:
                    f.state = FrameState.DROPPED
                    f.vars.clear()
                    dropped.append(f)
                    break
        return dropped

    def merge_context(self, new_vars: Dict[str, Any], target_depth: int = 0) -> Optional[ContextFrame]:
        """
        状态 ③: 条件合并 (Merge Context / State Rebase)
        将打断过程中的新参数增量重构合并到目标栈帧 (通常为主线)，发起新规划
        """
        for f in self.frames:
            if f.depth == target_depth:
                f.vars.update(new_vars)
                f.state = FrameState.ACTIVE
                return f
        return None

    def tick_turn(self) -> List[ContextFrame]:
        """
        状态 ④: 对话轮次时钟推进与超时淘汰 (Timeout Expire)
        每轮对话执行一次 tick，挂起栈帧 TTL 递减，耗尽则物理归档
        """
        expired = []
        for f in self.frames:
            if f.state == FrameState.SUSPENDED:
                f.ttl -= 1
                if f.ttl <= 0:
                    f.state = FrameState.EXPIRED
                    f.vars.clear()
                    expired.append(f)
        return expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_depth": self.current_depth,
            "active_topic": self.active_frame().topic if self.active_frame() else None,
            "frames": [
                {
                    "depth": f.depth,
                    "topic": f.topic,
                    "state": f.state.value,
                    "vars": dict(f.vars),
                    "ttl": f.ttl,
                    "snapshot_ms": f.snapshot_timestamp_ms,
                    "invariance_score": f.interruption_invariance_score
                }
                for f in self.frames
            ]
        }
