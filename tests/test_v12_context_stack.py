#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 四态对话断点状态栈 (Context Call Stack) 单元测试
覆盖：
1. Resume Full: 恢复中断快照与打断不变性评分
2. Drop Stale: 彻底废弃与 Tombstone 墓碑
3. Merge Context: 条件合并与 State Rebase
4. Timeout Expire: TTL 衰减与超时淘汰归档
"""

import unittest
from voice_qa_engine.runtime.context_stack import (
    ContextCallStack,
    FrameState,
    ContextFrame
)


class TestContextCallStack(unittest.TestCase):
    def setUp(self):
        self.stack = ContextCallStack(default_ttl=5)

    def test_initial_state(self):
        active = self.stack.active_frame()
        self.assertIsNotNone(active)
        self.assertEqual(active.depth, 0)
        self.assertEqual(active.topic, "mainline")
        self.assertEqual(active.state, FrameState.ACTIVE)

    def test_push_subtask_suspends_mainline(self):
        sub = self.stack.push_subtask("query_weather", {"city": "Beijing"}, snapshot_ms=1500.0)
        self.assertEqual(sub.depth, 1)
        self.assertEqual(sub.topic, "query_weather")
        self.assertEqual(sub.state, FrameState.ACTIVE)

        # 检查主线是否被挂起
        main = self.stack.frames[0]
        self.assertEqual(main.state, FrameState.SUSPENDED)
        self.assertEqual(main.snapshot_timestamp_ms, 1500.0)

    def test_resume_full_restores_mainline(self):
        self.stack.push_subtask("query_weather", {"city": "Beijing"}, snapshot_ms=1500.0)
        resumed = self.stack.resume_full(target_depth=0)

        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.depth, 0)
        self.assertEqual(resumed.state, FrameState.ACTIVE)
        self.assertGreaterEqual(resumed.interruption_invariance_score, 0.95)

    def test_drop_stale_marks_tombstone(self):
        self.stack.push_subtask("cancelled_topic", {"key": "val"})
        dropped_list = self.stack.drop_stale(target_depth=1)

        self.assertEqual(len(dropped_list), 1)
        self.assertEqual(dropped_list[0].state, FrameState.DROPPED)
        self.assertEqual(len(dropped_list[0].vars), 0)

    def test_merge_context_rebases_mainline(self):
        self.stack.push_subtask("change_owner", {"owner": "李老师"})
        merged = self.stack.merge_context({"day2_owner": "李老师", "budget": 50000}, target_depth=0)

        self.assertIsNotNone(merged)
        self.assertEqual(merged.vars["day2_owner"], "李老师")
        self.assertEqual(merged.vars["budget"], 50000)
        self.assertEqual(merged.state, FrameState.ACTIVE)

    def test_timeout_expire_after_ttl_exhaustion(self):
        self.stack.push_subtask("temporary_query", {"temp": True})
        # 主线现在挂起，TTL = 5
        self.assertEqual(self.stack.frames[0].state, FrameState.SUSPENDED)

        # 推进 4 轮，尚未超时
        for _ in range(4):
            expired = self.stack.tick_turn()
            self.assertEqual(len(expired), 0)

        # 第 5 轮推进，触发超时淘汰
        expired = self.stack.tick_turn()
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].state, FrameState.EXPIRED)


if __name__ == "__main__":
    unittest.main()
