#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 四态对话断点状态栈 (Context Call Stack) 单元测试
覆盖：
1. Resume Full: 恢复中断快照与打断不变性评分 (动态计算与 SLA >= 95% 门禁)
2. Drop Stale: 彻底废弃与 Tombstone 墓碑
3. Merge Context: 条件合并与 State Rebase
4. Timeout Expire: TTL 衰减与超时淘汰归档
5. FDB v3 场景 9: 改口回滚 (Rollback & Rebase) 原子状态重构
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

    def test_resume_full_dynamic_invariance_score_calculation(self):
        # 初始化主线带特定状态
        self.stack.frames[0].vars = {"hotel": "希尔顿", "nights": 3, "breakfast": True}
        self.stack.push_subtask("interruption_chat", {"intent": "ask_time"})

        # 1. 恢复上下文且状态完全一致 -> 得分 1.0 (>= 95%)
        resumed = self.stack.resume_full(
            target_depth=0,
            verification_vars={"hotel": "希尔顿", "nights": 3, "breakfast": True},
            expected_topic="mainline"
        )
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.interruption_invariance_score, 1.0)

        # 2. 模拟变量丢失/漂移 -> 不变性得分下降
        self.stack.push_subtask("another_sub", {})
        resumed_drift = self.stack.resume_full(
            target_depth=0,
            verification_vars={"hotel": "希尔顿"},  # 缺失 nights 和 breakfast
            expected_topic="mainline"
        )
        self.assertIsNotNone(resumed_drift)
        self.assertLess(resumed_drift.interruption_invariance_score, 0.5)

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

    def test_fdb_v3_correction_rollback_and_rebase(self):
        """测试 FDB v3 场景 9: 用户改口回滚与参数重构原子操作"""
        self.stack.frames[0].vars = {
            "origin": "北京",
            "destination": "上海",
            "train_code": "G101",
            "seat": "商务座"
        }
        # 用户改口：“等等，我不去上海了，改去杭州，坐一等座”
        rebased = self.stack.rollback_and_rebase(
            rollback_keys=["destination", "train_code", "seat"],
            updated_vars={"destination": "杭州", "train_code": "G33", "seat": "一等座"},
            target_depth=0
        )
        self.assertIsNotNone(rebased)
        self.assertEqual(rebased.vars["origin"], "北京")  # 始发地保持不变
        self.assertEqual(rebased.vars["destination"], "杭州")  # 目的地成功改口
        self.assertEqual(rebased.vars["seat"], "一等座")
        self.assertNotIn("上海", rebased.vars.values())  # 旧数据完全清除

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
