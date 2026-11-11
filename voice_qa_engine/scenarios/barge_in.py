#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 场景 B：播报中打断 (BargeInScenario)
断言：
1. 收到打断请求后，本地旧 Generation 立即失效
2. 待播队列清空 (queue.flushed)
3. 底层 Sink 停止并哑音 (sink.stopped)
4. 打断后延迟到达的旧代次 Chunk 全部拒绝
"""

import time
from typing import List
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import EventType


class BargeInScenario(BaseVoiceScenario):
    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        # 依赖外部 runtime 调度
        finished_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="barge_in",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        # 1. 必须发生打断请求
        interrupt_events = [e for e in events if e.event_type == EventType.PLAYBACK_INTERRUPT_REQUESTED]
        has_interrupt = len(interrupt_events) > 0
        results.append(AssertionResult(
            name="assert_interrupt_requested",
            passed=has_interrupt,
            severity=AssertionSeverity.CRITICAL,
            message="必须记录到用户插话打断请求事件"
        ))

        if not has_interrupt:
            return results

        interrupt_evt = interrupt_events[0]
        interrupt_ns = interrupt_evt.monotonic_ns
        old_gen = interrupt_evt.payload.get("old_generation_id", 0)
        new_gen = interrupt_evt.payload.get("new_generation_id", 0)

        # 2. 本地代次严格递增
        results.append(AssertionResult(
            name="assert_generation_incremented",
            passed=(new_gen > old_gen),
            severity=AssertionSeverity.CRITICAL,
            message=f"打断后本地播放代次必须严格递增 (old: {old_gen}, new: {new_gen})"
        ))

        # 3. 必须在打断后发生列清空与 Sink 停止
        flushed_events = [e for e in events if e.event_type == EventType.PLAYBACK_QUEUE_FLUSHED and e.monotonic_ns >= interrupt_ns]
        sink_stopped_events = [e for e in events if e.event_type == EventType.PLAYBACK_SINK_STOPPED and e.monotonic_ns >= interrupt_ns]

        results.append(AssertionResult(
            name="assert_playback_queue_flushed",
            passed=len(flushed_events) > 0,
            severity=AssertionSeverity.CRITICAL,
            message="打断发生后待播队列必须物理清空"
        ))

        results.append(AssertionResult(
            name="assert_playback_sink_stopped",
            passed=len(sink_stopped_events) > 0,
            severity=AssertionSeverity.CRITICAL,
            message="打断发生后物理 Sink 必须被调用停止并清空缓冲"
        ))

        # 4. 打断之后到达的旧 Generation Chunk 绝不能触发播放
        post_interrupt_play = [
            e for e in events
            if e.event_type == EventType.PLAYBACK_STARTED
            and e.monotonic_ns >= interrupt_ns
            and e.generation_id <= old_gen
        ]
        results.append(AssertionResult(
            name="assert_no_ghost_audio_played",
            passed=len(post_interrupt_play) == 0,
            severity=AssertionSeverity.CRITICAL,
            message="打断之后绝对禁止播放旧 Generation 音频"
        ))

        return results
