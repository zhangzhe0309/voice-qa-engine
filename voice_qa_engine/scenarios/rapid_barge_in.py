#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 场景 C：快速连续两次打断 (RapidDoubleBargeInScenario)
断言：
1. 两次打断事件均被捕获且代次依次递增
2. 第二次打断发生后，第一次和第二次打断所涉及的旧代次绝不再恢复播放
"""

import time
from typing import List
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import EventType


class RapidDoubleBargeInScenario(BaseVoiceScenario):
    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        finished_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="rapid_double_barge_in",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        interrupts = [e for e in events if e.event_type == EventType.PLAYBACK_INTERRUPT_REQUESTED]
        has_two_interrupts = len(interrupts) >= 2
        results.append(AssertionResult(
            name="assert_two_interrupts_recorded",
            passed=has_two_interrupts,
            severity=AssertionSeverity.CRITICAL,
            message=f"必须记录到至少两次打断请求事件 (实际: {len(interrupts)})"
        ))

        if not has_two_interrupts:
            return results

        first_old = interrupts[0].payload.get("old_generation_id", 0)
        second_old = interrupts[1].payload.get("old_generation_id", 0)
        second_ns = interrupts[1].monotonic_ns

        # 1. 第二次打断代次必须严格大于第一次
        results.append(AssertionResult(
            name="assert_second_interrupt_generation_greater",
            passed=(second_old > first_old),
            severity=AssertionSeverity.CRITICAL,
            message=f"第二次打断的代次必须大于第一次代次 (first: {first_old}, second: {second_old})"
        ))

        # 2. 第二次打断之后，旧 Generation (<= second_old) 绝不能重新开始播放
        resumed_stale_plays = [
            e for e in events
            if e.event_type == EventType.PLAYBACK_STARTED
            and e.monotonic_ns >= second_ns
            and e.generation_id <= second_old
        ]

        results.append(AssertionResult(
            name="assert_stale_generation_never_resumes",
            passed=len(resumed_stale_plays) == 0,
            severity=AssertionSeverity.CRITICAL,
            message="第二次打断之后旧 Generation 绝不能恢复播放 (防伪通过验证)"
        ))

        return results
