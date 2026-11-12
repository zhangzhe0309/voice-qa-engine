#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 高阶专项评测对抗场景 (Advanced Scenarios)
依据飞书评测规范 v1.4.0 / v1.4.1 实现四大专项用例断言：
1. BackchannelScenario: 用例 A / 场景 4 微弱附和倾听无缝免打断 (精度 >= 98.5%, 零 CANCEL/FLUSH)
2. HesitationToleranceScenario: 用例 B / 场景 2 思考留白阶梯停顿对抗 (0.5s~2.5s 抢话率 <= 1.5%, Holding 态)
3. AudioImpulseNoiseScenario: 用例 D 短时脉冲杂音抗扰与音量平滑回弹 (误打断 == 0 次)
"""

import time
from typing import List, Dict, Any, Optional
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType


class BackchannelScenario(BaseVoiceScenario):
    """用例 A: 高频微弱附和免打断场景 (Backchanneling Precision >= 98.5%)"""

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        finished_ns = time.monotonic_ns()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="backchanneling",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        # 1. 弱附和期间绝不能发出 CANCEL 或 FLUSH
        cancel_events = [e for e in events if e.event_type in (EventType.RESPONSE_CANCELLED, EventType.PLAYBACK_QUEUE_FLUSHED)]
        results.append(AssertionResult(
            name="assert_no_cancel_or_flush_during_backchannel",
            passed=len(cancel_events) == 0,
            severity=AssertionSeverity.CRITICAL,
            message="微弱附和 (嗯/对/好的) 期间严禁发出 CANCEL 信令或执行待播队列 Flush"
        ))

        # 2. 下行推流必须持续产生 AUDIO_CHUNK
        audio_chunks = [e for e in events if e.event_type == EventType.AUDIO_CHUNK]
        results.append(AssertionResult(
            name="assert_audio_stream_continuity",
            passed=len(audio_chunks) > 0,
            severity=AssertionSeverity.MAJOR,
            message="附和期间下行音频推流必须保持连续流式合成，无非预期顿挫"
        ))

        return results


class HesitationToleranceScenario(BaseVoiceScenario):
    """用例 B: 思考留白阶梯停顿对抗场景 (0.5s~2.5s 留白误抢话率 <= 1.5%)"""

    def __init__(self, *args, pause_duration_s: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_duration_s = pause_duration_s

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        finished_ns = time.monotonic_ns()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="hesitation_tolerance",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions,
            metrics={"pause_duration_s": self.pause_duration_s}
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        # 检查是否在 2.5s 留白区间内发生提前抢话
        premature_speeches = [
            e for e in events
            if e.event_type == EventType.RESPONSE_STARTED
            and e.payload.get("is_premature_interruption", False)
        ]

        if self.pause_duration_s <= 2.5:
            results.append(AssertionResult(
                name="assert_holding_state_no_premature_takeover",
                passed=len(premature_speeches) == 0,
                severity=AssertionSeverity.CRITICAL,
                message=f"在 {self.pause_duration_s}s 思考留白正常区间内，系统必须保持 Holding 态，严禁提前抢话 (SLA <= 1.5%)"
            ))
        else:
            # 超过 3.0s 超时交权边界
            timeout_takeovers = [
                e for e in events
                if e.event_type == EventType.RESPONSE_STARTED
            ]
            results.append(AssertionResult(
                name="assert_graceful_timeout_turn_shift",
                passed=len(timeout_takeovers) > 0,
                severity=AssertionSeverity.MAJOR,
                message="超过 3.0s 超时边界后系统应礼貌性发起交权或提示"
            ))

        return results


class AudioImpulseNoiseScenario(BaseVoiceScenario):
    """用例 D: 短时脉冲杂音抗扰与音量回弹 (50~120ms 咳嗽喷麦误打断 == 0 次)"""

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        finished_ns = time.monotonic_ns()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="impulse_noise_robustness",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        # 短时杂音不得触发有效打断
        false_interrupts = [
            e for e in events
            if e.event_type == EventType.PLAYBACK_INTERRUPT_REQUESTED
            and e.payload.get("trigger") == "impulse_noise"
        ]

        results.append(AssertionResult(
            name="assert_zero_false_barge_in_on_impulse_noise",
            passed=len(false_interrupts) == 0,
            severity=AssertionSeverity.CRITICAL,
            message="短时脉冲杂音 (50~120ms 咳嗽喷麦) 误打断次数必须严格等于 0 次"
        ))

        return results
