#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 场景 A：正常问答场景 (NormalQAScenario)
验证持续上行音频、服务端正常响应、文本增量与下行音频完整消费
"""

import time
from typing import List
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import EventType


class NormalQAScenario(BaseVoiceScenario):
    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        finished_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        assertions = self.assertions(events)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="normal_qa",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        results = []

        # 1. 连接与就绪断言
        has_ready = any(e.event_type == EventType.SESSION_READY for e in events)
        results.append(AssertionResult(
            name="assert_session_ready",
            passed=has_ready,
            severity=AssertionSeverity.CRITICAL,
            message="会话必须成功建立并进入就绪状态"
        ))

        # 2. 行音频流断言
        input_sent_events = [e for e in events if e.event_type == EventType.INPUT_AUDIO_SENT]
        has_input_end = any(e.event_type == EventType.INPUT_AUDIO_END for e in events)
        results.append(AssertionResult(
            name="assert_audio_upload_continuous",
            passed=len(input_sent_events) > 0 and has_input_end,
            severity=AssertionSeverity.CRITICAL,
            message=f"上行音频帧必须持续完整发送 (帧数: {len(input_sent_events)}, 是否完结: {has_input_end})"
        ))

        # 3. 响应开始与音频下行断言
        has_resp_start = any(e.event_type == EventType.RESPONSE_STARTED for e in events)
        chunk_events = [e for e in events if e.event_type == EventType.AUDIO_CHUNK]
        results.append(AssertionResult(
            name="assert_response_and_audio_chunks_received",
            passed=has_resp_start and len(chunk_events) > 0,
            severity=AssertionSeverity.CRITICAL,
            message=f"服务端必须产生响应并下发音频分片 (分片数: {len(chunk_events)})"
        ))

        # 4. 全程无未处理 Error
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        results.append(AssertionResult(
            name="assert_no_unhandled_errors",
            passed=len(error_events) == 0,
            severity=AssertionSeverity.CRITICAL,
            message=f"问答流程中不得发生异常错误 (异常数: {len(error_events)})"
        ))

        return results
