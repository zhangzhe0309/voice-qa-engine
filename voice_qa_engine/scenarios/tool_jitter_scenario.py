#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 用例 C: 工具调用并发抖动对抗场景 (Tool Concurrency Jitter Scenario)
遵循飞书评测规范 §7.4 / 用例 C：
在异步外部 HTTP/RPC 工具链并发调用下，严密核验音频管道的稳定性。
SLA 硬指标：
1. P95 Jitter <= 12ms
2. Max Jitter <= 15ms
3. Audio Buffer Underrun == 0 (零卡顿/零下溢)
4. Delta TTFT <= 50ms (工具执行对首字时延的额外延迟冲击 <= 50ms)
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType


class ToolConcurrencyJitterScenario(BaseVoiceScenario):
    """用例 C: 工具链并发下音频推流抖动与缓冲欠载测试"""

    def __init__(self, *args, baseline_ttft_ms: float = 400.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_ttft_ms = baseline_ttft_ms

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        events = self.timeline.snapshot()
        finished_ns = time.monotonic_ns()

        metrics = self._calculate_jitter_and_underrun(events)
        assertions = self.assertions(events, metrics)
        all_passed = all(a.passed for a in assertions)
        status = TestStatus.PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="tool_concurrency_jitter",
            status=status,
            started_ns=started_ns,
            finished_ns=finished_ns,
            assertions=assertions,
            metrics=metrics
        )

    def _calculate_jitter_and_underrun(self, events: list) -> Dict[str, Any]:
        """计算音频 Chunk 投递间隔抖动与欠载"""
        audio_events = [e for e in events if e.event_type == EventType.AUDIO_CHUNK]
        if len(audio_events) < 2:
            return {
                "p95_jitter_ms": 0.0,
                "max_jitter_ms": 0.0,
                "underrun_count": 0,
                "delta_ttft_ms": 0.0,
                "audio_chunk_count": len(audio_events)
            }

        intervals_ms = []
        for i in range(1, len(audio_events)):
            dt_ms = (audio_events[i].monotonic_ns - audio_events[i - 1].monotonic_ns) / 1_000_000.0
            intervals_ms.append(dt_ms)

        nominal_interval = 20.0  # 标称每 20ms 一帧
        jitters = [abs(dt - nominal_interval) for dt in intervals_ms]

        p95_jitter = float(np.percentile(jitters, 95))
        max_jitter = float(np.max(jitters))

        # 欠载事件统计
        underrun_events = [e for e in events if e.event_type == EventType.PLAYBACK_SINK_UNDERRUN]
        underrun_count = len(underrun_events)

        # Delta TTFT 统计
        input_ends = [e for e in events if e.event_type == EventType.INPUT_AUDIO_END]
        first_texts = [e for e in events if e.event_type in (EventType.TEXT_DELTA, EventType.RESPONSE_STARTED)]
        delta_ttft = 0.0
        if input_ends and first_texts:
            actual_ttft = (first_texts[0].monotonic_ns - input_ends[0].monotonic_ns) / 1_000_000.0
            delta_ttft = max(0.0, actual_ttft - self.baseline_ttft_ms)

        return {
            "p95_jitter_ms": round(p95_jitter, 2),
            "max_jitter_ms": round(max_jitter, 2),
            "underrun_count": underrun_count,
            "delta_ttft_ms": round(delta_ttft, 2),
            "audio_chunk_count": len(audio_events)
        }

    def assertions(self, events: list, metrics: Optional[Dict[str, Any]] = None) -> List[AssertionResult]:
        if metrics is None:
            metrics = self._calculate_jitter_and_underrun(events)

        results = []

        # 1. P95 抖动 <= 12ms
        p95 = metrics.get("p95_jitter_ms", 0.0)
        results.append(AssertionResult(
            name="assert_p95_jitter_sla",
            passed=p95 <= 12.0,
            severity=AssertionSeverity.CRITICAL,
            message=f"并发工具执行时音频下行 P95 抖动为 {p95}ms (SLA <= 12.0ms)"
        ))

        # 2. Max 抖动 <= 15ms
        max_j = metrics.get("max_jitter_ms", 0.0)
        results.append(AssertionResult(
            name="assert_max_jitter_sla",
            passed=max_j <= 15.0,
            severity=AssertionSeverity.MAJOR,
            message=f"并发工具执行时音频下行最大抖动为 {max_j}ms (SLA <= 15.0ms)"
        ))

        # 3. Buffer 零欠载 Underrun == 0
        underrun = metrics.get("underrun_count", 0)
        results.append(AssertionResult(
            name="assert_zero_audio_buffer_underrun",
            passed=underrun == 0,
            severity=AssertionSeverity.CRITICAL,
            message=f"播放缓冲发生欠载 {underrun} 次，必须严格等于 0 次 (SLA: Underrun == 0)"
        ))

        # 4. Delta TTFT <= 50ms
        delta_ttft = metrics.get("delta_ttft_ms", 0.0)
        results.append(AssertionResult(
            name="assert_delta_ttft_tool_impact",
            passed=delta_ttft <= 50.0,
            severity=AssertionSeverity.MAJOR,
            message=f"工具并发导致的首包时延增量为 {delta_ttft}ms (SLA <= 50.0ms)"
        ))

        return results
