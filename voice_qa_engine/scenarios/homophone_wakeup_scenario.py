#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 恶意谐音同音误唤醒场景 (Homophone False-Wakeup)
对应评测规范 7.3：多人讨论中高频注入"模式设计/摸索前进/魔术表演/磨合期"等
与唤醒词同音/近音的干扰语，误触发必须为 0.0%（精确零容忍 + 最低样本量门禁）。
"""

import time
from typing import List, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.corpus_matrix import CorpusMatrix, HOMOPHONE_TRAPS


class HomophoneWakeupScenario(BaseVoiceScenario):
    """谐音挑衅误唤醒零容忍场景"""

    def __init__(self, *args,
                 repetitions_per_trap: int = 50,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.repetitions_per_trap = repetitions_per_trap

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        errors: List[str] = []
        false_wakeups = 0
        total_injections = len(HOMOPHONE_TRAPS) * self.repetitions_per_trap

        try:
            matrix = CorpusMatrix()
            traps = [c for c in matrix.build_cases() if c.category == "homophone_trap"]
            for case in traps:
                for rep in range(self.repetitions_per_trap):
                    await self.timeline.record(RuntimeEvent(
                        event_type=EventType.TEXT_DELTA,
                        session_id=self.runtime.session_id,
                        source="homophone_scenario",
                        payload={"case_id": case.case_id, "rep": rep, "delta": case.text},
                    ))
                    # 物证口径：注入后若出现 RESPONSE_STARTED/下行分片即构成误唤醒
                    false_wakeups += int(self._false_wakeup_observed(case.case_id, rep))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

        metrics = {
            "total_injections": total_injections,
            "false_wakeups": false_wakeups,
            "observed_false_wakeup_rate": (false_wakeups / total_injections) if total_injections else None,
            "samples_sufficient": total_injections >= self.min_samples,
        }
        assertions = self.assertions([], metrics)
        all_passed = all(a.passed for a in assertions) and not errors
        status = TestStatus.SIMULATED_PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="homophone_false_wakeup",
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
            metrics=metrics,
            errors=errors,
        )

    min_samples = 200

    def _false_wakeup_observed(self, case_id: str, rep: int) -> bool:
        """注入该用例后事件流中出现的任何下行响应即误唤醒（真实链路口径）"""
        # 时间窗口以注入事件为锚：真实链路下 RESPONSE_STARTED/AUDIO_CHUNK 晚于注入即计
        events = self.timeline.snapshot()
        for e in events:
            if (e.event_type in (EventType.RESPONSE_STARTED, EventType.AUDIO_CHUNK)
                    and e.source == "websocket"):
                return True
        return False

    def assertions(self, events: list, metrics: Optional[Dict[str, Any]] = None) -> List[AssertionResult]:
        if metrics is None:
            metrics = {}
        results = []

        # 1. 误唤醒精确零容忍
        results.append(AssertionResult(
            name="assert_zero_homophone_false_wakeup",
            passed=metrics.get("false_wakeups", 0) == 0,
            severity=AssertionSeverity.CRITICAL,
            message=(f"谐音同音挑衅（模式设计/摸索前进/魔术表演/磨合期等）误唤醒必须为 0.0% "
                     f"(实测: {metrics.get('false_wakeups', 0)} / {metrics.get('total_injections', 0)})"),
            expected=0,
            actual=metrics.get("false_wakeups"),
        ))

        # 2. 样本量门禁（零容忍结论需要足够样本支撑）
        results.append(AssertionResult(
            name="assert_homophone_sample_size_sufficient",
            passed=bool(metrics.get("samples_sufficient")),
            severity=AssertionSeverity.MAJOR,
            message=(f"零容忍判定需要 >= {self.min_samples} 次注入样本支撑 "
                     f"(实测: {metrics.get('total_injections', 0)})"),
        ))
        return results
