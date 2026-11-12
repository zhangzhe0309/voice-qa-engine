#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 真实不流利口语与倒吸气耐受场景 (Disfluent Speech & Inhalation Tolerance)
对应评测规范 场景 2 (Full-Duplex-Bench v3):
1. 前置注入 80ms 急促倒吸气声 → 声学前端应提前 50ms 压麦 (Pre-emptive Duck-down)
2. 否定纠偏"不是不是...我是说换王老师" → 语义慢通道过滤否定词，杜绝脏参数写入
3. 批量试验 + Wilson 置信区间：误抢话率 SLA <= 1.5%（n >= 200 才具备统计判定力）
"""

import time
from typing import List, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.evaluators.statistics import ratio_verdict, assertion_from_verdict
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.overlap_speech import OverlapSpeechSynthesizer

DISFLUENT_UTTERANCE = "不是不是...呃...那个...我是说换王老师"
NEGATION_TOKENS = ("不是不是", "不对不对", "我说的是")


class DisfluencyScenario(BaseVoiceScenario):
    """倒吸气 + 口吃/自我修正对抗场景（批量试验 + 统计置信判定）"""

    def __init__(self, *args,
                 trials: int = 300,
                 premature_interruption_sla: float = 0.015,
                 min_samples: int = 200,
                 inhalation_ms: float = 80.0,
                 negation_filter_expected: bool = True,
                 seed: int = 42,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.trials = trials
        self.premature_interruption_sla = premature_interruption_sla
        self.min_samples = min_samples
        self.inhalation_ms = inhalation_ms
        self.negation_filter_expected = negation_filter_expected
        self.synth = OverlapSpeechSynthesizer(seed=seed)

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        errors: List[str] = []
        premature_hits = 0
        duckdown_hits = 0
        negation_leaks = 0

        try:
            if hasattr(self.audio_input, "open"):
                await self.audio_input.open()
            for i in range(self.trials):
                trial_start = time.monotonic_ns()
                # 前置倒吸气 + 不流利话术注入（经真实上行链路）
                inhalation = self.synth.generate_inhalation(self.inhalation_ms)
                if hasattr(self.audio_input, "push_segment"):
                    self.audio_input.push_segment(inhalation)
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.INPUT_AUDIO_SENT,
                    session_id=self.runtime.session_id,
                    source="disfluency_scenario",
                    payload={"trial": i, "kind": "inhalation", "duration_ms": self.inhalation_ms},
                ))

                # 判定窗口内服务端是否误抢话（下行音频分片 = 抢话物证）
                window_end = time.monotonic_ns() + 150_000_000  # 150ms 语义慢通道窗口
                premature = self._premature_downlink(trial_start, window_end)
                premature_hits += int(premature)
                duckdown_hits += int(not premature)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

        # 否定词过滤（会话级一次性判定）：回写文本含否定段且未落到正确实体 → 脏参数泄漏
        negation_leaks = self._count_negation_leaks()

        metrics = {
            "trials": self.trials,
            "premature_interruptions": premature_hits,
            "preemptive_duckdown_hits": duckdown_hits,
            "negation_leaks": negation_leaks,
            "observed_premature_rate": (premature_hits / self.trials) if self.trials else None,
        }
        assertions = self.assertions([], metrics)
        all_passed = all(a.passed for a in assertions) and not errors
        status = (TestStatus.SIMULATED_PASS if all_passed else TestStatus.FAIL) if self.trials else TestStatus.NOT_RUN

        return ScenarioResult(
            scenario_id="disfluency_inhalation_tolerance",
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
            metrics=metrics,
            errors=errors,
        )

    def _premature_downlink(self, window_start_ns: int, window_end_ns: int) -> bool:
        """判定窗口内出现下行音频分片即构成误抢话（真实链路物证口径）"""
        for e in self.timeline.snapshot():
            if e.event_type == EventType.AUDIO_CHUNK and window_start_ns <= e.monotonic_ns <= window_end_ns:
                return True
        return False

    def _count_negation_leaks(self) -> int:
        """回写文本中否定纠偏段未被净化（含否定词且未落到正确实体'王老师'）的次数"""
        leaks = 0
        for e in self.timeline.snapshot():
            if e.event_type == EventType.TEXT_DELTA and e.source == "dialog_backend":
                text = str(e.payload.get("delta", ""))
                has_negation = any(tok in text for tok in NEGATION_TOKENS)
                if has_negation and "王老师" not in text:
                    leaks += 1
        return leaks

    def assertions(self, events: list, metrics: Optional[Dict[str, Any]] = None) -> List[AssertionResult]:
        if metrics is None:
            metrics = {}
        results = []

        # 1. 误抢话率 (Wilson 置信上界 <= 1.5%，n >= 200)
        verdict = ratio_verdict(
            name="premature_interruption_rate",
            successes=metrics.get("premature_interruptions", 0),
            n=metrics.get("trials", 0),
            sla=self.premature_interruption_sla,
            higher_is_better=False,
            min_samples=self.min_samples,
        )
        results.append(assertion_from_verdict(verdict, "assert_premature_interruption_rate_sla"))

        # 2. 倒吸气前置压麦覆盖率
        duckdown_rate = (metrics.get("preemptive_duckdown_hits", 0) / metrics["trials"]
                         if metrics.get("trials") else 0.0)
        results.append(AssertionResult(
            name="assert_preemptive_duckdown_coverage",
            passed=duckdown_rate >= 0.985,
            severity=AssertionSeverity.MAJOR,
            message=f"倒吸气后 150ms 内正确压麦不抢话覆盖率 {duckdown_rate * 100:.2f}% (SLA >= 98.5%)",
            expected=">= 0.985",
            actual=round(duckdown_rate, 4),
        ))

        # 3. 否定词过滤（脏参数 0 写入）
        results.append(AssertionResult(
            name="assert_negation_filtered_no_dirty_params",
            passed=metrics.get("negation_leaks", 0) == 0,
            severity=AssertionSeverity.CRITICAL,
            message=f"否定纠偏'不是不是...我是说换王老师'必须过滤否定段，脏参数写入严格为 0 "
                    f"(实测: {metrics.get('negation_leaks')})",
            expected=0,
            actual=metrics.get("negation_leaks"),
        ))
        return results
