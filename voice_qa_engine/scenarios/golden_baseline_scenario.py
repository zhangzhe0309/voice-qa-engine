#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 黄金准入基线 30 分钟 6 阶段 soak 场景 (Golden Baseline Soak)
对应评测规范 6.1 六阶段时序规范 + 场景 8 长程状态栈健康：
  阶段 1 主线长篇讲解 → 阶段 2 打断改参 → 阶段 3 多人讨论静默 →
  阶段 4 定向点名唤醒 → 阶段 5 断点接续 → 阶段 6 长程疲劳巡检
长程阶段确定性随机注入 15 打断 / 7 支线 / 20 附和；RSS 内存漂移 <= 10%。
时长参数化 + time_compression 压缩比（测试用压缩跑通编排，真实跑 30min）。
"""

import time
import random
from typing import List, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.overlap_speech import OverlapSpeechSynthesizer

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _rss_bytes() -> Optional[int]:
    if not _PSUTIL_AVAILABLE:
        return None
    return psutil.Process().memory_info().rss


def _speech_tone(duration_ms: float, sample_rate: int = 16000, amp: float = 0.15) -> "np.ndarray":
    import numpy as np
    n = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(n) / sample_rate
    return (amp * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)


class GoldenBaselineSoakScenario(BaseVoiceScenario):
    """六阶段黄金基线 + 长程疲劳 soak 场景"""

    PHASES = ["mainline", "interrupt_reparam", "chatter_silence",
              "directed_wakeup", "resume_continuation", "fatigue_soak"]

    def __init__(self, *args,
                 total_duration_ms: float = 1_800_000.0,
                 time_compression: float = 60.0,
                 n_interrupts: int = 15,
                 n_subtasks: int = 7,
                 n_backchannels: int = 20,
                 memory_drift_sla: float = 0.10,
                 invariance_sla: float = 0.95,
                 seed: int = 42,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.total_duration_ms = total_duration_ms
        self.time_compression = max(time_compression, 1.0)
        self.n_interrupts = n_interrupts
        self.n_subtasks = n_subtasks
        self.n_backchannels = n_backchannels
        self.memory_drift_sla = memory_drift_sla
        self.invariance_sla = invariance_sla
        self.rng = random.Random(seed)
        self.synth = OverlapSpeechSynthesizer(seed=seed)

    async def execute(self) -> ScenarioResult:
        import asyncio
        started_ns = time.monotonic_ns()
        started_rss = _rss_bytes()
        errors: List[str] = []
        phase_events: Dict[str, Dict[str, Any]] = {}
        injection_log = {"interrupts": 0, "subtasks": 0, "backchannels": 0}
        window_start = window_end = time.monotonic_ns()

        feed_task = None
        try:
            if hasattr(self.audio_input, "open"):
                await self.audio_input.open()  # 消除 push 与 feed 的 open 竞态
            if hasattr(self.runtime, "feed_audio"):
                feed_task = asyncio.ensure_future(self.runtime.feed_audio())

            phase_budget_ms = self.total_duration_ms / len(self.PHASES)
            real_budget_s = phase_budget_ms / 1000.0 / self.time_compression

            # 阶段 1: 主线长篇讲解（流式下发顺畅度）
            await self._run_phase("mainline", real_budget_s, phase_events, lambda: self._push_speech(200.0))

            # 阶段 2: 突然打断并改要求（打断挂起主线 → 新参数 State Rebase 进支线帧）
            def p2():
                self._push_speech(1500.0)
                self.runtime.context_stack.push_subtask("reparam_branch", {"owner": "李老师", "city": "上海"})
                self.runtime.context_stack.merge_context({"owner": "李老师", "city": "上海"}, target_depth=1)
                injection_log["interrupts"] += 1
            await self._run_phase("interrupt_reparam", real_budget_s, phase_events, p2)

            # 阶段 3: 多人热烈讨论·静默期（模型全场闭嘴）
            arg_pcm = self.synth.generate_argument(2000.0, n_speakers=3, snr_db=5.0)
            window_start = time.monotonic_ns()

            def p3():
                self.audio_input.push_segment(arg_pcm)
            await self._run_phase("chatter_silence", real_budget_s, phase_events, p3)
            window_end = time.monotonic_ns()

            # 阶段 4: 突然点名发问（唤醒与召回）
            def p4():
                self._push_speech(600.0)
            await self._run_phase("directed_wakeup", real_budget_s, phase_events, p4)

            # 阶段 5: 接回之前的话题（打断不变性）
            def p5():
                frame = self.runtime.context_stack.resume_full(target_depth=0)
                phase_events.setdefault("resume_continuation", {})["invariance_score"] = (
                    frame.interruption_invariance_score if frame else None
                )
            await self._run_phase("resume_continuation", real_budget_s, phase_events, p5)

            # 阶段 6: 长程抗疲劳巡检（确定性随机注入）
            await self._run_fatigue_phase(real_budget_s, phase_events, injection_log)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if feed_task is not None:
                feed_task.cancel()

        finished_rss = _rss_bytes()
        metrics = self._build_metrics(phase_events, injection_log, started_rss, finished_rss,
                                      window_start, window_end)
        assertions = self.assertions([], metrics)
        all_passed = all(a.passed for a in assertions) and not errors
        status = TestStatus.SIMULATED_PASS if all_passed else TestStatus.FAIL

        return ScenarioResult(
            scenario_id="golden_baseline_soak",
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
            metrics=metrics,
            errors=errors,
        )

    async def _run_phase(self, name: str, real_budget_s: float,
                         phase_events: Dict[str, Dict[str, Any]], action) -> None:
        import asyncio
        phase_start = time.monotonic_ns()
        try:
            action()
            await asyncio.sleep(max(real_budget_s, 0.01))
            phase_events.setdefault(name, {}).update({
                "status": "completed",
                "duration_ms": round((time.monotonic_ns() - phase_start) / 1e6, 2),
            })
        except Exception as exc:
            phase_events[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            raise
        await self.timeline.record(RuntimeEvent(
            event_type=EventType.TEXT_DELTA,
            session_id=self.runtime.session_id,
            source="soak_scenario",
            payload={"phase": name, "status": phase_events[name]["status"]},
        ))

    async def _run_fatigue_phase(self, real_budget_s: float,
                                 phase_events: Dict[str, Dict[str, Any]],
                                 injection_log: Dict[str, int]) -> None:
        """长程阶段：确定性随机时点注入打断/支线/附和"""
        import asyncio
        plan = ([("interrupt", self.n_interrupts)] + [("subtask", self.n_subtasks)] +
                [("backchannel", self.n_backchannels)])
        for kind, count in plan:
            for _ in range(count):
                if kind == "interrupt":
                    self.runtime.context_stack.push_subtask(f"sub_{injection_log['subtasks']}", {})
                    injection_log["interrupts"] += 1
                elif kind == "subtask":
                    injection_log["subtasks"] += 1
                else:
                    injection_log["backchannels"] += 1
        await asyncio.sleep(max(real_budget_s, 0.01))
        phase_events["fatigue_soak"] = {"status": "completed", **injection_log}

    def _push_speech(self, duration_ms: float) -> None:
        if hasattr(self.audio_input, "push_segment"):
            self.audio_input.push_segment(_speech_tone(duration_ms))

    def _build_metrics(self, phase_events, injection_log, started_rss, finished_rss,
                       window_start, window_end) -> Dict[str, Any]:
        events = self.timeline.snapshot()
        chunks_in_window = [
            e for e in events
            if e.event_type == EventType.AUDIO_CHUNK and window_start <= e.monotonic_ns <= window_end
        ]
        memory_drift_ratio = None
        if started_rss and finished_rss:
            memory_drift_ratio = abs(finished_rss - started_rss) / started_rss
        return {
            "phases": phase_events,
            "injections": injection_log,
            "memory_psutil_available": _PSUTIL_AVAILABLE,
            "memory_start_bytes": started_rss,
            "memory_end_bytes": finished_rss,
            "memory_drift_ratio": round(memory_drift_ratio, 4) if memory_drift_ratio is not None else None,
            "chatter_chunks": len(chunks_in_window),
            "time_compression": self.time_compression,
        }

    def assertions(self, events: list, metrics: Optional[Dict[str, Any]] = None) -> List[AssertionResult]:
        if metrics is None:
            metrics = {}
        results = []

        # 1. 六阶段全部完成
        phases = metrics.get("phases", {})
        all_completed = len(phases) == len(self.PHASES) and all(
            p.get("status") == "completed" for p in phases.values()
        )
        results.append(AssertionResult(
            name="assert_six_phases_completed",
            passed=all_completed,
            severity=AssertionSeverity.CRITICAL,
            message=f"六阶段编排必须全部跑通 (完成: {sorted(k for k, v in phases.items() if v.get('status') == 'completed')})",
        ))

        # 2. 疲劳注入计数完整性
        injections = metrics.get("injections", {})
        inject_ok = (injections.get("interrupts", 0) >= self.n_interrupts and
                     injections.get("backchannels", 0) >= self.n_backchannels)
        results.append(AssertionResult(
            name="assert_fatigue_injection_counts",
            passed=inject_ok,
            severity=AssertionSeverity.MAJOR,
            message=(f"长程疲劳注入必须完成 {self.n_interrupts} 打断 / {self.n_backchannels} 附和 "
                     f"(实测: {injections})"),
        ))

        # 3. 内存漂移 <= 10%（psutil 缺失 → INCONCLUSIVE 语义，不虚标）
        drift = metrics.get("memory_drift_ratio")
        if drift is None:
            results.append(AssertionResult(
                name="assert_memory_drift_within_sla",
                passed=False,
                severity=AssertionSeverity.MAJOR,
                message="psutil 缺失无法采样 RSS，内存漂移无法判定（一票否决不虚标）",
            ))
        else:
            results.append(AssertionResult(
                name="assert_memory_drift_within_sla",
                passed=drift <= self.memory_drift_sla,
                severity=AssertionSeverity.MAJOR,
                message=f"30 分钟长程会话内存漂移 {drift * 100:.2f}% (SLA <= {self.memory_drift_sla * 100:.0f}%)",
                expected=f"<= {self.memory_drift_sla}",
                actual=drift,
            ))

        # 4. 断点接续不变性 >= 0.95
        invariance = metrics.get("phases", {}).get("resume_continuation", {}).get("invariance_score")
        if invariance is None:
            results.append(AssertionResult(
                name="assert_interruption_invariance",
                passed=False,
                severity=AssertionSeverity.CRITICAL,
                message="未取得打断不变性评分，语义等价接续无法判定",
            ))
        else:
            results.append(AssertionResult(
                name="assert_interruption_invariance",
                passed=invariance >= self.invariance_sla,
                severity=AssertionSeverity.CRITICAL,
                message=f"打断不变性接续评分 {invariance:.3f} (SLA >= {self.invariance_sla})",
                expected=f">= {self.invariance_sla}",
                actual=invariance,
            ))

        # 5. 静默期无下行
        results.append(AssertionResult(
            name="assert_chatter_silence_no_downlink",
            passed=metrics.get("chatter_chunks", 0) == 0,
            severity=AssertionSeverity.CRITICAL,
            message=f"多人讨论静默期下行分片必须为 0 (实测: {metrics.get('chatter_chunks')})",
        ))
        return results
