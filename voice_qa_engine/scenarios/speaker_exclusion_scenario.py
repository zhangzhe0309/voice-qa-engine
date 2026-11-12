#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 声纹排他与多人讨论拒识场景 (Speaker Exclusion / Voiceprint Exclusion)
对应评测规范 场景 5 + 指标 4.5/4.6：
1. 背景多人争吵 (SNR 5dB) 持续注入期间：下行推流增量 0 字节、误插话 0 次
2. 争吵中定向点名唤醒：首包时延 <= 600ms + 争议焦点召回率
判定纪律：进程内仿真服务端 → SIMULATED_PASS；仅真实链路证据全绿 → PASS
"""

import time
import wave
import os
from typing import List, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.audio.overlap_speech import OverlapSpeechSynthesizer


class SpeakerExclusionScenario(BaseVoiceScenario):
    """背景争吵拒识 + 定向唤醒召回场景"""

    def __init__(self, *args,
                 background_duration_ms: float = 7000.0,
                 n_speakers: int = 2,
                 snr_db: float = 5.0,
                 wakeup_latency_sla_ms: float = 600.0,
                 expected_keyword: str = "李老师",
                 seed: int = 42,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.background_duration_ms = background_duration_ms
        self.n_speakers = n_speakers
        self.snr_db = snr_db
        self.wakeup_latency_sla_ms = wakeup_latency_sla_ms
        self.expected_keyword = expected_keyword
        self.synth = OverlapSpeechSynthesizer(seed=seed)

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        errors: List[str] = []

        # 1. 合成并注入背景争吵音频（经真实上行链路推流）
        background = self.synth.generate_argument(
            self.background_duration_ms, n_speakers=self.n_speakers, snr_db=self.snr_db
        )
        window_start_ns = time.monotonic_ns()
        try:
            if hasattr(self.audio_input, "open"):
                await self.audio_input.open()  # 消除 push 与 feed 的 open 竞态
            self.audio_input.push_segment(background)
            feed_task = None
            if hasattr(self.runtime, "feed_audio"):
                import asyncio
                feed_task = asyncio.ensure_future(self.runtime.feed_audio())
                await asyncio.sleep(min(self.background_duration_ms / 1000.0, 2.0))
                feed_task.cancel()
        except Exception as exc:
            errors.append(f"background_injection: {type(exc).__name__}: {exc}")
        window_end_ns = time.monotonic_ns()

        # 2. 定向唤醒子场景：争吵中点名提问 → 服务端响应（事件由真实链路/桩注入 timeline）
        wakeup_metrics = self._wakeup_metrics()

        events = self.timeline.snapshot()
        metrics = self._calculate_metrics(events, window_start_ns, window_end_ns, wakeup_metrics)
        assertions = self.assertions(events, metrics)
        status = self._decide_status(assertions, metrics)

        return ScenarioResult(
            scenario_id="speaker_exclusion_rejection",
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
            metrics=metrics,
            errors=errors,
        )

    def _wakeup_metrics(self) -> Dict[str, Any]:
        """定向唤醒指标：默认无真实唤醒事件 → INCONCLUSIVE 语义（None），严禁虚标"""
        return {"wakeup_observed": False, "wakeup_latency_ms": None, "recall_hit": None}

    def _calculate_metrics(self, events: list, window_start_ns: int, window_end_ns: int,
                           wakeup_metrics: Dict[str, Any]) -> Dict[str, Any]:
        chunks_in_window = [
            e for e in events
            if e.event_type == EventType.AUDIO_CHUNK
            and window_start_ns <= e.monotonic_ns <= window_end_ns
        ]
        bytes_in_window = sum(
            int(e.payload.get("bytes_length", 0) or 0) for e in chunks_in_window
        )
        return {
            "background_window_ms": round((window_end_ns - window_start_ns) / 1e6, 2),
            "n_speakers": self.n_speakers,
            "snr_db": self.snr_db,
            "audio_chunks_in_window": len(chunks_in_window),
            "downlink_bytes_in_window": bytes_in_window,
            "wakeup_observed": wakeup_metrics["wakeup_observed"],
            "wakeup_latency_ms": wakeup_metrics["wakeup_latency_ms"],
            "recall_hit": wakeup_metrics["recall_hit"],
        }

    def assertions(self, events: list, metrics: Optional[Dict[str, Any]] = None) -> List[AssertionResult]:
        if metrics is None:
            metrics = {}
        results = []

        # 1. 争吵期间下行 0 字节（核心 SLA 4.5）
        results.append(AssertionResult(
            name="assert_zero_downlink_during_chatter",
            passed=metrics.get("downlink_bytes_in_window", 0) == 0,
            severity=AssertionSeverity.CRITICAL,
            message=(
                f"旁人争吵窗口 ({metrics.get('background_window_ms')}ms) 内下行推流增量必须为 0 字节 "
                f"(实测: {metrics.get('downlink_bytes_in_window')} 字节 / "
                f"{metrics.get('audio_chunks_in_window')} 分片)"
            ),
            expected=0,
            actual=metrics.get("downlink_bytes_in_window"),
        ))

        # 2. 误插话 0 次
        results.append(AssertionResult(
            name="assert_zero_false_bargein",
            passed=metrics.get("audio_chunks_in_window", 0) == 0,
            severity=AssertionSeverity.CRITICAL,
            message="争吵期间模型必须全场闭嘴当听众，误插话次数严格为 0",
            expected=0,
            actual=metrics.get("audio_chunks_in_window"),
        ))

        # 3. 定向唤醒（观察到才判定；未观察到 → INCONCLUSIVE 不虚标）
        if metrics.get("wakeup_observed"):
            lat = metrics.get("wakeup_latency_ms")
            results.append(AssertionResult(
                name="assert_directed_wakeup_latency",
                passed=lat is not None and lat <= self.wakeup_latency_sla_ms,
                severity=AssertionSeverity.CRITICAL,
                message=f"点名唤醒首包时延 {lat}ms (SLA <= {self.wakeup_latency_sla_ms}ms)",
                expected=f"<= {self.wakeup_latency_sla_ms}",
                actual=lat,
            ))
        else:
            results.append(AssertionResult(
                name="assert_directed_wakeup_evidence_present",
                passed=False,
                severity=AssertionSeverity.MAJOR,
                message="未观察到定向唤醒物理证据（唤醒时延/召回率无法判定），一票否决为不确定",
            ))
        return results

    @staticmethod
    def _decide_status(assertions: List[AssertionResult], metrics: Dict[str, Any]) -> TestStatus:
        core_fail = any(
            not a.passed and a.name in ("assert_zero_downlink_during_chatter",
                                        "assert_zero_false_bargein")
            for a in assertions
        )
        if core_fail:
            return TestStatus.FAIL
        if not metrics.get("wakeup_observed"):
            return TestStatus.INCONCLUSIVE
        if all(a.passed for a in assertions):
            return TestStatus.SIMULATED_PASS
        return TestStatus.FAIL
