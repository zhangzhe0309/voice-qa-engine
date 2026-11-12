#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 场景注册表与统一执行器 (Scenario Registry & Suite Runner)
统一调度场景生命周期：上下文装配 → 执行 → 证据落盘 → SHA-256 防篡改校验 → 汇总判定
"""

import time
import os
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, TestStatus
from voice_qa_engine.evaluators.evidence_validator import EvidenceValidator


@dataclass
class ScenarioContext:
    """单场景独立依赖包（每场景全新实例，防跨场景状态污染）"""
    runtime: Any
    audio_input: Any
    playback_engine: Any
    timeline: Any
    recorder: Any
    extras: Dict[str, Any]


class ScenarioRegistry:
    def __init__(self):
        self._scenarios: List[type] = []

    def register(self, scenario_cls: type) -> type:
        """注册场景类或场景工厂 callable(ctx_deps...) -> BaseVoiceScenario 实例"""
        is_class = isinstance(scenario_cls, type) and issubclass(scenario_cls, BaseVoiceScenario)
        if not is_class and not callable(scenario_cls):
            raise TypeError(f"{scenario_cls!r} 不是 BaseVoiceScenario 子类或场景工厂")
        self._scenarios.append(scenario_cls)
        return scenario_cls

    @property
    def scenario_classes(self) -> List[type]:
        return list(self._scenarios)


class SuiteRunner:
    """
    统一执行器。

    context_factory: async (scenario_id) -> ScenarioContext，由调用方装配每场景全新依赖。
    overall 判定优先级：FAIL > ERROR > INCONCLUSIVE > NOT_RUN > SIMULATED_PASS > PASS。
    """

    def __init__(
        self,
        suite_name: str,
        context_factory: Callable,
        version: str = "1.3.0",
    ):
        self.suite_name = suite_name
        self.context_factory = context_factory
        self.version = version

    async def run(self, registry: ScenarioRegistry) -> Dict[str, Any]:
        suite_started = time.time()
        results: List[Dict[str, Any]] = []

        for scenario_cls in registry.scenario_classes:
            scenario_id = self._scenario_id(scenario_cls)
            entry = await self._run_one(scenario_cls, scenario_id)
            results.append(entry)

        return {
            "suite": self.suite_name,
            "version": self.version,
            "timestamp": suite_started,
            "overall_status": self._overall_status(results),
            "scenarios": results,
        }

    async def _run_one(self, scenario_cls: type, scenario_id: str) -> Dict[str, Any]:
        ctx: Optional[ScenarioContext] = None
        try:
            ctx = await self.context_factory(scenario_id)
            await ctx.recorder.initialize()
            deps = dict(runtime=ctx.runtime, audio_input=ctx.audio_input,
                        playback_engine=ctx.playback_engine, timeline=ctx.timeline)
            scenario = scenario_cls(**deps)
            result: ScenarioResult = await scenario.execute()

            evidence_valid, evidence_reason = await self._finalize_and_validate_async(ctx, result)
            if not evidence_valid and result.status == TestStatus.PASS:
                # 关键证据缺失/被篡改 → 一票否决降级
                result.status = TestStatus.INCONCLUSIVE
                result.errors.append(f"evidence_invalid: {evidence_reason}")

            return self._to_report_entry(result, evidence_valid, evidence_reason)
        except Exception as exc:  # 场景崩溃不得拖垮整个套件
            return {
                "scenario_id": scenario_id,
                "status": TestStatus.ERROR,
                "assertions": [],
                "metrics": {},
                "errors": [f"{type(exc).__name__}: {exc}"],
                "evidence_valid": False,
                "evidence_reason": "context_or_execute_crashed",
            }
        finally:
            if ctx is not None:
                await self._safe_close(ctx)

    async def _finalize_and_validate_async(self, ctx: ScenarioContext, result: ScenarioResult) -> (bool, str):
        await ctx.recorder.finalize(
            metrics=dict(result.metrics),
            session_meta={
                "scenario_id": result.scenario_id,
                "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                "assertions": [self._assertion_dict(a) for a in result.assertions],
            },
        )
        session_dir = os.path.join(ctx.recorder.base_dir, ctx.recorder.session_id)
        valid, reason, _ = EvidenceValidator.validate_session_dir(session_dir)
        return valid, reason

    async def _safe_close(self, ctx: ScenarioContext):
        try:
            if hasattr(ctx.runtime, "close"):
                await ctx.runtime.close()
        except Exception:
            pass

    @staticmethod
    def _scenario_id(scenario_cls: type) -> str:
        return scenario_cls.__name__.lstrip("_").replace("Scenario", "").lower()

    @staticmethod
    def _assertion_dict(a: AssertionResult) -> Dict[str, Any]:
        return {
            "name": a.name,
            "passed": a.passed,
            "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
            "message": a.message,
        }

    @staticmethod
    def _to_report_entry(result: ScenarioResult, evidence_valid: bool, evidence_reason: str) -> Dict[str, Any]:
        return {
            "scenario_id": result.scenario_id,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "assertions": [SuiteRunner._assertion_dict(a) for a in result.assertions],
            "metrics": result.metrics,
            "errors": result.errors,
            "evidence_valid": evidence_valid,
            "evidence_reason": evidence_reason,
        }

    @staticmethod
    def _overall_status(results: List[Dict[str, Any]]) -> TestStatus:
        statuses = [r["status"] for r in results]
        if not statuses:
            return TestStatus.NOT_RUN
        for bad in (TestStatus.FAIL, TestStatus.ERROR, TestStatus.INCONCLUSIVE, TestStatus.NOT_RUN):
            if bad.value in statuses:
                return bad
        if all(s == TestStatus.SIMULATED_PASS.value for s in statuses):
            return TestStatus.SIMULATED_PASS
        return TestStatus.PASS
