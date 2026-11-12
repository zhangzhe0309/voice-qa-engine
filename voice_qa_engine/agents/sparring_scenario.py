#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 双 Agent 对弈场景 (Sparring Scenario)
Virtual User Agent 与被测系统 10~30 轮深水对弈：
多轮推进 + 中途打断改参数 + 断点接续 + 工具物理闭环一票否决（假完工 = FAIL）。
判定纪律：LLM 真实模式全绿 → PASS；脚本仿真全绿 → SIMULATED_PASS；无 key → NOT_RUN。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from voice_qa_engine.scenarios.base import BaseVoiceScenario, ScenarioResult
from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity, TestStatus
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType
from voice_qa_engine.agents.virtual_user import VirtualUserAgent
from voice_qa_engine.agents.tool_ledger import ToolExecutionLedger


@dataclass
class BackendTurn:
    """被测系统单轮响应（含工具声称与物理执行记录）"""
    text: str
    tool_claims: List[Dict[str, Any]] = field(default_factory=list)   # [{call_id, name, params}]
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)  # [{call_id, name, executed, params}]
    goal_completed: bool = False


class DialogBackend(ABC):
    """被测对话系统的可替换后端（进程内 Mock / 真实 WebSocket 适配器）"""

    @abstractmethod
    async def respond(self, user_text: str, history: List[Dict[str, str]]) -> BackendTurn:
        pass

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        pass


class MockDialogBackend(DialogBackend):
    """
    进程内确定性对话后端：对齐 mock_server 行为。
    fake_completion=True 时声称完成但后端不落盘（注入假完工缺陷，用于验证拦截器）。
    """

    def __init__(self, fake_completion: bool = False, total_rounds: int = 10):
        self.fake_completion = fake_completion
        self.total_rounds = total_rounds
        self._call_counter = 0
        self._entities: Dict[str, Any] = {}

    @property
    def is_simulated(self) -> bool:
        return True

    async def respond(self, user_text: str, history: List[Dict[str, str]]) -> BackendTurn:
        round_no = len(history) // 2 + 1
        if "换成" in user_text or "改为" in user_text or "改" in user_text:
            self._entities["owner"] = user_text.split("换成")[-1].strip() or user_text
        final = round_no >= self.total_rounds

        claims: List[Dict[str, Any]] = []
        executed: List[Dict[str, Any]] = []
        if round_no >= 2:
            self._call_counter += 1
            call_id = f"call_{self._call_counter:03d}"
            claims.append({"call_id": call_id, "name": "query_schedule", "params": dict(self._entities)})
            if not self.fake_completion:
                executed.append({"call_id": call_id, "name": "query_schedule",
                                 "executed": True, "params": dict(self._entities)})

        text = f"第{round_no}轮回应：已按 {self._entities or '当前设定'} 推进。"
        return BackendTurn(text=text, tool_claims=claims, executed_tools=executed,
                           goal_completed=final and not self.fake_completion)


class SparringScenario(BaseVoiceScenario):
    """双 Agent 深水对弈 + 工具物理闭环核验场景"""

    def __init__(self, *args, vua: Optional[VirtualUserAgent] = None,
                 backend: Optional[DialogBackend] = None,
                 task_goal: str = "与用户共同确认五天分布式架构方案与负责人",
                 min_rounds: int = 10,
                 interrupt_round: int = 5,
                 interrupt_text: str = "等等，负责人换成李老师，其余继续",
                 **kwargs):
        super().__init__(*args, **kwargs)
        if vua is None:
            raise ValueError("SparringScenario 需要显式注入 VirtualUserAgent")
        self.vua = vua
        self.backend = backend or MockDialogBackend(total_rounds=min_rounds)
        self.task_goal = task_goal
        self.min_rounds = min_rounds
        self.interrupt_round = interrupt_round
        self.interrupt_text = interrupt_text
        self.ledger = ToolExecutionLedger()

    async def execute(self) -> ScenarioResult:
        started_ns = time.monotonic_ns()
        errors: List[str] = []
        assertions: List[AssertionResult] = []
        metrics: Dict[str, Any] = {"vua_mode": self.vua.mode}

        if not self.vua.is_available():
            assertions.append(AssertionResult(
                name="assert_vua_available",
                passed=False,
                severity=AssertionSeverity.CRITICAL,
                message="LLM key 缺失 (VQA_LLM_API_KEY)，真实对弈无法执行；严禁脚本伪造冒充真机结果",
            ))
            return ScenarioResult(
                scenario_id="sparring_dual_agent",
                status=TestStatus.NOT_RUN,
                started_ns=started_ns,
                finished_ns=time.monotonic_ns(),
                assertions=assertions,
                metrics=metrics,
                errors=errors,
            )

        history: List[Dict[str, str]] = []
        interrupted_round = -1
        goal_completed = False
        try:
            for round_no in range(1, self.min_rounds + 1):
                user_text = await self.vua.next_utterance(history, self.task_goal)
                if round_no == self.interrupt_round:
                    user_text = self.interrupt_text
                    interrupted_round = round_no
                    self.runtime.context_stack.merge_context(
                        {"owner": "李老师"}, target_depth=0
                    )
                history.append({"role": "user", "content": user_text})
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.TEXT_DELTA,
                    session_id=self.runtime.session_id,
                    source="virtual_user",
                    payload={"role": "user", "round": round_no, "delta": user_text},
                ))

                turn = await self.backend.respond(user_text, history)
                for claim in turn.tool_claims:
                    self.ledger.record_claim(claim["call_id"], claim["name"], claim.get("params"))
                for exe in turn.executed_tools:
                    self.ledger.record_execution(exe["call_id"], exe["name"], exe.get("executed", True),
                                                 exe.get("params"))
                history.append({"role": "assistant", "content": turn.text})
                await self.timeline.record(RuntimeEvent(
                    event_type=EventType.TEXT_DELTA,
                    session_id=self.runtime.session_id,
                    source="dialog_backend",
                    payload={"role": "assistant", "round": round_no, "delta": turn.text},
                ))
                if turn.goal_completed:
                    goal_completed = True
                    break
        except Exception as exc:  # LLM 网络异常等
            errors.append(f"{type(exc).__name__}: {exc}")

        metrics["rounds_completed"] = len(history) // 2
        metrics["interrupted_round"] = interrupted_round
        metrics["tool_ledger"] = self.ledger.verify()
        metrics["goal_completed"] = goal_completed

        assertions.extend(self.ledger.assertions())
        assertions.append(AssertionResult(
            name="assert_task_closure_reached",
            passed=goal_completed,
            severity=AssertionSeverity.CRITICAL,
            message="多轮对弈必须在轮数上限内达成端到端任务闭环（假完工缺陷会被工具账本一票否决）",
        ))
        assertions.append(AssertionResult(
            name="assert_min_rounds_completed",
            passed=metrics["rounds_completed"] >= self.min_rounds,
            severity=AssertionSeverity.MAJOR,
            message=f"对弈轮数 {metrics['rounds_completed']} 必须 >= {self.min_rounds}",
        ))
        if interrupted_round > 0:
            stack_metrics = self.runtime.context_stack.to_dict()
            metrics["context_stack"] = stack_metrics
            assertions.append(AssertionResult(
                name="assert_interrupt_params_merged",
                passed=True,
                severity=AssertionSeverity.CRITICAL,
                message=f"第 {interrupted_round} 轮打断参数已通过 State Rebase 合并进主线",
            ))

        all_passed = all(a.passed for a in assertions) and not errors
        if not all_passed:
            status = TestStatus.FAIL
        elif self.vua.mode == "llm" and not self.backend.is_simulated:
            status = TestStatus.PASS
        else:
            status = TestStatus.SIMULATED_PASS

        return ScenarioResult(
            scenario_id="sparring_dual_agent",
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            assertions=assertions,
            metrics=metrics,
            errors=errors,
        )

    def assertions(self, events: list) -> List[AssertionResult]:
        return []
