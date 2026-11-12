#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 确定性工具物理状态拦截器 (Tool Execution Ledger)
借鉴 ServiceNow EVA 闭环核验：对每笔"服务端声称完成的工具调用"与
"后端物理实际落盘的执行记录"做确定性比对，不一致一票否决（假完工缺陷）。
"""

from typing import Dict, Any, List, Optional

from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity


class ToolExecutionLedger:
    """工具调用"声称 vs 实际"物理状态账本"""

    def __init__(self):
        # call_id -> {"tool_name", "params", "claimed_ns"}
        self.claims: Dict[str, Dict[str, Any]] = {}
        # call_id -> {"tool_name", "executed", "params", "executed_ns"}
        self.executions: Dict[str, Dict[str, Any]] = {}

    def record_claim(self, call_id: str, tool_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """登记服务端声称发起/完成的工具调用（来自 response.tool_call 事件）"""
        self.claims[call_id] = {
            "tool_name": tool_name,
            "params": params or {},
            "claimed_ns": 0,
        }

    def record_execution(self, call_id: str, tool_name: str, executed: bool,
                         params: Optional[Dict[str, Any]] = None) -> None:
        """登记后端物理实际执行记录（来自 ground_truth_log / 后端审计）"""
        self.executions[call_id] = {
            "tool_name": tool_name,
            "executed": executed,
            "params": params or {},
            "executed_ns": 0,
        }

    def verify(self) -> Dict[str, Any]:
        """
        确定性比对：
        - 每笔声称必须存在 executed=True 的物理执行
        - 任何无声称的执行视为幽灵执行（同样可疑）
        - 声称了但 executed=False 即"假完工"
        """
        missing_executions: List[str] = []
        fake_completions: List[str] = []
        for call_id, claim in self.claims.items():
            exe = self.executions.get(call_id)
            if exe is None:
                missing_executions.append(call_id)
            elif not exe["executed"]:
                fake_completions.append(call_id)
            elif exe["tool_name"] != claim["tool_name"]:
                fake_completions.append(call_id)  # 落盘工具与声称不符

        unclaimed_executions: List[str] = [
            call_id for call_id in self.executions if call_id not in self.claims
        ]
        consistent = not (missing_executions or fake_completions or unclaimed_executions)
        return {
            "consistent": consistent,
            "claim_count": len(self.claims),
            "execution_count": len(self.executions),
            "missing_executions": missing_executions,
            "fake_completions": fake_completions,
            "unclaimed_executions": unclaimed_executions,
        }

    def assertions(self) -> List[AssertionResult]:
        report = self.verify()
        results = [
            AssertionResult(
                name="assert_tool_claims_physically_executed",
                passed=not report["missing_executions"] and not report["fake_completions"],
                severity=AssertionSeverity.CRITICAL,
                message=(
                    f"全部 {report['claim_count']} 笔声称调用必须与后端物理落盘一致 "
                    f"(缺失: {report['missing_executions']}, 假完工: {report['fake_completions']})"
                ),
                expected="claim == execution",
                actual=report,
            )
        ]
        if report["unclaimed_executions"]:
            results.append(AssertionResult(
                name="assert_no_ghost_tool_executions",
                passed=False,
                severity=AssertionSeverity.MAJOR,
                message=f"发现无声称的幽灵工具执行: {report['unclaimed_executions']}",
                expected="0",
                actual=report["unclaimed_executions"],
            ))
        return results
