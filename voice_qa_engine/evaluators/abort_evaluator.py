#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 打断停算熔断断言评估器 (Inference Abort Evaluator)
对应评测规范 9.3 补充 SLA 门禁：
- 服务端推理引擎收到 CANCEL → 中止推理时延 <= 15ms
- 打断后废弃 Token 上限 <= 3 tokens（算力浪费 <= 30ms）
证据缺失时记 null → INCONCLUSIVE 语义，严禁虚标。
"""

from typing import List, Optional, Dict, Any

from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity

ABORT_SLA_MS = 15.0
MAX_DISCARDED_TOKENS = 3


def inference_abort_assertions(
    abort_ms: Optional[float],
    discarded_tokens: Optional[int],
    abort_sla_ms: float = ABORT_SLA_MS,
    max_discarded_tokens: int = MAX_DISCARDED_TOKENS,
) -> List[AssertionResult]:
    results: List[AssertionResult] = []

    if abort_ms is None:
        results.append(AssertionResult(
            name="assert_inference_abort_latency",
            passed=False,
            severity=AssertionSeverity.MAJOR,
            message="未观察到停算熔断时延物理证据（无 CANCEL→中止遥测），一票否决为不确定",
            expected=f"<= {abort_sla_ms}ms",
            actual=None,
        ))
    else:
        results.append(AssertionResult(
            name="assert_inference_abort_latency",
            passed=abort_ms <= abort_sla_ms,
            severity=AssertionSeverity.CRITICAL,
            message=f"服务端 CANCEL→推理中止时延 {abort_ms}ms (SLA <= {abort_sla_ms}ms)",
            expected=f"<= {abort_sla_ms}",
            actual=abort_ms,
        ))

    if discarded_tokens is None:
        results.append(AssertionResult(
            name="assert_discarded_tokens_within_cap",
            passed=False,
            severity=AssertionSeverity.MAJOR,
            message="未观察到废弃 Token 计数证据，算力浪费无法判定",
            expected=f"<= {max_discarded_tokens}",
            actual=None,
        ))
    else:
        results.append(AssertionResult(
            name="assert_discarded_tokens_within_cap",
            passed=discarded_tokens <= max_discarded_tokens,
            severity=AssertionSeverity.CRITICAL,
            message=f"打断后废弃 Token {discarded_tokens} (上限 <= {max_discarded_tokens})",
            expected=f"<= {max_discarded_tokens}",
            actual=discarded_tokens,
        ))
    return results


def extract_abort_metrics(ground_truth: Dict[str, Any]) -> (Optional[float], Optional[int]):
    """从 mock_server ground_truth_log 提取停算熔断遥测"""
    return ground_truth.get("inference_abort_ms"), ground_truth.get("discarded_tokens")
