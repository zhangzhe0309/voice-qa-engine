#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 比率类指标统计置信框架 (Statistical Confidence)
误抢话率 <=1.5%、附和成功率 >=98.5% 等比率 SLA 在小样本下无判定意义：
- Wilson 置信区间替代单点值
- 样本量 < min_samples → INCONCLUSIVE（一票否决，严禁虚报通过率）
"""

import math
from dataclasses import dataclass
from typing import Optional

from voice_qa_engine.evaluators.assertion_models import AssertionResult, AssertionSeverity


@dataclass
class RatioVerdict:
    name: str
    successes: int
    n: int
    observed_rate: float
    wilson_low: float
    wilson_high: float
    sla: float
    higher_is_better: bool
    min_samples: int
    passed: bool
    inconclusive: bool
    message: str


def wilson_interval(successes: int, n: int, z: float = 1.96) -> (float, float):
    """Wilson score 置信区间（默认 95%）"""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def ratio_verdict(
    name: str,
    successes: int,
    n: int,
    sla: float,
    higher_is_better: bool = True,
    min_samples: int = 200,
    z: float = 1.96,
) -> RatioVerdict:
    """
    比率 SLA 判定：
    - n < min_samples → inconclusive=True（不判定，防小样本虚报）
    - 通过性看 Wilson 边界：higher_is_better 取置信下界，否则取置信上界（最不利于系统的边界）
    """
    observed = (successes / n) if n > 0 else 0.0
    low, high = wilson_interval(successes, n, z)
    inconclusive = n < min_samples

    if inconclusive:
        passed = False
        message = (f"样本量 {n} < 最低要求 {min_samples}，比率 '{name}' 无法统计判定"
                   f"（观测 {observed * 100:.2f}%，Wilson 95% CI [{low * 100:.2f}%, {high * 100:.2f}%]），"
                   f"一票否决为不确定，严禁小样本虚报 SLA 达成")
    else:
        critical_bound = low if higher_is_better else high
        passed = (critical_bound >= sla) if higher_is_better else (critical_bound <= sla)
        direction = "下界" if higher_is_better else "上界"
        message = (f"比率 '{name}': 观测 {observed * 100:.2f}%，Wilson 95% CI "
                   f"[{low * 100:.2f}%, {high * 100:.2f}%]，以{direction}对比 SLA "
                   f"{'≥' if higher_is_better else '≤'} {sla * 100:.2f}% → "
                   f"{'通过' if passed else '未通过'} (n={n})")

    return RatioVerdict(
        name=name, successes=successes, n=n, observed_rate=observed,
        wilson_low=low, wilson_high=high, sla=sla, higher_is_better=higher_is_better,
        min_samples=min_samples, passed=passed, inconclusive=inconclusive, message=message,
    )


def assertion_from_verdict(
    verdict: RatioVerdict,
    assertion_name: str,
    severity: AssertionSeverity = AssertionSeverity.CRITICAL,
) -> AssertionResult:
    return AssertionResult(
        name=assertion_name,
        passed=verdict.passed,
        severity=severity,
        message=verdict.message,
        expected=(f"{'≥' if verdict.higher_is_better else '≤'} {verdict.sla} (n ≥ {verdict.min_samples})"),
        actual={
            "observed_rate": round(verdict.observed_rate, 4),
            "wilson_95_ci": [round(verdict.wilson_low, 4), round(verdict.wilson_high, 4)],
            "n": verdict.n,
            "inconclusive": verdict.inconclusive,
        },
    )
