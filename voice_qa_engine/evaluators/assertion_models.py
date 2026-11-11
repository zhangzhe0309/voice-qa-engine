#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 六态断言与测试判定模型
严格遵循防虚假全绿门禁，缺失关键物理证据一票否决降级为 INCONCLUSIVE
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Any


class AssertionSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    INCONCLUSIVE = "INCONCLUSIVE"
    SIMULATED_PASS = "SIMULATED_PASS"
    NOT_RUN = "NOT_RUN"


@dataclass
class AssertionResult:
    name: str
    passed: bool
    severity: AssertionSeverity = AssertionSeverity.CRITICAL
    message: str = ""
    expected: Any = None
    actual: Any = None
    evidence_event_ids: List[str] = field(default_factory=list)
