#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 场景执行抽象基类与结果结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from voice_qa_engine.evaluators.assertion_models import AssertionResult, TestStatus


@dataclass
class ScenarioResult:
    scenario_id: str
    status: TestStatus
    started_ns: int
    finished_ns: int
    events: List[Dict[str, Any]] = field(default_factory=list)
    assertions: List[AssertionResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class BaseVoiceScenario(ABC):
    def __init__(self, runtime, audio_input, playback_engine, timeline, evaluator=None):
        self.runtime = runtime
        self.audio_input = audio_input
        self.playback_engine = playback_engine
        self.timeline = timeline
        self.evaluator = evaluator

    @abstractmethod
    async def execute(self) -> ScenarioResult:
        """执行测试场景核心行为链"""
        pass

    @abstractmethod
    def assertions(self, events: list) -> List[AssertionResult]:
        """断言事件序列与物理指标"""
        pass
