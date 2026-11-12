#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - Virtual User Agent (VUA)
双 Agent 对弈中的"虚拟用户"驱动器。
- LLMPoweredVUA: OpenAI 兼容接口动态生成用户话术（真实模式，key 缺失 → 明确 NOT_RUN，严禁伪造）
- ScriptedVUA   : 显式脚本化用户（显式 SIMULATED_PASS 仿真模式，禁止冒充真机结果）
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VirtualUserAgent(ABC):
    mode: str = "abstract"

    @abstractmethod
    async def next_utterance(self, history: List[Dict[str, str]], task_goal: str) -> str:
        """根据对话历史与任务目标生成下一轮用户话术"""

    @abstractmethod
    def is_available(self) -> bool:
        pass


class LLMPoweredVUA(VirtualUserAgent):
    """LLM 驱动虚拟用户（OpenAI 兼容 Chat Completions）"""

    mode = "llm"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.8,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("VQA_LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("VQA_LLM_BASE_URL", "")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("VQA_LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def next_utterance(self, history: List[Dict[str, str]], task_goal: str) -> str:
        if not self.is_available():
            raise RuntimeError("LLM key 缺失 (VQA_LLM_API_KEY)：对弈引擎拒绝伪造真实用户话术")

        system_prompt = (
            "你是全双工语音评测中的虚拟用户。请严格围绕任务目标，用一句口语化中文短句推进对话，"
            "可以自然地修改参数、追问、附和或打断。禁止输出旁白、引号或解释，只输出你要说的话。"
        )
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"任务目标: {task_goal}"}]
        messages.extend(history[-12:])

        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 120,
        }
        resp = await asyncio.to_thread(
            requests.post,
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip().strip("\"'“”")


class ScriptedVUA(VirtualUserAgent):
    """显式脚本化虚拟用户：话术队列耗尽后复用队尾话术"""

    mode = "scripted"

    def __init__(self, script: List[str]):
        if not script:
            raise ValueError("ScriptedVUA 需要至少一句脚本话术")
        self.script = list(script)
        self._cursor = 0

    def is_available(self) -> bool:
        return True

    async def next_utterance(self, history: List[Dict[str, str]], task_goal: str) -> str:
        utterance = self.script[min(self._cursor, len(self.script) - 1)]
        self._cursor += 1
        return utterance
