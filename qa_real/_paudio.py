"""pyaudiowpatch 懒加载入口（WASAPI loopback 依赖它，缺失时给出明确报错）。"""

from __future__ import annotations

from typing import Any


def pa_module() -> Any:
    """返回 pyaudiowpatch 模块；未安装时抛出明确错误（而非 None 属性错误）。"""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyaudiowpatch 未安装：pip install pyaudiowpatch") from exc
    return pyaudio
