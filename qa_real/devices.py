"""设备枚举与选择。

所有真实测量必须把设备指纹写进 session.json，否则跨机器结果不可比。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]


@dataclass
class DeviceInfo:
    index: int
    name: str
    host_api: str
    sample_rate: int
    in_ch: int
    out_ch: int
    is_loopback: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _host_api_name(p: Any, idx: int) -> str:
    try:
        return str(p.get_host_api_info_by_index(idx)["name"])
    except (RuntimeError, KeyError, TypeError, ValueError):
        return f"api{idx}"


def list_devices() -> list[DeviceInfo]:
    """枚举全部设备（含 WASAPI loopback）。"""
    if pyaudio is None:
        raise RuntimeError("pyaudiowpatch 未安装：pip install pyaudiowpatch")
    p = pyaudio.PyAudio()
    out: list[DeviceInfo] = []
    try:
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            out.append(
                DeviceInfo(
                    index=i,
                    name=str(d.get("name", "")),
                    host_api=_host_api_name(p, int(d.get("hostApi", 0))),
                    sample_rate=int(d.get("defaultSampleRate", 0) or 0),
                    in_ch=int(d.get("maxInputChannels", 0) or 0),
                    out_ch=int(d.get("maxOutputChannels", 0) or 0),
                    is_loopback=bool(d.get("isLoopbackDevice", False)),
                )
            )
    finally:
        p.terminate()
    return out


def find_loopback(name_contains: str | None = None, prefer_speaker: bool = True) -> DeviceInfo:
    """挑一个 WASAPI loopback 设备（回采系统渲染出的音频）。

    prefer_speaker=True 时优先选带物理扬声器的输出（"Speaker"），
    这样 loopback 与麦克风能听到同一个物理声源，声学路径才可测。
    """
    cands = [d for d in list_devices() if d.is_loopback and d.in_ch > 0]
    if name_contains:
        cands = [d for d in cands if name_contains.lower() in d.name.lower()]
    if not cands:
        raise RuntimeError("未找到 WASAPI loopback 设备（需 Windows + pyaudiowpatch）")
    if prefer_speaker:
        spk = [d for d in cands if "speaker" in d.name.lower() or "扬声器" in d.name.lower()]
        if spk:
            cands = spk
    # 优先 48k，便于与主流 TTS 下行采样率对齐
    cands.sort(key=lambda d: (d.sample_rate != 48000, d.index))
    return cands[0]


def find_mic(name_contains: str | None = None) -> DeviceInfo | None:
    """挑一个真实麦克风（用于声学路径测量；没有则返回 None）。"""
    try:
        cands = [
            d
            for d in list_devices()
            if d.in_ch > 0 and not d.is_loopback and "声音映射器" not in d.name and "主声音捕获" not in d.name
        ]
    except RuntimeError:
        return None
    if name_contains:
        cands = [d for d in cands if name_contains.lower() in d.name.lower()]
    if not cands:
        return None
    cands.sort(key=lambda d: (d.sample_rate != 48000, d.index))
    return cands[0]


def find_output(parent_of: DeviceInfo | None = None) -> DeviceInfo:
    """挑输出设备：默认取 loopback 设备名对应的物理输出。"""
    devs = [d for d in list_devices() if d.out_ch > 0 and d.host_api.lower().find("wasapi") >= 0]
    if not devs:
        devs = [d for d in list_devices() if d.out_ch > 0]
    if not devs:
        raise RuntimeError("未找到可用的输出设备")
    if parent_of is not None:
        core = parent_of.name.replace("[Loopback]", "").strip()
        for d in devs:
            if core and (core in d.name or d.name in core):
                return d
    devs.sort(key=lambda d: (d.sample_rate != 48000, d.index))
    return devs[0]


def probe_manifest() -> dict[str, Any]:
    """一次性拿到本次测量用到的全部设备指纹。"""
    lb = find_loopback()
    mic = find_mic()
    out = find_output(lb)
    return {
        "loopback": lb.to_dict(),
        "microphone": mic.to_dict() if mic else None,
        "output": out.to_dict(),
    }
