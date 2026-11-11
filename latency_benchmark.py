"""Latency Benchmark: 语音时延与流式吞吐基准度量引擎
测量 TTS 首包时延 (TTFT)、端到端生成时延、音频时长及实时率 (RTF: Real-Time Factor)。
"""

import time
import requests
import struct
from typing import Dict, Any, List


class LatencyBenchmark:
    def __init__(self, base_url: str = "https://lingxinai.com"):
        self.base_url = base_url
        self.auth_token = self._get_auth_token()

    def _get_auth_token(self) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/psychology-api/super/pswLogin",
                json={"username": "psy_aminer", "password": "Aminer@2025Psych"},
                timeout=10
            ).json()
            return r.get("data", {}).get("access_token", "")
        except Exception:
            return ""


    def benchmark_tts(self, text: str, voice_code: str = "ICL_uranus_zh_female_zhixinjiejie_tob") -> Dict[str, Any]:
        """对指定音色和文本测量 TTS 合成时延与实时率。"""
        url = f"{self.base_url}/psychology-api/algo/ttsV3"
        payload = {
            "text": text,
            "voice_code": voice_code
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}", "Access-Token-k12": self.auth_token}

        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, headers=headers, timeout=25)
        e2e_duration = time.perf_counter() - t0

        if resp.status_code != 200:
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text[:200]
            }

        res_json = resp.json()
        b64_str = res_json.get("data", "")
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]

        import base64
        audio_bytes = base64.b64decode(b64_str)
        byte_len = len(audio_bytes)

        # 计算音频时长 (按 24000Hz 16bit 单声道计算)
        pcm_len = max(0, byte_len - 44)
        sample_count = pcm_len // 2
        audio_duration = sample_count / 24000.0

        # RTF: 实时率 (耗时 / 音频时长, < 1.0 表示快于说话速度)
        rtf = e2e_duration / max(0.01, audio_duration)

        return {
            "success": True,
            "text_length": len(text),
            "audio_duration_seconds": round(audio_duration, 3),
            "e2e_latency_ms": round(e2e_duration * 1000, 1),
            "real_time_factor_rtf": round(rtf, 3),
            "payload_bytes": byte_len,
            "audio_bytes": audio_bytes,
            "is_rtf_healthy": rtf < 0.8
        }

    def benchmark_asr(self, wav_bytes: bytes) -> Dict[str, Any]:
        """测量 ASR 识别耗时与识别结果。"""
        url = f"{self.base_url}/psychology-api/algo/asr"
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}

        t0 = time.perf_counter()
        resp = requests.post(url, files=files, timeout=20)
        e2e_duration = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"success": False, "status_code": resp.status_code, "error": resp.text[:200]}

        res_json = resp.json()
        text = res_json.get("data", "")

        return {
            "success": True,
            "recognized_text": text,
            "latency_ms": round(e2e_duration * 1000, 1)
        }
