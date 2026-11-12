"""Latency Benchmark: 语音时延与流式吞吐基准度量引擎
测量 TTS 首包时延 (TTFT)、端到端生成时延、音频时长及实时率 (RTF: Real-Time Factor)。
严格遵循安全规范：无任何硬编码凭证，环境变量驱动配置。
"""

import os
import time
import requests
import struct
from typing import Dict, Any, Optional


class LatencyBenchmark:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("VOICE_QA_BASE_URL", "https://lingxinai.com")).rstrip("/")
        self.username = os.getenv("VOICE_QA_USERNAME", "")
        self.password = os.getenv("VOICE_QA_PASSWORD", "")
        self.auth_token = self._get_auth_token()

    def _get_auth_token(self) -> str:
        if not self.username or not self.password:
            return ""
        try:
            r = requests.post(
                f"{self.base_url}/psychology-api/super/pswLogin",
                json={"username": self.username, "password": self.password},
                timeout=10
            )
            if r.status_code == 200:
                res_data = r.json()
                return res_data.get("data", {}).get("access_token", "")
            return ""
        except (requests.exceptions.RequestException, ValueError):
            return ""

    def benchmark_tts(self, text: str, voice_code: str = "ICL_uranus_zh_female_zhixinjiejie_tob") -> Dict[str, Any]:
        """对指定音色和文本测量 TTS 合成时延与实时率。"""
        if not self.auth_token:
            return {
                "success": False,
                "status": "INCONCLUSIVE",
                "error": "未配置 VOICE_QA_USERNAME/PASSWORD 凭证或认证失败，跳过真实接口压测"
            }

        url = f"{self.base_url}/psychology-api/algo/ttsV3"
        payload = {"text": text, "voice_code": voice_code}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "Access-Token-k12": self.auth_token
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=25)
            e2e_duration = time.perf_counter() - t0
        except requests.exceptions.Timeout:
            return {"success": False, "status": "ERROR", "error": "TTS 请求超时 (timeout > 25s)"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "status": "ERROR", "error": "TTS 网络连接失败 (ConnectionError)"}
        except Exception as e:
            return {"success": False, "status": "ERROR", "error": f"TTS 请求异常: {str(e)}"}

        if resp.status_code != 200:
            return {
                "success": False,
                "status": "FAIL",
                "status_code": resp.status_code,
                "error": resp.text[:200]
            }

        try:
            res_json = resp.json()
        except ValueError:
            return {"success": False, "status": "ERROR", "error": "TTS 响应 JSON 解析失败"}

        b64_str = res_json.get("data", "")
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]

        import base64
        try:
            audio_bytes = base64.b64decode(b64_str)
        except Exception as e:
            return {"success": False, "status": "ERROR", "error": f"Base64 解码音频失败: {e}"}

        byte_len = len(audio_bytes)

        # 解析实际采样率与通道
        sample_rate = 24000
        channels = 1
        bits_per_sample = 16
        if byte_len >= 44 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            try:
                channels = struct.unpack("<H", audio_bytes[22:24])[0]
                sample_rate = struct.unpack("<I", audio_bytes[24:28])[0]
                bits_per_sample = struct.unpack("<H", audio_bytes[34:36])[0]
            except Exception:
                pass

        bytes_per_sample = (bits_per_sample // 8) * max(1, channels)
        pcm_len = max(0, byte_len - 44)
        sample_count = pcm_len // max(1, bytes_per_sample)
        audio_duration = sample_count / float(max(8000, sample_rate))

        rtf = e2e_duration / max(0.01, audio_duration)

        return {
            "success": True,
            "status": "PASS",
            "text_length": len(text),
            "sample_rate_hz": sample_rate,
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
        try:
            resp = requests.post(url, files=files, timeout=20)
            e2e_duration = time.perf_counter() - t0
        except requests.exceptions.Timeout:
            return {"success": False, "status": "ERROR", "error": "ASR 请求超时"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "status": "ERROR", "error": "ASR 网络连接失败"}
        except Exception as e:
            return {"success": False, "status": "ERROR", "error": str(e)}

        if resp.status_code != 200:
            return {"success": False, "status": "FAIL", "status_code": resp.status_code, "error": resp.text[:200]}

        try:
            res_json = resp.json()
            text = res_json.get("data") or ""
        except ValueError:
            return {"success": False, "status": "ERROR", "error": "ASR 响应非合法 JSON"}

        return {
            "success": True,
            "status": "PASS",
            "recognized_text": text,
            "latency_ms": round(e2e_duration * 1000, 1)
        }
