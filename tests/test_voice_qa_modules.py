"""自动化单元测试套件: 校验声学质检、CER 计算、WebRTC 增量算法与全双工状态隔离"""

import unittest
import struct
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from acoustic_inspector import AcousticInspector
from noise_evaluator import NoiseEvaluator
from webrtc_getstats_inspector import WebRTCStatsInspector
from audio_window_mixer import AudioWindowMixer
from full_duplex_evaluator import (
    FullDuplexEvaluator,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_SIMULATED_PASS,
    STATUS_INCONCLUSIVE
)


class TestVoiceQAModules(unittest.TestCase):
    def setUp(self):
        self.inspector = AcousticInspector()
        self.noise_eval = NoiseEvaluator(default_seed=123)
        self.webrtc_inspector = WebRTCStatsInspector()
        self.fd_eval = FullDuplexEvaluator()

    def test_cer_calculation(self):
        """测试中文字错率编辑距离"""
        ref = "今天天气真好"
        hyp = "今天天气真好"
        res = NoiseEvaluator.calculate_cer(ref, hyp)
        self.assertEqual(res["cer"], 0.0)
        self.assertEqual(res["accuracy"], 1.0)

        hyp2 = "今天天气很好"  # 1 替换
        res2 = NoiseEvaluator.calculate_cer(ref, hyp2)
        self.assertEqual(res2["distance"], 1)
        self.assertAlmostEqual(res2["cer"], 1 / 6, places=3)

    def test_noise_rng_independence(self):
        """测试独立的随机数生成器不会污染全局 random 状态"""
        random.seed(999)
        pre_val = random.random()

        # 重新设定以观测独立性
        random.seed(999)
        _ = self.noise_eval.inject_noise_to_pcm([1000] * 100, snr_db=15.0)
        post_val = random.random()

        # 全局 random.random() 应完全一致，说明 noise_eval 未调用全局 random.seed()
        self.assertEqual(pre_val, post_val)

    def test_webrtc_delta_packet_calculation(self):
        """测试 WebRTC 单调递增计数器的增量差分计算，杜绝虚假累加"""
        # 正常无丢包样本
        samples_clean = [
            {"jitter": 0.005, "packetsReceived": 100, "packetsLost": 0, "audioLevel": 0.5},
            {"jitter": 0.006, "packetsReceived": 200, "packetsLost": 0, "audioLevel": 0.5},
            {"jitter": 0.005, "packetsReceived": 300, "packetsLost": 0, "audioLevel": 0.5}
        ]
        res = self.webrtc_inspector.parse_inbound_rtp_report(samples_clean, is_simulated=True)
        metrics = res["metrics"]
        # 总接收包应为 300 (100 + 100 + 100)，而不是 600
        self.assertEqual(metrics["total_packets_received"], 300)
        self.assertEqual(metrics["total_packets_lost"], 0)
        self.assertAlmostEqual(metrics["jitter_p50_ms"], 5.0, places=1)
        self.assertEqual(res["status"], STATUS_SIMULATED_PASS)

        # 出现丢包时应被硬性拦截
        samples_lossy = [
            {"jitter": 0.005, "packetsReceived": 100, "packetsLost": 0, "audioLevel": 0.5},
            {"jitter": 0.005, "packetsReceived": 200, "packetsLost": 2, "audioLevel": 0.5}
        ]
        res_lossy = self.webrtc_inspector.parse_inbound_rtp_report(samples_lossy, is_simulated=True)
        self.assertEqual(res_lossy["status"], STATUS_FAIL)
        self.assertEqual(res_lossy["metrics"]["total_packets_lost"], 2)

    def test_acoustic_unsupported_format(self):
        """测试对 MP3 等压缩音频的明确拒绝与不支持格式识别"""
        mp3_mock = b"ID3\x03\x00\x00\x00\x00\x00\x20" + b"\x00" * 50
        res = self.inspector.inspect_wav_bytes(mp3_mock)
        self.assertFalse(res["valid"])
        self.assertEqual(res["status"], "UNSUPPORTED_FORMAT")

    def test_full_duplex_state_isolation(self):
        """测试全双工时延断言的状态隔离体系：模拟数据绝不返回真实 PASS"""
        mock_events = {
            "user_speech_end_ts": 10.0,
            "first_text_event_ts": 10.4,
            "first_audio_chunk_ts": 10.5,
            "speaker_out_ts": 10.6
        }
        res_sim = self.fd_eval.evaluate_latency_breakdown(mock_events, is_simulated=True)
        self.assertEqual(res_sim["status"], STATUS_SIMULATED_PASS)
        self.assertEqual(res_sim["measurement_source"], "mock_events")

        res_real = self.fd_eval.evaluate_latency_breakdown(mock_events, is_simulated=False)
        self.assertEqual(res_real["status"], STATUS_PASS)
        self.assertEqual(res_real["measurement_source"], "real_event_timestamps")

    def test_audio_window_mixer_ducking(self):
        """测试加窗淡入淡出混音与压麦比"""
        mixer = AudioWindowMixer(frame_duration_ms=20, sample_rate=16000)
        base = [10000] * 1600  # 100ms
        bargein = [5000] * 800   # 50ms
        mixed = mixer.mix_bargein_at_sample(base, bargein, inject_sample_idx=800, duck_ratio=0.2, fade_ms=10)
        self.assertEqual(len(mixed), 1600)
        self.assertTrue(all(abs(m) <= 32767 for m in mixed))


if __name__ == "__main__":
    unittest.main()
