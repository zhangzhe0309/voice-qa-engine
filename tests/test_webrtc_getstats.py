#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC getStats 实时统计与音轨时序解析套件单元测试
"""

import unittest
from webrtc_getstats_inspector import WebRTCStatsInspector


class TestWebRTCStatsInspector(unittest.TestCase):
    def setUp(self):
        self.inspector = WebRTCStatsInspector(
            jitter_p95_threshold_ms=12.0,
            jitter_max_threshold_ms=15.0
        )

    def test_empty_samples_returns_inconclusive(self):
        res = self.inspector.parse_inbound_rtp_report([])
        self.assertEqual(res["status"], "INCONCLUSIVE")

    def test_valid_inbound_rtp_report_pass(self):
        # 模拟 3 个间隔的 getStats 样本点
        samples = [
            {
                "timestamp": 1000.0,
                "jitter": 0.005,  # 5ms
                "packetsReceived": 50,
                "packetsLost": 0,
                "audioLevel": 0.35,
                "totalAudioEnergy": 0.5,
                "roundTripTime": 0.040  # 40ms
            },
            {
                "timestamp": 1001.0,
                "jitter": 0.008,  # 8ms
                "packetsReceived": 100,
                "packetsLost": 0,
                "audioLevel": 0.40,
                "totalAudioEnergy": 1.0,
                "roundTripTime": 0.042
            },
            {
                "timestamp": 1002.0,
                "jitter": 0.006,  # 6ms
                "packetsReceived": 150,
                "packetsLost": 0,
                "audioLevel": 0.38,
                "totalAudioEnergy": 1.5,
                "roundTripTime": 0.041
            }
        ]
        res = self.inspector.parse_inbound_rtp_report(samples, is_simulated=True)
        self.assertIn(res["status"], ("PASS", "SIMULATED_PASS"))
        self.assertLessEqual(res["metrics"]["jitter_p95_ms"], 12.0)
        self.assertLessEqual(res["metrics"]["jitter_max_ms"], 15.0)
        self.assertEqual(res["metrics"]["total_packets_lost"], 0)

    def test_jitter_exceeds_threshold_fails(self):
        samples = [
            {
                "timestamp": 1000.0,
                "jitter": 0.025,  # 25ms > 12ms
                "packetsReceived": 50,
                "packetsLost": 0,
                "audioLevel": 0.3,
                "roundTripTime": 0.040
            },
            {
                "timestamp": 1001.0,
                "jitter": 0.030,  # 30ms > 15ms
                "packetsReceived": 100,
                "packetsLost": 0,
                "audioLevel": 0.3,
                "roundTripTime": 0.045
            }
        ]
        res = self.inspector.parse_inbound_rtp_report(samples, is_simulated=False)
        self.assertEqual(res["status"], "FAIL")
        self.assertFalse(res["gate_assertions"]["jitter_p95_within_sla"])


if __name__ == "__main__":
    unittest.main()
