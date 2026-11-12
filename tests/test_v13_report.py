#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1.3 大盘与自激扫描测试: AEC 边界扫描 / scorecard 聚合 / 防篡改联动 / 终端渲染"""

import os
import json
import tempfile
import unittest

from voice_qa_engine.evaluators.aec_evaluator import AECCollisionEvaluator
from voice_qa_engine.reporting.scorecard import build_scorecard, render_scorecard_text
from voice_qa_engine.evidence.recorder import EvidenceRecorder
from voice_qa_engine.evaluators.assertion_models import TestStatus


class TestSelfOscillationScan(unittest.TestCase):
    def test_ideal_aec_never_fails(self):
        scan = AECCollisionEvaluator().scan_self_oscillation_boundary(duration_ms=200.0)
        self.assertEqual(scan["boundary_summary"]["failing_points"], 0)
        self.assertEqual(scan["boundary_summary"]["total_points"],
                         len(scan["scan_grid"]))
        self.assertEqual(scan["boundary_summary"]["max_safe_gain_at_0ms"], 1.0)

    def test_weak_aec_exposes_boundary(self):
        # 弱 AEC（残余系数 0.1 ≈ 仅 20dB 抑制）→ ERLE=20dB < 40dB SLA → 全网格失效
        scan = AECCollisionEvaluator().scan_self_oscillation_boundary(
            duration_ms=100.0, residual_factor=0.1)
        self.assertGreater(scan["boundary_summary"]["failing_points"], 0)
        self.assertIsNone(scan["boundary_summary"]["max_safe_gain_at_0ms"])

    def test_grid_shape(self):
        scan = AECCollisionEvaluator().scan_self_oscillation_boundary(
            loopback_delays_ms=[0.0, 100.0], loopback_gains=[0.5], duration_ms=50.0)
        self.assertEqual(len(scan["scan_grid"]), 2)


def _write_session(base: str, sid: str, scenario_id: str, status: str,
                   metrics: dict, tamper: bool = False):
    recorder = EvidenceRecorder(base_dir=base, session_id=sid)
    recorder.session_dir = os.path.join(base, sid)
    os.makedirs(recorder.session_dir, exist_ok=True)
    # 手工落盘最小证据集（等价 finalize 产物结构）
    with open(os.path.join(recorder.session_dir, "input_audio.wav"), "wb") as f:
        import wave
        with wave.open(f, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 160)
    with open(os.path.join(recorder.session_dir, "output_audio.wav"), "wb") as f:
        import wave
        with wave.open(f, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 160)
    with open(os.path.join(recorder.session_dir, "events.jsonl"), "w", encoding="utf-8") as f:
        f.write("")
    with open(os.path.join(recorder.session_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False)
    with open(os.path.join(recorder.session_dir, "session.json"), "w", encoding="utf-8") as f:
        json.dump({"scenario_id": scenario_id, "status": status}, f, ensure_ascii=False)
    import hashlib
    from voice_qa_engine.evidence.manifest import generate_manifest
    manifest = generate_manifest(
        recorder.session_dir,
        ["input_audio.wav", "output_audio.wav", "events.jsonl", "metrics.json", "session.json"],
    )
    with open(os.path.join(recorder.session_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if tamper:
        with open(os.path.join(recorder.session_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"ttft_ms": 1.0}, f)  # 篡改后与 manifest 不符


class TestScorecard(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        _write_session(self.base, "s_normal", "normal_qa", "PASS",
                       {"ttft_ms": 420.0, "first_audio_latency_ms": 500.0})
        _write_session(self.base, "s_soak", "golden_baseline", "PASS",
                       {"ttft_ms": 800.0})  # P95 拉爆 → FAIL
        _write_session(self.base, "s_bad", "crisis_safety", "FAIL",
                       {"ttft_ms": 300.0})

    def test_build_aggregates_sla_matrix(self):
        card = build_scorecard(self.base)
        self.assertEqual(card["total_sessions"], 3)
        ttft = next(r for r in card["sla_matrix"] if r["metric"] == "ttft_ms")
        self.assertEqual(ttft["samples"], 3)
        self.assertEqual(ttft["verdict"], "FAIL")  # P95 = 800 > 600
        self.assertEqual(card["overall_status"], "FAIL")  # 含 FAIL 会话

    def test_status_distribution_per_scenario(self):
        card = build_scorecard(self.base)
        self.assertEqual(card["scenarios"]["normal_qa"]["statuses"], {"PASS": 1})
        self.assertEqual(card["scenarios"]["crisis_safety"]["statuses"], {"FAIL": 1})

    def test_tampered_session_flagged(self):
        _write_session(self.base, "s_tamper", "normal_qa", "PASS",
                       {"ttft_ms": 100.0}, tamper=True)
        card = build_scorecard(self.base)
        self.assertEqual(len(card["evidence_tampered_sessions"]), 1)
        self.assertIn("s_tamper", card["evidence_tampered_sessions"][0])

    def test_render_contains_matrix_and_verdicts(self):
        text = render_scorecard_text(build_scorecard(self.base))
        self.assertIn("TTFT", text)
        self.assertIn("FAIL", text)
        self.assertIn("crisis_safety", text)

    def test_empty_reports_root(self):
        empty = tempfile.mkdtemp()
        card = build_scorecard(empty)
        self.assertEqual(card["overall_status"], "NO_DATA")
        self.assertTrue(all(r["verdict"] == "NO_DATA" for r in card["sla_matrix"]))

    def test_negative_latency_filtered(self):
        _write_session(self.base, "s_neg", "normal_qa", "PASS", {"ttft_ms": -531.0})
        card = build_scorecard(self.base)
        ttft = next(r for r in card["sla_matrix"] if r["metric"] == "ttft_ms")
        self.assertNotIn(-531.0, [ttft["p50"], ttft["p95"]])  # 负值物理不可能 → 过滤


if __name__ == "__main__":
    unittest.main()
