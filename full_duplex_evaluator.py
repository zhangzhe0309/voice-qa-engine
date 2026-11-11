"""Full-Duplex Realtime Voice Interaction Evaluator (全双工语音交互端到端质量评估引擎)
融合 MOSS-Live、FDB v1~v3、ServiceNow EVA 与 voice-agent-barge-in-tests 核心能力:
1. 四段物理时延测量模型 (TTFT, First Audio Packet, EOT-to-Speech, Barge-in Stop)
2. 播放回执状态机与待播队列物理清空断言 (AudioContext Queue Flush & playback.chunk.completed)
3. 打断行为三阶对抗评测 (正向打断截断, 附和词免打断, 思考停顿防误抢)
4. 双 Agent 对弈驱动与任务物理闭环核验
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional

class FullDuplexEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "max_barge_in_stop_latency_ms": 300.0,
            "max_ttft_ms": 600.0,
            "max_first_audio_ms": 700.0,
            "min_backchannel_precision": 0.98,
            "max_premature_interruption_rate": 0.015,
            "audio_sample_rate": 16000,
            "frame_duration_ms": 20
        }

    def evaluate_latency_breakdown(self, events: Dict[str, Any]) -> Dict[str, Any]:
        user_end = events.get("user_speech_end_ts", 0)
        upload_start = events.get("upload_start_ts", user_end)
        first_text = events.get("first_text_event_ts", 0)
        first_audio = events.get("first_audio_chunk_ts", 0)
        speaker_out = events.get("speaker_out_ts", 0)

        ttft_ms = (first_text - user_end) * 1000 if first_text and user_end else 0.0
        first_audio_ms = (first_audio - user_end) * 1000 if first_audio and user_end else 0.0
        eot_to_speech_ms = (speaker_out - user_end) * 1000 if speaker_out and user_end else 0.0
        client_buffer_lag_ms = (speaker_out - first_audio) * 1000 if speaker_out and first_audio else 0.0

        is_ttft_pass = ttft_ms <= self.config["max_ttft_ms"]
        is_first_audio_pass = first_audio_ms <= self.config["max_first_audio_ms"]
        is_eot_pass = eot_to_speech_ms <= (self.config["max_first_audio_ms"] + 150.0)

        return {
            "metrics": {
                "ttft_ms": round(ttft_ms, 2),
                "first_audio_packet_latency_ms": round(first_audio_ms, 2),
                "eot_to_speech_latency_ms": round(eot_to_speech_ms, 2),
                "client_playback_pipeline_lag_ms": round(client_buffer_lag_ms, 2)
            },
            "sla": {
                "ttft_target_ms": self.config["max_ttft_ms"],
                "first_audio_target_ms": self.config["max_first_audio_ms"]
            },
            "status": "成功" if (is_ttft_pass and is_first_audio_pass and is_eot_pass) else "失败"
        }

    def evaluate_barge_in_and_playback_flush(
        self,
        barge_in_start_ts: float,
        interrupted_signal_ts: float,
        last_audible_frame_ts: float,
        pending_queue_size_after_flush: int,
        unconsumed_buffer_ms: float
    ) -> Dict[str, Any]:
        stop_latency_ms = (last_audible_frame_ts - barge_in_start_ts) * 1000
        signaling_latency_ms = (interrupted_signal_ts - barge_in_start_ts) * 1000
        client_eviction_latency_ms = (last_audible_frame_ts - interrupted_signal_ts) * 1000

        is_latency_pass = stop_latency_ms <= self.config["max_barge_in_stop_latency_ms"]
        is_queue_flushed = (pending_queue_size_after_flush == 0)
        is_buffer_cleared = (unconsumed_buffer_ms <= 20.0)

        overall_status = "成功" if (is_latency_pass and is_queue_flushed and is_buffer_cleared) else "失败"

        return {
            "barge_in_stop_latency_ms": round(stop_latency_ms, 2),
            "interrupted_signal_rtt_ms": round(signaling_latency_ms, 2),
            "client_queue_eviction_ms": round(client_eviction_latency_ms, 2),
            "pending_queue_size": pending_queue_size_after_flush,
            "unconsumed_buffer_ms": round(unconsumed_buffer_ms, 2),
            "assertions": {
                "barge_in_stop_within_sla": is_latency_pass,
                "pending_queue_strictly_cleared": is_queue_flushed,
                "zero_audio_leakage": is_buffer_cleared
            },
            "status": overall_status
        }

    def evaluate_three_tier_adversarial_matrix(self, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "true_barge_in": {"total": 0, "success": 0, "avg_stop_latency_ms": 0.0},
            "backchanneling": {"total": 0, "correct_uninterrupted": 0, "false_interrupted": 0},
            "hesitation": {"total": 0, "correct_holding": 0, "premature_interrupted": 0}
        }

        total_latencies = []
        for r in test_records:
            case_type = r.get("type")
            if case_type == "true_barge_in":
                summary["true_barge_in"]["total"] += 1
                lat = r.get("stop_latency_ms", 0)
                if r.get("stopped_successfully") and lat <= self.config["max_barge_in_stop_latency_ms"]:
                    summary["true_barge_in"]["success"] += 1
                    total_latencies.append(lat)
            elif case_type == "backchanneling":
                summary["backchanneling"]["total"] += 1
                if not r.get("was_interrupted"):
                    summary["backchanneling"]["correct_uninterrupted"] += 1
                else:
                    summary["backchanneling"]["false_interrupted"] += 1
            elif case_type == "hesitation":
                summary["hesitation"]["total"] += 1
                if not r.get("was_premature_interrupted"):
                    summary["hesitation"]["correct_holding"] += 1
                else:
                    summary["hesitation"]["premature_interrupted"] += 1

        b_total = summary["backchanneling"]["total"]
        backchannel_acc = (summary["backchanneling"]["correct_uninterrupted"] / b_total) if b_total > 0 else 1.0

        h_total = summary["hesitation"]["total"]
        premature_rate = (summary["hesitation"]["premature_interrupted"] / h_total) if h_total > 0 else 0.0

        t_total = summary["true_barge_in"]["total"]
        barge_in_success_rate = (summary["true_barge_in"]["success"] / t_total) if t_total > 0 else 1.0

        avg_lat = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0.0
        summary["true_barge_in"]["avg_stop_latency_ms"] = avg_lat

        pass_all = (
            barge_in_success_rate >= 0.95 and
            backchannel_acc >= self.config["min_backchannel_precision"] and
            premature_rate <= self.config["max_premature_interruption_rate"]
        )

        return {
            "barge_in_success_rate": round(barge_in_success_rate * 100, 2),
            "backchannel_precision": round(backchannel_acc * 100, 2),
            "premature_interruption_rate": round(premature_rate * 100, 2),
            "average_stop_latency_ms": avg_lat,
            "details": summary,
            "status": "成功" if pass_all else "失败"
        }

    def evaluate_dual_agent_task_closure(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        turns = session_data.get("dialogue_turns", 0)
        tool_calls = session_data.get("tool_invocations", [])
        backend_executed = session_data.get("backend_verified_state", False)
        verbal_claimed = session_data.get("agent_verbal_claim_completed", False)

        closure_defect = (verbal_claimed and not backend_executed)
        task_success = (verbal_claimed and backend_executed)

        return {
            "total_interactive_turns": turns,
            "tool_call_count": len(tool_calls),
            "agent_verbal_claimed": verbal_claimed,
            "backend_physics_verified": backend_executed,
            "anti_surrender_closure_check": "通过" if not closure_defect else "存在口头虚假完成缺陷",
            "task_completion_status": "成功" if task_success else "失败"
        }

    def run_e2e_verification_suite(self) -> Dict[str, Any]:
        print(">> [Full-Duplex Engine] 正在执行全双工四段时延与状态机核验...")
        sample_latency_events = {
            "user_speech_end_ts": 100.000,
            "upload_start_ts": 100.020,
            "first_text_event_ts": 100.410,
            "first_audio_chunk_ts": 100.520,
            "speaker_out_ts": 100.580
        }
        lat_res = self.evaluate_latency_breakdown(sample_latency_events)

        print(">> [Full-Duplex Engine] 正在核验打断信号与客户端播放队列 Flush 清空断言...")
        barge_res = self.evaluate_barge_in_and_playback_flush(
            barge_in_start_ts=200.000,
            interrupted_signal_ts=200.120,
            last_audible_frame_ts=200.245,
            pending_queue_size_after_flush=0,
            unconsumed_buffer_ms=0.0
        )

        print(">> [Full-Duplex Engine] 正在注入打断三阶对抗矩阵 (True Barge-in / Backchanneling / Hesitation)...")
        adversarial_samples = [
            {"type": "true_barge_in", "stopped_successfully": True, "stop_latency_ms": 235.0},
            {"type": "true_barge_in", "stopped_successfully": True, "stop_latency_ms": 260.0},
            {"type": "true_barge_in", "stopped_successfully": True, "stop_latency_ms": 248.0},
            {"type": "backchanneling", "was_interrupted": False},
            {"type": "backchanneling", "was_interrupted": False},
            {"type": "backchanneling", "was_interrupted": False},
            {"type": "hesitation", "was_premature_interrupted": False},
            {"type": "hesitation", "was_premature_interrupted": False}
        ]
        matrix_res = self.evaluate_three_tier_adversarial_matrix(adversarial_samples)

        print(">> [Full-Duplex Engine] 正在评估双 Agent 对弈与真实物理任务闭环...")
        dual_res = self.evaluate_dual_agent_task_closure({
            "dialogue_turns": 18,
            "tool_invocations": [{"tool": "query_weather", "status": "executed"}],
            "agent_verbal_claim_completed": True,
            "backend_verified_state": True
        })

        all_passed = (
            lat_res["status"] == "成功" and
            barge_res["status"] == "成功" and
            matrix_res["status"] == "成功" and
            dual_res["task_completion_status"] == "成功"
        )

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": "全双工实时语音交互评测链路 (MOSS-Live / WebRTC / SenseVoice)",
            "overall_status": "成功" if all_passed else "失败",
            "latency_breakdown": lat_res,
            "barge_in_queue_flush": barge_res,
            "adversarial_matrix": matrix_res,
            "dual_agent_closure": dual_res
        }
        return report

if __name__ == "__main__":
    evaluator = FullDuplexEvaluator()
    rep = evaluator.run_e2e_verification_suite()
    print("\n" + "=" * 60)
    print("全双工实时交互自动化评测报告:")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
