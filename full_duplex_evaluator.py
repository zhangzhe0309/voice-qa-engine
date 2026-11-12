"""Full-Duplex Voice Evaluator: 全双工时序与状态机断言计算引擎
负责四段时延计算、打断状态机断言、三阶对抗统计与双 Agent 任务闭环度量。
严格遵循可信度与审计原则：
1. 模拟数据执行必须明确打上 SIMULATED_PASS 与 simulation_used: true，严禁冒充真实物理测试；
2. 确立 PASS / FAIL / ERROR / INCONCLUSIVE / SIMULATED_PASS / NOT_RUN 六级标准状态枚举；
3. 增加 measurement_source 字段明确标注来源 (protocol_observed / physical_loopback / simulated)。
"""

import time
from typing import Dict, List, Any, Optional

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ERROR = "ERROR"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_SIMULATED_PASS = "SIMULATED_PASS"
STATUS_NOT_RUN = "NOT_RUN"


class FullDuplexEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "max_barge_in_stop_latency_ms": 300.0,
            "barge_in_p50_target_ms": 200.0,
            "barge_in_p95_target_ms": 300.0,
            "barge_in_hard_limit_ms": 350.0,
            "max_ttft_ms": 600.0,
            "max_first_audio_ms": 700.0,
            "min_backchannel_precision": 0.985,  # 规范 v1.4.0/v1.4.1 核心 SLA: 附和精准度 >= 98.5%
            "max_premature_interruption_rate": 0.015,  # 留白 0.5s~2.5s 误抢话率 <= 1.5%
            "audio_sample_rate": 16000,
            "frame_duration_ms": 20
        }

    def evaluate_latency_breakdown(
        self,
        events: Dict[str, Any],
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """四段时延解耦与计算
        events 包含: user_speech_end_ts, upload_start_ts, first_text_event_ts, first_audio_chunk_ts, speaker_out_ts
        """
        user_end = events.get("user_speech_end_ts")
        first_text = events.get("first_text_event_ts")
        first_audio = events.get("first_audio_chunk_ts")
        speaker_out = events.get("speaker_out_ts")

        if user_end is None or first_text is None or first_audio is None:
            return {
                "status": STATUS_INCONCLUSIVE,
                "error": "关键时序事件点不完整，无法精确计算四段物理时延"
            }

        ttft_ms = (first_text - user_end) * 1000.0
        first_audio_ms = (first_audio - user_end) * 1000.0
        eot_to_speech_ms = (speaker_out - user_end) * 1000.0 if speaker_out else 0.0
        client_buffer_lag_ms = (speaker_out - first_audio) * 1000.0 if speaker_out and first_audio else 0.0

        is_ttft_pass = ttft_ms <= self.config["max_ttft_ms"]
        is_first_audio_pass = first_audio_ms <= self.config["max_first_audio_ms"]
        is_eot_pass = (eot_to_speech_ms <= (self.config["max_first_audio_ms"] + 150.0)) if speaker_out else True

        passed = (is_ttft_pass and is_first_audio_pass and is_eot_pass)
        status = STATUS_SIMULATED_PASS if (passed and is_simulated) else (STATUS_PASS if passed else STATUS_FAIL)

        return {
            "status": status,
            "measurement_source": "mock_events" if is_simulated else "real_event_timestamps",
            "metrics": {
                "ttft_ms": round(ttft_ms, 2),
                "first_audio_packet_latency_ms": round(first_audio_ms, 2),
                "eot_to_speech_latency_ms": round(eot_to_speech_ms, 2) if speaker_out else None,
                "client_playback_pipeline_lag_ms": round(client_buffer_lag_ms, 2) if speaker_out else None
            },
            "sla": {
                "ttft_target_ms": self.config["max_ttft_ms"],
                "first_audio_target_ms": self.config["max_first_audio_ms"]
            }
        }

    def evaluate_barge_in_and_playback_flush(
        self,
        barge_in_start_ts: float,
        interrupted_signal_ts: float,
        last_audible_frame_ts: float,
        pending_queue_size_after_flush: int,
        unconsumed_buffer_ms: float,
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """打断刹车时延与客户端播放队列 Flush 硬断言"""
        stop_latency_ms = (last_audible_frame_ts - barge_in_start_ts) * 1000.0
        signaling_latency_ms = (interrupted_signal_ts - barge_in_start_ts) * 1000.0
        client_eviction_latency_ms = (last_audible_frame_ts - interrupted_signal_ts) * 1000.0

        is_latency_pass = stop_latency_ms <= self.config["max_barge_in_stop_latency_ms"]
        is_queue_flushed = (pending_queue_size_after_flush == 0)
        is_buffer_cleared = (unconsumed_buffer_ms <= 20.0)

        passed = (is_latency_pass and is_queue_flushed and is_buffer_cleared)
        status = STATUS_SIMULATED_PASS if (passed and is_simulated) else (STATUS_PASS if passed else STATUS_FAIL)

        return {
            "status": status,
            "measurement_source": "mock_bargein_events" if is_simulated else "playback_engine_telemetry",
            "barge_in_stop_latency_ms": round(stop_latency_ms, 2),
            "interrupted_signal_rtt_ms": round(signaling_latency_ms, 2),
            "client_queue_eviction_ms": round(client_eviction_latency_ms, 2),
            "pending_queue_size": pending_queue_size_after_flush,
            "unconsumed_buffer_ms": round(unconsumed_buffer_ms, 2),
            "assertions": {
                "barge_in_stop_within_sla": is_latency_pass,
                "pending_queue_strictly_cleared": is_queue_flushed,
                "zero_audio_leakage": is_buffer_cleared
            }
        }

    def evaluate_three_tier_adversarial_matrix(
        self,
        test_records: List[Dict[str, Any]],
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """打断行为三阶对抗评测"""
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

        status = STATUS_SIMULATED_PASS if (pass_all and is_simulated) else (STATUS_PASS if pass_all else STATUS_FAIL)

        return {
            "status": status,
            "measurement_source": "mock_samples" if is_simulated else "scenario_suite_runs",
            "barge_in_success_rate": round(barge_in_success_rate * 100, 2),
            "backchannel_precision": round(backchannel_acc * 100, 2),
            "premature_interruption_rate": round(premature_rate * 100, 2),
            "average_stop_latency_ms": avg_lat,
            "details": summary
        }

    def evaluate_dual_agent_task_closure(
        self,
        session_data: Dict[str, Any],
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """双 Agent 对弈与真实物理任务闭环核验"""
        turns = session_data.get("dialogue_turns", 0)
        tool_calls = session_data.get("tool_invocations", [])
        backend_executed = session_data.get("backend_verified_state", False)
        verbal_claimed = session_data.get("agent_verbal_claim_completed", False)

        closure_defect = (verbal_claimed and not backend_executed)
        task_success = (verbal_claimed and backend_executed)

        status = STATUS_SIMULATED_PASS if (task_success and is_simulated) else (STATUS_PASS if task_success else STATUS_FAIL)

        return {
            "status": status,
            "measurement_source": "mock_session" if is_simulated else "real_dual_agent_execution",
            "total_interactive_turns": turns,
            "tool_call_count": len(tool_calls),
            "agent_verbal_claimed": verbal_claimed,
            "backend_physics_verified": backend_executed,
            "anti_surrender_closure_check": "通过" if not closure_defect else "存在口头虚假完成缺陷"
        }

    def run_unit_simulation_suite(self) -> Dict[str, Any]:
        """运行算法与指标断言单元仿真测试 (明确标记为 SIMULATED_PASS，不替代真实 E2E)"""
        sample_latency_events = {
            "user_speech_end_ts": 100.000,
            "upload_start_ts": 100.020,
            "first_text_event_ts": 100.410,
            "first_audio_chunk_ts": 100.520,
            "speaker_out_ts": 100.580
        }
        lat_res = self.evaluate_latency_breakdown(sample_latency_events, is_simulated=True)

        barge_res = self.evaluate_barge_in_and_playback_flush(
            barge_in_start_ts=200.000,
            interrupted_signal_ts=200.120,
            last_audible_frame_ts=200.245,
            pending_queue_size_after_flush=0,
            unconsumed_buffer_ms=0.0,
            is_simulated=True
        )

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
        matrix_res = self.evaluate_three_tier_adversarial_matrix(adversarial_samples, is_simulated=True)

        dual_res = self.evaluate_dual_agent_task_closure({
            "dialogue_turns": 18,
            "tool_invocations": [{"tool": "query_weather", "status": "executed"}],
            "agent_verbal_claim_completed": True,
            "backend_verified_state": True
        }, is_simulated=True)

        all_sim_passed = (
            lat_res["status"] == STATUS_SIMULATED_PASS and
            barge_res["status"] == STATUS_SIMULATED_PASS and
            matrix_res["status"] == STATUS_SIMULATED_PASS and
            dual_res["status"] == STATUS_SIMULATED_PASS
        )

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": "全双工时序与断言单元仿真 (Full-Duplex Unit Simulation)",
            "execution_mode": "unit_test_simulation",
            "measurement_source": "mock_events",
            "protocol_observed": False,
            "audio_capture_verified": False,
            "simulation_used": True,
            "evidence_complete": False,
            "overall_status": STATUS_SIMULATED_PASS if all_sim_passed else STATUS_FAIL,
            "latency_breakdown": lat_res,
            "barge_in_queue_flush": barge_res,
            "adversarial_matrix": matrix_res,
            "dual_agent_closure": dual_res
        }
