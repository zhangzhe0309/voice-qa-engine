"""Voice QA Engine: 工业级全链路语音自动化测试与评估套件
一键执行物理声学质检、环境噪声抗扰度、ASR字错率(CER)、TTS时延与实时率(RTF)，
以及全双工语音交互端到端质量评估 (四段物理时延, 打断与队列Flush断言, 三阶对抗矩阵与双Agent对弈)。
"""

import os
import sys
import json
import time

VOICE_ENGINE_DIR = "/root/company-qa-workspace/voice-qa-engine"
if VOICE_ENGINE_DIR not in sys.path:
    sys.path.append(VOICE_ENGINE_DIR)

# 导入底层基础模块
from acoustic_inspector import AcousticInspector
from noise_evaluator import NoiseEvaluator
from latency_benchmark import LatencyBenchmark
from full_duplex_evaluator import FullDuplexEvaluator


def run_full_suite():
    print("=" * 70)
    print("🎙️ 聆芯 AI 语音场景工业级自动化测试与全双工端到端评测套件 (Voice QA Engine)")
    print("=" * 70)

    inspector = AcousticInspector()
    noise_eval = NoiseEvaluator()
    latency_bench = LatencyBenchmark()
    full_duplex_eval = FullDuplexEvaluator()

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_target": "知心姐姐 & MOSS-Live 实时全双工语音链路 (TTS & ASR & VAD & Barge-in)",
        "modules": {}
    }

    # -------------------------------------------------------------
    # 模块 1: 物理声学质量与爆音扫描 (Acoustic Anomaly & Pop Check)
    # -------------------------------------------------------------
    print("\n[Phase 1] 正在执行真实音频物理声学质检与爆音扫描...")
    sample_wav_path = "/var/www/tts_isolated/samples/zhixin_tts_pop_test.wav"

    if os.path.exists(sample_wav_path):
        with open(sample_wav_path, "rb") as f:
            sample_bytes = f.read()

        acoustic_res = inspector.inspect_wav_bytes(sample_bytes)
        report_data["modules"]["acoustic_inspection"] = acoustic_res
        metrics = acoustic_res.get("metrics", {})
        fmt = acoustic_res.get("format_info", {})

        print(f"  --> 音频时长: {fmt.get('duration_seconds')}s | 采样率: {fmt.get('sample_rate_hz')}Hz | 总样点: {fmt.get('total_samples')}")
        print(f"  --> 最大一阶差分跳跃: {metrics.get('max_sample_jump')} (发生在 {metrics.get('max_jump_time_s')}s)")
        print(f"  --> 超过阈值(8000)的突变次数: {metrics.get('jumps_exceeding_threshold_count')}")
        print(f"  --> 直流残留偏置 (DC Offset): {metrics.get('dc_offset')}")
        print(f"  --> 削波样点数: {metrics.get('clipping_sample_count')} (削波率: {metrics.get('clipping_rate_pct')}%)")
        print(f"  --> 爆音风险综合评分 (0-100): {metrics.get('pop_risk_score')}")

        if acoustic_res.get("header_anomalies"):
            print(f"  [警报] 头文件异常: {acoustic_res['header_anomalies']}")
        else:
            print("  状态：头文件合规 (无 0xFFFFFFFF 假长度，大小闭合)")
    else:
        print("  [提示] 未找到测试样本音频，跳过此项。")

    # -------------------------------------------------------------
    # 模块 2: 环境噪声注入与 ASR/VAD 抗扰度衰减矩阵 (Noise Robustness)
    # -------------------------------------------------------------
    print("\n[Phase 2] 正在执行环境噪声注入与 VAD/ASR 抗扰度衰减矩阵评测...")
    if os.path.exists(sample_wav_path):
        with open(sample_wav_path, "rb") as f:
            f.seek(44)
            pcm_raw = f.read()

        import struct
        pcm_samples = list(struct.unpack(f"<{len(pcm_raw)//2}h", pcm_raw))
        ref_text = "好呀你打算多会儿去北京"

        snr_levels = [25.0, 15.0, 10.0, 5.0]
        noise_results = []

        for snr in snr_levels:
            noisy_pcm = noise_eval.inject_noise_to_pcm(pcm_samples, snr_db=snr, noise_type="hallway_murmur")
            noisy_wav = noise_eval.pcm_to_wav_bytes(noisy_pcm, sample_rate=24000)

            asr_res = latency_bench.benchmark_asr(noisy_wav)
            hyp_text = asr_res.get("recognized_text", "").replace("，", "").replace("？", "").replace("。", "")
            cer_res = noise_eval.calculate_cer(ref_text, hyp_text)

            env_desc = "安静环境" if snr >= 20 else ("走廊轻噪" if snr >= 10 else "严重嘈杂")
            noise_results.append({
                "snr_db": snr,
                "environment_sim": env_desc,
                "hyp_text": hyp_text,
                "cer_rate": cer_res["cer"],
                "accuracy_rate": cer_res["accuracy"],
                "latency_ms": asr_res.get("latency_ms")
            })

            print(f"  --> SNR={snr:4.1f}dB ({env_desc}) | 识别: '{hyp_text}' | CER={cer_res['cer']*100:5.1f}% | 耗时: {asr_res.get('latency_ms')}ms")

        report_data["modules"]["noise_robustness"] = noise_results

    # -------------------------------------------------------------
    # 模块 3: 时延与流式吞吐实时率基准 (Latency & RTF Benchmark)
    # -------------------------------------------------------------
    print("\n[Phase 3] 正在对知心姐姐音色测量短句、中句、长文本 TTS 生成时延与实时率 (RTF)...")
    corpus = [
        ("短句 (8字)", "你好呀，我是知心姐姐。"),
        ("中句 (22字)", "今天在学校里过得开心吗？遇到烦恼可以随时跟我说哦。"),
        ("长句 (45字)", "学习压力大是很正常的现象，我们可以试着把大目标拆解成小任务，每天进步一点点就是很棒的自己啦。")
    ]

    tts_benchmarks = []
    for label, text in corpus:
        res = latency_bench.benchmark_tts(text)
        if res.get("success"):
            tts_benchmarks.append({
                "label": label,
                "text_length": res["text_length"],
                "audio_duration_s": res["audio_duration_seconds"],
                "e2e_latency_ms": res["e2e_latency_ms"],
                "rtf": res["real_time_factor_rtf"],
                "is_healthy": res["is_rtf_healthy"]
            })
            print(f"  --> [{label}] 耗时: {res['e2e_latency_ms']}ms | 音频时长: {res['audio_duration_seconds']}s | 实时率 RTF: {res['real_time_factor_rtf']} (行业红线 < 0.8)")
        else:
            print(f"  --> [{label}] 失败: {res.get('error')}")

    report_data["modules"]["latency_benchmark"] = tts_benchmarks

    # -------------------------------------------------------------
    # 模块 4: 全双工实时语音交互与打断状态机评测 (Full-Duplex Evaluation)
    # -------------------------------------------------------------
    print("\n[Phase 4] 正在执行全双工交互端到端质量评估 (四段时延/打断队列Flush/三阶对抗/双Agent)...")
    fd_report = full_duplex_eval.run_e2e_verification_suite()
    report_data["modules"]["full_duplex_e2e_evaluation"] = fd_report

    lat_m = fd_report["latency_breakdown"]["metrics"]
    barge_m = fd_report["barge_in_queue_flush"]
    adv_m = fd_report["adversarial_matrix"]
    dual_m = fd_report["dual_agent_closure"]

    print(f"  --> 四段物理时延: TTFT={lat_m['ttft_ms']}ms | 首音频包={lat_m['first_audio_packet_latency_ms']}ms | 扬声器首声={lat_m['eot_to_speech_latency_ms']}ms | 播放链路滞后={lat_m['client_playback_pipeline_lag_ms']}ms")
    print(f"  --> 打断刹车时延: {barge_m['barge_in_stop_latency_ms']}ms (SLA <= 300ms) | 客户端待播队列清空断言: 队列={barge_m['pending_queue_size']} (状态: {barge_m['status']})")
    print(f"  --> 三阶打断对抗: 正向打断成功率={adv_m['barge_in_success_rate']}% | 附和倾听精准率={adv_m['backchannel_precision']}% | 思考停顿误抢率={adv_m['premature_interruption_rate']}%")
    print(f"  --> 双Agent对弈闭环: 轮次={dual_m['total_interactive_turns']} | 工具执行={dual_m['tool_call_count']} | 物理真实落盘核验: {dual_m['anti_surrender_closure_check']}")

    # -------------------------------------------------------------
    # 物理落盘与综合健康结论输出
    # -------------------------------------------------------------
    os.makedirs("/root/company-qa-workspace/voice-qa-engine/reports", exist_ok=True)
    out_json = "/root/company-qa-workspace/voice-qa-engine/reports/voice_qa_full_baseline_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    feishu_report = "/root/feishu-data/voice_qa_full_baseline_report.json"
    with open(feishu_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"状态：全套物理测试完成！数据已落盘至:")
    print(f"  1. {out_json}")
    print(f"  2. {feishu_report}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_suite()
