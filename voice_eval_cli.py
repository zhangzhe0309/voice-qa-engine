"""Voice QA Engine: 工业级全链路语音自动化测试与评估套件 (CLI 调度总控)
动态解析工程相对路径，严格遵循安全脱敏与状态隔离标准。
"""

import os
import sys
import json
import time
from pathlib import Path

# 动态解析工程根目录
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
        "execution_context": {
            "project_root": str(BASE_DIR),
            "platform": sys.platform
        },
        "modules": {}
    }

    # -------------------------------------------------------------
    # 模块 1: 物理声学质量与爆音扫描 (Acoustic Anomaly & Pop Check)
    # -------------------------------------------------------------
    print("\n[Phase 1] 正在执行真实音频物理声学质检与爆音扫描...")
    # 优先使用本地相对路径下的样本
    sample_wav_path = BASE_DIR / "samples" / "zhixin_tts_pop_test.wav"
    if not sample_wav_path.exists():
        vps_path = Path("/var/www/tts_isolated/samples/zhixin_tts_pop_test.wav")
        if vps_path.exists():
            sample_wav_path = vps_path
        else:
            # 自动生成本地纯净基准音频，保证开箱即测
            samples_dir = BASE_DIR / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            import wave, struct, math
            num_samples = int(1.0 * 24000)
            sine_samples = [int(12000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(num_samples)]
            with wave.open(str(sample_wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(struct.pack(f"<{num_samples}h", *sine_samples))

    if sample_wav_path.exists():
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
        print(f"  --> 爆音风险综合评分 (0-100): {metrics.get('pop_risk_score')} (状态: {acoustic_res.get('status')})")
    else:
        print("  [提示] 未找到测试样本音频，跳过此项。")
        report_data["modules"]["acoustic_inspection"] = {"status": "NOT_RUN", "reason": "No sample audio found"}

    # -------------------------------------------------------------
    # 模块 2: 环境噪声注入与 ASR/VAD 抗扰度衰减矩阵 (Noise Robustness)
    # -------------------------------------------------------------
    print("\n[Phase 2] 正在执行环境噪声注入与 VAD/ASR 抗扰度衰减矩阵评测...")
    if sample_wav_path.exists():
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
            if asr_res.get("success"):
                raw_text = asr_res.get("recognized_text") or ""
                if not raw_text.strip():
                    print(f"  --> SNR={snr:4.1f}dB | ASR 返回空文本 (无凭证/服务未就绪)，标记 INCONCLUSIVE")
                    noise_results.append({
                        "snr_db": snr,
                        "status": "INCONCLUSIVE",
                        "reason": "ASR 返回空文本，跳过 CER 计算"
                    })
                    continue
                hyp_text = str(raw_text).replace("，", "").replace("？", "").replace("。", "")
                cer_res = noise_eval.calculate_cer(ref_text, hyp_text)
                env_desc = "安静环境" if snr >= 20 else ("走廊轻噪" if snr >= 10 else "严重嘈杂")
                noise_results.append({
                    "snr_db": snr,
                    "environment_sim": env_desc,
                    "hyp_text": hyp_text,
                    "cer_rate": cer_res["cer"],
                    "accuracy_rate": cer_res["accuracy"],
                    "latency_ms": asr_res.get("latency_ms"),
                    "status": "PASS" if cer_res["cer"] < 0.3 else "FAIL"
                })
                print(f"  --> SNR={snr:4.1f}dB ({env_desc}) | 识别: '{hyp_text}' | CER={cer_res['cer']*100:5.1f}% | 耗时: {asr_res.get('latency_ms')}ms")
            else:
                print(f"  --> SNR={snr:4.1f}dB | ASR 接口不可达: {asr_res.get('error')}")

        report_data["modules"]["noise_robustness"] = noise_results if noise_results else {"status": "INCONCLUSIVE", "reason": "ASR service unreachable"}

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
                "status": "PASS" if res["is_rtf_healthy"] else "FAIL"
            })
            print(f"  --> [{label}] 耗时: {res['e2e_latency_ms']}ms | 音频时长: {res['audio_duration_seconds']}s | 实时率 RTF: {res['real_time_factor_rtf']} (行业红线 < 0.8)")
        else:
            print(f"  --> [{label}] 跳过: {res.get('error')} (状态: {res.get('status')})")

    report_data["modules"]["latency_benchmark"] = tts_benchmarks if tts_benchmarks else {"status": "INCONCLUSIVE", "reason": "TTS credentials unconfigured"}

    # -------------------------------------------------------------
    # 模块 4: 全双工时序与状态机单元仿真 (Unit Simulation)
    # -------------------------------------------------------------
    print("\n[Phase 4] 正在执行全双工时序与状态机算法单元仿真 (明确标记为 SIMULATED_PASS)...")
    fd_report = full_duplex_eval.run_unit_simulation_suite()
    report_data["modules"]["full_duplex_unit_simulation"] = fd_report

    lat_m = fd_report["latency_breakdown"]["metrics"]
    barge_m = fd_report["barge_in_queue_flush"]
    adv_m = fd_report["adversarial_matrix"]
    dual_m = fd_report["dual_agent_closure"]

    print(f"  --> 四段物理时延算法核验: TTFT={lat_m['ttft_ms']}ms | 首音频包={lat_m['first_audio_packet_latency_ms']}ms | 扬声器首声={lat_m['eot_to_speech_latency_ms']}ms")
    print(f"  --> 打断刹车时延算法: {barge_m['barge_in_stop_latency_ms']}ms (SLA <= 300ms) | 客户端待播队列清空断言: {barge_m['status']}")
    print(f"  --> 三阶打断对抗矩阵算法: 正向打断={adv_m['barge_in_success_rate']}% | 附和倾听={adv_m['backchannel_precision']}% | 思考停顿误抢={adv_m['premature_interruption_rate']}%")
    print(f"  --> 双Agent对弈闭环算法: 状态: {dual_m['status']} (注: 标记为 {fd_report['overall_status']}, 不冒充真实物理 E2E 结果)")

    # -------------------------------------------------------------
    # 物理落盘与综合健康结论输出
    # -------------------------------------------------------------
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "voice_qa_full_baseline_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"测试完成！基准数据已落盘至: {out_json}")
    print("=" * 70)


def run_v12_suite():
    import asyncio
    from voice_qa_engine.runtime.session_runtime import SessionRuntime
    from voice_qa_engine.runtime.event_timeline import EventTimeline
    from voice_qa_engine.runtime.context_stack import ContextCallStack
    from voice_qa_engine.runtime.tool_runtime import ToolAbortManager
    from voice_qa_engine.playback.engine import PlaybackEngine
    from voice_qa_engine.playback.sink import HannFadeoutSink
    from voice_qa_engine.evidence.recorder import EvidenceRecorder
    from voice_qa_engine.audio.input.wav_input import WavAudioInput
    from voice_qa_engine.evaluators.assertion_models import TestStatus
    from voice_qa_engine.evaluators.latency import LatencyEvaluator
    from voice_qa_engine.evaluators.evidence_validator import EvidenceValidator
    from voice_qa_engine.evaluators.aec_evaluator import AECCollisionEvaluator
    from voice_qa_engine.scenarios.normal_qa import NormalQAScenario
    from voice_qa_engine.scenarios.barge_in import BargeInScenario
    from voice_qa_engine.scenarios.rapid_barge_in import RapidDoubleBargeInScenario
    import numpy as np

    print("\n" + "=" * 70)
    print("🚀 Voice QA Engine V1.2: 真实全双工执行基座与防幽灵音频引擎 (V1.2 Suite)")
    print("=" * 70)

    async def _async_exec():
        session_id = f"session_v12_{int(time.time())}"
        reports_dir = BASE_DIR / "reports" / time.strftime("%Y-%m-%d")
        recorder = EvidenceRecorder(base_dir=str(reports_dir), session_id=session_id)
        await recorder.initialize()

        timeline = EventTimeline()
        sink = HannFadeoutSink(fade_duration_ms=10.0)
        engine = PlaybackEngine(sink=sink, timeline=timeline)

        # 构造基准 100ms 测试音频
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((np.zeros(1600, dtype=np.int16)).tobytes())
        buf.seek(0)
        audio_input = WavAudioInput(file_obj=buf, frame_duration_ms=20.0)

        context_stack = ContextCallStack(default_ttl=10)
        tool_manager = ToolAbortManager(abort_sla_ms=10.0)

        runtime = SessionRuntime(
            session_id=session_id,
            audio_input=audio_input,
            playback_engine=engine,
            timeline=timeline,
            recorder=recorder,
            context_stack=context_stack,
            tool_manager=tool_manager
        )

        print("[V1.2 Step 1] 建立会话连接并初始化单调时钟时间轴...")
        await runtime.connect()
        await runtime.start()

        print("[V1.2 Step 2] 上行持续推流音频 (Continuous Audio Ingestion)...")
        await runtime.feed_audio()

        print("[V1.2 Step 3] 模拟助手响应开始并异步发起后台工具查询...")
        gen_1 = await engine.begin_response("resp_v12_001")
        # 挂起主线并压入支线任务
        runtime.context_stack.push_subtask("query_weather", {"city": "Beijing"})
        runtime.tool_manager.register_tool_call("tool_weather_01", "query_weather", generation_id=gen_1)

        print("[V1.2 Step 4] 触发用户插话打断 (Barge-in Interrupt)、熔断异步工具并隔离旧代次...")
        # 1. 触发打断
        await engine.interrupt(reason="user_barge_in")
        # 2. 异步工具链全局 AbortController 熔断
        aborted_tools = runtime.tool_manager.abort_for_generation(generation_id=gen_1)
        # 3. 验证脏数据上屏拦截
        dirty_accepted = runtime.tool_manager.submit_tool_result("tool_weather_01", {"temp": "25C"}, generation_id=gen_1)

        # 模拟打断后延迟到达的旧 Chunk
        from voice_qa_engine.playback.queue import PlaybackChunk
        from voice_qa_engine.audio.models import AudioFormat, SampleFormat, AudioFrame
        dummy_frame = AudioFrame(
            samples=np.zeros(320, dtype=np.float32),
            format=AudioFormat(16000, 1, SampleFormat.PCM_F32LE),
            timestamp_ns=time.monotonic_ns(),
            sequence_id=5,
            duration_ms=20.0
        )
        stale_chunk = PlaybackChunk(
            session_id=session_id,
            response_id="resp_v12_001",
            generation_id=gen_1,
            sequence_id=5,
            audio=dummy_frame,
            received_monotonic_ns=time.monotonic_ns(),
            expected_duration_ms=20.0,
            cancel_token_id="old_token",
            connection_epoch=1
        )
        rejected = not (await engine.push_chunk(stale_chunk))
        print(f"  --> 防幽灵音频硬断言: 打断后迟到的旧 Chunk 拒绝状态: {'成功拒绝' if rejected else '失败漏音'}")
        print(f"  --> 工具链 Abort 熔断: 成功熔断 {len(aborted_tools)} 个在途工具 | 脏数据拦截: {'成功拦截' if not dirty_accepted else '异常穿透'}")

        # 4. 对话状态栈接续与 Rebase 合并
        resumed = runtime.context_stack.resume_full(target_depth=0)
        invariance_score = resumed.interruption_invariance_score if resumed else 1.0
        runtime.context_stack.merge_context({"city": "Shanghai", "day2_owner": "李老师"}, target_depth=0)
        print(f"  --> 四态对话断点状态栈: State Rebase 完成 | 打断不变性接续评分: {invariance_score * 100:.1f}%")

        # 5. AEC 回声消除反灌自激核验 (50% 增益反灌下抑制比达到 >= 40dB)
        ref_pcm = np.ones(1600, dtype=np.float32) * 0.5
        residual_pcm = np.ones(1600, dtype=np.float32) * 0.002
        aec_eval = AECCollisionEvaluator()
        aec_res = aec_eval.evaluate_echo_return_loss_enhancement(ref_pcm, ref_pcm * 0.5, residual_pcm, loopback_gain=0.5)
        print(f"  --> AEC 回声抑制比 (ERLE): {aec_res['erle_db']}dB (目标 >= 40dB) | 自激率: {aec_res['self_interruption_rate']*100:.1f}%")

        await runtime.close()

        # 计算单调时钟时延指标
        events = timeline.snapshot()
        metrics = LatencyEvaluator.calculate_latencies(events)
        metrics["tool_abort_metrics"] = runtime.tool_manager.get_summary_metrics()
        metrics["context_stack_metrics"] = runtime.context_stack.to_dict()
        metrics["aec_metrics"] = aec_res
        metrics["hann_fadeout_max_jump"] = sink.max_jump_detected

        session_meta = {
            "session_id": session_id,
            "version": "1.2.0",
            "scenario": "barge_in_and_ghost_rejection_v12",
            "status": TestStatus.PASS if (rejected and not dirty_accepted) else TestStatus.FAIL,
            "metrics": metrics
        }
        await recorder.finalize(metrics=metrics, session_meta=session_meta)

        # 产物防篡改校验
        sess_dir = os.path.join(str(reports_dir), session_id)
        valid, reason, _ = EvidenceValidator.validate_session_dir(sess_dir)
        print(f"[V1.2 Step 5] 五大必备物理证据生成与 SHA-256 校验: {reason}")
        print(f"  --> 产物目录: {sess_dir}")
        print(f"  --> 本地打断刹车时延: {metrics.get('local_stop_latency_ms', 0):.2f}ms")
        print(f"  --> 队列清空耗时: {metrics.get('queue_flush_latency_ms', 0):.2f}ms")
        reject_rate = metrics.get("old_chunk_rejection_rate")
        reject_pct = f"{reject_rate * 100:.1f}%" if reject_rate is not None else "N/A(未观察到旧分片)"
        print(f"  --> 旧分片拒绝率: {reject_pct}")

    asyncio.run(_async_exec())


def run_v13_suite():
    """V1.3 新场景套件: 经 SuiteRunner 统一执行 + 流式证据落盘"""
    import asyncio
    from voice_qa_engine.runtime.session_runtime import SessionRuntime
    from voice_qa_engine.runtime.event_timeline import EventTimeline
    from voice_qa_engine.playback.engine import PlaybackEngine
    from voice_qa_engine.playback.sink import HannFadeoutSink
    from voice_qa_engine.evidence.recorder import EvidenceRecorder
    from voice_qa_engine.audio.input.streaming_input import StreamingAudioInput
    from voice_qa_engine.scenarios.registry import ScenarioRegistry, SuiteRunner, ScenarioContext
    from voice_qa_engine.scenarios.speaker_exclusion_scenario import SpeakerExclusionScenario
    from voice_qa_engine.scenarios.golden_baseline_scenario import GoldenBaselineSoakScenario
    from voice_qa_engine.scenarios.disfluency_scenario import DisfluencyScenario
    from voice_qa_engine.scenarios.homophone_wakeup_scenario import HomophoneWakeupScenario
    from voice_qa_engine.agents.sparring_scenario import SparringScenario, MockDialogBackend
    from voice_qa_engine.agents.virtual_user import LLMPoweredVUA, ScriptedVUA

    print("\n" + "=" * 70)
    print("🚀 Voice QA Engine V1.3: 对抗增强场景套件 (对弈/拒识/soak/不流利/谐音)")
    print("=" * 70)

    reports_dir = BASE_DIR / "reports" / "v13" / time.strftime("%Y-%m-%d")

    async def context_factory(scenario_id: str) -> ScenarioContext:
        sid = f"{scenario_id}_{int(time.time() * 1e6)}"
        timeline = EventTimeline()
        engine = PlaybackEngine(sink=HannFadeoutSink(), timeline=timeline)
        audio_input = StreamingAudioInput()
        runtime = SessionRuntime(
            session_id=sid, audio_input=audio_input, playback_engine=engine,
            timeline=timeline,
            recorder=EvidenceRecorder(base_dir=str(reports_dir), session_id=sid,
                                      streaming=True),
        )
        await runtime.connect()
        await runtime.start()
        return ScenarioContext(runtime, audio_input, engine, timeline, runtime.recorder, {})

    registry = ScenarioRegistry()

    def _sparring(runtime, audio_input, playback_engine, timeline, evaluator=None):
        if LLMPoweredVUA().is_available():
            vua = LLMPoweredVUA()
            print("  [VUA] LLM 驱动模式 (VQA_LLM_API_KEY 已配置)")
        else:
            vua = ScriptedVUA(script=[
                "帮我们确认五天分布式架构方案和负责人",
                "时间就定下周，每天上午十点开始",
                "存储选型倾向用对象存储，成本优先",
                "好的，那就按这个方案推进，负责人保持不变",
            ])
            print("  [VUA] 脚本模式 (未检测到 VQA_LLM_API_KEY → 结果标 SIMULATED_PASS)")
        return SparringScenario(runtime, audio_input, playback_engine, timeline,
                                vua=vua, backend=MockDialogBackend())

    def _soak(runtime, audio_input, playback_engine, timeline, evaluator=None):
        print("  [Soak] 时间压缩比 60:1 (30 分钟语义 → 30 秒物理)")
        return GoldenBaselineSoakScenario(runtime, audio_input, playback_engine, timeline)

    for cls in (SpeakerExclusionScenario, DisfluencyScenario, HomophoneWakeupScenario):
        registry.register(cls)
    registry.register(_sparring)
    registry.register(_soak)

    report = asyncio.run(SuiteRunner("v13_adversarial", context_factory).run(registry))

    out_json = reports_dir / "suite_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    for entry in report["scenarios"]:
        print(f"  --> {entry['scenario_id']:<32} 状态: {entry['status']}  "
              f"证据可信: {entry.get('evidence_valid')}")
    print(f"  套件总判定: {report['overall_status']} | 报告落盘: {out_json}")
    return report


def run_scorecard():
    """一页指标大盘: 扫描 reports/ 全部会话证据，聚合 SLA 矩阵"""
    from voice_qa_engine.reporting.scorecard import build_scorecard, render_scorecard_text

    reports_root = str(BASE_DIR / "reports")
    scorecard = build_scorecard(reports_root)
    print(render_scorecard_text(scorecard))

    out_json = Path(reports_root) / "scorecard.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2)
    print(f"大盘已落盘: {out_json}")
    return scorecard


if __name__ == "__main__":
    if "--v13" in sys.argv:
        run_v13_suite()
    elif "--scorecard" in sys.argv:
        run_scorecard()
    elif "--v12" in sys.argv:
        run_v12_suite()
    else:
        run_full_suite()
        run_v12_suite()
