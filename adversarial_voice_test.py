"""Adversarial Voice Test Suite: 语音模型门禁反向对抗变异测试
通过主动注入 4 大声学恶性变异体，反向验证 AcousticInspector 质检套件 100% 具备一票否决能力。
动态解析工程相对路径，消除硬编码。
"""

import json
import math
import struct
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from acoustic_inspector import AcousticInspector  # noqa: E402

inspector = AcousticInspector()
MUTATION_DIR = BASE_DIR / "mutations"
MUTATION_DIR.mkdir(parents=True, exist_ok=True)


def generate_clean_base_pcm(duration_sec: float = 1.0, sample_rate: int = 24000) -> bytes:
    """生成一段 1.0 秒 440Hz 标准纯净正弦波 (无爆音无削波)"""
    num_samples = int(duration_sec * sample_rate)
    samples = []
    for i in range(num_samples):
        val = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.append(val)
    return struct.pack(f"<{num_samples}h", *samples)


def make_standard_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    fake_riff_size: int | None = None,
    fake_data_size: int | None = None,
) -> bytes:
    data_size = len(pcm_data) if fake_data_size is None else fake_data_size
    riff_size = (len(pcm_data) + 36) if fake_riff_size is None else fake_riff_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size
    )
    return header + pcm_data


def test_adversarial_mutations():
    print("=" * 70)
    print("🧬 启动语音门禁反向对抗变异测试 (Adversarial Mutation Tests)")
    print("=" * 70)

    clean_pcm = generate_clean_base_pcm(1.0, 24000)
    sample_rate = 24000
    results = []

    # -------------------------------------------------------------
    # 变异体 1: 瞬态样点断崖式阶跃突变 (Spike/Pop 爆音突刺)
    # -------------------------------------------------------------
    num_s = len(clean_pcm) // 2
    samples_1 = list(struct.unpack(f"<{num_s}h", clean_pcm))
    spike_idx = 12000
    samples_1[spike_idx] = -15000
    samples_1[spike_idx + 1] = 16000
    mutated_pcm_1 = struct.pack(f"<{num_s}h", *samples_1)
    mutated_wav_1 = make_standard_wav(mutated_pcm_1, sample_rate)

    wav_path_1 = MUTATION_DIR / "bad_pop_spike.wav"
    with open(wav_path_1, "wb") as f:
        f.write(mutated_wav_1)

    res_1 = inspector.inspect_wav_bytes(mutated_wav_1)
    metrics_1 = res_1.get("metrics", {})
    killed_1 = metrics_1.get("max_sample_jump", 0) >= 8000 and metrics_1.get("pop_risk_score", 0) >= 25
    results.append({
        "mutation_id": "MUT-001-POP-SPIKE",
        "name": "瞬态一阶样点突变 (>8000 跳变)",
        "injected_fault": "采样点断崖阶跃: -15000 -> 16000 (Delta 31000)",
        "detected_jump": metrics_1.get("max_sample_jump"),
        "risk_score": metrics_1.get("pop_risk_score"),
        "mutant_killed": killed_1,
        "status": "PASS" if killed_1 else "FAIL"
    })
    print(f"  --> [MUT-001] 爆音突刺突变体 | 检出 Jump: {metrics_1.get('max_sample_jump')} | 拦截状态: {'成功杀死突变体' if killed_1 else '漏报'}")

    # -------------------------------------------------------------
    # 变异体 2: 4GB 假长度未闭合音频流 (0xFFFFFFFF Header)
    # -------------------------------------------------------------
    mutated_wav_2 = make_standard_wav(clean_pcm, sample_rate, fake_riff_size=0xFFFFFFFF, fake_data_size=0xFFFFFFFF)
    wav_path_2 = MUTATION_DIR / "bad_fake_4gb_header.wav"
    with open(wav_path_2, "wb") as f:
        f.write(mutated_wav_2)

    res_2 = inspector.inspect_wav_bytes(mutated_wav_2)
    has_crit_2 = any(h["severity"] == "CRITICAL" for h in res_2.get("header_anomalies", []))
    killed_2 = has_crit_2 and res_2.get("metrics", {}).get("pop_risk_score", 0) >= 50
    results.append({
        "mutation_id": "MUT-002-FAKE-4GB-HEADER",
        "name": "0xFFFFFFFF 假长度未闭合流",
        "injected_fault": "RIFF 与 data 尺寸覆盖为 0xFFFFFFFF",
        "anomalies_detected": len(res_2.get("header_anomalies", [])),
        "mutant_killed": killed_2,
        "status": "PASS" if killed_2 else "FAIL"
    })
    print(f"  --> [MUT-002] 假长度头文件突变体 | 检出异常: {len(res_2.get('header_anomalies', []))}项 | 拦截状态: {'成功杀死突变体' if killed_2 else '漏报'}")

    # -------------------------------------------------------------
    # 变异体 3: 幅值满幅削波失真 (Clipping Distortion)
    # -------------------------------------------------------------
    samples_3 = list(struct.unpack(f"<{num_s}h", clean_pcm))
    for i in range(5000, 7000):
        samples_3[i] = 32767 if samples_3[i] >= 0 else -32768
    mutated_pcm_3 = struct.pack(f"<{num_s}h", *samples_3)
    mutated_wav_3 = make_standard_wav(mutated_pcm_3, sample_rate)

    wav_path_3 = MUTATION_DIR / "bad_clipping_square.wav"
    with open(wav_path_3, "wb") as f:
        f.write(mutated_wav_3)

    res_3 = inspector.inspect_wav_bytes(mutated_wav_3)
    metrics_3 = res_3.get("metrics", {})
    killed_3 = metrics_3.get("clipping_sample_count", 0) >= 1000 and metrics_3.get("clipping_rate_pct", 0) > 0.1
    results.append({
        "mutation_id": "MUT-003-CLIPPING-SQUARE",
        "name": "连续 2000 样点方波满幅削波",
        "injected_fault": "幅值削波至 +/-32768",
        "clipping_samples": metrics_3.get("clipping_sample_count"),
        "clipping_pct": metrics_3.get("clipping_rate_pct"),
        "mutant_killed": killed_3,
        "status": "PASS" if killed_3 else "FAIL"
    })
    print(f"  --> [MUT-003] 严重削波突变体 | 检出削波点: {metrics_3.get('clipping_sample_count')} | 拦截状态: {'成功杀死突变体' if killed_3 else '漏报'}")

    # -------------------------------------------------------------
    # 变异体 4: 结尾非零硬截断 (Hard Tail Cutoff)
    # -------------------------------------------------------------
    samples_4 = list(struct.unpack(f"<{num_s}h", clean_pcm))
    for i in range(num_s - 50, num_s):
        samples_4[i] = 12000
    mutated_pcm_4 = struct.pack(f"<{num_s}h", *samples_4)
    mutated_wav_4 = make_standard_wav(mutated_pcm_4, sample_rate)

    wav_path_4 = MUTATION_DIR / "bad_hard_cut_step.wav"
    with open(wav_path_4, "wb") as f:
        f.write(mutated_wav_4)

    res_4 = inspector.inspect_wav_bytes(mutated_wav_4)
    has_step_4 = any(t["type"] == "HARD_TAIL_CUTOFF" for t in res_4.get("transient_anomalies", []))
    killed_4 = has_step_4 and res_4.get("metrics", {}).get("tail_step_level", 0) >= 500
    results.append({
        "mutation_id": "MUT-004-HARD-TAIL-STEP",
        "name": "结尾最后样点高位硬截断 (12000)",
        "injected_fault": "Tail Step 直接以 12000 截断不归零",
        "tail_step_level": res_4.get("metrics", {}).get("tail_step_level"),
        "mutant_killed": killed_4,
        "status": "PASS" if killed_4 else "FAIL"
    })
    print(f"  --> [MUT-004] 结尾截断突变体 | 检出阶跃: {res_4.get('metrics', {}).get('tail_step_level')} | 拦截状态: {'成功杀死突变体' if killed_4 else '漏报'}")

    all_killed = all(r["mutant_killed"] for r in results)
    report_path = BASE_DIR / "reports" / "adversarial_mutation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": "ALL_MUTANTS_KILLED" if all_killed else "LEAKED", "results": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"门禁有效性判定: {'状态：通过 (4/4 恶性突变体 100% 杀死并被物理拦截)' if all_killed else '状态：失败 (存在漏检)'}")
    print(f"报告已保存: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    test_adversarial_mutations()
