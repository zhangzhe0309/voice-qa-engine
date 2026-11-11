#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音场景专属反向对抗测试套件 (Negative / Mutation Voice QA Test Suite)
目的: 主动注入恶性反例 (爆音突变、严重削波、假长度头、换源硬阶跃)，
验证质检门禁在遭遇异常时能否 100% 敏锐报警并一票否决，杜绝虚假全绿。
"""

import os
import struct
import math
import json
import sys

sys.path.insert(0, '/root/company-qa-workspace/voice-qa-engine')
from acoustic_inspector import AcousticInspector

inspector = AcousticInspector()
MUTATION_DIR = "/root/company-qa-workspace/voice-qa-engine/mutations"
os.makedirs(MUTATION_DIR, exist_ok=True)

def generate_base_tone(duration_s=2.0, sr=24000, freq=440.0):
    samples = []
    num_samples = int(duration_s * sr)
    for i in range(num_samples):
        val = int(16000.0 * math.sin(2.0 * math.pi * freq * (i / sr)))
        samples.append(val)
    return samples, sr

def save_wav(filename, samples, sr=24000, override_header_size=None):
    raw_data = struct.pack(f"<{len(samples)}h", *samples)
    data_size = len(raw_data)
    riff_size = data_size + 36
    
    header = bytearray()
    header.extend(b'RIFF')
    if override_header_size:
        header.extend(override_header_size[0])
    else:
        header.extend(struct.pack('<I', riff_size))
    header.extend(b'WAVEfmt ')
    header.extend(struct.pack('<I', 16))
    header.extend(struct.pack('<H', 1))
    header.extend(struct.pack('<H', 1))
    header.extend(struct.pack('<I', sr))
    header.extend(struct.pack('<I', sr * 2))
    header.extend(struct.pack('<H', 2))
    header.extend(struct.pack('<H', 16))
    header.extend(b'data')
    if override_header_size:
        header.extend(override_header_size[1])
    else:
        header.extend(struct.pack('<I', data_size))
        
    full_wav = bytes(header) + raw_data
    with open(filename, 'wb') as f:
        f.write(full_wav)
    return full_wav

def inspect_file(filepath):
    with open(filepath, 'rb') as f:
        return inspector.inspect_wav_bytes(f.read())

def run_adversarial_suite():
    print("=================================================================")
    print("🔥 启动语音全链路反向对抗与变异破坏测试 (Adversarial Tests)")
    print("=================================================================")
    
    adversarial_results = {}
    
    # -------------------------------------------------------------
    # 反例 1: 恶性爆音冲激注入 (瞬态突变 Jump = 29500)
    # 期望: metrics.max_sample_jump > 12000, 爆音风险分加权飙升
    # -------------------------------------------------------------
    print("\n[反例 1] 构造恶性突变爆音样本 (模拟 DAC 开关脉冲 / 未做软压限)...")
    samples, sr = generate_base_tone(2.0, 24000)
    inject_idx = sr
    samples[inject_idx] = -14000
    samples[inject_idx + 1] = 15500
    samples[inject_idx + 2] = -14000
    
    pop_file = f"{MUTATION_DIR}/bad_pop_spike.wav"
    save_wav(pop_file, samples, sr)
    
    res1 = inspect_file(pop_file)
    m1 = res1.get("metrics", {})
    max_jump1 = m1.get("max_sample_jump", 0)
    intercepted_1 = max_jump1 >= 12000 and m1.get("jumps_exceeding_threshold_count", 0) > 0
    print(f"  -> 最大差分跳跃: {max_jump1} (红线门限: 8000)")
    print(f"  -> 越限跳跃次数: {m1.get('jumps_exceeding_threshold_count')}")
    print(f"  -> 爆音风险评分: {m1.get('pop_risk_score')} / 100")
    print(f"  -> 门禁敏锐拦截断言: {'[状态: 成功拦截]' if intercepted_1 else '[状态: 漏测放行]'}")
    adversarial_results["bad_pop_spike"] = {
        "case_name": "恶性爆音瞬态冲击 (Jump > 28000)",
        "expected_intercept": True,
        "max_sample_jump": max_jump1,
        "pop_risk_score": m1.get("pop_risk_score"),
        "pass": intercepted_1
    }

    # -------------------------------------------------------------
    # 反例 2: 严重削波失真注入 (Clipping 破音方波)
    # -------------------------------------------------------------
    print("\n[反例 2] 构造严重削波音频 (模拟增益失控导致爆音)...")
    samples, sr = generate_base_tone(2.0, 24000)
    for i in range(sr // 2, sr):
        samples[i] = 32767 if samples[i] > 0 else -32768
    
    clip_file = f"{MUTATION_DIR}/bad_clipping_square.wav"
    save_wav(clip_file, samples, sr)
    
    res2 = inspect_file(clip_file)
    m2 = res2.get("metrics", {})
    clip_rate = m2.get("clipping_rate_pct", 0)
    intercepted_2 = clip_rate > 0.0
    print(f"  -> 削波率: {clip_rate}% (红线门限: 严格等于 0.00%)")
    print(f"  -> 削波样点数: {m2.get('clipping_sample_count')}")
    print(f"  -> 门禁敏锐拦截断言: {'[状态: 成功拦截]' if intercepted_2 else '[状态: 漏测放行]'}")
    adversarial_results["bad_clipping"] = {
        "case_name": "幅值全满量程削波 (Clipping)",
        "expected_intercept": True,
        "clipping_rate_pct": clip_rate,
        "pass": intercepted_2
    }

    # -------------------------------------------------------------
    # 反例 3: 损坏容器头注入 (0xFFFFFFFF 4GB 假长度)
    # -------------------------------------------------------------
    print("\n[反例 3] 构造 0xFFFFFFFF 假长度损坏 WAV 头 (模拟历史豆包未闭合流)...")
    samples, sr = generate_base_tone(2.0, 24000)
    fake_header_file = f"{MUTATION_DIR}/bad_fake_4gb_header.wav"
    save_wav(fake_header_file, samples, sr, override_header_size=(b'\xff\xff\xff\xff', b'\xff\xff\xff\xff'))
    
    res3 = inspect_file(fake_header_file)
    anomalies = res3.get("header_anomalies", [])
    has_fake_len = any("0xFFFFFFFF" in a.get("message", "") or "4GB" in a.get("message", "") for a in anomalies)
    intercepted_3 = has_fake_len
    print(f"  -> 检出头部异常项数: {len(anomalies)}")
    for a in anomalies:
        print(f"     [{a.get('severity')}] {a.get('message')}")
    print(f"  -> 门禁敏锐拦截断言: {'[状态: 成功拦截]' if intercepted_3 else '[状态: 漏测放行]'}")
    adversarial_results["bad_fake_header"] = {
        "case_name": "WAV 头部 4GB 0xFFFFFFFF 假长度",
        "expected_intercept": True,
        "header_anomalies_count": len(anomalies),
        "pass": intercepted_3
    }

    # -------------------------------------------------------------
    # 反例 4: 硬截断非零直流起振收尾
    # -------------------------------------------------------------
    print("\n[反例 4] 构造首尾硬截断音频 (模拟换源无 Crossfade 产生切歌脉冲)...")
    samples, sr = generate_base_tone(1.0, 24000)
    samples[0] = 16000
    samples[-1] = -16000
    step_file = f"{MUTATION_DIR}/bad_hard_cut_step.wav"
    save_wav(step_file, samples, sr)
    
    res4 = inspect_file(step_file)
    m4 = res4.get("metrics", {})
    head_step = m4.get("head_step_level", 0)
    tail_step = m4.get("tail_step_level", 0)
    trans = res4.get("transient_anomalies", [])
    intercepted_4 = (head_step > 500 or tail_step > 300) and len(trans) > 0
    print(f"  -> 起振阶跃: {head_step} | 收尾阶跃: {tail_step} (红线门限: ≤300)")
    print(f"  -> 瞬态告警数: {len(trans)}")
    for t in trans:
        print(f"     [{t.get('severity')}] {t.get('message')}")
    print(f"  -> 门禁敏锐拦截断言: {'[状态: 成功拦截]' if intercepted_4 else '[状态: 漏测放行]'}")
    adversarial_results["bad_hard_cut"] = {
        "case_name": "首尾硬截断与换源阶跃 (Step > 300)",
        "expected_intercept": True,
        "head_step_level": head_step,
        "tail_step_level": tail_step,
        "pass": intercepted_4
    }

    all_passed = all(v["pass"] for v in adversarial_results.values())
    print("\n=================================================================")
    print(f"🎯 反向对抗变异测试汇总: 共注入 4 类恶性反例 | 全部拦截成功: {all_passed}")
    print("=================================================================")
    
    report_path = "/root/company-qa-workspace/voice-qa-engine/reports/adversarial_mutation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(adversarial_results, f, ensure_ascii=False, indent=2)
    print(f"反向对抗测试证据已落盘至: {report_path}")

if __name__ == "__main__":
    run_adversarial_suite()
