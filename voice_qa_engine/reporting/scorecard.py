#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 一页指标大盘 (Scorecard Aggregator)
扫描 reports/**/session_*/{metrics.json, session.json, manifest.json}，
经 EvidenceValidator 防篡改校验后聚合全部 SLA 判定，输出 scorecard.json + 终端矩阵。
SLA 口径对齐评测规范 v1.4.0。
"""

import os
import glob
import json
from typing import Dict, Any, List, Optional

from voice_qa_engine.evaluators.evidence_validator import EvidenceValidator

# 已知 SLA 口径表: metric_key -> (sla, 方向 upper|lower, 单位, 名称)
SLA_TABLE: Dict[str, Dict[str, Any]] = {
    "ttft_ms": {"sla": 600.0, "direction": "upper", "unit": "ms", "name": "TTFT 首响应时延"},
    "first_audio_latency_ms": {"sla": 700.0, "direction": "upper", "unit": "ms", "name": "首音频到达时延"},
    "e2e_stop_latency_ms": {"sla": 300.0, "direction": "upper", "unit": "ms", "name": "打断刹车时延 P95"},
    "local_stop_latency_ms": {"sla": 200.0, "direction": "upper", "unit": "ms", "name": "本地刹车时延"},
    "queue_flush_latency_ms": {"sla": 20.0, "direction": "upper", "unit": "ms", "name": "队列清空残存"},
    "tool_abort_latency_ms": {"sla": 10.0, "direction": "upper", "unit": "ms", "name": "工具熔断时延"},
    "inference_abort_ms": {"sla": 15.0, "direction": "upper", "unit": "ms", "name": "停算熔断时延"},
    "hann_fadeout_max_jump": {"sla": 8000.0, "direction": "upper", "unit": "", "name": "Hann 渐出爆音 Jump"},
}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def scan_sessions(reports_root: str) -> List[Dict[str, Any]]:
    """扫描全部会话目录，提取 (scenario, status, evidence_valid, metrics)"""
    sessions: List[Dict[str, Any]] = []
    pattern = os.path.join(reports_root, "**", "session.json")
    for session_path in glob.glob(pattern, recursive=True):
        session_dir = os.path.dirname(session_path)
        try:
            with open(session_path, encoding="utf-8") as f:
                session_meta = json.load(f)
            metrics_path = os.path.join(session_dir, "metrics.json")
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, encoding="utf-8") as f:
                    metrics = json.load(f)
            valid, reason, _ = EvidenceValidator.validate_session_dir(session_dir)
            sessions.append({
                "session_dir": session_dir,
                "scenario_id": session_meta.get("scenario_id", "unknown"),
                "status": session_meta.get("status", "UNKNOWN"),
                "evidence_valid": valid,
                "evidence_reason": reason,
                "metrics": metrics,
            })
        except (OSError, json.JSONDecodeError) as exc:
            sessions.append({
                "session_dir": session_dir,
                "scenario_id": "unreadable",
                "status": "ERROR",
                "evidence_valid": False,
                "evidence_reason": f"{type(exc).__name__}: {exc}",
                "metrics": {},
            })
    return sessions


def build_scorecard(reports_root: str) -> Dict[str, Any]:
    sessions = scan_sessions(reports_root)
    scenario_summary: Dict[str, Dict[str, Any]] = {}
    latency_samples: Dict[str, List[float]] = {}

    for s in sessions:
        sid = s["scenario_id"]
        entry = scenario_summary.setdefault(sid, {
            "total": 0, "statuses": {}, "evidence_valid_ratio": [],
        })
        entry["total"] += 1
        entry["statuses"][s["status"]] = entry["statuses"].get(s["status"], 0) + 1
        entry["evidence_valid_ratio"].append(1.0 if s["evidence_valid"] else 0.0)

        for key, value in _flatten_metrics(s["metrics"]):
            # 时延负值物理不可能（历史会话时钟基准偏差）→ 采样时过滤，防污染分位数
            if key in SLA_TABLE and _numeric(value) and value > 0:
                latency_samples.setdefault(key, []).append(float(value))

    sla_matrix: List[Dict[str, Any]] = []
    for key, spec in SLA_TABLE.items():
        values = latency_samples.get(key, [])
        p50, p95 = _percentile(values, 50), _percentile(values, 95)
        critical = p95 if spec["direction"] == "upper" else p50
        verdict = None
        if critical is not None:
            verdict = ("PASS" if critical <= spec["sla"] else "FAIL") if spec["direction"] == "upper" else \
                      ("PASS" if critical >= spec["sla"] else "FAIL")
        sla_matrix.append({
            "metric": key,
            "name": spec["name"],
            "sla": spec["sla"],
            "direction": spec["direction"],
            "unit": spec["unit"],
            "samples": len(values),
            "p50": p50,
            "p95": p95,
            "verdict": verdict or "NO_DATA",
        })

    statuses = [s["status"] for s in sessions]
    overall = "NO_DATA"
    if statuses:
        if "FAIL" in statuses or "ERROR" in statuses:
            overall = "FAIL"
        elif "INCONCLUSIVE" in statuses or "NOT_RUN" in statuses:
            overall = "INCONCLUSIVE"
        elif all(v == "SIMULATED_PASS" for v in statuses):
            overall = "SIMULATED_PASS"
        else:
            overall = "PASS"

    return {
        "reports_root": reports_root,
        "total_sessions": len(sessions),
        "overall_status": overall,
        "evidence_tampered_sessions": [s["session_dir"] for s in sessions if not s["evidence_valid"]],
        "scenarios": {
            sid: {
                "total": e["total"],
                "statuses": e["statuses"],
                "evidence_valid_ratio": round(
                    sum(e["evidence_valid_ratio"]) / len(e["evidence_valid_ratio"]), 3
                ) if e["evidence_valid_ratio"] else None,
            }
            for sid, e in scenario_summary.items()
        },
        "sla_matrix": sla_matrix,
    }


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = ""):
    for key, value in metrics.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten_metrics(value, prefix=f"{full_key}.")
        elif key in SLA_TABLE and _numeric(value):
            # 顶层或嵌套末段 key 命中 SLA 表（全路径仅在非命中时保留语义，此处避免双计）
            yield full_key, value


def render_scorecard_text(scorecard: Dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        f"Voice QA Engine 一页指标大盘  |  总判定: {scorecard['overall_status']}  "
        f"|  会话数: {scorecard['total_sessions']}",
        "=" * 78,
        f"{'指标':<24}{'SLA':>10}{'P50':>12}{'P95':>12}{'样本':>8}  {'判定':>6}",
        "-" * 78,
    ]
    for row in scorecard["sla_matrix"]:
        lines.append(
            f"{row['name']:<24}"
            f"{row['sla']:>8.1f}{row['unit']:<2}"
            f"{('%0.2f' % row['p50']) if row['p50'] is not None else 'N/A':>12}"
            f"{('%0.2f' % row['p95']) if row['p95'] is not None else 'N/A':>12}"
            f"{row['samples']:>8}  {row['verdict']:>6}"
        )
    lines.append("-" * 78)
    for sid, e in scorecard["scenarios"].items():
        lines.append(f"  {sid:<36} n={e['total']:<4} 状态分布: {e['statuses']}  "
                     f"证据可信率: {e['evidence_valid_ratio']}")
    if scorecard["evidence_tampered_sessions"]:
        lines.append(f"  ⚠ 篡改/证据缺失会话: {scorecard['evidence_tampered_sessions']}")
    lines.append("=" * 78)
    return "\n".join(lines)
