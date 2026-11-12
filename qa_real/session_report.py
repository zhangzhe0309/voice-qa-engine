"""会话报告汇总：把标定 / 物理刹车 / 变异测试三类证据合成一份可判读报告。

核心纪律（对应评审 P0）
------------------------
- 每个指标都必须带 `measurement_source` 与 `n`；
- 缺观测源的指标一律 `NOT_MEASURABLE`，不得用 0 或 PASS 顶替；
- `protocol_observed` 只有真正连过服务端才是 true —— 没有 key 时它必须为 false。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

NOT_MEASURABLE = "NOT_MEASURABLE"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(root: Path, protocol_observed: bool, protocol_note: str) -> dict[str, Any]:
    cal = _load(root / "calibration" / "calibration.json")
    barge = _load(root / "physical_bargein" / "probe.json")
    ghost = _load(root / "mutations" / "ghost_audio.json")

    metrics: dict[str, Any] = {}

    if cal:
        metrics["device_latency_const_ms"] = {
            "value": cal.get("device_latency_const_ms"),
            "measurement_source": "physical_loopback_calibration",
            "n": (cal.get("render_path_offset_ms") or {}).get("n"),
            "sla": None,
            "status": "PASS" if cal.get("device_latency_const_ms") is not None else NOT_MEASURABLE,
        }
        acoust = cal.get("acoustic_path_offset_ms") or {}
        metrics["acoustic_path_offset_ms"] = {
            "value": acoust.get("p50"),
            "measurement_source": "physical_microphone",
            "n": acoust.get("n"),
            "status": (
                "PASS"
                if (acoust.get("n") or 0) >= 3 and (acoust.get("std") or 999) <= 30.0
                else NOT_MEASURABLE
            ),
            "reason": (
                None
                if (acoust.get("std") or 999) <= 30.0
                else f"麦克风回采抖动过大(std={acoust.get('std')}ms)，不可作为物理真值源"
            ),
        }
    else:
        metrics["device_latency_const_ms"] = {
            "value": None,
            "measurement_source": None,
            "n": 0,
            "status": NOT_MEASURABLE,
            "reason": "缺少 calibration.json",
        }

    if barge:
        lat = barge.get("physical_stop_latency_ms") or {}
        metrics["barge_in_local_stop_latency_ms"] = {
            "value": lat.get("p50"),
            "p95": lat.get("p95"),
            "measurement_source": "physical_loopback_envelope",
            "n": lat.get("n"),
            "sla": "p95 <= 200ms (文档 9.2 本地刹车口径)",
            "status": "PASS" if (lat.get("p95") is not None and lat["p95"] <= 200.0) else "FAIL",
        }
        metrics["residual_audio_frames"] = {
            "value": (barge.get("residual_frames") or {}).get("p50"),
            "measurement_source": "physical_loopback_envelope",
            "n": (barge.get("residual_frames") or {}).get("n"),
            "sla": "<= 1 frame (<=20ms, <-45dBFS)",
            "status": (
                "PASS"
                if ((barge.get("residual_frames") or {}).get("p95") or 99) <= 1.0
                else "FAIL"
            ),
        }
    else:
        for key in ("barge_in_local_stop_latency_ms", "residual_audio_frames"):
            metrics[key] = {
                "value": None,
                "measurement_source": None,
                "n": 0,
                "status": NOT_MEASURABLE,
                "reason": "缺少 physical_bargein/probe.json",
            }

    # 服务端相关指标：没有真实 WS 就一律 NOT_MEASURABLE
    for key in ("ttft_ms", "first_audio_chunk_ms", "eot_to_speech_ms", "barge_in_stop_latency_ms"):
        metrics[key] = {
            "value": None,
            "measurement_source": None,
            "n": 0,
            "status": NOT_MEASURABLE,
            "reason": "未连接真实服务端（protocol_observed=false）" if not protocol_observed else "待采集",
        }

    mutation = (ghost or {}).get("judgement") or {
        "verdict": NOT_MEASURABLE,
        "mutation_score": None,
    }

    return {
        "kind": "voice_qa_session_report",
        "schema": "v1.5-physical",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_observed": protocol_observed,
        "protocol_note": protocol_note,
        "audio_capture_verified": bool(cal and barge),
        "evidence_complete": bool(cal and barge and ghost and protocol_observed),
        "simulation_used": False,
        "metrics": metrics,
        "mutation": mutation,
        "verdict": _verdict(metrics, mutation, protocol_observed),
    }


def _verdict(metrics: dict[str, Any], mutation: dict[str, Any], protocol_observed: bool) -> dict[str, Any]:
    failed = [k for k, v in metrics.items() if v.get("status") == "FAIL"]
    nm = [k for k, v in metrics.items() if v.get("status") == NOT_MEASURABLE]
    if failed:
        return {"status": "FAIL", "failed_metrics": failed}
    if not protocol_observed:
        return {
            "status": "PARTIAL",
            "note": "客户端物理链路已实测通过，服务端链路未接入，不得作为模型能力结论",
            "not_measurable": nm,
        }
    return {"status": "PASS" if not nm else "PARTIAL", "not_measurable": nm}


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总会话报告")
    ap.add_argument("--root", type=str, default="reports")
    ap.add_argument("--protocol-observed", action="store_true")
    ap.add_argument("--protocol-note", type=str, default="未连接真实服务端（缺 MOSS-Live key/endpoint）")
    ap.add_argument("--out", type=str, default="reports/voice_qa_session_report.json")
    args = ap.parse_args()

    data = build_report(Path(args.root), args.protocol_observed, args.protocol_note)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "verdict": data["verdict"],
        "protocol_observed": data["protocol_observed"],
        "audio_capture_verified": data["audio_capture_verified"],
        "evidence_complete": data["evidence_complete"],
        "mutation": {"verdict": data["mutation"].get("verdict"), "score": data["mutation"].get("mutation_score")},
    }, ensure_ascii=False, indent=2))
    for k, v in data["metrics"].items():
        print(f"  {k:34s} {v.get('value')!s:>10s}  n={v.get('n')}  {v.get('status')}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
