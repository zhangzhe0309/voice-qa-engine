#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 证据链完整性与防伪校验器
对 Session 目录下的物理产物执行存在性与 SHA-256 哈希比对
"""

import os
import json
from typing import Dict, Any, Tuple
from voice_qa_engine.evidence.manifest import compute_file_sha256


class EvidenceValidator:
    REQUIRED_FILES = [
        "input_audio.wav",
        "output_audio.wav",
        "events.jsonl",
        "metrics.json",
        "session.json",
        "manifest.json"
    ]

    @classmethod
    def validate_session_dir(cls, session_dir: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        验证指定测试会话目录证据链
        返回: (is_valid, reason, details)
        """
        if not os.path.exists(session_dir):
            return False, f"目录不存在: {session_dir}", {}

        # 1. 检查必备文件是否存在
        for filename in cls.REQUIRED_FILES:
            filepath = os.path.join(session_dir, filename)
            if not os.path.exists(filepath):
                return False, f"必备证据文件缺失: {filename}", {}

        # 2. 读取并核验 manifest.json
        manifest_path = os.path.join(session_dir, "manifest.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            return False, f"Manifest JSON 解析异常: {e}", {}

        artifacts = manifest_data.get("artifacts", {})
        if not artifacts:
            return False, "Manifest 未记录任何 artifacts (空转校验无效)", {}
        missing_in_manifest = [f for f in cls.REQUIRED_FILES if f != "manifest.json" and f not in artifacts]
        if missing_in_manifest:
            return False, f"Manifest 未覆盖必备产物: {missing_in_manifest}", {}
        hash_results = {}

        for filename, info in artifacts.items():
            expected_hash = info.get("sha256")
            filepath = os.path.join(session_dir, filename)
            if not os.path.exists(filepath):
                return False, f"Manifest 记录的文件未落盘: {filename}", {}

            actual_hash = compute_file_sha256(filepath)
            if actual_hash != expected_hash:
                return False, f"文件哈希校验失败 (篡改嫌疑): {filename}", {
                    "expected": expected_hash,
                    "actual": actual_hash
                }
            hash_results[filename] = "matched"

        return True, "证据链完整且哈希校验一致", hash_results
