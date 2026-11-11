#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 证据 Manifest 与不可篡改哈希计算
"""

import os
import hashlib
from typing import Dict, Any


def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest(session_dir: str, artifacts: list[str]) -> Dict[str, Any]:
    manifest = {
        "session_dir": os.path.abspath(session_dir),
        "artifacts": {}
    }
    for filename in artifacts:
        full_path = os.path.join(session_dir, filename)
        if os.path.exists(full_path):
            manifest["artifacts"][filename] = {
                "sha256": compute_file_sha256(full_path),
                "size_bytes": os.path.getsize(full_path)
            }
    return manifest
