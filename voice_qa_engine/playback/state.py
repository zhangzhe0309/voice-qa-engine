#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 播放状态机枚举
"""

from enum import Enum


class PlaybackState(str, Enum):
    IDLE = "idle"
    BUFFERING = "buffering"
    PLAYING = "playing"
    INTERRUPTING = "interrupting"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
