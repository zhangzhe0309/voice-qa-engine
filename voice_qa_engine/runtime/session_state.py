#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 会话与连接生命周期状态机
解耦会话执行生命周期 (SessionState) 与网络物理连接状态 (ConnectionState)
"""

from enum import Enum


class SessionState(str, Enum):
    CREATED = "created"
    CONNECTING = "connecting"
    HANDSHAKING = "handshaking"
    READY = "ready"
    RUNNING = "running"
    DRAINING = "draining"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"
    FAILED = "failed"
