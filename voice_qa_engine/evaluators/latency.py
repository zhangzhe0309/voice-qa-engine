#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.2 - 客观单调时钟物理时延计算器
严格依据事件单调时间戳 (monotonic_ns) 计算 TTFT、首音频到达时延与本地刹车停止时延
"""

from typing import List, Dict, Any, Optional
from voice_qa_engine.clients.protocol_models import RuntimeEvent, EventType


class LatencyEvaluator:
    @staticmethod
    def calculate_latencies(events: List[RuntimeEvent]) -> Dict[str, Any]:
        metrics = {
            "ttft_ms": None,
            "first_audio_chunk_ms": None,
            "local_stop_latency_ms": None,
            "queue_flush_latency_ms": None,
            "post_interrupt_old_chunk_count": 0,
            "post_interrupt_old_chunk_rejected_count": 0,
            "old_chunk_rejection_rate": None
        }

        # 找到输入结束时间点
        input_end_events = [e for e in events if e.event_type == EventType.INPUT_AUDIO_END]
        input_end_ns = input_end_events[0].monotonic_ns if input_end_events else None

        if input_end_ns:
            # 1. 计算 TTFT (首文本/首事件时延)
            first_text_events = [
                e for e in events
                if e.event_type in (EventType.TEXT_DELTA, EventType.RESPONSE_STARTED)
                and e.monotonic_ns >= input_end_ns
            ]
            if first_text_events:
                metrics["ttft_ms"] = (first_text_events[0].monotonic_ns - input_end_ns) / 1_000_000.0

            # 2. 计算首个有效音频分片到达时延
            first_audio_events = [
                e for e in events
                if e.event_type == EventType.AUDIO_CHUNK
                and e.monotonic_ns >= input_end_ns
            ]
            if first_audio_events:
                metrics["first_audio_chunk_ms"] = (first_audio_events[0].monotonic_ns - input_end_ns) / 1_000_000.0

        # 3. 计算打断本地刹车停止时延 (local_stop_latency_ms)
        interrupt_events = [e for e in events if e.event_type == EventType.PLAYBACK_INTERRUPT_REQUESTED]
        if interrupt_events:
            int_ns = interrupt_events[0].monotonic_ns
            # 队列清空耗时
            flush_events = [e for e in events if e.event_type == EventType.PLAYBACK_QUEUE_FLUSHED and e.monotonic_ns >= int_ns]
            if flush_events:
                metrics["queue_flush_latency_ms"] = (flush_events[0].monotonic_ns - int_ns) / 1_000_000.0

            # 硬件 Sink 停止耗时
            sink_stop_events = [e for e in events if e.event_type == EventType.PLAYBACK_SINK_STOPPED and e.monotonic_ns >= int_ns]
            if sink_stop_events:
                metrics["local_stop_latency_ms"] = (sink_stop_events[0].monotonic_ns - int_ns) / 1_000_000.0

            # 统计打断后到达的旧 Chunk 拒绝情况
            rejected_events = [e for e in events if e.event_type == EventType.AUDIO_CHUNK_REJECTED and e.monotonic_ns >= int_ns]
            metrics["post_interrupt_old_chunk_rejected_count"] = len(rejected_events)
            metrics["post_interrupt_old_chunk_count"] = len(rejected_events)

            if metrics["post_interrupt_old_chunk_count"] > 0:
                metrics["old_chunk_rejection_rate"] = 1.0  # 全部被拒绝
            else:
                metrics["old_chunk_rejection_rate"] = None  # 未观察到旧Chunk到达，不能虚标100%

        return metrics
