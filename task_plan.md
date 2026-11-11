# Voice QA Engine V1.2 工程演进跟踪看板 (Task Plan)

- [x] 阶段 1-1：内部统一归一化 float32 PCM 数据模型与格式适配器 (`AudioFormat`, `PCMFormatAdapter`, `AudioResampler`)
- [x] 阶段 1-2：流式 WAV 分帧输入器 (`WavAudioInput`)
- [x] 阶段 2-1：解耦会话与网络状态机 (`SessionState`, `ConnectionState`)
- [x] 阶段 2-2：单调时钟事件时间轴 (`EventTimeline`) 与代次绑定注册表 (`ResponseRegistry`, `CancelToken`)
- [x] 阶段 3-1：单一 Actor 串行化防幽灵音频播放引擎 (`PlaybackEngine`)
- [x] 阶段 4-1：工业级六态断言模型 (`AssertionResult`, `TestStatus`)
- [x] 阶段 4-2：打断与双打断场景断言门禁 (`BargeInScenario`, `RapidDoubleBargeInScenario`)
- [x] 阶段 5-1：五大必备物理证据记录器与 SHA-256 Manifest (`EvidenceRecorder`, `generate_manifest`)
- [x] 阶段 6-1：WebSocket 异步客户端与并发收发任务 (`MossLiveClient`)
- [x] 阶段 6-2：MOSS-Live 协议双向映射适配器 (`MossLiveProtocolAdapter`)
- [x] 阶段 6-3：正常问答场景执行器 (`NormalQAScenario`)
- [x] 阶段 6-4：客观物理时延计算器与产物校验 (`LatencyEvaluator`, `EvidenceValidator`)
- [x] 阶段 7-1：CLI 入口集成与全双工端到端演示 (`voice_eval_cli.py`)
- [x] 阶段 7-2：全量回归测试套件与编码防护审查
- [x] 阶段 7-3：Git 提交并安全推送到 GitHub
