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
- [x] 阶段 8-1：飞书评测规范 (v1.4.0/v1.4.1) 核心 SLA 对齐（附和倾听精准度 ≥ 98.5%、思考留白 ≤ 1.5%、刹车时延 P50 ≤ 200ms/P95 ≤ 300ms/硬上限 ≤ 350ms）
- [x] 阶段 8-2：防爆音物理硬门禁：实现 `HannFadeoutSink` 10ms 半 Hann 窗渐出，确保 Jump ≤ 8000
- [x] 阶段 8-3：四态对话断点状态栈：实现 `ContextCallStack`（Resume Full、Drop Stale、Merge Context、Timeout Expire）
- [x] 阶段 8-4：MOSS-Live 专项对抗场景库扩展（`BackchannelScenario`、`HesitationToleranceScenario`、`AudioImpulseNoiseScenario`）
- [x] 阶段 8-5：旧分片拒绝率与客观物理时延计算模型优化，修复 CLI 双重执行 Bug 与直接运行 pytest 模块路径问题

# Voice QA Engine V1.3 对抗增强与统计置信 (对照飞书评测规范 v1.4.0)

- [x] 阶段 V13-1：基础设施 — `ScenarioRegistry`/`SuiteRunner` 统一执行器、`StreamingAudioInput` 全时推流源、`EvidenceRecorder` 流式落盘模式、`TOOL_CALLED/TOOL_RESULTED` 协议事件
- [x] 阶段 V13-2：LLM 驱动双 Agent 对弈 — `LLMPoweredVUA`(环境变量 key)/`ScriptedVUA` + `ToolExecutionLedger` 假完工一票否决 + `SparringScenario` 打断改参/断点接续
- [x] 阶段 V13-3：声纹排他拒识 — `OverlapSpeechSynthesizer` 多人争吵合成 (SNR 定标) + `SpeakerExclusionScenario` 下行 0 字节/定向唤醒
- [x] 阶段 V13-4：30 分钟黄金基线 soak — 6 阶段编排/确定性注入/RSS 漂移 ≤10%/时间压缩比
- [x] 阶段 V13-5：对抗增强包 — 不流利/倒吸气耐受、碰撞窗口分级 + Floor Yielding、谐音误唤醒矩阵、停算熔断断言 (≤15ms/≤3 tokens)
- [x] 阶段 V13-6：增强包 — Wilson 统计置信 (n≥200 门禁)、scorecard 一页大盘 (含防篡改联动)、CI 回归 workflow、AEC 自激边界扫描、语料健康度矩阵 (27 组)
- [x] 阶段 V13-7：全量回归 101 passed + CLI `--v13`/`--scorecard` 冒烟验证 + 文档同步
