# Voice QA Engine (工业级全双工语音交互质量评估套件)

面向实时全双工语音大模型（Full-Duplex Voice AI，如 MOSS-Live、Moshi、GPT-4o Realtime）及多模态实时语音智能体的端到端工业级自动化评测套件。

全面跳出传统语音接口“单向请求、文本转写、字错率统计”的单向死胡同，将测试范围从单向 HTTP/SSE 扩展为**双向音频流、物理时延解耦、打断状态机、播放队列清空与对抗对弈**的真实端到端交互评估。

---

## 核心特性与测试域

### 1. 四段真实物理时延解耦与测量模型
全双工系统下，模型常在用户发声未完全结束时开始预测，传统以“用户话毕”作为唯一统计起点的方法已失效。本套件将时序严格解耦为四段物理指标：
- **TTFT (Time To First Text / Event)**: 从用户有效语音截止到接收到首个服务端文本/控制事件时延（SLA ≤ 600ms）。
- **First Audio Chunk Latency**: 接收到首个下行 PCM 音频分片到达时延（SLA ≤ 700ms）。
- **EOT-to-Speech (扬声器物理出声时延)**: 从用户话毕到扬声器真实产生震动出声耗时，单独考核客户端播放管线时滞（SLA ≤ 850ms）。
- **Barge-in Stop Latency (打断刹车时延)**: 用户插话起点到扬声器旧回复彻底哑音的物理时延（P50 ≤ 200ms, P95 ≤ 300ms）。

### 2. 状态机与播放回执硬断言
- **全时上行推流 (Continuous Audio Ingestion)**: 16kHz PCM 单声道持续收音，AI 播放期间麦克风保持全时收音，静音期持续发送静音帧填充维持长连接与上下文。
- **待播队列清空断言 (AudioContext Queue Flush)**: 收到打断信号 (`response.interrupted`) 时，硬断言客户端待播队列必须物理清零（`pending_queue_size == 0`），残余溢出音频 ≤ 1 帧（≤ 20ms），彻底消除残存半个字的爆音与漏音。
- **真实播放进度回执**: 客户端依照真实音频采样率与物理播放进度上报 `playback.chunk.completed`，严禁测试客户端提前虚假确认，防止服务端会话状态机失真。

### 3. 三阶打断对抗测试矩阵
- **正向打断 (True Barge-in)**: 用户发出实质新指令，AI 必须在 ≤ 300ms 刹车并转向新话题（通过率 ≥ 98%）。
- **微弱附和免打断 (Backchanneling)**: 用户倾听中发出“嗯、对、好的”弱能量短语（≤ 350ms, -20dBFS），AI 保持平稳陈述，严禁触发非预期停顿（精准度 ≥ 98.5%）。
- **思考停顿防误抢 (Hesitation & Pause Tolerance)**: 模拟人类 0.5s ~ 2.5s 正常留白思考，AI 保持 Holding 倾听，严禁过早插话切断用户语句（误抢话率 ≤ 1.5%）。

### 4. 双 Agent 对弈与真实业务闭环核验 (借鉴 ServiceNow EVA)
- 引入 **Virtual User Agent** 动态对弈被测系统，自适应推进 10~30 轮深水多轮对话。
- 引入**确定性工具物理状态拦截器**，对任务闭环进行真实环境状态断言，一票否决“口头承诺完成但后台未实际落盘”的假完工缺陷。

### 5. 物理声学质检与网络韧性
- **声学跳变与爆音扫描**: 16-bit PCM 单步一阶差分突变 Jump ≤ 8,000，削波率 0.00%，直流偏置 < 5.0。
- **弱网混沌断言 (tc-netem)**: 15% 丢包与 150ms 抖动下，验证 JitterBuffer 队列无饥饿且打断 Flush 正常。

### 6. V1.3 对抗增强与统计置信 (评测规范 v1.4.0 对齐)
- **双 Agent LLM 对弈**: `LLMPoweredVUA`（OpenAI 兼容接口，key 走环境变量 `VQA_LLM_API_KEY`，无 key 拒绝伪造 → NOT_RUN/脚本模式标 SIMULATED_PASS）+ `ToolExecutionLedger` 声称/落盘比对，假完工一票否决。
- **声纹排他拒识**: 背景多人争吵（SNR 5dB 合成）下行 0 字节 + 定向唤醒 ≤ 600ms；无唤醒证据 → INCONCLUSIVE 不虚标。
- **30 分钟黄金基线 soak**: 6 阶段时序编排 + 确定性随机注入（15 打断/7 支线/20 附和）+ psutil RSS 漂移 ≤ 10% + 时间压缩比参数化。
- **不流利与倒吸气耐受**: 80ms 倒吸气前置压麦 + 否定纠偏脏参数 0 写入；误抢话率走 Wilson 置信上界判定（n ≥ 200 才具统计判定力，否则 INCONCLUSIVE）。
- **谐音误唤醒矩阵**: 6 组谐音陷阱 × 声线交叉注入，误唤醒严格 0 + 样本量门禁。
- **碰撞窗口分级**: 0-150ms 无害交叠 / 150-250ms 决策缓冲 / > 250ms 恶性碰撞；Floor Yielding ≥ 98%。
- **AEC 自激边界扫描**: 回灌延迟 0~300ms × 增益 0~1 网格逐点 ERLE 评估，输出自激失效边界曲线。
- **一页指标大盘**: `--scorecard` 聚合全部会话 SLA P50/P95 矩阵 + EvidenceValidator 防篡改联动；篡改会话一票否决。

---

## 快速开始

### 环境依赖
- Python 3.10+
- `numpy`, `requests`

### 执行全链路基线测试
```bash
python3 voice_eval_cli.py
```

### 执行 V1.3 对抗增强场景套件 (对弈/拒识/soak/不流利/谐音)
```bash
python3 voice_eval_cli.py --v13

# LLM 驱动对弈（可选）: 配置环境变量后自动切换 LLM VUA
export VQA_LLM_API_KEY=sk-...        # OpenAI 兼容
export VQA_LLM_BASE_URL=https://...  # 可选，默认 OpenAI
export VQA_LLM_MODEL=gpt-4o-mini     # 可选
```

### 生成一页指标大盘 (SLA 矩阵 + 证据防篡改)
```bash
python3 voice_eval_cli.py --scorecard
```

---

## 模块拓扑

```text
├── full_duplex_evaluator.py      # 全双工四段时延、打断Flush断言与三阶对抗引擎
├── voice_eval_cli.py             # 综合评测 CLI 与调度总控 (--v12 / --v13 / --scorecard)
├── acoustic_inspector.py         # 物理声学质量、爆音跳跃与削波扫描
├── latency_benchmark.py          # TTS / ASR / VAD 首包时延与实时率 (RTF) 基准
├── noise_evaluator.py            # 环境噪声注入、SNR 衰减与 CER 字错率评估
├── mfa_phoneme_aligner.py        # MFA 音素级强制对齐注水
├── tc_netem_chaos_evaluator.py   # 弱网丢包抖动与混沌评测
├── webrtc_getstats_inspector.py  # WebRTC getStats 实时统计巡检
├── voice_qa_engine/
│   ├── scenarios/                # 场景库: 正常问答/打断/附和/对弈/拒识/soak/不流利/谐音
│   │   └── registry.py           #   ScenarioRegistry + SuiteRunner 统一执行器
│   ├── agents/                   # LLM 驱动 Virtual User Agent + 工具物理状态拦截器
│   ├── runtime/                  # 会话状态机/事件时间轴/上下文栈/轮次仲裁
│   ├── audio/                    # 流式输入源/重叠语音合成/语料矩阵
│   ├── evidence/                 # 五大物理证据流式落盘 + SHA-256 Manifest
│   ├── evaluators/               # 六态断言/时延/AEC/统计置信/证据校验
│   └── reporting/                # scorecard 一页指标大盘
├── .github/workflows/regression.yml  # CI 回归: pytest + v13 套件 + SLA 漂移门禁
├── docs/                         # 详细评测原理与规范说明
└── reports/                      # 自动化测试输出基准报告 (JSON)
```

---

## 授权协议
MIT License
