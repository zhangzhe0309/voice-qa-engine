#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice QA Engine V1.3 - 语料前置标定矩阵 (Corpus Calibration Matrix)
对应评测规范 第 7 节：语料工场防带坑门禁。
- 方言口音对抗覆盖（平翘舌/鼻边音/粤普/儿化/吴语）
- 极端语速与声线（1.5x 抢话/0.7x 结巴/儿童高音/老人沙哑）
- 恶意谐音同音挑衅矩阵（"模式设计/摸索前进/魔术表演/磨合期" 误唤醒必须 0.0%）
每组用例入库前必须过标准 ASR 校验：普通话基准样本 CER = 0.0% 才算合格。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CorpusCase:
    case_id: str
    category: str          # dialect | speed_voice | homophone_trap
    text: str
    expected_wake: bool    # 是否应当触发唤醒/响应
    expected_cer: float = 0.0  # 入库前 ASR 校验基准
    notes: str = ""


# 恶意谐音同音挑衅集：与唤醒词"小模"构成同音/近音干扰
HOMOPHONE_TRAPS: List[str] = [
    "这个模式设计得不错",
    "我们先摸索前进",
    "魔术表演开始了",
    "项目还在磨合期",
    "模棱两可的说法不行",
    "莫莫莫，先等等",
]

DIALECT_TEXTS: List[Dict[str, str]] = [
    {"subcat": "平翘舌不分", "text": "你老师和李老师四谁负责"},        # 是→四
    {"subcat": "鼻边音不分", "text": "来农，把牛奶拿来"},             # 龙→农
    {"subcat": "粤普前后鼻音", "text": "呢个方案好唔好，你话事"},
    {"subcat": "北方儿化快语速", "text": "今儿个咱把这活儿干完嘞"},
    {"subcat": "江浙吴语调", "text": "格个办法灵个灵个"},
]

SPEED_VOICE_PROFILES: List[Dict[str, Any]] = [
    {"subcat": "1.5倍速抢话连读", "speed": 1.5, "voice": "adult_male"},
    {"subcat": "0.7倍速结巴犹豫", "speed": 0.7, "voice": "adult_female"},
    {"subcat": "儿童高音调", "speed": 1.1, "voice": "child"},
    {"subcat": "老人低沉沙哑", "speed": 0.85, "voice": "elder"},
]


class CorpusMatrix:
    """24 组方言用例矩阵构建器（方言 5 组 × 声线 4 组展开 + 谐音挑衅 6 组 + 正例）"""

    def build_cases(self) -> List[CorpusCase]:
        cases: List[CorpusCase] = []
        # 方言 × 声线交叉展开为 20 组
        for d in DIALECT_TEXTS:
            for v in SPEED_VOICE_PROFILES:
                cases.append(CorpusCase(
                    case_id=f"dialect_{len(cases) + 1:02d}",
                    category="dialect",
                    text=d["text"],
                    expected_wake=True,
                    notes=f"{d['subcat']} × {v['subcat']} (speed={v['speed']}, voice={v['voice']})",
                ))
        # 谐音挑衅 6 组：绝不允许误唤醒
        for i, text in enumerate(HOMOPHONE_TRAPS):
            cases.append(CorpusCase(
                case_id=f"homophone_trap_{i + 1:02d}",
                category="homophone_trap",
                text=text,
                expected_wake=False,
                notes="恶意谐音同音挑衅，误唤醒必须为 0.0%",
            ))
        # 普通话基准正例（CER=0.0% 门禁基准）
        cases.append(CorpusCase(
            case_id="mandarin_baseline_01",
            category="baseline",
            text="小模，帮我把五天方案排出来",
            expected_wake=True,
            notes="普通话基准样本，入库 CER 门禁 = 0.0%",
        ))
        return cases

    @staticmethod
    def corpus_health_report(cases: List[CorpusCase], asr_results: Dict[str, float]) -> Dict[str, Any]:
        """
        语料健康度：asr_results: {case_id: 实测 CER}。
        普通话基准必须 CER=0.0%；超门禁用例列入 blocked 清单。
        """
        passed, blocked = [], []
        for c in cases:
            cer = asr_results.get(c.case_id)
            entry = {"case_id": c.case_id, "category": c.category, "cer": cer}
            if cer is None:
                blocked.append({**entry, "reason": "missing_asr_result"})
            elif cer > c.expected_cer:
                blocked.append({**entry, "reason": f"cer_exceeds_{c.expected_cer}"})
            else:
                passed.append(entry)
        return {
            "total_cases": len(cases),
            "passed": passed,
            "blocked": blocked,
            "healthy": not blocked,
        }
