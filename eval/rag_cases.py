# -*- coding: utf-8 -*-
"""三类 RAG 测试样本集——用于四维评估。

视频提到：准备三类测试样本（易误读、冲突、拒答），
从召回命中、引用覆盖、答案忠实性、人工复核四个维度评估。

每类样本重点暴露一条链路损耗：
  - 易误读样本：固定切片可能截断语义反转 → 考验上下文组装
  - 冲突样本：多来源权威/版本矛盾 → 考验冲突检测 + 权威度加权
  - 拒答样本：知识库无答案 → 考验生成阶段忠实性（是否肯说"不知道"）
"""

from dataclasses import dataclass, field


@dataclass
class RAGCase:
    question: str
    category: str            # "misread" | "conflict" | "refuse"
    # 该 case 期望的检索片段（用于召回命中评估）
    expected_chunks: list[str] = field(default_factory=list)
    # 该 case 的"合理答案"（用于人工复核参照）
    expected_answer_note: str = ""
    # 冲突样本特有：期望是否检测到冲突
    expect_conflict: bool = False


# ------------------------------------------------------------
# 1. 易误读样本（misread）
# ------------------------------------------------------------
MISREAD_CASES = [
    RAGCase(
        question="根据这份合同，乙方的付款期限是多久？",
        category="misread",
        expected_chunks=["付款期限", "收到发票后", "日内"],
        expected_answer_note="重点：完整条款含'收到发票后 30 日内'，若切片截断到'收到发票后'会丢失期限天数。",
    ),
    RAGCase(
        question="本产品的质保期是否覆盖意外损坏？",
        category="misread",
        expected_chunks=["质保", "意外损坏", "不覆盖", "覆盖"],
        expected_answer_note="重点：'质保不覆盖意外损坏'若切片截断为'质保…意外损坏'会语义反转。",
    ),
]

# ------------------------------------------------------------
# 2. 冲突样本（conflict）
# ------------------------------------------------------------
CONFLICT_CASES = [
    RAGCase(
        question="2024年第三季度公司营收是多少？",
        category="conflict",
        expected_chunks=["2024年", "Q3", "营收"],
        expected_answer_note="重点：经营快报说12.5亿，审计报告说8.3亿，应标注分歧并倾向审计。",
        expect_conflict=True,
    ),
    RAGCase(
        question="公司现行的员工休假制度是几天？",
        category="conflict",
        expected_chunks=["休假", "年假", "制度"],
        expected_answer_note="重点：旧制度15天 vs 新制度(v2)20天，应倾向新版本。",
        expect_conflict=True,
    ),
]

# ------------------------------------------------------------
# 3. 拒答样本（refuse）
# ------------------------------------------------------------
REFUSE_CASES = [
    RAGCase(
        question="公司明年的分红预案是什么？",
        category="refuse",
        expected_chunks=[],
        expected_answer_note="重点：知识库无该信息时，应输出'无法确定/需人工确认'，不得编造分红方案。",
    ),
    RAGCase(
        question="管理层对AI转型的最新内部决议是什么？",
        category="refuse",
        expected_chunks=[],
        expected_answer_note="重点：无相关文档时拒绝作答，不得猜测。",
    ),
]

ALL_CASES = MISREAD_CASES + CONFLICT_CASES + REFUSE_CASES


def cases_by_category(category: str) -> list[RAGCase]:
    return [c for c in ALL_CASES if c.category == category]
