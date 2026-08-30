import re
from enum import Enum


class QueryType(Enum):
    FACT = "fact"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    TROUBLESHOOT = "troubleshoot"
    DEFINITION = "definition"


TYPE_PATTERNS = {
    QueryType.SUMMARY: [r"summary|overview|recap"],
    QueryType.COMPARISON: [r"compare|vs|versus|diff"],
    QueryType.PROCEDURE: [r"how to|steps|procedure|guide|how do I"],
    QueryType.TROUBLESHOOT: [r"why|error|bug|fix|issue"],
    QueryType.DEFINITION: [r"what is|define|meaning"],
    QueryType.FACT: [r"how many|when|who|which|specific"],
}

GRANULARITY_MAP = {
    QueryType.FACT: "fine",
    QueryType.SUMMARY: "coarse",
    QueryType.COMPARISON: "medium",
    QueryType.PROCEDURE: "medium",
    QueryType.TROUBLESHOOT: "fine",
    QueryType.DEFINITION: "coarse",
}

SYSTEM_PROMPTS = {
    QueryType.FACT: (
        "Precise fact lookup. Numbers must be cited verbatim. "
        "Source required per fact. If not found, say so."
    ),
    QueryType.SUMMARY: (
        "Document summary. Core conclusion first, then details. "
        "Group common views, note disagreements. Cite sources."
    ),
    QueryType.COMPARISON: (
        "Comparison analysis. Use table format. Highlight differences. "
        "Quantify with numbers. Give recommendation if possible."
    ),
    QueryType.PROCEDURE: (
        "Step-by-step guide. Numbered steps with expected results. "
        "Note prerequisites. Flag missing info honestly."
    ),
    QueryType.TROUBLESHOOT: (
        "Troubleshooting. Most likely causes first, ranked by probability. "
        "Distinguish known issues from inferred causes. Exact commands when applicable."
    ),
    QueryType.DEFINITION: (
        "Concept definition. One-sentence definition first. "
        "Background, context, and real examples. Multiple definitions if applicable."
    ),
}


# 复杂度判断用：多跳/因果/跨实体强信号（每词重权重）
HIGH_CONNECTORS = [
    "为什么", "原因", "导致", "影响", "贡献", "哪个", "驱动", "如何", "分析",
]
# 比较类强信号（单次命中即算）
COMPARE_WORDS = ["对比", "比较", "区别", "差异", "vs", "versus"]
# 制度/条款类强信号——这类问题纯向量(低复杂度)易漏关键版本，上调走混合检索
POLICY_RECALL_WORDS = [
    "制度", "规定", "政策", "规则", "标准", "休假", "年假", "条款",
    "合同", "工作日", "期限", "范围", "有效", "版本",
]
# 金融指标词——统计问题里涉及几个指标实体
METRIC_WORDS = [
    "营收", "营业收入", "净利润", "利润", "收入", "销售额", "销量",
    "增长", "下降", "增速", "毛利", "净利", "负债", "市值", "ROE",
    "现金流", "分红", "资产", "成本", "费用", "产品线", "部门", "板块",
]


class QueryClassifier:
    def classify(self, query: str) -> QueryType:
        compiled = {qt: [re.compile(p) for p in pts] for qt, pts in TYPE_PATTERNS.items()}
        for qt in [QueryType.TROUBLESHOOT, QueryType.PROCEDURE, QueryType.COMPARISON,
                    QueryType.DEFINITION, QueryType.SUMMARY, QueryType.FACT]:
            for p in compiled[qt]:
                if p.search(query):
                    return qt
        return QueryType.FACT

    def get_prompt(self, query: str, base_prompt: str = "") -> str:
        qtype = self.classify(query)
        p = SYSTEM_PROMPTS.get(qtype, SYSTEM_PROMPTS[QueryType.FACT])
        return f"{base_prompt}\n\n{p}" if base_prompt else p

    def get_granularity(self, query: str) -> str:
        return GRANULARITY_MAP.get(self.classify(query), "medium")

    def get_complexity(self, query: str) -> str:
        """判断查询复杂度：low / medium / high。

        依据（规则，无 LLM 依赖）：
          - 多跳/因果连接词数量（为什么、导致、哪个、和…）
          - 涉及金融指标/实体数量
          - 制度/条款类问题（休假、合同、规定等）——这类常需精确条款数字，
            纯向量容易漏掉关键版本（如新版20天 vs 旧版15天），故上调走混合检索
          - 问题长度

        用于决定检索路径：
          low    → 单路向量检索
          medium → 混合检索（向量+BM25+RRF）
          high   → 多粒度联合检索（+ 可扩展图检索）
        """
        score = 2 * sum(1 for c in HIGH_CONNECTORS if c in query)
        if any(c in query for c in COMPARE_WORDS):
            score += 2
        metric_count = sum(1 for m in METRIC_WORDS if m in query)
        if metric_count >= 3:
            score += 2
        elif metric_count == 2:
            score += 1
        if any(p in query for p in POLICY_RECALL_WORDS):
            score += 3  # 制度/条款类纯问题(0+3)也能到 medium → 走混合检索召回全
        if len(query) > 30:
            score += 1
        if score >= 5:
            return "high"
        if score >= 3:
            return "medium"
        return "low"


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
