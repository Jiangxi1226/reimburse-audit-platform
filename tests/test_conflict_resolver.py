# -*- coding: utf-8 -*-
"""冲突识别 + 复杂度判断 + 动态权重 + 智能路由 的单元测试。

不依赖真实 LLM / ChromaDB：冲突模块纯逻辑直测，
pipeline 路由用 monkeypatch 打桩验证调用路径。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rag.conflict_resolver import ConflictResolver
from rag.query_classifier import QueryClassifier, QueryType
from rag.pipeline import RAGPipeline
from rag import answer_generator as ag
from rag.metadata import version_rank


# ------------------------------------------------------------
# 1. 数值矛盾检测
# ------------------------------------------------------------
def _chunk(cid, text, authority=0.5, source="doc", date="2025-01-01"):
    return {
        "id": cid, "text": text, "similarity": 0.9,
        "metadata": {"source_path": source, "authority_level": authority,
                     "document_date": date},
    }


def test_numeric_conflict_detected():
    chunks = [
        _chunk("a", "2024年Q3公司营收为12.5亿元，同比增长30%",
               authority=0.6, source="快报.xlsx", date="2024-10-05"),
        _chunk("b", "2024年Q3公司营收为8.3亿元，同比下降15%",
               authority=1.0, source="审计.pdf", date="2025-03-30"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    assert report["has_conflict"] is True
    assert report["count"] == 1
    assert report["conflicts"][0]["type"] == "numeric"
    assert report["conflicts"][0]["metric"] == "营收"


def test_no_conflict_when_same_source():
    # 同来源不自判冲突（同文档内通常自洽）
    chunks = [
        _chunk("a", "2024年Q3营收为12.5亿元", source="同.doc"),
        _chunk("b", "2024年Q3营收为12.5亿元", source="同.doc"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    assert report["has_conflict"] is False


def test_no_conflict_when_consistent_values():
    # 相同数值不冲突（即使来源不同）
    chunks = [
        _chunk("a", "2024年Q3营收为12.5亿元", source="a.pdf"),
        _chunk("b", "2024年Q3营收为12.5亿元", source="b.pdf"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    assert report["has_conflict"] is False


def test_unit_conversion_no_false_conflict():
    # 12.5亿元 == 125000万元，不该判为冲突
    chunks = [
        _chunk("a", "2024年Q3营收为12.5亿元", source="a.pdf"),
        _chunk("b", "2024年Q3营收为125000万元", source="b.pdf"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    assert report["has_conflict"] is False


# ------------------------------------------------------------
# 2. 来源加权解决
# ------------------------------------------------------------
def test_resolution_favors_higher_authority():
    chunks = [
        _chunk("a", "2024年Q3公司营收为12.5亿元", authority=0.6, source="快报.xlsx"),
        _chunk("b", "2024年Q3公司营收为8.3亿元", authority=1.0, source="审计.pdf"),
        _chunk("c", "2024年Q3净利润为2.1亿元", authority=0.95, source="年报.pdf"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    res = ConflictResolver().resolve(chunks, report)

    # 审计(权威1.0) 应在加权排序首位
    assert res["ranked"][0]["id"] == "b"
    # 冲突解决倾向权威度更高的来源
    conflict = res["conflicts"][0]
    assert conflict["winner_id"] == "b"
    assert conflict["favor_a"] is False


def test_authority_ranking_order():
    # 权威度越高越靠前（权威1.0 > 0.95 > 0.6）
    chunks = [
        _chunk("a", "营收12.5亿元", authority=0.6, source="a"),
        _chunk("b", "营收8.3亿元", authority=1.0, source="b"),
        _chunk("c", "营收10.0亿元", authority=0.95, source="c"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    res = ConflictResolver().resolve(chunks, report)
    assert [x["id"] for x in res["ranked"]] == ["b", "c", "a"]


# ------------------------------------------------------------
# 3. 复杂度判断
# ------------------------------------------------------------
def test_complexity_low():
    qc = QueryClassifier()
    assert qc.get_complexity("Q3营收多少") == "low"
    assert qc.get_complexity("本季度毛利率") == "low"


def test_complexity_medium():
    qc = QueryClassifier()
    assert qc.get_complexity("营收和净利润对比一下") == "medium"


def test_complexity_high():
    qc = QueryClassifier()
    assert qc.get_complexity(
        "为什么营收下降，由哪个产品线导致，对各部门影响如何") == "high"


# ------------------------------------------------------------
# 4. 动态权重（alpha 按查询类型调优）
# ------------------------------------------------------------
def test_dynamic_alpha_by_query_type():
    # 精确数字/故障 权重最高，综述/概念 权重最低
    assert RAGPipeline._alpha_for(QueryType.FACT) == 0.5
    assert RAGPipeline._alpha_for(QueryType.TROUBLESHOOT) == 0.5
    assert RAGPipeline._alpha_for(QueryType.COMPARISON) == 0.35
    assert RAGPipeline._alpha_for(QueryType.SUMMARY) == 0.15
    assert RAGPipeline._alpha_for(None) == 0.3  # 兜底


def test_alpha_applied_in_ranking():
    # 权威度加权重排：权威越高、加权分越高
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._chunk_metadata = {
        "a": {"authority_level": 0.5},
        "b": {"authority_level": 1.0},
    }
    results = [
        {"id": "a", "text": "x", "similarity": 0.9},
        {"id": "b", "text": "y", "similarity": 0.9},
    ]
    ranked = pipeline._apply_authority_ranking(results, QueryType.FACT)
    # FACT 权威权重 0.5：权威1.0 的 b 加权分更高
    assert ranked[0]["id"] == "b"
    assert ranked[1]["id"] == "a"


# ------------------------------------------------------------
# 5. 智能路由（monkeypatch，避免真实 ChromaDB）
# ------------------------------------------------------------
def test_smart_search_routing(monkeypatch):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    calls = {"low": 0, "medium": 0, "high": 0, "high_k": 0}

    def fake_search(query, top_k):
        calls["low"] = top_k
        return []

    def fake_hybrid(query, top_k):
        calls["medium"] = top_k
        return []

    def fake_multi(query, top_k):
        calls["high"] += 1
        calls["high_k"] = top_k
        return []

    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(pipeline, "search_hybrid", fake_hybrid)
    monkeypatch.setattr(pipeline, "search_multi", fake_multi)
    pipeline._query_classifier = QueryClassifier()

    pipeline.smart_search("Q3营收多少", top_k=5)          # low
    pipeline.smart_search("营收和净利润对比一下", top_k=5)  # medium
    pipeline.smart_search("为什么营收下降，哪个产品线导致", top_k=5)  # high

    assert calls["low"] == 5
    assert calls["medium"] == 5
    # 高复杂度走 search_multi 且 top_k 翻倍（动态多召回）
    assert calls["high"] == 1
    assert calls["high_k"] == 10


# ------------------------------------------------------------
# 6. 结构化过滤（先缩范围再精检）
# ------------------------------------------------------------
def test_search_filtered_builds_where_and_calls_search(monkeypatch):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    captured = {}

    def fake_search(query, top_k, metadata_filter=None):
        captured["metadata_filter"] = metadata_filter
        return []

    monkeypatch.setattr(pipeline, "search", fake_search)

    pipeline.search_filtered("营收", top_k=5, file_type="PDF",
                             min_quality=0.7, chunk_type="table")
    f = captured["metadata_filter"]
    assert isinstance(f, dict) and "$and" in f
    assert {"file_type": {"$eq": "PDF"}} in f["$and"]
    assert {"quality_score": {"$gte": 0.7}} in f["$and"]
    assert {"chunk_type": {"$eq": "table"}} in f["$and"]

    # 无过滤条件时传 None
    pipeline.search_filtered("营收", top_k=5)
    assert captured["metadata_filter"] is None


# ------------------------------------------------------------
# 7. 版本 / 生效日期冲突（旧制度 vs 新制度）
# ------------------------------------------------------------
def _chunk_v(cid, text, authority=0.5, source="doc", date="2025-01-01",
             effective=None, version=""):
    return {
        "id": cid, "text": text, "similarity": 0.9,
        "metadata": {"source_path": source, "authority_level": authority,
                     "document_date": date,
                     "effective_date": effective or date, "version": version},
    }


def test_newer_version_wins_on_same_authority():
    # 同权威度、同生效日期下，新版本(v2)压旧版本(v1)
    chunks = [
        _chunk_v("old", "2024年公司营收为15亿元", authority=0.5,
                 source="旧版.pdf", date="2025-01-01", effective="2025-01-01",
                 version="1"),
        _chunk_v("new", "2024年公司营收为20亿元", authority=0.5,
                 source="新版.pdf", date="2025-01-01", effective="2025-01-01",
                 version="2"),
    ]
    report = ConflictResolver().detect(chunks, use_llm=False)
    res = ConflictResolver().resolve(chunks, report)
    assert res["conflicts"][0]["winner_id"] == "new"


def test_version_rank_ordering():
    assert version_rank("2.1") > version_rank("2.0")
    assert version_rank("修订版") > version_rank("")
    assert version_rank("旧版") < version_rank("新版")


# ------------------------------------------------------------
# 8. 上下文组装（相邻 chunk 带入）
# ------------------------------------------------------------
def test_build_context_includes_neighbors():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._chunk_texts = {
        "doc_0": "本协议自签署之日起生效", "doc_1": "但甲方不得单方面",
        "doc_2": "终止本合同。",
    }
    pipeline._chunk_order = {"doc": ["doc_0", "doc_1", "doc_2"]}

    # 中间块：上下文含前一块 + 当前块 + 后一块
    ctx = pipeline._build_context("doc_1")
    assert "本协议" in ctx and "但甲方" in ctx and "终止本合同" in ctx

    # 头块：无前一块，上下文 = 当前块 + 后一块
    ctx_head = pipeline._build_context("doc_0")
    assert ctx_head.startswith("本协议") and "但甲方" in ctx_head


# ------------------------------------------------------------
# 9. 数字回指校验（生成忠实性硬约束）
# ------------------------------------------------------------
def test_verify_numbers_catches_unreferenced_number():
    chunks = [{"id": "a", "text": "Q3营收为12.5亿元", "context": "Q3营收为12.5亿元"}]
    # 答案含"12.5"（有依据）和"99.9"（无依据）
    unverified = ag._verify_numbers_in_answer("营收12.5亿元，增长99.9%", chunks)
    assert "99.9" in unverified
    assert "12.5" not in unverified


def test_verify_numbers_clean_when_all_referenced():
    chunks = [{"id": "a", "text": "营收12.5亿元", "context": "营收12.5亿元"}]
    assert ag._verify_numbers_in_answer("营收12.5亿元", chunks) == []
