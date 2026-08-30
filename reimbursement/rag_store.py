# -*- coding: utf-8 -*-
"""报销问答的**向量语义检索**层（RAG 的检索部分）。

为什么需要它：
- 结构化实体过滤（决策/月份/类别/金额）对"明确数字问题"准确，但**口语化、模糊表述**
  （"那笔可疑的住宿" "为毛会被拒"）匹配不到显式关键词，召回会漏。
- 这里复用 `rag/` 的向量库(ChromaStore, sqlite+faiss 确定性落盘) 与 Embedder，
  把每条审核记录文本化后 embedding；问题向量化 → 按语义相似度召回相关记录，
  交给 LLM **依据召回事实**组织回答 —— 这是真·向量 RAG。

与结构化检索的关系：
- 语义召回负责"找到相关的记录"（口语/解释类，灵活）；
- 结构化聚合负责"数字锚定"（多少笔/总额/核减，准）。
  两者在 api.py 里融合：数字用结构化 stats，生成用语义召回的事实。

惰性建索引：首次查询时把当前审核记录全量 embedding 入库；记录新增后再查会重建
（记录量小，重建成本可忽略）。索引失败/不可用时回退结构化检索，不影响主流程。
"""
import os, threading

_RECORDS = "audit_records.json"
_COLLECTION = "reimburse"
_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db")

_lock = threading.Lock()
_store = None          # ChromaStore 单例
_embedder = None       # Embedder 单例
_built = False         # 索引是否与当前记录数一致
_built_n = -1


def _ensure_embedder():
    global _embedder
    if _embedder is None:
        from rag.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _ensure_store():
    global _store
    if _store is None:
        from rag.chroma_store import ChromaStore
        _store = ChromaStore(collection_name=_COLLECTION, persist_dir=_PERSIST_DIR)
    return _store


def _load_records():
    """读全部审核记录（供建索引/召回）。"""
    try:
        from reimbursement import record_store as rs
        return rs._load()
    except Exception:
        return []


def record_text(rec: dict) -> str:
    """把一条审核记录文本化，供 embedding 与事实引用。"""
    dec = {"approve": "通过", "partial": "部分核准", "reject": "不予报销",
           "manual_review": "待人工复核"}.get(rec.get("decision"), str(rec.get("decision", "")))
    s = f"{str(rec.get('ts', ''))[:10]} 事由「{rec.get('purpose') or '未填写'}」 结论[{dec}] " \
        f"申报{rec.get('claimed_amount', 0):.2f}元 核准{rec.get('approved_amount', 0):.2f}元 " \
        f"拒{rec.get('rejected_amount', 0):.2f}元 命中{rec.get('issue_count', 0)}项规则"
    for it in rec.get("items_adjudication", []):
        s += f"；条目「{it.get('desc', '')}」类别{it.get('category', '')} " \
             f"{it.get('amount', 0):.2f}→{it.get('approved', 0):.2f}元 {it.get('decision', '')}"
    for iss in rec.get("issues", []):
        s += f"；规则{iss.get('issue_code', '')} {iss.get('description', '')}"
    s += f"；{rec.get('summary', '')}"
    return s


def _build_index_if_needed(records: list[dict]):
    """若索引缺失或记录数变化，则重建。记录量小，重建成本可忽略。"""
    global _built, _built_n
    with _lock:
        if _built and _built_n == len(records):
            return
        store = _ensure_store()
        try:
            if records:
                texts = [record_text(r) for r in records]
                vecs = _ensure_embedder().encode_batch(texts)
                # 全量重建：直接建新 collection 由 add_batch 覆盖（同 id 覆盖）
                store.add_batch([r.get("id", f"R{i}") for i, r in enumerate(records)],
                                vecs, texts,
                                [{"purpose": r.get("purpose") or "", "decision": r.get("decision") or ""}
                                 for r in records])
            _built, _built_n = True, len(records)
        except Exception:
            # 模型/向量库不可用 → 置未建，后续回退结构化
            _built, _built_n = False, -1


def semantic_search(question: str, top_k: int = 8) -> list[dict]:
    """问题向量化 → 向量库语义召回相关审核记录（含相似度，sklit 降序）。

    返回 [{record, similarity}]；失败/无索引返回 []（上层回退结构化检索）。
    """
    try:
        records = _load_records()
        if not records:
            return []
        _build_index_if_needed(records)
        store = _ensure_store()
        qv = _ensure_embedder().encode(question)
        hits = store.search(qv, top_k=min(top_k, len(records)))
        id2rec = {r.get("id"): r for r in records}
        out = []
        for h in hits:
            rec = id2rec.get(h.get("id"))
            if rec:
                out.append({"record": rec, "similarity": h.get("similarity", 0.0)})
        return out
    except Exception:
        return []
