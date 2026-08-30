import uuid, hashlib
from rag.embedder import Embedder
from rag.chroma_store import ChromaStore
from rag.image_embedder import ImageEmbedder
from rag.reranker import Reranker
from rag.query_classifier import get_classifier


DEDUP_SIMILARITY_THRESHOLD = 0.95

# 入库批量提交大小：攒够这批才一次性 add_batch 到 Chroma。逐条写会频繁触发 compaction 致
# HNSW 不落盘；批次越大写入越稳，但内存峰值越高。50 是兼顾稳定与内存的折中。
_BATCH = 50


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RAGPipeline:
    def __init__(self, use_reranker: bool = True,
                 use_langchain_chunker: bool = True,
                 use_semantic_chunker: bool = False,
                 use_table_preserving: bool = False,
                 use_multi_granularity: bool = True,
                 use_knowledge_graph: bool = False,
                 use_abbreviation_expand: bool = False,
                 llm=None):
        if use_semantic_chunker:
            from rag.semantic_chunker import SemanticChunker
            base_chunker = SemanticChunker(chunk_size=200, overlap=50)
        elif use_langchain_chunker:
            from rag.langchain_chunker import LangChainChunker
            base_chunker = LangChainChunker(chunk_size=200, overlap=50)
        else:
            base_chunker = LangChainChunker(chunk_size=200, overlap=50)
        self._chunker_type = "semantic" if use_semantic_chunker else "langchain"

        if use_table_preserving:
            from rag.table_chunker import TablePreservingChunker
            self.chunker = TablePreservingChunker(base_chunker)
        else:
            self.chunker = base_chunker

        self.embedder = Embedder()

        self.use_multi_granularity = use_multi_granularity
        self.store = ChromaStore(collection_name="rag_text")
        self.store_coarse = ChromaStore(collection_name="rag_text_coarse") if use_multi_granularity else None
        self.store_fine = ChromaStore(collection_name="rag_text_fine") if use_multi_granularity else None

        if use_semantic_chunker:
            from rag.semantic_chunker import SemanticChunker
            self._coarse_chunker = SemanticChunker(chunk_size=800, overlap=100)
            self._fine_chunker = SemanticChunker(chunk_size=50, overlap=15)
        elif use_langchain_chunker:
            from rag.langchain_chunker import LangChainChunker
            self._coarse_chunker = LangChainChunker(chunk_size=800, overlap=100)
            self._fine_chunker = LangChainChunker(chunk_size=50, overlap=15)
        else:
            self._coarse_chunker = LangChainChunker(chunk_size=800, overlap=100)
            self._fine_chunker = LangChainChunker(chunk_size=50, overlap=15)

        self.image_embedder = ImageEmbedder()
        self.image_store = ChromaStore(collection_name="rag_images")

        self.reranker = Reranker() if use_reranker else None

        self._bm25 = None

        self._kg = None
        self._kg_llm = llm
        if use_knowledge_graph and llm:
            from rag.graph_extractor import KnowledgeGraph, extract_triples
            self._kg = KnowledgeGraph()
            self._extract_triples_fn = extract_triples
            self._kg_llm = llm

        self._abbrev_resolver = None
        if use_abbreviation_expand:
            from rag.abbreviation_resolver import get_resolver
            self._abbrev_resolver = get_resolver()

        self._query_classifier = get_classifier()

        self._text_hashes: set[str] = set()

        self._chunk_metadata: dict[str, dict] = {}
        # 相邻 chunk 缓存：解决"固定切片截断语义反转"。key=chunk_id, value=文本
        self._chunk_texts: dict[str, str] = {}
        # 每个来源(source prefix)的 chunk 顺序，用于组装相邻上下文
        self._chunk_order: dict[str, list[str]] = {}


    def _flush_batches(self, main, coarse, fine):
        """把攒批的三层向量一次性提交到 Chroma。空批跳过；失败不阻断主流程。"""
        try:
            if main:
                ids = [x[0] for x in main]; vecs = [x[1] for x in main]
                docs = [x[2] for x in main]; metas = [x[3] for x in main]
                self.store.add_batch(ids, vecs, docs, metas)
            if coarse and self.store_coarse:
                ids = [x[0] for x in coarse]; vecs = [x[1] for x in coarse]
                docs = [x[2] for x in coarse]; metas = [x[3] for x in coarse]
                self.store_coarse.add_batch(ids, vecs, docs, metas)
            if fine and self.store_fine:
                ids = [x[0] for x in fine]; vecs = [x[1] for x in fine]
                docs = [x[2] for x in fine]; metas = [x[3] for x in fine]
                self.store_fine.add_batch(ids, vecs, docs, metas)
        except Exception:
            pass  # 向量化已消费，落库失败不影响主流程（后续 stats 会体现缺口）

    def add_document(self, text: str, source: str = None,
                     file_metadata: dict = None, quality_info: dict = None,
                     authority_level: float = None,
                     document_date: str = None,
                     effective_date: str = None,
                     version: str = None):
        """文本入库：质量路由 → 表格感知切分 → 多粒度索引 → 元数据记录。

        Args:
            text: 文档纯文本内容
            source: 来源标识（文件路径）
            file_metadata: 文件级元数据（来自 metadata.extract_file_metadata）
            quality_info: 质量检测结果（来自 quality_detector.detect_quality）
            authority_level: 来源权威度 0~1（覆盖 file_metadata 推断值）
            document_date: 文档日期 ISO 字符串（覆盖 file_metadata 推断值）
            effective_date: 生效日期（覆盖 file_metadata 推断值）
            version: 版本号（覆盖 file_metadata 推断值）
        """
        if source:
            self._delete_by_source(source)

        from rag.langchain_chunker import LangChainChunker
        from rag.semantic_chunker import SemanticChunker
        if hasattr(self.chunker, 'split') and not isinstance(self.chunker, (LangChainChunker, SemanticChunker)):
            chunk_dicts = self.chunker.split(text)
        else:
            raw_chunks = list(self.chunker.split(text))
            chunk_dicts = [{"text": c, "type": "text"} for c in raw_chunks]

        quality_score = quality_info.get("score", 1.0) if quality_info else 1.0
        strategy = quality_info.get("strategy", {}) if quality_info else {}
        is_low_quality = quality_score < 0.5

        if strategy.get("reject"):
            return

        prefix = source.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if source else str(uuid.uuid4())[:8]

        # 攒批缓冲：主/粗/细三层各攒 _BATCH 条后一次性 add_batch，避免逐条写触发 compaction。
        _pending_main: list = []
        _pending_coarse: list = []
        _pending_fine: list = []

        for i, cd in enumerate(chunk_dicts):
            if isinstance(cd, str):
                chunk_text = cd
                chunk_type = "text"
            else:
                chunk_text = cd["text"]
                chunk_type = cd.get("type", "text")

            h = _hash_text(chunk_text)
            if h in self._text_hashes:
                continue
            self._text_hashes.add(h)

            chunk_id = f"{prefix}_{i}"

            from rag.metadata import build_chunk_metadata
            page_meta = cd.get("page_metadata") if isinstance(cd, dict) and "page_metadata" in cd else None
            meta = build_chunk_metadata(
                file_meta=file_metadata or {},
                page_meta=page_meta,
                chunk_idx=i,
                quality_score=quality_score,
                chunk_type=chunk_type,
                custom_tags=cd.get("tags", []) if isinstance(cd, dict) and "tags" in cd else [],
                authority_level=authority_level,
                document_date=document_date,
                effective_date=effective_date,
                version=version,
            )
            if chunk_type == "table" and isinstance(cd, dict):
                meta["table_format"] = cd.get("table_format", "markdown")
                meta["table_summary"] = cd.get("table_summary", "")
                meta["table_json"] = str(cd.get("table_json", {}))

            self._chunk_metadata[chunk_id] = meta
            self._chunk_texts[chunk_id] = chunk_text
            self._chunk_order.setdefault(prefix, []).append(chunk_id)

            vec = self.embedder.encode(chunk_text)
            # 不再逐条 collection.add：分批攒齐后一次提交。逐条高频写会频繁触发 chromadb 的
            # compaction，600+ 条小写入下 HNSW 图文件易丢，重启后检索报 "Error constructing hnsw
            # segment"。这里按 _BATCH 条聚成一笔批量 add，显著降低 compaction 频率、保证落盘。
            _pending_main.append((chunk_id, vec, chunk_text, meta))

            if self.use_multi_granularity and not is_low_quality:
                for j, coarse_chunk in enumerate(self._coarse_chunker.split(chunk_text)):
                    coarse_id = f"{prefix}_coarse_{i}_{j}"
                    coarse_vec = self.embedder.encode(coarse_chunk)
                    coarse_meta = {**meta, "granularity": "coarse"}
                    _pending_coarse.append((coarse_id, coarse_vec, coarse_chunk, coarse_meta))

                for k, fine_chunk in enumerate(self._fine_chunker.split(chunk_text)):
                    fine_id = f"{prefix}_fine_{i}_{k}"
                    fine_vec = self.embedder.encode(fine_chunk)
                    fine_meta = {**meta, "granularity": "fine"}
                    _pending_fine.append((fine_id, fine_vec, fine_chunk, fine_meta))

            if len(_pending_main) >= _BATCH:
                self._flush_batches(_pending_main, _pending_coarse, _pending_fine)
                _pending_main, _pending_coarse, _pending_fine = [], [], []

        if _pending_main:
            self._flush_batches(_pending_main, _pending_coarse, _pending_fine)

        if self._kg and self._kg_llm:
            for cd in chunk_dicts[:10]:
                try:
                    triples = self._extract_triples_fn(cd["text"], self._kg_llm)
                    self._kg.add_triples(triples, doc_id=source or prefix)
                except Exception:
                    pass

    def _delete_by_source(self, source: str):
        """删除指定来源的所有旧块（三层粒度全部清理）。"""
        prefix = source.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        stores = [self.store]
        if self.use_multi_granularity:
            stores.extend([s for s in [self.store_coarse, self.store_fine] if s])
        for store in stores:
            try:
                all_ids = store.collection.get()["ids"]
                to_delete = [id_ for id_ in all_ids if id_.startswith(prefix)]
                if to_delete:
                    store.collection.delete(ids=to_delete)
            except Exception:
                pass
        self._text_hashes.clear()


    def search(self, query: str, top_k: int = 5,
               granularity: str = "auto",
               metadata_filter: dict = None) -> list[dict]:
        """智能检索——自动选择检索粒度 + 缩写展开 + 元数据过滤。

        Args:
            query: 用户查询
            top_k: 返回条数
            granularity: "auto"|"coarse"|"medium"|"fine"
                "auto" → 根据查询类型自动选择（默认）
            metadata_filter: ChromaDB where 条件（过滤文件类型/页码/质量等）

        Returns:
            [{"id":..., "text":..., "similarity":..., "metadata":..., "citation":...}, ...]
        """
        if self._abbrev_resolver:
            query = self._abbrev_resolver.expand(query, mode="append")

        qtype = self._query_classifier.classify(query)
        if granularity == "auto":
            granularity = self._query_classifier.get_granularity(query)

        if granularity == "coarse" and self.store_coarse:
            store = self.store_coarse
        elif granularity == "fine" and self.store_fine:
            store = self.store_fine
        else:
            store = self.store

        query_vec = self.embedder.encode(query)
        n_candidates = top_k * 4 if self.reranker else top_k
        candidates = store.search(query_vec, top_k=n_candidates, where=metadata_filter)

        if self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k * 2)

        candidates = self._dedup_results(candidates, DEDUP_SIMILARITY_THRESHOLD)
        candidates = self._apply_authority_ranking(candidates, qtype)

        results = candidates[:top_k]
        for r in results:
            chunk_id = r.get("id", "")
            r["metadata"] = self._chunk_metadata.get(chunk_id, {})
            r["context"] = self._build_context(chunk_id)
            if r["metadata"]:
                from rag.metadata import format_citation
                r["citation"] = format_citation(r["metadata"], r.get("text", ""))
            else:
                r["citation"] = ""
            r["granularity"] = granularity

        return results

    @staticmethod
    def _alpha_for(query_type=None) -> float:
        """按查询类型动态决定权威度权重 alpha。

        精确数字/故障类查询：答案必须可靠 → 权威度权重高（0.5）
        比较类：兼顾双方 → 中（0.35）
        综述/概念/流程：重覆盖面 → 权威度权重低（0.15~0.3）

        weighted = base × ((1-alpha) + alpha×authority)
        """
        key = ""
        if query_type is not None:
            key = query_type.value if hasattr(query_type, "value") else str(query_type)
        table = {
            "fact": 0.5, "troubleshoot": 0.5, "comparison": 0.35,
            "procedure": 0.3, "definition": 0.2, "summary": 0.15,
        }
        return table.get(key, 0.3)

    def _apply_authority_ranking(self, results: list[dict],
                                 query_type=None) -> list[dict]:
        """权重重排：相关度 × ((1-alpha) + alpha×来源权威度)。

        alpha 按查询类型动态调优（_alpha_for）：
          精确数字(FACT)权威度权重高，综述(SUMMARY)权重低。
        改的是排序依据，不丢任何候选，只调顺序。
        """
        if not results:
            return results
        alpha = self._alpha_for(query_type)
        for r in results:
            meta = self._chunk_metadata.get(r.get("id", ""), {})
            authority = float(meta.get("authority_level", 0.5))
            base = r.get("_rerank_score", r.get("similarity", 0)) or 0
            r["_weighted_score"] = round(base * ((1 - alpha) + alpha * authority), 5)
        results.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
        return results

    def _build_context(self, chunk_id: str) -> str:
        """组装相邻上下文：取当前 chunk 及其前一/后一块，避免切片截断语义反转。

        例：合同条款"本协议自签署之日起生效，但[截断]甲方不得…"——
        单独检索后半句会语义反转，带上相邻块可还原完整条款。
        返回 "前一块\n\n当前块\n\n后一块"。
        """
        cur = self._chunk_texts.get(chunk_id, "")
        if not cur:
            return cur
        prev_id, next_id = None, None
        for ids in self._chunk_order.values():
            if chunk_id in ids:
                i = ids.index(chunk_id)
                if i > 0:
                    prev_id = ids[i - 1]
                if i < len(ids) - 1:
                    next_id = ids[i + 1]
                break
        parts = []
        if prev_id:
            parts.append(self._chunk_texts.get(prev_id, ""))
        parts.append(cur)
        if next_id:
            parts.append(self._chunk_texts.get(next_id, ""))
        return "\n\n".join(p for p in parts if p)

    def get_query_complexity(self, query: str) -> str:
        """查询复杂度判断（low/medium/high），用于动态选择检索路径。"""
        return self._query_classifier.get_complexity(query)

    def smart_search(self, query: str, top_k: int = 5) -> list[dict]:
        """复杂度感知的智能检索：按问题复杂度自动选择检索路径 + 动态召回数量。

        low    → search          单路向量，top_k 不变（快、省）
        medium → search_hybrid   向量+BM25+RRF（召回更全）
        high   → search_multi    三粒度联合，top_k×2 多召回（宏观+精确兼顾）

        实现"简单少召回、复杂多召回"——高复杂度问题自动扩大候选池。

        Returns: 和 search 一致的 [{id,text,similarity,metadata,citation}, ...]
        """
        complexity = self.get_query_complexity(query)
        if complexity == "high":
            return self.search_multi(query, top_k * 2)
        if complexity == "medium":
            return self.search_hybrid(query, top_k)
        return self.search(query, top_k)

    def search_filtered(self, query: str, top_k: int = 5,
                        file_type: str = "", min_quality: float = 0.0,
                        chunk_type: str = "") -> list[dict]:
        """先缩范围再精检：按 文件类型/最低质量/块类型 构造过滤条件后再检索。

        对应方案里的"先缩范围再精检"——海量知识库先按类目/质量初筛收敛检索空间，
        再精准检索，兼顾效率与效果。
        """
        conditions = []
        if file_type:
            conditions.append({"file_type": {"$eq": file_type.upper()}})
        if min_quality and min_quality > 0:
            conditions.append({"quality_score": {"$gte": min_quality}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        return self.search(query, top_k, metadata_filter=where)

    def _dedup_results(self, results: list[dict], threshold: float) -> list[dict]:
        if len(results) <= 1:
            return results
        keep = []
        for r in results:
            is_dup = False
            for k in keep:
                if self._texts_too_similar(r["text"], k["text"], threshold):
                    is_dup = True
                    break
            if not is_dup:
                keep.append(r)
        return keep

    def _texts_too_similar(self, text_a: str, text_b: str, threshold: float) -> bool:
        if text_a == text_b:
            return True
        set_a = set(text_a)
        set_b = set(text_b)
        if not set_a or not set_b:
            return False
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return (intersection / union) > threshold if union > 0 else False


    def search_multi(self, query: str, top_k: int = 5) -> list[dict]:
        """多粒度联合检索——三路同时搜，合并去重后取 top_k。

        适用场景：复杂查询（既需要宏观总结又需要精确数字）。
        代价：3 路检索 + 合并，耗时约为单路检索的 2~3 倍。
        """
        if not self.use_multi_granularity:
            return self.search(query, top_k, granularity="medium")

        if self._abbrev_resolver:
            query = self._abbrev_resolver.expand(query, mode="append")

        query_vec = self.embedder.encode(query)
        n_candidates = top_k * 3

        seen: dict[str, dict] = {}
        for store, label in [(self.store_coarse, "coarse"),
                              (self.store, "medium"),
                              (self.store_fine, "fine")]:
            if store is None:
                continue
            for r in store.search(query_vec, top_k=n_candidates):
                rid = r["id"]
                if rid not in seen or r["similarity"] > seen[rid]["similarity"]:
                    r["granularity"] = label
                    r["metadata"] = self._chunk_metadata.get(rid, {})
                    seen[rid] = r

        merged = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
        if self.reranker and len(merged) > top_k:
            merged = self.reranker.rerank(query, merged, top_k * 2)
        merged = self._dedup_results(merged, DEDUP_SIMILARITY_THRESHOLD)
        merged = self._apply_authority_ranking(
            merged, self._query_classifier.classify(query))

        for r in merged[:top_k]:
            if r.get("metadata") and not r.get("citation"):
                from rag.metadata import format_citation
                r["citation"] = format_citation(r["metadata"], r.get("text", ""))
            if not r.get("context"):
                r["context"] = self._build_context(r.get("id", ""))

        return merged[:top_k]


    def add_image(self, image_path: str, ocr_text: str = ""):
        img_id = str(uuid.uuid4())[:8]
        img_vector = self.image_embedder.encode_image(image_path)
        self.image_store.add(f"img_{img_id}", img_vector,
                             f"[图片: {image_path}] OCR文字: {ocr_text[:200]}")
        if ocr_text and ocr_text.strip():
            self.add_document(
                f"[图片来源: {image_path}]\n{ocr_text}",
                source=f"ocr:{image_path}"
            )
        return img_id

    def search_images(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.image_embedder.encode_text(query)
        results = self.image_store.search(query_vector, top_k)
        for r in results:
            r["type"] = "image"
        return results

    def search_all(self, query: str, top_k: int = 5) -> list[dict]:
        text_results = self.search(query, top_k)
        for r in text_results:
            r["type"] = "text"
        image_results = self.search_images(query, top_k)
        merged = text_results + image_results
        merged.sort(key=lambda x: x["similarity"], reverse=True)
        return merged[:top_k]


    def search_hybrid(self, query: str, top_k: int = 5,
                       use_reranker: bool = True) -> list[dict]:
        from rag.hybrid_search import reciprocal_rank_fusion
        if self._abbrev_resolver:
            query = self._abbrev_resolver.expand(query, mode="append")
        if self._bm25 is None:
            self._build_bm25_index()
        query_vec = self.embedder.encode(query)
        n_candidates = top_k * 4
        vec_results = self.store.search(query_vec, top_k=n_candidates)
        bm25_results: list[dict] = []
        if self._bm25 and self._bm25.N > 0:
            bm25_hits = self._bm25.search(query, top_k=n_candidates)
            bm25_results = [
                {"id": f"bm25_{idx}", "text": self._bm25.corpus[idx],
                 "similarity": round(score / max(1, max(s[1] for s in bm25_hits)), 4)}
                for idx, score in bm25_hits
            ]
        merged = reciprocal_rank_fusion(vec_results, bm25_results) if bm25_results else vec_results
        if use_reranker and self.reranker and len(merged) > top_k:
            merged = self.reranker.rerank(query, merged, top_k * 2)
        merged = self._dedup_results(merged, DEDUP_SIMILARITY_THRESHOLD)
        merged = self._apply_authority_ranking(
            merged, self._query_classifier.classify(query))
        return merged[:top_k]

    def _build_bm25_index(self):
        try:
            from rag.hybrid_search import BM25Scorer
            all_data = self.store.collection.get(include=["documents"])
            texts = all_data.get("documents") or []
            if texts:
                self._bm25 = BM25Scorer()
                self._bm25.add_corpus(texts)
        except Exception:
            self._bm25 = None

    def search_graph(self, entity_names: list[str], max_depth: int = 2) -> dict:
        if not self._kg:
            return {"entities": [], "relations": [], "context": "知识图谱未启用。"}
        return self._kg.query(entity_names, max_depth)

    @property
    def knowledge_graph_stats(self) -> dict | None:
        return self._kg.stats() if self._kg else None


    @property
    def text_count(self) -> int:
        return self.store.count()

    @property
    def image_count(self) -> int:
        return self.image_store.count()

    def reset(self):
        self._text_hashes.clear()
        self._chunk_metadata.clear()
        try:
            self.store.collection.delete(where={})
            self.image_store.collection.delete(where={})
            if self.store_coarse:
                self.store_coarse.collection.delete(where={})
            if self.store_fine:
                self.store_fine.collection.delete(where={})
        except Exception:
            pass

    def get_metadata(self, chunk_id: str) -> dict:
        """查询单个 chunk 的元数据——用于精细化溯源。"""
        return self._chunk_metadata.get(chunk_id, {})
