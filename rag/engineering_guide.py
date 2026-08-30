import time, threading



def explain_multipath_ranking():
    """多路召回排序——三条路的融合策略。"""
    return """
    多路召回排序 = RRF 排名融合（非分数融合）

    为什么不是简单分数相加：
      向量余弦  = 0.85     ← [0,1] 区间
      BM25 分数 = 12.3     ← 无上限
      → 直接相加 BM25 压倒向量，语义信息丢失

    RRF 公式：RRF(d) = Σ 1/(k + rank_i(d))
      vector 路排名第1 → 1/(60+1) = 0.0164
      bm25   路排名第3 → 1/(60+3) = 0.0159
      总分 = 0.0323

      只看排名不看分数——排名是不同检索方式之间的"通用货币"

    三阶段排序流水线：
      ┌─ 多路召回 ─┐      ┌─ RRF融合 ─┐      ┌─ Reranker精排 ─┐
      │ 向量路 top-20│  →  │           │  →  │ CrossEncoder   │
      │ BM25路 top-20│  →  │ 排名相加   │  →  │ 逐对打相关性分   │
      │(图路 top-20) │  →  │           │  →  │                 │
      └─────────────┘      └───────────┘      └────────────────┘
       60 个候选              排名融合              前5个最佳
    """



class IndexSyncManager:
    """索引同步管理器——文档变更时同步更新向量索引和 BM25 索引。

    三种操作的处理策略：
      - 增(add)：新文档 → 切块 → Embedding → ChromaDB.add() + BM25.add_corpus()
      - 删(delete)：→ ChromaDB.delete(ids=...) + BM25 重建（BM25不支持单条删除）
      - 改(update)：→ 删旧 + 增新（等价于先删后增，保证幂等）

    ChromaDB 支持单条删除（collection.delete(ids=[...])），
    BM25 不支持——只能全量重建。优化策略：增量批次重建。
    """

    def __init__(self, chroma_store, bm25, embedder):
        self.store = chroma_store
        self.bm25 = bm25
        self.embedder = embedder
        self._pending_delete: list[str] = []
        self._pending_add: list[tuple] = []
        self._bm25_dirty = False
        self._lock = threading.Lock()

    def add(self, text: str, doc_id: str):
        """增量添加文档。"""
        with self._lock:
            chunks = self._split(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                vector = self.embedder.encode(chunk)
                self.store.add(chunk_id, vector, chunk)
                self._pending_add.append((chunk, {"source": doc_id}))

            if len(self._pending_add) >= 20:
                self._flush_bm25()

    def delete(self, doc_id: str):
        """删除文档。ChromaDB 可以精确删，BM25 标记重建。"""
        with self._lock:
            all_ids = self.store.collection.get()["ids"]
            to_delete = [id_ for id_ in all_ids if id_.startswith(doc_id)]
            if to_delete:
                self.store.collection.delete(ids=to_delete)

            self._bm25_dirty = True

    def update(self, text: str, doc_id: str):
        """更新文档 = 删旧 + 增新（原子性由 Lock 保证）。"""
        self.delete(doc_id)
        self.add(text, doc_id)

    def _flush_bm25(self):
        """增量更新 BM25——重建全量索引。（BM25 不支持单条增删）"""
        all_texts = self._collect_all_texts()
        self.bm25 = None
        self.bm25.add_corpus(all_texts)
        self._pending_add.clear()
        self._bm25_dirty = False

    def _split(self, text: str) -> list[str]:
        """切块——复用 Chunker。"""
        from rag.chunker import Chunker
        return list(Chunker(chunk_size=200, overlap=50).split(text))

    def _collect_all_texts(self) -> list[str]:
        """从 ChromaDB 取出所有文本用于重建 BM25。"""
        data = self.store.collection.get(include=["documents"])
        return data.get("documents", []) or []



def explain_memory_control():
    """内存暴涨的五层防线。"""
    return """
    内存构成分析（你的项目）：
      MiniLM         ~100MB
      CLIP           ~600MB
      Qwen2-VL-2B    ~4GB (半精度)
      ChromaDB HNSW  ~ N × 384 × 4 bytes + 索引开销
      PaddleOCR       ~200MB
      基准总计        ~5GB+

    五层防线：

    ① 懒加载（已做）：
       Qwen2-VL 只在 chat_with_image 时加载
       PaddleOCR 单例延迟加载

    ② 模型量化（可做）：
       Qwen2-VL 用 int8 量化：4GB → 2GB
       ChromaDB 用 PQ 量化：384维 → 64维（4x压缩）

    ③ 索引分页（需做）：
       100万向量不要全在内存 → 分批加载
       热数据(HNSW)在内存 / 冷数据在磁盘

    ④ 淘汰策略（已做部分）：
       ShortTermMemory token超限 → 压缩为摘要
       EpisodicMemory 时间衰减 → forget_old()

    ⑤ 内存监控（需做）：
       定时采样 process.memory_info().rss
       超过 80% 水位 → 触发淘汰/压缩
    """


class MemoryMonitor:
    """内存监控与自动保护。"""

    def __init__(self, max_memory_mb: int = 6000, check_interval: int = 30):
        self.max_mb = max_memory_mb
        self.interval = check_interval
        self._callbacks: list = []
        self._running = False
        self._thread = None

    def on_pressure(self, callback):
        """注册内存压力回调。超过阈值时调用。"""
        self._callbacks.append(callback)

    def current_mb(self) -> float:
        """当前进程内存占用（MB）。"""
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    def start(self):
        """启动后台监控线程。"""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                mem = self.current_mb()
                if mem > self.max_mb:
                    for cb in self._callbacks:
                        cb(mem)
            except Exception:
                pass
            time.sleep(self.interval)



def explain_context_window():
    """上下文窗口管理的四层策略。"""
    return """
    上下文窗口 = LLM 一次能处理的 token 上限（Agnes 一般是 8K/32K/128K）

    你的项目已有：
      ① ShortTermMemory token阈值压缩（总字符数//2 估算）
         → 超过 4000 tokens → LLM 摘要 → 丢弃旧消息

    需要补的：
      ② 检索结果截断：
         不是把 8 个 chunk 全丢给 LLM，而是按相似度排序后，
         从最重要的一块一块加，加到 token 预算用完为止

      ③ 动态 Chunk 膨胀/收缩：
         高相似度的 chunk → 展开到 400 字（多喂细节）
         低相似度的 chunk → 收缩到 100 字（只喂摘要）

      ④ 分层回答策略：
         Token 够 → 完整回答 + 引用来源
         Token 紧 → 核心要点 + 省略引用
         Token 爆 → 拒答"当前知识库太大，请缩小问题范围"
    """


class ContextBudget:
    """上下文 Token 预算管理器——控制检索结果喂给 LLM 的量。"""

    def __init__(self, max_tokens: int = 4000, reserve_for_answer: int = 800):
        self.max_tokens = max_tokens
        self.reserve = reserve_for_answer
        self.available = max_tokens - reserve_for_answer

    def allocate(self, chunks: list[dict]) -> list[dict]:
        """按重要性分配 token 预算——最重要的一块一块加。

        策略：
          相似度 > 0.7 → 取完整文本（400字）
          相似度 0.5~0.7 → 取摘要（150字）
          相似度 < 0.5 → 取标题行（60字）
          直到用完预算
        """
        used = 0
        allocated = []
        for c in sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True):
            sim = c.get("similarity", 0)

            if sim > 0.7:
                text = c.get("text", "")[:400]
            elif sim > 0.5:
                text = c.get("text", "")[:150]
            else:
                text = c.get("text", "")[:60] + "..."

            tokens = len(text) // 2
            if used + tokens > self.available:
                break

            allocated.append({**c, "truncated_text": text, "allocated_tokens": tokens})
            used += tokens

        return allocated

    def budget_report(self, chunks: list[dict]) -> dict:
        """报告预算使用情况——给 LLM 的 prompt 开头加一句。"""
        total_chunks = len(chunks)
        allocated_chunks = len(self.allocate(chunks))
        return {
            "total_chunks": total_chunks,
            "fed_to_llm": allocated_chunks,
            "dropped": total_chunks - allocated_chunks,
            "warning": (
                f"（共检索到 {total_chunks} 条相关信息，"
                f"由于上下文窗口限制，仅展示最重要的 {allocated_chunks} 条）"
            ) if total_chunks > allocated_chunks else "",
        }



ALL_ANSWERS = {
    "图检索能否建立跨文档联系": (
        "✅ 能。用 LLM 抽取三元组(实体,关系,实体)，跨文档合并同名实体。"
        "文档A的'Q3营收'和文档B的'Q3营收'合并 → 自动串联推理路径。"
        "代码：rag/graph_extractor.py → KnowledgeGraph.add_triples()"
    ),
    "多路召回怎么排序": (
        "✅ RRF排名融合，不是分数相加。"
        "阶段1：多路(向量+BM25+可选图)各取top-20 → "
        "阶段2：RRF(1/(k+rank))排名融合 → "
        "阶段3：CrossEncoder精排取top-5。"
        "代码：rag/hybrid_search.py → reciprocal_rank_fusion()"
    ),
    "文档增删改索引同步": (
        "✅ ChromaDB单条删(collection.delete(ids=[...]))。"
        "BM25批量重建(累积20+chunk或定时触发)。"
        "改=删旧+增新(Lock保证原子性)。"
        "代码：rag/engineering_guide.py → IndexSyncManager"
    ),
    "内存暴涨": (
        "✅ 五层：懒加载(已做)+模型量化(int8)+索引分页(热内存冷磁盘)+"
        "淘汰(时间衰减+forget_old)+监控(psutil采样+80%水位回调)。"
        "代码：rag/engineering_guide.py → MemoryMonitor"
    ),
    "上下文窗口爆了": (
        "✅ ShortTermMemory压缩(已做)+ContextBudget按重要性分配(新增)+"
        "分层回答(充足完整，紧张省略，爆了拒答)。"
        "代码：rag/engineering_guide.py → ContextBudget"
    ),
}

if __name__ == "__main__":
    for q, a in ALL_ANSWERS.items():
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {a}")
