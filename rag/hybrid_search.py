import math, re



class BM25Scorer:
    """BM25 关键词检索引擎——轻量实现，无外部依赖。

    BM25 公式：score(d,q) = Σ IDF(qi) × (f(qi,d) × (k1+1)) / (f(qi,d) + k1×(1-b+b×|d|/avgdl))

    参数：
      k1=1.5：词频饱和度——词出现 1 次和 10 次的权重不再是 10 倍
      b=0.75：长度惩罚——长文档不占便宜
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.tokenized: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0
        self.N: int = 0

    def add_corpus(self, texts: list[str]):
        """批量添加文档。"""
        for text in texts:
            tokens = self._tokenize(text)
            self.corpus.append(text)
            self.tokenized.append(tokens)

        self.N = len(self.corpus)
        if self.N == 0:
            return

        total_len = sum(len(t) for t in self.tokenized)
        self.avgdl = total_len / self.N if self.N > 0 else 1

        for tokens in self.tokenized:
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.idf[token] = self.idf.get(token, 0) + 1
                    seen.add(token)

        for token, df in self.idf.items():
            self.idf[token] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """检索——返回 [(文档索引, BM25分数), ...] 按分数降序。"""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.tokenized):
            score = 0.0
            doc_len = len(doc_tokens)
            tf = {}
            for t in query_tokens:
                tf[t] = tf.get(t, 0) + 1

            for token, freq in tf.items():
                if token not in self.idf:
                    continue
                idf = self.idf[token]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((idx, round(score, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """中文分词——简易 2-gram + 单字切分。
        生产环境应换 jieba 分词，这里最小化依赖。
        """
        cleaned = re.sub(r'[^一-鿿\w]', ' ', text.lower())
        tokens = []
        for word in cleaned.split():
            if len(word) <= 3:
                tokens.append(word)
            else:
                for i in range(len(word) - 1):
                    tokens.append(word[i:i+2])
        return tokens



def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """RRF 融合两路检索结果。

    公式：RRF(d) = Σ 1/(k + rank_i(d))
      rank_i(d)：文档 d 在第 i 路检索中的排名

    为什么 RRF 不是简单的分数相加：
      - 向量余弦相似度(0~1)和 BM25 分数(无上限)在完全不同的量纲
      - RRF 只看排名不看分数——排名是不同的"语言"之间的通用货币
      - k=60 是经验值（Elasticsearch 默认），平滑排名差异

    Args:
        vector_results: 向量检索结果 [{"id","text","similarity"}, ...]
        bm25_results: BM25 检索结果 [{"id","text","bm25_score"}, ...]
        k: 平滑参数，越大排名差异越被压平

    Returns:
        RRF 融合后的排序列表
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("id", doc.get("text", ""))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("id", doc.get("text", ""))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    merged = [
        {**doc_map[doc_id], "rrf_score": round(score, 6)}
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return merged



def hybrid_search(
    query: str,
    vector_db,
    embedder,
    bm25: BM25Scorer = None,
    top_k: int = 5,
    use_reranker: bool = False,
    reranker=None,
) -> list[dict]:
    """混合检索——向量 + BM25 + 可选 Reranker。

    三阶段：
      1. 向量路搜 top_k×4（宽进）
      2. BM25 路搜 top_k×4
      3. RRF 融合 → 可选 Reranker 精排 → 取 top_k
    """
    query_vec = embedder.encode(query)
    vec_results = vector_db.search(query_vec, top_k=top_k * 4)

    bm25_results = []
    if bm25 and bm25.N > 0:
        bm25_hits = bm25.search(query, top_k=top_k * 4)
        bm25_results = [
            {"id": f"bm25_{idx}", "text": bm25.corpus[idx], "bm25_score": score}
            for idx, score in bm25_hits
        ]

    if bm25_results:
        merged = reciprocal_rank_fusion(vec_results, bm25_results)
    else:
        merged = vec_results

    if use_reranker and reranker and len(merged) > top_k:
        merged = reranker.rerank(query, merged, top_k)

    return merged[:top_k]
