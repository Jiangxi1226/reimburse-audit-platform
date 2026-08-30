from sentence_transformers import CrossEncoder


class Reranker:
    """CrossEncoder 重排序器

    模型：ms-marco-MiniLM-L-6-v2
      - ms-marco：微软 MS MARCO 搜索排序数据集训练
      - MiniLM-L-6：6 层轻量 Transformer
      - 首次下载 ~80MB
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """对候选列表精排：query 和每个候选文本拼接 → 模型打分 → 按分排序。

        Args:
            query: 用户原始查询
            candidates: 粗排结果 [{"id": ..., "text": ..., "similarity": ...}, ...]
            top_k: 精排后保留条数

        Returns:
            按 _rerank_score 降序的 top_k 条结果
        """
        if len(candidates) <= top_k:
            return candidates

        pairs = [(query, c["text"]) for c in candidates]

        scores = self.model.predict(pairs)

        for i, c in enumerate(candidates):
            c["_rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return candidates[:top_k]
