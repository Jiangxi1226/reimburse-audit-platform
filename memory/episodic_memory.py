import math, uuid, re
from datetime import datetime
from rag.embedder import Embedder
from rag.chroma_store import ChromaStore

DEDUP_SIMILARITY = 0.90


class MemoryItem:
    def __init__(self, content: str, importance: float = 0.5, category: str = "general"):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.importance = importance
        self.category = category
        self.timestamp = datetime.now()


class EpisodicMemory:
    """情景记忆：记住跨会话的关键事件和用户偏好"""

    def __init__(self, embedder: Embedder = None, store: ChromaStore = None):
        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore(collection_name="episodic_memory")


    def remember(self, content: str, importance: float = 0.5,
                 category: str = "general") -> str:
        """存储一条情景记忆——自动去重和更新。

        流程：
          1. 编码新记忆 → 向量
          2. 查现有记忆中有没有高度相似的（冗余去重）
          3. 同类别有相似 → 更新旧记忆（保留旧 ID，换新内容）
          4. 同类别有矛盾 → 删旧存新（偏好变更）
          5. 全新记忆 → 直接插入
        """
        item = MemoryItem(content, importance, category)
        vector = self.embedder.encode(content)

        existing = self._find_duplicate(vector, category, DEDUP_SIMILARITY)

        if existing:
            old_content = existing["text"]
            old_importance = self._parse_importance(old_content)

            if self._is_contradictory(content, old_content):
                self.store.collection.delete(ids=[existing["id"]])
            else:
                old_time_match = re.search(r"(\d{2}-\d{2} \d{2}:\d{2})", old_content)
                old_time = old_time_match.group(1) if old_time_match else item.timestamp.strftime('%m-%d %H:%M')
                importance = max(importance, old_importance)
                self.store.collection.delete(ids=[existing["id"]])

        packed = (f"[{category}] 重要性:{importance:.1f} | "
                  f"{item.timestamp.strftime('%m-%d %H:%M')} | {content}")
        self.store.add(item.id, vector, packed)
        return item.id

    def _find_duplicate(self, vector: list[float], category: str,
                        threshold: float) -> dict | None:
        """在现有记忆中查找高度相似的记录。
        只检索同 category 的记忆（偏好不和事实比较）。"""
        try:
            candidates = self.store.search(vector, top_k=3)
            for c in candidates:
                if c["similarity"] >= threshold:
                    cat = self._parse_category(c["text"])
                    if cat == category:
                        return c
        except Exception:
            pass
        return None

    def _is_contradictory(self, new_text: str, old_text: str) -> bool:
        """启发式矛盾检测：同类别记忆出现否定词或对立表述。

        例："用户喜欢表格" vs "用户不喜欢表格，要求用图表" → 矛盾
            "Q3目标1.5亿"  vs "Q3实际1.2亿"         → 不矛盾（目标和实际不同维度）
        """
        negation_words = ["不", "不要", "不喜欢", "改为", "换成", "不再", "取消"]
        has_negation = any(w in new_text for w in negation_words)
        if not has_negation:
            return False

        import re as _re
        new_keywords = set(_re.findall(r'[一-鿿]{2,}', new_text))
        old_keywords = set(_re.findall(r'[一-鿿]{2,}', old_text))
        overlap = len(new_keywords & old_keywords) / max(len(new_keywords | old_keywords), 1)

        return overlap > 0.3


    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """检索相关记忆——时间近因性 + 重要性加权重排。"""
        query_vec = self.embedder.encode(query)
        candidates = self.store.search(query_vec, top_k=top_k * 3)

        results = []
        for c in candidates:
            importance = self._parse_importance(c["text"])
            age_hours = self._parse_age(c["text"])

            vec_score = c["similarity"]
            recency = max(0.1, math.exp(-0.1 * age_hours / 24))
            base = vec_score * 0.8 + recency * 0.2
            importance_weight = 0.8 + importance * 0.4
            final_score = base * importance_weight

            results.append({**c, "episodic_score": round(final_score, 4)})

        results.sort(key=lambda x: x["episodic_score"], reverse=True)
        return results[:top_k]


    def forget_old(self, days: int = 30, min_importance: float = 0.3):
        """清理低重要性旧记忆（按时间 + 重要性双重筛选）。

        策略：
          - 超过 days 天 + 重要性低于 min_importance → 删除
          - 重要性 ≥ 0.7 的记忆永远不会被自动清理（核心偏好/关键决策）
          - 实际删除操作：get → 筛选 → delete(ids=...)
        """
        try:
            all_data = self.store.collection.get(include=["documents"])
            if not all_data["ids"]:
                return 0

            to_delete = []
            threshold_hours = days * 24

            for id_, text in zip(all_data["ids"], all_data["documents"] or []):
                importance = self._parse_importance(text)
                age_hours = self._parse_age(text)

                if importance >= 0.7:
                    continue
                if age_hours > threshold_hours and importance < min_importance:
                    to_delete.append(id_)

            if to_delete:
                self.store.collection.delete(ids=to_delete)
            return len(to_delete)
        except Exception:
            return 0


    @staticmethod
    def _parse_importance(text: str) -> float:
        m = re.search(r"重要性:([\d.]+)", text)
        return float(m.group(1)) if m else 0.5

    @staticmethod
    def _parse_category(text: str) -> str:
        m = re.search(r"\[(\w+)\]", text)
        return m.group(1) if m else "general"

    @staticmethod
    def _parse_age(text: str) -> float:
        """从记录文本中解析"MM-DD HH:MM"并计算距今小时数(情感近因加权)。

        若存储时只落月日（无年份），则默认为当前年份，避免跨年后时间衰减
        计算错误。单位换算为小时。
        """
        m = re.search(r"(\d{2}-\d{2} \d{2}:\d{2})", text)
        if m:
            try:
                year = datetime.now().year
                dt = datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M")
                return (datetime.now() - dt).total_seconds() / 3600
            except ValueError:
                pass
        return 0.0

    def stats(self) -> dict:
        return {"total_episodes": self.store.count()}
