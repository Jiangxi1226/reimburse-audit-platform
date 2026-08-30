import re


class LongContextManager:
    """长上下文管理器——多层漏斗式降维。"""

    def __init__(self, llm, max_chunks: int = 8, max_tokens: int = 4000):
        self.llm = llm
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens


    def coarse_filter(self, chunks: list[dict], top_n: int = None) -> list[dict]:
        """按相似度排序，取 top-N。

        不是取 top-k 就完了——k 是固定的，但检索质量不稳定。
        一次检索 top-5 的相似度是 [0.91, 0.85, 0.52, 0.48, 0.45]，
        后三条几乎不相关。粗筛：只保留相似度 > 0.5 的。
        """
        if top_n is None:
            top_n = self.max_chunks

        seen_texts = set()
        filtered = []
        for c in sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True):
            text = c.get("text", "")

            if c.get("similarity", 0) < 0.5:
                continue

            text_fingerprint = text[:60]
            if text_fingerprint in seen_texts:
                continue
            seen_texts.add(text_fingerprint)

            filtered.append(c)

        return filtered[:top_n]


    def fine_filter(self, query: str, chunks: list[dict]) -> list[dict]:
        """让 LLM 判断每块是否真的相关。

        不是每块都和用户问题相关——粗筛只看向量相似度，可能把"Q3营收"和
        "Q3利润"都排在前面。精筛让 LLM 逐块判断：真的回答了问题吗？
        """
        if len(chunks) <= 3:
            return chunks

        candidates_text = "\n\n---\n".join(
            f"[{i}] {c.get('text', '')[:200]}" for i, c in enumerate(chunks)
        )
        prompt = (
            f"用户问题：{query}\n\n"
            f"以下是检索到的文本块，请判断哪些真正与问题相关。\n"
            f"只返回相关块的编号，用逗号分隔（如：0,2,5）。\n\n"
            f"{candidates_text}"
        )

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.1
            )
            numbers = re.findall(r'\d+', response)
            relevant_indices = {int(n) for n in numbers if 0 <= int(n) < len(chunks)}
            if relevant_indices:
                return [chunks[i] for i in sorted(relevant_indices)]
        except Exception:
            pass

        return chunks


    def compress_chunk(self, text: str, max_len: int = 400) -> str:
        """对单个长块做摘要压缩——保留核心信息。

        不是简单的截断（text[:400]），而是让 LLM 把 1000 字的核心
        信息浓缩到 400 字以内。代价是多一次 LLM 调用。
        """
        if len(text) <= max_len:
            return text

        prompt = (
            f"将以下文本压缩到 {max_len} 字以内，保留所有关键事实、数字和结论。\n\n{text}"
        )
        try:
            compressed = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.2
            )
            return compressed[:max_len]
        except Exception:
            return text[:max_len] + "..."


    def process(self, query: str, chunks: list[dict],
                enable_fine_filter: bool = True) -> tuple[list[dict], dict]:
        """完整的上下文处理管线——返回精炼上下文 + 处理报告。

        Returns:
            (filtered_chunks, report) — 精炼后的块 + 处理统计
        """
        initial_count = len(chunks)
        initial_tokens = sum(len(c.get("text", "")) for c in chunks) // 2

        chunks = self.coarse_filter(chunks)

        if enable_fine_filter and len(chunks) > 3:
            chunks = self.fine_filter(query, chunks)

        for c in chunks:
            if len(c.get("text", "")) > 400:
                c["text"] = self.compress_chunk(c["text"], max_len=400)

        final_tokens = sum(len(c.get("text", "")) for c in chunks) // 2

        report = {
            "initial_chunks": initial_count,
            "final_chunks": len(chunks),
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
            "compression_ratio": round(final_tokens / max(initial_tokens, 1), 2),
            "dropped": initial_count - len(chunks),
        }

        return chunks, report
