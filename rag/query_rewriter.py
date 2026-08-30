class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str, strategies: list[str] | None = None) -> list[str]:
        """按策略列表改写查询，返回查询变体列表（含原始查询）。
        单个策略失败不影响其他策略——except Exception → continue。
        """
        if strategies is None:
            strategies = ["basic"]
        results = [query]
        for strategy in strategies:
            try:
                if strategy == "basic":
                    r = self._basic_rewrite(query)
                    if r and r != query:
                        results.append(r)
                elif strategy == "multi_query":
                    for v in self._multi_query_generate(query, 3):
                        if v and v != query and v not in results:
                            results.append(v)
                elif strategy == "hyde":
                    h = self._hyde_expand(query)
                    if h and h not in results:
                        results.append(h)
            except Exception:
                continue
        return results

    def _basic_rewrite(self, query: str) -> str:
        """口语 → 规范检索查询。
        temperature=0.2：不需要创造性，只需要稳定规范化。
        例："这玩意儿咋用" → "如何使用该产品"
        """
        prompt = [
            {"role": "system", "content": "你是搜索查询规范化助手。将用户口语转为规范检索查询。只输出改写后的一句话，不要解释。"},
            {"role": "user", "content": f"原始问题：{query}\n请改写为规范检索查询："}
        ]
        return self.llm.chat(prompt, temperature=0.2).strip()

    def _multi_query_generate(self, query: str, n: int = 3) -> list[str]:
        """生成 N 个语义等价但表述不同的查询变体。
        temperature=0.7：需要多样性——如果三个变体用词一样就没有意义了。

        返回清洗后的变体列表：
        - 每行去掉前导的 "-  " 序号标记
        - 跳过空行
        - 跳过和原始查询完全相同的结果
        - 最多返回 n 个
        """
        prompt = [
            {"role": "system", "content": "生成语义等价的多个查询表述。中文、简短、每行一个。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述，每行一个。"}
        ]
        text = self.llm.chat(prompt, temperature=0.7)
        lines = [ln.strip("-  \t") for ln in text.splitlines()]
        return [ln for ln in lines if ln and ln != query][:n]

    def _hyde_expand(self, query: str) -> str | None:
        """HyDE：让 LLM 根据问题编一段假答案段落（100-200字）。
        原理：用户问句和文档陈述句在 Embedding 空间有天然偏移。
              编的假答案虽不准确，但"样子和语气"更像文档。
              用假答案搜真文档 → 跨空间对齐，提升检索命中率。
        temperature=0.3：稍微有些发挥空间但不能胡说。
        """
        prompt = [
            {"role": "system", "content": "根据用户问题写一段可能的答案段落（100-200字），包含关键术语。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段答案段落："}
        ]
        text = self.llm.chat(prompt, temperature=0.3).strip()
        return text if text else None
