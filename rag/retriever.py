from rag.query_rewriter import QueryRewriter


def _basic_rewrite(query: str, llm) -> str:
    """口语 → 规范查询。例："这玩意儿咋用" → "如何使用该产品"
    Embedding 模型对正式文本和口语的向量编码距离很大，
    改写后更接近文档用词，提升余弦相似度匹配效果。"""
    return QueryRewriter(llm)._basic_rewrite(query)


def _mqe_expand(query: str, llm, n: int = 3) -> list[str]:
    """MQE（Multi-Query Expansion）：同一语义 × 3 种表述。
    例："Q3营收" → ["第三季度营业收入", "Q3季度营收数据", "2024年第三季度财务收入"]
    每种表述可能命中不同的文档块——一份报告写"营业收入"、另一份写"营收数据"。"""
    return QueryRewriter(llm)._multi_query_generate(query, n)


def _hyde_expand(query: str, llm) -> str | None:
    """HyDE（Hypothetical Document Embeddings）：LLM 编一段假答案。
    原理：用户问句和文档陈述句在向量空间有天然偏移。
    让 LLM 编一段"可能是正确答案"的段落 → 用这段假答案搜真文档。
    假答案和真文档都是陈述句 → 向量空间更接近 → 检索命中的可能性更高。
    代价：多一次 LLM 调用，token 消耗和延迟翻倍。默认关闭。"""
    return QueryRewriter(llm)._hyde_expand(query)


def search_expanded(query: str, pipeline, llm, top_k: int = 5,
                    enable_basic: bool = True,
                    enable_mqe: bool = True,
                    enable_hyde: bool = False,
                    enable_hybrid: bool = False):
    """三路查询扩展 + 多路检索 + 合并去重。

    Args:
        query: 用户原始查询
        pipeline: RAGPipeline 实例（提供 .search() / .search_hybrid() 方法）
        llm: LLM 实例（用于查询改写）
        top_k: 最终返回条数
        enable_basic: 是否启用 Basic 改写（默认开）
        enable_mqe: 是否启用 MQE 多查询扩展（默认开）
        enable_hyde: 是否启用 HyDE 假设文档（默认关，要额外的 LLM 调用）
        enable_hybrid: 是否用混合检索替代纯向量检索（默认关）

    Returns:
        去重合并后的 top_k 条结果，按 similarity 降序
    """
    search_fn = pipeline.search_hybrid if enable_hybrid else pipeline.search

    queries = [query]

    if enable_basic:
        rewritten = _basic_rewrite(query, llm)
        if rewritten and rewritten != query:
            queries.append(rewritten)

    if enable_mqe:
        queries.extend(_mqe_expand(query, llm))

    if enable_hyde:
        hyde_text = _hyde_expand(query, llm)
        if hyde_text:
            queries.append(hyde_text)

    seen: dict[str, dict] = {}

    for q in queries:
        for r in search_fn(q, top_k=top_k * 2):
            if r["id"] not in seen or r["similarity"] > seen[r["id"]]["similarity"]:
                seen[r["id"]] = r

    merged = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

    return merged[:top_k]
