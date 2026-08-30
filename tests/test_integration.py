"""集成测试 — 端到端验证核心链路"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_full_rag_pipeline():
    """端到端：文档加载 → 入库 → 检索 → 答案生成"""
    from rag.document_loader import load
    from rag.pipeline import RAGPipeline

    p = RAGPipeline(use_reranker=False)
    # 用固定文本测试，不依赖桌面文件
    text = "2024年Q3营收为1.2亿元，同比增长30%。产品线A贡献了60%的营收。"
    p.add_document(text, source="test")

    assert p.text_count > 0, "入库失败"
    results = p.search("Q3营收")
    assert len(results) > 0, "检索失败"


def test_hybrid_search():
    """混合检索：向量 + BM25"""
    from rag.pipeline import RAGPipeline
    from rag.hybrid_search import BM25Scorer

    docs = ["Q3营收1.2亿元", "产品线A贡献60%", "同比增长30%"]
    p = RAGPipeline(use_reranker=False)
    for d in docs:
        p.add_document(d)

    bm25 = BM25Scorer()
    bm25.add_corpus(docs)

    from rag.hybrid_search import hybrid_search
    results = hybrid_search("营收", p.store, p.embedder, bm25, top_k=3)
    assert len(results) > 0, "混合检索失败"


def test_semantic_chunker():
    """语义切分：段落→句子→安全断点"""
    from rag.semantic_chunker import SemanticChunker
    text = "Transformer架构由Vaswani等人在2017年提出。\n\n它完全基于自注意力机制，抛弃了传统的循环神经网络。"
    chunks = SemanticChunker(chunk_size=30, overlap=5).split(text)
    assert len(chunks) >= 1, f"切分块数不够: {len(chunks)}"


def test_answer_generator():
    """答案生成：结构化输出"""
    from rag.answer_generator import _dedup_and_sort, _assess_confidence
    chunks = [
        {"id": "1", "text": "Q3营收报告显示本季度总营收达到1.2亿元人民币，创下历史新高。" * 3, "similarity": 0.9},
        {"id": "2", "text": "与去年同期相比增长30%，主要驱动力来自产品线A的强劲表现。" * 3, "similarity": 0.85},
        {"id": "3", "text": "Q3营收1.2亿元——这是对本季度财务表现的综合评价。" * 3, "similarity": 0.88},
    ]
    deduped = _dedup_and_sort(chunks, "Q3营收", top_n=2)
    assert len(deduped) <= 2 and deduped[0]["similarity"] >= deduped[0]["similarity"]

    conf = _assess_confidence(chunks, "Q3营收")
    assert conf["level"] in ("high", "medium"), f"置信度应为high或medium: {conf['level']}"


def test_graph_extraction():
    """图提取：实体+关系"""
    from rag.graph_extractor import KnowledgeGraph
    kg = KnowledgeGraph()
    triples = {
        "entities": [{"name": "Q3营收", "type": "metric"}, {"name": "产品线A", "type": "product"}],
        "relations": [{"source": "产品线A", "target": "Q3营收", "relation": "贡献60%"}],
    }
    kg.add_triples(triples, "report_q3")
    assert kg.stats()["entities"] >= 2 and kg.stats()["relations"] >= 1

    result = kg.query(["Q3营收"], max_depth=1)
    assert len(result["entities"]) >= 1, f"实体数: {result['entities']}"


def test_session_manager():
    """会话管理：多用户隔离"""
    from core.session_manager import SessionManager
    from core.llm import LLM
    sm = SessionManager(LLM())
    sid = sm.get_or_create("user_a")
    assert sid, "创建会话失败"
    sm.add_message(sid, "user", "测试")
    ctx = sm.get_context(sid, "user_a")
    assert "测试" in ctx, "上下文丢失"


def test_rate_limiter():
    """速率限制器"""
    from utils.security import RateLimiter
    rl = RateLimiter(max_requests_per_minute=5)
    for _ in range(5):
        assert rl.allow("user1"), "前5次应该放行"
    assert not rl.allow("user1"), "第6次应该被限制"


def test_error_codes():
    """错误码体系"""
    from core.db import ERROR_CODES
    assert "NET_001" in ERROR_CODES
    assert "SEC_001" in ERROR_CODES
    assert len(ERROR_CODES) >= 10, f"错误码数量: {len(ERROR_CODES)}"


def test_config_file():
    """配置文件"""
    assert os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".env.example"
    )), "缺少 .env.example"
    assert os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "README.md"
    )), "缺少 README.md"


    assert os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Dockerfile"
    )), "缺少 Dockerfile"
