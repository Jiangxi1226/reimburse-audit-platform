"""核心模块基础测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_llm_init():
    from core.llm import LLM
    llm = LLM()
    assert llm.model is not None
    assert llm.client is not None


def test_tool_base():
    from core.base import Tool
    t = Tool("test", "测试工具")
    assert t.name == "test"
    assert t.execute("nonexistent") != ""


def test_registry():
    from core.registry import ToolRegistry
    from core.base import Tool
    r = ToolRegistry()
    r.register(Tool("test", "desc"))
    assert len(r.list_tools()) == 1


def test_chunker():
    from rag.chunker import Chunker
    c = Chunker(chunk_size=10, overlap=2)
    chunks = list(c.split("A" * 30))
    assert len(chunks) >= 3


def test_react_agent_parse():
    from core.react_agent import ReactAgent
    from core.llm import LLM
    from core.registry import ToolRegistry
    agent = ReactAgent("t", LLM(), ToolRegistry())
    t, a, k = agent.parse_action("rag.search(query=hello, top_k=5)")
    assert t == "rag" and a == "search"
    t2, a2, k2 = agent.parse_action("Finish[answer=done]")
    assert t2 == "Finish" and k2["answer"] == "done"
