import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm import LLM
from core.registry import ToolRegistry
from core.react_agent import ReactAgent
from core.workflow_agent import WorkflowAgent, RAG_WORKFLOW
from tools.rag_tool import RAGTool

llm = LLM()
registry = ToolRegistry()
registry.register(RAGTool())
agent = ReactAgent(name="财务报销审核助手", llm=llm, tool_registry=registry)
workflow_agent = WorkflowAgent(
    name="财务报销审核工作流", llm=llm, tool_registry=registry,
    workflow=RAG_WORKFLOW,
)

print("财务报销审核平台已就绪（多模态票据识别 + LLM 兜底）")
print("/add <文本> | /file <路径> | /image <图片路径> | /workflow <问题> | /hybrid <问题> | q 退出")
while True:
    try:
        user_input = input("\n用户: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break
    if not user_input: continue
    if user_input in ("q", "quit", "exit"): break
    if user_input.startswith("/add "):
        print(agent.tool_registry.execute("rag", "add", text=user_input[5:]))
        continue
    if user_input.startswith("/file "):
        print(agent.tool_registry.execute("rag", "add_file", file_path=user_input[6:]))
        continue
    if user_input.startswith("/image "):
        print(agent.tool_registry.execute("rag", "add_image", image_path=user_input[7:]))
        continue
    if user_input.startswith("/workflow "):
        result = workflow_agent.run(user_input[10:])
        print(f"Workflow: {result}")
        continue
    result = agent.run(user_input)
    print(f"Agent: {result}")
