# -*- coding: utf-8 -*-
"""Agent 编排端到端验证：LLM(ReactAgent/FC) → 选 reimbursement 工具 → 通过 Runtime 闸门 → 出结论。

模拟 api.py 的启动装配：
  LLM + ToolRegistry(注册 ReimbursementTool) + Runtime Guard(报销策略) + ReactAgent
然后以自然语言让 Agent 走"解析票据 → 审核 → 结论"整条链路。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.llm import LLM
from core.registry import ToolRegistry
from core.react_agent import ReactAgent
from core.guard_policies import build_guard, make_ctx
from reimbursement.reimbursement_tool import ReimbursementTool

SR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_receipts")


def main():
    llm = LLM()
    registry = ToolRegistry()
    registry.register(ReimbursementTool())
    registry.set_guard(build_guard(on_confirm="deny"))
    registry.set_ctx_default(make_ctx(role="analyst", user_id="api_default"))

    agent = ReactAgent(name="报销审核助手", llm=llm, tool_registry=registry,
                       system_prompt="你是财务报销审核助手，走规则为主+LLM兜底链路。", max_steps=4)

    receipt = os.path.join(SR, "餐饮_商务宴请.png")
    q = (f"帮我审核这张报销票据：{receipt}。票据日期2026-08-01，事由出差商务宴请，"
         "请解析票据并给出审核结论（是否可报销、可报销金额、需修正/拒项）。")
    print("【用户】", q, "\n")

    answer = agent.run(q, ctx=make_ctx(role="analyst", user_id="demo_user"))
    print("【Agent 结论】\n", answer, "\n")

    # 打印本次工具调用轨迹
    try:
        from utils.logger import TraceLogger
        logs = TraceLogger().list_traces(limit=1)
        if logs:
            print("【轨迹】")
            for i, (step, tool, obs, lat, ok) in enumerate(logs[0].get("steps", [])):
                print(f"  step{i+1}: {tool} -> {(obs or '')[:80]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
