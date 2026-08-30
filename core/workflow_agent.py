import time, json
from core.agent import Agent
from core.llm import LLM
from core.registry import ToolRegistry
from memory.short_term_memory import ShortTermMemory
from utils.logger import TraceLogger


class WorkflowAgent(Agent):
    """带工作流约束的 Agent。

    工作流 = 有序步骤列表，每步有：
      - name: 步骤名（展示用）
      - tool_name / action_name: 强制使用的工具（可选，不指定则 LLM 自选）
      - required: 是否必须执行（False 则 LLM 可以跳过）
      - max_tool_calls: 本步最多调几次工具
      - prompt: 本步的 LLM 指令
    """

    def __init__(self, name: str, llm: LLM, tool_registry: ToolRegistry,
                 workflow: list[dict], system_prompt: str = ""):
        super().__init__(name, llm, system_prompt)
        self.tool_registry = tool_registry
        self.workflow = workflow
        self.tracer = TraceLogger()
        self.memory = ShortTermMemory(llm, max_rounds=20, max_tokens=4000)

    def run(self, user_input: str) -> str:
        """按工作流逐步执行，每步内 LLM 自主决策。"""
        trace_id = self.tracer.start_trace(user_input, self.name)
        self.memory.add_message("user", user_input)

        collected_info: list[str] = []

        for step_idx, step in enumerate(self.workflow):
            step_name = step.get("name", f"Step{step_idx+1}")
            is_required = step.get("required", True)

            if not is_required:
                decision = self._should_skip(step, user_input, collected_info)
                if decision:
                    self.tracer.log_step(step_idx + 1, "-", f"跳过:{step_name}", "LLM 判断不需要")
                    continue

            max_calls = step.get("max_tool_calls", 3)
            messages = self._build_step_messages(step, user_input, collected_info)

            for call_idx in range(max_calls):
                step_start = time.perf_counter()

                try:
                    response = self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=messages,
                        tools=self.tool_registry.get_fc_schemas(),
                        temperature=0.3,
                    )
                except Exception:
                    collected_info.append(f"[{step_name}] 网络异常，跳过后续检索")
                    break

                msg = response.choices[0].message
                latency = (time.perf_counter() - step_start) * 1000

                if msg.content and not msg.tool_calls:
                    collected_info.append(f"[{step_name}] {msg.content}")
                    self.tracer.log_step(step_idx + 1, f"{step_name}(Finish)", "OK", msg.content[:100],
                                         latency_ms=latency, success=True)
                    break

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    action_name = args.pop("action", "")

                    if "tool_name" in step and tool_name != step["tool_name"]:
                        messages.append({"role": "user", "content": f"此步骤只能使用 {step['tool_name']} 工具。"})
                        continue

                    result = self.tool_registry.execute(tool_name, action_name, **args)
                    self.tracer.log_step(step_idx + 1, f"{step_name}", f"{tool_name}.{action_name}",
                                         result[:100], latency_ms=latency, success=True)
                    self.memory.add_tool_result(tool_name, result, success=True)

                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            else:
                collected_info.append(f"[{step_name}] 达到最大调用次数")

        return self._generate_final_answer(user_input, collected_info)

    def _build_step_messages(self, step: dict, user_input: str,
                             collected: list[str]) -> list[dict]:
        """构造本步的 LLM 上下文。"""
        step_name = step.get("name", "")
        step_prompt = step.get("prompt", "完成当前步骤")
        tool_hint = ""
        if "tool_name" in step:
            tool_hint = f"\n你只能使用 {step['tool_name']} 工具。"

        context = self.memory.get_context()
        collected_text = "\n".join(collected[-5:])

        return [
            {"role": "system", "content": (
                f"你是{self.name}，正在执行任务。\n"
                f"当前步骤: {step_name}\n"
                f"步骤指令: {step_prompt}{tool_hint}\n"
                "如果已有足够信息完成当前步骤，直接回复结论。"
            )},
            {"role": "user", "content": (
                f"用户问题: {user_input}\n"
                f"已有信息:\n{collected_text}\n"
                f"对话上下文:\n{context}\n\n"
                f"请完成 {step_name} 步骤。"
            )}
        ]

    def _should_skip(self, step: dict, user_input: str,
                     collected: list[str]) -> bool:
        """LLM 判断可选步骤是否需要执行。简单问题可以跳过补充检索。"""
        prompt = (
            f"用户问题: {user_input}\n"
            f"已收集信息: {'; '.join(collected[-3:])}\n"
            f"可选步骤: {step.get('name', '')} — {step.get('prompt', '')}\n"
            f"是否需要执行此步骤？已回答用户问题则回复 NO，否则回复 YES。\n"
            f"只回复 YES 或 NO。"
        )
        try:
            answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            return "NO" in answer.upper()
        except Exception:
            return step.get("default_skip", False)

    def _generate_final_answer(self, user_input: str,
                               collected: list[str]) -> str:
        """基于所有收集的信息生成最终答案。"""
        prompt = (
            f"根据以下信息回答用户问题:\n\n"
            f"用户问题: {user_input}\n\n"
            f"收集的信息:\n" + "\n".join(f"- {c}" for c in collected[-10:]) + "\n\n"
            f"请用中文给出完整、准确的答案。如信息不足，如实说明。"
        )
        try:
            return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        except Exception:
            return f"基于已收集信息: {collected[-1] if collected else '信息不足'}"



RAG_WORKFLOW = [
    {
        "name": "检索文档",
        "tool_name": "rag",
        "prompt": "在知识库中检索与用户问题相关的文档内容。尝试不同的关键词以获得全面结果。",
        "max_tool_calls": 3,
        "required": True,
    },
    {
        "name": "补充检索（可选）",
        "tool_name": "rag",
        "prompt": "如果首次检索结果不足以完整回答用户问题，进行一次补充检索。",
        "max_tool_calls": 1,
        "required": False,
    },
]

