import re, time, json
from core.agent import Agent
from core.llm import LLM
from core.registry import ToolRegistry
from memory.short_term_memory import ShortTermMemory
from utils.logger import TraceLogger
from utils.security import ContentFilter
from utils.decorators import retry
from core.runtime_guard import AccessContext

REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。

可用工具:
{tools}

每次回应严格按以下格式：
**Thought:** 分析当前问题，思考需要什么信息
**Action:** 工具名.操作名(参数1=值1, 参数2=值2)
或 Finish[answer=最终答案]

一次只调一个工具。工具结果不够就继续调用。

当前问题: {user_input}
对话上下文: {context}
"""


class ReactAgent(Agent):
    def __init__(self, name: str, llm: LLM, tool_registry: ToolRegistry,
                 system_prompt: str = "", max_steps: int = 5,
                 max_retries: int = 2, use_function_calling: bool = True,
                 min_steps: int = 1, max_consecutive_similar: int = 3):
        super().__init__(name, llm, system_prompt)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.use_fc = use_function_calling
        self.min_steps = min_steps
        self.max_consecutive_similar = max_consecutive_similar
        self.tracer = TraceLogger()
        self.memory = ShortTermMemory(llm, max_rounds=20, max_tokens=4000)

    def run(self, user_input: str, ctx: AccessContext = None) -> str:
        """ctx：本次调用的用户身份上下文（鉴权后传入）。缺省按 guest 最低信任处理，
        RuntimeGuard 会据此决定能调哪些工具——模型无法自己绕过。"""
        if ctx is None:
            ctx = AccessContext(user_id="anonymous", role="guest")
        trace_id = self.tracer.start_trace(user_input, self.name)
        context = self.memory.get_context()

        if self.use_fc:
            return self._run_function_calling(user_input, context, trace_id, ctx)
        return self._run_react(user_input, context, trace_id, ctx)


    def _run_function_calling(self, user_input: str, context: str, trace_id: str,
                              ctx: AccessContext) -> str:
        base_prompt = (
            f"你是{self.name}，用中文回答。可用工具见 tools 列表。\n"
            "行为约束：\n"
            "1. 每次只调用一个工具，根据返回结果决定下一步。\n"
            "2. 如果连续两次检索返回高度重叠的内容，说明信息已经足够——请直接回答，不要重复搜索。\n"
            "3. 如果工具返回错误（❌开头），分析错误原因后换一个工具或参数，不要重试同样的调用。\n"
            "4. 如果已经收集了足够信息回答用户问题，立即 Finish，不要做多余的检索。\n"
            "5. 如果两个工具都能完成任务，选择最直接的那个——不要为了调而调。\n"
            "6. 当用户问题涉及'和...什么关系'、'影响/驱动因素'这类多跳推理时，优先用 rag.search_graph 而非 rag.search——它能在跨文档实体间建立逻辑联系。"
        )
        try:
            from rag.query_classifier import get_classifier
            type_prompt = get_classifier().get_prompt(user_input)
            base_prompt += "\n\n" + type_prompt
        except Exception:
            pass

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": f"上下文:\n{context}\n\n问题: {user_input}"}
        ]

        tools_schema = self._build_tools_schema()
        last_action = None
        last_3_actions: list[str] = []
        consecutive_same = 0
        best_answer = None
        best_answer_score = 0.0

        for step in range(self.max_steps):
            step_start = time.perf_counter()

            try:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=messages,
                    tools=tools_schema,
                    temperature=0.3,
                )
            except Exception as e:
                print(f"[网络错误] FC API 调用失败 ({e})，降级到 ReAct 模式...")
                # 降级也必须带 ctx，否则 ReAct 工具调用落到 guest 基线、写操作被拒，
                # 与 FC 主路径行为不一致（安全/行为一致性 bug）。
                return self._run_react(user_input, context, trace_id, ctx=ctx)

            msg = response.choices[0].message
            latency = (time.perf_counter() - step_start) * 1000

            if msg.content and not msg.tool_calls:
                if step < self.min_steps:
                    messages.append({"role": "assistant", "content": msg.content})
                    messages.append({"role": "user", "content": (
                        "你似乎还没有调用任何工具来检索信息。请先检索相关知识再回答。"
                        "如果确实不需要检索，请再次以 Finish 结束并说明理由。"
                    )})
                    self.tracer.log_step(step + 1, "Finish(过早)", "(被拦截)",
                                         "步数不足，要求继续检索", latency_ms=latency, success=False)
                    continue

                self.tracer.log_step(step + 1, msg.content[:100], "Finish", "任务完成", latency_ms=latency)
                self.tracer.end_trace(msg.content, success=True)
                self.memory.add_message("user", user_input)
                self.memory.add_message("assistant", msg.content)
                return msg.content

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                # function 名 = "{tool}_{action}"。不能无脑 rsplit：工具名可含下划线（image_analysis），
                # action 也可含下划线（search_all）。正确做法是用注册表做前缀匹配——找到 fname 以哪个
                # 已知工具名 + "_" 开头，剩下一段就是 action。这样 rag_search_all → (rag, search_all)、
                # image_analysis_analyze → (image_analysis, analyze)。
                fname = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_name, action_name = self._split_fc_name(fname)
                args.pop("action", None)
                action_str = f"{tool_name}.{action_name}({args})"

                action_key = f"{tool_name}.{action_name}"
                if action_key == last_action:
                    consecutive_same += 1
                else:
                    consecutive_same = 0
                last_action = action_key

                last_3_actions.append(action_key)
                if len(last_3_actions) > 3:
                    last_3_actions = last_3_actions[-3:]

                observation, success = self._execute_with_retry(tool_name, action_name, args, step + 1, ctx)

                scan = ContentFilter.scan(observation)
                if not scan["safe"]:
                    observation = ContentFilter.sanitize(observation)

                self.tracer.log_step(step + 1, f"FC: {action_str}", action_str, observation,
                                     latency_ms=latency, success=success)
                self.memory.add_tool_result(tool_name, observation, success)

                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

                score = self._score_observation(observation)
                if score > best_answer_score:
                    best_answer_score = score
                    best_answer = observation

                if consecutive_same >= self.max_consecutive_similar:
                    messages.append({"role": "user", "content": (
                        f"你已经连续 {consecutive_same} 次调用了 {action_key}。"
                        "已收集了足够的信息。请基于现有检索结果给出最终答案，不要再调用工具。"
                    )})
                    consecutive_same = 0

                if len(last_3_actions) == 3:
                    if last_3_actions[0] == last_3_actions[2] and last_3_actions[0] != last_3_actions[1]:
                        messages.append({"role": "user", "content": (
                            f"检测到你反复在 {last_3_actions[0]} 和 {last_3_actions[1]} 之间切换。"
                            "这说明你已经有了足够的信息。请基于已有检索结果直接回答用户问题，不要再调用工具。"
                        )})
                        last_3_actions = []

        if best_answer and best_answer_score > 0.3:
            fallback = (
                f"经过 {self.max_steps} 步检索，以下是基于已收集信息的部分结果：\n\n"
                f"{best_answer[:1000]}\n\n（未能在限定步数内完成完整分析，建议缩小问题范围重试）"
            )
            self.tracer.end_trace(fallback, success=False, error="max_steps_partial")
        else:
            fallback = f"抱歉，在 {self.max_steps} 步内未能完成任务。"
            self.tracer.end_trace(fallback, success=False, error="max_steps")
        return fallback

    @staticmethod
    def _score_observation(observation: str) -> float:
        """工具输出质量评分 0~1。保留最佳中途结果用于 max_steps 兜底。"""
        if not observation:
            return 0.0
        if observation.startswith("❌") or "失败" in observation[:50]:
            return 0.0
        score = 0.3
        if '"ok": true' in observation or '"ok":True' in observation:
            score = 0.6
        length_bonus = min(len(observation) / 2000, 0.4)
        return min(score + length_bonus, 1.0)

    def _build_tools_schema(self):
        """把 ToolRegistry 转成 OpenAI Function Calling 格式。
        schema 生成逻辑下放到了 Tool.get_fc_schemas()，这里只做聚合。
        """
        return self.tool_registry.get_fc_schemas()

    @retry(max_attempts=3, delay=1.0)
    def _llm_chat_with_retry(self, messages: list[dict]) -> str:
        """带自动重试的 LLM 对话——网络抖动时最多重试 3 次。"""
        return self.llm.chat(messages, temperature=0.3)


    def _run_react(self, user_input: str, context: str, trace_id: str,
                   ctx: AccessContext = None) -> str:
        """ReAct 推理循环。上下文管理委托给 ShortTermMemory，
        不再用 context += 拼接字符串，避免雪球式膨胀。
        """
        self.memory.add_message("user", user_input)

        for step in range(self.max_steps):
            step_start = time.perf_counter()
            context = self.memory.get_context()

            prompt = REACT_PROMPT.format(
                tools=self.tool_registry.describe(),
                user_input=user_input, context=context
            )
            response = self._llm_chat_with_retry([{"role": "user", "content": prompt}])
            latency = (time.perf_counter() - step_start) * 1000

            thought = re.search(r"\*\*Thought:\*\*(.*)", response)
            action_match = re.search(r"\*\*Action:\*\*(.*)", response)
            thought_str = thought.group(1).strip() if thought else ""
            action_str = action_match.group(1).strip() if action_match else ""

            if not action_match:
                self.tracer.log_step(step + 1, thought_str, "(格式错误)", "未按格式输出", latency_ms=latency, success=False)
                continue

            tool_name, action_name, kwargs = self.parse_action(action_str)
            if tool_name == "Finish":
                answer = kwargs.get("answer", "")
                self.memory.add_message("assistant", answer)
                self.tracer.log_step(step + 1, thought_str, "Finish", "完成", latency_ms=latency, success=True)
                self.tracer.end_trace(answer, success=True)
                return answer

            observation, success = self._execute_with_retry(tool_name, action_name, kwargs, step + 1, ctx)
            scan = ContentFilter.scan(observation)
            if not scan["safe"]:
                observation = ContentFilter.sanitize(observation)
            self.tracer.log_step(step + 1, thought_str, action_str, observation, latency_ms=latency, success=success)
            self.memory.add_tool_result(tool_name, observation, success)

        fallback = f"抱歉，在 {self.max_steps} 步内未能完成任务。"
        self.tracer.end_trace(fallback, success=False, error="max_steps")
        return fallback


    def _execute_with_retry(self, tool_name, action_name, kwargs, step_num,
                            ctx: AccessContext = None):
        """执行工具调用，带智能重试。
        核心逻辑：
        - 权限拒绝 / 白名单外 / 需人工确认 → 不重试（确定性事实，重试只会浪费步数，
          需要 LLM 换工具或走降级，而不是重发同样的调用）
        - "不支持"/"未找到" 类错误 → 不重试（重试也没用，需要 LLM 换工具）
        - 运行时异常 → 重试 max_retries 次（可能是网络抖动等瞬态错误）
        执行权归 Runtime：无论是否传 ctx，都进 tool_registry.execute 的闸门。
        """
        NON_RETRYABLE = ["不支持的操作", "未找到工具", "权限拒绝", "不在白名单",
                         "需人工确认", "缺少用户身份", "无权调用", "参数校验失败",
                         "路径越界", "超出调用预算"]
        for attempt in range(self.max_retries + 1):
            try:
                result = self.tool_registry.execute(tool_name, action_name, ctx=ctx, **kwargs)
                if any(kw in result for kw in NON_RETRYABLE):
                    return f"❌ {result}", False
                if "失败" in result[:80] and attempt < self.max_retries:
                    continue
                return result, True
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                return f"❌ {tool_name}.{action_name} 执行异常: {e}", False
        return f"❌ {tool_name}.{action_name} 重试{self.max_retries}次后仍失败", False

    def _split_fc_name(self, fname: str) -> tuple[str, str]:
        """把 FC function 名 "{tool}_{action}" 拆成 (tool, action)。

        用注册表前缀匹配而非无脑 split：工具名可含下划线（image_analysis），action 也可含
        下划线（search_all）。遍历已知工具名，找 fname 以哪个工具名 + "_" 开头，剩余即 action。
        """
        tools = getattr(self.tool_registry, "_tools", {}) or {}
        for tname in tools:
            prefix = tname + "_"
            if fname.startswith(prefix):
                return tname, fname[len(prefix):]
        # 兜底：按最后一个下划线拆（能处理无下划线工具名 + 无下划线 action 的最简形态）
        if "_" in fname:
            return fname.rsplit("_", 1)[0], fname.rsplit("_", 1)[1]
        return fname, ""

    def parse_action(self, action_str: str):
        if action_str.startswith("Finish"):
            ans = re.search(r"Finish\[answer=(.*)\]", action_str)
            return ("Finish", None, {"answer": ans.group(1) if ans else ""})
        parts = action_str.split(".", 1)
        tool_name, right = parts[0], parts[1]
        action_name, params_str = right.split("(", 1)

        # ReAct 里 LLM 可能直接引用 FC schema 的组合 function 名（{tool}_{action}）当作工具名，
        # 如 "rag_search.advanced(...)" 会把 tool_name 解析成 "rag_search"，导致白名单按它查不到而误拒。
        # 用注册表前缀匹配还原真实工具名（能处理 image_analysis / search_all 等含下划线名）。
        tools = getattr(self.tool_registry, "_tools", {})
        if tool_name not in tools:
            real_tool, _ = self._split_fc_name(tool_name)
            if real_tool in tools:
                tool_name = real_tool
        params_str = params_str[:-1]
        kwargs = {}
        if params_str:
            for pair in params_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    kwargs[k.strip()] = v.strip()
        return tool_name, action_name, kwargs

    def get_last_trace(self):
        traces = self.tracer.list_traces(limit=1)
        return traces[0] if traces else None
