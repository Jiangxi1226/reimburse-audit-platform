class ShortTermMemory:
    def __init__(self, llm, max_rounds: int = 20, max_tool_result_len: int = 300,
                 max_tokens: int = 4000):
        self.llm = llm
        self.max_rounds = max_rounds
        self.max_tool_result_len = max_tool_result_len
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.summary: str = ""
        self._retrieval_cache: dict[str, str] = {}


    def add_message(self, role: str, content: str):
        if role == "tool" and len(content) > self.max_tool_result_len:
            content = content[:self.max_tool_result_len] + "..."
        self.messages.append({"role": role, "content": content})
        self._maybe_compress()

    def add_tool_result(self, tool_name: str, result: str, success: bool = True):
        status = "✅" if success else "❌"
        formatted = f"[{status} {tool_name}]: {result}"
        if len(formatted) > self.max_tool_result_len:
            formatted = formatted[:self.max_tool_result_len] + "..."
        self.messages.append({"role": "tool", "content": formatted})
        if "检索" in tool_name or "搜索" in tool_name or "查" in tool_name:
            cache_key = f"{tool_name}:{result[:80]}"
            self._retrieval_cache[cache_key] = result[:self.max_tool_result_len]
        self._maybe_compress()


    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[对话历史摘要]\n{self.summary}")
        if self._retrieval_cache:
            parts.append("\n[历史检索结果（缓存）]")
            for i, (key, value) in enumerate(self._retrieval_cache.items(), 1):
                parts.append(f"{i}. {value}")
        parts.append("\n[最近对话]")
        for msg in self.messages:
            role_label = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}
            parts.append(f"{role_label.get(msg['role'], msg['role'])}: {msg['content']}")
        return "\n".join(parts)


    def clear(self):
        self.messages.clear()
        self.summary = ""
        self._retrieval_cache.clear()


    def _maybe_compress(self):
        """用 token 估算代替消息数阈值，更贴近 context window 实际限制"""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        estimated_tokens = total_chars // 2
        if estimated_tokens > self.max_tokens or len(self.messages) > self.max_rounds * 5:
            self._compress()

    def _compress(self):
        """保留最近消息，压缩早期消息为摘要。检索缓存不受影响。"""
        keep_count = self.max_rounds * 2
        if len(self.messages) <= keep_count:
            return
        old = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]
        conv = "\n".join(
            f"[{m['role']}]: {m['content'][:300]}" for m in old
        )
        prompt = [
            {"role": "system", "content": "将对话压缩为一段简洁摘要（200字内）。保留：用户所有提问、已获取的关键信息、当前任务状态。不要省略具体的文档内容或数据。"},
            {"role": "user", "content": f"对话：\n{conv}\n\n摘要："}
        ]
        try:
            new_summary = self.llm.chat(prompt, temperature=0.2)
            self.summary = f"{self.summary}\n{new_summary}" if self.summary else new_summary
            if len(self.summary) > 800:
                self.summary = self.summary[-800:]
        except Exception:
            pass
