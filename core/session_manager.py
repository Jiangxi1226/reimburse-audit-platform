import threading
from datetime import datetime
from memory.short_term_memory import ShortTermMemory
from memory.episodic_memory import EpisodicMemory
from memory.long_context import LongContextManager


class SessionManager:
    def __init__(self, llm, max_sessions_per_user: int = 10,
                 session_timeout_minutes: int = 60,
                 enable_long_context: bool = False):
        self.llm = llm
        self.max_sessions_per_user = max_sessions_per_user
        self.session_timeout_minutes = session_timeout_minutes

        self.long_context = LongContextManager(llm) if enable_long_context else None

        self._user_sessions: dict[str, dict[str, dict]] = {}
        self._memories: dict[str, ShortTermMemory] = {}

        self.episodic = EpisodicMemory()

        self._lock = threading.RLock()


    def get_or_create(self, user_id: str, session_id: str = None) -> str:
        """获取或创建会话。返回 session_id。

        - 传 session_id 且未过期 → 复用
        - 传 session_id 但已过期 → 新建
        - 不传 session_id → 新建
        """
        with self._lock:
            if session_id and session_id in self._memories:
                if not self._is_expired(session_id):
                    self._touch(session_id)
                    return session_id
                else:
                    self._cleanup_session(user_id, session_id)

            return self._create(user_id)

    def _create(self, user_id: str) -> str:
        """创建新会话，超限时清理最旧的。"""
        import uuid
        sid = str(uuid.uuid4())[:12]
        now = datetime.now()

        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = {}
        if len(self._user_sessions[user_id]) >= self.max_sessions_per_user:
            oldest = min(self._user_sessions[user_id].keys(),
                        key=lambda k: self._user_sessions[user_id][k].get("last_active", now))
            self._cleanup_session(user_id, oldest)

        self._memories[sid] = ShortTermMemory(self.llm, max_rounds=20, max_tokens=4000)
        self._user_sessions[user_id][sid] = {
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }

        try:
            from core.db import create_session
            create_session(user_id, title="")
        except Exception:
            pass

        return sid

    def _is_expired(self, session_id: str) -> bool:
        """检查会话是否过期。"""
        for user_sessions in self._user_sessions.values():
            if session_id in user_sessions:
                last_active = user_sessions[session_id].get("last_active")
                if last_active:
                    elapsed = (datetime.now() - last_active).total_seconds() / 60
                    return elapsed > self.session_timeout_minutes
        return True

    def _touch(self, session_id: str):
        """更新会话最后活跃时间。"""
        for user_sessions in self._user_sessions.values():
            if session_id in user_sessions:
                user_sessions[session_id]["last_active"] = datetime.now()
                user_sessions[session_id]["message_count"] += 1
                break

    def _cleanup_session(self, user_id: str, session_id: str):
        """清理过期/被淘汰的会话。"""
        if session_id in self._memories:
            del self._memories[session_id]
        if user_id in self._user_sessions and session_id in self._user_sessions[user_id]:
            del self._user_sessions[user_id][session_id]


    def get_context(self, session_id: str, user_id: str = None) -> str:
        """获取会话上下文 + 用户情景记忆。"""
        with self._lock:
            memory = self._memories.get(session_id)
            if not memory:
                return ""

            context = memory.get_context()

            if user_id:
                try:
                    relevant = self.episodic.recall(
                        f"用户{user_id}的偏好和重要信息", top_k=3
                    )
                    if relevant:
                        context += "\n[用户偏好记忆]\n" + "\n".join(
                            m["text"][:200] for m in relevant
                        )
                except Exception:
                    pass

            return context

    def add_message(self, session_id: str, role: str, content: str):
        """记录对话消息。"""
        with self._lock:
            memory = self._memories.get(session_id)
            if memory:
                memory.add_message(role, content)

    def add_tool_result(self, session_id: str, tool_name: str,
                        result: str, success: bool = True):
        """记录工具调用结果。"""
        with self._lock:
            memory = self._memories.get(session_id)
            if memory:
                memory.add_tool_result(tool_name, result, success)


    def remember_preference(self, user_id: str, content: str, importance: float = 0.7):
        """记录用户偏好到情景记忆。"""
        self.episodic.remember(content, importance=importance, category="preference")

    def filter_retrieval_results(self, query: str, chunks: list[dict]) -> tuple[list[dict], dict]:
        """用 LongContextManager 对检索结果做三层漏斗式降维。

        未启用 long_context 时直接返回原结果。
        Returns: (filtered_chunks, report)
        """
        if not self.long_context or not chunks:
            return chunks, {"skipped": True, "reason": "long_context 未启用"}
        return self.long_context.process(query, chunks, enable_fine_filter=True)


    def stats(self) -> dict:
        with self._lock:
            total_sessions = sum(len(s) for s in self._user_sessions.values())
            total_users = len(self._user_sessions)
            active_sessions = sum(
                1 for s in self._user_sessions.values()
                for sid in s if not self._is_expired(sid)
            )
            episodic_stats = self.episodic.stats()
            return {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "episodic_memories": episodic_stats.get("total_episodes", 0),
            }
