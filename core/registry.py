from core.base import Tool


class ToolRegistry:
    """工具注册表 + Runtime 执行入口。

    设计原则（对齐架构图）：模型产物（tool_name/action/args）只是**意图**，
    `execute()` 是唯一执行通道——若注入了 RuntimeGuard，则任何调用先过闸门，
    由 Guard 决定 允许 / 需确认 / 拒绝，再决定是否真正落到后端工具。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._guard = None          # RuntimeGuard | None（None = 不拦截，向后兼容）
        self._default_ctx = None    # 入口级"默认信任基线"（未显式传身份时兜底）

    # ---- Guard 对接 ----
    def set_guard(self, guard):
        """注入权限闸门。执行权交给 Runtime。"""
        self._guard = guard
        if guard is not None and guard.audit_log is None:
            try:
                from core.db import log_audit
                guard.audit_log = log_audit
            except Exception:
                pass

    def set_ctx_default(self, ctx):
        """设置本入口的默认身份基线：Gradio 可设 admin，API 可设 analyst 等。
        显式传入的 ctx 优先级高于此默认值。"""
        self._default_ctx = ctx

    @property
    def guard(self):
        return self._guard

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def execute(self, tool_name: str, action: str, ctx=None, **kwargs) -> str:
        if self._guard is None:
            return self._run(tool_name, action, **kwargs)

        # 无显式身份上下文时，用入口配置的"默认信任基线"兜底（执行权仍在 Runtime）。
        # 未配置基线时退到最低信任 guest（写操作会被拒）。
        if ctx is None:
            if self._default_ctx is not None:
                ctx = self._default_ctx
            else:
                from core.runtime_guard import AccessContext
                ctx = AccessContext(user_id="anonymous", role="guest")
        decision, result = self._guard.execute(tool_name, action, kwargs, ctx,
                                               dispatcher=self._run)
        return result

    def _run(self, tool_name: str, action: str, **kwargs) -> str:
        """真正的后端工具执行（Guard 放行后才会到这里）。

        契约：与 RuntimeGuard 的 dispatcher 对齐 —— 参数以 **kwargs 展开传入，
        确保 Guard 用 dispatcher(tool, action, **args) 调用时能正确落到后端。
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return f"未找到工具 '{tool_name}'"
        return tool.execute(action, **kwargs)

    def describe(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines) if lines else "暂无可用工具"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_fc_schemas(self) -> list[dict]:
        """聚合所有工具的 FC schemas，不再需要外部字符串解析"""
        schemas = []
        for tool in self._tools.values():
            schemas.extend(tool.get_fc_schemas())
        return schemas
