from abc import ABC


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


    def execute(self, action: str, **kwargs) -> str:
        method_name = f"_{action}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            try:
                return method(**kwargs)
            except Exception as e:
                return f"[{self.name}] 执行 '{action}' 失败: {e}"
        return f"[{self.name}] 不支持的操作: '{action}'"

    def parameters(self) -> str:
        """人类可读的工具描述（给 ReAct prompt 和调试用）"""
        actions = self.list_actions()
        return f"工具 {self.name}: {self.description}\n支持操作: {', '.join(actions)}"


    def list_actions(self) -> list[str]:
        """返回操作名列表（去下划线前缀），替代原来的字符串解析。

        过滤规则：
          - 仅保留单下划线前缀的方法（_xxx => xxx），即"action"约定。
          - 排除双下划线私有方法（__xxx）——它们会被 Python name-mangling 成
            `_ClassName__xxx`，仍以单下划线开头，只能靠含 "__" 判断剔除。
          - 排除框架约定的辅助方法（_get_param_hints / _get_required_params 等）。
        """
        _builtin = {"_get_param_hints", "_get_required_params"}
        return [
            attr[1:] for attr in dir(self)
            if attr.startswith("_") and "__" not in attr
            and callable(getattr(self, attr))
            and attr not in _builtin
        ]

    def get_fc_schemas(self) -> list[dict]:
        """为每个操作生成一个 OpenAI Function Calling schema。
        替代原来在 react_agent 里字符串解析 parameters() 的做法。
        """
        schemas = []
        for action in self.list_actions():
            schemas.append({
                "type": "function",
                "function": {
                    # 每个操作都必须有唯一 function 名：DeepSeek 等地要求 Tool names must be unique，
                    # 若统一用工具名，RAGTool 的 N 个 action 会生成 N 个同名 "rag"，被 LLM 以 400 拒绝。
                    # 用 {tool}_{action} 组合名，runtime 侧再按最后一个下划线切回 tool/action。
                    "name": f"{self.name}_{action}",
                    "description": f"{self.description}（操作: {action}）",
                    "parameters": {
                        "type": "object",
                        "properties": self._get_param_hints(action),
                        "required": self._get_required_params(action),
                    }
                }
            })
        return schemas

    def _get_param_hints(self, action: str) -> dict:
        """返回该操作需要的参数 schema。子类可以覆盖此方法来精确约束参数。

        qa 动作后端（RAGTool._qa）用的是命名为 question 的参数；search 系列用 query。
        若只返回一个含 query/question 的超集，LLM 可能为 qa 错填 query，后端就会收到
        "unexpected keyword argument 'query'"。这里按 action 精确区分文本框参数名。
        """
        text_key = "question" if action == "qa" else "query"
        text_desc = "问题内容" if action == "qa" else "搜索内容或操作参数"
        return {
            "action": {"type": "string", "const": action},
            text_key: {"type": "string", "description": text_desc},
            "top_k":  {"type": "integer", "default": 5},
            "file_path":  {"type": "string", "description": "文件路径"},
            "image_path": {"type": "string", "description": "图片路径"},
            "text":        {"type": "string", "description": "文本内容"},
            "expression":  {"type": "string", "description": "数学表达式"},
        }

    def _get_required_params(self, action: str) -> list[str]:
        """返回该操作必须填写的参数名列表。默认只需要 action。
        qa 必须填 question；其余检索类必须填 query（若 hint 含 query 时）。"""
        if action == "qa":
            return ["action", "question"]
        return ["action"]
