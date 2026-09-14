"""默认工具权限策略 + Guard 工厂。

把「哪些工具算读 / 哪些算写 / 谁够格调 / 多高危 / 是否需审批」集中声明在这里，
app.py 与 api.py 各调一次 build_guard() 即可获得一致的运行时闸门。
"""

from __future__ import annotations

import os

from core.runtime_guard import RuntimeGuard, AccessContext
from core.runtime_guard import RiskLevel, ToolPolicy


# 允许上传/读取文件的根目录（图④ 路径越界校验）。
# 默认取项目根，使越界校验默认开启——否则 add_file/add_image 可传入任意绝对路径
# （如 C:\Windows\system32 下的文件）读取知识库之外的内容。生产可用环境变量覆盖。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD_PATH_ROOT = os.getenv("GUARD_PATH_ROOT", "") or os.path.normpath(_PROJECT_ROOT)

# 检索类只读操作：任何角色可用，低风险
READ_ONLY_ACTIONS = [
    "search", "search_advanced", "search_all", "search_hybrid",
    "search_multi", "search_filtered", "search_graph", "qa", "stats",
]

# 写入知识库操作：高险、需确认、记预算、可回滚
WRITE_ACTIONS = {
    "add":      {"required": ["text"],             "budget": 100},
    "add_file": {"required": ["file_path"],        "budget": 50},
    "add_image": {"required": ["image_path"],      "budget": 50},
}


def _rollback_source(pipeline):
    """为文件/图片写入提供「尽力回滚」回调（图 回滚策略）。

    利用 source（文件路径）反查删除；pipeline 无该接口时优雅降级为"需人工回滚"。
    """
    def fn(**args):
        src = args.get("file_path") or args.get("image_path") or args.get("source")
        deleter = getattr(pipeline, "_delete_by_source", None)
        if deleter and src:
            try:
                deleter(src)
                return "ok"
            except Exception:
                pass
        return "no-handler"
    return fn


def build_guard(rag_tool=None, on_confirm: str = "deny",
                persist: bool = False) -> RuntimeGuard:
    """按项目现状构建默认权限闸门。

    rag_tool：用于给 rag 写操作挂回滚钩子；不传则写操作无回滚（降级）。
    on_confirm：高险操作被审批拦截时的策略。默认 'deny'（安全），
                'approve' 用于演示/测试放行路径。
    persist：是否把「幂等/预算」落到 SQLite（重启后仍生效）。生产入口应开启；
             默认 False（测试/无 DB 场景保持纯内存行为不变）。
    """
    policies: list[ToolPolicy] = []

    # ---- 1. 只读检索：任意角色可用，低风险 ----
    for action in READ_ONLY_ACTIONS:
        p = ToolPolicy(
            tool="rag", action=action,
            risk=RiskLevel.LOW,
            redact_text=False,          # 检索结果默认不整段脱敏，仅标记注入
            redact_fields=[],           # 最小化返回已在 RAGTool._simplify 做截断
        )
        # 检索结果里若夹带证件号/金额，做最小化脱敏兜底（图「数据脱敏」）
        policies.append(p)

    # ---- 2. 知识库写入：高险 + 需 admin + 需确认 + 预算 + 回滚 ----
    for action, cfg in WRITE_ACTIONS.items():
        redact_fields = ["file_path", "text"]
        policies.append(ToolPolicy(
            tool="rag", action=action,
            risk=RiskLevel.HIGH,
            roles={"admin"},
            needs_confirm=True,
            mutable=True,
            budget=cfg["budget"],
            required_args=cfg["required"],
            path_root=GUARD_PATH_ROOT,
            redact_fields=redact_fields,
            rollback=_rollback_source(getattr(rag_tool, "pipeline", None)) if rag_tool else None,
        ))

    # ---- 3. 计算器：eval 需谨慎，需 analyst/ admin ----
    policies.append(ToolPolicy(
        tool="calculator", action="calc",
        risk=RiskLevel.MEDIUM,
        roles={"admin", "analyst"},
        arg_types={"expression": str},
        required_args=["expression"],
        redact_fields=["result"],
    ))

    # ---- 4. 图片分析：读文件，路径越界校验 + 存在性兜底 ----
    policies.append(ToolPolicy(
        tool="image_analysis", action="analyze",
        risk=RiskLevel.MEDIUM,
        roles={"admin", "analyst", "user"},
        arg_types={"image_path": str, "question": str},
        required_args=["image_path"],
        path_root=GUARD_PATH_ROOT,
    ))

    # ---- 5. 报销审核：只读票据 + 规则计算，不改知识库，低险，任意角色可调 ----
    # 对齐"模型只出意图、执行权归 Runtime"：即使 LLM 想调报销工具，也必须过闸门。
    # 三个 action 均为读/计算（mutable=False），不需人工确认；file/票据路径受 path_root 越界校验。
    policies.append(ToolPolicy(
        tool="reimbursement", action="audit_claim",
        risk=RiskLevel.LOW,
        roles={"admin", "analyst", "user"},
        required_args=[],
        arg_types={"items": list, "receipt_file": str, "purpose": str, "claimed_total": float},
        path_root=GUARD_PATH_ROOT,
    ))
    policies.append(ToolPolicy(
        tool="reimbursement", action="parse_receipt",
        risk=RiskLevel.LOW,
        roles={"admin", "analyst", "user"},
        required_args=["file_path"],
        arg_types={"file_path": str, "use_llm": bool},
        path_root=GUARD_PATH_ROOT,
    ))
    policies.append(ToolPolicy(
        tool="reimbursement", action="policy_lookup",
        risk=RiskLevel.LOW,
        roles={"admin", "analyst", "user"},
        required_args=[],
        arg_types={"category": str, "city": str},
    ))

    guard = RuntimeGuard(policies=policies, on_confirm=on_confirm, persist=persist)
    return guard


def make_ctx(role: str = "user", user_id: str = "demo_user",
             tenant_id: str = "t0", scopes: set[str] | None = None) -> AccessContext:
    """快捷构造一个身份上下文（便于入口 / 测试复用）。"""
    return AccessContext(user_id=user_id, tenant_id=tenant_id,
                         role=role, scopes=scopes or set())
