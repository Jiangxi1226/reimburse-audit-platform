# -*- coding: utf-8 -*-
"""MCP Server —— 把报销审核能力标准化成 MCP 工具，供任意 MCP 客户端调用。

设计要点：
  1. **执行权不下放**：MCP 客户端传来的只是一个"调用意图"，真正执行仍走
     `ToolRegistry.execute()` → `RuntimeGuard` 七道闸门（白名单/角色/参数
     schema/路径越界/风险分级/审计/幂等）。MCP 只是多了一个"入口"，
     不新增任何绕过闸门的旁路。
  2. **身份由部署方决定**：MCP 客户端本身不带可信身份，因此角色从环境变量读
     `MCP_USER_ROLE`（默认最低信任的 user）。要更高权限必须由部署方显式配置，
     不接受客户端在参数里自报角色。
  3. **两种原语都用上**：`tools` 暴露动作（审单/解析票据/查规则），
     `resources` 暴露只读的规则库（policy://rules），客户端可直接读取当上下文。

启动（stdio，供 Claude Desktop / Cursor 等客户端配置）：
    python mcp_server.py

自检（不依赖任何客户端）：
    python mcp_client_demo.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core.guard_policies import build_guard, make_ctx  # noqa: E402
from core.registry import ToolRegistry  # noqa: E402
from reimbursement.reimbursement_tool import ReimbursementTool  # noqa: E402


# ---------- 身份与装配 ----------
# 角色只认部署方的环境变量，不接受调用方在请求参数里自报（防止越权提权）。
MCP_USER_ROLE = os.getenv("MCP_USER_ROLE", "user")
MCP_USER_ID = os.getenv("MCP_USER_ID", "mcp_client")
MCP_TENANT_ID = os.getenv("MCP_TENANT_ID", "t0")

# 是否允许"语义模糊点"调 LLM 兜底；关闭后整条链路零 LLM 依赖（可离线跑）。
_USE_LLM_FALLBACK = os.getenv("REIMBURSEMENT_LLM_FALLBACK", "1") not in ("0", "false", "False")

_registry = ToolRegistry()
_registry.register(ReimbursementTool(use_llm_fallback=_USE_LLM_FALLBACK))
_registry.set_guard(build_guard())
_ctx = make_ctx(role=MCP_USER_ROLE, user_id=MCP_USER_ID, tenant_id=MCP_TENANT_ID)

mcp = FastMCP("reimbursement-audit")


def _call(action: str, **kwargs) -> str:
    """唯一执行通道：意图 → RuntimeGuard → 后端工具。

    None 值不往下传：Guard 会按 arg_types 做类型校验，显式 None 会被判成类型不符。
    """
    payload = {k: v for k, v in kwargs.items() if v is not None}
    return _registry.execute("reimbursement", action, ctx=_ctx, **payload)


# ---------- tools ----------
@mcp.tool()
def audit_claim(
    items: list[dict] | None = None,
    receipt_file: str | None = None,
    purpose: str | None = None,
    claimed_total: float | None = None,
) -> str:
    """审核一组报销条目（或一张票据），返回结构化审核结论。

    裁决规则：金额/额度/类别/日期等可枚举判断由确定性规则引擎完成，
    仅当存在语义模糊点（如费用是否业务相关）时才由 LLM 兜底裁决。

    Args:
        items: 报销条目列表，每条形如
            {"category": "住宿|餐饮|交通", "date": "YYYY-MM-DD", "amount": 350.0,
             "desc": "事由说明", "city": "南京", "itemization": "是否明细"}
        receipt_file: 票据文件路径（png/jpg/pdf）。与 items 二选一或同时传。
        purpose: 本次报销的事由。
        claimed_total: 申报总额，用于与核定金额交叉核对。
    """
    return _call(
        "audit_claim",
        items=items,
        receipt_file=receipt_file,
        purpose=purpose,
        claimed_total=claimed_total,
    )


@mcp.tool()
def parse_receipt(file_path: str, use_llm: bool = True) -> str:
    """解析单张票据（图片/PDF）为结构化字段，并给出金额校准结论。

    Args:
        file_path: 票据文件路径（png/jpg/pdf）。
        use_llm: 是否允许调用 LLM 做版式理解；False 时走纯 OCR + 规则解析。
    """
    return _call("parse_receipt", file_path=file_path, use_llm=use_llm)


@mcp.tool()
def policy_lookup(category: str | None = None, city: str | None = None) -> str:
    """查询报销规则（确定性计算，不经过 LLM）。

    Args:
        category: 规则类别，如「住宿」「餐饮」「交通」。不传返回全部规则。
        city: 城市，影响住宿类的高档城市上限判定。
    """
    return _call("policy_lookup", category=category, city=city)


# ---------- resources ----------
@mcp.resource("policy://rules")
def policy_rules() -> str:
    """报销规则库（只读）。客户端可直接读取作为上下文，无需先调工具。"""
    return _call("policy_lookup")


@mcp.resource("policy://capabilities")
def capabilities() -> str:
    """本服务暴露的能力清单与身份基线（只读，便于客户端自检）。"""
    return json.dumps(
        {
            "server": "reimbursement-audit",
            "tools": ["audit_claim", "parse_receipt", "policy_lookup"],
            "resources": ["policy://rules", "policy://capabilities"],
            "execution": "所有调用经 RuntimeGuard 闸门后才落到后端工具",
            "identity": {"role": MCP_USER_ROLE, "user_id": MCP_USER_ID,
                         "tenant_id": MCP_TENANT_ID},
            "llm_fallback": _USE_LLM_FALLBACK,
        },
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    # stdio 传输：stdout 必须是纯协议帧，任何日志只能走 stderr。
    mcp.run()
