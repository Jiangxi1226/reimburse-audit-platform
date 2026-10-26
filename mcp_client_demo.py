# -*- coding: utf-8 -*-
"""MCP 客户端自检脚本：连上本项目的 MCP Server，走一遍「发现 → 读资源 → 调工具」。

用途有两个：
  1. 不装 Claude Desktop / Cursor 也能验证 MCP Server 是否真的可用；
  2. 截图或录屏即可作为「MCP 已落地」的证据。

跑法：
    python mcp_client_demo.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _preview(payload, limit: int = 600) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"…（共 {len(text)} 字符）"


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(_ROOT, "mcp_server.py")],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("== 工具发现 ==")
            for t in tools.tools:
                desc = (t.description or "").strip().splitlines()[0]
                print(f"  - {t.name}: {desc}")

            resources = await session.list_resources()
            print("\n== 资源发现 ==")
            for r in resources.resources:
                print(f"  - {r.uri}")

            print("\n== 读资源 policy://rules ==")
            rule_res = await session.read_resource("policy://rules")
            for c in rule_res.contents:
                print("  " + _preview(getattr(c, "text", ""), 300).replace("\n", "\n  "))

            print("\n== 调工具 policy_lookup(住宿, 南京) ==")
            res = await session.call_tool("policy_lookup", {"category": "住宿", "city": "南京"})
            for c in res.content:
                print("  " + _preview(getattr(c, "text", ""), 400).replace("\n", "\n  "))

            print("\n== 调工具 audit_claim（含一条超额住宿）==")
            items = [
                {"category": "住宿", "date": "2026-09-12", "amount": 380,
                 "desc": "南京出差住宿", "city": "南京"},
                {"category": "住宿", "date": "2026-09-13", "amount": 980,
                 "desc": "南京出差住宿", "city": "南京"},
            ]
            res = await session.call_tool(
                "audit_claim", {"items": items, "purpose": "客户拜访", "claimed_total": 1360}
            )
            for c in res.content:
                print("  " + _preview(getattr(c, "text", ""), 800).replace("\n", "\n  "))

    print("\nMCP 链路自检完成。")


if __name__ == "__main__":
    asyncio.run(main())
