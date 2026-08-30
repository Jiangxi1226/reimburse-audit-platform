# -*- coding: utf-8 -*-
"""报销审核 Tool —— 接入现有 Tool 基类 + Registry（含 Runtime 权限闸门）。

设计（对齐"规则为主 + LLM 兜底"）：
  1. 确定性层（reimbursement.audit_engine.ReimbursementEngine）：
     金额、额度、类别合规、日期、票据校准全部代码判定 —— 不靠 LLM。
  2. LLM 兜底层：规则引擎产出的结论里，仅当存在"语义模糊点"（如某笔费用是否业务相关、
     费用类别存疑）时，才调 LLM 做最终裁决；明确违规/明确合规的**不经过 LLM**。

  对外 action（均以 `_` 前缀命名，base 反射按此枚举；内部辅助用 `__` 前缀避免被误认成 action）：
    _audit_claim     传入 items（或已解析票据），走 规则→(可选)LLM兜底→结构化结论
    _parse_receipt   解析一张票据图/PDF → 结构化字段 + 确定性校准
    _policy_lookup   查询规则（确定性，非 LLM）
"""
import json
from core.base import Tool
from reimbursement.service import audit_claim, err_wrap as _err
from reimbursement.invoice_parser import parse_receipt


def _ok(data) -> str:
    return json.dumps({"ok": True, "data": data, "error": ""}, ensure_ascii=False)


class ReimbursementTool(Tool):

    def __init__(self, use_llm_fallback: bool = True):
        super().__init__("reimbursement", "报销审核工具：审核报销单/查规则/解析票据。返回 JSON。")
        self.use_llm_fallback = use_llm_fallback

    # ---------- action（_ 前缀，供 base 反射枚举）----------
    def _audit_claim(self, items=None, receipt_file=None, purpose=None,
                     claimed_total=None) -> str:
        """主入口：审核一组报销条目（或一张票据）。

        复用 reimbursement.service 的统一编排（规则为主 + LLM 兜底仅针对语义模糊点）。
        use_llm_fallback 仅控制"审核裁决"环节是否允许调 LLM；票据识别用的 LLM 不受它限制。
        """
        receipt_files = None
        if receipt_file:
            receipt_files = [receipt_file]
        result = audit_claim(items=items, receipt_files=receipt_files,
                             purpose=purpose, claimed_total=claimed_total,
                             use_llm_arbitration=self.use_llm_fallback)
        if not result.get("ok"):
            return _err(str(result.get("error")))
        data = result["data"]
        # 仅返回业务结论给 Agent（耗时/阶段属观测值，简化 Agent 上下文）
        return _ok(data["verdict"])

    def _parse_receipt(self, file_path=None, use_llm: bool = True) -> str:
        """解析单张票据图/PDF → 结构化字段 + 金额校准结论。"""
        if not file_path:
            return _err("缺少 file_path")
        parsed = parse_receipt(file_path, use_llm=use_llm)
        if not parsed.get("ok"):
            return _err(str(parsed.get("error")))
        return _ok(parsed["data"])

    def _policy_lookup(self, category=None, city=None) -> str:
        """查询报销规则（确定性，非 LLM）。"""
        from reimbursement.audit_engine import ReimbursementEngine
        return _ok(ReimbursementEngine().policy_lookup(category, city))

    def _get_param_hints(self, action: str) -> dict:
        """按 action 精确返回参数 schema，**不继承** base 的通用 hint。

        否则 LLM 会看到 image_path/query/text/file_path 等一堆与报销无关的参数，
        把 file_path 传成 image_path（LLM 选错参数的根因，上面端到端已复现）。
        """
        action = action.lstrip("_")
        # 各 action 只暴露自己的参数，避免交给 LLM 的 schema 里出现无关字段
        per_action = {
            "audit_claim": {
                "items": {"type": "array", "items": {"type": "object"},
                          "description": "报销条目列表（category/date/amount/desc/city/itemization）"},
                "receipt_file": {"type": "string", "description": "票据文件路径（png/jpg/pdf）"},
                "purpose": {"type": "string", "description": "报销事由"},
                "claimed_total": {"type": "number", "description": "申报总额（与核定交叉核对）"},
            },
            "parse_receipt": {
                "file_path": {"type": "string", "description": "票据文件路径（png/jpg/pdf）"},
                "use_llm": {"type": "boolean", "default": True},
            },
            "policy_lookup": {
                "category": {"type": "string", "description": "规则类别（住宿/餐饮/交通）"},
                "city": {"type": "string", "description": "城市（影响住宿高档城市上限）"},
            },
        }
        hints = {"action": {"type": "string", "const": action}}
        hints.update(per_action.get(action, {}))
        return hints

    def _get_required_params(self, action: str) -> list[str]:
        action = action.lstrip("_")
        # 识别类必填票据路径，其余只需 action
        if action == "parse_receipt":
            return ["action", "file_path"]
        return ["action"]
