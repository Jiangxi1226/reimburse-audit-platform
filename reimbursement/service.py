# -*- coding: utf-8 -*-
"""报销审核服务中心 —— 整条链路的统一编排（端点 / Agent 工具复用同一份逻辑）。

专业职责（对齐"规则为主 + LLM 兜底、少依赖 LLM"）：
  1. OCR：复用 rag.document_loader（PaddleOCR）拿票据原文。
  2. 字段抽取：LLM 仅做"识别"（票据类型/金额/日期/明细），不做任何计算。
  3. 确定性校准：reconcile 校验分项之和 == 票面总额，超容差标记待人工。
  4. 规则审核：ReimbursementEngine.audit() 全代码判定（金额/额度/合规/日期）。
  5. LLM 兜底裁决：仅当结论为"合规(approve)"但仍存在语义模糊点（如类别是"其他"、
     描述含"杂项"）时，才调 LLM 判断是否业务相关；明确合规/明确违规绝不交 LLM。
  6. 多票据 + 事由/申报总额 交叉核对：申报总额与票据核定额不一致 → 提示。

产出：结构化审核结论 + 每阶段耗时 + llm_used 标志（用作"少依赖 LLM"量化证据）。
"""
import os, json, time, re

from reimbursement.audit_engine import ReimbursementEngine
from reimbursement.invoice_parser import parse_receipt, to_audit_items


# LLM 兜底裁决提示：只有"语义模糊"才触发；已明确合规/违规不进来。
_LLM_ARBITRATE_PROMPT = '''你是财务报销审核裁决助手。下面是确定性规则引擎已给出一份初判结论，
以及一个"语义模糊点"（无法用硬规则判定、需要理解业务意图的问题）。请只针对这个模糊点裁决，输出 JSON：

{{"is_legitimate": true/false, "reason": "一段话说明"}}

模糊点：{ambiguity}
规则初判摘要：{summary}

只输出 JSON，不要多余文字。
'''


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                return {}
    return {}


def find_ambiguity(items: list[dict], verdict: dict) -> str:
    """识别"无法用硬规则判定、需业务语义判断"的模糊点。

    规则引擎已明确 reject / partial / manual_review 的，属于确定性结论，不算模糊点；
    只有 approve 仍带疑点（类别"其他"、描述"杂项/其他/待定"）才需要语义裁决。
    """
    if verdict.get("decision") != "approve":
        return ""
    cons = []
    for it in items or []:
        desc = str(it.get("desc", "") or "")
        cat = str(it.get("category", "") or "")
        if "其他" in cat or any(w in desc for w in ("杂项", "其他", "待定")):
            cons.append(f"{it.get('date','')} 类目[{cat}] 描述[{desc}]")
    return "；".join(cons)


def llm_arbitrate(ambiguity: str, summary: str) -> dict:
    """调 LLM 裁决一个语义模糊点。失败则保守返回 {}（不阻断，转人工）。"""
    try:
        from core.llm import LLM
        prompt = _LLM_ARBITRATE_PROMPT.format(ambiguity=ambiguity, summary=summary)
        resp = LLM().chat([{"role": "user", "content": prompt}], temperature=0.1)
        return _extract_json(resp)
    except Exception:
        return {}


def _load_items_from_files(receipt_files: list[str]) -> tuple[list[dict], list[str]]:
    """解析一组票据文件 → (审计条目, 解析错误清单)。识别必须用 LLM（OCR 文本→结构化字段）。"""
    items: list[dict] = []
    errors: list[str] = []
    for path in receipt_files:
        parsed = parse_receipt(path, use_llm=True)
        if not parsed.get("ok"):
            errors.append(f"{os.path.basename(path)}: {parsed.get('error')}")
            continue
        sub = to_audit_items(parsed["data"])
        if not sub:
            errors.append(f"{os.path.basename(path)}: 未抽出可审核条目")
            continue
        items.extend(sub)
    return items, errors


def audit_claim(items: list[dict] = None, receipt_files: list[str] = None,
                purpose: str = "", claimed_total: float = None,
                use_llm_arbitration: bool = True, engine: ReimbursementEngine = None,
                employee_id: str = "") -> dict:
    """报销审核主编排入口。入参二选一：已结构化的 items，或票据文件列表 receipt_files。

    employee_id: 可选申报人工号（透传审计落盘，不做金额判定、不落姓名）。
    返回结构化结论 + 阶段耗时 + llm_used 标志。
    """
    t0 = time.perf_counter()
    stages = {"ocr_extract": 0.0, "audit": 0.0, "arbitrate": 0.0}
    # 注入历史发票号做唯一性校验(防一票多报)。仅在显式传了 engine 时跳过，否则默认接入，
    # 保证生产链路真实生效(此前 engine 每次新建不传 seen_invoices，DUPLICATE_INVOICE 恒不触发)。
    if engine is None:
        try:
            from reimbursement.record_store import load_seen_invoices
            engine = ReimbursementEngine(seen_invoices=load_seen_invoices())
        except Exception:
            engine = ReimbursementEngine()
    parse_errors: list[str] = []
    llm_used = False

    # ---- 1. 从票据文件识别条目（LLM 只做识别，不算钱）----
    if receipt_files:
        t_s = time.perf_counter()
        items, parse_errors = _load_items_from_files(receipt_files)
        stages["ocr_extract"] = round((time.perf_counter() - t_s) * 1000, 1)
        if not items:
            return {
                "ok": False, "data": None,
                "error": "票据均未抽出可审核条目：" + "；".join(parse_errors),
            }

    if items is None:
        return {"ok": False, "data": None, "error": "未提供 items 或 receipt_files"}
    # items 为空列表：合法（无可审核条目），交给引擎产出 reject/0/0 的确定性结论

    # ---- 2. 确定性规则审核 ----
    t_s = time.perf_counter()
    verdict = engine.audit(items, employee_id=employee_id)
    stages["audit"] = round((time.perf_counter() - t_s) * 1000, 1)

    # ---- 3. LLM 兜底裁决（仅 approve + 语义模糊）----
    arbit_final = None
    if use_llm_arbitration:
        t_s = time.perf_counter()
        ambiguity = find_ambiguity(items, verdict)
        if ambiguity:
            arbit_final = llm_arbitrate(ambiguity, verdict["summary"])
            llm_used = True
            if arbit_final and not arbit_final.get("is_legitimate", True):
                verdict["decision"] = "manual_review"
                verdict["issues"].append({
                    "issue_code": "LLM_ARBITRATE",
                    "description": arbit_final.get("reason", "业务相关性存疑，转人工复核"),
                    "severity": "medium",
                })
                verdict["issue_count"] = len(verdict["issues"])
                verdict["summary"] += "；存在业务相关性存疑，转人工复核"
            elif not arbit_final:
                # LLM 兜底失败 → 显式降级标记（对齐项目二"绝不静默失败"，审计可追溯）
                verdict["issues"].append({
                    "issue_code": "LLM_FALLBACK_FAIL",
                    "description": "语义兜底调用失败，未得到裁决，建议人工复核该模糊点",
                    "severity": "medium",
                })
                # 兜底失败=无法确认业务相关性 → 结论同步降级为待人工复核。
                # 否则 summary 说"待人工复核"而 decision 仍是 approve，前端只读
                # decision 会把存疑单据直接放行（言行不一）。
                if verdict.get("decision") == "approve":
                    verdict["decision"] = "manual_review"
                verdict["issue_count"] = len(verdict["issues"])
                verdict["summary"] += "；语义兜底失败，待人工复核"
        stages["arbitrate"] = round((time.perf_counter() - t_s) * 1000, 1)

    # ---- 4. 申报总额交叉核对（确定性）----
    ct = claimed_total if claimed_total is not None else verdict["claimed_amount"]
    try:
        ct = float(ct)
    except (TypeError, ValueError):
        ct = verdict["claimed_amount"]
    total_issue = None
    if abs(round(ct, 2) - round(verdict["claimed_amount"], 2)) > 0.5:
        total_issue = {
            "issue_code": "CLAIM_TOTAL_MISMATCH",
            "description": f"申报总额 {ct} 与票据核定总额 {verdict['claimed_amount']} 不一致",
            "severity": "medium",
        }
        verdict["issues"].append(total_issue)
        verdict["issue_count"] = len(verdict["issues"])
        if verdict["decision"] == "approve":
            verdict["decision"] = "manual_review"
            verdict["summary"] += "；申报总额与核定不一致，转人工核对"

    verdict["purpose"] = purpose or ""
    verdict["claimed_input"] = ct
    verdict["parse_errors"] = parse_errors
    verdict["employee_id"] = employee_id or ""
    verdict["adjusted_by_rules"] = round(verdict["claimed_amount"] - verdict["approved_amount"] + (
        verdict["claimed_amount"] - ct if total_issue and ct > verdict["claimed_amount"] else 0.0), 2)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "ok": True,
        "data": {
            "verdict": verdict,
            "items": items,
            "llm_used": llm_used,
            "arbitration": arbit_final,
            "stages_ms": stages,
            "elapsed_ms": elapsed_ms,
        },
    }


def err_wrap(msg) -> str:
    return json.dumps({"ok": False, "data": None, "error": msg}, ensure_ascii=False)


if __name__ == "__main__":
    # 无 LLM 冒烟：直接用结构化条目跑规则引擎
    demo = [
        {"category": "住宿", "date": "2026-08-01", "amount": 620, "city": "北京", "desc": "酒店标准间"},
        {"category": "餐饮", "date": "2026-08-01", "amount": 250, "desc": "晚餐商务宴请", "itemization": ["晚餐"]},
        {"category": "交通", "date": "2026-08-01", "amount": 50, "desc": "地铁"},
        {"category": "个人消费", "date": "2026-08-02", "amount": 300, "desc": "面膜护肤品"},
    ]
    print(json.dumps(audit_claim(items=demo, use_llm_arbitration=False), ensure_ascii=False, indent=2))
