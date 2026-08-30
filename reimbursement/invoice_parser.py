# -*- coding: utf-8 -*-
"""发票/票据解析 —— OCR + 结构化字段抽取 + 确定性金额校准。

职责分工（对齐"少依赖 LLM + 不让 LLM 算钱"）：
  1. OCR：复用 rag.document_loader._ocr_image_file（PaddleOCR）拿原始文本。
  2. 字段抽取：用 LLM 把 OCR 文本转成结构化 JSON（识别票据类型/金额/日期/明细是语义任务，
     交给 LLM 合理；但这只是"识别"，不是"计算"）。
  3. 确定性校准：抽取出的金额/分项之和，必须过 reconcile（代码校验是否等于票面总额），
     金额差超容差 → 标记 manual_review，**绝不信 LLM 报的整数当作最终金额**。
"""
import os, json, re
from pathlib import Path


def _ok(data) -> str:
    return json.dumps({"ok": True, "data": data, "error": ""}, ensure_ascii=False)


def _err(msg) -> str:
    return json.dumps({"ok": False, "data": None, "error": msg}, ensure_ascii=False)


_EXTRACT_PROMPT = '''你是发票字段抽取器。从下面 OCR 文本中抽取结构化字段，只输出 JSON：

{
  "invoice_type": "增值税专用发票|增值税普通发票|出租车票|餐饮发票|航空行程单|住宿发票|收据|其他",
  "vendor": "开票方",
  "date": "YYYY-MM-DD",
  "amount": 100.00,
  "city": "若票据含城市/地址（如住宿、交通票据）则填城市名，如\"北京\"，无法确定则空串",
  "items": [{"name": "...", "amount": 100.00}],
  "invoice_number": "",
  "notes": "无法认定的信息写这里，不要编造"
}

规则：
- amount 必须是票面价税合计金额。
- items 是从该票能拆出的分项（金额之和应等于 amount）。
- 无法确定的字段给空值，严禁臆造。
- 只输出 JSON，不要多余文字。

OCR 文本：
<<<DOC_OCR>>>
'''


def ocr_from_file(file_path: str) -> str:
    """OCR 一张票据图，返回文本。复用现有 PaddleOCR 链路。"""
    try:
        from rag.document_loader import _ocr_image_file
        txt = _ocr_image_file(str(file_path))
        return txt if txt and txt.strip() else ""
    except Exception as e:
        raise RuntimeError(f"OCR失败: {e}")


def _extract_with_llm(text: str, llm) -> dict:
    """LLM 结构化抽取（仅识别，不做计算）。"""
    from core.llm import LLM
    llm = llm or LLM()
    prompt = _EXTRACT_PROMPT.replace("<<<DOC_OCR>>>", text[:4000])
    # chat() 以 messages 列表为参数（llm.chat 不接收裸字符串）
    resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
    return _parse_json(resp)


def _parse_json(text: str):
    """从 LLM 响应里抠 JSON（容忍 ``` 包裹 / 前后缀）。"""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        # 兜底：截取首个 { 到最后一个 }
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                return {}
    return {}


def reconcile(extracted: dict, tolerance: float = 1.0) -> dict:
    """确定性校验：分项金额之和是否等于票面总额。

    借鉴 expense-ai reconcile_items —— 分项加总必须对上票面总额，
    对不上 → 走人工复核，不轻信 LLM 报的整数。"""
    total = 0.0
    items = extracted.get("items") or []
    priced = [it for it in items if isinstance(it, dict) and it.get("amount") is not None]
    try:
        total = round(float(extracted.get("amount") or 0.0), 2)
    except (TypeError, ValueError):
        total = 0.0
    items_total = round(sum(float(it.get("amount") or 0.0) for it in priced), 2)
    diff = round(total - items_total, 2)
    reconciled = bool(priced) and abs(diff) <= tolerance
    return {
        "reconciled": reconciled,
        "items_total": items_total,
        "receipt_total": total,
        "difference": diff,
        "reason": "" if reconciled else f"分项和({items_total})与票面总额({total})相差 {diff}，需人工复核",
    }


def parse_receipt(file_path: str, llm=None, use_llm=True) -> dict:
    """解析单张票据 → 结构化字段 + 校准结论。"""
    if not os.path.exists(file_path):
        return {"ok": False, "error": f"票据不存在: {file_path}", "data": None}
    try:
        text = ocr_from_file(file_path)
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}
    if not text:
        return {"ok": False, "error": "OCR未识别到文字", "data": None}

    if not use_llm:
        # 纯确定性兜底：无 LLM 时只返回 OCR 文本，交规则引擎后续判定
        return {"ok": True, "data": {"ocr_text": text, "extracted": {}, "reconciled": None}}

    try:
        extracted = _extract_with_llm(text, llm)
    except Exception as e:
        return {"ok": True, "data": {"ocr_text": text, "extracted": {},
                                     "reconciled": {"reconciled": False,
                                                    "reason": f"抽取失败: {e}"}}}

    rec = reconcile(extracted)
    return {
        "ok": True,
        "data": {
            "ocr_text": text,
            "extracted": extracted,
            "reconciled": rec,
            "needs_manual": not rec["reconciled"],
        },
    }


def to_audit_items(parsed: dict) -> list[dict]:
    """把解析结果转成审计引擎 audit() 要的 items 列表。"""
    ext = (parsed or {}).get("extracted") or {}
    if not ext:
        return []
    items = []
    for it in ext.get("items") or []:
        if not isinstance(it, dict):
            continue
        items.append({
            "category": it.get("category") or _infer_category(ext.get("invoice_type", "")),
            "date": ext.get("date", ""),
            "amount": it.get("amount"),
            "desc": it.get("name", "") or ext.get("vendor", ""),
            "city": it.get("city") or ext.get("city", ""),
            "itemization": [it.get("name", "")],
            # 防骗保(可选字段)：发票号→唯一性校验，开票方→供应商黑名单。未抽到为空串则不触发。
            "invoice_no": ext.get("invoice_number", ""),
            "supplier": ext.get("vendor", ""),
        })
    if not items and ext.get("amount") is not None:
        items.append({
            "category": _infer_category(ext.get("invoice_type", "")),
            "date": ext.get("date", ""),
            "amount": ext.get("amount"),
            "desc": ext.get("vendor", "") or ext.get("invoice_type", ""),
            "city": ext.get("city", ""),
            "invoice_no": ext.get("invoice_number", ""),
            "supplier": ext.get("vendor", ""),
        })
    return items


def _infer_category(invoice_type: str) -> str:
    it = invoice_type or ""
    if any(k in it for k in ("住宿", "酒店")):
        return "住宿"
    if any(k in it for k in ("餐饮", "餐")):
        return "餐饮"
    if any(k in it for k in ("航空", "行程", "飞机")):
        return "交通-航空"
    if any(k in it for k in ("出租", "打车", "交通")):
        return "交通-出租"
    return "其他"


if __name__ == "__main__":
    # 无 LLM 冒烟：直接测 reconcile 逻辑
    demo = {"amount": 500, "items": [{"name": "房费", "amount": 300}, {"name": "早餐", "amount": 200}]}
    print(json.dumps(reconcile(demo), ensure_ascii=False))
    demo2 = {"amount": 500, "items": [{"name": "房费", "amount": 320}]}
    print(json.dumps(reconcile(demo2), ensure_ascii=False))
