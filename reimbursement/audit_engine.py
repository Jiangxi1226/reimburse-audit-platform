# -*- coding: utf-8 -*-
"""报销审核确定性引擎 —— 规则为主，LLM 只做兜底裁决。

设计原则（对齐"少依赖 LLM"）：
  金额、额度、类别合规、日期、票据有效性一律用**确定性代码**判定，
  不交给 LLM 心算；只在"该笔费用是否属业务相关"这类真正需要语义判断的点，
  才把候选结论交给 LLM 做最终裁决（见 reimbursement_tool）。

产出一条结构化审核结论（用于后续 Agent 拼接 + 落盘）：
  decision: approve | partial | reject | manual_review
  approved_amount / rejected_amount / claimed_amount
  issues: [{issue_code, description, severity, amount?}]
  policy_references: [规则条文]
"""
import json, os, re
from datetime import date, datetime


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RULES_PATH = os.path.join(_THIS_DIR, "reimbursement_rules.json")


def _load_rules(path: str = _RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _money(x) -> float:
    """任意输入转非负数；NaN/异常归 0。"""
    try:
        v = float(x)
        return v if v == v and v > 0 else 0.0  # 排除 NaN/负
    except (TypeError, ValueError):
        return 0.0


def _is_high_cost_city(city: str, rules: dict) -> bool:
    cities = rules.get("lodging", {}).get("high_cost_cities", [])
    return any(c in (city or "") for c in cities)


class ReimbursementEngine:
    """纯规则审核引擎：入参为票据结构化的条目，出结构化结论。

    入参 items 形如：
      [{"category": "住宿", "date": "2026-08-01", "amount": 620.0,
        "city": "北京", "desc": "酒店标间", "itemization": ["标准间"]}]
    """

    def __init__(self, rules: dict = None, seen_invoices: dict | None = None):
        """rules: 规则库；seen_invoices: 可选，已审核发票号集合{inv_no: rec_id}，
        用于发票唯一性校验（一票多报 → 重复报销整拒）。不传则不做该检查，保持无状态。
        """
        self.rules = rules or _load_rules()
        self.seen_invoices = seen_invoices or {}

    # ---------- 单品类额度校验 ----------
    def _check_lodging(self, item: dict) -> list[dict]:
        cap = self.rules["lodging"]
        high = _is_high_cost_city(item.get("city", ""), self.rules)
        limit = cap["cap_high_cost"] if high else cap["cap_standard"]
        amt = _money(item.get("amount"))
        issues = []
        if amt > limit:
            issues.append({
                "issue_code": "LODGING_OVER_CAP",
                "description": f"住宿 {item.get('city','')} 房价 {amt} 超上限 {limit}",
                "severity": "high",
                "amount": round(amt - limit, 2),
            })
        if not item.get("itemization"):
            issues.append({
                "issue_code": "NO_ITEMIZATION",
                "description": "住宿缺少明细（天数×单价）",
                "severity": "medium",
            })
        return issues

    def _check_meals(self, item: dict) -> list[dict]:
        lim = self.rules["meals"]["daily_limit"]
        amt = _money(item.get("amount"))
        issues = []
        if amt > lim:
            issues.append({
                "issue_code": "MEAL_OVER_DAILY",
                "description": f"餐饮单日 {amt} 超上限 {lim}",
                "severity": "high",
                "amount": round(amt - lim, 2),
            })
        # 单餐分项上限（如无分项则按整笔判定）
        brk = self.rules["meals"].get("breakdown", {})
        sub = item.get("itemization") or [item.get("desc") or ""]
        for s in sub:
            for meal_type, cap in brk.items():
                if meal_type in str(s):
                    # 粗略：若含"晚餐/早餐/午餐"字样且整笔超该餐上限
                    if amt > cap:
                        issues.append({
                            "issue_code": "MEAL_OVERRUN_SEGMENT",
                            "description": f"「{meal_type}」{amt} 超该餐上限 {cap}",
                            "severity": "low",
                            "amount": round(amt - cap, 2),
                        })
                    break
        return issues

    def _check_transport(self, item: dict) -> list[dict]:
        amt = _money(item.get("amount"))
        cat_name = item.get("category", "")
        issues = []
        if "航空" in cat_name or "飞机" in cat_name:
            threshold = self.rules["transport"]["airline_high_end_threshold"]
            if amt > threshold:
                issues.append({
                    "issue_code": "AIRLINE_HIGH_END",
                    "description": f"航空 {amt} 超高端线 {threshold}（需说明舱位）",
                    "severity": "medium",
                    # 高端舱位是"需说明"的合规提示，非自动扣减 → 不带金额，落 hint → manual_review
                    "amount": None,
                })
        if "出租" in cat_name or "打车" in cat_name:
            cap = self.rules["transport"]["taxi_daily_expense_cap"]
            if amt > cap:
                issues.append({
                    "issue_code": "TAXI_OVER_CAP",
                    "description": f"出租 {amt} 超单日上限 {cap}",
                    "severity": "medium",
                    "amount": round(amt - cap, 2),
                })
        return issues

    def _check_category_compliance(self, item: dict) -> list[dict]:
        """类别合规：禁报类别 / 红旗词。

        检查范围覆盖 category + desc + itemization（明细也要扫红旗词，
        否则"面膜/护肤品"单靠 desc 可能因含分隔符/拆词漏判——对齐红旗拆词教训）。
        """
        cat_name = item.get("category", "")
        desc = str(item.get("desc", "")) + " " + " ".join(item.get("itemization") or [])
        issues = []
        for bad in self.rules["non_reimbursable_categories"]:
            if bad in cat_name or bad in desc:
                issues.append({
                    "issue_code": "NON_REIMBURSABLE",
                    "description": f"命中禁报类别「{bad}」",
                    "severity": "high",
                    "amount": _money(item.get("amount")),
                })
        rfs = self.rules.get("red_flags", {})
        for word in rfs.get("clear_personal", []):
            if word in cat_name or word in desc:
                issues.append({
                    "issue_code": "PERSONAL_ITEM",
                    "description": f"疑似个人消费（{word}）",
                    "severity": "high",
                    "amount": _money(item.get("amount")),
                })
        for word in rfs.get("generic_itemization", []):
            if word in desc:
                issues.append({
                    "issue_code": "VAGUE_ITEMIZATION",
                    "description": f"明细过于笼统（{word}）",
                    "severity": "medium",
                })
        return issues

    def _check_supplier_blacklist(self, item: dict) -> list[dict]:
        """供应商黑名单：命中即整拒（对齐红旗"硬决定、不靠 LLM"）。
        条目未带 supplier 字段则跳过（可选字段，不入侵现有调用）。
        """
        supplier = str(item.get("supplier", "") or "").strip()
        if not supplier:
            return []
        issues = []
        for bad in self.rules.get("supplier_blacklist", []):
            if bad and bad in supplier:
                issues.append({
                    "issue_code": "BLACKLISTED_SUPPLIER",
                    "description": f"命中供应商黑名单「{bad}」",
                    "severity": "high",
                    "amount": _money(item.get("amount")),
                })
        return issues

    def _check_invoice_uniqueness(self, item: dict) -> list[dict]:
        """发票唯一性：同一发票号已被审核过 → 一票多报，整拒。
        条目未带 invoice_no 或未启用该规则则跳过；seen_invoices 为空时不检查（无状态）。
        """
        inv = str(item.get("invoice_no", "") or "").strip()
        if not inv:
            return []
        cfg = self.rules.get("invoice_uniqueness", {}) or {}
        if not cfg.get("enabled", True):
            return []
        if not self.seen_invoices:
            return []
        if inv in self.seen_invoices:
            return [{
                "issue_code": "DUPLICATE_INVOICE",
                "description": f"发票号 {inv} 已审核过（{self.seen_invoices[inv]}），疑似重复报销",
                "severity": cfg.get("dup_severity", "high"),
                "amount": _money(item.get("amount")),
            }]
        return []

    def _expiry_check(self, item: dict) -> list[dict]:
        win = self.rules.get("submission_window_days", 30)
        ds = item.get("date")
        issues = []
        try:
            d = datetime.strptime(str(ds)[:10], "%Y-%m-%d").date()
            if (date.today() - d).days > win:
                issues.append({
                    "issue_code": "PAST_WINDOW",
                    "description": f"票据日期 {d} 已超报销窗口 {win} 天",
                    "severity": "medium",
                })
        except (TypeError, ValueError):
            issues.append({
                "issue_code": "BAD_DATE",
                "description": f"票据日期无法解析（{ds}）",
                "severity": "medium",
            })
        return issues

    # ---------- 主入口 ----------
    def _adjudicate_item(self, it: dict) -> tuple[dict, list[dict]]:
        """对**单条**票据做逐条判定，贴近真实财务审核（不整单一刀切）。

        返回 (item_verdict, item_issues)。item_verdict = reject | partial | manual_review | approve。
        """
        cat = it.get("category", "")
        issues = []
        # 各类别专项额度校验
        if "住宿" in cat:
            issues += self._check_lodging(it)
        elif "餐" in cat or "饮" in cat:
            issues += self._check_meals(it)
        elif any(k in cat for k in ("交通", "航空", "出租", "车", "打车")):
            issues += self._check_transport(it)
        # 通用类别合规 + 日期
        issues += self._check_category_compliance(it)
        issues += self._expiry_check(it)
        # 防骗保硬校验：供应商黑名单 / 发票唯一性（可选字段，不带不查）
        issues += self._check_supplier_blacklist(it)
        issues += self._check_invoice_uniqueness(it)

        amt = _money(it.get("amount"))
        # 硬违规：禁报类别/个人消费/黑名单供应商/重复发票 → 该笔整拒
        hard = [i for i in issues if i["issue_code"] in (
            "NON_REIMBURSABLE", "PERSONAL_ITEM", "BLACKLISTED_SUPPLIER", "DUPLICATE_INVOICE")]
        # 超上限：medium/high 级别带金额的扣减（业务相关但超额）
        excess = sum(i.get("amount", 0.0) or 0.0 for i in issues
                     if i.get("amount") is not None and i["severity"] in ("medium", "high"))
        # 提示类：medium/high 但**不带金额**（缺明细/超期/日期异常/明细笼统）→ 需人工复核。
        # 注意不能用 get(key, 0.0) 判空 —— 缺键会落 0.0，导致提示被漏判成合规。
        # 用 get(key)（缺键返回 None）区分"提示（无金额）"与"超限（有金额）"。
        hint = [i for i in issues
                if i["severity"] in ("medium", "high") and i.get("amount") is None]
        # 纯 low 级软提示（如单餐超软限）→ 记录但不强制人工，避免误判 noise
        if hard:
            verdict = "reject"
            approved = 0.0
            # 硬违规只拒该笔金额，不牵连同单其他合规条目
        elif excess > 0:
            verdict = "partial"
            approved = round(max(0.0, amt - excess), 2)
        elif hint:  # 需人工核实（缺明细/超期）→ 按上限批、记问题
            verdict = "manual_review"
            approved = round(amt, 2)
        else:
            verdict, approved = "approve", round(amt, 2)

        item_verdict = {
            "desc": it.get("desc", ""),
            "category": cat,
            "date": it.get("date", ""),
            "amount": round(amt, 2),
            "approved": approved,
            "decision": verdict,
            # 可选审计字段：带则透传（用于落盘/核验），不带为 None
            "invoice_no": it.get("invoice_no") or "",
            "supplier": it.get("supplier") or "",
        }
        # 逐条只保留最大的核减依据（降低输出噪声，但全量 issues 仍返给 Agent 级）
        return item_verdict, issues

    def audit(self, items: list[dict], employee_id: str = "") -> dict:
        """对一组票据条目做**逐条**审核，再按最少违规原则汇总。

        对齐真实报销审核（不整单一刀切）：
          - 某笔含禁报/个人消费 → 仅拒该笔，同单其他合规条目标注后照批。
          - 某笔超上限 → 封顶核准（超额部分拒）。
          - 仅提示 → 记问题、按上限批，建议人工复核。
          - 汇总决策：有硬违规单 → reject（并注明"仅拒违规条目"）；有超额 → partial；
            仅提示 → manual_review；全合规 → approve。
        与 LLM 兜底（find_ambiguity）区分：确定性判定的不交给 LLM。

        employee_id: 可选，申报人工号（仅落盘/审计用，不做金额判定，也不落姓名）。
        """
        claimed = 0.0
        approved_sum = 0.0
        all_issues: list[dict] = []
        item_verdicts: list[dict] = []
        has_hard = False
        has_over = False
        has_hint = False

        for it in items:
            claimed += _money(it.get("amount"))
            iv, issues = self._adjudicate_item(it)
            item_verdicts.append(iv)
            approved_sum += iv["approved"]
            all_issues += issues
            if iv["decision"] == "reject":
                has_hard = True
            elif iv["decision"] == "partial":
                has_over = True
            elif iv["decision"] == "manual_review":
                has_hint = True

        if not items:
            decision, approved = "reject", 0.0
        elif has_hard:
            decision = "reject"
        elif has_over:
            decision = "partial"
        elif has_hint:
            decision = "manual_review"
        else:
            decision = "approve"
        approved = round(approved_sum, 2)

        # 汇总文案（逐条口径）
        if not items:
            summary = "无可审核条目。"
        elif decision == "reject":
            n = len([v for v in item_verdicts if v["decision"] == "reject"])
            kept = len([v for v in item_verdicts if v["approved"] > 0])
            hard_codes = {i["issue_code"] for i in all_issues}
            reason = "涉及禁报/个人消费"
            if "BLACKLISTED_SUPPLIER" in hard_codes:
                reason = "供应商命中黑名单"
            elif "DUPLICATE_INVOICE" in hard_codes:
                reason = "发票号重复报销"
            summary = (f"{n} 笔{reason}，不予报销；"
                       + (f"其余 {kept} 笔合规条目已按规则核准。" if kept else "无合规条目可批。"))
        elif decision == "partial":
            summary = "存在超上限项，超额部分不予报销，按上限核准；其余合规条目照批。"
        elif decision == "manual_review":
            summary = "存在需人工核实的提示项（如缺明细/超期），建议人工复核后再批。"
        else:
            summary = "各票据均符合规则，可全额批准。"

        return {
            "claimed_amount": round(claimed, 2),
            "approved_amount": approved,
            "rejected_amount": round(claimed - approved, 2),
            "decision": decision,
            "issues": all_issues,
            "issue_count": len(all_issues),
            "items_adjudication": item_verdicts,
            "policy_references": sorted({i["issue_code"] for i in all_issues}),
            "summary": summary,
            "employee_id": employee_id or "",
        }

    def policy_lookup(self, category: str = None, city: str = None) -> dict:
        """供 Agent 查询规则（确定性，非 LLM）。返回可读规则。"""
        rules = self.rules
        out = {
            "currency": rules.get("currency"),
            "submission_window_days": rules.get("submission_window_days"),
            "non_reimbursable_categories": rules.get("non_reimbursable_categories"),
        }
        if category and any(c in category for c in ("住宿", "酒店", "房")):
            l = rules["lodging"]
            out["lodging"] = {
                **l,
                "effective_cap": (l["cap_high_cost"] if _is_high_cost_city(city, rules)
                                  else l["cap_standard"]),
            }
        if category and any(c in category for c in ("餐", "饮", "饭")):
            out["meals"] = rules["meals"]
        if category and any(c in category for c in ("交通", "车")):
            out["transport"] = rules["transport"]
        return out


if __name__ == "__main__":
    e = ReimbursementEngine()
    demo = [
        {"category": "住宿", "date": "2026-08-01", "amount": 620, "city": "北京", "desc": "酒店标准间"},
        {"category": "餐饮", "date": "2026-08-01", "amount": 250, "desc": "晚餐商务宴请", "itemization": ["晚餐"]},
        {"category": "交通", "date": "2026-08-01", "amount": 50, "desc": "地铁"},
        {"category": "个人消费", "date": "2026-08-02", "amount": 300, "desc": "面膜护肤品"},
    ]
    r = e.audit(demo)
    print(json.dumps(r, ensure_ascii=False, indent=2))
