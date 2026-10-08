# -*- coding: utf-8 -*-
"""审核记录持久化 —— 让"上传→审核→结果"不再关页即失，对齐真实公司报销系统。

能力：
  - 每次 /v1/reimburse 审核完，把结构化结论(verdict/逐条核定/规则命中/事由)落盘。
  - 支持历史检索：时间倒序、按 决策/关键词/日期 过滤；返回简洁列表项。
  - 支持对单笔的确定性问答：读该条记录的核定结论，用规则直接回答，不靠 LLM 瞎编。
  - 上限裁剪：只保留最近 MAX_RECORDS 条，防日志无限膨胀(对齐项目二 audit 的保留策略)。

线程安全：文件写操作用锁串行，避免并发审核互相覆盖。JSON 单文件读写足够当前量级。
"""
import json, os, threading, time, re

_RECORDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "audit_records.json")

MAX_RECORDS = 2000          # 保留最近 2000 条
_lock = threading.Lock()    # 串行化写盘


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _trim(recs: list[dict]) -> list[dict]:
    """只保留最近 MAX_RECORDS 条(新在后)。"""
    if len(recs) > MAX_RECORDS:
        return recs[-MAX_RECORDS:]
    return recs


def _load() -> list[dict]:
    try:
        with open(_RECORDS_PATH, encoding="utf-8") as f:
            recs = json.load(f)
            return recs if isinstance(recs, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_seen_invoices() -> dict[str, str]:
    """收集历史记录里已审核过的发票号 → {invoice_no: rec_id}，供引擎做发票唯一性校验。

    仅取带 invoice_no 的记录；空库/无发票号记录返回空 dict（引擎则跳过该检查，保持无状态）。
    """
    seen: dict[str, str] = {}
    for r in _load():
        inv = str(r.get("invoice_no", "") or "").strip()
        if inv:
            seen.setdefault(inv, r.get("id", ""))
    return seen


def _save(recs: list[dict]) -> None:
    try:
        os.makedirs(os.path.dirname(_RECORDS_PATH), exist_ok=True)
        with open(_RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
    except Exception:
        pass  # 落盘失败不阻断审核主流程


# ---------------- 写入 ----------------

def save_record(inputs: dict, verdict: dict, llm_used: bool,
                server_elapsed_ms: float) -> dict:
    """落盘一次审核，返回记录 dict(含 id)。inputs: {purpose, claimed_input, items, parse_errors}"""
    rec_id = f"R{int(time.time() * 1000)}{os.getpid()}"
    # 顶层发票号/供应商从 items 聚合（逐次取首个非空），供 load_seen_invoices 发票去重；
    # 员工工号只存 ID 不落姓名（规避个信风险）。
    _inv, _sup = "", ""
    for it in (inputs.get("items", []) or []):
        if not _inv:
            _inv = str(it.get("invoice_no", "") or "").strip()
        if not _sup:
            _sup = str(it.get("supplier", "") or "").strip()
        if _inv and _sup:
            break
    record = {
        "id": rec_id,
        "ts": _now(),
        "purpose": inputs.get("purpose", "") or "",
        "claimed_input": inputs.get("claimed_input"),
        "decision": verdict.get("decision", ""),
        "claimed_amount": verdict.get("claimed_amount", 0.0),
        "approved_amount": verdict.get("approved_amount", 0.0),
        "rejected_amount": verdict.get("rejected_amount", 0.0),
        "summary": verdict.get("summary", ""),
        "items": inputs.get("items", []),                      # 审核入参条目
        "items_adjudication": verdict.get("items_adjudication", []),  # 逐条核定
        "issues": verdict.get("issues", []),                      # 命中的规则/问题
        "issue_count": verdict.get("issue_count", 0),
        "policy_refs": verdict.get("policy_references", []),     # 命中的规则编码
        "llm_used": bool(llm_used),
        "server_elapsed_ms": round(server_elapsed_ms, 1),
        "parse_errors": inputs.get("parse_errors", []),
        "employee_id": inputs.get("employee_id", "") or "",
        "invoice_no": _inv or inputs.get("invoice_no", "") or "",
        "supplier": _sup or inputs.get("supplier", "") or "",
    }
    with _lock:
        recs = _load()
        recs.append(record)
        _save(_trim(recs))
    return record


# ---------------- 人工复核回流（闭环） ----------------
# "规则建议 → 人工终判"的反馈：没有它，manual_review 就是个黑洞——规则说"待人工
# 复核"，之后无人知晓到底批没批，规则也无从校准。回填后即可统计一致性、定位
# 哪条规则总被人工推翻。

# 负面程度：越大越"拒"。用于判定规则是偏严还是偏松。
_SEVERITY = {"approve": 0, "partial": 1, "manual_review": 2, "reject": 3}


def _agreement(rule_decision: str, final_decision: str) -> str:
    """规则建议 vs 人工终判的关系：agree / rule_strict(规则偏严) / rule_lenient(规则偏松)。

    按负面程度比较：终判比规则建议更正面 → 规则偏严（动不动就拒）；
    更负面 → 规则偏松（放得太松）。
    """
    a = _SEVERITY.get(rule_decision, 1)
    b = _SEVERITY.get(final_decision, 1)
    if b == a:
        return "agree"
    return "rule_strict" if b < a else "rule_lenient"


def apply_review(rec_id: str, final_decision: str, reviewer: str = "",
                 note: str = "") -> dict:
    """回填某条记录的**人工终判**，形成"规则建议 → 人工终判"的闭环。

    final_decision: approve | partial | reject | manual_review
    返回 {ok, record} 或 {ok: False, error}。
    """
    fd = (final_decision or "").strip()
    if fd not in _SEVERITY:
        return {"ok": False,
                "error": f"非法终判：{final_decision}（可选 approve/partial/reject/manual_review）"}
    with _lock:
        recs = _load()
        for r in recs:
            if r.get("id") == rec_id:
                r["review"] = {
                    "final_decision": fd,
                    "reviewer": (reviewer or "").strip(),
                    "note": (note or "").strip(),
                    "ts": _now(),
                    "agreement": _agreement(str(r.get("decision", "")), fd),
                }
                _save(recs)
                return {"ok": True, "record": r}
    return {"ok": False, "error": f"记录不存在：{rec_id}"}


def review_stats() -> dict:
    """复核统计：一致率、规则偏严/偏松分布、被推翻最多的规则编码。

    回答"规则判错了怎么发现"：一致率低、或某规则命中后总被推翻 → 该规则需校准。
    """
    recs = _load()
    reviewed = [r for r in recs if r.get("review")]
    pending = [r for r in recs
               if r.get("decision") == "manual_review" and not r.get("review")]
    total = len(reviewed)
    agree = sum(1 for r in reviewed if r["review"].get("agreement") == "agree")
    strict = sum(1 for r in reviewed if r["review"].get("agreement") == "rule_strict")
    lenient = sum(1 for r in reviewed if r["review"].get("agreement") == "rule_lenient")
    # 被推翻最多的规则编码：不一致记录中命中的 policy_refs 计数
    by_policy: dict[str, int] = {}
    for r in reviewed:
        if r["review"].get("agreement") == "agree":
            continue
        for code in (r.get("policy_refs") or []):
            by_policy[code] = by_policy.get(code, 0) + 1
    return {
        "total_records": len(recs),
        "reviewed": total,
        "pending_review": len(pending),          # 待人工复核但尚未回填
        "agree": agree,
        "rule_strict": strict,                   # 规则偏严（人工判得更宽）
        "rule_lenient": lenient,                 # 规则偏松（人工判得更严）
        "agreement_rate": (agree / total * 100) if total else None,
        "top_overridden_policies": sorted(by_policy.items(), key=lambda kv: -kv[1])[:5],
    }


# ---------------- 检索 ----------------

def list_records(keyword: str = "", decision: str = "", limit: int = 50,
                 employee_id: str = "", invoice_no: str = "") -> list[dict]:
    """历史列表(新在前)。keyword 匹配 事由/摘要/条目描述；decision 四档过滤。
    employee_id/invoice_no 可选精确过滤（支持"查某工号的报销"）。"""
    recs = _load()
    kw = (keyword or "").strip()
    dec = (decision or "").strip()
    emp = (employee_id or "").strip()
    inv = (invoice_no or "").strip()
    out = []
    for r in reversed(recs):  # 新在前
        if dec and r.get("decision") != dec:
            continue
        if emp and str(r.get("employee_id", "") or "").strip() != emp:
            continue
        if inv and str(r.get("invoice_no", "") or "").strip() != inv:
            continue
        if kw:
            blob = " ".join([
                str(r.get("purpose", "")), str(r.get("summary", "")),
                " ".join(str(i.get("desc", "")) for i in r.get("items", [])),
            ])
            if kw not in blob:
                continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def get_record(rec_id: str) -> dict | None:
    for r in _load():
        if r.get("id") == rec_id:
            return r
    return None


# ---------------- 确定性问答 ----------------

def answer_question(rec: dict, question: str) -> dict:
    """对**单条**审核记录做确定性问答。答案全部由记录与规则推出，0 LLM、可验证。

    返回 {ok, answer}。覆盖常见问题；未识别关键词返回引导文案(列出记录里有什么)。
    """
    q = (question or "").strip()
    # 归一化：转小写、去空白，避免「LLM 吗 / LLM吗 / 用大模型」等写法差异导致漏匹配
    qn = q.lower().replace(" ", "").replace("　", "")

    def _has(*kws) -> bool:
        return any(k in qn for k in kws)
    decision = rec.get("decision", "")
    amt = rec.get("claimed_amount", 0.0)
    appr = rec.get("approved_amount", 0.0)
    rej = rec.get("rejected_amount", 0.0)
    issues = rec.get("issues", [])
    items = rec.get("items_adjudication", [])
    hard_codes = {"NON_REIMBURSABLE", "PERSONAL_ITEM"}

    # —— 结论 ——
    if _has("什么结论", "结果怎么", "能报吗", "批了吗", "通过吗", "能不能报销", "结论"):
        d = {"approve": "通过，全部可报销。", "partial": "部分通过，超额部分已核减。",
             "reject": "不通过，含禁报/个人消费条目。", "manual_review": "需人工复核后决定。"}
        return _ans(f"该单结论为「{decision}」：{d.get(decision,'')} {rec.get('summary','')}",
                    rec, ["结论"])

    # —— 金额 ——
    if _has("多少钱", "金额", "报销多少", "能报多少", "批多少", "总共有多少"):
        return _ans(
            f"申报总额 ¥{amt:.2f}，核准 ¥{appr:.2f}，拒付 ¥{rej:.2f}。", rec, ["金额"])

    # —— 为什么拒 ——
    if _has("为什么拒", "为啥拒", "拒的理由", "拒付原因", "哪笔拒", "为何不能", "被谁拒", "拒了",
            "这笔拒", "这单拒", "这笔被拒", "这单被拒", "被拒", "不能报", "不予报销", "不报销", "冤"):
        hard = [i for i in items if i.get("decision") == "reject"]
        if hard:
            descs = "；".join(f"{i.get('desc','')}({i.get('category','')} ¥{i.get('amount',0):.0f})" for i in hard)
            return _ans(f"共 {len(hard)} 笔被拒：{descs}。命中禁报/个人消费规则，按规则整拒该笔。",
                        rec, [i.get("issue_code","") for i in issues if i.get("issue_code") in hard_codes])
        return _ans(f"该单无逐笔拒付条目，整体结论 {decision}。核减发生在超额/提示类。", rec, ["拒付"])

    # —— 超了多少钱 ——
    if _has("超了", "超多少", "多报", "扣多少", "超限", "超额", "超支", "亏"):
        if rej > 0:
            over = [i for i in issues if (i.get("amount") or 0) > 0]
            main = max(over, key=lambda x: x.get("amount", 0), default=None)
            txt = f"核减 ¥{rej:.2f}。"
            if main:
                txt += f"最大一笔：{main.get('issue_code','')} {main.get('description','')}（核减 ¥{main.get('amount',0):.2f}）。"
            return _ans(txt, rec, [i.get("issue_code","") for i in issues])
        return _ans("无超额核减，全部按规则核准。", rec, ["超额"])

    # —— 哪条规则 ——
    if _has("哪条规则", "什么规则", "命中规则", "违反", "几项问题", "什么问题", "哪些问题"):
        if issues:
            codes = "；".join(f"{i.get('issue_code','')}（{i.get('severity','')}）{i.get('description','')}" for i in issues)
            return _ans(f"该单命中 {len(issues)} 项：{codes}", rec, [i.get("issue_code","") for i in issues])
        return _ans("未命中任何规则问题，全部合规。", rec, ["规则"])

    # —— 用了 LLM 吗 ——
    if _has("用llm", "用大模型", "llm吗", "调用llm", "大模型吗", "走llm", "用模型", "llm", "大模型"):
        return _ans(f"{'调用了 LLM 兜底(存在语义模糊点)' if rec.get('llm_used') else '未调用 LLM，全规则引擎判定'}。",
                    rec, ["LLM依赖"])

    # —— 未命中关键词：引导 ——
    return _ans(
        ("我能回答该单的：结论/金额/拒付原因/超额多少/命中规则/是否走LLM。"
         "例如问「为什么拒」「超了多少钱」。") + f"（{rec.get('id','')} 共 {len(items)} 笔、命中 {rec.get('issue_count',0)} 项规则）",
        rec, ["帮助"])


def _ans(text: str, rec: dict, intent: list[str]) -> dict:
    """统一 回答拼接 + 附来源(记录id/决策/逐笔)，让答案可核验。"""
    detail = "；".join(
        f"{i.get('desc','')}→{i.get('decision','')} ¥{i.get('approved',0):.0f}" for i in rec.get("items_adjudication", [])[:5])
    return {"ok": True, "answer": text, "intent": intent, "record_id": rec.get("id", ""),
            "decision": rec.get("decision", ""),
            "llm_used": bool(rec.get("llm_used", False)),
            "items_brief": detail}


# ================= 全系统问答(跨全部记录：统计 / 查询 / 计算) =================
# 定位：像真实公司里「财务系统」的提问——不是追单笔，而是对整库做聚合。
# 全部结论由落盘记录 + 汇总公式推出，0 LLM、可验证(附明细)。规则与 answer_question 同源。

DECISION_LABEL = {"approve": "通过", "partial": "部分", "reject": "拒报", "manual_review": "人工"}


def _aggregate(recs: list[dict]) -> dict:
    """对一组记录做聚合统计(总额/批准/拒付/笔数/决策分布/类别核减/月份分布)。"""
    agg = {
        "n": len(recs),
        "claimed": 0.0, "approved": 0.0, "rejected": 0.0,
        "by_decision": {}, "by_category": {}, "by_month": {},
        "llm_used_n": 0,
    }
    for r in recs:
        agg["claimed"] += r.get("claimed_amount", 0.0) or 0.0
        agg["approved"] += r.get("approved_amount", 0.0) or 0.0
        agg["rejected"] += r.get("rejected_amount", 0.0) or 0.0
        d = r.get("decision", "")
        agg["by_decision"][d] = agg["by_decision"].get(d, 0) + 1
        if r.get("llm_used"):
            agg["llm_used_n"] += 1
        m = (r.get("ts", "") or "")[:7]  # YYYY-MM
        if m:
            agg["by_month"][m] = agg["by_month"].get(m, 0.0) + (r.get("claimed_amount", 0.0) or 0.0)
        # 类别核减：从逐条核定的 rejected 金额里按 category 汇总
        for it in r.get("items_adjudication", []):
            cat = it.get("category", "") or "未知"
            rej = (it.get("amount", 0.0) or 0.0) - (it.get("approved", 0.0) or 0.0)
            agg["by_category"].setdefault(cat, {"n": 0, "rejected": 0.0, "rejected_n": 0})
            agg["by_category"][cat]["n"] += 1
            agg["by_category"][cat]["rejected"] += max(0.0, rej)
            if max(0.0, rej) > 0:
                agg["by_category"][cat]["rejected_n"] += 1
    return agg


def _fmt_money(v: float) -> str:
    return f"¥{v:,.2f}"


def _norm_meta(q: str) -> str:
    return q.lower().replace(" ", "").replace("　", "").replace("，", "").replace("。", "")


def _month_q(qn: str) -> str | None:
    """从问句里提取形如 '2026-09' / '9月' / '09月' 的月份，返回 'YYYY-MM' 或 None。"""
    import re as _re
    m = _re.search(r"(20\d{2})[-年/](0?[1-9]|1[0-2])", qn)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = _re.search(r"(?<!\d)(0?[1-9]|1[0-2])月", qn)
    if m:
        # 无年份默认用最近出现的月份年份
        month = int(m.group(1))
        best = None
        for k in sorted((r.get("ts", "") for r in _load()), reverse=True):
            try:
                if int(k[5:7]) == month:
                    best = k[:7]
                    break
            except Exception:
                continue
        if best:
            return best
        return f"-{month:02d}"
    return None


def _extract_entities(qn: str) -> dict:
    """从归一化问句里抽取实体(决策/月份/类别/事由关键词/金额下限)，供通用查询器用。"""
    ent = {"decision": None, "month": None, "category": None, "keyword": None, "money_gte": None}

    # 决策
    if any(k in qn for k in ["被拒", "不予报", "拒绝", "拒报", "驳回", "不给报", "不报销", "拒了"]):
        ent["decision"] = "reject"
    elif any(k in qn for k in ["通过", "批准", "批了", "准予", "合规", "能报"]):
        ent["decision"] = "approve"
    elif any(k in qn for k in ["部分", "超额", "超标", "封顶", "核减", "超上限"]):
        ent["decision"] = "partial"
    elif any(k in qn for k in ["待人工", "人工复核", "转人工", "待审", "需人工"]):
        ent["decision"] = "manual_review"

    ent["month"] = _month_q(qn)

    for c in ["住宿", "餐饮", "机票", "航空", "出租", "交通", "礼品", "个人消费", "娱乐", "罚款", "其他"]:
        if c in qn:
            ent["category"] = c
            break

    # 金额下限：如 "500以上 / 超过500 / 大于500 / 到500"（带数字）
    m = re.search(r"(超过|大于|高于|以上|满|≥|>)(\d+(?:\.\d+)?)", qn)
    if m:
        ent["money_gte"] = float(m.group(2))
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:以上|起)", qn)
        if m:
            ent["money_gte"] = float(m.group(1))

    # 事由关键词：仅在无其他实体时才识别，避免把查询词/指令词误当事由
    if not (ent["decision"] or ent["month"] or ent["category"] or ent.get("money_gte") is not None):
        stop = set("多少几笔总量总额合计汇总共有几哪些列出所有全部都按为的嘛呢吗啊呀各其这那这些那些哪个分别"
                   "大于超过以上一下到元块钱万报销审核记录单子如何什么怎样怎么被拒通过部分人工待审月份类别分类最近"
                   "查看看找跟我告诉还有以及可能帮")
        cand = []
        for seg in re.findall(r"[一-龥]{2,8}", qn):
            if seg in stop or all(ch in stop for ch in seg):
                continue
            if any(x in seg for x in ["怎么", "如何", "什么", "多少", "几笔", "哪些", "有几", "总额",
                                      "审核", "报销", "记录", "单笔", "每笔", "每次", "每单",
                                      "看到", "能看", "显示", "展示", "查询", "有多少", "报多少"]):
                continue
            cand.append(seg)
        if cand:
            ent["keyword"] = max(cand, key=len)

    return ent


def _has_entities(ent: dict) -> bool:
    return any(ent.get(k) for k in ("decision", "month", "category", "keyword", "money_gte"))


_DEC_LABEL_CN = {"reject": "拒报", "approve": "通过", "partial": "部分核准", "manual_review": "待人工复核"}


def _ent_label(ent: dict) -> str:
    """把实体拼成一句筛选描述，如 '9月 · 拒报 · 住宿 · 含“差旅” · ≥500元'。"""
    parts = []
    if ent.get("month"):
        parts.append(ent["month"])
    if ent.get("decision"):
        parts.append(_DEC_LABEL_CN.get(ent["decision"], ent["decision"]))
    if ent.get("category"):
        parts.append(ent["category"])
    if ent.get("keyword"):
        parts.append(f"含「{ent['keyword']}」")
    if ent.get("money_gte") is not None:
        parts.append(f"≥{ent['money_gte']:g}元")
    return " · ".join(parts) if parts else "当前筛选"


def _query_by_entities(ent: dict, recs: list[dict]) -> list[dict]:
    """按实体筛选记录。配合多轮承接：缺失维度由 context 补齐。"""
    out = []
    for r in recs:
        if ent.get("decision") and r.get("decision") != ent["decision"]:
            continue
        if ent.get("month") and not (r.get("ts", "") or "").startswith(ent["month"]):
            continue
        blob = str(r.get("purpose", "")) + str(r.get("summary", ""))
        items_blob = " ".join(str(i.get("desc", "")) + str(i.get("category", "")) for i in r.get("items", []))
        if ent.get("category") and ent["category"] not in blob and ent["category"] not in items_blob:
            continue
        if ent.get("keyword") and ent["keyword"] not in blob and ent["keyword"] not in items_blob:
            continue
        if ent.get("money_gte") is not None and (r.get("claimed_amount", 0) or 0) < ent["money_gte"]:
            continue
        out.append(r)
    return out


# 承接词：上一轮已在问某实体，本轮用指示词接续（如 "那几笔呢""哪些""分别""再说"）
_FOLLOW_UP = set("这这些这些笔那那些那几笔哪几笔它们它都该再来具体再再说几笔呢有哪些分别多少另外其剩下的其余别")


def _is_followup(qn: str, ent: dict) -> bool:
    return not _has_entities(ent) and any(k in qn for k in _FOLLOW_UP)


def _merge_entities(prev: dict, cur: dict) -> dict:
    """本轮实体优先，缺失维度沿用上一轮(多轮承接)。"""
    merged = dict(prev or {})
    for k, v in (cur or {}).items():
        if v:
            merged[k] = v
    return merged


def answer_system(question: str, context: dict | None = None) -> dict:
    """对**整库**做确定性聚合问答：统计/查询/计算。0 LLM、可验证。

    返回 {ok, answer, intent, records, stats, context}。`context` 为上一轮实体，可多轮承接；
    返回的 context 存回前端供下一轮使用。
    """
    q = (question or "").strip()
    qn = _norm_meta(q)
    if not qn:
        return {"ok": True, "answer": "请输入要查询的问题，例如「累计核准多少钱」「有多少笔被拒」「9月的报销总额」。",
                "intent": ["帮助"], "records": [], "stats": {}, "context": context or {}}

    recs = _load()  # 新在后
    all_agg = _aggregate(recs)

    def _has(*kws) -> bool:
        return any(k in qn for k in kws)

    def _dec(dist: dict) -> str:
        return "、".join(f"{DECISION_LABEL.get(k,k)} {v}笔" for k, v in sorted(dist.items())) or "无记录"

    def _pick(rs: list[dict], limit: int = 5) -> list[dict]:
        return list(reversed(rs))[:limit]  # 新的在前，最多取若干条

    def _brief(rs: list[dict]) -> str:
        if not rs:
            return "（无匹配记录）"
        lines = [f"- {r.get('ts','')[:10]} {r.get('purpose','') or '(未填事由)'}："
                 f"{DECISION_LABEL.get(r.get('decision',''),r.get('decision',''))} "
                 f"申报{_fmt_money(r.get('claimed_amount',0) or 0)}→核准{_fmt_money(r.get('approved_amount',0) or 0)}"
                 for r in _pick(rs)]
        return "\n".join(lines)

    # ==== 实体路径(支持多轮承接 + 任意筛选组合) ====
    cur_ent = _extract_entities(qn)
    ent = _merge_entities(context or {}, cur_ent) if (_is_followup(qn, cur_ent) or _has_entities(cur_ent)) else cur_ent
    if _has_entities(ent):
        asking = "哪些" in qn or "列出" in qn or "都是" in qn or "分别" in qn or "看看" in qn or "有哪" in qn
        want_stats = any(k in qn for k in ["多少", "几笔", "总额", "合计", "汇总", "总共", "几单", "总额是", "是多少"])
        sub = _query_by_entities(ent, recs)
        a = _aggregate(sub)
        if not sub:
            return {"ok": True,
                    "answer": f"没有符合{_ent_label(ent)}的记录。",
                    "intent": ["实体查询"], "records": [], "stats": {"n": 0},
                    "context": ent}
        lines = []
        if want_stats or not asking:
            lines.append(f"{_ent_label(ent)}共 {a['n']} 笔：申报{_fmt_money(a['claimed'])}、" \
                         f"核准{_fmt_money(a['approved'])}、拒付{_fmt_money(a['rejected'])}。")
        if asking or not want_stats:
            lines.append(_brief(sub))
        return {"ok": True, "answer": "\n".join(lines),
                "intent": ["实体查询"] + (["列表"] if asking else ["统计"]),
                "records": _pick(sub), "stats": a, "context": ent}

    # ==== 简单问题：沿用确定性 if 链(整库统计/类别/决策分布/总额/列表) ====

    # ---- 1. 月份/时间段 查询与统计 ----
    mq = _month_q(qn)
    if mq:
        sub = [r for r in recs if (r.get("ts", "") or "").startswith(mq)]
        if sub:
            a = _aggregate(sub)
            head = f"{mq} 共 {a['n']} 笔：申报{_fmt_money(a['claimed'])}、" \
                   f"核准{_fmt_money(a['approved'])}、拒付{_fmt_money(a['rejected'])}。"
            body = _brief(sub)
            return {"ok": True, "answer": head + "\n" + body,
                    "intent": ["月份统计"], "records": _pick(sub), "stats": a}

    # ---- 2. 类别维度 统计/核减 ----
    if _has("类别", "分类", "各类", "按类", "哪种", "哪些类", "核减最多"):
        cats = sorted(all_agg["by_category"].items(), key=lambda kv: kv[1]["rejected"], reverse=True)
        head = f"全库 {len(recs)} 笔，类别核减排序：\n"
        if not cats:
            return {"ok": True, "answer": head + "（暂无可统计的条目）", "intent": ["类别统计"],
                    "records": [], "stats": all_agg}
        lines = [f"- {c}：{v['n']}笔，其中核减 {v['rejected_n']}笔，累计核减{_fmt_money(v['rejected'])}"
                 for c, v in cats]
        return {"ok": True, "answer": head + "\n".join(lines), "intent": ["类别统计"],
                "records": [], "stats": all_agg}

    # ---- 3. 待人工复核 列表 ----
    if _has("待人工", "人工复核", "转人工", "需人工", "人工确认", "人工"):
        sub = [r for r in recs if r.get("decision") == "manual_review"]
        n = len(sub)
        if not sub:
            return {"ok": True, "answer": "当前没有待人工复核的记录。", "intent": ["待人工"], "records": [], "stats": {}}
        return {"ok": True, "answer": f"待人工复核共 {n} 笔：\n" + _brief(sub),
                "intent": ["待人工"], "records": _pick(sub),
                "stats": {"n": n, "claimed": _aggregate(sub)["claimed"]}}

    # ---- 4. 决策分布 统计 ----
    if _has("多少笔被拒", "被拒多少", "拒了多少", "被拒", "拒报多少", "几笔拒", "有多少笔拒", "多少笔"):
        sub = [r for r in recs if r.get("decision") == "reject"]
        return {"ok": True, "answer": f"全库共 {len(recs)} 笔；被拒 {len(sub)} 笔；决策分布：{_dec(all_agg['by_decision'])}。",
                "intent": ["决策分布"], "records": _pick(sub), "stats": all_agg}
    if _has("通过多少", "通过几笔", "批了几笔", "批了多少", "通过率", "通过比例", "占多少"):
        appr = all_agg["by_decision"].get("approve", 0)
        rate = (appr / len(recs) * 100) if recs else 0.0
        return {"ok": True,
                "answer": f"全库 {len(recs)} 笔，通过 {appr} 笔（通过率 {rate:.1f}%）。决策分布：{_dec(all_agg['by_decision'])}。",
                "intent": ["决策分布"], "records": [], "stats": all_agg}

    # ---- 5. 总额 统计 ----
    if _has("累计", "一共", "总共", "总额", "总金额", "合共", "总共报销", "累计核准", "总核准", "总拒付"):
        if not recs:
            return {"ok": True, "answer": "当前没有审核记录。", "intent": ["总额"], "records": [], "stats": {}}
        rate = (all_agg["rejected"] / all_agg["claimed"] * 100) if all_agg["claimed"] else 0.0
        return {"ok": True,
                "answer": f"全库累计 {len(recs)} 笔：申报{_fmt_money(all_agg['claimed'])}、" \
                          f"核准{_fmt_money(all_agg['approved'])}、拒付{_fmt_money(all_agg['rejected'])}" \
                          f"（核减率 {rate:.1f}%）；其中 {all_agg['llm_used_n']} 笔走了 LLM 兜底。",
                "intent": ["总额"], "records": [], "stats": all_agg}

    # ---- 6. 关键词 检索 ----
    if _has("列出", "哪些", "所有", "全部", "最近的", "最近", "报销记录", "单子"):
        # 问句里除指令词外，若还带了具体词则当关键词过滤事由/摘要/条目
        stop = ["列出", "哪些", "所有", "全部", "最近的", "最近", "报销记录", "报销", "单子", "记录", "的", "笔"]
        kw = ""
        for tok in stop:
            qn = qn.replace(tok, "")
        kw = qn
        # 若过滤后只剩很短的碎片(如"别的")则忽略，展示全库
        sub = [r for r in recs if (not kw) or (kw in (r.get("purpose", "") + r.get("summary", "")))] if kw else recs
        label = f"含「{kw}」" if kw else ""
        shown = _pick(sub)
        n = len(sub)
        if not sub:
            return {"ok": True, "answer": f"没有匹配{label}的审核记录。", "intent": ["列表"],
                    "records": [], "stats": {"n": 0}}
        a = _aggregate(sub)
        return {"ok": True,
                "answer": f"{label}共 {n} 笔（申报{_fmt_money(a['claimed'])}、核准{_fmt_money(a['approved'])}）：\n" + _brief(sub),
                "intent": ["列表"], "records": shown, "stats": a}

    # ---- 7. 用数字/币种 模糊汇总(兜底给个总览) ----
    return {"ok": True,
            "answer": f"我能对整库统计：累计总额/各决策笔数/类别核减/指定月份/待人工/按关键词列单。\n" \
                      f"当前全库 {len(recs)} 笔：申报{_fmt_money(all_agg['claimed'])}、" \
                      f"核准{_fmt_money(all_agg['approved'])}、拒付{_fmt_money(all_agg['rejected'])}。\n" \
                      f"可试试「累计核准多少钱」「有多少笔被拒」「按类别统计核减」「9月的报销总额」「列出所有待人工复核」。",
            "intent": ["帮助"], "records": [], "stats": all_agg}
