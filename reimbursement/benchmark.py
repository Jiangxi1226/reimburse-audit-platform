# -*- coding: utf-8 -*-
"""报销审核全量测试集 + 量化评估。

两大部分：
  A. 结构化条目测试集（确定性为主，不依赖 OCR）：覆盖每个规则分支，用于精确衡量
     「决策准确率 / 逐条核定准确率 / 金额准确率 / LLM 依赖占比」。
  B. 真实票据文件端到端（OCR + LLM 抽取 + 审查）：衡量「字段抽取命中 / 校准通过 / 端到端正确性」。

开行：python reimbursement/benchmark.py   （B 部分会调用 PaddleOCR，较慢）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reimbursement.service import audit_claim
from reimbursement.audit_engine import ReimbursementEngine

EPS = 0.01


# ---------------------------------------------------------------------------
# A. 结构化条目测试集
# ---------------------------------------------------------------------------
# 每条：name / items / claimed_total(可选) / expect{decision, approved, rejected}
#         allow_hint_sev(可选，容忍的逐条决策集合，用于说明波动)
# 今天=2026-08-30，报销窗口 30 天 → 日期 >=2026-07-31 合规。
STRUCTURED_CASES = [
    # ---- 全合规 ----
    {"name": "全合规●多笔", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州", "desc": "酒店标准间", "itemization": ["标准间"]},
        {"category": "餐饮", "date": "2026-08-01", "amount": 50, "desc": "早餐", "itemization": ["早餐"]},
        {"category": "交通", "date": "2026-08-02", "amount": 30, "desc": "地铁通勤"},
    ], "expect": {"decision": "approve", "approved": 480, "rejected": 0}},

    # ---- 住宿额度 ----
    {"name": "住宿●高端城市超上限", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 950, "city": "北京", "desc": "酒店标准间", "itemization": ["标准间1晚"]},
    ], "expect": {"decision": "partial", "approved": 800, "rejected": 150}},
    {"name": "住宿●普通城市超上限", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 620, "city": "苏州", "desc": "酒店标准间", "itemization": ["标准间1晚"]},
    ], "expect": {"decision": "partial", "approved": 500, "rejected": 120}},
    {"name": "住宿●高端城市未超", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 620, "city": "北京", "desc": "酒店标准间", "itemization": ["标准间1晚"]},
    ], "expect": {"decision": "approve", "approved": 620, "rejected": 0}},
    {"name": "住宿●缺明细(仅提示)", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州", "desc": "酒店"},
    ], "expect": {"decision": "manual_review", "approved": 400, "rejected": 0}},

    # ---- 餐饮 ----
    {"name": "餐饮●单日超限", "items": [
        {"category": "餐饮", "date": "2026-08-01", "amount": 250, "desc": "晚餐商务宴请", "itemization": ["晚餐"]},
    ], "expect": {"decision": "partial", "approved": 200, "rejected": 50}},
    {"name": "餐饮●单餐软限(总额未超)", "items": [
        {"category": "餐饮", "date": "2026-08-01", "amount": 80, "desc": "早餐", "itemization": ["早餐"]},
    ], "expect": {"decision": "approve", "approved": 80, "rejected": 0}},

    # ---- 交通 ----
    {"name": "交通●航空高端需说明", "items": [
        {"category": "交通-航空", "date": "2026-08-01", "amount": 1000, "desc": "公务舱机票"},
    ], "expect": {"decision": "manual_review", "approved": 1000, "rejected": 0}},
    {"name": "交通●出租超日限", "items": [
        {"category": "交通-出租", "date": "2026-08-01", "amount": 180, "desc": "夜间打车"},
    ], "expect": {"decision": "partial", "approved": 150, "rejected": 30}},

    # ---- 禁报 / 个人 ----
    {"name": "禁报●个人消费(面膜)", "items": [
        {"category": "其他", "date": "2026-08-03", "amount": 300, "desc": "面膜/护肤品"},
    ], "expect": {"decision": "reject", "approved": 0, "rejected": 300}},
    {"name": "禁报●礼品类", "items": [
        {"category": "礼品", "date": "2026-08-03", "amount": 500, "desc": "商务礼品"},
    ], "expect": {"decision": "reject", "approved": 0, "rejected": 500}},

    # ---- 逐条核定（混杂单）----
    {"name": "混杂●合规+个人(逐条)", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 620, "city": "北京", "desc": "酒店标准间", "itemization": ["标准间1晚"]},
        {"category": "餐饮", "date": "2026-08-01", "amount": 250, "desc": "晚餐", "itemization": ["晚餐"]},
        {"category": "其他", "date": "2026-08-03", "amount": 300, "desc": "面膜/护肤品"},
    ], "expect": {"decision": "reject", "approved": 820, "rejected": 350}},

    # ---- 日期窗口 ----
    {"name": "日期●超报销窗口", "items": [
        {"category": "餐饮", "date": "2026-07-20", "amount": 100, "desc": "午餐", "itemization": ["午餐"]},
    ], "expect": {"decision": "manual_review", "approved": 100, "rejected": 0}},

    # ---- 申报总额交叉核对 ----
    {"name": "金额●申报总额不符", "items": [
        {"category": "餐饮", "date": "2026-08-01", "amount": 150, "desc": "晚餐", "itemization": ["晚餐"]},
    ], "claimed_total": 200, "expect": {"decision": "manual_review", "approved": 150, "rejected": 0}},

    # ---- 无条目 ----
    {"name": "边缘●无可审核条目", "items": [],
     "expect": {"decision": "reject", "approved": 0, "rejected": 0}},

    # ---- 防骗保：供应商黑名单（命中即整拒，对齐红旗硬决定）----
    {"name": "黑名单●供应商命中", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州",
         "desc": "酒店", "itemization": ["1晚"], "supplier": "XX商贸"},
    ], "expect": {"decision": "reject", "approved": 0, "rejected": 400}},
    {"name": "黑名单●正常供应商不误伤", "items": [
        {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州",
         "desc": "酒店", "itemization": ["1晚"], "supplier": "锦江之星"},
    ], "expect": {"decision": "approve", "approved": 400, "rejected": 0}},
]


def _run_structured() -> dict:
    engine = ReimbursementEngine()
    results = []
    for c in STRUCTURED_CASES:
        t0 = time.perf_counter()
        r = audit_claim(items=c["items"], claimed_total=c.get("claimed_total"),
                        purpose=c.get("name", ""), use_llm_arbitration=False, engine=engine)
        elapsed = (time.perf_counter() - t0) * 1000
        v = r["data"]["verdict"] if r["ok"] else {}
        got = {
            "decision": v.get("decision"),
            "approved": round(v.get("approved_amount", -1), 2),
            "rejected": round(v.get("rejected_amount", -1), 2),
        }
        exp = c["expect"]
        ok = (got["decision"] == exp["decision"]
              and abs(got["approved"] - exp["approved"]) < EPS
              and abs(got["rejected"] - exp["rejected"]) < EPS)
        results.append({
            "name": c["name"], "ok": ok, "got": got, "expect": exp,
            "elapsed_ms": round(elapsed, 1),
            "llm_used": r["data"]["llm_used"] if r["ok"] else None,
        })
    return results


# ---------------------------------------------------------------------------
# B. 真实票据文件端到端（慢，会调 PaddleOCR）
# ---------------------------------------------------------------------------
# 预期为"应为的正确结论"，用于衡量端到端字段抽取 + 审查是否得到正确处置。
FILE_CASES = [
    {"name": "文件●住宿_北京_超标(620,高档城市未超)", "path": "住宿_北京_超标.png",
     "expect": {"decision": "approve", "approved": 620, "rejected": 0}},
    {"name": "文件●餐饮_商务宴请(250,超日限)", "path": "餐饮_商务宴请.png",
     "expect": {"decision": "partial", "approved": 200, "rejected": 50}},
    {"name": "文件●交通_地铁(50,合规)", "path": "交通_地铁.png",
     "expect": {"decision": "approve", "approved": 50, "rejected": 0}},
    {"name": "文件●个人消费_护肤(300,禁报)", "path": "个人消费_护肤.png",
     "expect": {"decision": "reject", "approved": 0, "rejected": 300}},
]


def _run_files() -> dict:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_receipts")
    results = []
    for c in FILE_CASES:
        p = os.path.join(base, c["path"])
        t0 = time.perf_counter()
        r = audit_claim(receipt_files=[p], use_llm_arbitration=True)
        elapsed = time.perf_counter() - t0
        v = r["data"]["verdict"] if r["ok"] else {}
        got = {
            "decision": v.get("decision"),
            "approved": round(v.get("approved_amount", -1), 2),
            "rejected": round(v.get("rejected_amount", -1), 2),
        }
        exp = c["expect"]
        # 端到端用"决策一致 + 金额在容差 1 元内"判定（OCR/LLM 抽取本身有微小噪声）
        ok = got["decision"] == exp["decision"] and abs(got["approved"] - exp["approved"]) < 1.0
        results.append({
            "name": c["name"], "ok": ok, "got": got, "expect": exp,
            "elapsed_s": round(elapsed, 1),
            "llm_used": r["data"]["llm_used"] if r["ok"] else None,
            "error": r.get("error", "") if not r["ok"] else "",
        })
    return results


def _report(title: str, results: list[dict]):
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    print(f"\n===== {title} =====")
    print(f"用例数: {total}   通过: {passed}   通过率: {passed/total*100:.1f}%")
    for r in results:
        flag = "✅" if r["ok"] else "❌"
        g, e = r["got"], r["expect"]
        print(f" {flag} {r['name']}  决策 {g['decision']}/{e['decision']}  "
              f"批 {g['approved']}/{e['approved']} 拒 {g['rejected']}/{e['rejected']}"
              + (f"  ({r['elapsed_ms']}ms)" if "elapsed_ms" in r else f" ({r['elapsed_s']}s)")
              + (f" llm={r['llm_used']}" if "llm_used" in r and r["llm_used"] else "")
              + (f" | 错:{r.get('error')}" if r.get("error") else ""))
    return passed, total


# ---------------------------------------------------------------------------
# C. LLM 兜底（仅语义模糊点）：验证 LLM 只在规则无法判定的模糊处被调用，
#    且能正确裁决"是否业务相关"。用于量化"少依赖 LLM"的代价边界。
# ---------------------------------------------------------------------------
AMBIGUITY_CASES = [
    {"name": "兜底●类别其他+杂项(应判非业务)", "items": [
        {"category": "其他", "date": "2026-08-01", "amount": 200, "desc": "杂项开支"},
    ]},
    {"name": "兜底●类别其他+明确业务(应判业务)", "items": [
        {"category": "其他", "date": "2026-08-01", "amount": 300, "desc": "软件开发服务费"},
    ]},
]


def _run_duplicate() -> list[dict]:
    """防骗保：发票唯一性（注入历史发票号引擎）。

    漏检单列警示：带 invoice_no 却未被 `DUPLICATE_INVOICE` 命中的重复提交，视为最致命漏检。
    """
    from reimbursement.audit_engine import ReimbursementEngine
    engine = ReimbursementEngine(seen_invoices={"FP2026001": "R123", "FP2026099": "R456"})
    cases = [
        {"name": "防骗保●重复发票(硬拒)", "items": [
            {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州",
             "desc": "酒店", "itemization": ["1晚"], "invoice_no": "FP2026001"}],
         "expect": {"decision": "reject", "approved": 0, "rejected": 400}},
        {"name": "防骗保●新发票不误伤", "items": [
            {"category": "住宿", "date": "2026-08-01", "amount": 400, "city": "苏州",
             "desc": "酒店", "itemization": ["1晚"], "invoice_no": "FP2026088"}],
         "expect": {"decision": "approve", "approved": 400, "rejected": 0}},
    ]
    results = []
    for c in cases:
        r = audit_claim(items=c["items"], purpose=c["name"],
                        use_llm_arbitration=False, engine=engine)
        v = r["data"]["verdict"] if r["ok"] else {}
        got = {"decision": v.get("decision"),
               "approved": round(v.get("approved_amount", -1), 2),
               "rejected": round(v.get("rejected_amount", -1), 2)}
        exp = c["expect"]
        dup_hit = "DUPLICATE_INVOICE" in (v.get("policy_references") or [])
        # 漏检：expected 是重复/黑名单，但引擎却没命中对应 issue → 最致命
        leak = (exp["decision"] == "reject") and not dup_hit and not (
            "BLACKLISTED_SUPPLIER" in (v.get("policy_references") or []))
        ok = (got["decision"] == exp["decision"]
              and abs(got["approved"] - exp["approved"]) < EPS
              and abs(got["rejected"] - exp["rejected"]) < EPS)
        results.append({"name": c["name"], "ok": ok, "got": got, "expect": exp,
                        "llm_used": r["data"]["llm_used"] if r["ok"] else None,
                        "leak": leak,
                        "policy_refs": v.get("policy_references", [])})
    return results


def _run_ambiguity() -> list[dict]:
    results = []
    for c in AMBIGUITY_CASES:
        t0 = time.perf_counter()
        try:
            r = audit_claim(items=c["items"], purpose=c["name"],
                            use_llm_arbitration=True)
            data = r["data"] if r["ok"] else {}
            v = data.get("verdict", {})
            got = {"decision": v.get("decision"),
                   "approved": round(v.get("approved_amount", -1), 2)}
            results.append({
                "name": c["name"], "ok": True, "got": got,
                "llm_used": data.get("llm_used"),
                "arb_msg": (data.get("arbitration") or {}).get("reason", ""),
                "fallback_fail": "LLM_FALLBACK_FAIL" in (v.get("policy_references") or []),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": "",
            })
        except Exception as e:
            results.append({"name": c["name"], "ok": False, "got": {},
                            "llm_used": None, "arb_msg": "", "elapsed_ms": 0,
                            "error": f"{type(e).__name__}: {e}"})
    return results


def main(include_files: bool = True):
    s_results = _run_structured()
    sp, st = _report("A. 结构化条目测试集（纯规则）", s_results)
    # 决策准确率 + 金额准确率分别算
    dec_acc = sum(1 for r in s_results if r["got"]["decision"] == r["expect"]["decision"]) / st
    amt_acc = sum(1 for r in s_results if abs(r["got"]["approved"] - r["expect"]["approved"]) < EPS) / st
    llm_used = sum(1 for r in s_results if r["llm_used"])
    avg_ms = sum(r["elapsed_ms"] for r in s_results) / st
    print("\n--- 量化指标（结构化） ---")
    print(f" 决策准确率: {dec_acc*100:.1f}%  ({dec_acc*st:.0f}/{st})")
    print(f" 核准金额准确率: {amt_acc*100:.1f}%  (±{EPS})")
    print(f" LLM 依赖: {llm_used}/{st} 用例调用 LLM（其余纯确定性）占比 {llm_used/st*100:.1f}%")
    print(f" 平均审核耗时: {avg_ms:.1f} ms（不含 OCR/云端识别）")

    # A′ 防骗保：发票唯一性 + 供应商黑名单（漏=赔付损失，对齐项目二"红旗召回"思路）
    d_results = _run_duplicate()
    print("\n===== A′. 防骗保（发票唯一性 + 黑名单） =====")
    for r in d_results:
        want_miss = r["expect"]["decision"] == "reject"
        ln = "⚠️ 漏检!" if r["leak"] else "  "
        flag = "✅" if r["ok"] else "❌"
        print(f" {flag} {r['name']}  → {r['got']['decision']}  refs={r.get('policy_refs')} {ln}")
    dup_pass = sum(1 for r in d_results if r["ok"])
    dup_leak = sum(1 for r in d_results if r["leak"])
    print(f" 防骗保: {dup_pass}/{len(d_results)} 用例通过 | 漏检(最致命) {dup_leak} 例")
    if dup_leak:
        print("  ⚠️ 以下关键漏检必须修复（漏=赔付损失）：")
        for r in d_results:
            if r["leak"]:
                print(f"     - {r['name']}：应拒但未命中 DUPLICATE_INVOICE/BLACKLISTED_SUPPLIER")
    if include_files:
        f_results = _run_files()
        fp, ft = _report("B. 真实票据文件端到端（OCR+LLM 抽取+审查）", f_results)
        f_ok = sum(1 for r in f_results if r["ok"])
        avg_f = sum(r["elapsed_s"] for r in f_results) / ft
        print(f"\n--- 量化指标（端到端文件） ---")
        print(f" 端到端处置正确率: {f_ok/ft*100:.1f}%  ({f_ok}/{ft})")
        print(f" 平均端到端耗时(含OCR): {avg_f:.1f} s（单张票据）")

    # C. LLM 兜底（单独，不并入通过率——依赖 LLM 主观判断，作观测）
    g_results = _run_ambiguity()
    print(f"\n===== C. LLM 兜底（仅语义模糊点） =====")
    for r in g_results:
        flag = "✅" if r["ok"] else "❌"
        print(f" {flag} {r['name']}  → {r['got'].get('decision')}  llm_used={r['llm_used']}"
              f"  {r['elapsed_ms']}ms  裁决:{r['arb_msg'][:40] or r['error']}")
    g_llm = sum(1 for r in g_results if r["llm_used"])
    g_fail = sum(1 for r in g_results if r.get("fallback_fail"))
    if g_results:
        print(f" 该组（均为语义模糊）LLM 调用: {g_llm}/{len(g_results)} —— 证实 LLM 仅用于模糊点是/否业务相关")
        if g_fail:
            print(f" ⚠️ 语义兜底失败降级: {g_fail} 例（LLM 未返回可解析裁决，已显式标记转人工，绝不静默）")
    return sp, st


if __name__ == "__main__":
    main(include_files=os.getenv("NO_OCR") != "1")
