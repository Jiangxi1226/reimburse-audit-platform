# -*- coding: utf-8 -*-
"""四维 RAG 评估脚本——跑三类测试样本，定位链路损耗层级。

四个维度：
  1. 召回命中：检索结果是否覆盖 expected_chunks 关键词
  2. 引用覆盖：答案是否带来源引用/片段回指
  3. 答案忠实性：用 RAGEvaluator 算 faithfulness + 数字回指校验
  4. 人工复核：输出 answer 全文供人工判断（尤其拒答/冲突是否正确）

用法：
    python eval/run_evaluation.py            # 跑全部三类样本
    python eval/run_evaluation.py conflict   # 只跑冲突样本
    python eval/run_evaluation.py --no-llm   # 跳过 LLM 生成（只评估检索层）

依赖真实 LLM（OPENAI_API_KEY）与已入库文档；无 key 时检索层仍可跑。
"""
import sys, os, json, argparse
# Windows 终端 GBK 打不出 ⚠ 等字符，强制 UTF-8，避免 print_report 中途 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_env():
    """读取项目根 .env，灌入进程环境（LLM 实例依赖 os.getenv）。

    core/llm.py 只读环境变量、不自动加载 .env；评估脚本独立运行，
    需要自给自足地把 .env 注入，否则 LLM() 拿不到 BASE_URL/KEY/MODEL。
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'").strip(" "))


_load_env()

from eval.rag_cases import ALL_CASES
from tools.rag_tool import RAGTool


def _recall_metrics(results, case):
    """维度1 召回命中：期望关键词出现在检索文本中的比例。"""
    if not case.expected_chunks:
        return {"recall": None, "matched": [], "missed": case.expected_chunks}
    all_text = " ".join(r.get("text", "") or r.get("context", "") for r in results)
    matched = [k for k in case.expected_chunks if k in all_text]
    missed = [k for k in case.expected_chunks if k not in all_text]
    return {
        "recall": round(len(matched) / len(case.expected_chunks), 2) if case.expected_chunks else None,
        "matched": matched, "missed": missed,
    }


def _citation_coverage(answer, results):
    """维度2 引用覆盖：答案是否引用/回指了至少一个来源。"""
    has_citation = "[来源" in answer or "[来源:" in answer or "来源:" in answer
    return {"citation_present": has_citation, "n_sources": len(results)}


def evaluate_case(case, rag_tool, use_llm=True):
    """跑单个 case，返回四维结果。"""
    results = rag_tool.pipeline.smart_search(case.question, top_k=6)

    recall = _recall_metrics(results, case)
    citation = {"citation_present": False, "n_sources": len(results)}

    answer_full = ""
    faithfulness = None
    hallucination = None
    reference_accuracy = None
    conflict_detected = None

    if use_llm:
        from rag.answer_generator import generate_answer
        from eval.rag_evaluator import RAGEvaluator, HallucinationDetector
        from core.llm import LLM
        llm = LLM()
        gen = generate_answer(case.question, results, llm, use_llm_conflict=True)
        answer_full = gen["answer"]
        citation = _citation_coverage(answer_full, results)
        conflict_detected = bool(gen.get("conflicts"))

        contexts = [r.get("context", "") or r.get("text", "") for r in results]
        evaluator = RAGEvaluator(llm)
        faithfulness = evaluator._eval_faithfulness(answer_full, contexts)
        hallucination = HallucinationDetector(llm).detect(answer_full, contexts)
        reference_accuracy = evaluator.reference_accuracy(answer_full, contexts)

    return {
        "question": case.question,
        "category": case.category,
        "recall": recall,
        "citation": citation,
        "faithfulness": faithfulness,
        "reference_accuracy": reference_accuracy,
        "hallucination": hallucination,
        "conflict_detected": conflict_detected,
        "expect_conflict": case.expect_conflict,
        "answer": answer_full,
        "answer_note": case.expected_answer_note,
    }


def print_report(results):
    """输出可读报告。"""
    print("=" * 60)
    for r in results:
        cat_label = {"misread": "易误读", "conflict": "冲突", "refuse": "拒答"}[r["category"]]
        print(f"\n[{cat_label}] {r['question']}")
        rec = r["recall"]["recall"] if r["recall"].get("recall") is not None else "-"
        print(f"  ① 召回命中: {rec}  (漏: {r['recall'].get('missed', [])})")
        print(f"  ② 引用覆盖: {'有' if r['citation']['citation_present'] else '无'}  (来源数:{r['citation']['n_sources']})")
        if r["faithfulness"]:
            print(f"  ③ 忠实度: {r['faithfulness'].get('score')}  {r['faithfulness'].get('verdict')}")
        if r.get("reference_accuracy"):
            ra = r["reference_accuracy"]
            print(f"  ②′引用准确率: {ra.get('score')}  {ra.get('verdict')}"
                  f"  (有效引用 {ra.get('verified_citations')}/{ra.get('total_citations')})")
        if r["category"] == "conflict":
            det = "是" if r["conflict_detected"] else "否"
            exp = "是" if r["expect_conflict"] else "否"
            print(f"  ④ 冲突检测: {det} (期望 {exp}) {'OK' if det==exp else '⚠️'}")
        print(f"     [参照] {r['answer_note']}")
        if r["answer"]:
            print(f"     [答案] {r['answer'][:300]}")


def _diagnose(results: list[dict]) -> dict:
    """评估→诊断映射：把各维度分数翻译成可执行的调参建议（供人工采纳）。

    刻意不做"自动改参"——那是为了闭环而闭环。真实闭环第一步是让评估结论
    结构化、可回溯、能指向具体参数；是否调整由人决定。
    """
    if not results:
        return {"actionable": [], "summary": "无有效样本"}

    n = len(results)
    avg_faith = [r["faithfulness"]["score"] for r in results if r.get("faithfulness")]
    avg_recall = [r["recall"]["recall"] for r in results if r["recall"].get("recall") is not None]
    avg_ref = [r["reference_accuracy"]["score"] for r in results
               if r.get("reference_accuracy") and r["reference_accuracy"]["score"] is not None]
    avg_conflict = sum(1 for r in results if r.get("conflict_detected")) / n

    actionable = []

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None

    f, rec, ref = _avg(avg_faith), _avg(avg_recall), _avg(avg_ref)

    if f is not None and f < 0.5:
        actionable.append("忠实度低：答案与证据脱节。查 ①生成 prompt 是否放开'合理补全' ②context 是否塞入无关块 ③--no-llm 先确认检索层是否带了正确证据。")
    if rec is not None and rec < 0.7:
        actionable.append("召回低：证据没找全。建议 pipeline.search → search_hybrid（向量+BM25+RRF）或 search_multi（多粒度），或开 use_abbreviation_expand / 查询改写（enable_mqe）。")
    if ref is not None and ref < 0.3:
        actionable.append("引用准确率低：引用标注后结论数字无文档支撑，多为补全。检查 include_sources/来源格式，及生成 prompt 第 7 条忠实性硬约束是否生效。")
    if avg_conflict > 0 and avg_conflict < 0.9:
        actionable.append("冲突检测不稳定：期望检测到冲突的样本未全命中。查 conflict_resolver 的 LLM 复核开关 use_llm_conflict，或权威度/日期元数据是否完整。")
    if not actionable:
        actionable.append("各维度达标。可复跑回归确认稳定，无需调整。")

    return {"summary": f"样本 {n} 个", "actionable": actionable,
            "note": "诊断供人工采纳，当前为人工闭环——未自动回写任何检索/prompt/工具参数。"}


def main():
    parser = argparse.ArgumentParser(description="四维 RAG 评估")
    parser.add_argument("category", nargs="?", default="all",
                        choices=["all", "misread", "conflict", "refuse"])
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 生成，只评估检索层")
    parser.add_argument("--out", default="", help="评估结果落盘路径（JSON），默认不落盘")
    args = parser.parse_args()

    cases = ALL_CASES if args.category == "all" \
        else [c for c in ALL_CASES if c.category == args.category]

    rag_tool = RAGTool()
    results = []
    for case in cases:
        try:
            results.append(evaluate_case(case, rag_tool, use_llm=not args.no_llm))
        except Exception as e:
            print(f"[跳过] {case.question}: {e}")
    print_report(results)

    diagnosis = _diagnose(results)
    print("\n" + "=" * 60)
    print("诊断与调参建议（人工采纳）")
    for i, tip in enumerate(diagnosis["actionable"], 1):
        print(f"  {i}. {tip}")

    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, "eval_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump({"category": args.category, "results": results,
                       "diagnosis": diagnosis}, fp, ensure_ascii=False, indent=2)
        print(f"\n[落盘] 评估结果 + 诊断已写入 {out_path}")
    except Exception as e:
        print(f"[落盘失败] {e}")


if __name__ == "__main__":
    main()
