from memory.short_term_memory import ShortTermMemory


class MemoryEvaluator:
    """ShortTermMemory 评测

    使用方式：
        from eval.memory_eval import MemoryEvaluator
        evaluator = MemoryEvaluator(llm)
        report = evaluator.evaluate()
        evaluator.print_report(report)
    """

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self) -> dict:
        """跑三项评测，返回完整报告"""
        return {
            "compression": self._test_compression(),
            "context_retention": self._test_context_retention(),
            "tool_truncation": self._test_tool_truncation(),
        }


    def _test_compression(self) -> dict:
        """超限后旧消息是否被正确压缩为摘要"""
        mem = ShortTermMemory(self.llm, max_rounds=2)

        mem.add_message("user", "我叫张三，我喜欢用表格格式展示数据")
        mem.add_message("assistant", "好的张三，我会用表格")
        for i in range(8):
            mem.add_message("user", f"第{i}条普通对话内容")
            mem.add_message("assistant", f"第{i}条回复")

        has_summary = bool(mem.summary)
        name_retained = "张三" in mem.summary if has_summary else False
        preference_retained = "表格" in mem.summary if has_summary else False

        return {
            "name": "压缩质量",
            "passed": has_summary and name_retained and preference_retained,
            "has_summary": has_summary,
            "name_retained": name_retained,
            "preference_retained": preference_retained,
            "summary_preview": mem.summary[:100] if mem.summary else "",
            "message_count": len(mem.messages),
            "expect": "摘要应保留'张三'和'表格'两个关键信息",
        }


    def _test_context_retention(self) -> dict:
        """多轮对话后关键信息是否仍可获取"""
        mem = ShortTermMemory(self.llm, max_rounds=3)

        mem.add_message("user", "我想了解RAG系统的评估方法")
        mem.add_message("assistant", "评估RAG主要有三个指标：召回率、准确率、MRR")
        for i in range(6):
            mem.add_message("user", f"第{i}条无关对话")
            mem.add_message("assistant", f"第{i}条回复")

        context = mem.get_context()
        key_retained = "召回率" in context and "MRR" in context

        return {
            "name": "上下文保留",
            "passed": key_retained,
            "key_retained": key_retained,
            "context_length": len(context),
            "expect": "关键信息'召回率'和'MRR'应在上下文或摘要中保留",
        }


    def _test_tool_truncation(self) -> dict:
        """长工具输出是否被正确截断"""
        mem = ShortTermMemory(self.llm, max_rounds=10, max_tool_result_len=100)

        long_result = "A" * 500
        mem.add_tool_result("search", long_result)

        last_msg = mem.messages[-1]["content"] if mem.messages else ""
        truncated = len(last_msg) <= 150

        return {
            "name": "工具返回值截断",
            "passed": truncated,
            "original_length": 500,
            "stored_length": len(last_msg),
            "expect": f"500 字截断到 <=150 字",
        }


    def print_report(self, report: dict):
        """打印格式化的评估报告"""
        tests = [report["compression"], report["context_retention"], report["tool_truncation"]]
        passed = sum(1 for t in tests if t["passed"])
        total = len(tests)

        print("\n" + "=" * 50)
        print(f"  ShortTermMemory 评估报告")
        print(f"  通过: {passed}/{total}")
        print("-" * 50)

        for t in tests:
            status = "✅" if t["passed"] else "❌"
            print(f"  {status} {t['name']}")
            print(f"     期望: {t['expect']}")
            if not t["passed"]:
                for k, v in t.items():
                    if k not in ("name", "passed", "expect"):
                        print(f"     {k}: {v}")
        print("=" * 50 + "\n")
        return passed == total
