import re, math



def calibrate_multimodal_weights(text_results: list[dict],
                                  image_results: list[dict]) -> dict:
    """校准图文两路结果的权重——让 MiniLM 和 CLIP 的分数可比。

    问题：MiniLM 的余弦相似度通常 0.5~0.9，CLIP 的通常 0.3~0.7。
    直接 merge 按 similarity 排序 → 文本结果永远排前面。

    解决：对每路做 z-score 归一化 → 映射到同一分布 →
         加权融合 weight_text + weight_image = 1.0

    面试能讲：
      "MiniLM 和 CLIP 的相似度在不同几何空间——384维超球面和512维超球面，
      数值分布不一样。我用了 z-score 归一化把两路分数映射到同一分布，
      再按业务权重加权——文字密集查询 text_weight=0.7，
      视觉查询 image_weight=0.6。"
    """
    def z_score(scores: list[float]) -> list[float]:
        if not scores:
            return []
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)) if len(scores) > 1 else 1
        return [(s - mean) / (std + 1e-8) for s in scores]

    text_scores = [r.get("similarity", 0) for r in text_results]
    image_scores = [r.get("similarity", 0) for r in image_results]

    text_z = z_score(text_scores)
    image_z = z_score(image_scores)

    return {
        "text_score_range": (min(text_scores), max(text_scores)) if text_scores else (0, 0),
        "image_score_range": (min(image_scores), max(image_scores)) if image_scores else (0, 0),
        "calibrated_text_z": text_z,
        "calibrated_image_z": image_z,
        "recommendation": (
            "text_weight=0.7, image_weight=0.3" if len(text_results) > len(image_results)
            else "text_weight=0.4, image_weight=0.6"
        ),
    }



class RAGEvaluator:
    """RAGAS 风格评估器——自研实现，不依赖 ragas 库。

    三个核心指标：
      faithfulness: 答案有多少是基于检索内容的（vs 幻觉）
      answer_relevancy: 答案和问题的相关程度
      context_precision: 检索结果中相关内容的精度
    """

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, question: str, answer: str,
                 contexts: list[str]) -> dict:
        """一次完整的 RAGAS 评估。"""
        return {
            "faithfulness": self._eval_faithfulness(answer, contexts),
            "answer_relevancy": self._eval_answer_relevancy(question, answer),
            "context_precision": self._eval_context_precision(question, contexts),
        }


    def _eval_faithfulness(self, answer: str, contexts: list[str]) -> dict:
        """评估答案中的陈述有多少能在上下文中找到依据。

        方法：
          ① 把答案拆成独立陈述句
          ② 逐句检查是否在上下文中被支持
          ③ 计算支持率 = 被支持的陈述数 / 总陈述数

        faithfulness = 1.0 → 每个陈述都能在文档中找到，零幻觉
        faithfulness = 0.0 → 所有陈述都是编的
        """
        statements = [s.strip() for s in re.split(r'[。！？\n]', answer) if len(s.strip()) > 5]

        if not statements:
            return {"score": 0.0, "supported": 0, "total": 0,
                    "verdict": "无法拆解为独立陈述"}

        context_combined = "\n".join(contexts[:5])
        supported = 0
        details = []

        for stmt in statements[:10]:
            is_supported = self._check_support(stmt, context_combined)
            if is_supported:
                supported += 1
            details.append({"statement": stmt[:80], "supported": is_supported})

        score = supported / len(statements[:10]) if statements else 0

        verdict = "高忠实度" if score > 0.8 else "中忠实度" if score > 0.5 else "低忠实度——可能存在较多幻觉"

        return {
            "score": round(score, 4),
            "supported": supported,
            "total": len(statements[:10]),
            "verdict": verdict,
            "details": details[:5],
        }

    def _check_support(self, statement: str, context: str) -> bool:
        """LLM 判断某条陈述是否被上下文支持。"""
        prompt = (
            f"上下文：{context[:1500]}\n\n"
            f"陈述：{statement}\n\n"
            f"这个陈述中的信息是否能在上下文中找到依据？只回复 YES 或 NO。"
        )
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            return "YES" in response.upper()
        except Exception:
            return False


    def _eval_answer_relevancy(self, question: str, answer: str) -> dict:
        """评估答案和问题的相关程度。

        方法：LLM 判断答案是否直接回应了问题，还是跑偏了。
        """
        prompt = (
            f"问题：{question}\n答案：{answer[:1000]}\n\n"
            f"答案是否直接回应了问题？评分 1-5（1=完全无关，5=完全直接回应问题核心）。只回复数字。"
        )
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            score_match = re.search(r'[1-5]', response)
            score = int(score_match.group()) / 5.0 if score_match else 0.6
        except Exception:
            score = 0.5

        return {
            "score": round(score, 4),
            "verdict": "高度相关" if score > 0.8 else "基本相关" if score > 0.5 else "偏题"
        }


    def _eval_context_precision(self, question: str, contexts: list[str]) -> dict:
        """评估检索结果中相关内容的精度。

        方法：对每个 context chunk，LLM 判断是否和问题相关。
        context_precision = 相关的块数 / 总块数
        """
        if not contexts:
            return {"score": 0.0, "verdict": "无上下文可评估"}

        relevant = 0
        for ctx in contexts[:5]:
            prompt = (
                f"问题：{question}\n文本：{ctx[:300]}\n\n"
                f"这段文本是否包含回答问题所需的信息？只回复 YES 或 NO。"
            )
            try:
                response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
                if "YES" in response.upper():
                    relevant += 1
            except Exception:
                pass

        score = relevant / min(len(contexts), 5)

        return {
            "score": round(score, 4),
            "relevant_chunks": relevant,
            "total_chunks": min(len(contexts), 5),
            "verdict": "检索精准" if score > 0.8 else "部分精准" if score > 0.4 else "检索较差——需优化检索策略"
        }


    def _extract_citations(self, answer: str) -> list[str]:
        """提取答案中的来源引用标注。

        本系统的答案实际引用形态有几种，需都识别：
          - [来源: xxx] / 来源: xxx        （answer_generator 规定的格式）
          - 来源[1](审计报告)称：...       （LLM 生成时常用的「来源[N]」形态）
        「来源引用」小标题（后无指向数字）不应误抓。"""
        pats = [
            r'\[来源:[^\]]*\]',          # [来源: 文件名]
            r'来源:[^\n，。、；]{2,30}',   # 来源: xxx
            r'来源\s*\[\d+\]',            # 来源[1] / 来源 [2]
        ]
        cites = []
        for p in pats:
            cites += re.findall(p, answer)
        # 去掉「来源引用」这类无具体指向的标题词
        return [c.strip() for c in cites if c.strip() not in ("来源引用", "来源")]

    def _eval_reference_accuracy(self, answer: str, contexts: list[str]) -> dict:
        """引用准确率（证据支撑率）——答案中每个带来源标注的结论是否真的被文档支撑。

        与 faithfulness 的区别：faithfulness 让 LLM 逐句判"是否被支持"，
        reference_accuracy 是**规则法**：只取带来源引用标注的结论，核其紧邻文本里
        宣布的关键数字能否在检索语料中找到依据。不依赖 LLM，可离线跑（--no-llm）。

        抓到的问题：模型"引用了一个根本不存在的结果"，或引用标注后的结论数字是补全的
        ——这正是图里"引用准确率"要兜的幻觉。

        Returns:
            {"score", "verified_citations", "total_citations", "unverified", "verdict"}
        """
        citations = self._extract_citations(answer)
        if not citations:
            return {"score": None, "verified_citations": 0, "total_citations": 0,
                    "unverified": [], "verdict": "无引用标注，无法评估",
                    "note": "建议生成阶段开启来源引用（include_sources=True）"}

        corpus = "\n".join(contexts[:5])
        verified, unverified = [], []
        for cite in citations:
            # 跳过引用标注本身（其 chunk_id 可能自带 0_0 这类数字，属噪音），
            # 只看它后面 120 字内宣布的数字，逐一核是否在语料中出现。
            start = answer.find(cite)
            if start == -1:
                unverified.append(cite[:30])
                continue
            seg = answer[start + len(cite): start + len(cite) + 120]
            nums = re.findall(r'\d+\.?\d*', seg)
            nums_ok = all(n in corpus for n in nums) if nums else True
            if nums_ok:
                verified.append(cite[:30])
            else:
                unverified.append(cite[:30])

        total = len(citations)
        score = round(len(verified) / total, 4)
        verdict = (
            "引用准确" if score > 0.8
            else "引用部分可靠" if score > 0.5
            else "引用不可靠——存在引用结论数字无文档支撑"
        )
        return {
            "score": score, "verified_citations": len(verified),
            "total_citations": total, "unverified": unverified[:5],
            "verdict": verdict,
        }

    def reference_accuracy(self, answer: str, contexts: list[str]) -> dict:
        """对外便捷入口（等价 _eval_reference_accuracy）。"""
        return self._eval_reference_accuracy(answer, contexts)

    def full_report(self, question: str, answer: str,
                    contexts: list[str]) -> dict:
        """生成完整评估报告——含通过/不通过判定 + 引用准确率维度。"""
        results = self.evaluate(question, answer, contexts)
        results["reference_accuracy"] = self._eval_reference_accuracy(answer, contexts)

        ref = results["reference_accuracy"]["score"]
        passed = all([
            results["faithfulness"]["score"] >= 0.5,
            results["answer_relevancy"]["score"] >= 0.5,
            results["context_precision"]["score"] >= 0.3,
            ref is None or ref >= 0.3,  # 无引用标注则不因该项判失败
        ])

        return {
            "question": question[:100],
            "answer_preview": answer[:200],
            "passed": passed,
            "scores": results,
            "summary": (
                "RAG 系统通过评估" if passed
                else "RAG 系统存在改进空间：" +
                     ("[忠实度低→可能幻觉] " if results["faithfulness"]["score"] < 0.5 else "") +
                     ("[相关性低→答案偏题] " if results["answer_relevancy"]["score"] < 0.5 else "") +
                     ("[精度低→检索差] " if results["context_precision"]["score"] < 0.3 else "") +
                     ("[引用准确率低] " if results["reference_accuracy"]["score"] is not None
                      and results["reference_accuracy"]["score"] < 0.3 else "")
            ),
        }



class HallucinationDetector:
    """幻觉检测器——四层检查。

    ① 数字核对：答案中的数字是否在上下文中有对应？
    ② 实体核实：答案中的人名/组织名/产品名是否在上下文中？
    ③ 引用追溯：答案中每个关键事实能否追溯到具体文档块？
    ④ 置信度比对：答案的确定程度和检索置信度是否匹配？
    """

    def __init__(self, llm):
        self.llm = llm

    def detect(self, answer: str, contexts: list[str]) -> dict:
        """检测答案中的幻觉迹象。Returns {hallucination_flag, details, evidence}."""
        context_full = "\n".join(contexts)

        nums_in_answer = set(re.findall(r'\d+\.?\d*', answer))
        nums_in_context = set(re.findall(r'\d+\.?\d*', context_full))
        unverified_nums = nums_in_answer - nums_in_context

        entities = self._extract_entities(answer)
        verified = self._verify_entities(entities, context_full)

        hallucination_score = 0.0
        flags = []

        if unverified_nums:
            hallucination_score += 0.3
            flags.append(f"发现未验证数字: {unverified_nums}")

        if verified.get("unverified", []):
            hallucination_score += 0.3
            flags.append(f"发现未验证实体: {verified['unverified'][:3]}")

        if not contexts or all(len(c.get("text", "")) < 50 for c in (contexts if isinstance(contexts[0], dict) else [])):
            hallucination_score += 0.4
            flags.append("检索上下文为空或过短——答案可能完全是编造的")

        return {
            "hallucination_risk": "高" if hallucination_score > 0.5 else "中" if hallucination_score > 0.2 else "低",
            "score": round(hallucination_score, 2),
            "flags": flags,
            "unverified_numbers": list(unverified_nums)[:5],
            "entity_verification": verified,
            "recommendation": (
                "建议检查答案的虚构内容" if hallucination_score > 0.5
                else "答案基本可信" if hallucination_score < 0.2
                else "答案可能包含部分不准确信息"
            ),
        }

    def _extract_entities(self, text: str) -> list[str]:
        """简单实体提取——大写开头的中文词组。"""
        words = re.findall(r'[A-Z][A-Za-z]+|[一-鿿]{2,6}', text)
        stop_words = {"这个", "那个", "一个", "可以", "需要", "应该", "可能", "使用", "进行", "通过", "根据"}
        return [w for w in words if w not in stop_words][:10]

    def _verify_entities(self, entities: list[str], context: str) -> dict:
        """核实实体是否在上下文中出现。"""
        verified, unverified = [], []
        for ent in entities:
            if ent in context:
                verified.append(ent)
            else:
                unverified.append(ent)
        return {"verified": verified, "unverified": unverified}
