import os, json
from core.base import Tool
from rag.pipeline import RAGPipeline
from rag.retriever import search_expanded
from utils.security import validate_file, scan_retrieval_results


def _ok(data) -> str:
    return json.dumps({"ok": True, "data": data, "error": ""}, ensure_ascii=False)

def _err(msg) -> str:
    return json.dumps({"ok": False, "data": None, "error": msg}, ensure_ascii=False)


# 检索 top_k 上限：LLM 通过工具参数可随意传值，若不钳制，Agent 传一个极大值
# 会让底层做 top_k*4 的路数召回，瞬间打爆向量库。这里统一 clamp 到 [1, MAX]。
_MAX_TOP_K = 20

def _clamp_top_k(top_k: int) -> int:
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    return max(1, min(top_k, _MAX_TOP_K))


class RAGTool(Tool):
    def __init__(self, use_multi_granularity: bool = True,
                 use_table_preserving: bool = True,
                 use_abbreviation_expand: bool = True):
        super().__init__("rag", (
            "知识库工具：add/add_file/add_image/search/search_advanced/"
            "search_hybrid/search_multi/search_filtered/search_all/qa/stats。全部返回 JSON。"
        ))
        self.pipeline = RAGPipeline(
            use_multi_granularity=use_multi_granularity,
            use_table_preserving=use_table_preserving,
            use_abbreviation_expand=use_abbreviation_expand,
        )

    def _add(self, text: str) -> str:
        if not text or not text.strip():
            return _err("文本内容为空")
        before = self.pipeline.text_count
        self.pipeline.add_document(text)
        return _ok({"added_chunks": self.pipeline.text_count - before, "total_chunks": self.pipeline.text_count})

    def _add_file(self, file_path: str) -> str:
        check = validate_file(file_path)
        if not check["valid"]:
            return _err(f"文件校验失败: {check['reason']}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp"):
                return self._add_image(file_path)

            from rag.document_loader import load_with_quality
            result = load_with_quality(file_path)
            text = result["text"]
            quality = result["quality"]
            file_meta = result["file_metadata"]

            before = self.pipeline.text_count
            self.pipeline.add_document(
                text, source=file_path,
                file_metadata=file_meta,
                quality_info=quality,
            )
            added = self.pipeline.text_count - before
            return _ok({
                "added_chunks": added,
                "total_chunks": self.pipeline.text_count,
                "quality": quality["quality"].value,
                "quality_score": quality["score"],
                "warnings": quality.get("warnings", []),
                "file_type": file_meta.get("file_type", ""),
            })
        except ValueError as e:
            return _err(f"不支持的格式: {e}")
        except Exception as e:
            return _err(f"文件处理失败: {e}")

    def _add_image(self, image_path: str) -> str:
        check = validate_file(image_path)
        if not check["valid"]:
            return _err(f"图片校验失败: {check['reason']}")
        try:
            from rag.document_loader import _ocr_image_file
            ocr_text = _ocr_image_file(image_path)
            img_id = self.pipeline.add_image(image_path, ocr_text)
            return _ok({"image_id": img_id, "ocr_text": ocr_text[:200], "image_count": self.pipeline.image_count})
        except Exception as e:
            return _err(f"图片处理失败: {e}")

    def _search(self, query: str, top_k: int = 5) -> str:
        """复杂度感知的智能检索：
        低复杂度→单路向量；中→混合检索(向量+BM25+RRF)；高→多粒度联合。
        """
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        results = self.pipeline.smart_search(query, top_k)
        complexity = self.pipeline.get_query_complexity(query)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "complexity": complexity,
                    "results": self._simplify(results), "count": len(results)})

    def _search_advanced(self, query: str, top_k: int = 5, enable_mqe: bool = True, enable_hyde: bool = False) -> str:
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        try:
            from core.llm import LLM
            results = search_expanded(query, self.pipeline, LLM(), top_k, enable_mqe, enable_hyde)
        except Exception:
            results = self.pipeline.search(query, top_k)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "strategy": "mqe+hyde" if enable_hyde else "mqe", "results": self._simplify(results), "count": len(results)})

    def _search_all(self, query: str, top_k: int = 5) -> str:
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        results = self.pipeline.search_all(query, top_k)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "mode": "multimodal", "results": self._simplify(results), "count": len(results)})

    def _search_hybrid(self, query: str, top_k: int = 5) -> str:
        """混合检索：向量语义 + BM25 关键词 + RRF 融合。
        适合产品型号、日期、编号等精确关键词查询。"""
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        results = self.pipeline.search_hybrid(query, top_k)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "mode": "hybrid", "results": self._simplify(results), "count": len(results)})

    def _search_multi(self, query: str, top_k: int = 5) -> str:
        """多粒度联合检索：同时搜粗/中/细三层索引，合并去重。
        适合复杂查询——既需要宏观总结又需要精确数字。"""
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        results = self.pipeline.search_multi(query, top_k)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "mode": "multi_granularity", "results": self._simplify(results), "count": len(results)})

    def _search_filtered(self, query: str, top_k: int = 5,
                          file_type: str = "", min_quality: float = 0.0,
                          chunk_type: str = "") -> str:
        """元数据过滤检索：按文件类型/质量/块类型过滤后检索。
        例：只搜PPT文件（file_type="PPTX"）、只看表格（chunk_type="table"）。"""
        if not query or not query.strip():
            return _err("搜索问题不能为空")
        top_k = _clamp_top_k(top_k)
        kwargs = {}
        if file_type:
            kwargs["file_type"] = file_type
        if min_quality > 0:
            kwargs["min_quality"] = min_quality
        if chunk_type:
            kwargs["chunk_type"] = chunk_type
        results = self.pipeline.search_filtered(query, top_k, **kwargs)
        results = scan_retrieval_results(results)
        return _ok({"query": query, "mode": "filtered", "filters": kwargs, "results": self._simplify(results), "count": len(results)})

    def _search_graph(self, query: str) -> str:
        """图检索：从知识图谱中查找实体关系，用于多跳推理问题。
        需 pipeline 初始化时开启 use_knowledge_graph=True。"""
        if not query or not query.strip():
            return _err("查询不能为空")
        entities = [w for w in query.replace("，", ",").replace("、", ",").split(",") if w.strip()]
        if not entities:
            entities = [query[:10]]
        result = self.pipeline.search_graph(entities)
        if not result.get("entities"):
            return _ok({"message": "知识图谱未启用或未找到相关实体", "stats": self.pipeline.knowledge_graph_stats})
        return _ok({"entities": result["entities"], "relations": result["relations"][:10], "context": result["context"][:1500], "stats": result["stats"]})

    def _qa(self, question: str, top_k: int = 8) -> str:
        """财报问答：检索 → 去重 → 冲突检测 → 置信度评估 → 结构化答案 → 溯源。
        和 _search 的区别：_search 返回原始 chunk 列表，
        _qa 返回结构化答案（带置信度、来源引用、追问引导）。
        """
        if not question or not question.strip():
            return _err("问题不能为空")
        top_k = _clamp_top_k(top_k)
        try:
            from rag.answer_generator import rag_qa
            from core.llm import LLM
            result = rag_qa(question, self.pipeline, LLM(), top_k)
        except Exception as e:
            return _err(f"问答生成失败: {e}")
        return _ok({
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "followups": result["followups"],
            "conflicts": result.get("conflicts", []),
        })

    def _stats(self) -> str:
        return _ok({"text_chunks": self.pipeline.text_count, "images": self.pipeline.image_count})

    def _simplify(self, results: list[dict]) -> list[dict]:
        return [{
            "id": r.get("id", ""),
            "similarity": r.get("similarity", 0),
            "type": r.get("type", r.get("metadata", {}).get("chunk_type", "text")),
            "text": r["text"][:200],
            "citation": r.get("citation", ""),
            "page": r.get("metadata", {}).get("page"),
        } for r in results]
