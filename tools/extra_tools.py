import json, re
from core.base import Tool
from core.llm import LLM


def _ok(data) -> str:
    return json.dumps({"ok": True, "data": data, "error": ""}, ensure_ascii=False)

def _err(msg) -> str:
    return json.dumps({"ok": False, "data": None, "error": msg}, ensure_ascii=False)


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "计算器：计算数学表达式。返回 JSON。")

    def _calc(self, expression: str) -> str:
        if not re.match(r'^[\d\s+\-*/().%**]+$', expression):
            return _err("表达式包含不安全字符")
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return _ok({"expression": expression, "result": result})
        except Exception as e:
            return _err(str(e))


class ImageAnalysisTool(Tool):
    def __init__(self, llm: LLM = None):
        super().__init__("image_analysis", "图片分析：DeepSeek 多模态识图，返回 JSON。")
        self._llm = llm

    def _analyze(self, image_path: str, question: str = "请描述这张图片的内容") -> str:
        import os
        if not os.path.exists(image_path):
            return _err(f"图片不存在: {image_path}")
        try:
            llm = self._llm or LLM()
            img_b64 = llm.encode_image(image_path)
            answer = llm.chat_with_image(question, img_b64)
            return _ok({"image_path": image_path, "question": question, "analysis": answer})
        except Exception as e:
            return _err(str(e))
