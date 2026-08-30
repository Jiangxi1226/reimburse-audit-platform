import json
from pydantic import BaseModel


def extract_json(text: str) -> str:
    """从 LLM 回复中提取纯 JSON 字符串

    处理三种常见情况：
      1. 裸 JSON: {"key": "value"}
      2. 代码块包裹: ```json\n{"key": "value"}\n```
      3. 带前缀说明: 这是结果：\n{"key": "value"}
    """
    text = text.replace('```json', '').replace('```', '')
    text = text.strip()

    start = text.find('{')
    if start < 0:
        return text

    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text


def structured_output(
    llm,
    prompt: str,
    output_model: type[BaseModel] | None = None,
    temperature: float = 0.1,
    max_retries: int = 2,
    example_json: str = None,
    business_validator=None,
    on_fail: str = "empty",
) -> dict:
    """让 LLM 稳定输出 JSON，失败自动重试。

    已纳入分层严格 JSON（MODEL schema 约束 → GATE1 解析/结构 → GATE2 业务 → ACCEPT），
    内部复用 strict_json。成功返回 data；失败行为由 on_fail 决定：
      - "empty"（默认）：返回 {} —— 兼容旧调用点，不改变调用契约。
      - "raise"：抛 JsonStructuredError（携带 JsonResult，含 gate/reason/data），
        显式暴露失败态，供调用方降级或人工接管，绝不悄悄吞掉。

    Args:
        llm: LLM 实例
        prompt: 用户提示词（会自动追加 JSON 格式要求 + schema 约束）
        output_model: 可选 Pydantic 模型——传了就做 GATE1 结构校验
        temperature: 默认 0.1（极低，保证稳定性）
        max_retries: 解析/结构失败重试次数
        example_json: 可选示例 JSON，帮助 LLM 理解格式
        business_validator: 可选业务校验器（GATE2），非法不重试
        on_fail: 失败策略 "empty" | "raise"

    Returns:
        解析后的 dict

    用法：
        result = structured_output(llm, "分析曼城vs阿森纳的战术",
                                   example_json='{"formation": "4-3-3"}')
    """
    res = strict_json(
        llm, prompt,
        output_model=output_model,
        business_validator=business_validator,
        temperature=temperature,
        max_retries=max_retries,
        example_json=example_json,
    )
    if res.ok:
        return res.data
    if on_fail == "raise":
        raise JsonStructuredError(res)
    return {}


def structured_chat(
    llm,
    messages: list[dict],
    temperature: float = 0.1,
    max_retries: int = 2,
    business_validator=None,
    on_fail: str = "empty",
) -> dict:
    """和 structured_output 一样，但接受完整 messages（带对话历史的场景）。

    同样纳入分层严格 JSON，复用 strict_json（失败行为同 structured_output）。
    """
    prompt = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
    )
    res = strict_json(
        llm, prompt,
        business_validator=business_validator,
        temperature=temperature,
        max_retries=max_retries,
    )
    if res.ok:
        return res.data
    if on_fail == "raise":
        raise JsonStructuredError(res)
    return {}


# ============================================================
# 分层严格 JSON（图示方法论落地）
#   MODEL BOUNDARY  : Schema 约束生成（required / enum / type 注入提示）
#   APPLICATION     : GATE1 解析 + 结构校验（可重试）
#                     GATE2 业务规则校验（真实ID/权限/值域，业务非法不重试）
#                     ACCEPT 可用对象
#   FAILURE CONTROL : 有限重试 → 显式失败（降级/人工），绝不悄悄 {} 后继续
# ============================================================

from dataclasses import dataclass, field
from enum import Enum


class JsonGate(str, Enum):
    MODEL = "model"          # 模型生成阶段（未到校验）
    PARSE = "parse"          # GATE1-解析（是否合法 JSON）
    STRUCTURE = "structure"  # GATE1-结构（字段/类型/required/enum）
    BUSINESS = "business"    # GATE2-业务规则（真实ID/权限/值域/范围）
    FAIL = "fail"            # 超出重试上限
    ACCEPT = "accept"        # 通过——可用对象进入业务系统


class JsonStructuredError(Exception):
    """结构化输出未通过校验时抛出，携带 JsonResult 供调用方分级处理。

    由 structured_output/structured_chat 在 on_fail="raise" 时抛出，
    替代旧的「静默返回 {}」——失败态显式化，交调用方降级或人工接管。
    """
    def __init__(self, result: "JsonResult"):
        self.result = result
        super().__init__(f"结构化输出失败[gate={result.gate}] {result.reason}")


@dataclass
class JsonResult:
    ok: bool
    gate: str                 # JsonGate 的值
    data: dict | None = None
    reason: str = ""
    raw: str = ""             # 模型原始输出（供人工/降级排查）
    attempts: int = 0
    degraded: bool = False    # 是否降级（失败态，而非被接受）
    errors: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.ok and self.gate == JsonGate.ACCEPT.value


def _schema_constraints(output_model) -> str:
    """MODEL BOUNDARY：把 pydantic 模型的 required / enum / type 转成约束说明，
    注入提示，让模型在生成期就受 schema 约束（而非生成后再判错）。"""
    if not output_model:
        return ""
    lines = ["输出必须是符合以下 schema 的 JSON 对象："]
    for name, info in output_model.model_fields.items():
        ann = info.annotation
        anno = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        required = "required" if info.is_required() else "optional"
        seg = [f"{name} ({required}, {anno})"]
        if hasattr(ann, "__members__"):  # Enum
            seg.append("enum=" + "|".join(str(m) for m in ann.__members__))
        lines.append("  - " + " ".join(seg))
    return "\n".join(lines)


def _parse_gate(raw: str) -> tuple[dict | None, str]:
    """GATE1-解析：从模型输出中解出 JSON。返回 (data, 错误信息)。"""
    try:
        return json.loads(extract_json(raw)), ""
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败：{e}"


def _structure_gate(data: dict, output_model) -> str | None:
    """GATE1-结构：用 pydantic 校验字段/类型/required/enum。返回错误信息或 None。"""
    if not output_model:
        return None  # 未声明 schema，结构放行
    try:
        output_model.model_validate(data)
        return None
    except Exception as e:
        return f"结构校验失败：{e}"


def _business_gate(data: dict, business_validator) -> str | None:
    """GATE2-业务：校验值域/真实ID/权限/范围。业务非法返回错误信息，永不重试。

    business_validator 返回约定：
      - None  → 通过(不校验/无约束)
      - True  → 通过
      - False → 失败(生成通用失败原因)
      - (ok, reason) → 二元组，ok=False 时 reason 为失败原因
    注意 None 是「通过」，绝不能与 False 混同——否则所有返回 None 的校验器都会被误判失败。
    (与 medical_assistant 版本保持一致，修复此前 `bool(None)=False` 误判的缺陷。)
    """
    if not business_validator:
        return None
    try:
        result = business_validator(data)
        if isinstance(result, tuple):
            ok, reason = result
            return None if ok else (reason or "业务规则校验未通过")
        # 非 tuple：None / True 表示通过；只有显式 False 才算失败
        return None if result is not False else "业务规则校验未通过"
    except Exception as e:
        return f"业务校验异常：{e}"


def strict_json(
    llm,
    prompt: str,
    output_model: type[BaseModel] | None = None,
    business_validator=None,
    temperature: float = 0.1,
    max_retries: int = 2,
    example_json: str = None,
) -> JsonResult:
    """分层严格 JSON 生成：MODEL → GATE1(解析+结构) → GATE2(业务) → ACCEPT。

    - 结构失败（解析/字段类型）可重试：把错误回喂模型改正。
    - 业务失败不重试：业务非法是事实，重试是浪费，直接显式失败交调用方降级/人工。
    - 失败一律返回 JsonResult(ok=False, gate, reason, degraded=True)，绝不悄悄 {}。
    """
    sys_prompt = "你是一个 JSON 输出器。只输出合法 JSON 对象，不要输出任何其他文字、解释或 markdown 标记。"
    if example_json:
        sys_prompt += f"\n\n输出必须严格遵循以下格式示例：\n{example_json}"
    schema_block = _schema_constraints(output_model)
    if schema_block:
        sys_prompt += "\n\n" + schema_block
    full_prompt = f"{sys_prompt}\n\n{prompt}"

    result = JsonResult(ok=False, gate=JsonGate.MODEL.value, attempts=0)
    last_err = ""
    gate_after = JsonGate.FAIL.value

    for attempt in range(max_retries + 1):
        result.attempts = attempt + 1
        # 模型调用（瞬时错误可整轮重试）
        try:
            raw = llm.chat([{"role": "user", "content": full_prompt}], temperature=temperature)
            result.raw = raw
        except Exception as e:
            last_err = f"模型调用失败：{e}"
            if attempt < max_retries:
                continue
            result.gate = JsonGate.FAIL.value
            result.reason = last_err
            break

        # GATE1-解析
        data, perr = _parse_gate(raw)
        if data is None:
            last_err = perr
            gate_after = JsonGate.PARSE.value
            if attempt < max_retries:
                full_prompt = (sys_prompt + f"\n\n上次输出不是合法 JSON，错误：{perr}\n"
                               f"请修正后重新输出 JSON。\n\n{prompt}")
                continue
            result.gate = gate_after
            result.reason = perr
            break

        # GATE1-结构
        serr = _structure_gate(data, output_model)
        if serr:
            last_err = serr
            gate_after = JsonGate.STRUCTURE.value
            if attempt < max_retries:
                full_prompt = (sys_prompt + f"\n\n上次输出未通过结构校验：{serr}\n"
                               f"请按 schema 修正后重新输出。\n\n{prompt}")
                continue
            result.gate = gate_after
            result.reason = serr
            break

        # 结构通过 → 先 build 规范化 pydantic 对象(补齐缺省 default)再交给业务校验。
        # 业务校验器因此拿到"带默认值的可用对象"，字段缺省不会 KeyError(与项目二对齐)。
        # result.data 仍返回原始 dict，保持本项目既有调用契约(get 取值型调用方友好)。
        obj = output_model.model_validate(data) if output_model else data

        # GATE2-业务（不重试）
        berr = _business_gate(obj, business_validator)
        if berr:
            result.ok = False
            result.gate = JsonGate.BUSINESS.value
            result.reason = berr
            result.data = data
            result.degraded = True
            return result

        result.ok = True
        result.gate = JsonGate.ACCEPT.value
        result.data = data
        return result

    # 超过重试上限 —— 显式失败（降级/人工接管），不悄悄返回 {}
    result.ok = False
    result.gate = gate_after
    result.reason = result.reason or last_err or "超出重试上限"
    result.degraded = True
    result.errors.append(result.reason) if result.reason not in result.errors else None
    return result
