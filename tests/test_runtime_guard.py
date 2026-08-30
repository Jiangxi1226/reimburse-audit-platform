"""Runtime 权限闸门测试 —— 覆盖架构图的七道闸门 + 三档处置。

不依赖 LLM / 外部服务，只测 RuntimeGuard 判定与 Registry 集成。
运行：pytest tests/test_runtime_guard.py -v
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runtime_guard import (
    RuntimeGuard, AccessContext, ToolPolicy, GuardDecision,
    RiskLevel, Verdict, Redactor,
)
from core.registry import ToolRegistry
from core.base import Tool
from core.guard_policies import build_guard, make_ctx


# ---------------------------------------------------------------------------
# 测试替身：一个假工具，记录调用次数，便于验证闸门是否真的"放行/拦截"
# ---------------------------------------------------------------------------
class FakeTool(Tool):
    def __init__(self):
        super().__init__("fake", "测试工具")
        self.writes = 0
        self.read_result = "持仓金额 ¥12000，手机 13812345678，身份证 110101199001011234"

    def _write(self, text: str = "") -> str:
        self.writes += 1
        return json.dumps({"ok": True, "data": text}, ensure_ascii=False)

    def _read(self) -> str:
        return json.dumps({"ok": True, "data": self.read_result}, ensure_ascii=False)


def _policy_for(guard, tool, action):
    return guard.policy(tool, action)


# ---------------------------------------------------------------------------
# ① 工具白名单
# ---------------------------------------------------------------------------
def test_whitelist_deny():
    g = RuntimeGuard(policies=[])
    d = g.gate("unknown", "op", {}, make_ctx("admin"))
    assert d.verdict == Verdict.DENY
    assert "白名单" in d.reason


# ---------------------------------------------------------------------------
# ② 身份缺失
# ---------------------------------------------------------------------------
def test_identity_deny():
    g = build_guard()
    d = g.gate("rag", "search", {"query": "x"}, None)
    assert d.verdict == Verdict.DENY
    assert "身份" in d.reason


# ---------------------------------------------------------------------------
# ③ Role / Scope：guest 调 calculator 被拒，analyst 放行
# ---------------------------------------------------------------------------
def test_role_deny_guest_calc():
    g = build_guard()
    d = g.gate("calculator", "calc", {"expression": "1+1"}, make_ctx("guest"))
    assert d.verdict == Verdict.DENY
    assert "无权调用" in d.reason

def test_role_allow_analyst_calc():
    g = build_guard()
    d = g.gate("calculator", "calc", {"expression": "1+1"}, make_ctx("analyst"))
    assert d.verdict == Verdict.ALLOW
    assert d.risk == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# ④ 参数 Schema 校验：必填参数缺失 → 拒绝
# ---------------------------------------------------------------------------
def test_args_required_deny():
    g = build_guard()
    d = g.gate("rag", "add", {}, make_ctx("admin"))
    assert d.verdict == Verdict.DENY
    assert "缺少必填参数" in d.reason

def test_args_type_deny():
    g = build_guard()
    # calculator 的 expression 必须是 str
    d = g.gate("calculator", "calc", {"expression": 123}, make_ctx("analyst"))
    assert d.verdict == Verdict.DENY


# ---------------------------------------------------------------------------
# ⑤⑥ 风险分级 + 审批：高险写操作需确认 → 默认 deny 策略拦截
# ---------------------------------------------------------------------------
def test_high_risk_confirm_deny_by_default():
    g = build_guard(on_confirm="deny")
    # 使用项目根内的合法绝对路径，验证"高险写操作默认需确认"(而非被路径越界拦下)
    import os
    in_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "a.pdf")
    d = g.gate("rag", "add_file", {"file_path": in_root}, make_ctx("admin"))
    assert d.verdict == Verdict.CONFIRM
    assert d.confirm_token

def test_path_escape_for_write_action():
    # 写操作传入项目根之外的绝对路径 → 应被路径越界校验 DENY(而不是走 confirm)
    g = build_guard(on_confirm="deny")
    d = g.gate("rag", "add_file", {"file_path": "/etc/passwd"}, make_ctx("admin"))
    assert d.verdict == Verdict.DENY

def test_confirm_approve_flows():
    # on_confirm='approve' 时，高险写操作走确认后放行
    g = build_guard(on_confirm="approve")
    calls = []
    g.audit_log = lambda **kw: calls.append(kw)
    d, res = g.execute("rag", "add", {"text": "test"}, make_ctx("admin"),
                       dispatcher=lambda tool, action, **kw: '{"ok": true}')
    assert d.verdict == Verdict.ALLOW
    assert "OK" in res or "ok" in res


# ---------------------------------------------------------------------------
# 路径越界（图④ 参数校验的扩展）
# ---------------------------------------------------------------------------
def test_path_escape_deny():
    g = RuntimeGuard(policies=[ToolPolicy(
        tool="img", action="analyze", risk=RiskLevel.MEDIUM,
        path_root="/safe/root", required_args=["image_path"])])
    d = g.gate("img", "analyze", {"image_path": "/etc/passwd"}, make_ctx("admin"))
    assert d.verdict == Verdict.DENY
    assert "路径越界" in d.reason


# ---------------------------------------------------------------------------
# ⑦ 预算：超限拒绝
# ---------------------------------------------------------------------------
def test_budget_exceeded():
    g = RuntimeGuard(policies=[ToolPolicy(
        tool="r", action="x", risk=RiskLevel.LOW, budget=1)])
    ctx = make_ctx("admin")
    assert g.gate("r", "x", {}, ctx).verdict == Verdict.ALLOW
    assert g.gate("r", "x", {}, ctx).verdict == Verdict.DENY


# ---------------------------------------------------------------------------
# 数据脱敏（图「数据脱敏——最小化返回」）
# ---------------------------------------------------------------------------
def test_redactor_masks():
    text = "金额 ¥12000.5，手机 13812345678，身份证 110101199001011234"
    masked, changed = Redactor.mask_text(text)
    assert changed
    # 打码规则：保留前3后4，中间以 **** 遮住（金额整段打码）
    assert "¥**" in masked and "¥12000.5" not in masked
    assert "138****5678" in masked
    assert "110****1234" in masked
    assert "3456" not in masked  # 证件号中间段不应露出


# ---------------------------------------------------------------------------
# 幂等：mutable 相同参数只执行一次（图⑦）
# ---------------------------------------------------------------------------
def test_idempotent_write():
    tool = FakeTool()
    g = RuntimeGuard(policies=[
        ToolPolicy(tool="fake", action="write", risk=RiskLevel.HIGH,
                   mutable=True, roles={"admin"})],
        on_confirm="approve")  # 高险写操作默认需人工确认，此处预置为已批准
    ctx = make_ctx("admin")
    _, r1 = g.execute("fake", "write", {"text": "dup"}, ctx,
                      dispatcher=lambda t, a, **kw: tool._write(**kw))
    _, r2 = g.execute("fake", "write", {"text": "dup"}, ctx,
                      dispatcher=lambda t, a, **kw: tool._write(**kw))
    assert tool.writes == 1, "幂等应保证相同参数只落一次"
    assert r1 == r2


# ---------------------------------------------------------------------------
# 审计日志（图「审计日志——完整记录操作与结果」）
# ---------------------------------------------------------------------------
def test_audit_recorded():
    g = RuntimeGuard(policies=[ToolPolicy(tool="fake", action="read", risk=RiskLevel.LOW)])
    audit = []
    g.audit_log = lambda **kw: audit.append(kw)
    ctx = make_ctx("analyst", user_id="u1")
    g.execute("fake", "read", {}, ctx, dispatcher=lambda t, a, **kw: "ok")
    assert len(audit) == 1
    rec = audit[0]
    assert rec["tool"] == "fake" and rec["verdict"] == "allow" and rec["ok"] is True


# ---------------------------------------------------------------------------
# Registry 集成：执行权归 Runtime，放行才真正调用后端
# ---------------------------------------------------------------------------
def test_registry_guard_integration():
    tool = FakeTool()
    reg = ToolRegistry()
    reg.register(tool)
    audit = []
    g = RuntimeGuard(policies=[
        ToolPolicy(tool="fake", action="write", risk=RiskLevel.HIGH,
                   roles={"admin"}, mutable=True),
        ToolPolicy(tool="fake", action="read", risk=RiskLevel.LOW, redact_text=True)],
        on_confirm="approve")  # 写操作预置已批准，专注验证"角色拦截 + 放行"两分支
    g.audit_log = lambda **kw: audit.append(kw)
    reg.set_guard(g)

    # guest 调用 write → 被拦，后端不执行
    res = reg.execute("fake", "write", ctx=make_ctx("guest"), text="x")
    assert tool.writes == 0
    assert "权限拒绝" in res

    # admin 调用 write → 放行
    res = reg.execute("fake", "write", ctx=make_ctx("admin"), text="hi")
    assert tool.writes == 1
    assert "hi" in res

    # 默认基线兜底：未显式传 ctx 时按入口基线
    reg.set_ctx_default(make_ctx("admin"))
    reg.execute("fake", "write", text="zzz")
    assert tool.writes == 2


# ---------------------------------------------------------------------------
# 回滚策略（图「回滚策略——出现异常可回滚/补偿」）
# ---------------------------------------------------------------------------
def test_rollback_registered_and_triggered():
    rolled = []
    g = RuntimeGuard(policies=[ToolPolicy(
        tool="fake", action="write", risk=RiskLevel.HIGH,
        roles={"admin"}, mutable=True,
        rollback=lambda **a: rolled.append(a))],
        on_confirm="approve")  # 高险写操作预置批准以走到 rollback 登记
    ctx = make_ctx("admin")
    g.execute("fake", "write", {"text": "x"}, ctx,
              dispatcher=lambda t, a, **kw: json.dumps({"ok": True}))
    keys = g.rollback_pending()
    assert len(keys) == 1
    assert g.rollback(keys[0]) is True
    assert rolled and rolled[0]["text"] == "x"
    assert g.rollback_pending() == []
