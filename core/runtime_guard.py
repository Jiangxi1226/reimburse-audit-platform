"""Runtime 权限闸门 —— 模型只有意图，执行权在 Runtime 手里。

对应架构图：
    模型输出 Tool Call —— 只是意图，不是已授权命令
    → Runtime 权限闸门：
      ① 工具白名单    ② 用户身份/租户   ③ Scope / Role / Policy
      ④ 参数 Schema 校验  ⑤ 风险分级    ⑥ 审批 / 二次确认    ⑦ 限流 / 预算 / 幂等
    → 三档处置：允许执行 / 需要确认(人工确认后执行) / 拒绝·降级
    → 后端再次校验 + 数据脱敏(最小化返回) + 审计日志 + 回滚策略

设计取舍（面试可讲）：
- 采用「默认拒绝」白名单思想：未注册进策略表的 (tool, action) 一律 DENY，
  而不是默认放行。宁可断，不可错。
- ALLOW / CONFIRM / DENY 三档是显式枚举，避免 boolean 二值挡不住"需要审批"
  这类中间态。
- 风险分级用「规则 + 策略」而非纯 LLM 判断：可解释、可测试、不依赖模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 1. 枚举与数据类
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Verdict(str, Enum):
    """三档处置结果。"""
    ALLOW = "allow"        # 允许执行
    CONFIRM = "confirm"    # 需要确认（人工确认后执行）
    DENY = "deny"          # 拒绝 / 降级


@dataclass
class AccessContext:
    """发起本次工具调用的用户上下文。

    - user_id / tenant_id：用户身份与租户隔离（图②）
    - role：admin / analyst / user / guest（图③ Scope/Role/Policy）
    - scopes：细分权限点，可叠加细化到"某功能是否可调"
    """
    user_id: str
    tenant_id: str = ""
    role: str = "user"
    scopes: set[str] = field(default_factory=set)

    def to_json(self) -> str:
        return json.dumps(
            {"user_id": self.user_id, "tenant_id": self.tenant_id,
             "role": self.role, "scopes": sorted(self.scopes)},
            ensure_ascii=False)


@dataclass
class GuardDecision:
    """一次 gate 的结论。"""
    verdict: Verdict
    risk: RiskLevel
    reason: str = ""
    policy_name: str = ""
    confirm_token: str = ""
    masked: bool = False

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW

    def to_str(self) -> str:
        return f"[Guard:{self.verdict.value}/{self.risk.value}] {self.reason}"


# ---------------------------------------------------------------------------
# 2. 工具策略声明（一条 = 一个 (tool, action) 的权限规则）
# ---------------------------------------------------------------------------

@dataclass
class ToolPolicy:
    """对单个工具操作的权限规则。未注册的操作 = 默认拒绝。"""
    tool: str
    action: str
    risk: RiskLevel = RiskLevel.LOW
    roles: set[str] = field(default_factory=set)      # 空 = 任意角色可调
    scopes: set[str] = field(default_factory=set)     # 空 = 任意 scope 可调
    needs_confirm: bool = False                        # 图⑥ 审批/二次确认
    mutable: bool = False                              # 写操作（需幂等 + 回滚）
    budget: int = 0                                    # 图⑦ 每用户每会话调用上限（0=不限）
    required_args: list[str] = field(default_factory=list)   # 图④ 必填参数
    arg_types: dict = field(default_factory=dict)            # 图④ 参数类型约束
    path_root: str = ""                                  # 文件/图片路径根目录（防越界）
    redact_fields: list[str] = field(default_factory=list)   # 返回结果脱敏字段
    redact_text: bool = False                            # 整段文本脱敏（金额/证件号等）
    rollback: callable = None                            # 图 回滚策略 回调

    @property
    def key(self) -> str:
        return f"{self.tool}.{self.action}"


# ---------------------------------------------------------------------------
# 3. 数据脱敏 —— 最小化返回（图下半部分）
# ---------------------------------------------------------------------------

# 金额：¥/￥ 后跟数字
_RE_AMOUNT = re.compile(r"[¥￥]\s?\d[\d,]*(?:\.\d+)?")
# 身份证：18 位（末位可 x）
_RE_ID = re.compile(r"\b\d{17}[\dXx]\b")
# 手机号：11 位大陆手机
_RE_PHONE = re.compile(r"\b1[3-9]\d{9}\b")
# 银行卡：16~19 位纯数字
_RE_CARD = re.compile(r"\b\d{16,19}\b")


class Redactor:
    """对工具返回结果做最小化脱敏（图「数据脱敏——最小化返回」）。

    只打码、不删除；保留前后几位便于核对。作为"后端再次校验"的一层兜底。
    """

    @staticmethod
    def mask_text(text: str) -> tuple[str, bool]:
        """对一段文本做脱敏，返回 (脱敏后, 是否有改动)。"""
        orig = text
        text = _RE_AMOUNT.sub(lambda m: "¥**", text)
        text = _RE_ID.sub(lambda m: _mask(m.group(0), 3, 4), text)
        text = _RE_PHONE.sub(lambda m: _mask(m.group(0), 3, 4), text)
        text = _RE_CARD.sub(lambda m: _mask(m.group(0), 4, 4), text)
        return text, text != orig

    @staticmethod
    def mask_fields(obj, fields: list[str]):
        """对 dict 中指定字段值打码（金额/证件号/手机号/银行卡全套正则）。"""
        if not isinstance(obj, dict):
            return obj
        for f in fields:
            if f in obj and isinstance(obj[f], str):
                obj[f] = Redactor.mask_text(obj[f])[0]
        return obj

    @staticmethod
    def mask_obj(obj):
        """递归最小化脱敏任意 dict/list——审计入库前用。

        不再依赖"字段名恰好叫 text/query"：凡是字符串值，命中金额/证件号/手机号/
        银行卡正则就脱敏，避免敏感字段换了名字（phone、id_card…）就漏脱。
        返回新对象，不原地修改入参。
        """
        if isinstance(obj, dict):
            return {k: Redactor.mask_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Redactor.mask_obj(v) for v in obj]
        if isinstance(obj, str):
            return Redactor.mask_text(obj)[0]
        return obj


def _mask(s: str, head: int, tail: int) -> str:
    return s[:head] + "****" + s[-tail:]


# ---------------------------------------------------------------------------
# 4. RuntimeGuard —— 七道闸门 + 三档处置
# ---------------------------------------------------------------------------

class RuntimeGuard:
    """唯一入口：gate() 判定 + execute() 分档执行。

    任何工具调用必须先过 gate —— 模型生成的 tool_name / action_name / args
    都视为不可信的意图，能否执行、以何种风险执行，均由本 Guard 决定。
    """

    # 内部状态上限：这些都是进程内字典，若只写不删会随运行时间无限增长（内存泄漏）。
    # 保底做法是给每个 dict 设硬上限 + 对确认令牌设有效期，超出按最早登记顺序裁剪。
    _MAX_IDEM = 5000          # 幂等缓存条数上限
    _MAX_CONFIRM = 1000       # 待确认队列上限
    _CONFIRM_TTL = 15 * 60    # 待确认令牌有效期（秒）——过期令牌不可再用
    _MAX_ROLLBACK = 1000      # 回滚登记条数上限
    _MAX_BUDGET_KEYS = 10000  # 预算记录键数上限

    def __init__(self, policies: list[ToolPolicy] | None = None,
                 on_confirm: str = "deny", persist: bool = False):
        self._policies: dict[str, ToolPolicy] = {}
        for p in (policies or []):
            self._policies[p.key] = p
        # ① 白名单思想：register_policy 之外的操作 = 默认拒绝（不在这里开清单式放行）
        self._confirm_waiting: dict[str, dict] = {}   # confirm_token -> 待执行信息
        self._budget_spent: dict[str, list] = {}       # key:(user,tool.action) -> 时间戳/次数
        self._idem: dict[str, str] = {}                # 幂等 key -> 上次结果
        self._rollback: dict[str, callable] = {}       # 登记的回滚回调
        self.on_confirm = on_confirm                   # confirm 命中后 auto 行为: deny/approve
        self.audit_log = None                          # 由 registry 注入 db.log_audit
        # 持久化：幂等/预算落 SQLite，跨进程重启依然有效（默认关闭，测试/无 DB 场景不受影响）。
        self._db = self._load_db() if persist else None
        self._restore_state()

    # ---- 持久化：加载 / 恢复 ----
    @staticmethod
    def _load_db():
        """惰性获取 core.db；不可用时返回 None（持久化静默降级为纯内存）。"""
        try:
            from core import db as _db
            return _db
        except Exception:
            return None

    def _restore_state(self) -> None:
        """进程启动时把上次的幂等记录与预算窗口读回内存。失败不阻断启动。"""
        if self._db is None:
            return
        try:
            self._idem.update(self._db.guard_idem_load(self._MAX_IDEM))
        except Exception:
            pass
        try:
            now = time.time()
            restored = self._db.guard_budget_load(now - 24 * 3600)
            for k, ts in restored.items():
                recent = [t for t in ts if now - t < 24 * 3600]
                if recent:
                    self._budget_spent[k] = recent
        except Exception:
            pass

    # ---- 策略增删 ----
    def register(self, policy: ToolPolicy):
        self._policies[policy.key] = policy

    def policy(self, tool: str, action: str) -> ToolPolicy | None:
        p = self._policies.get(f"{tool}.{action}")
        if p is not None:
            return p
        # 组合 function 名兜底：FC/ReAct 可能直接引用 schema 的 "{tool}_{action}" 名（如
        # "rag_search.advanced" 或 tool="rag_search"）。此时 tool 本身不在策略表，按最后一个
        # 下划线拆回 "rag"+"advanced"。注意真实工具名可含下划线（image_analysis），但它在策略表
        # 里有完整 "{tool}.{action}" 条目，上一行已命中直接返回，不会落到这里，故不会误拆。
        if "_" in tool:
            cand_tool, cand_action = tool.rsplit("_", 1)
            if cand_action == action:
                return self._policies.get(f"{cand_tool}.{action}")
        return None

    # ---- 图① 工具白名单 ----
    def _is_whitelisted(self, tool: str, action: str) -> ToolPolicy | None:
        return self.policy(tool, action)

    # ---- 图② 用户身份 / 租户 ----
    def _check_identity(self, ctx: AccessContext | None) -> GuardDecision | None:
        if ctx is None or not ctx.user_id:
            return GuardDecision(Verdict.DENY, RiskLevel.HIGH,
                                 reason="缺少用户身份(user_id)——未授权调用")

    # ---- 图③ Scope / Role / Policy ----
    def _check_role_scope(self, p: ToolPolicy, ctx: AccessContext) -> GuardDecision | None:
        if p.roles and ctx.role not in p.roles:
            return GuardDecision(Verdict.DENY, p.risk,
                                 reason=f"角色 {ctx.role} 无权调用 {p.key}（需 {sorted(p.roles)}）")
        if p.scopes and not (p.scopes & ctx.scopes):
            return GuardDecision(Verdict.DENY, p.risk,
                                 reason=f"缺少 {p.key} 所需 scope {sorted(p.scopes)}")

    # ---- 图④ 参数 Schema 校验 ----
    def _check_args(self, p: ToolPolicy, tool: str, action: str, args: dict) -> GuardDecision | None:
        for req in p.required_args:
            if not args.get(req):
                return GuardDecision(Verdict.DENY, p.risk,
                                     reason=f"参数校验失败：缺少必填参数 '{req}'")
        for name, typ in p.arg_types.items():
            v = args.get(name)
            if v is not None and not isinstance(v, typ):
                return GuardDecision(Verdict.DENY, p.risk,
                                     reason=f"参数校验失败：'{name}' 应为 {typ.__name__}，实为 {type(v).__name__}")
        # 路径越界校验（参数里所有 *_path 与 receipt_file 皆受 `path_root` 约束）
        for k, v in args.items():
            if k in ("_path",) or k.endswith("_path") or k == "receipt_file":
                if isinstance(v, str) and v:
                    if p.path_root and not self._within_root(v, p.path_root):
                        return GuardDecision(Verdict.DENY, p.risk,
                                             reason=f"路径越界：'{k}' 不在允许的根目录 {p.path_root} 内")

    @staticmethod
    def _within_root(path: str, root: str) -> bool:
        try:
            return os.path.realpath(path).startswith(os.path.realpath(root) + os.sep)
        except Exception:
            return False

    # ---- 图⑤⑥ 风险分级 + 审批 ----
    def _needs_confirm(self, p: ToolPolicy) -> bool:
        # 高险写操作默认需确认；策略也可显式声明
        return p.needs_confirm or (p.risk == RiskLevel.HIGH and p.mutable)

    # ---- 图⑦ 限流 / 预算 / 幂等 ----
    def _check_budget(self, p: ToolPolicy, ctx: AccessContext, tool: str, action: str) -> GuardDecision | None:
        if p.budget <= 0:
            return None
        key = f"{ctx.user_id}:{p.key}"
        now = time.time()
        recent = [t for t in self._budget_spent.get(key, []) if now - t < 24 * 3600]
        recent.append(now)
        self._budget_spent[key] = recent
        # 落库：重启后预算仍能延续（失败静默降级为内存态，不阻断业务）
        if self._db is not None:
            try:
                self._db.guard_budget_add(key, now, now - 24 * 3600)
            except Exception:
                pass
        # 键数按 (用户×工具) 增长，超限时清理"全部时间戳都已过期"的空键，防内存膨胀
        if len(self._budget_spent) > self._MAX_BUDGET_KEYS:
            for k in [k for k, ts in self._budget_spent.items()
                      if not [t for t in ts if now - t < 24 * 3600]]:
                self._budget_spent.pop(k, None)
        if len(recent) > p.budget:
            return GuardDecision(Verdict.DENY, p.risk,
                                 reason=f"超出调用预算（{p.key} 每用户每天上限 {p.budget} 次）")

    # ---- 主判定：七道闸门依次放行 ----
    def gate(self, tool: str, action: str, args: dict,
             ctx: AccessContext, skip_budget: bool = False) -> GuardDecision:
        # ① 白名单
        p = self._is_whitelisted(tool, action)
        if p is None:
            return GuardDecision(Verdict.DENY, RiskLevel.HIGH,
                                 reason=f"工具的 '{tool}.{action}' 不在白名单，默认拒绝",
                                 policy_name="whitelist")
        # ② 身份
        d = self._check_identity(ctx)
        if d:
            return d
        # ③ 角色 / scope
        d = self._check_role_scope(p, ctx)
        if d:
            return d
        # ④ 参数 schema
        d = self._check_args(p, tool, action, args)
        if d:
            return d
        # ⑦ 预算（先于可执行判定，防刷）
        # skip_budget：幂等重放不消耗预算——它没有真正执行后端，只是回放上次结果。
        # 其余闸门（白名单/身份/角色/参数）仍照常校验，权限不会被绕过。
        if not skip_budget:
            d = self._check_budget(p, ctx, tool, action)
            if d:
                return d
        # ⑤⑥ 风险分级 + 审批 → 决定 ALLOW 还是 CONFIRM
        if self._needs_confirm(p):
            token = self._issue_confirm(p, tool, action, args, ctx)
            return GuardDecision(Verdict.CONFIRM, p.risk,
                                 reason=f"高险操作 {p.key} 需人工确认后执行",
                                 policy_name=p.key, confirm_token=token)
        return GuardDecision(Verdict.ALLOW, p.risk, reason=f"通过闸门 {p.key}", policy_name=p.key)

    def _sweep_confirms(self) -> None:
        """清理过期的待确认令牌——令牌有有效期，且防止队列无限增长。"""
        now = time.time()
        for k in [k for k, v in self._confirm_waiting.items()
                  if now - v.get("ts", 0) > self._CONFIRM_TTL]:
            self._confirm_waiting.pop(k, None)
        while len(self._confirm_waiting) > self._MAX_CONFIRM:
            self._confirm_waiting.pop(next(iter(self._confirm_waiting)), None)

    def _issue_confirm(self, p: ToolPolicy, tool, action, args, ctx) -> str:
        self._sweep_confirms()
        token = hashlib.sha256(f"{ctx.user_id}:{p.key}:{time.time_ns()}".encode()).hexdigest()[:16]
        self._confirm_waiting[token] = {"tool": tool, "action": action,
                                        "args": args, "ctx": ctx, "ts": time.time()}
        return token

    # ---- 人工审批（图⑥ 二次确认）：approve/pending 后重放 ----
    def confirm_pending(self) -> list[dict]:
        self._sweep_confirms()   # 先清过期，列表里不再出现已失效的令牌
        return [{"token": k, "tool": v["tool"], "action": v["action"],
                 "args": v["args"], "user": v["ctx"].user_id, "ts": v["ts"]}
                for k, v in self._confirm_waiting.items()]

    def resolve_confirm(self, token: str, approved: bool) -> dict:
        """审批结果：approved 时返回可执行的 (tool, action, args, ctx)，否则清空。"""
        item = self._confirm_waiting.pop(token, None)
        if item is None:
            return {"ok": False, "reason": "确认令牌无效或已过期"}
        if not approved:
            return {"ok": False, "reason": "用户拒绝该操作"}
        return {"ok": True, **item}

    # ---- 幂等：相同参数重复提交只执行一次 ----
    def _idem_key(self, ctx: AccessContext, tool, action, args) -> str:
        return hashlib.sha256(
            json.dumps([ctx.user_id, tool, action, args], ensure_ascii=False, sort_keys=True,
                       default=str).encode()).hexdigest()[:24]

    # ---- 统一执行入口：三档处置 ----
    def execute(self, tool: str, action: str, args: dict, ctx: AccessContext,
                dispatcher) -> tuple[GuardDecision, str]:
        """dispatcher(tool, action, **args) -> str 负责真实调用后端工具。

        返回 (decision, result)。DENY / CONFIRM 不触发 dispatcher。
        CONFIRM 每次：on_confirm='deny' 默认拒绝（安全），'approve' 直接放行（演示/测试）。
        """
        # 先算幂等键、判断是否"重放"：重放时跳过预算（见 gate 的 skip_budget 说明）。
        policy = self.policy(tool, action)
        idem_key = self._idem_key(ctx, tool, action, args) if (policy and policy.mutable) else None
        replay = bool(idem_key and idem_key in self._idem)

        d = self.gate(tool, action, args, ctx, skip_budget=replay)

        if d.verdict == Verdict.DENY:
            self._audit(ctx, tool, action, args, d, ok=False, result=d.reason)
            return d, f"❌ 权限拒绝：{d.reason}"

        if d.verdict == Verdict.CONFIRM:
            # 无人工审批 UI 时，按 on_confirm 策略决定
            if self.on_confirm != "approve":
                self._audit(ctx, tool, action, args, d, ok=False,
                            result="需人工确认，当前策略为拒绝")
                return d, (f"⏸ 需人工确认：{d.reason}。请在审批界面确认后重放 "
                           f"confirm_token={d.confirm_token}；当前策略为「拒绝」，已拦截未执行。")
            # approve：把 confirm 当 ALLOW 继续
            d = GuardDecision(Verdict.ALLOW, d.risk, reason=f"人工确认通过 {tool}.{action}",
                              policy_name=d.policy_name)

        # ---- ALLOW：真正交给后端执行 ----
        # 幂等：mutable 写操作，相同参数只执行一次（policy/idem_key 已在入口处算好）
        if replay:
            result = self._idem[idem_key]
            self._audit(ctx, tool, action, args, d, ok=True, result="[幂等重放]",
                        masked=False)
            return d, result

        try:
            raw = dispatcher(tool, action, **args)
        except Exception as e:
            self._audit(ctx, tool, action, args, d, ok=False, result=f"执行异常:{e}")
            return d, f"❌ {tool}.{action} 执行异常: {e}"

        ok = not (raw.startswith("❌") or "失败" in raw[:80])
        if ok and idem_key:
            self._idem[idem_key] = raw
            if len(self._idem) > self._MAX_IDEM:   # 硬上限：按最早登记裁剪，防无限增长
                self._idem.pop(next(iter(self._idem)), None)
            # 落库：重启后相同写操作仍能识别为"已执行过"，不会重复退款/重复写入
            if self._db is not None:
                try:
                    self._db.guard_idem_put(idem_key, raw, self._MAX_IDEM)
                except Exception:
                    pass

        # 回滚策略登记：写操作成功 → 记录撤销钩子
        if ok and policy and policy.mutable and policy.rollback:
            undo_key = f"{ctx.user_id}:{idem_key}"
            self._rollback[undo_key] = (policy.rollback, dict(args))
            if len(self._rollback) > self._MAX_ROLLBACK:
                self._rollback.pop(next(iter(self._rollback)), None)

        # 后端再次校验的兜底：对返回结果做最小化脱敏（图「数据脱敏」）
        masked = False
        if policy and (policy.redact_fields or policy.redact_text):
            raw, masked = Redactor.mask_text(raw) if policy.redact_text else (raw, False)

        self._audit(ctx, tool, action, args, d, ok=ok, result=raw, masked=masked)
        return d, raw

    def rollback_pending(self) -> list[str]:
        return list(self._rollback.keys())

    def rollback(self, undo_key: str) -> bool:
        """触发一次补偿（图「回滚策略——出现异常可回滚/补偿」）。"""
        hook = self._rollback.pop(undo_key, None)
        if hook is None:
            return False
        fn, args = hook
        try:
            fn(**args)
            return True
        except Exception:
            return False

    def _audit(self, ctx, tool, action, args, d: GuardDecision, ok, result, masked=False):
        if self.audit_log is None:
            return
        try:
            # 审计入库前做全覆盖脱敏：递归扫描 args 全部字符串值，
            # 命中手机/证件号/银行卡/金额即打码（不再只认 text/query 两个字段名）。
            safe_args = Redactor.mask_obj(dict(args))
            self.audit_log(
                user_id=ctx.user_id, tenant_id=ctx.tenant_id, role=ctx.role,
                tool=tool, action=action, args=safe_args,
                risk=d.risk.value, verdict=d.verdict.value,
                ok=ok, result=(result or "")[:600], masked=masked,
            )
        except Exception:
            pass  # 审计失败不阻断业务
