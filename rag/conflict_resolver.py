"""冲突识别与解决模块 —— 金融财报场景。

问题：多来源检索结果可能内容矛盾。
  检索块A: "Q3营收12.5亿元，同比增长30%"  (来源: 经营快报, 权威度0.6)
  检索块B: "Q3营收8.3亿元，同比下降15%"   (来源: 审计报告, 权威度1.0)
  旧系统直接按 similarity 排序喂给 LLM，冲突交给 LLM 自行处理。

本模块在"检索 → 答案生成"之间插入一层：
  1. 数值矛盾检测（纯规则，无 LLM 依赖，快）
  2. 语义矛盾检测（可选 LLM，对疑似冲突二次确认 + 发现语义对立）
  3. 来源加权解决（结合权威度 authority_level + 文档时间 document_date）

用法：
    resolver = ConflictResolver(llm)
    report = resolver.detect(chunks)          # 发现冲突
    result  = resolver.resolve(chunks, report)  # 加权排序，返回冲突报告
"""

import re
from datetime import datetime


# ============================================================
# 1. 数值矛盾检测（纯规则）
# ============================================================

# 金融指标关键词 → 用于识别"同一指标的不同数值"
METRIC_PATTERNS = [
    r"营收", r"营业收入", r"净利润", r"利润", r"收入", r"销售额", r"销量",
    r"增长", r"下降", r"增速", r"亏损", r"毛利", r"净利", r"毛利率", r"净利率",
    r"总资产", r"净资产", r"负债", r"市值", r"EPS", r"每股收益", r"分红",
    r"ROE", r"每股净资产", r"经营现金流", r"研发投入",
]

# 数值 + 单位提取
VALUE_RE = re.compile(
    r"([-+]?\d[\d,，]*(?:\.\d+)?)\s*(万亿|千亿|百亿|十亿|亿元|亿|千万|百万|万元|万|千元|百元|元|%|％|倍)?"
)

# 单位 → 相对"元"的倍率（用于统一量纲后比较）
UNIT_RATE = {
    "万亿": 1e12, "千亿": 1e11, "百亿": 1e10, "十亿": 1e9,
    "亿元": 1e8, "亿": 1e8, "千万": 1e7, "百万": 1e6,
    "万元": 1e4, "万": 1e4, "千元": 1e3, "百元": 1e2, "元": 1.0,
}

# 相对值（百分比/倍数）单独比较，不做元换算
PCT_UNITS = {"%", "％", "倍"}


class ConflictResolver:
    """冲突识别 + 来源加权解决。

    Args:
        llm: 可选。传入后启用语义矛盾检测（对疑似冲突用 LLM 确认）。
            不传则只做数值规则检测（无 LLM 依赖，稳定）。
    """

    def __init__(self, llm=None):
        self.llm = llm

    # --------------------------------------------------------
    # 对外主入口
    # --------------------------------------------------------
    # 停用词：切问题关键词时剔除，避免把"公司"这类宽泛词当相关性特征
    STOPWORDS = {
        "公司", "现行", "这份", "本", "那个", "这个", "多久", "多少", "是否",
        "根据", "怎么", "什么", "如何", "为", "是", "的", "吗", "了", "在",
        "于", "和", "与", "或", "请", "问", "一下", "能", "可以", "应该",
    }

    def detect(self, chunks: list[dict], use_llm: bool = True,
               question: str = None) -> dict:
        """检测检索结果中的冲突。

        Args:
            chunks: [{"id","text","similarity","metadata":{...}}, ...]
            use_llm: 是否用 LLM 确认疑似数值冲突 + 找语义矛盾（需 self.llm）
            question: 若给出，只用"与问题相关"的块判冲突——避免把
                无关块的指标数值误当冲突（如"付款期限"问题把营收12.5/8.3判为矛盾）

        Returns:
            {"has_conflict": bool,
             "conflicts": [ {type, a_id, b_id, metric, value_a, value_b,
                             reason, favor_a, ...}, ... ]}
        """
        conflicts = []

        # 0. 与问题相关性门控：无关块不参与冲突判定（解决跨主题误报）
        scan_chunks = self._filter_by_question(chunks, question) if question else chunks

        # ① 数值矛盾：逐对扫描
        numeric_conflicts = self._detect_numeric_conflicts(scan_chunks)
        conflicts.extend(numeric_conflicts)

        # ② LLM 语义矛盾：仅当启用了 LLM 且数值检测已发现疑似冲突时，
        #    用 LLM 做二次确认；同时顺带发现"无共同数字但语义对立"的矛盾。
        if use_llm and self.llm is not None:
            pairs_to_check = [c for c in numeric_conflicts]
            # 为避免对每个 chunk 对都调 LLM（成本高），只对数值冲突所在块、
            # 及与它们同主题的块用 LLM 复核。
            involved_ids = {c["a_id"] for c in numeric_conflicts} | \
                           {c["b_id"] for c in numeric_conflicts}
            semantic_conflicts = self._detect_semantic_conflicts(
                scan_chunks, involved_ids, confirmed_numeric=pairs_to_check
            )
            for sc in semantic_conflicts:
                # 语义检测返回的是确认结果；若 LLM 判定数值冲突实际不成立，则移除
                conflicts = [c for c in conflicts if not self._same_pair(c, sc)]
            conflicts.extend(semantic_conflicts)

        # 去重
        conflicts = self._dedup_conflicts(conflicts)

        return {
            "has_conflict": bool(conflicts),
            "conflicts": conflicts,
            "count": len(conflicts),
        }

    def resolve(self, chunks: list[dict], report: dict) -> dict:
        """结合来源权威度 + 文档时间，对冲突给出倾向意见。

        Returns:
            {"ranked": 加权排序后的 chunks,
             "conflicts": 每条冲突附上 favor_a / winner_id / reason}
        """
        conflicts = []
        for c in report.get("conflicts", []):
            resolved = self._resolve_one(c, chunks)
            conflicts.append(resolved)

        # 加权重排（权威度 + 时间）——即使无冲突也应用，作为通用排序增强
        ranked = self._apply_authority_ranking(chunks)

        return {"ranked": ranked, "conflicts": conflicts}

    # --------------------------------------------------------
    # 数值矛盾检测
    # --------------------------------------------------------
    def _detect_numeric_conflicts(self, chunks: list[dict]) -> list[dict]:
        conflicts = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                a, b = chunks[i], chunks[j]
                # 同一来源（同文件）不判冲突——同文档内通常自洽
                if self._same_source(a, b):
                    continue
                pair = self._find_numeric_conflict(a, b)
                if pair:
                    conflicts.append({
                        "type": "numeric",
                        "a_id": a.get("id", ""), "b_id": b.get("id", ""),
                        "metric": pair["metric"],
                        "value_a": pair["value_a"], "value_b": pair["value_b"],
                        "reason": pair["reason"],
                    })
        return conflicts

    def _find_numeric_conflict(self, a: dict, b: dict) -> dict | None:
        """在一对 chunk 中找"同一指标 + 不同数值"的冲突。"""
        a_facts = self._extract_financial_facts(a.get("text", ""))
        b_facts = self._extract_financial_facts(b.get("text", ""))

        for fa in a_facts:
            for fb in b_facts:
                if fa["metric"] != fb["metric"]:
                    continue
                # 相对值（%）比较百分比本身；绝对值统一换算到元
                if self._values_conflict(fa, fb):
                    return {
                        "metric": fa["metric"],
                        "value_a": fa["display"], "value_b": fb["display"],
                        "reason": (
                            f"两来源对「{fa['metric']}」的表述冲突："
                            f"{fa['display']} vs {fb['display']}"
                        ),
                    }
        return None

    def _extract_financial_facts(self, text: str) -> list[dict]:
        """从文本中提取 [(指标, 数值, 单位), ...]。"""
        facts = []
        for metric in METRIC_PATTERNS:
            # 指标后紧跟一个数值
            for m in re.finditer(re.escape(metric) + r"\s*[为达是约共计合计]?\s*" +
                                 r"([-+]?\d[\d,，]*(?:\.\d+)?)\s*(万亿|千亿|百亿|十亿|亿元|亿|千万|百万|万元|万|千元|百元|元|%|％|倍)?",
                                 text):
                value_str = m.group(1).replace(",", "").replace("，", "")
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                unit = m.group(2) or ""
                facts.append({
                    "metric": metric, "value": value, "unit": unit,
                    "display": f"{m.group(1)}{unit}",
                })
        return facts

    def _values_conflict(self, fa: dict, fb: dict) -> bool:
        """判断两个同指标数值是否冲突。统一量纲后比较，排除量纲造成的假冲突。"""
        if fa["unit"] in PCT_UNITS or fb["unit"] in PCT_UNITS:
            # 百分比/倍数：直接比数值
            if fa["unit"] not in PCT_UNITS or fb["unit"] not in PCT_UNITS:
                return False  # 一个相对一个绝对，不强行比
            return abs(fa["value"] - fb["value"]) > 1e-6
        va = fa["value"] * UNIT_RATE.get(fa["unit"], 1.0)
        vb = fb["value"] * UNIT_RATE.get(fb["unit"], 1.0)
        if va == 0 or vb == 0:
            return False
        # 相对差异超过 10% 视为冲突（容忍口径/舍入差异）
        return abs(va - vb) / max(abs(va), abs(vb)) > 0.10

    # --------------------------------------------------------
    # 语义矛盾检测（LLM）
    # --------------------------------------------------------
    def _detect_semantic_conflicts(self, chunks, involved_ids, confirmed_numeric):
        """用 LLM 复核疑似冲突，并发现无语数字但对立的矛盾。

        只对涉及疑似冲突的块做两两判断，控制 LLM 调用成本（最多 6 对）。
        """
        semantic = []
        involved = [c for c in chunks if c.get("id") in involved_ids]
        # 没有疑似数值冲突时，退化为对 top 候选做语义扫描（限制对数）
        if not involved:
            involved = chunks[:4]

        checked = set()
        count = 0
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                a, b = involved[i], involved[j]
                key = (a.get("id", ""), b.get("id", ""))
                if key in checked or count >= 6:
                    continue
                checked.add(key)
                verdict = self._llm_judge(a.get("text", ""), b.get("text", ""))
                if verdict == "CONTRADICTION":
                    semantic.append({
                        "type": "semantic",
                        "a_id": a.get("id", ""), "b_id": b.get("id", ""),
                        "metric": "", "value_a": "", "value_b": "",
                        "reason": "两来源结论语义对立（LLM 判定）",
                    })
                    count += 1
        return semantic

    def _llm_judge(self, text_a: str, text_b: str) -> str:
        """LLM 判断两段文本关系：CONTRADICTION / NEUTRAL / ENTAILMENT。"""
        prompt = (
            "判断下面两段来自不同财报来源的文字是「矛盾(CONTRADICTION)」「无关/中立(NEUTRAL)」"
            "还是「一致/包含(ENTAILMENT)」。只回复一个单词。\n\n"
            f"文本A: {text_a[:500]}\n\n文本B: {text_b[:500]}"
        )
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            up = resp.upper()
            if "CONTRADICT" in up:
                return "CONTRADICTION"
            if "ENTAIL" in up:
                return "ENTAILMENT"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    # --------------------------------------------------------
    # 来源加权解决
    # --------------------------------------------------------
    def _resolve_one(self, conflict: dict, chunks: dict) -> dict:
        a = self._find_chunk(chunks, conflict.get("a_id", ""))
        b = self._find_chunk(chunks, conflict.get("b_id", ""))
        a_authority = self._authority_of(a)
        b_authority = self._authority_of(b)
        a_date = self._date_of(a)
        b_date = self._date_of(b)
        a_version = self._version_of(a)
        b_version = self._version_of(b)

        # 综合排序：权威度优先；同权威时看生效日期；再同日期看版本号
        a_score = (a_authority, self._date_rank(a_date), self._version_rank(a_version))
        b_score = (b_authority, self._date_rank(b_date), self._version_rank(b_version))
        favor_a = a_score >= b_score

        conflict["a_authority"] = a_authority
        conflict["b_authority"] = b_authority
        conflict["a_date"] = a_date or ""
        conflict["b_date"] = b_date or ""
        conflict["a_version"] = a_version or ""
        conflict["b_version"] = b_version or ""
        conflict["favor_a"] = favor_a
        conflict["winner_id"] = conflict["a_id"] if favor_a else conflict["b_id"]
        conflict["resolution"] = (
            f"倾向来源{'A' if favor_a else 'B'}（权威度/生效日期/版本更高者优先），"
            "冲突仍建议向用户标注。"
        )
        return conflict

    def _apply_authority_ranking(self, chunks: list[dict]) -> list[dict]:
        """按 相关度 + 权威度 + 时间 加权排序（重排增强）。"""
        ranked = []
        for c in chunks:
            sim = c.get("similarity", 0) or c.get("_rerank_score", 0) or 0
            authority = self._authority_of(c)
            weighted = sim * (0.7 + 0.3 * authority)
            ranked.append({**c, "_weighted_score": round(weighted, 5)})
        ranked.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
        return ranked

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------
    def _authority_of(self, chunk: dict | None) -> float:
        meta = (chunk or {}).get("metadata", {}) or {}
        return float(meta.get("authority_level", 0.5))

    def _date_of(self, chunk: dict | None) -> str:
        meta = (chunk or {}).get("metadata", {}) or {}
        # 生效日期优先（处理"旧制度 vs 新制度"），退化为文档日期
        return (meta.get("effective_date") or meta.get("document_date")
                or meta.get("modified_at", ""))

    @staticmethod
    def _date_rank(date_str: str) -> float:
        """日期字符串转可比较数值（越大越新）。无法解析返回 0。"""
        if not date_str:
            return 0.0
        try:
            return datetime.fromisoformat(str(date_str).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    def _version_of(self, chunk: dict | None) -> str:
        meta = (chunk or {}).get("metadata", {}) or {}
        return meta.get("version", "")

    @staticmethod
    def _version_rank(version: str) -> float:
        """版本号转可比较数值：v2.1 -> 2.1，修订版 -> 1.5。无版本 -> 0。"""
        if not version:
            return 0.0
        try:
            return float(version)
        except ValueError:
            return 0.0

    def _filter_by_question(self, chunks: list[dict], question: str) -> list[dict]:
        """只保留与问题主题相关的块，用于冲突检测。

        相关性代理：块命中问题的 **3-gram**（任一二字即可，很特异），
        或命中 **≥2 个 2-gram**（单个宽泛二字如"产品"容易误伤"产品线A"这类
        无关块，故 2-gram 需多个同时命中）。全部不命中则回退全量（避免误删证据）。
        """
        bigrams, trigrams = self._question_ngrams(question)
        if not bigrams and not trigrams:
            return chunks
        relevant = [c for c in chunks if self._chunk_matches(c, bigrams, trigrams)]
        return relevant if relevant else chunks

    @classmethod
    def _question_ngrams(cls, question: str) -> tuple[list[str], list[str]]:
        """提取相关性特征：中文 n-gram（2-gram + 3-gram）。

        不依赖分词器——中文无空格，整段切分易把"员工休假制度"连成一段，
        匹配不上文档里的"员工年休假制度"（中间多了个"年"）。n-gram 对任意
        "员工/年休假/制度"都两两/三三覆盖，免分词。
        """
        zh = re.sub(r'[^一-鿿]', '', question)  # 只留中文
        if len(zh) < 2:
            return ([zh] if zh else []), []
        bigrams = list({zh[i:i + 2] for i in range(len(zh) - 1)})
        trigrams = list({zh[i:i + 3] for i in range(len(zh) - 2)}) if len(zh) >= 3 else []
        return bigrams, trigrams

    @staticmethod
    def _chunk_matches(chunk: dict, bigrams: list[str], trigrams: list[str]) -> bool:
        text = chunk.get("text", "")
        if any(t in text for t in trigrams):
            return True
        return sum(1 for b in bigrams if b in text) >= 2

    def _same_source(self, a: dict, b: dict) -> bool:
        ma = (a.get("metadata", {}) or {}).get("source_path", "")
        mb = (b.get("metadata", {}) or {}).get("source_path", "")
        return ma and ma == mb

    def _find_chunk(self, chunks: list[dict], cid: str) -> dict | None:
        for c in chunks:
            if c.get("id") == cid:
                return c
        return None

    def _same_pair(self, c1: dict, c2: dict) -> bool:
        return {c1.get("a_id"), c1.get("b_id")} == {c2.get("a_id"), c2.get("b_id")}

    def _dedup_conflicts(self, conflicts: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for c in conflicts:
            key = (c.get("a_id", ""), c.get("b_id", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out


def format_conflict_prompt(conflicts: list[dict]) -> str:
    """把冲突列表转成注入 LLM 提示的文本——提示答案生成时标注分歧。"""
    if not conflicts:
        return ""
    lines = ["⚠️ **检索结果存在信息冲突，请在答案中明确标注，切勿自行调和或偏信一方：**"]
    for i, c in enumerate(conflicts, 1):
        lines.append(f"{i}. {c.get('reason', '来源表述冲突')}")
        if c.get("a_authority") is not None:
            lines.append(
                f"   - 来源A(权威度{c['a_authority']:.1f}/{c.get('a_date','未知日期')}) "
                f"vs 来源B(权威度{c['b_authority']:.1f}/{c.get('b_date','未知日期')})，"
                f"当前倾向：{c.get('resolution','')}"
            )
    return "\n".join(lines)


# 便捷函数
def detect_conflicts(chunks: list[dict], llm=None) -> dict:
    return ConflictResolver(llm).detect(chunks)


def resolve_conflicts(chunks: list[dict], llm=None) -> dict:
    resolver = ConflictResolver(llm)
    report = resolver.detect(chunks)
    return resolver.resolve(chunks, report)
