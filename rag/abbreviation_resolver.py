import re



DEFAULT_ABBREVIATIONS: dict[str, list[str]] = {
    "Q1": ["第一季度", "一季度"],
    "Q2": ["第二季度", "二季度"],
    "Q3": ["第三季度", "三季度"],
    "Q4": ["第四季度", "四季度"],
    "H1": ["上半年"],
    "H2": ["下半年"],
    "YoY": ["同比增长", "同比"],
    "QoQ": ["环比增长", "环比"],
    "YTD": ["年初至今", "Year-To-Date"],
    "MTD": ["月初至今", "Month-To-Date"],
    "ROI": ["投资回报率", "投入产出比"],
    "ROE": ["净资产收益率"],
    "EBITDA": ["息税折旧摊销前利润"],
    "KPI": ["关键绩效指标", "核心指标"],
    "OKR": ["目标与关键成果"],

    "API": ["应用程序接口"],
    "SDK": ["软件开发工具包"],
    "CI/CD": ["持续集成", "持续部署", "自动化部署"],
    "MVP": ["最小可行产品"],
    "POC": ["概念验证", "原型验证"],
    "RAG": ["检索增强生成"],
    "LLM": ["大语言模型", "大模型"],
    "OCR": ["文字识别", "光学字符识别"],
    "NER": ["命名实体识别"],

    "B2B": ["企业对企业"],
    "B2C": ["企业对消费者"],
    "SaaS": ["软件即服务"],
    "PaaS": ["平台即服务"],
    "CRM": ["客户关系管理"],
    "ERP": ["企业资源计划"],
    "HR": ["人力资源", "人事"],
    "PR": ["公共关系", "公关"],

    "DAU": ["日活跃用户"],
    "MAU": ["月活跃用户"],
    "GMV": ["商品交易总额", "成交额"],
    "ARPU": ["每用户平均收入"],
    "CVR": ["转化率"],
    "CTR": ["点击率"],
    "ROAS": ["广告支出回报率"],
}


class AbbreviationResolver:
    """缩写消歧器——查询时自动展开缩写为全称。

    用法：
      resolver = AbbreviationResolver()
      expanded = resolver.expand("Q3的ROI和KPI达标情况")
    """

    def __init__(self, custom_dict: dict[str, list[str]] = None):
        self.dict: dict[str, list[str]] = dict(DEFAULT_ABBREVIATIONS)
        if custom_dict:
            self.dict.update(custom_dict)

        self._sorted_keys = sorted(self.dict.keys(), key=len, reverse=True)

        self._pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self._sorted_keys) + r')\b',
            re.IGNORECASE
        )

    def expand(self, query: str, mode: str = "append") -> str:
        """展开查询中的缩写。

        Args:
            query: 用户原始查询
            mode: "append" → 原词 + 展开词（推荐，保持原始语义）
                  "replace" → 仅展开词（去掉缩写）
                  "both" → 返回 (原查询, 展开后查询) 元组

        Returns:
            展开后的查询字符串（或元组，取决于 mode）
        """
        if mode == "both":
            return query, self._expand_text(query)

        expanded = self._expand_text(query)
        if mode == "replace":
            return expanded
        return f"{query} {expanded}"

    def _expand_text(self, text: str) -> str:
        """替换文本中的缩写为其全称。"""
        result = text
        for abbr in self._sorted_keys:
            if re.search(r'\b' + re.escape(abbr) + r'\b', result, re.IGNORECASE):
                expansions = " ".join(self.dict[abbr])
                result = re.sub(
                    r'\b' + re.escape(abbr) + r'\b',
                    f"{abbr} {expansions}",
                    result,
                    flags=re.IGNORECASE
                )
        return result

    def get_expansions(self, abbr: str) -> list[str]:
        """查询单个缩写的展开。"""
        return self.dict.get(abbr.upper(), self.dict.get(abbr, []))

    def add(self, abbr: str, expansions: list[str]):
        """动态添加缩写（从企业文档中自动学习）。"""
        self.dict[abbr] = expansions
        self._sorted_keys = sorted(self.dict.keys(), key=len, reverse=True)
        self._pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self._sorted_keys) + r')\b',
            re.IGNORECASE
        )

    def learn_from_document(self, text: str):
        """从文档中自动检测"全称（缩写）"或"缩写——全称"模式，自动收录。"""
        for match in re.finditer(r'([一-鿿\w]{2,20})[（(]([A-Z]{2,8})[）)]', text):
            full_name, abbr = match.groups()
            if abbr.upper() not in self.dict:
                self.add(abbr.upper(), [full_name])

        for match in re.finditer(r'([A-Z]{2,8})[—–-]+([一-鿿\w]{2,20})', text):
            abbr, full_name = match.groups()
            if abbr.upper() not in self.dict:
                self.add(abbr.upper(), [full_name])



_resolver: AbbreviationResolver | None = None


def get_resolver() -> AbbreviationResolver:
    """获取全局缩写消歧器单例。"""
    global _resolver
    if _resolver is None:
        _resolver = AbbreviationResolver()
    return _resolver
