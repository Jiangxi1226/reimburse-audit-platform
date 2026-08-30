import json, re
from collections import defaultdict



EXTRACT_PROMPT = """从以下文本中提取所有重要的实体和它们之间的关系。
输出 JSON 格式，不要解释。

文本：
{text}

输出格式：
{{
  "entities": [
    {{"name": "实体名", "type": "person/organization/metric/product/date/other"}}
  ],
  "relations": [
    {{"source": "实体1名", "target": "实体2名", "relation": "关系描述"}}
  ]
}}

规则：
- 实体用中文全称，不要缩写
- 关系描述要具体（如"贡献60%营收"而非"相关"）
- 只提取文本中明确提到的，不要推测"""


def extract_triples(text: str, llm) -> dict:
    """LLM 抽取三元组——一条文本 → 实体列表 + 关系列表。"""
    prompt = EXTRACT_PROMPT.format(text=text[:2000])
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"entities": [], "relations": []}



class KnowledgeGraph:
    """跨文档知识图谱——同名实体自动合并，建立跨文档逻辑联系。

    例：文档A提到的"Q3营收"和文档B提到的"Q3营收"→ 合并为同一实体，
    由此文档A的"产品线A → 贡献 → Q3营收"和文档B的"Q3营收 → 增长 → 30%"
    自动串联成多跳推理路径。
    """

    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.relations: list[dict] = []
        self.adjacency: dict[str, set] = defaultdict(set)
        self._name_index: dict[str, str] = {}

    def add_triples(self, triples: dict, doc_id: str):
        """添加一批三元组到图谱——自动合并同名实体。"""
        for ent in triples.get("entities", []):
            name = ent["name"].strip()
            if not name:
                continue
            if name not in self.entities:
                self.entities[name] = {
                    "type": ent.get("type", "other"),
                    "sources": [doc_id],
                    "aliases": set(),
                }
            else:
                if doc_id not in self.entities[name]["sources"]:
                    self.entities[name]["sources"].append(doc_id)

        for rel in triples.get("relations", []):
            source = rel["source"].strip()
            target = rel["target"].strip()
            relation = rel.get("relation", "相关").strip()
            if not source or not target:
                continue

            self.relations.append({
                "source": source, "target": target,
                "relation": relation, "doc_id": doc_id,
            })
            self.adjacency[source].add(target)
            self.adjacency[target].add(source)

    def query(self, entity_names: list[str], max_depth: int = 2) -> dict:
        """从给定实体出发，提取 depth 层邻域子图——这就是图检索。

        和向量检索的关键区别：
          向量检索：返回 top-k 个相似的文本块
          图检索：返回包含实体+关系的结构化子图 → LLM 可以基于图推理
        """
        seed = set()
        for name in entity_names:
            for ename in self.entities:
                if name in ename or ename in name:
                    seed.add(ename)

        visited = set(seed)
        frontier = set(seed)
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        sub_entities = [
            {"name": e, "type": self.entities[e]["type"],
             "sources": self.entities[e]["sources"]}
            for e in visited
        ]
        sub_relations = [
            r for r in self.relations
            if r["source"] in visited and r["target"] in visited
        ]

        context = "知识图谱子图：\n"
        context += "实体：\n" + "\n".join(
            f"  - {e['name']} ({e['type']})" for e in sub_entities
        )
        context += "\n关系：\n" + "\n".join(
            f"  - {r['source']} --[{r['relation']}]--> {r['target']}"
            f" (来源: {r['doc_id']})" for r in sub_relations
        )

        return {
            "entities": sub_entities,
            "relations": sub_relations,
            "context": context,
            "stats": self.stats(),
        }

    def stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "connected_components": self._count_components(),
        }

    def _count_components(self) -> int:
        """统计连通分量数——衡量图谱碎片程度。"""
        visited = set()
        components = 0
        for node in self.entities:
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        stack.extend(self.adjacency.get(n, set()))
        return components



COMPLEMENT = """
向量 RAG 和 GraphRAG 的互补关系：

  用户问题 → 图检索（找出"谁和谁相关"）→ 向量检索（找出具体内容）

  例："Q3营收增长由哪个产品线驱动？"

  图检索先回答结构问题：
    Q3营收 ←[贡献60%]← 产品线A
    Q3营收 ←[贡献40%]← 产品线B
    Q3营收 ←[同比增长]← Q2营收
    → 结论：产品线A贡献最大

  向量检索再回答细节问题：
    "产品线A在Q3的具体表现为..."
    → 检索产品线A相关的文档段落

  两个不是替代关系，是互补关系。
"""
