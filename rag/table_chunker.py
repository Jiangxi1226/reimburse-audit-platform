import re



def is_table_row(line: str) -> bool:
    """判断单行是否为表格行（含 | 分隔符或 Tab 分隔的多列）。"""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2:
        return True
    if "\t" in stripped and len(stripped.split("\t")) >= 3:
        return True
    return False


def is_separator_row(line: str) -> bool:
    """判断是否为 Markdown 表格分隔行（|---|---|）。"""
    return bool(re.match(r'^\|?[\s]*[-:]+[\s|]*[-:\s|]+$', line.strip()))


def is_table_header(line: str) -> bool:
    """判断是否为表头行——含 | 且字段中有中文词。"""
    if not is_table_row(line):
        return False
    cells = [c.strip() for c in line.split("|") if c.strip()]
    chinese_cells = [c for c in cells if re.search(r'[一-鿿]', c)]
    return len(chinese_cells) >= 2



def extract_tables(text: str) -> tuple[list[str], list[dict]]:
    """从文本中提取表格区域。

    扫描全文，找到连续的表格行（≥3行），将其作为表格块提取出来。
    返回 (非表格文本段列表, 表格块列表)。
    """
    lines = text.split("\n")
    table_regions = []
    in_table = False
    table_start = 0

    for i, line in enumerate(lines):
        if is_table_row(line) and not is_separator_row(line):
            if not in_table:
                table_start = i
                in_table = True
        else:
            if in_table:
                if i - table_start >= 3:
                    table_regions.append((table_start, i - 1))
                elif i - table_start >= 2 and _has_numeric_columns(lines[table_start:i]):
                    table_regions.append((table_start, i - 1))
                in_table = False

    if in_table and len(lines) - table_start >= 2:
        table_regions.append((table_start, len(lines) - 1))

    tables_md = []
    tables_json = []
    non_table_chunks = []
    last_end = -1

    for start, end in table_regions:
        if last_end + 1 < start:
            segment = "\n".join(lines[last_end + 1:start]).strip()
            if segment:
                non_table_chunks.append(segment)

        table_lines = lines[start:end + 1]
        tables_md.append(_format_as_markdown(table_lines))
        tables_json.append(_format_as_json(table_lines))
        last_end = end

    if last_end < len(lines) - 1:
        segment = "\n".join(lines[last_end + 1:]).strip()
        if segment:
            non_table_chunks.append(segment)

    return non_table_chunks, tables_md, tables_json


def _format_as_markdown(lines: list[str]) -> str:
    """将表格行格式化为标准 Markdown table。"""
    if not lines:
        return ""

    clean_lines = [l for l in lines if not is_separator_row(l)]
    if not clean_lines:
        return ""

    header = clean_lines[0]
    cells = _split_cells(header)
    n_cols = len(cells)

    result = "| " + " | ".join(cells) + " |\n"
    result += "| " + " | ".join(["---"] * n_cols) + " |\n"

    for line in clean_lines[1:]:
        cells = _split_cells(line)
        while len(cells) < n_cols:
            cells.append("")
        cells = cells[:n_cols]
        result += "| " + " | ".join(cells) + " |\n"

    return result


def _format_as_json(lines: list[str]) -> dict:
    """将表格行格式化为 JSON 键值对列表。"""
    clean_lines = [l for l in lines if not is_separator_row(l)]
    if len(clean_lines) < 2:
        return {"headers": [], "rows": []}

    headers = _split_cells(clean_lines[0])
    rows = []
    for line in clean_lines[1:]:
        cells = _split_cells(line)
        while len(cells) < len(headers):
            cells.append("")
        row = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
        rows.append(row)

    return {"headers": headers, "rows": rows}


def _split_cells(line: str) -> list[str]:
    """按 | 或 Tab 拆分单元格。"""
    line = line.strip()
    if "|" in line:
        return [c.strip() for c in line.split("|") if c.strip() or True][1:-1] if line.startswith("|") else [c.strip() for c in line.split("|")]
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return [line]


def _has_numeric_columns(lines: list[str]) -> bool:
    """判断多行是否含有数值列——紧凑表格的特征。"""
    num_count = 0
    for line in lines:
        cells = _split_cells(line)
        if sum(1 for c in cells if re.search(r'\d+', c)) >= 2:
            num_count += 1
    return num_count >= len(lines) * 0.6



class TablePreservingChunker:
    """表格感知分块包装器——表格整块保留，非表格按原策略切分。

    不是替代 Chunker，而是在其之上加一层表格检测。
    用法：
      original = SemanticChunker(200, 50)
      wrapper = TablePreservingChunker(original)
      chunks = wrapper.split(text)
    """

    def __init__(self, base_chunker):
        self.base_chunker = base_chunker

    def split(self, text: str) -> list[dict]:
        """切分文本，表格块保留为结构化 chunk。

        Returns:
            [{"text": ..., "type": "text"|"table", "table_format": "markdown"|"json", ...}, ...]
        """
        if not text or not text.strip():
            return []

        non_table_chunks, tables_md, tables_json = extract_tables(text)
        results = []

        for segment in non_table_chunks:
            for chunk_text in self.base_chunker.split(segment):
                results.append({
                    "text": chunk_text,
                    "type": "text",
                })

        for i, (md, js) in enumerate(zip(tables_md, tables_json)):
            headers_str = ", ".join(js.get("headers", []))
            results.append({
                "text": md,
                "type": "table",
                "table_format": "markdown",
                "table_json": js,
                "table_id": f"table_{i}",
                "table_summary": f"[表格 {i+1}: {headers_str}] [{len(js.get('rows', []))} 行]",
            })

        return results



def demo():
    """对比普通切分和表格感知切分。"""
    text = """产品营收分析报告

2024年Q3各产品线营收情况如下：

| 产品线  | Q3营收(亿元) | 同比增长 | 占比  |
|---------|-------------|---------|------|
| 产品线A | 1.2         | 30%     | 60%  |
| 产品线B | 0.6         | 15%     | 30%  |
| 产品线C | 0.2         | -5%     | 10%  |

从表中可以看出，产品线A是主要营收来源，贡献了60%的营收。
产品线B增长稳定，产品线C出现负增长需要重点关注。"""

    from rag.semantic_chunker import SemanticChunker

    print("=== 普通切分（表格被拆碎）===")
    for i, c in enumerate(SemanticChunker(50, 10).split(text)):
        print(f"  [{i}] {c[:80]}...")

    print("\n=== 表格感知切分（表格完整保留）===")
    wrapper = TablePreservingChunker(SemanticChunker(50, 10))
    for i, c in enumerate(wrapper.split(text)):
        tag = f"[{c['type']}]" if c['type'] == 'table' else ""
        print(f"  [{i}]{tag} {c['text'][:100]}...")


if __name__ == "__main__":
    demo()
