import os, re, tempfile, zipfile
from html.parser import HTMLParser
import pdfplumber, docx, openpyxl, fitz
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from paddleocr import PaddleOCR

_ocr_instance = None

def _get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        # enable_mkldnn=False：规避 PaddleOCR 3.x 在部分 CPU 环境上 onednn 的
        # "ConvertPirAttribute2RuntimeAttribute not support ... ArrayAttribute" 崩溃。
        # 关闭 oneDNN 后 OCR 恢复稳定（实测），不影响识别质量。
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False)
    return _ocr_instance

_BYTE_SIGNATURES = [
    (b'\xff\xd8\xff', '.jpg'),
    (b'\x89PNG',      '.png'),
    (b'GIF8',         '.gif'),
    (b'RIFF',         '.webp'),
    (b'BM',           '.bmp'),
    (b'II*\x00',      '.tiff'),
    (b'MM\x00*',      '.tiff'),
]

def _detect_suffix(data: bytes) -> str:
    for magic, suffix in _BYTE_SIGNATURES:
        if data.startswith(magic):
            return suffix
    return '.png'

OCR_CONFIDENCE_THRESHOLD = 0.7
OCR_MIN_LINES = 3
OCR_CRITICAL_THRESHOLD = 0.4
OCR_COVERAGE_THRESHOLD = 0.3

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
TEXT_EXTS  = {".md", ".txt", ".py", ".json", ".csv", ".xml", ".log", ".tex", ".yaml", ".yml"}

_UPGRADE_HINTS = {
    ".doc":  "请用 Word 另存为 .docx 格式后上传",
    ".xls":  "请用 Excel 另存为 .xlsx 格式后上传",
    ".ppt":  "请用 PowerPoint 另存为 .pptx 格式后上传",
    ".rtf":  "请用 Word 打开后另存为 .docx 格式后上传",
    ".wps":  "请用 WPS 另存为 .docx 或 .pdf 格式后上传",
    ".odt":  "请另存为 .docx 格式后上传",
    ".ods":  "请另存为 .xlsx 格式后上传",
}

SUPPORTED_FORMATS = (
    "PDF(.pdf) | Word(.docx) | Excel(.xlsx) | PPT(.pptx) | "
    "HTML(.html) | EPUB(.epub) | 图片(.jpg/.png/.gif/.webp/.bmp/.tiff) | "
    "纯文本(.txt/.md/.py/.json/.csv/.xml/.log/.tex)"
)



def load(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":   return _load_pdf(file_path)
    if ext == ".docx":  return _load_docx(file_path)
    if ext == ".xlsx":  return _load_excel(file_path)
    if ext == ".pptx":  return _load_pptx(file_path)
    if ext in (".html", ".htm"): return _load_html(file_path)
    if ext == ".epub":  return _load_epub(file_path)
    if ext in TEXT_EXTS: return _load_text(file_path)
    if ext in IMAGE_EXTS: return _ocr_image_file(file_path)

    if ext in _UPGRADE_HINTS:
        raise ValueError(f"不支持 .{ext} 旧格式。{_UPGRADE_HINTS[ext]}。")
    raise ValueError(f"不支持的格式: {ext}。支持的文件格式：{SUPPORTED_FORMATS}")



def _extract_ocr_lines(ocr_page) -> tuple[list[str], list[float]]:
    """从 PaddleOCR 单页结果中抽 (文本列表, 置信度列表)。

    兼容两种返回结构：
      - PaddleOCR 3.x：OCRResult，文本在 `.json["res"]["rec_texts"]`，
        置信度在 `.json["res"]["rec_scores"]`。
      - PaddleOCR 0.x/1.x：`[[line[1][0], line[1][1]], ...]`，即 `[(text, score)]`。
    抽出后封装为统一结构，供 OCR/质量判定复用。
    """
    texts: list[str] = []
    confidences: list[float] = []
    try:
        # 3.x：dict-like OCRResult
        r = ocr_page.json.get("res") if hasattr(ocr_page, "json") else None
        if r and r.get("rec_texts"):
            texts = list(r["rec_texts"])
            confidences = list(r.get("rec_scores", []) or [])
            return [str(t) for t in texts], [float(c) for c in confidences]
    except Exception:
        pass
    # 0.x/1.x：逐行解包
    for line in ocr_page:
        try:
            texts.append(str(line[1][0]))
            confidences.append(float(line[1][1]))
        except (IndexError, TypeError):
            continue
    return texts, confidences


def _extract_ocr_boxes(ocr_page) -> list[list]:
    """抽取 OCR 边界框列表（供文字覆盖率计算）。

    兼容：
      - 3.x：OCRResult，框在 .json["res"]["rec_polys"]
      - 旧版：逐行 line[0] 即为坐标多边形
    """
    try:
        r = ocr_page.json.get("res") if hasattr(ocr_page, "json") else None
        if r and r.get("rec_polys"):
            return list(r["rec_polys"])
    except Exception:
        pass
    boxes = []
    for line in ocr_page:
        try:
            boxes.append(line[0])
        except (IndexError, TypeError):
            continue
    return boxes


def _ocr_image_file(file_path: str) -> str:
    """OCR + 按需视觉补充。OCR 结果是主体，文字覆盖率低时追加视觉描述。"""
    result = _get_ocr().ocr(file_path)
    if not result or not result[0]:
        return _vision_describe(file_path)

    texts, confidences = _extract_ocr_lines(result[0])
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    if avg_conf < OCR_CRITICAL_THRESHOLD:
        return _vision_describe(file_path)

    if avg_conf < OCR_CONFIDENCE_THRESHOLD and len(texts) < OCR_MIN_LINES:
        return _vision_describe(file_path)

    ocr_text = "\n".join(texts)

    if _is_likely_flowchart(texts, confidences):
        vision_text = _vision_describe(file_path)
        if vision_text:
            return f"[流程图/架构图分析]: {vision_text}\n\n[图中文字]: {ocr_text}"

    coverage = _calc_text_coverage(result[0], file_path)
    if coverage < OCR_COVERAGE_THRESHOLD:
        vision_text = _vision_describe(file_path)
        if vision_text:
            return ocr_text + "\n\n[视觉补充]: " + vision_text

    return ocr_text


def _calc_text_coverage(ocr_lines, file_path: str) -> float:
    """计算 OCR 文字边界框占图片总面积的百分比。
    返回值 0~1。无法计算时返回 1.0（假设全覆盖，不触发补充）。
    兼容 PaddleOCR 3.x（OCRResult，框在 json.res.rec_polys）与旧版（逐行 line[0]）。
    """
    try:
        boxes = _extract_ocr_boxes(ocr_lines)
        bbox_area = 0.0
        for coords in boxes:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            bbox_area += w * h

        from PIL import Image
        img = Image.open(file_path)
        img_area = img.width * img.height
        return bbox_area / img_area if img_area > 0 else 1.0
    except Exception:
        return 1.0


def _ocr_bytes(image_bytes: bytes) -> str:
    suffix = _detect_suffix(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return _ocr_image_file(tmp_path)
    finally:
        os.unlink(tmp_path)



def _filter_header_footer(text: str) -> str:
    """过滤常见的页眉页脚和页码模式。

    这些内容每页重复出现，会污染检索——关键词匹配到页码不是有效语义。
    """
    text = re.sub(r'^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*第\s*\d{1,4}\s*页\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page\s+\d{1,4}\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    lines = text.split('\n')
    short_line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 3 < len(stripped) < 20:
            short_line_counts[stripped] = short_line_counts.get(stripped, 0) + 1

    repeated = {k for k, v in short_line_counts.items() if v >= 3}
    if repeated:
        lines = [l for l in lines if l.strip() not in repeated]

    return '\n'.join(lines)



_FLOWCHART_KEYWORDS = [
    '流程图', '架构图', '示意图', '框图', '组织结构', '思维导图',
    '时序图', '用例图', '类图', '泳道', 'ER图', '拓扑',
    'flowchart', 'diagram', 'architecture', 'schematic',
]

def _is_likely_flowchart(ocr_texts: list[str], confidences: list[float]) -> bool:
    """检测图片是否可能是流程图/架构图。

    特征：
      - OCR 识别到短标签（平均 <15 字）→ 流程图里的节点文字
      - 关键词命中（"架构图""流程图"等）
      - 置信度参差不齐 → 手写/印刷文字和线条箭头混合
    """
    full_text = " ".join(ocr_texts)
    avg_len = sum(len(t) for t in ocr_texts) / max(len(ocr_texts), 1)

    has_keyword = any(kw in full_text.lower() for kw in _FLOWCHART_KEYWORDS)

    has_arrows = bool(re.search(r'[→←↑↓↔➔⟶⟹]|->|=>', full_text))
    is_short_labels = avg_len < 15 and len(ocr_texts) >= 3

    std_conf = 0
    if len(confidences) >= 2:
        mean = sum(confidences) / len(confidences)
        std_conf = (sum((c - mean) ** 2 for c in confidences) / len(confidences)) ** 0.5

    return has_keyword or (is_short_labels and has_arrows) or (is_short_labels and std_conf > 0.15)


def _vision_describe(image_input) -> str:
    try:
        from core.llm import LLM
        llm = LLM()
        return llm.chat_with_image("请用一段话描述这张图片的内容，尽量详细。", image_input)
    except Exception:
        return ""



def _load_pdf(file_path: str) -> str:
    all_text = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                text = page.extract_text()
                if text:
                    all_text.append(text)
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    rows = [" | ".join(str(c or "") for c in row) for row in table]
                    all_text.append(f"[PDF第{i}页表格{t_idx+1}]:\n" + "\n".join(rows))
            except Exception:
                all_text.append(f"[PDF第{i}页文本提取失败]")

    with fitz.open(file_path) as doc:
        toc = doc.get_toc()
        if toc:
            toc_lines = []
            for level, title, page in toc:
                indent = "  " * (level - 1)
                toc_lines.append(f"{indent}- {title} (第{page}页)")
            all_text.insert(0, "[文档目录]\n" + "\n".join(toc_lines))

        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (round(b[1] / 20) * 20, b[0]))
                for img in doc[page_index].get_images(full=True):
                    try:
                        ocr_text = _ocr_bytes(doc.extract_image(img[0])["image"])
                        if ocr_text:
                            all_text.append(f"[PDF第{page_index+1}页图片]: {ocr_text}")
                    except Exception:
                        all_text.append(f"[PDF第{page_index+1}页图片提取失败]")
            except Exception:
                all_text.append(f"[PDF第{page_index+1}页处理失败]")

    with fitz.open(file_path) as doc:
        headings = _extract_headings_by_font(doc)
        if headings:
            all_text.insert(0, "[文档标题层级]\n" + "\n".join(headings))

    merged = "\n\n".join(all_text)
    return _filter_header_footer(merged)



def _extract_headings_by_font(doc) -> list[str]:
    """通过字体大小推断标题层级——不依赖 ML 模型。

    fitz 的 get_text("dict") 返回每个文本块的字体大小。
    策略：收集所有字体大小 → 找最大的几种 → 按大小分 h1/h2/h3。
    比 marker/mineru 快 100 倍，对标准排版的文档足够。
    """
    import statistics

    all_spans = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    text = span["text"].strip()
                    if text:
                        all_spans.append((size, text))

    if not all_spans:
        return []

    sizes = [s for s, _ in all_spans]
    max_size = max(sizes)
    body_size = statistics.median(sizes)
    if body_size == max_size:
        return []

    headings = []
    for size, text in all_spans:
        if size > body_size * 1.2:
            if size >= max_size * 0.9:
                prefix = "## "
            elif size >= body_size * 1.5:
                prefix = "### "
            else:
                prefix = "# "
            headings.append(f"{prefix}{text}")

    seen = set()
    unique = []
    for h in headings:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique[:50]



def _load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    parts = []

    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())

    for t_idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[Word表格{t_idx+1}]:\n" + "\n".join(rows))

    for r_idx, rel in doc.part.rels.items():
        if "image" in rel.reltype:
            try:
                img_text = _ocr_bytes(rel.target_part.blob)
                if img_text:
                    parts.append(f"[Word图片]: {img_text}")
            except Exception:
                pass

    return _filter_header_footer("\n\n".join(parts))



def _load_excel(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    all_text = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_text.append(f"--- Sheet: {sheet_name} ---")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                all_text.append(" | ".join(cells))
        if hasattr(ws, "_images"):
            for img in ws._images:
                try:
                    t = _ocr_bytes(img._data())
                    if t:
                        all_text.append(f"[Excel图片]: {t}")
                except Exception:
                    pass
    return _filter_header_footer("\n".join(all_text))



def _load_pptx(file_path: str) -> str:
    prs = Presentation(file_path)
    all_text = []
    for i, slide in enumerate(prs.slides, 1):
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    if p.text.strip():
                        slide_text.append(p.text.strip())
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    t = _ocr_bytes(shape.image.blob)
                    if t:
                        slide_text.append(f"[幻灯片{i}图片]: {t}")
                except Exception:
                    pass
            if shape.has_table:
                table = shape.table
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(" | ".join(cells))
                if rows:
                    slide_text.append(f"[幻灯片{i}表格]:\n" + "\n".join(rows))
        if slide_text:
            all_text.append(f"--- 幻灯片 {i} ---")
            all_text.append("\n".join(slide_text))
    return _filter_header_footer("\n".join(all_text))



class _HTMLStripper(HTMLParser):
    """从 HTML 中提取纯文本，保留段落结构。"""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_tags = {"script", "style", "noscript", "meta", "link"}

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def handle_endtag(self, tag):
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._parts.append("\n")

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _load_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    stripper = _HTMLStripper()
    try:
        stripper.feed(content)
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return stripper.get_text()



def _load_epub(file_path: str) -> str:
    """EPUB 本质是 ZIP 包含 XHTML 文件，提取所有章节文本。"""
    all_text = []
    with zipfile.ZipFile(file_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith((".xhtml", ".html", ".htm")):
                try:
                    content = zf.read(name).decode("utf-8")
                    stripper = _HTMLStripper()
                    stripper.feed(content)
                    text = stripper.get_text()
                    if text.strip():
                        all_text.append(text)
                except Exception:
                    pass
    return _filter_header_footer("\n\n".join(all_text))



def _load_text(file_path: str) -> str:
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")



def load_with_quality(file_path: str) -> dict:
    """加载文档并返回文本 + 质量检测 + 元数据。

    Returns:
        {"text": str, "quality": dict, "file_metadata": dict,
         "page_count": int, "image_count": int}
    """
    from rag.quality_detector import detect_quality
    from rag.metadata import extract_file_metadata

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    page_count = 0
    image_count = 0
    if ext == ".pdf":
        try:
            import fitz
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                for page in doc:
                    if page.get_images():
                        image_count += 1
        except Exception:
            pass

    text = load(file_path)

    ocr_confidence = 1.0
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"):
        try:
            result = _get_ocr().ocr(file_path)
            if result and result[0]:
                confidences = [line[1][1] for line in result[0]]
                ocr_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        except Exception:
            ocr_confidence = 0.3

    quality = detect_quality(
        file_path=file_path, ext=ext, text=text,
        ocr_confidence=ocr_confidence,
        page_count=page_count, image_count=image_count,
    )

    file_meta = extract_file_metadata(file_path, ext)

    return {
        "text": text,
        "quality": quality,
        "file_metadata": file_meta,
        "page_count": page_count,
        "image_count": image_count,
    }
