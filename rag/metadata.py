import os, re
from datetime import datetime


# 金融财报来源权威度层级（0~1）
# 审计报告 > 正式财报 > 研报 > 经营快报 > 草稿/邮件
AUTHORITY_RULES = [
    # (文件名包含关键词, 权威度)
    ("审计", 1.0),
    ("审计报告", 1.0),
    ("年报", 0.95),
    ("财报", 0.9),
    ("定期报告", 0.9),
    ("季报", 0.85),
    ("研报", 0.7),
    ("研究报告", 0.7),
    ("经营快报", 0.6),
    ("公告", 0.6),
    ("新闻", 0.5),
    ("通讯稿", 0.4),
    ("初稿", 0.4),
    ("草稿", 0.3),
    ("邮件", 0.3),
    ("邮件内容", 0.3),
]

DEFAULT_AUTHORITY_LEVEL = 0.5


def infer_authority_level(file_path: str) -> float:
    """根据文件名推断来源权威度（金融层级）。"""
    name = os.path.basename(str(file_path)).lower()
    for keyword, level in AUTHORITY_RULES:
        if keyword in name:
            return level
    return DEFAULT_AUTHORITY_LEVEL


def extract_document_date(file_path: str, stat=None) -> str:
    """提取文档日期——优先用文件修改时间，作为冲突时'新压旧'的依据。"""
    if stat is None and os.path.exists(file_path):
        stat = os.stat(file_path)
    if stat is None:
        return ""
    return datetime.fromtimestamp(stat.st_mtime).isoformat()


# 文件名中的版本号特征（v2/v3/修订版/第二版...）
VERSION_RE = re.compile(r"([vV](\d+(?:\.\d+)?))|(修订版|第二版|第三版|新版|旧版)")
_EXTRA_VERSION_RANK = {
    "第三版": 3, "第二版": 2, "修订版": 1.5, "新版": 2.0, "旧版": 1.0,
}


def infer_version(file_path: str) -> str:
    """从文件名推断版本号，如 v2 / v2.1 / 修订版。用于冲突时'新版压旧版'。"""
    name = os.path.basename(str(file_path))
    m = VERSION_RE.search(name)
    if not m:
        return ""
    if m.group(1):  # v2 / v2.1
        return m.group(2)
    return m.group(3) or m.group(4) or m.group(5) or ""


def version_rank(version: str) -> float:
    """版本号转可比较数值：v2.1 -> 2.1，修订版 -> 1.5。无版本 -> 0。"""
    if not version:
        return 0.0
    if version in _EXTRA_VERSION_RANK:
        return _EXTRA_VERSION_RANK[version]
    try:
        return float(version)
    except ValueError:
        return 0.0


def extract_file_metadata(file_path: str, ext: str) -> dict:
    """提取文件级元数据（含来源权威度 + 文档时间戳 + 版本，供冲突解决用）。"""
    stat = os.stat(file_path) if os.path.exists(file_path) else None
    file_name = os.path.basename(file_path)

    return {
        "file_name": file_name,
        "file_path": file_path,
        "file_type": ext.lstrip(".").upper(),
        "file_size_kb": round(stat.st_size / 1024, 1) if stat else 0,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat() if stat else "",
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
        "document_date": extract_document_date(file_path, stat),
        "effective_date": extract_document_date(file_path, stat),
        "authority_level": infer_authority_level(file_path),
        "version": infer_version(file_path),
    }


def extract_page_metadata(page_num: int, total_pages: int,
                          has_images: bool = False,
                          has_tables: bool = False,
                          section_title: str = "") -> dict:
    """提取页面级元数据。"""
    return {
        "page": page_num,
        "total_pages": total_pages,
        "has_images": has_images,
        "has_tables": has_tables,
        "section": section_title,
        "location": f"第{page_num}页/共{total_pages}页" + (f" · {section_title}" if section_title else ""),
    }


def build_chunk_metadata(file_meta: dict, page_meta: dict = None,
                          chunk_idx: int = 0, quality_score: float = 1.0,
                          chunk_type: str = "text",
                          custom_tags: list[str] = None,
                          authority_level: float = None,
                          document_date: str = None,
                          effective_date: str = None,
                          version: str = None) -> dict:
    """构建单个 chunk 的完整元数据——存入 ChromaDB metadata。

    ChromaDB 的 metadata 支持过滤查询：
      collection.query(where={"file_type": "PDF"})
      collection.query(where={"page": 3})
      collection.query(where={"quality_score": {"$gte": 0.7}})

    金融领域新增：
      authority_level: 来源权威度 0~1（审计>财报>研报>快报>草稿）
      document_date:   文档创建日期（冲突时"新压旧"依据）
      effective_date:  生效日期（旧制度 vs 新制度时，新的生效日期优先）
      version:         版本号（v2>v1，修订版优先于旧版）
    """
    meta = {
        "file_name": file_meta.get("file_name", ""),
        "file_type": file_meta.get("file_type", ""),
        "source_path": file_meta.get("file_path", ""),
        "chunk_idx": chunk_idx,
        "chunk_type": chunk_type,
        "quality_score": quality_score,
        "tags": ",".join(custom_tags) if custom_tags else "",
        "indexed_at": datetime.now().isoformat(),
        "authority_level": authority_level if authority_level is not None
                           else file_meta.get("authority_level", 0.5),
        "document_date": document_date if document_date is not None
                         else file_meta.get("document_date", ""),
        "effective_date": effective_date if effective_date is not None
                          else file_meta.get("effective_date",
                                             file_meta.get("document_date", "")),
        "version": version if version is not None
                   else file_meta.get("version", ""),
    }

    if page_meta:
        meta.update({
            "page": page_meta.get("page", 0),
            "section": page_meta.get("section", ""),
            "location": page_meta.get("location", ""),
            "has_images": page_meta.get("has_images", False),
            "has_tables": page_meta.get("has_tables", False),
        })

    return meta


def build_metadata_filter(file_type: str = None,
                           min_page: int = None,
                           max_page: int = None,
                           min_quality: float = None,
                           tags: list[str] = None) -> dict:
    """构建 ChromaDB 元数据过滤条件。

    用法：
      pipeline.search("Q3营收", metadata_filter=build_metadata_filter(
          file_type="PDF", min_quality=0.7
      ))
    """
    filter_dict = {}

    if file_type:
        filter_dict["file_type"] = file_type.upper()

    if min_page is not None:
        filter_dict["page"] = {"$gte": min_page}

    if min_quality is not None:
        filter_dict["quality_score"] = {"$gte": min_quality}

    if tags:
        filter_dict["tags"] = {"$contains": tags[0]} if len(tags) == 1 else None

    conditions = []
    for k, v in filter_dict.items():
        if v is not None:
            if isinstance(v, dict):
                conditions.append({k: v})
            else:
                conditions.append({k: {"$eq": v}})

    if len(conditions) == 0:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def format_citation(chunk_metadata: dict, text_snippet: str = "") -> str:
    """格式化引用文本——用于答案中的来源标注。

    Returns:
        如 "📎 Q3营收报告.pdf 第3页 · Q3业绩分析" 或
           "📎 [表1] Q3营收报告.pdf · 产品营收对比表(5行) · 第2页"
    """
    parts = []

    if chunk_metadata.get("chunk_type") == "table":
        table_summary = chunk_metadata.get("table_summary", "")
        parts.append(f"[表] {table_summary}" if table_summary else "[表]")

    file_name = chunk_metadata.get("file_name", "")
    if file_name:
        parts.append(f"📎 {file_name}")

    location = chunk_metadata.get("location", "")
    if location:
        parts.append(location)
    elif chunk_metadata.get("page"):
        parts.append(f"第{chunk_metadata['page']}页")

    if chunk_metadata.get("quality_score", 1.0) < 0.5:
        parts.append("⚠️低质量")

    authority = chunk_metadata.get("authority_level")
    if authority is not None:
        parts.append(f"权威度{authority:.1f}")

    citation = " · ".join(parts)
    if text_snippet:
        citation += f"\n> {text_snippet[:150]}..."

    return citation
