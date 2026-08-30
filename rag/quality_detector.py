import re
from enum import Enum


class DocQuality(Enum):
    CLEAN = "clean"
    SCANNED = "scanned"
    OCR_DEGRADED = "ocr_degraded"
    MIXED = "mixed"
    CORRUPTED = "corrupted"


def detect_quality(file_path: str, ext: str, text: str = "",
                   ocr_confidence: float = 1.0,
                   page_count: int = 0,
                   image_count: int = 0) -> dict:
    """检测文档质量并返回分级 + 处理策略。

    Args:
        file_path: 文件路径
        ext: 扩展名（含点，如 ".pdf"）
        text: 已提取的文本内容
        ocr_confidence: OCR 平均置信度（0~1，非OCR文档为1.0）
        page_count: 总页数
        image_count: 图片页数

    Returns:
        {"quality": DocQuality, "score": 0~1, "strategy": str, "warnings": [...]}
    """
    quality = DocQuality.CLEAN
    score = 1.0
    warnings = []

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"):
        if ocr_confidence < 0.5:
            quality = DocQuality.OCR_DEGRADED
            score = ocr_confidence
            warnings.append(f"图片OCR置信度仅 {ocr_confidence:.0%}，可能存在识别错误")
        else:
            quality = DocQuality.SCANNED
            score = 0.7
        return _result(quality, score, warnings)

    if ext == ".pdf":
        if page_count > 0 and image_count > page_count * 0.8:
            quality = DocQuality.SCANNED
            score = 0.6
            warnings.append(f"PDF 中 {image_count}/{page_count} 页为图片，疑似扫描件")
        elif image_count > 0:
            quality = DocQuality.MIXED
            score = 0.75
            warnings.append(f"PDF 含 {image_count} 张图片，需图文分路径处理")
        elif not text or len(text.strip()) < 50:
            quality = DocQuality.CORRUPTED
            score = 0.0
            warnings.append("PDF 无法提取有效文本内容")

        if ocr_confidence < 0.5:
            quality = DocQuality.OCR_DEGRADED
            score = min(score, ocr_confidence)

        return _result(quality, score, warnings)

    if ext in (".docx", ".xlsx", ".pptx"):
        if not text or len(text.strip()) < 100:
            quality = DocQuality.CORRUPTED
            score = 0.0
            warnings.append("文档内容为空或过短，可能已损坏")
        elif _has_garbled_text(text):
            quality = DocQuality.OCR_DEGRADED
            score = 0.4
            warnings.append("检测到乱码字符，可能是编码问题或转换异常")
        return _result(quality, score, warnings)

    if ext in (".txt", ".md", ".py", ".json", ".csv"):
        if not text or not text.strip():
            quality = DocQuality.CORRUPTED
            score = 0.0
            warnings.append("文件为空")
        return _result(quality, score, warnings)

    return _result(quality, score, warnings)


def get_processing_strategy(quality: DocQuality, score: float) -> dict:
    """根据质量分级返回处理策略。"""
    strategies = {
        DocQuality.CLEAN: {
            "chunk_mode": "multi",
            "ocr_required": False,
            "metadata_track": True,
            "auto_publish": True,
        },
        DocQuality.SCANNED: {
            "chunk_mode": "single",
            "ocr_required": True,
            "metadata_track": True,
            "auto_publish": True,
        },
        DocQuality.OCR_DEGRADED: {
            "chunk_mode": "single",
            "ocr_required": True,
            "metadata_track": True,
            "auto_publish": False,
            "warning_tag": "⚠️ 此文档 OCR 质量较低，内容可能存在识别错误",
        },
        DocQuality.MIXED: {
            "chunk_mode": "multi",
            "ocr_required": False,
            "metadata_track": True,
            "auto_publish": True,
            "split_paths": True,
        },
        DocQuality.CORRUPTED: {
            "chunk_mode": "none",
            "ocr_required": False,
            "metadata_track": False,
            "auto_publish": False,
            "reject": True,
        },
    }
    base = strategies.get(quality, strategies[DocQuality.CLEAN])
    base["quality"] = quality.value
    base["score"] = score
    return base


def _has_garbled_text(text: str, threshold: float = 0.15) -> bool:
    """检测文本中是否存在大量乱码字符。"""
    if not text:
        return False
    garbled = len(re.findall(r'[^\x20-\x7E一-鿿　-〿＀-￯\n\r\t]', text))
    return garbled / max(len(text), 1) > threshold


def _result(quality: DocQuality, score: float, warnings: list) -> dict:
    return {
        "quality": quality,
        "score": round(score, 2),
        "strategy": get_processing_strategy(quality, score),
        "warnings": warnings,
    }
