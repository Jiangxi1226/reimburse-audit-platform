from sentence_transformers import SentenceTransformer
from PIL import Image


class ImageEmbedder:
    """CLIP 多模态向量化器：文字和图片 → 同一 512 维向量空间"""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def encode_image(self, image_path: str) -> list[float]:
        """图片文件 → 512 维向量。
        .convert("RGB")：PNG 可能是 RGBA（4通道），灰度图是单通道。
        CLIP 的 ViT 要求 3 通道 RGB 输入——不转换会报维度不匹配。
        """
        image = Image.open(image_path).convert("RGB")
        return self.model.encode(image).tolist()

    def encode_text(self, text: str) -> list[float]:
        """文字 → 512 维向量。
        和 encode_image 结果在同一向量空间——
        encode_text("架构图") 和 encode_image("arch.png") 的向量可以比较余弦相似度。
        """
        return self.model.encode(text).tolist()

    def encode_text_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文字编码——多条文字一次推理，GPU 并行加速"""
        return self.model.encode(texts).tolist()
