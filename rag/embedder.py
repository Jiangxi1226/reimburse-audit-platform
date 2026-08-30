from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        """单条文本 → 384 维浮点数向量。
        内部流程：Tokenizer（分词+token ID）→ 12层 Transformer → mean pooling → 向量
        .tolist()：把 numpy array(384,) 转成 Python list(384)
        """
        return self.model.encode(text).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码——多条文本打包一次推理，GPU 并行加速。
        100 个 chunk 一起编码比逐个调用 encode() 快 5~10 倍。
        """
        return self.model.encode(texts).tolist()
