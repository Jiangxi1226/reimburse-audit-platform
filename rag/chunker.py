class Chunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str):
        """生成器函数——逐块产出文本，不一次性创建所有块。
        用 yield 而非 return [...]：100 万字只占一个 chunk 的内存。

        Yields:
            text[i : i + chunk_size]  —— 从位置 i 开始取 chunk_size 个字符
        """
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            yield text[i:i + self.chunk_size]
