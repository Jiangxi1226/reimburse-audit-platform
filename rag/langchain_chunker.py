from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainChunker:
    """LangChain RecursiveCharacterTextSplitter 的轻量包装。

    接口和自研 Chunker / SemanticChunker 保持一致——split(text) 返回字符串列表。
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
