import re


class SemanticChunker:
    """语义感知分块器——递进式切分，每块保证语义完整性。"""

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        # 防御：overlap 必须严格小于 chunk_size，否则 _split_by_length 里
        # start = end - overlap 可能不回退甚至倒退，导致死循环/重复块。
        if overlap >= chunk_size:
            raise ValueError(f"overlap({overlap}) 必须小于 chunk_size({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        """递进切分：段落 → 句子 → 固定长度。"""
        paragraphs = self._split_by_paragraph(text)

        chunks = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                sentences = self._split_by_sentence(para)
                for sent in sentences:
                    if len(sent) <= self.chunk_size:
                        chunks.append(sent)
                    else:
                        for fixed_chunk in self._split_by_length(sent):
                            chunks.append(fixed_chunk)

        return self._merge_short(chunks)

    def _split_by_paragraph(self, text: str) -> list[str]:
        """按自然段落切分——双换行、标题标记。"""
        parts = re.split(r'\n\s*\n', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_sentence(self, text: str) -> list[str]:
        """按句子边界切分——句号、问号、感叹号、换行。"""
        parts = re.split(r'(?<=[。！？\n])', text)
        merged = []
        buffer = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) < 10 and buffer:
                buffer += part
            else:
                if buffer:
                    merged.append(buffer.strip())
                buffer = part
        if buffer:
            merged.append(buffer.strip())
        return merged

    def _split_by_length(self, text: str) -> list[str]:
        """固定长度切分——兜底策略。但在空白处切，不切断词。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            chunk = text[start:end]
            safe_pos = self._find_safe_break(chunk)
            if safe_pos > self.chunk_size // 2:
                end = start + safe_pos
            chunks.append(text[start:end].strip())
            start = end - self.overlap
        return [c for c in chunks if c]

    def _find_safe_break(self, text: str) -> int:
        """在文本中找安全的断点——空格、标点后的位置。从后往前找。"""
        safe_chars = ['。', '！', '？', '\n', '，', '；', '、', ' ', '.', '!', '?', ',']
        for i in range(len(text) - 1, len(text) // 2, -1):
            if text[i] in safe_chars:
                return i + 1
        return len(text)

    def _merge_short(self, chunks: list[str]) -> list[str]:
        """合并过短的相邻块——防止一句话被切得七零八落。"""
        if not chunks:
            return []
        merged = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) + len(chunk) < self.chunk_size * 1.2:
                buffer += chunk
            else:
                merged.append(buffer)
                buffer = chunk
        merged.append(buffer)
        return [c for c in merged if c]



def demo():
    """对比固定长度切分和语义切分的差异。"""
    text = (
        "Transformer架构由Vaswani等人在2017年提出。"
        "它抛弃了传统的循环神经网络结构，完全基于注意力机制。\n\n"
        "核心创新是自注意力机制：每个输入token都会和序列中的所有token"
        "计算注意力权重，从而捕获长距离依赖关系。这解决了RNN的长期记忆问题。"
    )

    from rag.chunker import Chunker as FixedChunker

    print("=== 固定长度切分 (chunk_size=50, overlap=10) ===")
    for i, c in enumerate(FixedChunker(50, 10).split(text)):
        print(f"  [{i}] {c}")

    print("\n=== 语义切分 (chunk_size=50, overlap=10) ===")
    for i, c in enumerate(SemanticChunker(50, 10).split(text)):
        print(f"  [{i}] {c}")


if __name__ == "__main__":
    demo()
