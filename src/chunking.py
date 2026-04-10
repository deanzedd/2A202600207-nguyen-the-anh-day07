from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Sử dụng Regex để tách nhưng vẫn giữ lại dấu câu (delimiter)
        # Các dấu câu để tách: ". ", "! ", "? ", ".\n"
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        
        sentences = []
        # Nối lại phần text và dấu câu đi kèm của nó
        for i in range(0, len(parts), 2):
            sent = parts[i]
            if i + 1 < len(parts):
                sent += parts[i+1]
            sent = sent.strip()
            if sent:  # Chỉ lấy các câu có nội dung
                sentences.append(sent)

        # Gộp các câu thành từng chunk theo giới hạn max_sentences_per_chunk
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sents = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sents))
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Trả về luôn nếu text đã nằm trong giới hạn
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Tìm separator phù hợp nhất (đầu tiên) có trong text
        separator = ""
        for sep in remaining_separators:
            if sep == "" or sep in current_text:
                separator = sep
                break

        # Nếu separator là chuỗi rỗng (không thể chia theo logic thông thường), 
        # bắt buộc phải cắt cứng theo độ dài (Fixed Size)
        if separator == "":
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Lấy danh sách separators tiếp theo để chuẩn bị đệ quy
        next_separators = remaining_separators[remaining_separators.index(separator) + 1:] if separator in remaining_separators else []

        # Tách chuỗi theo separator hiện tại
        splits = current_text.split(separator)
        
        final_chunks = []
        current_chunk = []
        current_len = 0

        for i, split in enumerate(splits):
            # Khôi phục lại separator (ngoại trừ phần tử cuối cùng)
            part = split + (separator if i < len(splits) - 1 else "")
            
            # Nếu riêng mảnh này đã vượt quá chunk_size, đẩy các mảnh đang gom vào final và đệ quy mảnh lớn
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Đệ quy mảnh lớn
                sub_chunks = self._split(part, next_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Gộp mảnh nếu tổng chiều dài chưa vượt chunk_size
                if current_len + len(part) > self.chunk_size:
                    final_chunks.append("".join(current_chunk))
                    current_chunk = [part]
                    current_len = len(part)
                else:
                    current_chunk.append(part)
                    current_len += len(part)

        # Đừng quên phần tử cuối cùng nếu còn sót lại
        if current_chunk:
            final_chunks.append("".join(current_chunk))

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if not text:
            return {}

        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_result = fixed_chunker.chunk(text)
        sentence_result = sentence_chunker.chunk(text)
        recursive_result = recursive_chunker.chunk(text)

        return {
            "fixed_size": {
                "count": len(fixed_result),
                "avg_length": sum(len(c) for c in fixed_result) / len(fixed_result) if fixed_result else 0,
                "chunks": fixed_result,
            },
            "by_sentences": {
                "count": len(sentence_result),
                "avg_length": sum(len(c) for c in sentence_result) / len(sentence_result) if sentence_result else 0,
                "chunks": sentence_result,
            },
            "recursive": {
                "count": len(recursive_result),
                "avg_length": sum(len(c) for c in recursive_result) / len(recursive_result) if recursive_result else 0,
                "chunks": recursive_result,
            },
        }
