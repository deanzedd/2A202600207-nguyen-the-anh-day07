from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu trữ reference tới vector store và hàm gọi LLM
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not question.strip():
            return "Vui lòng đặt một câu hỏi hợp lệ."

        # 1. Retrieve top-k relevant chunks from the store.
        # Gọi hàm search từ EmbeddingStore đã viết ở file trước
        search_results = self.store.search(query=question, top_k=top_k)

        # Trích xuất nội dung (content) từ các chunks tìm được
        if not search_results:
            context_text = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        else:
            # Lấy value của key 'content' và nối chúng lại với nhau
            chunks = [result.get("content", "") for result in search_results]
            context_text = "\n\n---\n\n".join(chunks)

        # 2. Build a prompt with the chunks as context.
        # Tạo prompt tiêu chuẩn cho mô hình RAG, yêu cầu LLM chỉ trả lời dựa trên context
        prompt = (
            "You are a helpful and accurate assistant. Use the following Context to answer the Question.\n"
            "If the answer is not contained within the Context, simply state that you don't have enough "
            "information to answer based on the provided documents. Do not hallucinate.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )

        # 3. Call the LLM to generate an answer.
        # Truyền prompt đã format vào hàm llm_fn được tiêm (inject) từ bên ngoài
        try:
            response = self.llm_fn(prompt)
            return response
        except Exception as e:
            return f"Lỗi trong quá trình gọi LLM: {str(e)}"
