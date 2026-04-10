from __future__ import annotations
import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # Initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        # Lấy metadata và nội dung từ đối tượng Document một cách an toàn
        content = getattr(doc, "content", getattr(doc, "text", getattr(doc, "page_content", "")))
        metadata = getattr(doc, "metadata", {})
        
        # Tạo ID duy nhất nếu doc không có
        doc_id = getattr(doc, "id", str(uuid.uuid4()))
        
        return {
            "id": doc_id,
            "content": content,
            "embedding": self._embedding_fn(content),
            "metadata": metadata
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []

        query_emb = self._embedding_fn(query)
        scored_records = []
        
        for record in records:
            # Tính độ tương đồng bằng tích vô hướng (Dot Product)
            score = _dot(query_emb, record["embedding"])
            # Gắn thêm trường score vào kết quả trả về
            result_record = record.copy()
            result_record["score"] = score
            scored_records.append((score, result_record))
            
        # Sắp xếp giảm dần theo điểm số
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về top_k kết quả
        return [rec for score, rec in scored_records[:top_k]]
    
    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection:
            ids = [rec["id"] for rec in records]
            documents = [rec["content"] for rec in records]
            embeddings = [rec["embedding"] for rec in records]
            metadatas = [rec["metadata"] for rec in records]
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            
            # Format kết quả từ Chroma về chuẩn chung giống in-memory
            formatted_results = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "score": 1.0 - distance if isinstance(distance, (int, float)) else 0.0,
                    })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        metadata_filter = metadata_filter or {}

        if self._use_chroma and self._collection:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter  # Chroma hỗ trợ dict filter
            )
            
            formatted_results = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })
            return formatted_results
        else:
            # In-memory: Lọc theo metadata trước
            filtered_records = []
            for rec in self._store:
                meta = rec.get("metadata", {})
                # Kiểm tra tất cả key-value trong bộ lọc phải khớp với metadata của record
                match = all(meta.get(k) == v for k, v in metadata_filter.items())
                if match:
                    filtered_records.append(rec)
                    
            # Tìm kiếm độ tương đồng trên các chunks đã được lọc
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection:
            initial_count = self._collection.count()
            self._collection.delete(
                where={"doc_id": doc_id}
            )
            return self._collection.count() < initial_count
        else:
            initial_count = len(self._store)
            # Lọc bỏ các chunk có id hoặc metadata.doc_id tương ứng
            self._store = [
                rec for rec in self._store
                if rec.get("id") != doc_id and rec.get("metadata", {}).get("doc_id") != doc_id
            ]
            return len(self._store) < initial_count
