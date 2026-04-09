"""
Day 7 — Data Foundations: Embedding & Vector Store
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change class/function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

Note on dependencies:
    ChromaDB is optional. If not installed, EmbeddingStore falls back to an
    in-memory implementation using dot-product similarity.
    Install ChromaDB with: pip install chromadb
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Document
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    A text document with optional metadata.

    Fields:
        id:       Unique identifier string.
        content:  The raw text content.
        metadata: Arbitrary key-value metadata (e.g. source, date, author).
    """
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task 2 — Chunking Functions
# ---------------------------------------------------------------------------

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text:       The input text.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters to repeat at the start of the next chunk.

    Returns:
        List of chunk strings.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share `overlap` characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].

    Example (chunk_size=10, overlap=2):
        "abcdefghijklmnop" →
        ["abcdefghij", "ijklmnopqr", "qrstuv..."]
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_by_sentences(text: str, max_sentences_per_chunk: int = 3) -> list[str]:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Args:
        text:                    The input text.
        max_sentences_per_chunk: Maximum sentences per chunk.

    Returns:
        List of chunk strings.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.

    Example (max_sentences=2):
        "A. B. C. D." → ["A. B.", "C. D."]
    """
    # TODO: split into sentences, group into chunks
    raise NotImplementedError("Implement chunk_by_sentences")


def chunk_recursive(
    text: str,
    separators: list[str] | None = None,
    chunk_size: int = 500,
) -> list[str]:
    """
    Recursively split text using a list of separators in priority order.

    Args:
        text:       The input text.
        separators: List of separator strings tried in order.
                    Default: ["\n\n", "\n", ". ", " ", ""]
        chunk_size: Target maximum chunk length in characters.

    Returns:
        List of chunk strings, each at most chunk_size characters.

    Algorithm:
        1. Try the first separator.
        2. Split the text on it.
        3. If any split produces a piece longer than chunk_size,
           recursively split that piece using the remaining separators.
        4. If no separators remain, return text as-is (fallback).
    """
    # TODO: implement recursive splitting
    raise NotImplementedError("Implement chunk_recursive")


# ---------------------------------------------------------------------------
# Mock embedding function (used by tests to avoid API keys)
# ---------------------------------------------------------------------------

def _mock_embed(text: str, dim: int = 64) -> list[float]:
    """
    Generate a deterministic pseudo-random embedding vector.
    Based on MD5 hash of text, so same text → same vector.
    """
    h = hashlib.md5(text.encode()).hexdigest()
    seed = int(h, 16)
    vec = []
    for i in range(dim):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        vec.append((seed / 0xFFFFFFFF) * 2 - 1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Args:
        vec_a: First vector as list of floats
        vec_b: Second vector as list of floats (same length as vec_a)

    Returns:
        Cosine similarity as float in range [-1, 1].
        Returns 0.0 if either vector has zero magnitude.

    TODO: Implement cosine similarity formula
    """
    raise NotImplementedError


def compare_chunking_strategies(text: str, chunk_size: int = 200) -> dict:
    """Run all three chunking strategies and compare results.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size (characters for fixed, sentences for sentence-based)

    Returns:
        dict with keys for each strategy:
          {
            'fixed_size': {'count': int, 'avg_length': float, 'chunks': list[str]},
            'by_sentences': {'count': int, 'avg_length': float, 'chunks': list[str]},
            'recursive': {'count': int, 'avg_length': float, 'chunks': list[str]},
          }

    TODO: Call each chunking function, compute stats, return comparison dict
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 3 — Embedding Store
# ---------------------------------------------------------------------------

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
        """
        Args:
            collection_name: Name for the ChromaDB collection (if used).
            embedding_fn:    Function str → list[float].
                             Defaults to _mock_embed (for tests) or a real API
                             embedding function in production.
        """
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        # TODO: try to initialise ChromaDB; fall back to in-memory list
        # In-memory fallback: self._store = []  (list of dicts with 'id','content','embedding')
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []

        try:
            import chromadb  # noqa: F401
            # TODO: initialise chromadb.Client() and get/create collection
            self._use_chroma = True
        except ImportError:
            pass  # Use in-memory fallback

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        Args:
            docs: List of Document objects to index.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        raise NotImplementedError("Implement add_documents")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        Args:
            query: The search string.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts, each with at least:
                'content' (str)  — the chunk text
                'score'   (float) — similarity score (higher = more similar)
            Sorted by score descending.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        raise NotImplementedError("Implement search")

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        raise NotImplementedError("Implement get_collection_size")

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter (all key-value pairs must match),
        then run similarity search on the filtered set.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            metadata_filter: Dict of metadata key-value pairs that must match.
                             None means no filtering.

        Returns:
            List of dicts with 'content', 'score', 'metadata' keys

        TODO: Filter by metadata, then search among filtered chunks
        """
        raise NotImplementedError

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document.

        Args:
            doc_id: Document ID to remove (matches Document.id used during add_documents)

        Returns:
            True if any chunks were removed, False if doc_id not found

        TODO: Remove all stored chunks where metadata['doc_id'] == doc_id
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 4 — Knowledge Base Agent
# ---------------------------------------------------------------------------

class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        pass

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using RAG.

        Args:
            question: The user's question.
            top_k:    Number of context chunks to retrieve.

        Returns:
            The LLM's answer string.
        """
        # TODO: retrieve chunks, build prompt, call llm_fn
        raise NotImplementedError("Implement KnowledgeBaseAgent.answer")


# ---------------------------------------------------------------------------
# Helpers for manual testing
# ---------------------------------------------------------------------------

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """
    Load documents from a list of file paths.

    Only .md and .txt files are accepted for the manual demo.
    Missing files are skipped with a message.
    """
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_files = [
        "data/python_intro.txt",
        "data/vector_store_notes.md",
        "data/rag_system_design.md",
        "data/customer_support_playbook.txt",
        "data/chunking_experiment_report.md",
        "data/vi_retrieval_notes.md",
    ]
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in sample_files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(sample_files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 solution/solution.py")
    else:
        print(f"\nLoaded {len(docs)} documents")
        for doc in docs:
            print(f"  - {doc.id}: {doc.metadata['source']}")

        store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=_mock_embed)
        store.add_documents(docs)

        print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")

        query = question
        print(f"\n=== EmbeddingStore Search Test ===")
        print(f"Query: {query}")
        search_results = store.search(query, top_k=3)
        for index, result in enumerate(search_results, start=1):
            print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
            print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

        print("\n=== KnowledgeBaseAgent Test ===")
        agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
        print(f"Question: {question}")
        answer = agent.answer(question, top_k=3)
        print("Agent answer:")
        print(answer)
