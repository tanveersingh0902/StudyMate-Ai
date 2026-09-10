"""
vector_store.py — FAISS-based vector store with HuggingFace embeddings.

Stores study material chunks as vectors for fast semantic search.
"""

import os
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .config import Config


class VectorStore:
    """
    Wraps FAISS + HuggingFace embeddings.
    Provides add_documents() and similarity_search() for the agent pipeline.
    """

    def __init__(self):
        print("⚙️  Loading embedding model (first run downloads ~90 MB)…")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store: Optional[FAISS] = None

    # ── Building the index ────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and index a list of Document chunks.
        Merges into existing index if one already exists (incremental upload).
        Returns total document count in the index.
        """
        if not documents:
            raise ValueError("Cannot index an empty document list.")

        new_store = FAISS.from_documents(documents, self.embeddings)

        if self._store is None:
            self._store = new_store
        else:
            self._store.merge_from(new_store)

        # Count documents by iterating over the index-to-docstore-id mapping
        return self._store.index.ntotal

    def clear(self):
        """Wipe the in-memory index."""
        self._store = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = Config.VECTOR_STORE_PATH):
        if self._store is None:
            raise RuntimeError("No documents indexed yet — nothing to save.")
        os.makedirs(path, exist_ok=True)
        self._store.save_local(path)

    def load(self, path: str = Config.VECTOR_STORE_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved index at: {path}")
        self._store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = Config.MAX_RETRIEVAL_DOCS) -> List[Document]:
        """Return the top-k most relevant document chunks for the query."""
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, k: int = Config.MAX_RETRIEVAL_DOCS):
        if self._store is None:
            return None
        return self._store.as_retriever(search_kwargs={"k": k})

    @property
    def is_ready(self) -> bool:
        return self._store is not None
