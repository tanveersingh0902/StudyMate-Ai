"""
document_processor.py — Load, split, and prepare study material for embedding.

Supported formats: PDF, TXT, DOCX, Markdown.
Uses LangChain document loaders + RecursiveCharacterTextSplitter.
"""

import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import Config


class DocumentProcessor:
    """
    Handles study material ingestion:
      1. Load raw text from file
      2. Split into overlapping chunks
      3. Return LangChain Document objects ready for embedding
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and return chunked Documents."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {Config.SUPPORTED_EXTENSIONS}"
            )

        raw_docs = self._load_raw(str(path), ext)
        chunks = self.splitter.split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        return chunks

    def load_from_bytes(self, file_bytes: bytes, file_name: str) -> List[Document]:
        """Load a file from bytes (for Streamlit UploadedFile)."""
        import tempfile
        ext = Path(file_name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def load_text(self, text: str, source_name: str = "manual_input") -> List[Document]:
        """Process a raw text string directly."""
        doc = Document(page_content=text, metadata={"source": source_name})
        return self.splitter.split_documents([doc])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_raw(self, path: str, ext: str) -> List[Document]:
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
        else:
            raise ValueError(f"Unhandled extension: {ext}")
        return loader.load()
