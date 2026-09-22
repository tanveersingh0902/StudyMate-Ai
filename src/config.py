"""
config.py — Central configuration for StudyMate AI
Supports Groq (GPT-OSS) and Google Gemini models.
All settings read from environment variables with sensible defaults.

Model IDs (verified from official docs, August 2026):
  Groq  → openai/gpt-oss-120b  (OpenAI open-weight flagship on Groq hardware)
           openai/gpt-oss-20b   (smaller/faster GPT-OSS on Groq)
  Gemini → gemini-3.7-flash     (latest stable)
            gemini-3.6-flash     (previous stable)
            gemini-3.5-flash     (legacy stable)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets first, then env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


class Config:
    # ── LLM Provider Selection ────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")   # "groq" or "gemini"
    LLM_PROVIDER: str = _get_secret("LLM_PROVIDER", "groq")   # "groq" or "gemini"

    # ── Groq — GPT-OSS models (OpenAI open-weight, hosted on Groq LPUs) ──────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")   # flagship 120B
    GROQ_API_KEY: str = _get_secret("GROQ_API_KEY", "")
    GROQ_MODEL: str = _get_secret("GROQ_MODEL", "openai/gpt-oss-120b")   # flagship 120B

    # ── Google Gemini ─────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")   # latest stable
    GEMINI_API_KEY: str = _get_secret("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = _get_secret("GEMINI_MODEL", "gemini-3.7-flash")   # latest stable

    # ── Shared LLM Settings ───────────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # ── Embeddings (HuggingFace — free, runs locally) ─────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Document Processing ───────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".txt", ".docx", ".md")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")

    # ── Agent Control ─────────────────────────────────────────────────────────
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

    # ── Study / Quiz Settings ─────────────────────────────────────────────────
    DEFAULT_DIFFICULTY: str = "intermediate"   # beginner / intermediate / advanced
    QUIZ_QUESTION_COUNT: int = 5
    STUDY_PLAN_DAYS: int = 7

    @classmethod
    def get_active_api_key(cls) -> str:
        """Return the API key for the currently selected provider."""
        if cls.LLM_PROVIDER == "gemini":
            return cls.GEMINI_API_KEY
        return cls.GROQ_API_KEY

    @classmethod
    def validate(cls) -> bool:
        """Check that required API keys are present for chosen provider."""
        if cls.LLM_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY is missing. "
                    "Add it to Streamlit secrets (on Streamlit Cloud) or your .env file. "
                    "Get a free key at https://aistudio.google.com/app/apikey"
                )
        else:
            if not cls.GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY is missing. "
                    "Get a free key at https://console.groq.com and add it to .env"
                    "Add it to Streamlit secrets (on Streamlit Cloud) or your .env file. "
                    "Get a free key at https://console.groq.com"
                )
        return True

    @classmethod
    def reset(cls):
        """Reset mutable settings to their defaults (used by Clear Session)."""
        cls.MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
        cls.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
        cls.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
        cls.QUIZ_QUESTION_COUNT = 5
        cls.STUDY_PLAN_DAYS = 7
