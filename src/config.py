"""
config.py — Central configuration for StudyMate AI
Supports Groq (LLaMA) and Google Gemini models.
All settings read from environment variables with sensible defaults.

Model IDs (verified from official docs):
  Groq   → llama-3.3-70b-versatile   (Meta LLaMA 3.3 70B on Groq LPUs)
            llama-3.1-8b-instant      (smaller/faster LLaMA 3.1 8B)
  Gemini → gemini-2.0-flash           (latest fast model)
            gemini-1.5-flash           (previous gen, high throughput)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Placeholder strings that ship in .env.example — treat as "no key set"
_PLACEHOLDER_KEYS = {
    "",
    "your_groq_api_key_here",
    "your_gemini_api_key_here",
}


class Config:
    # ── LLM Provider Selection ────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")   # "groq" or "gemini"

    # ── Groq — LLaMA models (Meta open-weight, hosted on Groq LPUs) ──────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Google Gemini ─────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Shared LLM Settings ───────────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
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
        """Return the API key for the currently selected provider.
        Returns empty string if the key is a placeholder or missing."""
        if cls.LLM_PROVIDER == "gemini":
            key = cls.GEMINI_API_KEY
        else:
            key = cls.GROQ_API_KEY
        return "" if key in _PLACEHOLDER_KEYS else key

    @classmethod
    def validate(cls) -> bool:
        """Check that required API keys are present for chosen provider."""
        if cls.LLM_PROVIDER == "gemini":
            if not cls.get_active_api_key():
                raise ValueError(
                    "GEMINI_API_KEY is missing. "
                    "Get a free key at https://aistudio.google.com/app/apikey"
                )
        else:
            if not cls.get_active_api_key():
                raise ValueError(
                    "GROQ_API_KEY is missing. "
                    "Get a free key at https://console.groq.com and add it to .env"
                )
        return True
