"""
llm_factory.py — Unified LLM factory supporting Groq (GPT-OSS) and Google Gemini.

Usage:
    from src.llm_factory import make_llm, call_llm

    llm = make_llm()                          # Uses Config.LLM_PROVIDER
    response = call_llm(system, user)        # One-shot call, returns string

Production features:
  - Retry with exponential backoff for transient API errors (429 / 503)
  - Distinct error messages for auth failures vs rate limits
  - Normalises multi-part response content to plain string
"""

import time
import logging
from functools import wraps

from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config

logger = logging.getLogger(__name__)

# ── Retry Decorator ───────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


def _retry_on_transient(func):
    """Retry LLM calls on transient errors (rate-limit / server overload)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()

                # Auth errors — don't retry, fail fast
                if any(tok in err_str for tok in ("401", "403", "invalid api key",
                                                   "authentication", "unauthorized")):
                    raise RuntimeError(
                        f"🔑 Authentication failed for {Config.LLM_PROVIDER.upper()}. "
                        f"Please check your API key in the sidebar."
                    ) from e

                # Model-not-found errors — don't retry
                if "404" in err_str or "model not found" in err_str:
                    raise RuntimeError(
                        f"❌ Model not found: {Config.LLM_MODEL}. "
                        f"Please select a valid model."
                    ) from e

                # Transient errors — retry with backoff
                if any(tok in err_str for tok in ("429", "503", "rate limit",
                                                   "overloaded", "capacity",
                                                   "timeout", "timed out")):
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"Transient error (attempt {attempt}/{MAX_RETRIES}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                    continue

                # Unknown errors on first attempts — retry once
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"Unexpected error (attempt {attempt}/{MAX_RETRIES}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                    continue

                raise
        raise last_exc
    return wrapper


# ── LLM Factory ───────────────────────────────────────────────────────────────

def make_llm(temperature: float = None, max_tokens: int = None):
    """
    Create and return an LLM instance based on Config.LLM_PROVIDER.

    Returns a LangChain-compatible chat model (Groq or Gemini).
    Temperature and max_tokens default to Config values if not supplied.
    """
    temp = temperature if temperature is not None else Config.TEMPERATURE
    tokens = max_tokens if max_tokens is not None else Config.MAX_TOKENS

    if Config.LLM_PROVIDER == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            google_api_key=Config.GEMINI_API_KEY,
            model=Config.GEMINI_MODEL,
            temperature=temp,
            max_output_tokens=tokens,   # Gemini uses max_output_tokens (not max_tokens)
        )
    else:
        # Default: Groq (GPT-OSS)
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=temp,
            max_tokens=tokens,
        )


@_retry_on_transient
def call_llm(system: str, user: str, temperature: float = None) -> str:
    """
    Make a single LLM call and return the response as a plain string.

    Args:
        system: The system/instruction prompt.
        user:   The user message / question.
        temperature: Override temperature (optional).

    Returns:
        The model's response as a stripped string.

    Raises:
        RuntimeError: On auth failures or model-not-found (no retry).
        Exception: On persistent transient errors (after MAX_RETRIES).
    """
    llm = make_llm(temperature=temperature)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    # .content can be a list of dicts for some providers — normalise to str
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def get_provider_name() -> str:
    """Return a human-readable name for the current LLM provider."""
    if Config.LLM_PROVIDER == "gemini":
        return f"Google Gemini ({Config.GEMINI_MODEL})"
    return f"Groq GPT-OSS ({Config.GROQ_MODEL})"
