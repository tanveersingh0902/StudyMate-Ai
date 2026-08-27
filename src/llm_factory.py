"""
llm_factory.py — Unified LLM factory supporting Groq (LLaMA) and Google Gemini.

Usage:
    from src.llm_factory import make_llm, call_llm

    llm = make_llm()                          # Uses Config.LLM_PROVIDER
    response = call_llm(system, user)        # One-shot call, returns string

"""

from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config


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
        # Default: Groq LLaMA
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=temp,
            max_tokens=tokens,
        )


def call_llm(system: str, user: str, temperature: float = None) -> str:
    """
    Make a single LLM call and return the response as a plain string.

    Args:
        system: The system/instruction prompt.
        user:   The user message / question.
        temperature: Override temperature (optional).

    Returns:
        The model's response as a stripped string.
    """
    llm = make_llm(temperature=temperature)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    # FIX: .content can be a list of dicts for some providers — normalise to str
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
        return f"Google Gemini ({Config.GEMINI_MODEL})"   # FIX: was using GROQ_MODEL for Gemini
    return f"Groq LLaMA ({Config.GROQ_MODEL})"
