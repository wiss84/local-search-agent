"""
Multi-provider LLM factory for the Local Search Agent framework.

Supported providers
-------------------
- "google"    : Google AI Studio via langchain-google-genai (Gemma 4 free tier)
- "ollama"    : Local Ollama via langchain-community
- "openai"    : OpenAI API via langchain-openai
- "anthropic" : Anthropic API via langchain-anthropic

All providers return a LangChain BaseChatModel instance with .bind_tools() support,
so the agent loop works identically regardless of provider.

Usage
-----
    from local_search_agent.agent.provider_factory import build_llm
    from local_search_agent.core.config import SearchAgentConfig

    llm = build_llm(config)
    llm_with_tools = llm.bind_tools([search_tool, fetch_tool])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from local_search_agent.core.config import SearchAgentConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_llm(config: SearchAgentConfig) -> "BaseChatModel":
    """
    Construct and return a LangChain ChatModel for the configured provider.

    Parameters
    ----------
    config : SearchAgentConfig with provider, api_key, and model_name set.

    Returns
    -------
    A LangChain BaseChatModel instance ready for .bind_tools().

    Raises
    ------
    ValueError       : If provider is unsupported.
    ImportError      : If the required langchain package is not installed.
    """
    provider = config.provider.lower()
    logger.info("Building LLM: provider=%r, model=%r", provider, config.model_name)

    if provider == "google":
        return _build_google(config)
    elif provider == "ollama":
        return _build_ollama(config)
    elif provider == "openai":
        return _build_openai(config)
    elif provider == "anthropic":
        return _build_anthropic(config)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Choose from: 'google', 'ollama', 'openai', 'anthropic'."
        )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _build_google(config: SearchAgentConfig) -> "BaseChatModel":
    """
    Google AI Studio via langchain-google-genai.

    Supports Gemma 4 free-tier models:
    - "gemma-4-31b-it"       (default — dense, strong reasoning)
    - "gemma-4-26b-a4b-it"   (MoE variant — faster, same free tier)

    Install: pip install "langchain-google-genai>=4.2.2"
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-genai is not installed. "
            "Run: pip install 'langchain-google-genai>=4.2.2'"
        ) from e

    if not config.api_key:
        raise ValueError(
            "api_key is required for provider='google'. "
            "Set GOOGLE_API_KEY env var or pass api_key to SearchAgentConfig."
        )

    return ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.api_key,
        temperature=0,  # Deterministic responses for RAG
        max_retries=5,
    )


def _build_ollama(config: SearchAgentConfig) -> "BaseChatModel":
    """
    Local Ollama via langchain-ollama.

    No API key needed. Ollama must be running at localhost:11434.
    Download Ollama from https://ollama.com and pull a model first:
        ollama pull llama3.2
        ollama pull mistral
        ollama pull qwen2.5
        ollama pull gemma3

    Install: pip install "langchain-ollama>=1.1.0"
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install 'langchain-ollama>=1.1.0'"
        ) from e

    model = config.model_name or "gemma4:e2b"
    return ChatOllama(
        model=model,
        temperature=0,
    )


def _build_openai(config: SearchAgentConfig) -> "BaseChatModel":
    """
    OpenAI API via langchain-openai.

    Install: pip install "langchain-openai>=1.1.10"
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install 'langchain-openai>=1.1.10'"
        ) from e

    if not config.api_key:
        raise ValueError(
            "api_key is required for provider='openai'. "
            "Set OPENAI_API_KEY env var or pass api_key to SearchAgentConfig."
        )

    return ChatOpenAI(
        model=config.model_name or "gpt-4o-mini",
        api_key=config.api_key,
        temperature=0,
        max_retries=5,
    )


def _build_anthropic(config: SearchAgentConfig) -> "BaseChatModel":
    """
    Anthropic API via langchain-anthropic.

    Install: pip install "langchain-anthropic>=1.4.2"
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "langchain-anthropic is not installed. Run: pip install 'langchain-anthropic>=1.4.2'"
        ) from e

    if not config.api_key:
        raise ValueError(
            "api_key is required for provider='anthropic'. "
            "Set ANTHROPIC_API_KEY env var or pass api_key to SearchAgentConfig."
        )

    return ChatAnthropic(
        model=config.model_name or "claude-sonnet-4-20250514",
        api_key=config.api_key,
        temperature=0,
        max_retries=5,
    )
