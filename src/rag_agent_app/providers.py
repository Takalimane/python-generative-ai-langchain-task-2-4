from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .config import AppConfig
from .diagnostics import ConfigError, sanitize_key


def build_chat_model(config: AppConfig) -> BaseChatModel:
    if config.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        key = sanitize_key(config.openrouter_key)
        if not key:
            raise ConfigError("OpenRouter API key is required")
        if not key.startswith("sk-or-"):
            raise ConfigError("OpenRouter key format is invalid")
        return ChatOpenAI(
            model=config.chat_model,
            temperature=0,
            streaming=True,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/Takalimane/python-generative-ai-langchain-task-2-4",
                "X-Title": "Knowledge Studio",
            },
        )

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        key = sanitize_key(config.openai_key)
        if not key:
            raise ConfigError("OpenAI API key is required")
        return ChatOpenAI(model=config.chat_model, temperature=0, streaming=True, api_key=key)

    from langchain_google_genai import ChatGoogleGenerativeAI

    key = sanitize_key(config.gemini_key)
    if not key:
        raise ConfigError("Gemini API key is required")
    return ChatGoogleGenerativeAI(
        model=config.chat_model,
        temperature=0,
        streaming=True,
        google_api_key=key,
    )


def build_embeddings(config: AppConfig) -> Embeddings:
    if config.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = sanitize_key(config.openai_key)
        if not key:
            raise ConfigError("OpenAI API key is required")
        return OpenAIEmbeddings(api_key=key)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    key = sanitize_key(config.gemini_key)
    if not key:
        raise ConfigError("Gemini API key is required")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )
