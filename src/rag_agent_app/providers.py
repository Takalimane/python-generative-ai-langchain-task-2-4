from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .config import AppConfig


def build_chat_model(config: AppConfig) -> BaseChatModel:
    if config.provider == "openrouter":
        import os
        from langchain_openai import ChatOpenAI

        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required")
        return ChatOpenAI(
            model=config.chat_model,
            temperature=0,
            streaming=True,
            openai_api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/Takalimane/python-generative-ai-langchain-task-2-4",
                "X-Title": "Knowledge Studio",
            },
        )

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.chat_model, temperature=0, streaming=True)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.chat_model,
        temperature=0,
        streaming=True,
    )


def build_embeddings(config: AppConfig) -> Embeddings:
    if config.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
