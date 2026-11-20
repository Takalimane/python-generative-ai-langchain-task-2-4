from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .config import AppConfig


def build_chat_model(config: AppConfig) -> BaseChatModel:
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
