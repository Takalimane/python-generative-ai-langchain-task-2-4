from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    provider: str
    chat_model: str
    embedding_provider: str
    tavily_key: str | None

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
        if provider not in {"openai", "gemini", "openrouter"}:
            raise ValueError("AI_PROVIDER must be 'gemini', 'openrouter', or 'openai'")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", provider).strip().lower()
        if embedding_provider not in {"openai", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'gemini'")
        if provider == "openai":
            chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "openrouter":
            chat_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        else:
            chat_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return cls(
            provider=provider,
            chat_model=chat_model,
            embedding_provider=embedding_provider,
            tavily_key=os.getenv("TAVILY_API_KEY") or None,
        )
