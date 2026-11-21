import pytest

from rag_agent_app.config import AppConfig
from rag_agent_app.providers import build_chat_model


def test_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="AI_PROVIDER"):
        AppConfig.from_env()


def test_openrouter_free_router_config(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/free")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

    config = AppConfig.from_env()

    assert config.provider == "openrouter"
    assert config.chat_model == "openrouter/free"


def test_openrouter_chat_model_receives_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    config = AppConfig(
        provider="openrouter",
        chat_model="openrouter/free",
        embedding_provider="gemini",
        tavily_key=None,
    )

    model = build_chat_model(config)

    assert model.openai_api_key.get_secret_value() == "test-openrouter-key"
