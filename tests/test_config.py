import pytest

from rag_agent_app.config import AppConfig


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
