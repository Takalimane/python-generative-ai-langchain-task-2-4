import pytest

from rag_agent_app.config import AppConfig


def test_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="AI_PROVIDER"):
        AppConfig.from_env()
