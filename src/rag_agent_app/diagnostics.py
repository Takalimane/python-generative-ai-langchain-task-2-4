from __future__ import annotations

import hashlib
import logging
import os
import re
from urllib.parse import urlparse
import httpx


class ConfigError(ValueError):
    pass


def sanitize_key(raw: str | None) -> str:
    if raw is None:
        return ""
    value = raw.strip().strip('"').strip("'").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if any(character.isspace() for character in value):
        raise ConfigError("API key contains whitespace")
    return value


def describe_key(key: str | None) -> dict:
    value = sanitize_key(key)
    return {
        "present": bool(value),
        "length": len(value),
        "prefix_ok": value.startswith(("sk-or-", "AIza")),
        "has_whitespace": bool(key and any(character.isspace() for character in key)),
        "has_quotes": bool(key and ("\"" in key or "'" in key)),
        "has_bearer_prefix": bool(key and key.strip().lower().startswith("bearer ")),
        "looks_like_other_provider": bool(
            value
            and not value.startswith(("sk-or-", "AIza"))
            and (
                value.startswith("sk-")
                or value.startswith("AIza")
                or len(value) < 12
            )
        ),
    }


def _host(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value if "://" in value else f"//{value}")
    return parsed.hostname


def describe_env() -> dict:
    env_keys = (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    )
    result = {key.lower(): bool(os.getenv(key)) for key in env_keys}

    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        result[name.lower()] = _host(os.getenv(name))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        result[name.lower()] = _host(os.getenv(name))
    return result


class RedactingFilter(logging.Filter):
    PATTERNS = (
        re.compile(r"sk-or-[A-Za-z0-9_-]+"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self.PATTERNS:
            message = pattern.sub("[REDACTED]", message)
        record.msg = message
        record.args = ()
        return True


def install_redacting_logs() -> None:
    redactor = RedactingFilter()
    for logger_name in ("", "httpx", "openai", "httpcore"):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, RedactingFilter) for item in logger.filters):
            logger.addFilter(redactor)


def preflight_openrouter(key: str, timeout: float = 10.0) -> tuple[int | None, str]:
    value = sanitize_key(key)
    if not value:
        return None, "Missing OpenRouter API key"
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={
                "Authorization": f"Bearer {value}",
                "HTTP-Referer": "https://github.com/Takalimane/python-generative-ai-langchain-task-2-4",
                "X-Title": "Knowledge Studio",
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return None, f"Network error: {type(exc).__name__}"

    if response.status_code == 200:
        return 200, "Verified"
    if response.status_code == 401:
        return 401, "OpenRouter rejected the key"
    if response.status_code == 402:
        return 402, "Account or model needs credits"
    if response.status_code == 404:
        return 404, "Model endpoint unavailable"
    if response.status_code == 429:
        return 429, "OpenRouter rate limit reached"
    return response.status_code, "OpenRouter connection failed"


def friendly_error(exc: Exception) -> tuple[str, str, bool]:
    status = getattr(exc, "status_code", None)
    name = exc.__class__.__name__
    if isinstance(exc, ConfigError) or name == "AuthenticationError" or status == 401:
        return "Authentication rejected", "Check the provider key and try again.", True
    if name == "RateLimitError" or status == 429:
        return "Rate limit reached", "Wait briefly, then retry.", True
    if name == "NotFoundError" or status == 404:
        return "Model unavailable", "The selected model or endpoint was not found.", True
    if name in {"APIConnectionError", "APITimeoutError"}:
        return "Connection failed", "Check the network or proxy and retry.", True
    if name == "ResourceExhausted":
        return "Provider quota reached", "The free quota is temporarily exhausted.", True
    if name == "InvalidArgument":
        return "Request rejected", "Check the provider and model settings.", True
    return "Request could not be completed", "Review the settings and retry.", True
