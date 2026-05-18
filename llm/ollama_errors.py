"""Shared Ollama Cloud model selection and safe error handling."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_MODEL = "deepseek-v4-pro:cloud"
OLLAMA_MODEL_FALLBACKS = (DEFAULT_OLLAMA_MODEL,)
OLLAMA_HOST = "https://ollama.com"


@dataclass(frozen=True)
class OllamaErrorDiagnostic:
    status_code: int | None
    text: str


def model_candidates() -> list[str]:
    return [DEFAULT_OLLAMA_MODEL]


def ollama_error_diagnostic(exc: Exception) -> OllamaErrorDiagnostic:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "") or str(exc)
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode("utf-8", errors="replace")
    text = sanitize_ollama_error_text(f"{type(exc).__name__} {raw_text}")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    return OllamaErrorDiagnostic(status_code=status_code, text=text)


def sanitize_ollama_error_text(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", clean, flags=re.IGNORECASE)
    clean = re.sub(r"Authorization:\s*[^,\n\r]+", "Authorization: [redacted]", clean, flags=re.IGNORECASE)
    return clean


def classify_ollama_error(exc: Exception) -> str:
    diagnostic = ollama_error_diagnostic(exc)
    error_text = diagnostic.text.lower()
    status_code = diagnostic.status_code

    if status_code == 429 or any(
        marker in error_text for marker in ("quota", "rate limit", "rate_limit", "too many requests")
    ):
        return "quota"
    if status_code == 401:
        return "invalid_key"
    if status_code in (403, 404):
        return "model_unavailable"
    if any(marker in error_text for marker in ("invalid api key", "invalid key", "bad key", "unauthenticated")):
        return "invalid_key"
    if any(
        marker in error_text
        for marker in (
            "model not found",
            "not found",
            "unavailable",
            "unsupported",
            "not supported",
            "permission denied",
            "not enabled",
            "forbidden",
            "model access",
            "models/",
        )
    ):
        return "model_unavailable"
    return "request_failed"


def ollama_error_message(category: str) -> str:
    messages = {
        "invalid_key": "Ollama rejected the API key. Check that it is active in your Ollama account.",
        "quota": "Ollama quota or rate limit was reached. Try again later.",
        "model_unavailable": (
            "Ollama Cloud request failed for the selected model. "
            "Check that your account can access deepseek-v4-pro:cloud."
        ),
        "malformed_response": "Ollama responded, but not in the expected format. Try again.",
        "missing_package": "The Ollama Python SDK is not installed. Install ollama and try again.",
    }
    return messages.get(category, "Ollama request failed. Check the API key and try again.")
