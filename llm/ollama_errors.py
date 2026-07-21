"""Shared Ollama Cloud model selection and safe error handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_OLLAMA_MODEL_LABEL = "GPT-OSS (120B)"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"
OLLAMA_MODEL_OPTIONS = {
    "GPT-OSS (120B)": "gpt-oss:120b",
    "Nemotron 3 Ultra": "nemotron-3-ultra",
    "MiniMax M3": "minimax-m3",
    "Gemma 4 (31B)": "gemma4:31b",
}
OLLAMA_MODEL_HELP = {
    "GPT-OSS (120B)": "Default. Largest gpt-oss tier, with fast and clean JSON output.",
    "Nemotron 3 Ultra": "Largest Nemotron 3 tier, built for high-throughput reasoning.",
    "MiniMax M3": "Coding and agentic frontier model. Thinking mode makes it slow.",
    "Gemma 4 (31B)": "Largest Gemma 4 cloud tier. Fast, but wraps JSON in code fences.",
}
OLLAMA_MODEL_FALLBACKS = tuple(OLLAMA_MODEL_OPTIONS.values())
OLLAMA_HOST = "https://ollama.com"


@dataclass(frozen=True)
class OllamaErrorDiagnostic:
    status_code: int | None
    text: str


def normalize_model_name(model_name: str = "") -> str:
    clean = (model_name or "").strip()
    if not clean:
        return ""
    return OLLAMA_MODEL_OPTIONS.get(clean, clean)


def model_candidates(preferred_model: str = "") -> list[str]:
    candidates: list[str] = []
    preferred = normalize_model_name(preferred_model)
    if preferred:
        candidates.append(preferred)
    candidates.extend(OLLAMA_MODEL_FALLBACKS)
    return list(dict.fromkeys(candidates))


def ollama_error_diagnostic(exc: Exception) -> OllamaErrorDiagnostic:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "") or str(exc)
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode("utf-8", errors="replace")
    text = sanitize_ollama_error_text(f"{type(exc).__name__} {raw_text}")
    if status_code is None:
        match = re.search(r"status code:\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            status_code = match.group(1)
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
            "requires a subscription",
            "upgrade for access",
        )
    ):
        return "model_unavailable"
    return "request_failed"


def ollama_error_message(category: str) -> str:
    messages = {
        "invalid_key": "Ollama rejected the API key. Check that it is active in your Ollama account.",
        "quota": "Ollama quota or rate limit was reached. Try again later.",
        "model_unavailable": (
            "Ollama Cloud could not access any approved model for this API key. "
            "Check that the key is active and has access to gpt-oss:120b, nemotron-3-ultra, minimax-m3, or gemma4:31b."
        ),
        "malformed_response": "Ollama responded, but not in the expected format. Try again.",
        "missing_package": "The Ollama Python SDK is not installed. Install ollama and try again.",
    }
    return messages.get(category, "Ollama request failed. Check the API key and try again.")
