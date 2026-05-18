import json
import sys
import types

import pytest

from analyzer.models import StaticAnalysisResult
from llm import ollama_errors
from llm import ollama_helper
from optimization.planner import OptimizationPlan
from scoring.optimizer_score import ScoreBreakdown


def _fixtures():
    analysis = StaticAnalysisResult(
        valid=True,
        raw_code="def two_sum(nums, target):\n    return []\n",
        estimated_time="O(n^2)",
        estimated_space="O(1)",
        confidence=0.8,
        metrics={"max_loop_depth": 2},
    )
    score = ScoreBreakdown(
        score=55,
        efficiency_percentage=55,
        severity="moderate",
        improvement_potential="Meaningful",
        bottlenecks=["Nested loops"],
    )
    plan = OptimizationPlan(summary="Use a hash map.")
    return analysis, score, plan


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, text: str) -> None:
        super().__init__(text)
        self.response = _FakeResponse(status_code, text)


def test_enhance_invalid_key_is_sanitized_and_does_not_leak_key(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def fail(api_key: str, prompt: str, *, json_mode: bool = False):
        raise ollama_helper.OllamaHelperError("invalid_key")

    monkeypatch.setattr(ollama_helper, "_request_ollama_text", fail)

    message = ollama_helper.enhance_with_ollama(secret, analysis.raw_code, analysis, score, plan)

    assert message == "Ollama enhancement failed. Ollama rejected the API key. Check that it is active in your Ollama account."
    assert secret not in message


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("quota", "Ollama quota or rate limit was reached. Try again later."),
        (
            "model_unavailable",
            "Ollama Cloud request failed for the selected model. Check that your account can access deepseek-v4-flash:cloud.",
        ),
    ],
)
def test_optimized_code_failure_categories_are_sanitized(monkeypatch, category, expected):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def fail(api_key: str, prompt: str, *, json_mode: bool = False):
        raise ollama_helper.OllamaHelperError(category)

    monkeypatch.setattr(ollama_helper, "_request_ollama_text", fail)

    candidate, error = ollama_helper.generate_optimized_code_with_ollama(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert candidate is None
    assert error == f"Ollama optimization generation failed. {expected}"
    assert secret not in error


def test_malformed_optimized_code_json_returns_safe_error(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def malformed(api_key: str, prompt: str, *, json_mode: bool = False):
        return "not json"

    monkeypatch.setattr(ollama_helper, "_request_ollama_text", malformed)

    candidate, error = ollama_helper.generate_optimized_code_with_ollama(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert candidate is None
    assert error == "Ollama optimization generation failed. Ollama responded, but not in the expected format. Try again."
    assert secret not in error


def test_successful_structured_optimized_code_generation(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()
    payload = {
        "step_by_step_plan": ["Store complements."],
        "optimized_code": "def two_sum(nums, target):\n    seen = {}\n    return []\n",
        "explanation": "Uses a hash map.",
        "validation_tests": ["assert two_sum([2, 7], 9) == [0, 1]"],
        "expected_time": "O(n)",
        "expected_space": "O(n)",
    }

    def success(api_key: str, prompt: str, *, json_mode: bool = False):
        assert api_key == secret
        assert json_mode
        return json.dumps(payload)

    monkeypatch.setattr(ollama_helper, "_request_ollama_text", success)

    candidate, error = ollama_helper.generate_optimized_code_with_ollama(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert error is None
    assert candidate is not None
    assert candidate.source == "ollama"
    assert candidate.code.startswith("def two_sum")
    assert candidate.step_by_step_plan == ["Store complements."]
    assert candidate.validation_tests == ["assert two_sum([2, 7], 9) == [0, 1]"]


def test_ollama_helper_uses_only_deepseek_flash_model(monkeypatch):
    fake_ollama = types.ModuleType("ollama")

    class FakeClient:
        def __init__(self, host: str, headers: dict) -> None:
            self.host = host
            self.headers = headers

    fake_ollama.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    monkeypatch.setenv("OLLAMA_MODEL", "unavailable-model")

    calls = []

    def fake_generate_content(client, model_name: str, prompt: str, json_mode=False):
        calls.append(model_name)
        assert json_mode
        assert client.host == "https://ollama.com"
        return '{"ok": true}'

    monkeypatch.setattr(ollama_helper, "_generate_content", fake_generate_content)

    text = ollama_helper._request_ollama_text("SECRET_TEST_KEY", "prompt", json_mode=True)

    assert text == '{"ok": true}'
    assert calls == ["deepseek-v4-flash:cloud"]


def test_ollama_helper_does_not_fallback_when_deepseek_flash_is_unavailable(monkeypatch):
    fake_ollama = types.ModuleType("ollama")

    class FakeClient:
        def __init__(self, host: str, headers: dict) -> None:
            self.host = host
            self.headers = headers

    fake_ollama.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    calls = []

    def fake_generate_content(client, model_name: str, prompt: str, json_mode=False):
        calls.append(model_name)
        raise RuntimeError("this model requires a subscription (status code: 403)")

    monkeypatch.setattr(ollama_helper, "_generate_content", fake_generate_content)

    with pytest.raises(ollama_helper.OllamaHelperError) as exc_info:
        ollama_helper._request_ollama_text("SECRET_TEST_KEY", "prompt", json_mode=True)

    assert exc_info.value.category == "model_unavailable"
    assert calls == ["deepseek-v4-flash:cloud"]


def test_ollama_helper_model_candidates_ignore_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    candidates = ollama_helper._model_candidates()

    assert candidates == ["deepseek-v4-flash:cloud"]


def test_generate_optimized_code_uses_env_api_key(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()
    monkeypatch.setenv("OLLAMA_API_KEY", secret)

    def success(api_key: str, prompt: str, *, json_mode: bool = False):
        assert api_key == secret
        assert json_mode
        return json.dumps(
            {
                "step_by_step_plan": [],
                "optimized_code": "def two_sum(nums, target):\n    return []\n",
                "explanation": "",
                "validation_tests": [],
                "expected_time": "O(n)",
                "expected_space": "O(1)",
            }
        )

    monkeypatch.setattr(ollama_helper, "_request_ollama_text", success)

    candidate, error = ollama_helper.generate_optimized_code_with_ollama(
        "",
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert error is None
    assert candidate is not None


@pytest.mark.parametrize(
    ("status_code", "text", "expected"),
    [
        (401, "invalid api key", "invalid_key"),
        (403, "permission denied for model", "model_unavailable"),
        (404, "model not found", "model_unavailable"),
        (429, "too many requests", "quota"),
        (None, "this model requires a subscription (status code: 403)", "model_unavailable"),
    ],
)
def test_ollama_error_status_classification(status_code, text, expected):
    assert ollama_errors.classify_ollama_error(_FakeHTTPError(status_code, text)) == expected


def test_ollama_error_diagnostic_sanitizes_authorization_header():
    secret = "SECRET_TEST_KEY"
    exc = _FakeHTTPError(403, f"Authorization: Bearer {secret}\npermission denied")

    diagnostic = ollama_errors.ollama_error_diagnostic(exc)

    assert secret not in diagnostic.text
    assert "Authorization: [redacted]" in diagnostic.text
