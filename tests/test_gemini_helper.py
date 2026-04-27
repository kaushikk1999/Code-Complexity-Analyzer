import json
import sys
import types

import pytest

from analyzer.models import StaticAnalysisResult
from llm import gemini_helper
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


def test_enhance_invalid_key_is_sanitized_and_does_not_leak_key(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def fail(api_key: str, prompt: str, *, json_mode: bool = False):
        raise gemini_helper.GeminiHelperError("invalid_key")

    monkeypatch.setattr(gemini_helper, "_request_gemini_text", fail)

    message = gemini_helper.enhance_with_gemini(secret, analysis.raw_code, analysis, score, plan)

    assert message == "Gemini enhancement failed. Gemini rejected the API key. Check that it is active in Google AI Studio."
    assert secret not in message


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("quota", "Gemini free-tier quota or rate limit was reached. Try again later."),
        (
            "model_unavailable",
            "The selected Gemini model is unavailable for this key. The app tried fallback models.",
        ),
    ],
)
def test_optimized_code_failure_categories_are_sanitized(monkeypatch, category, expected):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def fail(api_key: str, prompt: str, *, json_mode: bool = False):
        raise gemini_helper.GeminiHelperError(category)

    monkeypatch.setattr(gemini_helper, "_request_gemini_text", fail)

    candidate, error = gemini_helper.generate_optimized_code_with_gemini(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert candidate is None
    assert error == f"Gemini optimization generation failed. {expected}"
    assert secret not in error


def test_malformed_optimized_code_json_returns_safe_error(monkeypatch):
    secret = "SECRET_TEST_KEY"
    analysis, score, plan = _fixtures()

    def malformed(api_key: str, prompt: str, *, json_mode: bool = False):
        return "not json"

    monkeypatch.setattr(gemini_helper, "_request_gemini_text", malformed)

    candidate, error = gemini_helper.generate_optimized_code_with_gemini(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert candidate is None
    assert error == "Gemini optimization generation failed. Gemini responded, but not in the expected format. Try again."
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

    monkeypatch.setattr(gemini_helper, "_request_gemini_text", success)

    candidate, error = gemini_helper.generate_optimized_code_with_gemini(
        secret,
        analysis.raw_code,
        analysis,
        score,
        plan,
        entrypoint="two_sum",
    )

    assert error is None
    assert candidate is not None
    assert candidate.source == "gemini"
    assert candidate.code.startswith("def two_sum")
    assert candidate.step_by_step_plan == ["Store complements."]
    assert candidate.validation_tests == ["assert two_sum([2, 7], 9) == [0, 1]"]


def test_gemini_helper_model_fallback_tries_next_model(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_types = types.ModuleType("google.genai.types")

    class FakeConfig:
        def __init__(self, response_mime_type: str) -> None:
            self.response_mime_type = response_mime_type

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = object()

    fake_types.GenerateContentConfig = FakeConfig
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("GEMINI_MODEL", "unavailable-model")

    calls = []

    def fake_generate_content(client, model_name: str, prompt: str, config=None):
        calls.append(model_name)
        assert config is not None
        assert config.response_mime_type == "application/json"
        if model_name == "unavailable-model":
            raise RuntimeError("model not found")
        return types.SimpleNamespace(text='{"ok": true}')

    monkeypatch.setattr(gemini_helper, "_generate_content", fake_generate_content)

    text = gemini_helper._request_gemini_text("SECRET_TEST_KEY", "prompt", json_mode=True)

    assert text == '{"ok": true}'
    assert calls[:2] == ["unavailable-model", "gemini-3-flash-preview"]


def test_gemini_helper_model_candidates_start_with_env_override_then_gemini_3_flash(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")

    candidates = gemini_helper._model_candidates()

    assert candidates[:3] == [
        "custom-model",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ]
