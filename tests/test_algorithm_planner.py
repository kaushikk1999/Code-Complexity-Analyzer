import json
import sys
import types
from pathlib import Path

import pytest

from llm import algorithm_planner
from llm.algorithm_planner import (
    PLANNER_OUTPUT_LABELS,
    PlannerTestCase,
    benchmark_planner_solution,
    generate_algorithm_optimization_plan,
)


def test_empty_question_returns_validation_error():
    result = generate_algorithm_optimization_plan("")

    assert not result.valid
    assert "Enter a coding question" in result.error


def test_no_ollama_key_returns_local_estimated_plan_without_code_or_runtime(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    result = generate_algorithm_optimization_plan("Find two numbers in an array that sum to a target.")

    assert result.valid
    assert result.source == "local"
    assert result.final_optimized_python_code == ""
    assert result.time_complexity.startswith("Estimated:")
    assert result.space_complexity.startswith("Estimated:")
    assert not result.runtime.measured
    assert result.runtime.display_value == "Not measured"


def test_ollama_failure_is_sanitized_and_does_not_leak_key(monkeypatch):
    secret = "SECRET_TEST_KEY"

    def fail_with_secret(question: str, api_key: str):
        raise algorithm_planner.OllamaPlannerError("invalid_key")

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", fail_with_secret)

    result = generate_algorithm_optimization_plan("Solve two sum.", secret)

    assert not result.valid
    assert result.source == "ollama"
    assert result.error == "Ollama rejected the API key. Check that it is active in your Ollama account."
    assert secret not in result.error


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (
            "model_unavailable",
            "Ollama Cloud request failed for the selected model. Check that your account can access qwen3-coder-next:cloud.",
        ),
        ("malformed_response", "Ollama responded, but not in the expected planner format. Try again."),
    ],
)
def test_ollama_failure_categories_are_sanitized(monkeypatch, category, expected):
    secret = "SECRET_TEST_KEY"

    def fail(question: str, api_key: str):
        raise algorithm_planner.OllamaPlannerError(category)

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", fail)

    result = generate_algorithm_optimization_plan("Solve two sum.", secret)

    assert not result.valid
    assert result.error == expected
    assert secret not in result.error


def test_quota_failure_returns_local_fallback_plan_without_leaking_key(monkeypatch):
    secret = "SECRET_TEST_KEY"

    def fail(question: str, api_key: str):
        raise algorithm_planner.OllamaPlannerError("quota")

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", fail)

    result = generate_algorithm_optimization_plan("Find two numbers in an array that sum to a target.", secret)

    assert result.valid
    assert result.source == "local"
    assert result.error == "Ollama quota was exhausted, so Complexity Lab generated a local fallback plan."
    assert result.best_data_structure_algorithm_choice == "Hash map lookup"
    assert secret not in result.error


def test_unsafe_generated_code_is_blocked_before_benchmark(monkeypatch):
    def unsafe_payload(question: str, api_key: str):
        return {
            "problem_understanding": "Unsafe example.",
            "step_by_step_optimization_plan": ["Do not run unsafe code."],
            "best_data_structure_algorithm_choice": "N/A",
            "final_optimized_python_code": "def solve():\n    open('x.txt', 'w')\n    return 1\n",
            "entrypoint": "solve",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "test_cases": [{"name": "basic", "input": {"args": []}, "expected_output": 1}],
        }

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", unsafe_payload)

    result = generate_algorithm_optimization_plan("Return one.", "fake-key")

    assert not result.valid
    assert "Unsafe generated code" in result.error
    assert result.final_optimized_python_code == ""
    assert not result.runtime.measured


def test_safe_generated_code_reports_peak_runtime(monkeypatch):
    def safe_payload(question: str, api_key: str):
        return {
            "problem_understanding": "Increment a value.",
            "step_by_step_optimization_plan": ["Return x + 1."],
            "best_data_structure_algorithm_choice": "Simple arithmetic",
            "final_optimized_python_code": "def solve(value):\n    return value + 1\n",
            "entrypoint": "solve",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "test_cases": [
                {"name": "small", "input": {"args": [1]}, "expected_output": 2},
                {"name": "larger", "input": {"args": [100]}, "expected_output": 101},
            ],
        }

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", safe_payload)

    result = generate_algorithm_optimization_plan("Increment a number.", "fake-key")

    assert result.valid
    assert result.source == "ollama"
    assert result.runtime.measured
    assert result.runtime.peak_runtime_ms is not None
    assert result.runtime.display_value.endswith(" ms")
    assert not result.time_complexity.startswith("Estimated")
    assert not result.space_complexity.startswith("Estimated")


def test_benchmark_planner_solution_uses_peak_runtime_label_only():
    runtime = benchmark_planner_solution(
        "def solve(value):\n    return value * 2\n",
        "solve",
        [PlannerTestCase(name="basic", input_text='{"args": [4]}', expected_output="8")],
    )

    assert runtime.measured
    assert runtime.display_value.endswith(" ms")
    assert "Peak Runtime" in PLANNER_OUTPUT_LABELS
    assert "Average Runtime" not in PLANNER_OUTPUT_LABELS


def test_algorithm_planner_reuses_sidebar_ollama_key_input():
    app_source = Path("app.py").read_text(encoding="utf-8")
    planner_body = app_source.split("def _render_algorithm_planner_tab", 1)[1].split(
        "def _render_code_analyzer_workflow", 1
    )[0]

    assert '"Ollama API Key"' not in planner_body
    assert "st.text_input(" not in planner_body
    assert "_resolve_ollama_key_for_action" in planner_body
    assert "_queue_ollama_submit" in planner_body
    assert "key=_ollama_key_widget_key()" in app_source
    assert 'st.session_state.ollama_api_key = ""' in app_source
    assert "del st.session_state[widget_key]" in app_source


def test_algorithm_planner_uses_fixed_qwen_model(monkeypatch):
    fake_ollama = types.ModuleType("ollama")

    class FakeClient:
        def __init__(self, host: str, headers: dict) -> None:
            self.host = host
            self.headers = headers

    fake_ollama.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    monkeypatch.setenv("OLLAMA_MODEL", "unavailable-model")

    calls = []
    payload = {
        "problem_understanding": "Fallback worked.",
        "step_by_step_optimization_plan": ["Use a direct answer."],
        "best_data_structure_algorithm_choice": "Simple arithmetic",
        "final_optimized_python_code": "def solve(value):\n    return value\n",
        "entrypoint": "solve",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "test_cases": [],
    }

    def fake_generate_content(client, model_name: str, prompt: str, json_mode: bool):
        calls.append(model_name)
        assert json_mode
        assert client.host == "https://ollama.com"
        return json.dumps(payload)

    monkeypatch.setattr(algorithm_planner, "_generate_content", fake_generate_content)

    result = algorithm_planner._generate_with_ollama("Return the input.", "SECRET_TEST_KEY")

    assert result["problem_understanding"] == "Fallback worked."
    assert calls == ["qwen3-coder-next:cloud"]


def test_model_candidates_ignore_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    candidates = algorithm_planner._model_candidates()

    assert candidates == ["qwen3-coder-next:cloud"]
