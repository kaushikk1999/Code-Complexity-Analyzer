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
from utils.test_case_generator import DEFAULT_BENCHMARK_CASE_COUNT


def _planner_cases(count: int = DEFAULT_BENCHMARK_CASE_COUNT):
    return [
        {"name": f"case {index}", "input": {"args": [index]}, "expected_output": index + 1}
        for index in range(count)
    ]


def _planner_payload():
    return {
        "problem_understanding": "Increment a value.",
        "step_by_step_optimization_plan": ["Return x + 1."],
        "best_data_structure_algorithm_choice": "Simple arithmetic",
        "final_optimized_python_code": "def solve(value):\n    return value + 1\n",
        "entrypoint": "solve",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "test_cases": _planner_cases(),
    }


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
            "Ollama Cloud could not access any approved model for this API key. Check that the key is active and has access to deepseek-v4-pro:cloud, glm-5.1:cloud, or deepseek-v4-flash:cloud.",
        ),
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


def test_malformed_ollama_response_returns_linear_search_fallback_without_leaking_key(monkeypatch):
    secret = "SECRET_TEST_KEY"

    def fail(question: str, api_key: str):
        raise algorithm_planner.OllamaPlannerError("malformed_response")

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", fail)

    result = generate_algorithm_optimization_plan("write the easiest linear search code", secret)

    assert result.valid
    assert result.source == "local"
    assert result.error == "Ollama responded in an unexpected format, so Complexity Lab generated a local fallback plan."
    assert result.best_data_structure_algorithm_choice == "Linear scan"
    assert result.entrypoint == "linear_search"
    assert "def linear_search(items, target):" in result.final_optimized_python_code
    assert result.time_complexity == "O(n)"
    assert result.space_complexity == "O(1)"
    assert len(result.test_cases) == DEFAULT_BENCHMARK_CASE_COUNT
    assert result.runtime.measured
    assert secret not in result.error


def test_planner_extracts_json_from_wrapped_ollama_response():
    payload = _planner_payload()
    wrapped = {
        "model": "deepseek-v4-pro:cloud",
        "created_at": "2026-05-29T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": json.dumps(payload),
        },
    }

    assert algorithm_planner._extract_json_object(json.dumps(wrapped)) == payload


def test_planner_extracts_best_json_object_when_response_has_extra_braces():
    payload = _planner_payload()
    text = "I considered this shape first: {\"notes\": [\"not planner\"]}\n```json\n" + json.dumps(payload) + "\n```"

    assert algorithm_planner._extract_json_object(text) == payload


def test_planner_accepts_jsonish_numeric_expressions_in_test_cases():
    payload = _planner_payload()
    payload["test_cases"][0]["input"]["args"] = ["NUMERIC_EXPRESSION"]
    text = json.dumps(payload).replace('"NUMERIC_EXPRESSION"', "10**10")

    parsed = algorithm_planner._extract_json_object(text)

    assert parsed["test_cases"][0]["input"]["args"][0] == 10000000000


def test_planner_accepts_bounded_range_comprehensions_in_test_cases():
    payload = _planner_payload()
    payload["test_cases"][0]["input"]["args"] = ["RANGE_EXPRESSION"]
    text = json.dumps(payload).replace('"RANGE_EXPRESSION"', "[i for i in range(4)]")

    parsed = algorithm_planner._extract_json_object(text)

    assert parsed["test_cases"][0]["input"]["args"][0] == [0, 1, 2, 3]


def test_missing_ollama_sdk_returns_linear_search_fallback_without_leaking_key(monkeypatch):
    secret = "SECRET_TEST_KEY"

    def fail(question: str, api_key: str):
        raise algorithm_planner.OllamaPlannerError("missing_package")

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", fail)

    result = generate_algorithm_optimization_plan("write the easiest linear search code", secret)

    assert result.valid
    assert result.source == "local"
    assert result.error == "The Ollama Python SDK was unavailable, so Complexity Lab generated a local fallback plan."
    assert result.entrypoint == "linear_search"
    assert "def linear_search(items, target):" in result.final_optimized_python_code
    assert len(result.test_cases) == DEFAULT_BENCHMARK_CASE_COUNT
    assert result.runtime.measured
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


def test_local_linear_search_plan_includes_code_cases_and_runtime(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    result = generate_algorithm_optimization_plan("write the easiest linear search code")

    assert result.valid
    assert result.source == "local"
    assert result.entrypoint == "linear_search"
    assert result.final_optimized_python_code == (
        "def linear_search(items, target):\n"
        "    for index, value in enumerate(items):\n"
        "        if value == target:\n"
        "            return index\n"
        "    return -1\n"
    )
    assert result.time_complexity == "O(n)"
    assert result.space_complexity == "O(1)"
    assert len(result.test_cases) == DEFAULT_BENCHMARK_CASE_COUNT
    assert result.runtime.measured


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
            "test_cases": _planner_cases(),
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
            "test_cases": _planner_cases(),
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
        [
            PlannerTestCase(name=f"case {index}", input_text=f'{{"args": [{index}]}}', expected_output=str(index * 2))
            for index in range(DEFAULT_BENCHMARK_CASE_COUNT)
        ],
    )

    assert runtime.measured
    assert runtime.display_value.endswith(" ms")
    assert runtime.memory_display_value.endswith(" KB")
    assert "Peak Runtime" in PLANNER_OUTPUT_LABELS
    assert "Peak Memory" in PLANNER_OUTPUT_LABELS
    assert "Average Runtime" not in PLANNER_OUTPUT_LABELS


def test_planner_rejects_fewer_than_40_cases(monkeypatch):
    def short_payload(question: str, api_key: str):
        return {
            "problem_understanding": "Too few cases.",
            "step_by_step_optimization_plan": ["Return x."],
            "best_data_structure_algorithm_choice": "Simple arithmetic",
            "final_optimized_python_code": "def solve(value):\n    return value\n",
            "entrypoint": "solve",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "test_cases": _planner_cases(2),
        }

    monkeypatch.setattr(algorithm_planner, "_generate_with_ollama", short_payload)

    result = generate_algorithm_optimization_plan("Return the value.", "fake-key")

    assert not result.valid
    assert "40 are required" in result.error
    assert not result.runtime.measured


def test_algorithm_planner_uses_env_key_without_sidebar_ollama_key_input():
    app_source = Path("app.py").read_text(encoding="utf-8")
    planner_body = app_source.split("def _render_algorithm_planner_tab", 1)[1].split(
        "def _render_code_analyzer_workflow", 1
    )[0]

    assert '"Ollama API Key"' not in planner_body
    assert "Ollama API key (optional)" not in app_source
    assert "st.text_input(" not in planner_body
    assert "_resolve_ollama_key_for_action" in planner_body
    assert "_queue_ollama_submit" in planner_body
    assert "_ollama_key_widget_key" not in app_source
    assert "pending_ollama_api_key" not in app_source


def test_algorithm_planner_falls_back_to_glm_when_deepseek_pro_is_unavailable(monkeypatch):
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
        if model_name == "deepseek-v4-pro:cloud":
            raise RuntimeError("this model requires a subscription (status code: 403)")
        return json.dumps(payload)

    monkeypatch.setattr(algorithm_planner, "_generate_content", fake_generate_content)

    result = algorithm_planner._generate_with_ollama("Return the input.", "SECRET_TEST_KEY")

    assert result["problem_understanding"] == "Fallback worked."
    assert calls == ["deepseek-v4-pro:cloud", "glm-5.1:cloud"]


def test_algorithm_planner_repairs_non_json_ollama_response(monkeypatch):
    fake_ollama = types.ModuleType("ollama")

    class FakeClient:
        def __init__(self, host: str, headers: dict) -> None:
            self.host = host
            self.headers = headers

    fake_ollama.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    calls = []
    repaired_payload = {
        "problem_understanding": "Find a target value by scanning each item.",
        "step_by_step_optimization_plan": ["Check each element once.", "Return the first matching index."],
        "best_data_structure_algorithm_choice": "Linear scan",
        "final_optimized_python_code": "def linear_search(items, target):\n    for index, value in enumerate(items):\n        if value == target:\n            return index\n    return -1\n",
        "entrypoint": "linear_search",
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "test_cases": _planner_cases(),
    }

    def fake_generate_content(client, model_name: str, prompt: str, json_mode: bool):
        calls.append(prompt)
        if len(calls) == 1:
            return "Here is the easiest linear search code:\n```python\ndef linear_search(items, target): ..."
        return json.dumps(repaired_payload)

    monkeypatch.setattr(algorithm_planner, "_generate_content", fake_generate_content)

    result = algorithm_planner._generate_with_ollama("write the easiest linear search code", "SECRET_TEST_KEY")

    assert result["best_data_structure_algorithm_choice"] == "Linear scan"
    assert "Previous answer:" in calls[1]
    assert len(calls) == 2


def test_model_candidates_ignore_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    candidates = algorithm_planner._model_candidates()

    assert candidates == ["deepseek-v4-pro:cloud", "glm-5.1:cloud", "deepseek-v4-flash:cloud"]
