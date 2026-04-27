"""Natural-language algorithm optimization planner."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from benchmarking import run_benchmark
from benchmarking.sandbox import validate_code_for_execution

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL_FALLBACKS = (
    DEFAULT_GEMINI_MODEL,
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

PLANNER_OUTPUT_LABELS = (
    "Problem Understanding",
    "Step-by-Step Optimization Plan",
    "Best Data Structure / Algorithm Choice",
    "Final Optimized Python Code",
    "Time Complexity",
    "Space Complexity",
    "Peak Runtime",
    "Test Cases",
)

COMPLEXITY_NOTE = (
    "Big-O complexity is theoretical algorithm analysis, not an exact measured value. "
    "Runtime depends on hardware, input size, Python version, and the current environment."
)


class GeminiPlannerError(Exception):
    """Sanitized Gemini failure category for UI-safe error handling."""

    def __init__(self, category: str = "request_failed") -> None:
        super().__init__(category)
        self.category = category


@dataclass
class PlannerTestCase:
    name: str
    input_text: str
    expected_output: str = ""


@dataclass
class PlannerRuntimeResult:
    measured: bool = False
    peak_runtime_ms: Optional[float] = None
    details: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def display_value(self) -> str:
        if not self.measured or self.peak_runtime_ms is None:
            return "Not measured"
        return f"{self.peak_runtime_ms:.4f} ms"


@dataclass
class AlgorithmPlannerResult:
    valid: bool
    source: str
    problem_understanding: str = ""
    step_by_step_optimization_plan: List[str] = field(default_factory=list)
    best_data_structure_algorithm_choice: str = ""
    final_optimized_python_code: str = ""
    entrypoint: str = ""
    time_complexity: str = ""
    space_complexity: str = ""
    complexity_note: str = COMPLEXITY_NOTE
    test_cases: List[PlannerTestCase] = field(default_factory=list)
    runtime: PlannerRuntimeResult = field(default_factory=PlannerRuntimeResult)
    error: str = ""
    safety_error: str = ""


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(clean[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Gemini did not return a JSON object.")
    return payload


def _strip_code_fence(code: str) -> str:
    clean = (code or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:python)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _input_to_benchmark_text(raw_input: Any) -> str:
    if isinstance(raw_input, str):
        text = raw_input.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps({"args": [text], "kwargs": {}})
        return _input_to_benchmark_text(parsed)
    if isinstance(raw_input, dict) and ("args" in raw_input or "kwargs" in raw_input):
        args = raw_input.get("args", [])
        kwargs = raw_input.get("kwargs", {})
        if not isinstance(args, list):
            args = [args]
        if not isinstance(kwargs, dict):
            kwargs = {}
        return json.dumps({"args": args, "kwargs": kwargs})
    if isinstance(raw_input, list):
        return json.dumps({"args": raw_input, "kwargs": {}})
    return json.dumps({"args": [raw_input], "kwargs": {}})


def _coerce_test_cases(value: Any) -> List[PlannerTestCase]:
    if not isinstance(value, list):
        return []
    cases: List[PlannerTestCase] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            raw_input = item.get("input", item.get("args", []))
            expected = item.get("expected_output", item.get("expected", ""))
            name = str(item.get("name", f"Test case {index}")).strip() or f"Test case {index}"
        else:
            raw_input = item
            expected = ""
            name = f"Test case {index}"
        cases.append(
            PlannerTestCase(
                name=name,
                input_text=_input_to_benchmark_text(raw_input),
                expected_output=_stringify(expected),
            )
        )
    return cases[:8]


def benchmark_planner_solution(
    code: str,
    entrypoint: str,
    test_cases: List[PlannerTestCase],
) -> PlannerRuntimeResult:
    """Benchmark a generated planner solution and report only peak runtime."""
    code = (code or "").strip()
    entrypoint = (entrypoint or "").strip()
    if not code or not entrypoint or not test_cases:
        return PlannerRuntimeResult(error="Peak Runtime: Not measured")

    violations = validate_code_for_execution(code)
    if violations:
        return PlannerRuntimeResult(error="Peak Runtime: Not measured")

    peaks: List[float] = []
    details: List[str] = []
    for test_case in test_cases:
        result = run_benchmark(
            code=code,
            entrypoint=entrypoint,
            input_text=test_case.input_text,
            repeat_count=3,
            warmup_count=1,
            timeout_seconds=5.0,
            allow_top_level=False,
        )
        if result.success:
            peaks.append(result.summary.max_ms)
            details.append(f"{test_case.name}: peak {result.summary.max_ms:.4f} ms")
        else:
            details.append(f"{test_case.name}: not measured")

    if not peaks:
        return PlannerRuntimeResult(details=details, error="Peak Runtime: Not measured")
    return PlannerRuntimeResult(
        measured=True,
        peak_runtime_ms=round(max(peaks), 4),
        details=details,
    )


def _local_choice(question: str) -> tuple[str, str, str]:
    lowered = question.lower()
    if "two sum" in lowered or ("target" in lowered and ("pair" in lowered or "sum" in lowered)):
        return (
            "Hash map lookup",
            "Estimated: O(n)",
            "Estimated: O(n)",
        )
    if "binary search" in lowered or ("sorted" in lowered and "search" in lowered):
        return (
            "Binary search on sorted input",
            "Estimated: O(log n)",
            "Estimated: O(1)",
        )
    if "top k" in lowered or "k largest" in lowered or "k smallest" in lowered:
        return (
            "Heap of size k",
            "Estimated: O(n log k)",
            "Estimated: O(k)",
        )
    if "graph" in lowered or "bfs" in lowered or "dfs" in lowered:
        return (
            "Graph traversal with queue/stack and visited set",
            "Estimated: O(V + E)",
            "Estimated: O(V)",
        )
    if "palindrome" in lowered:
        return (
            "Two pointers",
            "Estimated: O(n)",
            "Estimated: O(1)",
        )
    return (
        "Problem-dependent; provide a Gemini API key for a tailored optimized solution.",
        "Estimated: problem-dependent",
        "Estimated: problem-dependent",
    )


def _local_plan(question: str) -> AlgorithmPlannerResult:
    algorithm, time_complexity, space_complexity = _local_choice(question)
    return AlgorithmPlannerResult(
        valid=True,
        source="local",
        problem_understanding=(
            "Local mode received a natural-language coding problem. Without Gemini, the app avoids "
            "inventing a full solution and provides only a conservative optimization outline."
        ),
        step_by_step_optimization_plan=[
            "Clarify input size, output requirements, and edge cases.",
            "Identify whether the bottleneck is repeated search, sorting, recursion, or graph traversal.",
            "Choose the simplest data structure that improves the dominant operation.",
            "Validate the approach with small examples before analyzing Big-O complexity.",
        ],
        best_data_structure_algorithm_choice=algorithm,
        final_optimized_python_code="",
        time_complexity=time_complexity,
        space_complexity=space_complexity,
        runtime=PlannerRuntimeResult(error="Peak Runtime: Not measured"),
    )


def _planner_prompt(question: str) -> str:
    return f"""
You are an expert competitive programmer. Given the following coding problem, produce the most time- and space-efficient Python solution. First explain the problem, then provide a step-by-step optimization plan, then identify the best data structure or algorithm, then produce the simplest correct optimized Python code. Include time complexity and space complexity. Prefer clear, maintainable Python. Do not over-engineer if a simple approach is already optimal. Include test cases. Problem: {question}

Return JSON only with this exact shape:
{{
  "problem_understanding": "short explanation",
  "step_by_step_optimization_plan": ["step 1", "step 2"],
  "best_data_structure_algorithm_choice": "best approach",
  "final_optimized_python_code": "complete Python code with one top-level callable function",
  "entrypoint": "the top-level function name to call",
  "time_complexity": "O(...)",
  "space_complexity": "O(...)",
  "test_cases": [
    {{"name": "basic", "input": {{"args": [], "kwargs": {{}}}}, "expected_output": "expected value"}}
  ]
}}

Rules:
- Optimize for correctness first, then lowest practical time complexity, then lowest practical space complexity.
- Use the best data structure or algorithm only when it is actually needed.
- Avoid unnecessary imports and avoid file, process, network, introspection, and dynamic execution APIs.
- Test case inputs must be JSON-compatible arguments for the entrypoint function.
- Do not claim Big-O complexity is exact measured runtime.
"""


def _model_candidates() -> List[str]:
    candidates = [os.getenv("GEMINI_MODEL", "").strip(), *GEMINI_MODEL_FALLBACKS]
    unique_candidates: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _classify_gemini_error(exc: Exception) -> str:
    error_text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in error_text for marker in ("quota", "rate limit", "rate_limit", "429", "resource_exhausted")):
        return "quota"
    if ("model" in error_text or "models/" in error_text) and any(
        marker in error_text
        for marker in (
            "not found",
            "unavailable",
            "unsupported",
            "not supported",
            "permission denied",
            "not enabled",
            "404",
        )
    ):
        return "model_unavailable"
    if any(
        marker in error_text
        for marker in (
            "api key",
            "apikey",
            "bad key",
            "invalid key",
            "unauthenticated",
            "authentication",
            "permission denied",
            "401",
            "403",
        )
    ):
        return "invalid_key"
    return "request_failed"


def _gemini_error_message(category: str) -> str:
    messages = {
        "invalid_key": "Gemini rejected the API key. Check that it is active in Google AI Studio.",
        "quota": "Gemini free-tier quota or rate limit was reached. Try again later.",
        "model_unavailable": "The selected Gemini model is unavailable for this key. The app tried fallback models.",
        "malformed_response": "Gemini responded, but not in the expected planner format. Try again.",
        "missing_package": "The Google Gen AI SDK is not installed. Install google-genai and try again.",
    }
    return messages.get(category, "Gemini request failed. Check the API key and try again.")


def _generate_content(client: Any, model_name: str, prompt: str, config: Any) -> Any:
    return client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )


def _generate_with_gemini(question: str, api_key: str) -> Dict[str, Any]:
    # The API key is used only to construct this request-scoped client.
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiPlannerError("missing_package") from exc

    client = genai.Client(api_key=api_key)
    prompt = _planner_prompt(question)
    config = types.GenerateContentConfig(response_mime_type="application/json")
    last_model_error: Exception | None = None

    for model_name in _model_candidates():
        try:
            response = _generate_content(client, model_name, prompt, config)
        except Exception as exc:
            category = _classify_gemini_error(exc)
            if category == "model_unavailable":
                last_model_error = exc
                continue
            raise GeminiPlannerError(category) from exc

        try:
            return _extract_json_object(getattr(response, "text", "") or "")
        except (json.JSONDecodeError, ValueError) as exc:
            raise GeminiPlannerError("malformed_response") from exc

    raise GeminiPlannerError("model_unavailable") from last_model_error


def _safe_gemini_failure(category: str) -> AlgorithmPlannerResult:
    return AlgorithmPlannerResult(
        valid=False,
        source="gemini",
        error=_gemini_error_message(category),
    )


def _result_from_payload(payload: Dict[str, Any]) -> AlgorithmPlannerResult:
    code = _strip_code_fence(str(payload.get("final_optimized_python_code", "")))
    entrypoint = str(payload.get("entrypoint", "")).strip()
    test_cases = _coerce_test_cases(payload.get("test_cases", []))

    result = AlgorithmPlannerResult(
        valid=True,
        source="gemini",
        problem_understanding=str(payload.get("problem_understanding", "")).strip(),
        step_by_step_optimization_plan=_coerce_string_list(payload.get("step_by_step_optimization_plan", [])),
        best_data_structure_algorithm_choice=str(payload.get("best_data_structure_algorithm_choice", "")).strip(),
        final_optimized_python_code=code,
        entrypoint=entrypoint,
        time_complexity=str(payload.get("time_complexity", "")).strip() or "Not provided",
        space_complexity=str(payload.get("space_complexity", "")).strip() or "Not provided",
        test_cases=test_cases,
    )

    if not code:
        result.valid = False
        result.error = "Gemini did not return optimized Python code."
        return result

    violations = validate_code_for_execution(code)
    if violations:
        result.valid = False
        result.final_optimized_python_code = ""
        result.error = "Unsafe generated code was blocked before execution."
        result.safety_error = "Generated code did not pass execution safety checks."
        result.runtime = PlannerRuntimeResult(error="Peak Runtime: Not measured")
        return result

    result.runtime = benchmark_planner_solution(code, entrypoint, test_cases)
    return result


def generate_algorithm_optimization_plan(question: str, api_key: str = "") -> AlgorithmPlannerResult:
    question = (question or "").strip()
    api_key = (api_key or "").strip()
    if not question:
        return AlgorithmPlannerResult(
            valid=False,
            source="validation",
            error="Enter a coding question before generating an optimization plan.",
        )

    if not api_key:
        return _local_plan(question)

    try:
        payload = _generate_with_gemini(question, api_key)
    except GeminiPlannerError as exc:
        if exc.category == "quota":
            result = _local_plan(question)
            result.error = "Gemini free-tier quota was exhausted, so Complexity Lab generated a local fallback plan."
            return result
        return _safe_gemini_failure(exc.category)
    except Exception as exc:
        return _safe_gemini_failure(_classify_gemini_error(exc))
    return _result_from_payload(payload)
