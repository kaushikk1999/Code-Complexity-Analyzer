"""Natural-language algorithm optimization planner."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from analyzer import analyze_code
from benchmarking import run_benchmark
from benchmarking.sandbox import validate_code_for_execution
from llm.ollama_errors import (
    OLLAMA_HOST,
    classify_ollama_error,
    model_candidates,
    ollama_error_message,
)
from optimization.planner import generate_verified_optimization_candidates
from scoring import calculate_optimization_score
from utils.test_case_generator import DEFAULT_BENCHMARK_CASE_COUNT

PLANNER_OUTPUT_LABELS = (
    "Problem Understanding",
    "Step-by-Step Optimization Plan",
    "Best Data Structure / Algorithm Choice",
    "Final Optimized Python Code",
    "Time Complexity",
    "Space Complexity",
    "Peak Runtime",
    "Peak Memory",
    "Test Cases",
)

COMPLEXITY_NOTE = (
    "Big-O complexity is theoretical algorithm analysis, not an exact measured value. "
    "Runtime depends on hardware, input size, Python version, and the current environment."
)


class OllamaPlannerError(Exception):
    """Sanitized Ollama failure category for UI-safe error handling."""

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
    peak_memory_kb: Optional[float] = None
    details: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def display_value(self) -> str:
        if not self.measured or self.peak_runtime_ms is None:
            return "Not measured"
        return f"{self.peak_runtime_ms:.4f} ms"

    @property
    def memory_display_value(self) -> str:
        if not self.measured or self.peak_memory_kb is None:
            return "Not measured"
        return f"{self.peak_memory_kb:.2f} KB"


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
    payload = _loads_planner_json(clean)
    if not isinstance(payload, dict):
        raise ValueError("Ollama did not return a JSON object.")
    return _unwrap_planner_payload(payload)


def _planner_schema_score(payload: Dict[str, Any]) -> int:
    expected_keys = {
        "problem_understanding",
        "step_by_step_optimization_plan",
        "best_data_structure_algorithm_choice",
        "final_optimized_python_code",
        "entrypoint",
        "time_complexity",
        "space_complexity",
        "test_cases",
    }
    return sum(1 for key in expected_keys if key in payload)


def _iter_json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    in_string = False
    escape = False
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1
    return candidates


def _loads_planner_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_error:
        try:
            return _loads_pythonish_literal(text)
        except (SyntaxError, ValueError):
            pass
        try:
            return json.loads(_replace_jsonish_numeric_expressions(text))
        except (json.JSONDecodeError, ValueError):
            pass
        best_payload: Dict[str, Any] | None = None
        best_score = -1
        last_error: Exception = direct_error
        for candidate in _iter_json_object_candidates(text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    payload = json.loads(_replace_jsonish_numeric_expressions(candidate))
                except (json.JSONDecodeError, ValueError):
                    try:
                        payload = _loads_pythonish_literal(candidate)
                    except (SyntaxError, ValueError) as pythonish_exc:
                        last_error = pythonish_exc
                        continue
            if isinstance(payload, dict):
                score = _planner_schema_score(_unwrap_planner_payload(payload, strict=False))
                if score > best_score:
                    best_payload = payload
                    best_score = score
        if best_payload is not None:
            return best_payload
        raise last_error from direct_error


def _loads_pythonish_literal(text: str) -> Any:
    return _evaluate_pythonish_node(ast.parse(text, mode="eval"), {})


def _evaluate_pythonish_node(node: ast.AST, names: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_pythonish_node(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_pythonish_node(node.operand, names)
        if not isinstance(value, (int, float)):
            raise ValueError("Unary operators are allowed only for numbers.")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _evaluate_pythonish_node(node.left, names)
        right = _evaluate_pythonish_node(node.right, names)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Binary operators are allowed only for numbers.")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and 0 <= right <= 18:
            return left**right
        raise ValueError("Unsupported numeric operator.")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate_pythonish_node(item, names) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _evaluate_pythonish_node(key, names): _evaluate_pythonish_node(value, names)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.ListComp):
        return _evaluate_range_list_comprehension(node, names)
    raise ValueError("Unsupported Python-like JSON value.")


def _evaluate_range_list_comprehension(node: ast.ListComp, names: Dict[str, Any]) -> List[Any]:
    if len(node.generators) != 1:
        raise ValueError("Only one-generator range comprehensions are supported.")
    generator = node.generators[0]
    if generator.ifs or not isinstance(generator.target, ast.Name):
        raise ValueError("Only simple range comprehensions are supported.")
    iterator = generator.iter
    if not (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
        and not iterator.keywords
        and 1 <= len(iterator.args) <= 3
    ):
        raise ValueError("Only range comprehensions are supported.")
    range_args = [_loads_pythonish_literal(ast.unparse(arg)) for arg in iterator.args]
    if not all(isinstance(arg, int) for arg in range_args):
        raise ValueError("Range arguments must be integers.")
    values = list(range(*range_args))
    if len(values) > 5000:
        raise ValueError("Generated range is too large.")
    target_name = generator.target.id
    return [_evaluate_pythonish_node(node.elt, {**names, target_name: value}) for value in values]


def _safe_numeric_expression_value(expression: str) -> int:
    tree = ast.parse(expression, mode="eval")
    allowed_nodes = (ast.Expression, ast.BinOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.USub, ast.UAdd)
    if not all(isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        raise ValueError("Only simple numeric expressions are supported.")

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Pow) and 0 <= right <= 18:
                return left**right
        raise ValueError("Unsupported numeric expression.")

    value = evaluate(tree)
    if abs(value) > 10**18:
        raise ValueError("Numeric expression is too large.")
    return value


def _replace_jsonish_numeric_expressions(text: str) -> str:
    expression_pattern = re.compile(r"(?<![\w.])-?\d+(?:\s*(?:\*\*|\*|\+|-)\s*-?\d+)+(?![\w.])")
    output: List[str] = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        match = expression_pattern.match(text, index)
        if match:
            output.append(str(_safe_numeric_expression_value(match.group(0))))
            index = match.end()
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _unwrap_planner_payload(payload: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    if _planner_schema_score(payload) >= 4:
        return payload

    nested_keys = ("planner", "plan", "result", "data", "output", "json", "arguments")
    for key in nested_keys:
        value = payload.get(key)
        if isinstance(value, dict):
            unwrapped = _unwrap_planner_payload(value, strict=False)
            if _planner_schema_score(unwrapped) >= 4:
                return unwrapped
        if isinstance(value, str) and value.strip():
            try:
                unwrapped = _extract_json_object(value)
            except (json.JSONDecodeError, ValueError):
                continue
            if _planner_schema_score(unwrapped) >= 4:
                return unwrapped

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            try:
                unwrapped = _extract_json_object(content)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if _planner_schema_score(unwrapped) >= 4:
                    return unwrapped

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            content = message.get("content") if isinstance(message, dict) else choice.get("text")
            if isinstance(content, str) and content.strip():
                try:
                    unwrapped = _extract_json_object(content)
                except (json.JSONDecodeError, ValueError):
                    continue
                if _planner_schema_score(unwrapped) >= 4:
                    return unwrapped

    if strict:
        raise ValueError("Ollama did not return planner JSON.")
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
    return cases


def benchmark_planner_solution(
    code: str,
    entrypoint: str,
    test_cases: List[PlannerTestCase],
) -> PlannerRuntimeResult:
    """Benchmark a generated planner solution and report peak runtime and memory."""
    code = (code or "").strip()
    entrypoint = (entrypoint or "").strip()
    if not code or not entrypoint or len(test_cases) < DEFAULT_BENCHMARK_CASE_COUNT:
        return PlannerRuntimeResult(error="Peak Runtime: Not measured")

    violations = validate_code_for_execution(code)
    if violations:
        return PlannerRuntimeResult(error="Peak Runtime: Not measured")

    peaks: List[float] = []
    peak_memory: List[float] = []
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
            peak_memory.append(result.summary.max_peak_memory_kb)
            details.append(
                f"{test_case.name}: peak {result.summary.max_ms:.4f} ms, "
                f"{result.summary.max_peak_memory_kb:.2f} KB"
            )
        else:
            details.append(f"{test_case.name}: not measured")

    if not peaks:
        return PlannerRuntimeResult(details=details, error="Peak Runtime: Not measured")
    return PlannerRuntimeResult(
        measured=True,
        peak_runtime_ms=round(max(peaks), 4),
        peak_memory_kb=round(max(peak_memory), 4) if peak_memory else None,
        details=details,
    )


def _local_choice(question: str) -> tuple[str, str, str]:
    lowered = question.lower()
    if "linear search" in lowered:
        return (
            "Linear scan",
            "Estimated: O(n)",
            "Estimated: O(1)",
        )
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
        "Problem-dependent; provide an Ollama API key for a tailored optimized solution.",
        "Estimated: problem-dependent",
        "Estimated: problem-dependent",
    )


def _linear_search_test_cases() -> List[PlannerTestCase]:
    cases: List[PlannerTestCase] = []
    for index in range(DEFAULT_BENCHMARK_CASE_COUNT):
        size = index + 1
        items = list(range(size))
        if index % 4 == 0:
            target = 0
            expected = 0
        elif index % 4 == 1:
            target = size - 1
            expected = size - 1
        elif index % 4 == 2:
            target = size // 2
            expected = size // 2
        else:
            target = size + 100
            expected = -1
        cases.append(
            PlannerTestCase(
                name=f"case {index + 1}",
                input_text=json.dumps({"args": [items, target], "kwargs": {}}),
                expected_output=str(expected),
            )
        )
    return cases


def _local_linear_search_plan() -> AlgorithmPlannerResult:
    code = (
        "def linear_search(items, target):\n"
        "    for index, value in enumerate(items):\n"
        "        if value == target:\n"
        "            return index\n"
        "    return -1\n"
    )
    test_cases = _linear_search_test_cases()
    return AlgorithmPlannerResult(
        valid=True,
        source="local",
        problem_understanding=(
            "Linear search checks each item in order until it finds the target value. "
            "It returns the matching index, or -1 when the value is not present."
        ),
        step_by_step_optimization_plan=[
            "Start from the first item in the list.",
            "Compare each value with the target exactly once.",
            "Return the index immediately when a match is found.",
            "Return -1 after the loop if no value matched.",
        ],
        best_data_structure_algorithm_choice="Linear scan",
        final_optimized_python_code=code,
        entrypoint="linear_search",
        time_complexity="O(n)",
        space_complexity="O(1)",
        test_cases=test_cases,
        runtime=benchmark_planner_solution(code, "linear_search", test_cases),
    )


def _local_plan(question: str) -> AlgorithmPlannerResult:
    if "linear search" in question.lower():
        return _local_linear_search_plan()

    algorithm, time_complexity, space_complexity = _local_choice(question)
    return AlgorithmPlannerResult(
        valid=True,
        source="local",
        problem_understanding=(
            "Local mode received a natural-language coding problem. Without Ollama, the app avoids "
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
You are an expert competitive programmer. Given the following coding problem, produce the most time- and space-efficient Python solution. First explain the problem, then provide a step-by-step optimization plan, then identify the best data structure or algorithm, then produce the simplest correct optimized Python code. Include time complexity and space complexity. Prefer clear, maintainable Python. Do not over-engineer if a simple approach is already optimal. Include exactly {DEFAULT_BENCHMARK_CASE_COUNT} test cases. Problem: {question}

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
- Use fully evaluated numeric literals in test cases. For example, write 10000000000 instead of 10**10 or 1e10.
- Return exactly {DEFAULT_BENCHMARK_CASE_COUNT} test cases. Runtime and memory will not be shown if fewer cases are returned.
- Do not claim Big-O complexity is exact measured runtime.
"""


def _planner_repair_prompt(question: str, previous_response: str) -> str:
    return f"""
The previous answer did not follow the required planner JSON schema.

Original coding problem:
{question}

Previous answer:
{previous_response}

Rewrite the answer as JSON only with this exact shape:
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

Return exactly {DEFAULT_BENCHMARK_CASE_COUNT} test cases. Do not include markdown, commentary, or code fences.
Use fully evaluated numeric literals in test cases. Do not use arithmetic expressions such as 10**10.
"""


def _model_candidates(preferred_model: str = "") -> List[str]:
    return model_candidates(preferred_model)


def _classify_ollama_error(exc: Exception) -> str:
    return classify_ollama_error(exc)


def _ollama_error_message(category: str) -> str:
    message = ollama_error_message(category)
    if category == "malformed_response":
        return "Ollama responded, but not in the expected planner format. Try again."
    return message


def _extract_message_text(response: Any) -> str:
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content).strip()
    if isinstance(response, dict):
        message_dict = response.get("message")
        if isinstance(message_dict, dict):
            return str(message_dict.get("content", "") or "").strip()
    return ""


def _generate_content(client: Any, model_name: str, prompt: str, json_mode: bool) -> str:
    request: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if json_mode:
        request["format"] = "json"
    return _extract_message_text(client.chat(**request))


def _generate_with_ollama(question: str, api_key: str, model_name: str = "") -> Dict[str, Any]:
    # The API key is used only to construct this request-scoped client.
    api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
    if not api_key:
        raise OllamaPlannerError("invalid_key")
    try:
        from ollama import Client
    except ImportError as exc:
        raise OllamaPlannerError("missing_package") from exc

    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    prompt = _planner_prompt(question)
    last_model_error: Exception | None = None
    last_parse_error: Exception | None = None

    for candidate_model in _model_candidates(model_name):
        try:
            text = _generate_content(client, candidate_model, prompt, json_mode=True)
        except Exception as exc:
            category = _classify_ollama_error(exc)
            if category == "model_unavailable":
                last_model_error = exc
                continue
            raise OllamaPlannerError(category) from exc

        try:
            return _extract_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_parse_error = exc

        try:
            repaired_text = _generate_content(
                client,
                candidate_model,
                _planner_repair_prompt(question, text),
                json_mode=True,
            )
            return _extract_json_object(repaired_text)
        except Exception as exc:
            category = _classify_ollama_error(exc)
            if category == "model_unavailable":
                last_model_error = exc
                continue
            if isinstance(exc, (json.JSONDecodeError, ValueError)):
                last_parse_error = exc
                continue
            raise OllamaPlannerError(category) from exc

    if last_parse_error is not None:
        raise OllamaPlannerError("malformed_response") from last_parse_error
    raise OllamaPlannerError("model_unavailable") from last_model_error


def _safe_ollama_failure(category: str) -> AlgorithmPlannerResult:
    return AlgorithmPlannerResult(
        valid=False,
        source="ollama",
        error=_ollama_error_message(category),
    )


def _result_from_payload(payload: Dict[str, Any]) -> AlgorithmPlannerResult:
    code = _strip_code_fence(str(payload.get("final_optimized_python_code", "")))
    entrypoint = str(payload.get("entrypoint", "")).strip()
    test_cases = _coerce_test_cases(payload.get("test_cases", []))

    result = AlgorithmPlannerResult(
        valid=True,
        source="ollama",
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
        result.error = "Ollama did not return optimized Python code."
        return result
    if len(test_cases) != DEFAULT_BENCHMARK_CASE_COUNT:
        result.valid = False
        result.error = (
            f"Ollama returned {len(test_cases)} executable test case(s); "
            f"{DEFAULT_BENCHMARK_CASE_COUNT} are required before runtime or memory metrics can be shown."
        )
        result.runtime = PlannerRuntimeResult(error="Peak Runtime: Not measured")
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
    if entrypoint and test_cases:
        analysis = analyze_code(code)
        if analysis.valid:
            score = calculate_optimization_score(analysis)
            verified_plan = generate_verified_optimization_candidates(
                original_code=code,
                analysis=analysis,
                score=score,
                entrypoint=entrypoint,
                benchmark_input=json.dumps(
                    {
                        "cases": [
                            {
                                **json.loads(test_case.input_text),
                                "name": test_case.name,
                            }
                            for test_case in test_cases
                        ]
                    }
                ),
            )
            if verified_plan.best_candidate:
                result.final_optimized_python_code = verified_plan.best_candidate.code
                result.time_complexity = verified_plan.best_candidate.actual_time or result.time_complexity
                result.space_complexity = verified_plan.best_candidate.actual_space or result.space_complexity
                result.step_by_step_optimization_plan.extend(
                    [
                        f"{verified_plan.best_candidate.title}: {verified_plan.best_candidate.acceptance_reason}",
                    ]
                )
    return result


def generate_algorithm_optimization_plan(
    question: str,
    api_key: str = "",
    model_name: str = "",
) -> AlgorithmPlannerResult:
    question = (question or "").strip()
    api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
    if not question:
        return AlgorithmPlannerResult(
            valid=False,
            source="validation",
            error="Enter a coding question before generating an optimization plan.",
        )

    if not api_key:
        return _local_plan(question)

    try:
        if model_name:
            payload = _generate_with_ollama(question, api_key, model_name=model_name)
        else:
            payload = _generate_with_ollama(question, api_key)
    except OllamaPlannerError as exc:
        if exc.category == "quota":
            result = _local_plan(question)
            result.error = "Ollama quota was exhausted, so Complexity Lab generated a local fallback plan."
            return result
        if exc.category == "malformed_response":
            result = _local_plan(question)
            result.error = "Ollama responded in an unexpected format, so Complexity Lab generated a local fallback plan."
            return result
        if exc.category == "missing_package":
            result = _local_plan(question)
            result.error = "The Ollama Python SDK was unavailable, so Complexity Lab generated a local fallback plan."
            return result
        return _safe_ollama_failure(exc.category)
    except Exception as exc:
        return _safe_ollama_failure(_classify_ollama_error(exc))
    return _result_from_payload(payload)
