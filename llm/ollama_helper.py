"""Optional Ollama Cloud enhancement for natural-language feedback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from analyzer.models import StaticAnalysisResult
from llm.ollama_errors import (
    OLLAMA_HOST,
    classify_ollama_error,
    model_candidates,
    ollama_error_message,
)
from optimization.planner import OptimizationPlan, OptimizedCodeCandidate
from scoring.optimizer_score import ScoreBreakdown
from utils.entrypoints import EntrypointDefinition
from utils.test_case_generator import DEFAULT_BENCHMARK_CASE_COUNT, GeneratedTestCase


class OllamaHelperError(Exception):
    """Ollama failure category safe to render in the UI."""

    def __init__(self, category: str = "request_failed") -> None:
        super().__init__(category)
        self.category = category


def _model_candidates(preferred_model: str = "") -> list[str]:
    return model_candidates(preferred_model)


def _classify_ollama_error(exc: Exception) -> str:
    return classify_ollama_error(exc)


def _ollama_error_message(category: str) -> str:
    return ollama_error_message(category)


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


def _generate_content(client: Any, model_name: str, prompt: str, json_mode: bool = False) -> str:
    request: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if json_mode:
        request["format"] = "json"
    return _extract_message_text(client.chat(**request))


def _request_ollama_text(api_key: str, prompt: str, *, json_mode: bool = False, model_name: str = "") -> str:
    # The API key is used only to construct this request-scoped client.
    api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
    if not api_key:
        raise OllamaHelperError("invalid_key")
    try:
        from ollama import Client
    except ImportError as exc:
        raise OllamaHelperError("missing_package") from exc

    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    last_model_error: Exception | None = None

    for candidate_model in _model_candidates(model_name):
        try:
            return _generate_content(client, candidate_model, prompt, json_mode)
        except Exception as exc:
            category = _classify_ollama_error(exc)
            if category == "model_unavailable":
                last_model_error = exc
                continue
            raise OllamaHelperError(category) from exc

    raise OllamaHelperError("model_unavailable") from last_model_error


def enhance_with_ollama(
    api_key: str,
    code: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    plan: OptimizationPlan,
    model_name: str = "",
) -> Optional[str]:
    """Return an optional Ollama-generated coaching summary.

    Ollama is not used for benchmark measurements, scoring, or static estimates.
    If the package/key is unavailable, the caller receives a readable message.
    """
    if not api_key:
        return None
    facts = {
        "estimated_time": analysis.estimated_time,
        "estimated_space": analysis.estimated_space,
        "confidence": analysis.confidence,
        "confidence_breakdown": analysis.confidence_breakdown,
        "score": score.score,
        "severity": score.severity,
        "bottlenecks": score.bottlenecks,
        "algorithm_patterns": [item.__dict__ for item in analysis.algorithm_patterns[:5]],
        "line_findings": [item.__dict__ for item in analysis.line_findings[:8]],
        "local_plan_summary": plan.summary,
        "has_local_rewrite": bool(plan.optimized_code),
    }
    prompt = f"""
You are an interview coach for Python data science coding interviews.
Use this JSON as the only source of product facts:
```json
{json.dumps(facts, indent=2)}
```

Grounding rules:
- Do not invent benchmark timings, memory usage, scores, or exact complexity.
- Say "estimated" for static complexity.
- Say "measured" only if a number appears in the JSON.
- If the JSON does not prove something, phrase it as a likely interview concern.

Code:
```python
{code[:6000]}
```

Return concise Markdown with exactly these headings:
## Summary
## Interview Answer
## Optimization Alternatives
## Trade-offs
## Edge-Case Questions
"""
    try:
        if model_name:
            text = _request_ollama_text(api_key, prompt, model_name=model_name)
        else:
            text = _request_ollama_text(api_key, prompt)
    except OllamaHelperError as exc:
        return f"Ollama enhancement failed. {_ollama_error_message(exc.category)}"
    except Exception as exc:
        return f"Ollama enhancement failed. {_ollama_error_message(_classify_ollama_error(exc))}"
    return text or "Ollama returned an empty response."


def _extract_json_object(text: str) -> dict:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start : end + 1])
        raise


def generate_optimized_code_with_ollama(
    api_key: str,
    code: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    plan: OptimizationPlan,
    entrypoint: str,
    level: str = "medium_refactor",
    retry_count: int = 0,
    rejection_reasons: Optional[list] = None,
    model_name: str = "",
) -> tuple[Optional[OptimizedCodeCandidate], Optional[str]]:
    """Ask Ollama for one structured optimized-code candidate.

    The returned candidate is not trusted by the app until the local planner
    validates syntax, safety checks, entrypoint preservation, and estimated
    complexity/score improvement.
    """
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return None, "Ollama API key was not provided."
    level_titles = {
        "quick_win": "Quick Win",
        "medium_refactor": "Medium Refactor",
        "advanced": "Advanced Improvement",
    }
    level_rules = {
        "quick_win": [
            "Level: Quick Win",
            "Generate the simplest safe cleanup.",
            "Do not change the algorithm unless clearly beneficial.",
            "Prioritize readability, edge cases, and avoiding obvious repeated work.",
            "The code must be short and interview-friendly.",
        ],
        "medium_refactor": [
            "Level: Medium Refactor",
            "Use better pruning, precomputation, and data structures when useful.",
            "Reduce avoidable loops, slicing, or repeated membership checks.",
            "Keep the code understandable.",
        ],
        "advanced": [
            "Level: Advanced Improvement",
            "Generate the fastest practical implementation for the same contract.",
            "Minimize runtime first, then peak memory.",
            "Use advanced algorithmic pruning or memoization if valid.",
            "Do not change return type or semantics.",
        ],
    }
    level = level if level in level_rules else "medium_refactor"
    level_rule_lines = "\n".join(f"- {rule}" for rule in level_rules[level])
    output_sensitive = bool(analysis.metrics.get("output_sensitive") or analysis.metrics.get("word_break_ii_output_sensitive"))
    facts = {
        "entrypoint": entrypoint,
        "requested_level": level_titles[level],
        "estimated_time": analysis.estimated_time,
        "estimated_space": analysis.estimated_space,
        "confidence": analysis.confidence,
        "score": score.score,
        "severity": score.severity,
        "bottlenecks": score.bottlenecks,
        "algorithm_patterns": [item.__dict__ for item in analysis.algorithm_patterns[:5]],
        "line_findings": [item.__dict__ for item in analysis.line_findings[:8]],
        "local_plan_summary": plan.summary,
        "output_sensitive": output_sensitive,
        "previous_rejection_reasons": rejection_reasons or [],
    }
    previous_rejection_reasons = "\n".join(f"- {reason}" for reason in (rejection_reasons or []))
    if not previous_rejection_reasons:
        previous_rejection_reasons = "- None"
    prompt = f"""
You generate optimized Python code for an interview-prep complexity analyzer.
Return JSON only. No Markdown wrapper.

Required JSON shape:
{{
  "step_by_step_plan": ["short concrete step", "..."],
  "optimized_code": "complete Python code",
  "explanation": "short explanation of why this is better",
  "validation_tests": ["assert ...", "..."],
  "expected_time": "O(...)",
  "expected_space": "O(...)"
}}

Rules:
- Preserve the configured entrypoint exactly: {entrypoint!r}.
- Generate exactly one {level_titles[level]} candidate.
{level_rule_lines}
- If the entrypoint contains a dot, such as "Solution.singleNumber", return the same class name and method name.
- For LeetCode-style code, keep the class wrapper.
- Keep the same method arguments and return type behavior.
- Keep the function callable with the same benchmark input shape.
- Generate the simplest correct code that improves or preserves the public behavior.
- Prefer lower time complexity first.
- Prefer lower auxiliary space second.
- Do not add extra data structures unless they reduce time complexity or benchmark runtime.
- Do not sort or precompute helper structures unless they are necessary or likely to improve runtime.
- If the current code is already close to optimal, return a minimal cleanup only.
- The candidate will be rejected if static score decreases, estimated time worsens, estimated space worsens, benchmark runtime is worse, benchmark peak memory is worse, or code is more complex without measured benefit.
- You must generate a candidate that is strictly better by at least one of: lower estimated time complexity, lower estimated space complexity, higher optimization score without worse time/space, lower measured runtime on the benchmark input without higher memory, or lower measured memory without higher runtime.
- If no better candidate exists, return the original algorithm cleaned only if it is not slower, not higher memory, and not lower score.
- Add edge-case handling when it does not change the required return contract.
- Return validation tests covering edge cases.
- Do not import filesystem, process, network, introspection, or dynamic execution modules.
- Do not call open, eval, exec, compile, input, getattr, setattr, globals, locals, or __import__.
- If the problem is output-sensitive, do not claim lower than output-size complexity. Optimize search overhead and memory use only.
- If previous rejection reasons are listed, correct them.
- Return only one candidate.

The previous candidate was rejected for these reasons:
{previous_rejection_reasons}

Local facts:
```json
{json.dumps(facts, indent=2)}
```

Original code:
```python
{code[:6000]}
```
"""
    try:
        if model_name:
            text = _request_ollama_text(api_key, prompt, json_mode=True, model_name=model_name)
        else:
            text = _request_ollama_text(api_key, prompt, json_mode=True)
        payload = _extract_json_object(text)
    except OllamaHelperError as exc:
        return None, f"Ollama optimization generation failed. {_ollama_error_message(exc.category)}"
    except (json.JSONDecodeError, ValueError):
        return None, f"Ollama optimization generation failed. {_ollama_error_message('malformed_response')}"
    except Exception as exc:
        return None, f"Ollama optimization generation failed. {_ollama_error_message(_classify_ollama_error(exc))}"

    optimized_code = str(payload.get("optimized_code", "")).strip()
    steps = payload.get("step_by_step_plan", [])
    tests = payload.get("validation_tests", [])
    if not isinstance(steps, list):
        steps = []
    if not isinstance(tests, list):
        tests = []
    if not optimized_code:
        return None, "Ollama did not return optimized_code."

    return OptimizedCodeCandidate(
        source="ollama",
        code=optimized_code,
        explanation=str(payload.get("explanation", "")).strip(),
        level=level,
        title=level_titles[level],
        expected_time=str(payload.get("expected_time", "")).strip(),
        expected_space=str(payload.get("expected_space", "")).strip(),
        step_by_step_plan=[str(item) for item in steps if str(item).strip()],
        validation_tests=[str(item) for item in tests if str(item).strip()],
        confidence=0.72,
        retry_count=retry_count,
    ), None


def generate_test_cases_with_ollama(
    api_key: str,
    code: str,
    definition: EntrypointDefinition,
    model_name: str = "",
    target_count: int = DEFAULT_BENCHMARK_CASE_COUNT,
) -> tuple[list[GeneratedTestCase], str]:
    """Ask Ollama for benchmark-ready cases for the selected entrypoint."""
    api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
    if not api_key:
        return [], "Ollama API key was not provided."
    signature = {
        "entrypoint": definition.callable_name,
        "args": definition.benchmark_args,
        "required_positional_count": definition.required_positional_count,
        "keyword_only_args": definition.keyword_only_args,
        "required_keyword_only_args": definition.required_keyword_only_args,
        "annotations": definition.annotations,
    }
    prompt = f"""
Generate exactly {target_count} executable benchmark test cases for this Python entrypoint.
Return JSON only with this exact shape:
{{
  "test_cases": [
    {{"name": "short name", "input": {{"args": [], "kwargs": {{}}}}, "expected_output": ""}}
  ]
}}

Entrypoint signature facts:
```json
{json.dumps(signature, indent=2)}
```

Rules:
- Generate exactly {target_count} test cases.
- Every input must be JSON-compatible.
- Prefer kwargs matching the entrypoint argument names.
- Cover empty, tiny, duplicate, negative, sorted, reverse-sorted, medium, and stress-style cases when relevant.
- Do not include code fences or Markdown.

Code:
```python
{code[:6000]}
```
"""
    try:
        if model_name:
            text = _request_ollama_text(api_key, prompt, json_mode=True, model_name=model_name)
        else:
            text = _request_ollama_text(api_key, prompt, json_mode=True)
        payload = _extract_json_object(text)
    except OllamaHelperError as exc:
        return [], _ollama_error_message(exc.category)
    except Exception as exc:
        return [], _ollama_error_message(_classify_ollama_error(exc))

    raw_cases = payload.get("test_cases", [])
    if not isinstance(raw_cases, list):
        return [], "Ollama did not return a test_cases list."
    cases: list[GeneratedTestCase] = []
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            continue
        raw_input = item.get("input", {"args": [], "kwargs": {}})
        if not isinstance(raw_input, dict):
            raw_input = {"args": [raw_input], "kwargs": {}}
        args = raw_input.get("args", [])
        kwargs = raw_input.get("kwargs", {})
        if not isinstance(args, list):
            args = [args]
        if not isinstance(kwargs, dict):
            kwargs = {}
        cases.append(
            GeneratedTestCase(
                name=str(item.get("name", f"LLM case {index}") or f"LLM case {index}"),
                benchmark_input=json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False),
                expected_output=str(item.get("expected_output", item.get("expected", "")) or ""),
                reason="Generated by the selected LLM model.",
            )
        )
    if len(cases) != target_count:
        return [], f"Ollama returned {len(cases)} valid test case(s), expected {target_count}."
    return cases, ""
