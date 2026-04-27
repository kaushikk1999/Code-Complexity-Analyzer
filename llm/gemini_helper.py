"""Optional Gemini enhancement for natural-language feedback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from analyzer.models import StaticAnalysisResult
from optimization.planner import OptimizationPlan, OptimizedCodeCandidate
from scoring.optimizer_score import ScoreBreakdown

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL_FALLBACKS = (
    DEFAULT_GEMINI_MODEL,
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)


class GeminiHelperError(Exception):
    """Gemini failure category safe to render in the UI."""

    def __init__(self, category: str = "request_failed") -> None:
        super().__init__(category)
        self.category = category


def _model_candidates() -> list[str]:
    candidates = [os.getenv("GEMINI_MODEL", "").strip(), *GEMINI_MODEL_FALLBACKS]
    unique_candidates: list[str] = []
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
        "malformed_response": "Gemini responded, but not in the expected format. Try again.",
        "missing_package": "The Google Gen AI SDK is not installed. Install google-genai and try again.",
    }
    return messages.get(category, "Gemini request failed. Check the API key and try again.")


def _generate_content(client: Any, model_name: str, prompt: str, config: Any = None) -> Any:
    request: dict[str, Any] = {"model": model_name, "contents": prompt}
    if config is not None:
        request["config"] = config
    return client.models.generate_content(**request)


def _request_gemini_text(api_key: str, prompt: str, *, json_mode: bool = False) -> str:
    # The API key is used only to construct this request-scoped client.
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiHelperError("missing_package") from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    last_model_error: Exception | None = None

    for model_name in _model_candidates():
        try:
            response = _generate_content(client, model_name, prompt, config)
        except Exception as exc:
            category = _classify_gemini_error(exc)
            if category == "model_unavailable":
                last_model_error = exc
                continue
            raise GeminiHelperError(category) from exc
        text = getattr(response, "text", None)
        return text.strip() if text else ""

    raise GeminiHelperError("model_unavailable") from last_model_error


def enhance_with_gemini(
    api_key: str,
    code: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    plan: OptimizationPlan,
) -> Optional[str]:
    """Return an optional Gemini-generated coaching summary.

    Gemini is not used for benchmark measurements, scoring, or static estimates.
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
        text = _request_gemini_text(api_key, prompt)
    except GeminiHelperError as exc:
        return f"Gemini enhancement failed. {_gemini_error_message(exc.category)}"
    except Exception as exc:
        return f"Gemini enhancement failed. {_gemini_error_message(_classify_gemini_error(exc))}"
    return text or "Gemini returned an empty response."


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


def generate_optimized_code_with_gemini(
    api_key: str,
    code: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    plan: OptimizationPlan,
    entrypoint: str,
    retry_count: int = 0,
    rejection_reasons: Optional[list] = None,
) -> tuple[Optional[OptimizedCodeCandidate], Optional[str]]:
    """Ask Gemini for one structured optimized-code candidate.

    The returned candidate is not trusted by the app until the local planner
    validates syntax, safety checks, entrypoint preservation, and estimated
    complexity/score improvement.
    """
    if not api_key:
        return None, "Gemini API key was not provided."
    facts = {
        "entrypoint": entrypoint,
        "estimated_time": analysis.estimated_time,
        "estimated_space": analysis.estimated_space,
        "confidence": analysis.confidence,
        "score": score.score,
        "severity": score.severity,
        "bottlenecks": score.bottlenecks,
        "algorithm_patterns": [item.__dict__ for item in analysis.algorithm_patterns[:5]],
        "line_findings": [item.__dict__ for item in analysis.line_findings[:8]],
        "local_plan_summary": plan.summary,
        "previous_rejection_reasons": rejection_reasons or [],
    }
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
- Preserve the configured entrypoint function name exactly: {entrypoint!r}.
- Keep the function callable with the same benchmark input shape.
- Generate the best practical version of the code.
- Prefer the lowest possible time complexity.
- If the current code is already asymptotically optimal, still return a cleaner, simpler, edge-case-safe reference implementation with the same function name.
- Prefer the lowest possible auxiliary space only when it does not worsen time complexity.
- Use data structures when they reduce asymptotic time or simplify correctness.
- Add edge-case handling when it does not change the required return contract.
- Do not import filesystem, process, network, introspection, or dynamic execution modules.
- Do not call open, eval, exec, compile, input, getattr, setattr, globals, locals, or __import__.
- If previous rejection reasons are listed, correct them.
- Return only one candidate.

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
        text = _request_gemini_text(api_key, prompt, json_mode=True)
        payload = _extract_json_object(text)
    except GeminiHelperError as exc:
        return None, f"Gemini optimization generation failed. {_gemini_error_message(exc.category)}"
    except (json.JSONDecodeError, ValueError):
        return None, f"Gemini optimization generation failed. {_gemini_error_message('malformed_response')}"
    except Exception as exc:
        return None, f"Gemini optimization generation failed. {_gemini_error_message(_classify_gemini_error(exc))}"

    optimized_code = str(payload.get("optimized_code", "")).strip()
    steps = payload.get("step_by_step_plan", [])
    tests = payload.get("validation_tests", [])
    if not isinstance(steps, list):
        steps = []
    if not isinstance(tests, list):
        tests = []
    if not optimized_code:
        return None, "Gemini did not return optimized_code."

    return OptimizedCodeCandidate(
        source="gemini",
        code=optimized_code,
        explanation=str(payload.get("explanation", "")).strip(),
        step_by_step_plan=[str(item) for item in steps if str(item).strip()],
        validation_tests=[str(item) for item in tests if str(item).strip()],
        confidence=0.72,
        retry_count=retry_count,
    ), None
