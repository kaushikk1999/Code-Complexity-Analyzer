"""Heuristic complexity rules.

The rules in this module intentionally estimate complexity instead of trying to
prove it. They are designed for interview-prep snippets where AST features are
useful signals, but dynamic behavior and input distributions are unknown.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

COMPLEXITY_ORDER = {
    "O(1)": 0,
    "O(log n)": 1,
    "O(n)": 2,
    "O(n log n)": 3,
    "O(n^2)": 4,
    "O(n^2 log n)": 5,
    "O(n^3)": 6,
    "O(2^n)": 7,
    "O(n!)": 8,
    "Unknown": 9,
}


def complexity_rank(label: str) -> int:
    return COMPLEXITY_ORDER.get(label, COMPLEXITY_ORDER["Unknown"])


def max_complexity(labels: Iterable[str]) -> str:
    labels = list(labels)
    if not labels:
        return "O(1)"
    return max(labels, key=complexity_rank)


def estimate_time_complexity(metrics: Dict[str, Any]) -> str:
    max_depth = int(metrics.get("max_loop_depth", 0) or 0)
    recursive_calls = int(metrics.get("recursive_calls", 0) or 0)
    max_recursive_calls_per_function = int(
        metrics.get("max_recursive_calls_per_function", recursive_calls) or 0
    )
    sort_calls = int(metrics.get("sort_calls", 0) or 0)
    sort_in_loop = int(metrics.get("sort_in_loop", 0) or 0)
    slicing_in_loop = int(metrics.get("slicing_in_loop", 0) or 0)
    nested_comprehension = int(metrics.get("max_comprehension_depth", 0) or 0)

    candidates: List[str] = ["O(1)"]

    if recursive_calls:
        if metrics.get("recursive_binary_search_markers", 0):
            candidates.append("O(log n)")
        elif metrics.get("memoization_markers", 0):
            candidates.append("O(n)")
        else:
            candidates.append("O(2^n)" if max_recursive_calls_per_function > 1 else "O(n)")

    binary_search_like = bool(metrics.get("binary_search_markers", 0) and max_depth <= 1)
    if binary_search_like:
        candidates.append("O(log n)")

    if max_depth >= 3 or nested_comprehension >= 3:
        candidates.append("O(n^3)")
    elif max_depth == 2 or nested_comprehension == 2:
        candidates.append("O(n^2)")
    elif (max_depth == 1 and not binary_search_like) or metrics.get("comprehension_count", 0):
        candidates.append("O(n)")

    if sort_calls:
        candidates.append("O(n^2 log n)" if sort_in_loop else "O(n log n)")

    if metrics.get("heap_calls", 0) and max_depth >= 1:
        candidates.append("O(n log n)")

    if slicing_in_loop and max_depth >= 1:
        candidates.append("O(n^2)")

    return max_complexity(candidates)


def estimate_space_complexity(metrics: Dict[str, Any]) -> str:
    if metrics.get("recursive_binary_search_markers", 0):
        return "O(log n)"
    if metrics.get("recursive_calls", 0):
        return "O(n)"
    if metrics.get("max_comprehension_depth", 0) >= 2:
        return "O(n^2)"
    if (
        metrics.get("list_builds", 0)
        or metrics.get("dict_builds", 0)
        or metrics.get("set_builds", 0)
        or metrics.get("slicing_count", 0)
        or metrics.get("comprehension_count", 0)
        or metrics.get("temp_objects", 0)
    ):
        return "O(n)"
    return "O(1)"


def estimate_confidence(metrics: Dict[str, Any], caveats: List[str]) -> float:
    confidence = 0.84
    confidence -= min(0.22, 0.04 * len(caveats))
    confidence -= min(0.12, 0.03 * int(metrics.get("while_loops", 0) or 0))
    confidence -= min(0.14, 0.04 * int(metrics.get("recursive_calls", 0) or 0))
    confidence -= min(0.12, 0.02 * int(metrics.get("unknown_calls", 0) or 0))
    confidence -= min(0.10, 0.04 * int(metrics.get("imports", 0) or 0))
    confidence += min(0.08, 0.02 * int(metrics.get("clear_linear_signals", 0) or 0))
    return round(max(0.35, min(0.92, confidence)), 2)


def complexity_label_for_score(score: float) -> str:
    if score >= 86:
        return "Excellent"
    if score >= 72:
        return "Strong"
    if score >= 55:
        return "Needs Review"
    if score >= 38:
        return "Risky"
    return "Critical"
