"""Anti-pattern detection derived from AST feature metrics."""

from __future__ import annotations

from typing import Any, Dict, List

from analyzer.models import AntiPattern, OptimizationTarget


def _pattern(
    name: str,
    category: str,
    severity: str,
    message: str,
    evidence: str,
    suggestion: str,
    weight: int,
) -> AntiPattern:
    return AntiPattern(
        name=name,
        category=category,
        severity=severity,
        message=message,
        evidence=evidence,
        suggestion=suggestion,
        weight=weight,
    )


def detect_anti_patterns(metrics: Dict[str, Any]) -> List[AntiPattern]:
    patterns: List[AntiPattern] = []

    if metrics.get("max_loop_depth", 0) >= 3:
        patterns.append(
            _pattern(
                "Deeply nested loops",
                "Algorithmic complexity",
                "critical",
                "Three or more loop levels usually indicate cubic or worse scaling.",
                "Detected maximum loop nesting depth >= 3.",
                "Look for hashing, sorting, prefix sums, two-pointer scans, or vectorized operations.",
                22,
            )
        )
    elif metrics.get("max_loop_depth", 0) == 2:
        patterns.append(
            _pattern(
                "Nested loop burden",
                "Algorithmic complexity",
                "high",
                "Nested loops often move the solution from linear to quadratic time.",
                "Detected a maximum loop nesting depth of 2.",
                "Check whether an inner search can become a dictionary/set lookup or a sorted two-pointer pass.",
                16,
            )
        )

    if metrics.get("sort_in_loop", 0):
        patterns.append(
            _pattern(
                "Sorting inside a loop",
                "Redundant work",
                "high",
                "Sorting repeatedly inside a loop can dominate runtime.",
                "Detected sorted(...) or .sort() while inside a loop.",
                "Sort once before the loop, maintain a heap, or update an ordered structure incrementally.",
                15,
            )
        )
    elif metrics.get("sort_calls", 0):
        patterns.append(
            _pattern(
                "Sorting cost",
                "Algorithmic complexity",
                "medium",
                "Sorting is usually O(n log n), which is fine when intentional but should be explained.",
                "Detected sorted(...) or .sort().",
                "Mention why sorting is acceptable or replace it with hashing if order is not needed.",
                6,
            )
        )

    if metrics.get("slicing_in_loop", 0):
        patterns.append(
            _pattern(
                "Repeated slicing",
                "Memory overhead",
                "high",
                "Slicing inside loops repeatedly copies data and can add hidden O(n) work per iteration.",
                "Detected slice expressions while inside a loop.",
                "Use index windows, iterators, or pass start/end offsets instead of copying sublists.",
                13,
            )
        )

    if metrics.get("list_membership_in_loop", 0):
        patterns.append(
            _pattern(
                "Linear membership checks in a loop",
                "Data structure choice",
                "high",
                "Using `x in list` inside a loop often creates an accidental quadratic scan.",
                "Detected membership checks against a list/tuple literal or likely list within a loop.",
                "Convert the searched collection to a set or dictionary before the loop.",
                14,
            )
        )

    if metrics.get("repeated_expensive_calls", 0):
        patterns.append(
            _pattern(
                "Repeated expensive calls",
                "Redundant work",
                "medium",
                "Repeated traversals such as sum/min/max/sorted over the same expression waste runtime.",
                "Detected repeated expensive call signatures.",
                "Cache the result, precompute once, or maintain a running aggregate.",
                10,
            )
        )

    if metrics.get("repeated_traversals", 0):
        patterns.append(
            _pattern(
                "Repeated traversal of same iterable",
                "Redundant work",
                "medium",
                "Walking the same input multiple times can be okay, but repeated passes should be intentional.",
                "Detected multiple loops over the same iterable expression.",
                "Combine passes when readability remains acceptable, or explain the trade-off.",
                7,
            )
        )

    if metrics.get("recursive_binary_search_markers", 0):
        patterns.append(
            _pattern(
                "Recursive binary search stack",
                "Call-stack and branching",
                "low",
                "Recursive binary search is logarithmic, but each recursive step uses call-stack space.",
                "Detected recursive calls that shrink around a midpoint.",
                "Mention O(log n) stack space or convert to iterative binary search for O(1) auxiliary space.",
                4,
            )
        )
    elif metrics.get("recursive_calls", 0):
        severity = "medium" if metrics.get("memoization_markers", 0) else (
            "critical" if metrics.get("max_recursive_calls_per_function", 0) > 1 else "medium"
        )
        message = "Memoized recursion needs a clear state count and stack-space explanation."
        suggestion = "State the number of states, recurrence transition cost, and stack depth."
        if not metrics.get("memoization_markers", 0):
            message = "Recursion may be elegant, but it needs a clear base case and stack-space explanation."
            suggestion = "Memoize overlapping subproblems or convert to iterative dynamic programming when needed."
        patterns.append(
            _pattern(
                "Recursive solution",
                "Call-stack and branching",
                severity,
                message,
                "Detected a direct recursive call.",
                suggestion,
                12 if severity == "critical" else 5,
            )
        )

    if metrics.get("binary_search_markers", 0):
        patterns.append(
            _pattern(
                "Binary search precondition",
                "Correctness and explanation",
                "low",
                "Binary search is strong only when the input ordering invariant is guaranteed.",
                "Detected left/right/mid boundary markers.",
                "Explicitly state that the input is sorted or explain where sorting happens.",
                4,
            )
        )

    if metrics.get("heap_calls", 0):
        patterns.append(
            _pattern(
                "Heap operation cost",
                "Algorithmic complexity",
                "low",
                "Heap operations are usually O(log k), which is often ideal for top-k and scheduling problems.",
                "Detected heapq-style calls.",
                "Explain why heap size stays bounded and how that affects complexity.",
                4,
            )
        )

    if metrics.get("non_vectorized_loops", 0):
        patterns.append(
            _pattern(
                "Non-vectorized data processing",
                "Data science performance",
                "medium",
                "Manual Python loops over tabular/array-style data can be slow for data science workloads.",
                "Detected loop-local aggregate or transformation patterns.",
                "For NumPy/Pandas inputs, prefer vectorized operations once correctness is clear.",
                8,
            )
        )

    if metrics.get("temp_objects", 0) >= 3:
        patterns.append(
            _pattern(
                "Many temporary objects",
                "Memory overhead",
                "medium",
                "Frequent temporary containers can increase peak memory and allocation overhead.",
                "Detected multiple list/dict/set constructions or conversions.",
                "Use generators, in-place updates, or streaming aggregation where possible.",
                7,
            )
        )

    return sorted(patterns, key=lambda item: item.weight, reverse=True)


def build_optimization_targets(patterns: List[AntiPattern]) -> List[OptimizationTarget]:
    targets: List[OptimizationTarget] = []
    for pattern in patterns[:6]:
        effort = "Low"
        if pattern.severity in {"high", "critical"} and "nested" in pattern.name.lower():
            effort = "Medium"
        if pattern.category == "Algorithmic complexity" and pattern.severity == "critical":
            effort = "High"
        impact = "High" if pattern.severity in {"high", "critical"} else "Medium"
        targets.append(
            OptimizationTarget(
                title=pattern.name,
                impact=impact,
                effort=effort,
                rationale=pattern.suggestion,
            )
        )
    return targets
