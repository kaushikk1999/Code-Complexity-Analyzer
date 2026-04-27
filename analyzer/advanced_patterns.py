"""Higher-level static analysis helpers for interview-oriented pattern detection."""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

from analyzer.models import AlgorithmPattern, LineFinding

CONTAINER_TYPE_BY_NODE = {
    ast.List: "list",
    ast.Tuple: "tuple",
    ast.Set: "set",
    ast.Dict: "dict",
}


def call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<dynamic>"


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


class FunctionCatalog(ast.NodeVisitor):
    """Collect function call graph and simple assignment data-flow hints."""

    def __init__(self) -> None:
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.container_types_by_scope: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._scope = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self._scope
        self._scope = node.name
        self.generic_visit(node)
        self._scope = previous

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Assign(self, node: ast.Assign) -> None:
        inferred = infer_container_type(node.value, self.container_types_by_scope[self._scope])
        if inferred:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.container_types_by_scope[self._scope][target.id] = inferred
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value and isinstance(node.target, ast.Name):
            inferred = infer_container_type(node.value, self.container_types_by_scope[self._scope])
            if inferred:
                self.container_types_by_scope[self._scope][node.target.id] = inferred
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node.func)
        if name != "<dynamic>":
            self.call_graph[self._scope].add(name)
        self.generic_visit(node)

    def as_call_graph(self, defined_functions: Iterable[str]) -> Dict[str, List[str]]:
        defined = set(defined_functions)
        graph: Dict[str, List[str]] = {}
        for scope, calls in self.call_graph.items():
            graph[scope] = sorted(name for name in calls if name in defined or name in {"heappush", "heappop"})
        return graph


def infer_container_type(node: ast.AST, known_types: Dict[str, str]) -> str:
    for cls, label in CONTAINER_TYPE_BY_NODE.items():
        if isinstance(node, cls):
            return label
    if isinstance(node, (ast.ListComp,)):
        return "list"
    if isinstance(node, ast.SetComp):
        return "set"
    if isinstance(node, ast.DictComp):
        return "dict"
    if isinstance(node, ast.Call):
        name = call_name(node.func)
        if name in {"list", "tuple", "set", "dict"}:
            return name
        if name == "Counter":
            return "dict"
    if isinstance(node, ast.Name):
        return known_types.get(node.id, "")
    return ""


def confidence_breakdown(metrics: Dict[str, Any], caveats: List[str], confidence: float) -> Dict[str, float]:
    syntax = 1.0
    structural = max(0.35, min(1.0, 0.88 - 0.06 * metrics.get("while_loops", 0)))
    data_flow = max(0.25, min(1.0, 0.76 + 0.03 * metrics.get("known_container_membership", 0) - 0.05 * metrics.get("unknown_container_membership", 0)))
    pattern = max(0.25, min(1.0, 0.58 + 0.08 * metrics.get("algorithm_pattern_count", 0)))
    caveat_score = max(0.25, min(1.0, 0.90 - 0.05 * len(caveats)))
    return {
        "syntax": round(syntax, 2),
        "structural_signals": round(structural, 2),
        "data_flow_hints": round(data_flow, 2),
        "pattern_recognition": round(pattern, 2),
        "caveat_pressure": round(caveat_score, 2),
        "overall": round(confidence, 2),
    }


def detect_algorithm_patterns(
    metrics: Dict[str, Any],
    evidence: List[str],
    calls: Set[str],
    code_text: str,
) -> List[AlgorithmPattern]:
    lowered = code_text.lower()
    patterns: List[AlgorithmPattern] = []

    def add(name: str, confidence: float, signals: List[str], note: str) -> None:
        patterns.append(
            AlgorithmPattern(
                name=name,
                confidence=round(max(0.0, min(1.0, confidence)), 2),
                evidence=signals,
                interview_note=note,
            )
        )

    if metrics.get("max_loop_depth", 0) >= 2 and "target" in lowered and ("nums" in lowered or "arr" in lowered):
        add(
            "Brute-force pair search",
            0.82,
            ["Nested loops", "`target` parameter/variable", "array-like variable name"],
            "A strong interviewer answer should mention replacing nested search with a hash map or sorting + two pointers.",
        )
    if metrics.get("dict_builds", 0) or metrics.get("set_builds", 0) or metrics.get("dict_membership_tests", 0):
        if "target" in lowered or "complement" in lowered:
            add(
                "Hash-map lookup solution",
                0.78,
                ["Dictionary/set construction", "membership lookup", "target/complement naming"],
                "Explain the O(n) memory trade-off for average O(1) lookup.",
            )
    if (
        metrics.get("max_loop_depth", 0) == 1
        and "prices" in lowered
        and ("min_price" in lowered or "minimum price" in lowered)
        and ("max_profit" in lowered or "profit" in lowered)
    ):
        add(
            "One-pass stock profit scan",
            0.82,
            ["single loop over prices", "minimum-price tracking", "maximum-profit tracking"],
            "Explain the invariant: each day compares against the lowest prior price, so total work is linear.",
        )
    if metrics.get("while_loops", 0) and metrics.get("index_pointer_updates", 0) >= 2:
        add(
            "Two-pointer scan",
            0.72,
            ["while-loop", "two index updates"],
            "Explain why each pointer moves monotonically so total work is linear.",
        )
    if metrics.get("binary_search_markers", 0):
        add(
            "Binary search",
            0.82,
            ["mid computation", "left/right boundary updates"],
            "State the sorted-input requirement and O(log n) search behavior.",
        )
    if metrics.get("recursive_binary_search_markers", 0):
        add(
            "Recursive binary search",
            0.88,
            ["recursive call", "midpoint calculation", "shrinking left/right bounds"],
            "Explain that only one half is searched each call, giving O(log n) time and O(log n) stack space.",
        )
    if metrics.get("sliding_window_markers", 0):
        add(
            "Sliding window",
            0.72,
            ["window variable naming or left/right pointer pattern"],
            "Explain the invariant maintained by the window and why each element enters/leaves once.",
        )
    if metrics.get("prefix_sum_markers", 0):
        add(
            "Prefix sum",
            0.76,
            ["prefix/cumulative naming or running sum"],
            "Explain how precomputation turns repeated range queries into O(1) or one-pass checks.",
        )
    if metrics.get("heap_calls", 0) or {"heappush", "heappop"} & calls:
        add(
            "Heap / priority queue",
            0.82,
            ["heapq operation"],
            "Mention O(log k) updates and why a heap is appropriate for top-k or scheduling problems.",
        )
    if metrics.get("recursive_binary_search_markers", 0):
        pass
    elif metrics.get("recursive_calls", 0) and (metrics.get("memoization_markers", 0) or "lru_cache" in calls):
        add(
            "Memoized recursion / dynamic programming",
            0.84,
            ["recursive call", "memo/cache marker"],
            "Explain overlapping subproblems and how memoization changes repeated recursion into state-count complexity.",
        )
    elif metrics.get("recursive_calls", 0):
        add(
            "Plain recursion",
            0.66,
            ["direct recursive call"],
            "Be ready to discuss stack space, base cases, and whether subproblems repeat.",
        )
    if metrics.get("sort_calls", 0) and metrics.get("index_pointer_updates", 0) >= 2:
        add(
            "Sorting + two pointers",
            0.76,
            ["sorting call", "pointer updates"],
            "Explain the O(n log n) sort and O(n) scan trade-off.",
        )
    if metrics.get("graph_markers", 0):
        add(
            "Graph traversal",
            0.70,
            ["visited/queue/stack/neighbors naming"],
            "State complexity in terms of vertices and edges: O(V + E).",
        )
    if metrics.get("non_vectorized_loops", 0) and any(token in lowered for token in ["dataframe", "pandas", "numpy", "np.", "pd."]):
        add(
            "Non-vectorized data processing",
            0.70,
            ["manual loop", "data science library marker"],
            "For data science interviews, mention when vectorization is clearer and faster.",
        )

    if not patterns and evidence:
        add(
            "General iterative solution",
            0.48,
            evidence[:2],
            "Explain the traversal count, auxiliary containers, and edge cases clearly.",
        )

    return sorted(patterns, key=lambda item: item.confidence, reverse=True)


def finding(
    line: int,
    title: str,
    severity: str,
    category: str,
    detail: str,
    suggestion: str,
) -> LineFinding:
    return LineFinding(
        line=line,
        title=title,
        severity=severity,
        category=category,
        detail=detail,
        suggestion=suggestion,
    )
