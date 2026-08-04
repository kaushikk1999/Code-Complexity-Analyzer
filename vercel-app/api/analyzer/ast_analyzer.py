"""AST-powered static analysis for pasted Python snippets."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from analyzer.advanced_patterns import (
    FunctionCatalog,
    confidence_breakdown,
    detect_algorithm_patterns,
    finding,
    infer_container_type,
    is_word_break_ii_pattern,
)
from analyzer.anti_patterns import build_optimization_targets, detect_anti_patterns
from analyzer.complexity_rules import (
    estimate_confidence,
    estimate_space_complexity,
    estimate_time_complexity,
    max_complexity,
)
from analyzer.models import AlgorithmPattern, FunctionAnalysis, LineFinding, StaticAnalysisResult

EXPENSIVE_CALLS = {"sum", "min", "max", "sorted", "list", "tuple", "set", "any", "all"}
DATA_PROCESSING_CALLS = {"append", "extend", "sum", "mean", "apply", "map", "filter"}


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _merge_metrics(metric_sets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = defaultdict(int)
    max_keys = {
        "max_loop_depth",
        "max_comprehension_depth",
        "max_recursive_calls_per_function",
    }
    for metrics in metric_sets:
        for key, value in metrics.items():
            if isinstance(value, bool):
                merged[key] = int(merged.get(key, 0)) + int(value)
            elif isinstance(value, (int, float)):
                if key in max_keys:
                    merged[key] = max(merged.get(key, 0), value)
                else:
                    merged[key] = merged.get(key, 0) + value
    return dict(merged)


def _names_in(node: ast.AST) -> Set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_recursive_call(node: ast.AST, function_name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function_name


def _has_mid_assignment(node: ast.AST) -> bool:
    value = None
    targets: List[ast.AST] = []
    if isinstance(node, ast.Assign):
        value = node.value
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        value = node.value
        targets = [node.target]
    if value is None:
        return False
    target_names = {
        target.id
        for target in targets
        if isinstance(target, ast.Name) and target.id.lower() in {"mid", "middle"}
    }
    if not target_names or not any(isinstance(child, ast.FloorDiv) for child in ast.walk(value)):
        return False
    boundary_names = _names_in(value)
    return len(boundary_names & {"left", "right", "lo", "hi", "low", "high", "start", "end"}) >= 2


def _has_boundary_base_case(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    names = _names_in(node)
    if len(names & {"left", "lo", "low", "start"}) < 1:
        return False
    if len(names & {"right", "hi", "high", "end"}) < 1:
        return False
    return any(isinstance(operator, (ast.Gt, ast.GtE)) for operator in node.ops)


def _shrinks_around_mid(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.BinOp) or not isinstance(child.op, (ast.Add, ast.Sub)):
            continue
        names = _names_in(child)
        constants = [
            grandchild.value
            for grandchild in ast.walk(child)
            if isinstance(grandchild, ast.Constant) and isinstance(grandchild.value, int)
        ]
        if names & {"mid", "middle"} and 1 in constants:
            return True
    return False


def _has_recursive_branching(body: List[ast.stmt], function_name: str) -> bool:
    wrapper = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(wrapper):
        if isinstance(node, ast.If) and any(_is_recursive_call(child, function_name) for child in ast.walk(node)):
            return True
    return False


def _detect_recursive_binary_search(function_name: str, body: List[ast.stmt]) -> bool:
    wrapper = ast.Module(body=body, type_ignores=[])
    recursive_calls = [node for node in ast.walk(wrapper) if _is_recursive_call(node, function_name)]
    if not recursive_calls:
        return False
    has_mid = any(_has_mid_assignment(node) for node in ast.walk(wrapper))
    has_base_case = any(_has_boundary_base_case(node) for node in ast.walk(wrapper))
    has_shrinking_call = any(_shrinks_around_mid(call) for call in recursive_calls)
    return has_mid and has_base_case and has_shrinking_call and _has_recursive_branching(body, function_name)


class FeatureCollector(ast.NodeVisitor):
    def __init__(
        self,
        function_name: Optional[str] = None,
        skip_nested_functions: bool = True,
        known_types: Optional[Dict[str, str]] = None,
    ) -> None:
        self.function_name = function_name
        self.skip_nested_functions = skip_nested_functions
        self.loop_depth = 0
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.evidence: List[str] = []
        self.caveats: List[str] = []
        self.line_findings: List[LineFinding] = []
        self.calls: Set[str] = set()
        self.known_types: Dict[str, str] = dict(known_types or {})
        self._call_signatures: Counter[str] = Counter()
        self._loop_iterables: Counter[str] = Counter()
        self._recursive_calls_in_current_function = 0

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.skip_nested_functions:
            self.metrics["nested_functions"] += 1
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.skip_nested_functions:
            self.metrics["nested_functions"] += 1
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.metrics["imports"] += len(node.names)
        names = ", ".join(alias.name for alias in node.names)
        self.caveats.append(f"Imported modules may hide runtime costs: {names}.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.metrics["imports"] += len(node.names)
        module = node.module or ""
        self.caveats.append(f"Imported module may hide runtime costs: {module}.")

    def visit_For(self, node: ast.For) -> None:
        self.metrics["loop_count"] += 1
        self.metrics["for_loops"] += 1
        iter_expr = _safe_unparse(node.iter)
        self._record_loop_bound(node.iter, getattr(node, "lineno", 0))
        self._loop_iterables[iter_expr] += 1
        if self._loop_iterables[iter_expr] > 1:
            self.metrics["repeated_traversals"] += 1
        self._enter_loop(f"for-loop over `{iter_expr}` at line {getattr(node, 'lineno', '?')}", getattr(node, "lineno", 0))
        self.visit(node.target)
        self.visit(node.iter)
        for child in node.body:
            self.visit(child)
        self.loop_depth -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_While(self, node: ast.While) -> None:
        self.metrics["loop_count"] += 1
        self.metrics["while_loops"] += 1
        self._detect_while_pattern(node)
        self.caveats.append("A while-loop depends on runtime state; exact iteration count is unknown.")
        self._enter_loop(f"while-loop at line {getattr(node, 'lineno', '?')}", getattr(node, "lineno", 0))
        self.visit(node.test)
        for child in node.body:
            self.visit(child)
        self.loop_depth -= 1
        for child in node.orelse:
            self.visit(child)

    def _enter_loop(self, evidence: str, line: int) -> None:
        self.loop_depth += 1
        self.metrics["max_loop_depth"] = max(self.metrics["max_loop_depth"], self.loop_depth)
        self.metrics["clear_linear_signals"] += 1
        self.evidence.append(evidence)
        if self.loop_depth >= 2:
            self.line_findings.append(
                finding(
                    line,
                    "Nested loop bottleneck",
                    "high",
                    "Algorithmic complexity",
                    "This loop is nested inside another loop, which often creates quadratic or worse growth.",
                    "Check whether hashing, sorting + two pointers, prefix sums, or pruning can remove the inner traversal.",
                )
            )

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, "list comprehension")
        self.metrics["list_builds"] += 1

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, "set comprehension")
        self.metrics["set_builds"] += 1

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, "dict comprehension")
        self.metrics["dict_builds"] += 1

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, "generator expression")

    def _visit_comprehension(self, node: ast.AST, label: str) -> None:
        generators = getattr(node, "generators", [])
        depth = len(generators)
        self.metrics["comprehension_count"] += 1
        self.metrics["max_comprehension_depth"] = max(self.metrics["max_comprehension_depth"], depth)
        self.evidence.append(f"Detected {label} with {depth} generator(s).")
        if depth >= 2:
            self.caveats.append("Nested comprehensions may be concise but can hide nested-loop cost.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name_value = self._call_name(node.func)
        self.calls.add(call_name_value)
        signature = f"{call_name_value}({_safe_unparse(node.args[0]) if node.args else ''})"
        self._call_signatures[signature] += 1

        if call_name_value == self.function_name:
            self.metrics["recursive_calls"] += 1
            self._recursive_calls_in_current_function += 1
            self.metrics["max_recursive_calls_per_function"] = max(
                self.metrics["max_recursive_calls_per_function"],
                self._recursive_calls_in_current_function,
            )
            self.evidence.append(f"Detected direct recursive call to `{call_name_value}`.")
            self.line_findings.append(
                finding(
                    getattr(node, "lineno", 0),
                    "Recursive call",
                    "medium",
                    "Recursion",
                    "Direct recursion affects stack usage and may repeat work without memoization.",
                    "Explain the base case and consider memoization or iterative dynamic programming when states repeat.",
                )
            )

        if call_name_value in {"sorted", "sort"}:
            self.metrics["sort_calls"] += 1
            self.evidence.append(f"Detected sorting call `{call_name_value}`.")
            if self.loop_depth:
                self.metrics["sort_in_loop"] += 1
                self.line_findings.append(
                    finding(
                        getattr(node, "lineno", 0),
                        "Sorting inside loop",
                        "high",
                        "Redundant work",
                        "Sorting while looping can create repeated O(n log n) work.",
                        "Sort once before the loop, use a heap, or maintain an ordered structure incrementally.",
                    )
                )
            else:
                self.line_findings.append(
                    finding(
                        getattr(node, "lineno", 0),
                        "Sorting cost",
                        "medium",
                        "Algorithmic complexity",
                        "Sorting usually contributes O(n log n) time.",
                        "Be ready to justify why ordering is needed.",
                    )
                )

        if call_name_value in EXPENSIVE_CALLS and self._call_signatures[signature] > 1:
            self.metrics["repeated_expensive_calls"] += 1
            self.line_findings.append(
                finding(
                    getattr(node, "lineno", 0),
                    "Repeated expensive call",
                    "medium",
                    "Redundant work",
                    f"`{signature}` appears more than once.",
                    "Cache the result or combine traversals when readability remains acceptable.",
                )
            )

        if call_name_value in {"list", "tuple", "set", "dict"}:
            self.metrics["temp_objects"] += 1
            if call_name_value == "list":
                self.metrics["list_builds"] += 1
            elif call_name_value == "set":
                self.metrics["set_builds"] += 1
            elif call_name_value == "dict":
                self.metrics["dict_builds"] += 1

        if call_name_value in {"append", "extend"} and self.loop_depth:
            self.metrics["list_builds"] += 1
            self.metrics["non_vectorized_loops"] += 1

        if call_name_value in DATA_PROCESSING_CALLS and self.loop_depth:
            self.metrics["non_vectorized_loops"] += 1

        if call_name_value in {"heappush", "heappop", "heapify", "nlargest", "nsmallest"}:
            self.metrics["heap_calls"] += 1

        if call_name_value in {"lru_cache", "cache"}:
            self.metrics["memoization_markers"] += 1

        if call_name_value == "<dynamic>":
            self.metrics["unknown_calls"] += 1

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Slice):
            self.metrics["slicing_count"] += 1
            self.evidence.append(f"Detected slicing expression `{_safe_unparse(node)}`.")
            if self.loop_depth:
                self.metrics["slicing_in_loop"] += 1
                self.line_findings.append(
                    finding(
                        getattr(node, "lineno", 0),
                        "Repeated slicing",
                        "high",
                        "Memory overhead",
                        "Slicing inside a loop copies data repeatedly.",
                        "Use index boundaries, iterators, or views instead of copied slices.",
                    )
                )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operator, comparator in zip(node.ops, node.comparators):
            if isinstance(operator, (ast.In, ast.NotIn)):
                self.metrics["membership_tests"] += 1
                target = _safe_unparse(comparator)
                inferred_type = self._container_type(comparator)
                self.evidence.append(f"Detected membership test against `{target}`.")
                if inferred_type:
                    self.metrics[f"{inferred_type}_membership_tests"] += 1
                    self.metrics["known_container_membership"] += 1
                    self.evidence.append(f"Membership target `{target}` appears to be a {inferred_type}.")
                else:
                    self.metrics["unknown_container_membership"] += 1
                if inferred_type in {"list", "tuple"}:
                    self.metrics["list_membership_tests"] += 1
                    if self.loop_depth:
                        self.metrics["list_membership_in_loop"] += 1
                        self.line_findings.append(
                            finding(
                                getattr(node, "lineno", 0),
                                "Linear membership in loop",
                                "high",
                                "Data structure choice",
                                f"`in {target}` is likely a linear scan inside a loop.",
                                "Convert the lookup collection to a set or dict before the loop.",
                            )
                        )
                elif inferred_type in {"set", "dict"}:
                    self.metrics["hash_membership_tests"] += 1
                elif self.loop_depth and isinstance(comparator, ast.Name):
                    self.metrics["list_membership_in_loop"] += 1
                    self.line_findings.append(
                        finding(
                            getattr(node, "lineno", 0),
                            "Unknown membership cost in loop",
                            "medium",
                            "Data structure choice",
                            f"The cost of `in {target}` depends on its runtime type.",
                            "Prefer a set/dict for repeated lookups, or document why the collection is small.",
                        )
                    )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        inferred = infer_container_type(node.value, self.known_types)
        if inferred:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.known_types[target.id] = inferred
                    self.metrics[f"{inferred}_assignments"] += 1
        self._detect_assignment_markers(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value and isinstance(node.target, ast.Name):
            inferred = infer_container_type(node.value, self.known_types)
            if inferred:
                self.known_types[node.target.id] = inferred
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target_name = _safe_unparse(node.target).lower()
        if any(token in target_name for token in ("sum", "total", "prefix", "running")):
            self.metrics["prefix_sum_markers"] += 1
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        if isinstance(getattr(node, "ctx", None), ast.Load):
            self.metrics["temp_objects"] += 1
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        self.metrics["set_builds"] += 1
        self.metrics["temp_objects"] += 1
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self.metrics["dict_builds"] += 1
        self.metrics["temp_objects"] += 1
        self.generic_visit(node)

    def _container_type(self, node: ast.AST) -> str:
        inferred = infer_container_type(node, self.known_types)
        if inferred:
            return inferred
        if isinstance(node, ast.Name):
            return self.known_types.get(node.id, "")
        return ""

    def _record_loop_bound(self, iter_node: ast.AST, line: int) -> None:
        if isinstance(iter_node, ast.Call) and self._call_name(iter_node.func) == "range":
            self.metrics["range_loops"] += 1
            args = iter_node.args
            if args and isinstance(args[0], ast.Call) and self._call_name(args[0].func) == "len":
                self.metrics["range_len_loops"] += 1
                self.evidence.append(f"Loop bound at line {line} is `range(len(...))`, a clear O(n) signal.")
            else:
                self.evidence.append(f"Loop bound at line {line} uses `range(...)`.")

    def _detect_while_pattern(self, node: ast.While) -> None:
        text = _safe_unparse(node).lower()
        if all(token in text for token in ("mid", "left", "right")) or all(token in text for token in ("mid", "lo", "hi")):
            self.metrics["binary_search_markers"] += 1
            self.line_findings.append(
                finding(
                    getattr(node, "lineno", 0),
                    "Binary-search-style loop",
                    "low",
                    "Algorithm pattern",
                    "This while-loop has left/right/mid boundary markers.",
                    "State the sorted-input precondition and O(log n) search behavior.",
                )
            )
        if any(token in text for token in ("left", "right", "window", "start", "end")):
            self.metrics["sliding_window_markers"] += 1

    def _detect_assignment_markers(self, node: ast.Assign) -> None:
        text = _safe_unparse(node).lower()
        targets = " ".join(_safe_unparse(target).lower() for target in node.targets)
        if any(token in targets or token in text for token in ("memo", "cache", "dp")):
            self.metrics["memoization_markers"] += 1
        if any(token in targets or token in text for token in ("prefix", "cumsum", "running_sum")):
            self.metrics["prefix_sum_markers"] += 1
        if any(token in targets for token in ("left", "right", "lo", "hi", "start", "end")):
            self.metrics["index_pointer_updates"] += 1
        if any(token in text for token in ("visited", "neighbors", "queue", "stack", "deque")):
            self.metrics["graph_markers"] += 1

    @staticmethod
    def _call_name(func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return "<dynamic>"

    def finalize(self) -> Dict[str, Any]:
        if self.metrics.get("loop_count", 0) == 0 and self.metrics.get("comprehension_count", 0) == 0:
            self.evidence.append("No explicit loops or comprehensions were detected in this scope.")
        return dict(self.metrics)


def _analyze_scope(name: str, lineno: int, body: List[ast.stmt]) -> FunctionAnalysis:
    wrapper = ast.Module(body=body, type_ignores=[])
    collector = FeatureCollector(function_name=name, skip_nested_functions=True)
    collector.visit(wrapper)
    metrics = collector.finalize()
    if _detect_recursive_binary_search(name, body):
        metrics["recursive_binary_search_markers"] = metrics.get("recursive_binary_search_markers", 0) + 1
        metrics["binary_search_markers"] = metrics.get("binary_search_markers", 0) + 1
        collector.evidence.append("Detected recursive binary search with midpoint and shrinking bounds.")
        collector.line_findings.append(
            finding(
                lineno,
                "Recursive binary search",
                "low",
                "Algorithm pattern",
                "Recursive calls shrink the left/right search bounds around a midpoint.",
                "State the sorted-input precondition, O(log n) time, and O(log n) recursion stack.",
            )
        )
    time_complexity = estimate_time_complexity(metrics)
    space_complexity = estimate_space_complexity(metrics)
    patterns = detect_anti_patterns(metrics)
    caveats = list(dict.fromkeys(collector.caveats))
    confidence = estimate_confidence(metrics, caveats)
    algorithm_patterns = detect_algorithm_patterns(
        metrics=metrics,
        evidence=collector.evidence,
        calls=collector.calls,
        code_text="\n".join(_safe_unparse(stmt) for stmt in body),
    )
    if algorithm_patterns:
        metrics["algorithm_pattern_count"] = len(algorithm_patterns)
    return FunctionAnalysis(
        name=name,
        lineno=lineno,
        estimated_time=time_complexity,
        estimated_space=space_complexity,
        confidence=confidence,
        evidence=list(dict.fromkeys(collector.evidence)),
        caveats=caveats,
        anti_patterns=patterns,
        line_findings=collector.line_findings,
        algorithm_patterns=algorithm_patterns,
        metrics=metrics,
    )


def _module_level_body(tree: ast.Module) -> List[ast.stmt]:
    return [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _top_level_function_names(tree: ast.Module) -> List[str]:
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(child.name)
                    names.append(f"{node.name}.{child.name}")
    return names


def analyze_code(code: str) -> StaticAnalysisResult:
    code = code or ""
    if not code.strip():
        return StaticAnalysisResult(
            valid=False,
            raw_code=code,
            parse_error="No Python code was provided.",
            caveats=["Paste a function or script to analyze static complexity."],
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticAnalysisResult(
            valid=False,
            raw_code=code,
            parse_error=f"SyntaxError at line {exc.lineno}: {exc.msg}",
            caveats=["Static analysis requires syntactically valid Python."],
        )

    defined_functions = _top_level_function_names(tree)
    catalog = FunctionCatalog()
    catalog.visit(tree)
    call_graph = catalog.as_call_graph(defined_functions)

    functions: List[FunctionAnalysis] = []
    module_body = _module_level_body(tree)
    if module_body:
        functions.append(_analyze_scope("<module>", 1, module_body))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_analyze_scope(node.name, node.lineno, node.body))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(_analyze_scope(f"{node.name}.{child.name}", child.lineno, child.body))

    if not functions:
        functions.append(_analyze_scope("<module>", 1, tree.body))

    metrics = _merge_metrics(function.metrics for function in functions)
    estimated_time = max_complexity(function.estimated_time for function in functions)
    estimated_space = max_complexity(function.estimated_space for function in functions)
    caveats: List[str] = []
    evidence: List[str] = []
    anti_patterns = []
    line_findings: List[LineFinding] = []
    algorithm_patterns = []
    for function in functions:
        caveats.extend(function.caveats)
        evidence.extend(f"{function.name}: {item}" for item in function.evidence)
        anti_patterns.extend(function.anti_patterns)
        line_findings.extend(function.line_findings)
        algorithm_patterns.extend(function.algorithm_patterns)

    if is_word_break_ii_pattern(code):
        output_caveat = (
            "Word Break II / sentence generation detected. Runtime and memory are output-sensitive "
            "because the function may return exponentially many sentences."
        )
        caveats.append(output_caveat)
        evidence.append("Detected Word Break II sentence-generation DFS with memoization and appended results.")
        metrics["word_break_ii_output_sensitive"] = 1
        metrics["output_sensitive"] = 1
        estimated_time = "O(k*n)"
        estimated_space = "O(k*n)"
        algorithm_patterns.append(
            AlgorithmPattern(
                name="Word Break II / sentence generation",
                confidence=0.92,
                evidence=["wordBreak entrypoint", "DFS", "memoization", "sentence result append"],
                interview_note=(
                    "This problem is output-sensitive. No algorithm can avoid producing all returned sentences; "
                    "optimize pruning, memoization, and avoid unnecessary work without claiming impossible lower complexity."
                ),
            )
        )

    caveats = list(dict.fromkeys(caveats))
    if len(functions) > 1:
        caveats.append("Overall complexity is the maximum estimate across analyzed scopes.")
    caveats.append("Static complexity is estimated from AST features and may differ from real runtime.")

    confidence = estimate_confidence(metrics, caveats)
    confidence_parts = confidence_breakdown(metrics, caveats, confidence)
    anti_patterns = sorted(anti_patterns, key=lambda item: item.weight, reverse=True)
    optimization_targets = build_optimization_targets(anti_patterns)

    return StaticAnalysisResult(
        valid=True,
        raw_code=code,
        estimated_time=estimated_time,
        estimated_space=estimated_space,
        confidence=confidence,
        functions=functions,
        evidence=list(dict.fromkeys(evidence)),
        caveats=caveats,
        anti_patterns=anti_patterns,
        optimization_targets=optimization_targets,
        line_findings=sorted(line_findings, key=lambda item: (item.line, item.severity)),
        algorithm_patterns=sorted(algorithm_patterns, key=lambda item: item.confidence, reverse=True),
        call_graph=call_graph,
        confidence_breakdown=confidence_parts,
        metrics=metrics,
    )
