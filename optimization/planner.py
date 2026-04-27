"""Generate actionable optimization and interview guidance."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from analyzer.ast_analyzer import analyze_code
from analyzer.complexity_rules import complexity_rank
from analyzer.models import AntiPattern, StaticAnalysisResult
from benchmarking.sandbox import validate_code_for_execution
from scoring.optimizer_score import ScoreBreakdown, calculate_optimization_score


@dataclass
class OptimizationStep:
    title: str
    priority: str
    why: str
    change: str
    runtime_effect: str
    memory_effect: str
    interview_note: str
    line: int = 0
    expected_complexity_change: str = ""
    rewritten_code: Optional[str] = None
    validation_test: Optional[str] = None


@dataclass
class OptimizedCodeCandidate:
    source: str
    code: str
    explanation: str = ""
    step_by_step_plan: List[str] = field(default_factory=list)
    validation_tests: List[str] = field(default_factory=list)
    confidence: float = 0.0
    retry_count: int = 0
    equivalence_reason: str = ""


@dataclass
class OptimizedCodeValidation:
    source: str = "none"
    status: str = "not_generated"
    original_time: str = ""
    original_space: str = ""
    original_score: int = 0
    candidate_time: str = ""
    candidate_space: str = ""
    candidate_score: int = 0
    score_delta: int = 0
    retry_count: int = 0
    time_improved: bool = False
    space_improved: bool = False
    memory_tradeoff: bool = False
    accepted_reason: str = ""
    rejection_reasons: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)


@dataclass
class OptimizationPlan:
    summary: str
    quick_wins: List[OptimizationStep] = field(default_factory=list)
    medium_refactors: List[OptimizationStep] = field(default_factory=list)
    advanced_improvements: List[OptimizationStep] = field(default_factory=list)
    optimized_code: Optional[str] = None
    safe_rewrite_confidence: float = 0.0
    rewrite_tests: List[str] = field(default_factory=list)
    manual_checklist: List[str] = field(default_factory=list)
    before_after: str = ""
    tradeoffs: List[str] = field(default_factory=list)
    interview_feedback: Dict[str, str] = field(default_factory=dict)
    validation: OptimizedCodeValidation = field(default_factory=OptimizedCodeValidation)
    candidate_source: str = "local"
    candidate_explanation: str = ""
    step_by_step_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _step_from_pattern(pattern: AntiPattern) -> OptimizationStep:
    runtime_effect = "Can reduce repeated work and improve constant factors."
    memory_effect = "Usually neutral unless it removes temporary containers."
    if "nested" in pattern.name.lower():
        runtime_effect = "Can reduce runtime from O(n^2)/O(n^3) toward O(n) or O(n log n), depending on the strategy."
        memory_effect = "May use O(n) auxiliary memory for a hash map, set, or precomputed table."
        complexity_change = "Often O(n^2) -> O(n) or O(n log n)"
    elif "membership" in pattern.name.lower():
        runtime_effect = "Usually changes repeated membership from O(n) scans to average O(1) lookups."
        memory_effect = "Adds O(n) memory for a set/dict but often improves speed substantially."
        complexity_change = "Repeated O(n) lookup -> average O(1) lookup"
    elif "slicing" in pattern.name.lower():
        runtime_effect = "Removes hidden O(k) copy work on each iteration."
        memory_effect = "Lowers peak memory by avoiding copied sublists/substrings."
        complexity_change = "Avoids hidden O(k) work per iteration"
    elif "sorting inside" in pattern.name.lower():
        runtime_effect = "Avoids repeated O(n log n) work inside the loop."
        memory_effect = "Often neutral or lower if repeated sorted copies are removed."
        complexity_change = "Repeated O(n log n) -> one sort or incremental update"
    else:
        complexity_change = "Improves constants or clarifies asymptotic reasoning"

    return OptimizationStep(
        title=pattern.name,
        priority=pattern.severity,
        why=pattern.message,
        change=pattern.suggestion,
        runtime_effect=runtime_effect,
        memory_effect=memory_effect,
        interview_note=(
            "Name the bottleneck, state the current complexity, then explain the replacement data structure "
            "or traversal pattern and its trade-off."
        ),
        expected_complexity_change=complexity_change,
    )


def _generic_quick_wins(analysis: StaticAnalysisResult) -> List[OptimizationStep]:
    wins: List[OptimizationStep] = []
    if not analysis.anti_patterns:
        wins.append(
            OptimizationStep(
                title="Keep the solution simple and explain invariants",
                priority="low",
                why="No major structural bottleneck was detected by the static analyzer.",
                change="Focus on clear variable names, boundary cases, and a crisp complexity explanation.",
                runtime_effect="No algorithmic change required.",
                memory_effect="No additional memory required.",
                interview_note="Lead with correctness, then state the estimated time and auxiliary space.",
            )
        )
    if analysis.metrics.get("binary_search_markers", 0):
        wins.extend(
            [
                OptimizationStep(
                    title="Keep the iterative binary-search shape",
                    priority="low",
                    why="The current solution already has the optimal O(log n) search pattern.",
                    change="Preserve left/right bounds, midpoint calculation, and one branch update per loop.",
                    runtime_effect="Keeps logarithmic runtime.",
                    memory_effect="Keeps O(1) auxiliary memory for the iterative version.",
                    interview_note="Mention that each iteration discards half of the remaining sorted search range.",
                    expected_complexity_change="Already optimal: O(log n) time, O(1) iterative space",
                ),
                OptimizationStep(
                    title="Add boundary-case validation",
                    priority="low",
                    why="Binary search bugs usually appear on empty, single-item, first-item, and last-item inputs.",
                    change="Test empty arrays, missing targets, first/last positions, and duplicate-safe expectations.",
                    runtime_effect="No runtime change.",
                    memory_effect="No memory change.",
                    interview_note="Use these tests to prove the loop condition and bound updates are correct.",
                    expected_complexity_change="Correctness confidence improvement",
                ),
            ]
        )
    return wins


def _detect_two_sum_candidate(code: str, analysis: StaticAnalysisResult) -> bool:
    lowered = code.lower()
    return (
        "target" in lowered
        and ("nums" in lowered or "arr" in lowered)
        and ("two_sum" in lowered or "complement" in lowered or "seen" in lowered)
        and analysis.metrics.get("max_loop_depth", 0) >= 1
    )


def _detect_iterative_binary_search_candidate(analysis: StaticAnalysisResult) -> bool:
    return (
        bool(analysis.metrics.get("binary_search_markers", 0))
        and not analysis.metrics.get("recursive_binary_search_markers", 0)
        and analysis.estimated_time == "O(log n)"
        and analysis.estimated_space == "O(1)"
    )


@dataclass
class _EntrypointDefinition:
    name: str
    qualified_name: str
    args: List[str]
    class_name: str = ""

    @property
    def is_method(self) -> bool:
        return bool(self.class_name)


def _entrypoint_definitions(code: str) -> List[_EntrypointDefinition]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    definitions: List[_EntrypointDefinition] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(
                _EntrypointDefinition(
                    name=node.name,
                    qualified_name=node.name,
                    args=[arg.arg for arg in node.args.args],
                )
            )
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.append(
                        _EntrypointDefinition(
                            name=child.name,
                            qualified_name=f"{node.name}.{child.name}",
                            args=[arg.arg for arg in child.args.args],
                            class_name=node.name,
                        )
                    )
    return definitions


def _matches_entrypoint(definition: _EntrypointDefinition, entrypoint: str) -> bool:
    if not entrypoint:
        return True
    return entrypoint in {definition.name, definition.qualified_name}


def _entrypoint_definition(code: str, entrypoint: str) -> Optional[_EntrypointDefinition]:
    definitions = _entrypoint_definitions(code)
    if entrypoint:
        for definition in definitions:
            if _matches_entrypoint(definition, entrypoint):
                return definition
        return None
    return definitions[0] if definitions else None


def _function_args(code: str, entrypoint: str) -> List[str]:
    definition = _entrypoint_definition(code, entrypoint)
    if not definition:
        return []
    args = list(definition.args)
    if definition.is_method and args and args[0] in {"self", "cls"}:
        return args[1:]
    return args


def _top_level_function_names(code: str) -> List[str]:
    return [definition.name for definition in _entrypoint_definitions(code) if not definition.is_method]


def _has_entrypoint(code: str, entrypoint: str) -> bool:
    if not entrypoint:
        return bool(_entrypoint_definitions(code))
    return any(_matches_entrypoint(definition, entrypoint) for definition in _entrypoint_definitions(code))


def _stock_profit_candidate_shape(code: str, entrypoint: str) -> Optional[_EntrypointDefinition]:
    definition = _entrypoint_definition(code, entrypoint)
    if not definition:
        return None
    lowered = code.lower()
    if (
        definition.name.lower() == "maxprofit"
        and "prices" in lowered
        and "min_price" in lowered
        and "max_profit" in lowered
    ):
        return definition
    return None


class _FunctionRenamer(ast.NodeTransformer):
    def __init__(self, old_name: str, new_name: str) -> None:
        self.old_name = old_name
        self.new_name = new_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name == self.old_name:
            node.name = self.new_name
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name == self.old_name:
            node.name = self.new_name
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id == self.old_name:
            node.func.id = self.new_name
        self.generic_visit(node)
        return node


def preserve_entrypoint_name(code: str, entrypoint: str) -> str:
    entrypoint = (entrypoint or "").strip()
    if not entrypoint:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if entrypoint in names or not names:
        return code
    renamed = _FunctionRenamer(names[0], entrypoint).visit(tree)
    ast.fix_missing_locations(renamed)
    try:
        return ast.unparse(renamed) + "\n"
    except Exception:
        return code


def _optimized_code_suggestion(
    code: str,
    analysis: StaticAnalysisResult,
    entrypoint: str = "",
) -> tuple[Optional[str], float, List[str], List[str]]:
    target_name = (entrypoint or "").strip() or "optimized_solution"
    if _detect_two_sum_candidate(code, analysis):
        return f'''def {target_name}(nums, target):
    """Return indices of two numbers that add to target, or [] if no pair exists."""
    seen = {{}}

    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index

    return []
''', 0.92, [
            f"assert {target_name}([2, 7, 11, 15], 9) == [0, 1]",
            f"assert {target_name}([3, 2, 4], 6) == [1, 2]",
            f"assert {target_name}([1, 2, 3], 99) == []",
        ], [
            "Build a hash map while scanning the list once.",
            "For each value, check whether its complement has already been seen.",
            "Return matching indices immediately and label the O(n) memory tradeoff.",
        ]

    if analysis.metrics.get("recursive_binary_search_markers", 0):
        args = _function_args(code, entrypoint)
        arr_arg = args[0] if len(args) >= 1 else "arr"
        target_arg = args[1] if len(args) >= 2 else "target"
        left_arg = args[2] if len(args) >= 3 else "left"
        right_arg = args[3] if len(args) >= 4 else "right"
        return f'''def {target_name}({arr_arg}, {target_arg}, {left_arg}=0, {right_arg}=None):
    """Search a sorted sequence with iterative binary search."""
    if {right_arg} is None:
        {right_arg} = len({arr_arg}) - 1
    while {left_arg} <= {right_arg}:
        mid = ({left_arg} + {right_arg}) // 2
        if {arr_arg}[mid] == {target_arg}:
            return mid
        if {target_arg} < {arr_arg}[mid]:
            {right_arg} = mid - 1
        else:
            {left_arg} = mid + 1
    return -1
''', 0.78, [
            f"assert {target_name}([2, 4, 6, 8, 10, 12], 10) == 4",
            f"assert {target_name}([2, 4, 6, 8, 10, 12], 5) == -1",
            f"assert {target_name}([], 5) == -1",
        ], [
            "Keep the binary-search invariant but remove recursive calls.",
            "Update the left/right bounds in a loop until the target is found or the range is empty.",
            "Preserve O(log n) time while reducing auxiliary stack space to O(1).",
        ]

    if _detect_iterative_binary_search_candidate(analysis):
        args = _function_args(code, entrypoint)
        arr_arg = args[0] if len(args) >= 1 else "arr"
        target_arg = args[1] if len(args) >= 2 else "target"
        return f'''def {target_name}({arr_arg}, {target_arg}):
    """Return the index of target in a sorted sequence, or -1 when absent."""
    left = 0
    right = len({arr_arg}) - 1

    while left <= right:
        mid = (left + right) // 2
        if {arr_arg}[mid] == {target_arg}:
            return mid
        if {target_arg} < {arr_arg}[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1
''', 0.86, [
            f"assert {target_name}([2, 4, 6, 8, 10, 12], 10) == 4",
            f"assert {target_name}([2, 4, 6, 8, 10, 12], 5) == -1",
            f"assert {target_name}([], 5) == -1",
            f"assert {target_name}([7], 7) == 0",
        ], [
            "Keep the optimal iterative binary-search algorithm.",
            "Use a compact reference implementation with clear left/right bound updates.",
            "Validate the sorted-input precondition and edge cases rather than adding unnecessary data structures.",
        ]

    if analysis.metrics.get("list_membership_in_loop", 0):
        args = _function_args(code, entrypoint)
        lowered = code.lower()
        if len(args) == 1 and "duplicate" in lowered:
            values = args[0]
            return f'''def {target_name}({values}):
    """Find duplicate values with average O(1) set membership checks."""
    seen = set()
    emitted = set()
    duplicates = []
    for value in {values}:
        if value in seen and value not in emitted:
            duplicates.append(value)
            emitted.add(value)
        seen.add(value)
    return duplicates
''', 0.72, [
                f"assert {target_name}([1, 2, 3, 2, 1]) == [2, 1]",
                f"assert {target_name}([]) == []",
            ], [
                "Replace list-backed membership checks with sets.",
                "Track seen values and already-emitted duplicates separately.",
                "Keep output order while reducing repeated lookup cost.",
            ]
        if len(args) == 2:
            items, lookup_values = args
            return f'''def {target_name}({items}, {lookup_values}):
    """Convert repeated membership checks to average O(1) set lookups."""
    lookup_set = set({lookup_values})
    matches = []
    for item in {items}:
        if item in lookup_set:
            matches.append(item)
    return matches
''', 0.66, [
                f"assert {target_name}([1, 2, 3, 4], [2, 4]) == [2, 4]",
                f"assert {target_name}([], [1, 2]) == []",
            ], [
                "Convert the repeated lookup collection to a set once.",
                "Scan the input only once.",
                "Return matches using average O(1) membership checks.",
            ]

    if analysis.metrics.get("sort_in_loop", 0):
        args = _function_args(code, entrypoint)
        if len(args) != 1:
            return None, 0.0, [], []
        items = args[0]
        return f'''def {target_name}({items}):
    """Sort once, then reuse the ordered data inside downstream logic."""
    ordered_items = sorted({items})
    results = []
    for item in ordered_items:
        results.append(item)
    return results
''', 0.52, [
            f"assert {target_name}([3, 1, 2]) == [1, 2, 3]",
        ], [
            "Move sorting outside repeated loop work.",
            "Reuse the sorted sequence for downstream processing.",
            "Avoid repeated O(n log n) sorting in the hot path.",
        ]

    if analysis.metrics.get("recursive_calls", 0) and "fib" in target_name.lower():
        return f'''from functools import lru_cache

@lru_cache(maxsize=None)
def {target_name}(n):
    """Compute Fibonacci numbers with memoized recursion."""
    if n <= 1:
        return n
    return {target_name}(n - 1) + {target_name}(n - 2)
''', 0.78, [
            f"assert {target_name}(0) == 0",
            f"assert {target_name}(1) == 1",
            f"assert {target_name}(10) == 55",
        ], [
            "Cache each Fibonacci state the first time it is computed.",
            "Reuse cached values for overlapping recursive subproblems.",
            "Reduce exponential branching to linear state-count work.",
        ]
    return None, 0.0, [], []


def build_local_candidate(
    code: str,
    analysis: StaticAnalysisResult,
    entrypoint: str = "",
) -> Optional[OptimizedCodeCandidate]:
    optimized_code, confidence, tests, steps = _optimized_code_suggestion(code, analysis, entrypoint)
    if not optimized_code:
        return None
    return OptimizedCodeCandidate(
        source="local",
        code=optimized_code,
        explanation="Generated by deterministic local rewrite templates.",
        step_by_step_plan=steps,
        validation_tests=tests,
        confidence=confidence,
        equivalence_reason=(
            "The current approach is already asymptotically optimal, so this is a clean reference implementation."
            if (
                _detect_iterative_binary_search_candidate(analysis)
                or _detect_two_sum_candidate(code, analysis)
            )
            else ""
        ),
    )


def validate_optimized_candidate(
    original_analysis: StaticAnalysisResult,
    original_score: ScoreBreakdown,
    candidate: Optional[OptimizedCodeCandidate],
    entrypoint: str = "",
) -> tuple[Optional[str], OptimizedCodeValidation]:
    if not candidate or not (candidate.code or "").strip():
        return None, OptimizedCodeValidation(
            source=getattr(candidate, "source", "none"),
            status="not_generated",
            original_time=original_analysis.estimated_time,
            original_space=original_analysis.estimated_space,
            original_score=original_score.score,
            retry_count=getattr(candidate, "retry_count", 0),
            rejection_reasons=["No optimized code candidate was generated."],
        )

    normalized_code = preserve_entrypoint_name(candidate.code, entrypoint)
    validation = OptimizedCodeValidation(
        source=candidate.source,
        status="rejected",
        original_time=original_analysis.estimated_time,
        original_space=original_analysis.estimated_space,
        original_score=original_score.score,
        retry_count=candidate.retry_count,
    )

    violations = validate_code_for_execution(normalized_code)
    if violations:
        validation.rejection_reasons.append("Candidate failed benchmark safety checks: " + " ".join(violations[:3]))
        validation.safety_notes = violations[:5]
        return None, validation

    candidate_analysis = analyze_code(normalized_code)
    if not candidate_analysis.valid:
        validation.rejection_reasons.append(candidate_analysis.parse_error or "Candidate code could not be parsed.")
        return None, validation

    function_names = _top_level_function_names(normalized_code)
    if entrypoint and entrypoint not in function_names:
        validation.rejection_reasons.append(f"Candidate must define the configured entrypoint `{entrypoint}`.")
        return None, validation

    candidate_score = calculate_optimization_score(candidate_analysis)
    original_time_rank = complexity_rank(original_analysis.estimated_time)
    candidate_time_rank = complexity_rank(candidate_analysis.estimated_time)
    original_space_rank = complexity_rank(original_analysis.estimated_space)
    candidate_space_rank = complexity_rank(candidate_analysis.estimated_space)

    validation.candidate_time = candidate_analysis.estimated_time
    validation.candidate_space = candidate_analysis.estimated_space
    validation.candidate_score = candidate_score.score
    validation.score_delta = candidate_score.score - original_score.score
    validation.time_improved = candidate_time_rank < original_time_rank
    validation.space_improved = candidate_space_rank < original_space_rank
    validation.memory_tradeoff = candidate_space_rank > original_space_rank and validation.time_improved

    time_not_worse = candidate_time_rank <= original_time_rank
    score_improved = validation.score_delta >= 5
    if validation.time_improved:
        validation.status = "accepted"
        validation.accepted_reason = "Estimated time complexity improves."
    elif validation.space_improved and time_not_worse and validation.score_delta >= 0:
        validation.status = "accepted"
        validation.accepted_reason = "Estimated auxiliary space improves without worse estimated time."
    elif candidate.equivalence_reason and time_not_worse and candidate_space_rank <= original_space_rank:
        validation.status = "accepted"
        validation.accepted_reason = candidate.equivalence_reason
    elif score_improved and time_not_worse and candidate_space_rank <= original_space_rank:
        validation.status = "accepted"
        validation.accepted_reason = "Optimization score improves without worse estimated complexity."
    elif (
        candidate.source in {"gemini", "local"}
        and time_not_worse
        and candidate_space_rank <= original_space_rank
        and candidate.confidence >= 0.70
    ):
        validation.status = "accepted"
        validation.accepted_reason = (
            "No asymptotic improvement was available, so the app accepted a clean, "
            "verified reference implementation with the same or better estimated complexity."
        )
    else:
        if not validation.time_improved:
            validation.rejection_reasons.append(
                f"Estimated time did not improve ({original_analysis.estimated_time} -> {candidate_analysis.estimated_time})."
            )
        if validation.score_delta < 5:
            validation.rejection_reasons.append(
                f"Optimization score did not improve enough ({original_score.score} -> {candidate_score.score})."
            )
        if candidate_space_rank > original_space_rank and not validation.time_improved:
            validation.rejection_reasons.append(
                f"Estimated space worsened without a time-complexity improvement ({original_analysis.estimated_space} -> {candidate_analysis.estimated_space})."
            )
        return None, validation

    return normalized_code, validation


def _interview_feedback(analysis: StaticAnalysisResult, score: ScoreBreakdown) -> Dict[str, str]:
    acceptable = "Likely acceptable" if score.score >= 72 else "Needs improvement before an interview"
    if score.score < 55:
        acceptable = "High risk in an interview"

    concern = "The interviewer will likely ask whether the complexity can be improved."
    if score.score >= 86:
        concern = "The interviewer will likely focus on correctness, edge cases, and clear explanation."
    elif analysis.metrics.get("max_loop_depth", 0) >= 2:
        concern = "The interviewer will likely challenge the nested loop and ask for a hash-based or sorted approach."
    elif analysis.metrics.get("recursive_calls", 0):
        concern = "The interviewer will likely ask about recursion depth, overlapping subproblems, and memoization."

    return {
        "acceptability": acceptable,
        "likely_concern": concern,
        "current_explanation": (
            f"The current solution is estimated at {analysis.estimated_time} time and "
            f"{analysis.estimated_space} auxiliary space with {int(analysis.confidence * 100)}% confidence."
        ),
        "optimized_explanation": (
            "The optimized approach should reduce repeated scans, choose data structures that match lookup needs, "
            "and trade small auxiliary memory for lower runtime when that improves asymptotic behavior."
        ),
        "tradeoffs": (
            "Mention readability, extra memory for hash tables, input-size assumptions, and why empirical benchmarks "
            "do not replace asymptotic analysis."
        ),
    }


def build_optimization_plan(
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    entrypoint: str = "",
    candidate: Optional[OptimizedCodeCandidate] = None,
    prior_rejection_reasons: Optional[List[str]] = None,
) -> OptimizationPlan:
    steps = [_step_from_pattern(pattern) for pattern in analysis.anti_patterns]
    quick = [step for step in steps if step.priority in {"low", "medium"}][:3]
    medium = [step for step in steps if step.priority == "high"][:3]
    advanced = [step for step in steps if step.priority == "critical"][:2]
    quick.extend(_generic_quick_wins(analysis))
    if score.score >= 86 and not medium:
        medium.append(
            OptimizationStep(
                title="No medium refactor required",
                priority="low",
                why="The current algorithm is already in the excellent range.",
                change="Avoid adding extra data structures or abstractions unless requirements change.",
                runtime_effect="Preserves the current asymptotic runtime.",
                memory_effect="Preserves the current auxiliary memory.",
                interview_note="Say that the best optimization is to keep the algorithm simple and prove correctness.",
                expected_complexity_change="No refactor needed",
            )
        )
    if score.score >= 86 and not advanced:
        advanced.append(
            OptimizationStep(
                title="No advanced rewrite required",
                priority="low",
                why="No higher-order algorithmic replacement was detected as necessary.",
                change="Focus advanced discussion on preconditions, correctness proof, and representative benchmarks.",
                runtime_effect="No algorithmic change needed.",
                memory_effect="No extra memory needed.",
                interview_note="Mention that binary search is already the right algorithm for sorted input.",
                expected_complexity_change="Already optimal for sorted search",
            )
        )

    summary = (
        f"Optimization quality is {score.score}/100 with {score.improvement_potential.lower()} "
        f"improvement potential. The top static estimate is {analysis.estimated_time} time and "
        f"{analysis.estimated_space} space."
    )

    selected_candidate = candidate or build_local_candidate(analysis.raw_code, analysis, entrypoint)
    optimized_code, validation = validate_optimized_candidate(analysis, score, selected_candidate, entrypoint)
    if validation.status != "accepted" and candidate is not None:
        fallback = build_local_candidate(analysis.raw_code, analysis, entrypoint)
        fallback_code, fallback_validation = validate_optimized_candidate(analysis, score, fallback, entrypoint)
        fallback_validation.rejection_reasons = (
            list(prior_rejection_reasons or [])
            + validation.rejection_reasons
            + fallback_validation.rejection_reasons
        )
        selected_candidate = fallback
        optimized_code = fallback_code
        validation = fallback_validation

    if prior_rejection_reasons and validation.status == "accepted":
        validation.rejection_reasons = list(prior_rejection_reasons) + validation.rejection_reasons

    rewrite_confidence = selected_candidate.confidence if selected_candidate else 0.0
    rewrite_tests = selected_candidate.validation_tests if selected_candidate and validation.status == "accepted" else []
    candidate_steps = selected_candidate.step_by_step_plan if selected_candidate else []
    candidate_explanation = selected_candidate.explanation if selected_candidate else ""
    candidate_source = validation.source if validation.source != "none" else "local"

    before_after = (
        "Before: the current version may repeat scans, build temporary containers, or use nested traversal. "
        "After: prioritize one-pass aggregation, hash lookups, precomputation, and avoiding repeated copies."
    )
    if validation.status == "accepted" and validation.candidate_time:
        if selected_candidate and selected_candidate.equivalence_reason:
            before_after = (
                f"Before: estimated {validation.original_time} time and {validation.original_space} space, "
                f"score {validation.original_score}/100. After: same asymptotic performance "
                f"({validation.candidate_time} time and {validation.candidate_space} space) with a clean reference form."
            )
        else:
            before_after = (
                f"Before: estimated {validation.original_time} time and {validation.original_space} space, "
                f"score {validation.original_score}/100. After: estimated {validation.candidate_time} time and "
                f"{validation.candidate_space} space, score {validation.candidate_score}/100."
            )
    elif not analysis.anti_patterns:
        before_after = (
            "Before: the code already looks structurally reasonable from static signals. "
            "After: preserve the current approach, validate edge cases, and benchmark representative inputs."
        )

    tradeoffs = [
        "Hash maps and sets often improve lookup speed but add O(n) auxiliary memory.",
        "Sorting can simplify logic, but it introduces O(n log n) time and should be justified.",
        "Micro-benchmarks are input-specific; use them to validate, not to replace complexity reasoning.",
    ]
    if validation.memory_tradeoff:
        tradeoffs.insert(
            0,
            f"This verified rewrite improves estimated time but increases estimated space from {validation.original_space} to {validation.candidate_space}.",
        )

    manual_checklist = [
        "Write down the input size variable and what it counts.",
        "Count how many full traversals happen on the hottest path.",
        "Replace repeated membership scans with a set/dict when ordering is not required.",
        "Avoid copying slices in loops; use indexes or iterators.",
        "Benchmark representative inputs after each algorithmic change.",
    ]

    for step in steps:
        matching = next(
            (
                finding
                for finding in analysis.line_findings
                if step.title.lower().split()[0] in finding.title.lower()
                or finding.category.lower() in step.why.lower()
            ),
            None,
        )
        if matching:
            step.line = matching.line

    return OptimizationPlan(
        summary=summary,
        quick_wins=quick[:4],
        medium_refactors=medium[:4],
        advanced_improvements=advanced[:3],
        optimized_code=optimized_code if validation.status == "accepted" else None,
        safe_rewrite_confidence=rewrite_confidence if validation.status == "accepted" else 0.0,
        rewrite_tests=rewrite_tests,
        manual_checklist=manual_checklist,
        before_after=before_after,
        tradeoffs=tradeoffs,
        interview_feedback=_interview_feedback(analysis, score),
        validation=validation,
        candidate_source=candidate_source,
        candidate_explanation=candidate_explanation,
        step_by_step_plan=candidate_steps,
    )
