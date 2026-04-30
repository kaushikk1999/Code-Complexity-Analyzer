"""Generate actionable optimization and interview guidance."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from analyzer.advanced_patterns import is_word_break_ii_pattern
from analyzer.ast_analyzer import analyze_code
from analyzer.complexity_rules import complexity_rank
from analyzer.models import AntiPattern, StaticAnalysisResult
from benchmarking.metrics import BenchmarkResult
from benchmarking.runner import run_benchmark
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
    level: str = ""
    title: str = ""
    expected_time: str = ""
    expected_space: str = ""
    step_by_step_plan: List[str] = field(default_factory=list)
    validation_tests: List[str] = field(default_factory=list)
    confidence: float = 0.0
    retry_count: int = 0
    equivalence_reason: str = ""


@dataclass
class CandidateBenchmarkComparison:
    original_success: bool = False
    candidate_success: bool = False
    original_avg_ms: float = 0.0
    candidate_avg_ms: float = 0.0
    original_peak_memory_kb: float = 0.0
    candidate_peak_memory_kb: float = 0.0
    runtime_ratio: float = 0.0
    memory_ratio: float = 0.0
    accepted: bool = False
    reason: str = ""


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
    benchmark_comparison: Optional[CandidateBenchmarkComparison] = None


@dataclass
class TieredOptimizationCandidate:
    tier: str
    title: str
    code: str
    expected_time: str
    expected_space: str
    explanation: str
    benchmark_comparison: Optional[CandidateBenchmarkComparison] = None
    accepted: bool = False
    rejection_reason: str = ""


@dataclass
class VerifiedOptimizationCandidate:
    level: str
    title: str
    code: str
    explanation: str
    expected_time: str
    expected_space: str
    actual_time: str = ""
    actual_space: str = ""
    benchmark_avg_ms: float = 0.0
    benchmark_peak_kb: float = 0.0
    original_avg_ms: float = 0.0
    original_peak_kb: float = 0.0
    status: str = "not_generated"
    acceptance_reason: str = ""
    rejection_reasons: List[str] = field(default_factory=list)
    validation_tests: List[str] = field(default_factory=list)
    source: str = "local"
    score: int = 0
    original_score: int = 0


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
    tiered_candidates: List[TieredOptimizationCandidate] = field(default_factory=list)
    verified_candidates: List[VerifiedOptimizationCandidate] = field(default_factory=list)
    best_candidate: Optional[VerifiedOptimizationCandidate] = None
    generation_notes: List[str] = field(default_factory=list)

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


def _detect_word_break_candidate(code: str, entrypoint: str) -> bool:
    lowered = f"{entrypoint}\n{code}".lower()
    return (
        "wordbreak" in lowered
        and "worddict" in lowered
        and "dp" in lowered
        and "s[" in lowered
    )


def _detect_word_break_ii_candidate(code: str, entrypoint: str = "") -> bool:
    return is_word_break_ii_pattern(f"{entrypoint}\n{code}")


LEVEL_LABELS = {
    "quick_win": "Quick Win",
    "medium_refactor": "Medium Refactor",
    "advanced": "Advanced Improvement",
}


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
    if "." in entrypoint:
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


def _word_break_code(entrypoint: str, variant: str = "quick") -> str:
    target_name = (entrypoint or "").strip() or "wordBreak"
    if "." in target_name:
        class_name, method_name = target_name.split(".", 1)
        signature = f"class {class_name}:\n    def {method_name}(self, s: str, wordDict: List[str]) -> bool:"
        indent = "        "
    else:
        signature = f"def {target_name}(s: str, wordDict: List[str]) -> bool:"
        indent = "    "

    if variant == "medium":
        body = f'''{indent}if not wordDict:
{indent}    return s == ""

{indent}words = set(wordDict)
{indent}lengths = sorted({{len(word) for word in words}})
{indent}n = len(s)
{indent}dp = [False] * (n + 1)
{indent}dp[0] = True

{indent}for end in range(1, n + 1):
{indent}    for length in lengths:
{indent}        if length > end:
{indent}            break
{indent}        begin = end - length
{indent}        if dp[begin] and s[begin:end] in words:
{indent}            dp[end] = True
{indent}            break

{indent}return dp[n]
'''
    elif variant == "advanced":
        body = f'''{indent}if not wordDict:
{indent}    return s == ""

{indent}trie = {{}}
{indent}terminal = "#"
{indent}for word in wordDict:
{indent}    node = trie
{indent}    for char in word:
{indent}        node = node.setdefault(char, {{}})
{indent}    node[terminal] = True

{indent}n = len(s)
{indent}reachable = [False] * (n + 1)
{indent}reachable[0] = True

{indent}for start in range(n):
{indent}    if not reachable[start]:
{indent}        continue
{indent}    node = trie
{indent}    for end in range(start, n):
{indent}        char = s[end]
{indent}        if char not in node:
{indent}            break
{indent}        node = node[char]
{indent}        if terminal in node:
{indent}            reachable[end + 1] = True
{indent}            if end + 1 == n:
{indent}                return True

{indent}return reachable[n]
'''
    else:
        body = f'''{indent}if not wordDict:
{indent}    return s == ""

{indent}words = set(wordDict)
{indent}max_len = max(map(len, words))
{indent}n = len(s)

{indent}dp = [False] * (n + 1)
{indent}dp[0] = True

{indent}for end in range(1, n + 1):
{indent}    start = max(0, end - max_len)
{indent}    for begin in range(start, end):
{indent}        if dp[begin] and s[begin:end] in words:
{indent}            dp[end] = True
{indent}            break

{indent}return dp[n]
'''

    return f"from typing import List\n\n{signature}\n{body}"


def _word_break_ii_signature(entrypoint: str) -> tuple[str, str]:
    target_name = (entrypoint or "").strip() or "wordBreak"
    if "." in target_name:
        class_name, method_name = target_name.split(".", 1)
        return f"class {class_name}:\n    def {method_name}(self, s: str, wordDict: List[str]) -> List[str]:", "        "
    return f"def {target_name}(s: str, wordDict: List[str]) -> List[str]:", "    "


def _word_break_ii_code(entrypoint: str, variant: str) -> str:
    signature, indent = _word_break_ii_signature(entrypoint)
    if variant == "medium_refactor":
        body = f'''{indent}words = set(wordDict)
{indent}if not words:
{indent}    return []
{indent}max_len = max(map(len, words))
{indent}memo = {{}}

{indent}def dfs(start: int) -> List[str]:
{indent}    if start == len(s):
{indent}        return [""]
{indent}    if start in memo:
{indent}        return memo[start]
{indent}    sentences = []
{indent}    limit = min(len(s), start + max_len)
{indent}    for end in range(start + 1, limit + 1):
{indent}        word = s[start:end]
{indent}        if word in words:
{indent}            for suffix in dfs(end):
{indent}                sentences.append(word if not suffix else word + " " + suffix)
{indent}    memo[start] = sentences
{indent}    return sentences

{indent}return dfs(0)
'''
    elif variant == "advanced":
        body = f'''{indent}words = set(wordDict)
{indent}if not words:
{indent}    return []
{indent}n = len(s)
{indent}lengths = sorted({{len(word) for word in words}})
{indent}can_break = [False] * (n + 1)
{indent}can_break[n] = True
{indent}for i in range(n - 1, -1, -1):
{indent}    for length in lengths:
{indent}        end = i + length
{indent}        if end > n:
{indent}            break
{indent}        if can_break[end] and s[i:end] in words:
{indent}            can_break[i] = True
{indent}            break
{indent}memo = {{}}

{indent}def dfs(start: int) -> List[str]:
{indent}    if start == n:
{indent}        return [""]
{indent}    if start in memo:
{indent}        return memo[start]
{indent}    if not can_break[start]:
{indent}        return []
{indent}    sentences = []
{indent}    for length in lengths:
{indent}        end = start + length
{indent}        if end > n:
{indent}            break
{indent}        word = s[start:end]
{indent}        if word in words and can_break[end]:
{indent}            for suffix in dfs(end):
{indent}                sentences.append(word if not suffix else word + " " + suffix)
{indent}    memo[start] = sentences
{indent}    return sentences

{indent}return dfs(0)
'''
    else:
        body = f'''{indent}words = set(wordDict)
{indent}memo = {{}}

{indent}def dfs(start: int) -> List[str]:
{indent}    if start == len(s):
{indent}        return [""]
{indent}    if start in memo:
{indent}        return memo[start]
{indent}    sentences = []
{indent}    for end in range(start + 1, len(s) + 1):
{indent}        word = s[start:end]
{indent}        if word not in words:
{indent}            continue
{indent}        for suffix in dfs(end):
{indent}            sentences.append(word if not suffix else word + " " + suffix)
{indent}    memo[start] = sentences
{indent}    return sentences

{indent}return dfs(0)
'''
    return f"from typing import List\n\n{signature}\n{body}"


def _word_break_ii_local_candidate(entrypoint: str, level: str) -> OptimizedCodeCandidate:
    titles = {
        "quick_win": "Cleaner DFS memoization reference",
        "medium_refactor": "Max-word-length DFS pruning",
        "advanced": "Reachability pruning before sentence construction",
    }
    explanations = {
        "quick_win": (
            "Keeps the original DFS + memoization approach, simplifies suffix assembly, and avoids claiming "
            "better asymptotic complexity for an output-sensitive problem."
        ),
        "medium_refactor": (
            "Adds max-word-length pruning so the DFS avoids substring checks that cannot match any dictionary word."
        ),
        "advanced": (
            "Precomputes reachable positions before constructing sentences. This can reduce useless DFS paths, "
            "but it adds setup memory and must win on the benchmark before being accepted."
        ),
    }
    return OptimizedCodeCandidate(
        source="local",
        code=_word_break_ii_code(entrypoint, level),
        level=level,
        title=titles[level],
        explanation=explanations[level],
        expected_time="O(k*n)",
        expected_space="O(k*n)",
        validation_tests=[
            'assert Solution().wordBreak("leetcode", ["leet", "code"]) == ["leet code"]',
            (
                'assert sorted(Solution().wordBreak("catsanddog", '
                '["cat", "cats", "and", "sand", "dog"])) == ["cat sand dog", "cats and dog"]'
            ),
        ],
        confidence=0.84,
        equivalence_reason=(
            "Word Break II is output-sensitive: k returned sentences can be exponential in n, so this is "
            "a same-complexity reference unless benchmarks prove lower overhead."
        ),
    )


def _optimized_code_suggestion(
    code: str,
    analysis: StaticAnalysisResult,
    entrypoint: str = "",
) -> tuple[Optional[str], float, List[str], List[str]]:
    target_name = (entrypoint or "").strip() or "optimized_solution"
    lowered = code.lower()

    if ("singlenumber" in lowered and "xor" in lowered) or (
        "singlenumber" in lowered and "^" in code
    ):
        return '''class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        Return the element that appears once when every other element appears twice.

        Time: O(n)
        Space: O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        return result
''', 0.95, [
            "assert Solution().singleNumber([2, 2, 1]) == 1",
            "assert Solution().singleNumber([4, 1, 2, 1, 2]) == 4",
            "assert Solution().singleNumber([1]) == 1",
            "assert Solution().singleNumber([-1, -1, -2]) == -2",
        ], [
            "Use XOR because a ^ a cancels to 0 and 0 ^ x returns x.",
            "Scan the array once.",
            "Keep O(1) auxiliary space without a hash map.",
            "Handle single-element and negative-number cases naturally.",
        ]

    if _detect_word_break_candidate(code, entrypoint):
        optimized_code = _word_break_code(entrypoint, "quick")
        return optimized_code, 0.90, [
            'assert Solution().wordBreak("leetcode", ["leet", "code"]) is True',
            'assert Solution().wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False',
            'assert Solution().wordBreak("applepenapple", ["apple", "pen"]) is True',
            'assert Solution().wordBreak("", ["a", "abc"]) is True',
            'assert Solution().wordBreak("a", []) is False',
        ], [
            "Keep the DP algorithm because it is already a standard solution.",
            "Add an empty dictionary guard.",
            "Use max word length pruning to reduce unnecessary substring checks.",
            "Do not add sorted unique lengths unless benchmarks prove it helps.",
        ]

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
                or _detect_word_break_candidate(code, entrypoint)
            )
            else ""
        ),
    )


def build_local_candidate_for_level(
    code: str,
    analysis: StaticAnalysisResult,
    entrypoint: str,
    level: str,
) -> Optional[OptimizedCodeCandidate]:
    if _detect_word_break_ii_candidate(code, entrypoint):
        return _word_break_ii_local_candidate(entrypoint, level)

    if _detect_word_break_candidate(code, entrypoint):
        variant = "quick"
        if level == "medium_refactor":
            variant = "medium"
        elif level == "advanced":
            variant = "advanced"
        return OptimizedCodeCandidate(
            source="local",
            code=_word_break_code(entrypoint, variant),
            level=level,
            title=f"{LEVEL_LABELS[level]} Word Break reference",
            explanation="Local Word Break reference candidate, benchmarked before acceptance.",
            expected_time="O(n^2)",
            expected_space="O(n)",
            confidence=0.78,
        )

    single_candidate = build_local_candidate(code, analysis, entrypoint)
    if not single_candidate:
        return None

    lowered = code.lower()
    preferred_level = "medium_refactor"
    if "singlenumber" in lowered:
        preferred_level = "quick_win"
    elif analysis.metrics.get("recursive_binary_search_markers", 0):
        preferred_level = "medium_refactor"
    elif _detect_iterative_binary_search_candidate(analysis):
        preferred_level = "quick_win"

    if level != preferred_level:
        return None
    single_candidate.level = level
    single_candidate.title = single_candidate.title or f"{LEVEL_LABELS[level]} local candidate"
    single_candidate.expected_time = single_candidate.expected_time or "Estimated from candidate"
    single_candidate.expected_space = single_candidate.expected_space or "Estimated from candidate"
    return single_candidate


def benchmark_candidate_against_original(
    original_code: str,
    candidate_code: str,
    entrypoint: str,
    benchmark_input: str,
    repeat_count: int = 3,
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    comparison = compare_candidate_benchmark(
        original_code=original_code,
        candidate_code=candidate_code,
        entrypoint=entrypoint,
        benchmark_input=benchmark_input,
        repeat_count=repeat_count,
        timeout_seconds=timeout_seconds,
    )
    return comparison.accepted, comparison.reason


def compare_candidate_benchmark(
    original_code: str,
    candidate_code: str,
    entrypoint: str,
    benchmark_input: str,
    repeat_count: int = 5,
    timeout_seconds: float = 5.0,
) -> CandidateBenchmarkComparison:
    if not benchmark_input.strip():
        return CandidateBenchmarkComparison(
            accepted=False,
            reason="No benchmark input was provided for original-vs-candidate validation.",
        )

    original = run_benchmark(
        code=original_code,
        entrypoint=entrypoint,
        input_text=benchmark_input,
        repeat_count=repeat_count,
        warmup_count=1,
        timeout_seconds=timeout_seconds,
        allow_top_level=False,
    )

    candidate = run_benchmark(
        code=candidate_code,
        entrypoint=entrypoint,
        input_text=benchmark_input,
        repeat_count=repeat_count,
        warmup_count=1,
        timeout_seconds=timeout_seconds,
        allow_top_level=False,
    )

    comparison = CandidateBenchmarkComparison(
        original_success=original.success,
        candidate_success=candidate.success,
    )

    if not original.success:
        comparison.reason = f"Original benchmark failed: {original.error}"
        return comparison

    if not candidate.success:
        comparison.reason = f"Candidate benchmark failed: {candidate.error}"
        return comparison

    original_avg = original.summary.avg_ms
    candidate_avg = candidate.summary.avg_ms
    original_mem = original.summary.max_peak_memory_kb
    candidate_mem = candidate.summary.max_peak_memory_kb

    comparison.original_avg_ms = original_avg
    comparison.candidate_avg_ms = candidate_avg
    comparison.original_peak_memory_kb = original_mem
    comparison.candidate_peak_memory_kb = candidate_mem
    comparison.runtime_ratio = candidate_avg / max(original_avg, 1e-9)
    comparison.memory_ratio = candidate_mem / max(original_mem, 1e-9)
    return comparison


def benchmark_candidate(
    original_code: str,
    candidate_code: str,
    entrypoint: str,
    benchmark_input: str,
    repeat_count: int = 5,
    timeout_seconds: float = 5.0,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    original = run_benchmark(
        code=original_code,
        entrypoint=entrypoint,
        input_text=benchmark_input,
        repeat_count=repeat_count,
        warmup_count=1,
        timeout_seconds=timeout_seconds,
        allow_top_level=False,
    )
    candidate = run_benchmark(
        code=candidate_code,
        entrypoint=entrypoint,
        input_text=benchmark_input,
        repeat_count=repeat_count,
        warmup_count=1,
        timeout_seconds=timeout_seconds,
        allow_top_level=False,
    )
    return original, candidate


def candidate_is_better(
    original_score: int,
    candidate_score: int,
    original_avg_ms: float,
    candidate_avg_ms: float,
    original_peak_kb: float,
    candidate_peak_kb: float,
    original_time_rank: int,
    candidate_time_rank: int,
    original_space_rank: int,
    candidate_space_rank: int,
) -> tuple[bool, str]:
    time_complexity_better = candidate_time_rank < original_time_rank
    time_complexity_not_worse = candidate_time_rank <= original_time_rank
    space_complexity_better = candidate_space_rank < original_space_rank
    runtime_better = candidate_avg_ms > 0 and original_avg_ms > 0 and candidate_avg_ms <= original_avg_ms * 0.95
    memory_better = candidate_peak_kb > 0 and original_peak_kb > 0 and candidate_peak_kb <= original_peak_kb * 0.95
    score_not_worse = candidate_score >= original_score
    runtime_not_worse = candidate_avg_ms <= original_avg_ms * 1.05
    memory_not_worse = candidate_peak_kb <= original_peak_kb * 1.10
    tiny_runtime_delta = abs(candidate_avg_ms - original_avg_ms) <= 0.01

    if time_complexity_better:
        return True, "Accepted because estimated time complexity improved."
    if space_complexity_better and (
        runtime_not_worse or (time_complexity_not_worse and score_not_worse and tiny_runtime_delta)
    ):
        return True, "Accepted because estimated space complexity improved without worse runtime."
    if runtime_better and memory_not_worse and score_not_worse:
        return True, "Accepted because benchmark runtime improved without worse memory or score."
    if memory_better and runtime_not_worse and score_not_worse:
        return True, "Accepted because peak memory improved without worse runtime or score."
    if score_not_worse and runtime_not_worse and memory_not_worse:
        return True, "Accepted as a same-or-better clean reference implementation."
    return False, "Rejected because it did not improve complexity, runtime, memory, or score."


def _same_normalized_code(left: str, right: str) -> bool:
    try:
        return ast.dump(ast.parse(left), include_attributes=False) == ast.dump(ast.parse(right), include_attributes=False)
    except SyntaxError:
        return left.strip() == right.strip()


def _empty_verified_candidate(level: str, reason: str) -> VerifiedOptimizationCandidate:
    return VerifiedOptimizationCandidate(
        level=level,
        title=f"{LEVEL_LABELS[level]} Candidate",
        code="",
        explanation="No candidate was generated for this level.",
        expected_time="Not generated",
        expected_space="Not generated",
        status="not_generated",
        rejection_reasons=[reason],
    )


def _verify_candidate_for_level(
    original_code: str,
    original_analysis: StaticAnalysisResult,
    original_score: ScoreBreakdown,
    candidate: Optional[OptimizedCodeCandidate],
    entrypoint: str,
    benchmark_input: str,
    level: str,
    repeat_count: int = 5,
    timeout_seconds: float = 5.0,
) -> VerifiedOptimizationCandidate:
    if not candidate or not (candidate.code or "").strip():
        return _empty_verified_candidate(level, f"No {LEVEL_LABELS[level].lower()} candidate was generated.")

    normalized_code = preserve_entrypoint_name(candidate.code, entrypoint)
    verified = VerifiedOptimizationCandidate(
        level=level,
        title=candidate.title or f"{LEVEL_LABELS[level]} Candidate",
        code=normalized_code,
        explanation=candidate.explanation,
        expected_time=candidate.expected_time or "Estimated from candidate",
        expected_space=candidate.expected_space or "Estimated from candidate",
        status="rejected",
        validation_tests=list(candidate.validation_tests),
        source=candidate.source,
        original_score=original_score.score,
    )

    violations = validate_code_for_execution(normalized_code)
    if violations:
        verified.rejection_reasons.append("Candidate rejected because safety checks failed: " + " ".join(violations[:3]))
        return verified

    candidate_analysis = analyze_code(normalized_code)
    if not candidate_analysis.valid:
        verified.rejection_reasons.append(candidate_analysis.parse_error or "Candidate code could not be parsed.")
        return verified

    if entrypoint and not _has_entrypoint(normalized_code, entrypoint):
        verified.rejection_reasons.append(f"Candidate must define the configured entrypoint `{entrypoint}`.")
        return verified

    if _same_normalized_code(original_code, normalized_code):
        verified.actual_time = candidate_analysis.estimated_time
        verified.actual_space = candidate_analysis.estimated_space
        verified.rejection_reasons.append("Candidate rejected because it is identical to the original code.")
        return verified

    verified.actual_time = candidate_analysis.estimated_time
    verified.actual_space = candidate_analysis.estimated_space

    if not benchmark_input.strip():
        verified.rejection_reasons.append("Candidate rejected because no benchmark input was provided for same-input validation.")
        return verified

    original_benchmark, candidate_benchmark = benchmark_candidate(
        original_code=original_code,
        candidate_code=normalized_code,
        entrypoint=entrypoint,
        benchmark_input=benchmark_input,
        repeat_count=repeat_count,
        timeout_seconds=timeout_seconds,
    )
    if not original_benchmark.success:
        verified.rejection_reasons.append(f"Candidate rejected because original benchmark failed: {original_benchmark.error}")
        return verified
    if not candidate_benchmark.success:
        verified.rejection_reasons.append("Candidate rejected because benchmark failed on the same input.")
        if candidate_benchmark.error:
            verified.rejection_reasons.append(candidate_benchmark.error)
        return verified

    verified.original_avg_ms = original_benchmark.summary.avg_ms
    verified.original_peak_kb = original_benchmark.summary.max_peak_memory_kb
    verified.benchmark_avg_ms = candidate_benchmark.summary.avg_ms
    verified.benchmark_peak_kb = candidate_benchmark.summary.max_peak_memory_kb

    candidate_score = calculate_optimization_score(candidate_analysis, candidate_benchmark)
    verified.score = candidate_score.score
    original_time_rank = complexity_rank(original_analysis.estimated_time)
    candidate_time_rank = complexity_rank(candidate_analysis.estimated_time)
    original_space_rank = complexity_rank(original_analysis.estimated_space)
    candidate_space_rank = complexity_rank(candidate_analysis.estimated_space)

    accepted, reason = candidate_is_better(
        original_score=original_score.score,
        candidate_score=candidate_score.score,
        original_avg_ms=verified.original_avg_ms,
        candidate_avg_ms=verified.benchmark_avg_ms,
        original_peak_kb=verified.original_peak_kb,
        candidate_peak_kb=verified.benchmark_peak_kb,
        original_time_rank=original_time_rank,
        candidate_time_rank=candidate_time_rank,
        original_space_rank=original_space_rank,
        candidate_space_rank=candidate_space_rank,
    )
    if not accepted:
        verified.rejection_reasons.append(reason)
        if candidate_score.score < original_score.score and candidate_time_rank >= original_time_rank:
            verified.rejection_reasons.append(
                f"Candidate score decreased ({original_score.score} -> {candidate_score.score}) without better time complexity."
            )
        return verified

    if (
        _detect_iterative_binary_search_candidate(original_analysis)
        and candidate_time_rank == original_time_rank
        and candidate_space_rank == original_space_rank
    ):
        reason = "Accepted as a same-or-better clean reference implementation."

    verified.acceptance_reason = reason
    verified.status = "same_complexity" if "same-or-better clean reference" in reason else "accepted"
    return verified


CandidateProvider = Callable[[str, List[str]], Tuple[Optional[OptimizedCodeCandidate], Optional[str]]]


def _select_best_candidate(candidates: List[VerifiedOptimizationCandidate]) -> Optional[VerifiedOptimizationCandidate]:
    accepted_candidates = [candidate for candidate in candidates if candidate.status in {"accepted", "same_complexity"}]
    accepted_candidates.sort(
        key=lambda candidate: (
            candidate.benchmark_avg_ms,
            candidate.benchmark_peak_kb,
            complexity_rank(candidate.actual_time),
            complexity_rank(candidate.actual_space),
        )
    )
    return accepted_candidates[0] if accepted_candidates else None


def generate_verified_optimization_candidates(
    original_code: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    entrypoint: str,
    benchmark_input: str,
    api_key: str = "",
    plan: Optional[OptimizationPlan] = None,
    candidate_provider: Optional[CandidateProvider] = None,
    seed_candidates: Optional[Dict[str, OptimizedCodeCandidate]] = None,
    repeat_count: int = 5,
    timeout_seconds: float = 5.0,
) -> OptimizationPlan:
    working_plan = plan or build_optimization_plan(
        analysis,
        score,
        entrypoint=entrypoint,
        benchmark_input="",
        generate_candidates=False,
    )
    seed_candidates = seed_candidates or {}
    verified_candidates: List[VerifiedOptimizationCandidate] = []
    generation_notes: List[str] = []
    rejection_context: List[str] = []
    used_provider = bool(api_key and candidate_provider)

    for level in ("quick_win", "medium_refactor", "advanced"):
        generated = seed_candidates.get(level)
        if generated is None and used_provider:
            generated, error = candidate_provider(level, rejection_context)
            if error:
                generation_notes.append(error)
                generated = None

        if generated is None:
            generated = build_local_candidate_for_level(original_code, analysis, entrypoint, level)
            if used_provider and generated is not None:
                generation_notes.append("Gemini candidate generation failed. Falling back to local verified optimizer.")

        verified = _verify_candidate_for_level(
            original_code=original_code,
            original_analysis=analysis,
            original_score=score,
            candidate=generated,
            entrypoint=entrypoint,
            benchmark_input=benchmark_input,
            level=level,
            repeat_count=repeat_count,
            timeout_seconds=timeout_seconds,
        )
        verified_candidates.append(verified)
        if verified.rejection_reasons:
            rejection_context = verified.rejection_reasons

    best_candidate = _select_best_candidate(verified_candidates)
    working_plan.verified_candidates = verified_candidates
    working_plan.best_candidate = best_candidate
    working_plan.validation = _validation_from_best_candidate(analysis, score, best_candidate, verified_candidates)
    working_plan.generation_notes = list(dict.fromkeys(generation_notes))
    working_plan.optimized_code = best_candidate.code if best_candidate else None
    working_plan.safe_rewrite_confidence = 0.84 if best_candidate else 0.0
    working_plan.rewrite_tests = best_candidate.validation_tests if best_candidate else []
    working_plan.candidate_source = best_candidate.source if best_candidate else "none"
    working_plan.candidate_explanation = best_candidate.explanation if best_candidate else ""
    working_plan.step_by_step_plan = [
        candidate.acceptance_reason
        for candidate in verified_candidates
        if candidate.status in {"accepted", "same_complexity"} and candidate.acceptance_reason
    ]

    if best_candidate:
        status_text = "same-complexity reference" if best_candidate.status == "same_complexity" else "verified improvement"
        working_plan.before_after = (
            f"Before: estimated {analysis.estimated_time} time and {analysis.estimated_space} space, "
            f"score {score.score}/100. After: {status_text} with estimated {best_candidate.actual_time} time "
            f"and {best_candidate.actual_space} space, score {best_candidate.score}/100."
        )
    else:
        working_plan.before_after = (
            "No verified better rewrite was found for this benchmark input. The original remains the best "
            "validated version. Try larger or more difficult inputs to evaluate pruning benefits."
        )

    return working_plan


def _validation_from_best_candidate(
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    best_candidate: Optional[VerifiedOptimizationCandidate],
    all_candidates: List[VerifiedOptimizationCandidate],
) -> OptimizedCodeValidation:
    if best_candidate:
        original_time_rank = complexity_rank(analysis.estimated_time)
        candidate_time_rank = complexity_rank(best_candidate.actual_time)
        original_space_rank = complexity_rank(analysis.estimated_space)
        candidate_space_rank = complexity_rank(best_candidate.actual_space)
        comparison = CandidateBenchmarkComparison(
            original_success=True,
            candidate_success=True,
            original_avg_ms=best_candidate.original_avg_ms,
            candidate_avg_ms=best_candidate.benchmark_avg_ms,
            original_peak_memory_kb=best_candidate.original_peak_kb,
            candidate_peak_memory_kb=best_candidate.benchmark_peak_kb,
            runtime_ratio=best_candidate.benchmark_avg_ms / max(best_candidate.original_avg_ms, 1e-9),
            memory_ratio=best_candidate.benchmark_peak_kb / max(best_candidate.original_peak_kb, 1e-9),
            accepted=True,
            reason=best_candidate.acceptance_reason,
        )
        return OptimizedCodeValidation(
            source=best_candidate.source,
            status=best_candidate.status,
            original_time=analysis.estimated_time,
            original_space=analysis.estimated_space,
            original_score=score.score,
            candidate_time=best_candidate.actual_time,
            candidate_space=best_candidate.actual_space,
            candidate_score=best_candidate.score,
            score_delta=best_candidate.score - score.score,
            time_improved=candidate_time_rank < original_time_rank,
            space_improved=candidate_space_rank < original_space_rank,
            memory_tradeoff=candidate_space_rank > original_space_rank and candidate_time_rank < original_time_rank,
            accepted_reason=best_candidate.acceptance_reason,
            benchmark_comparison=comparison,
        )

    reasons: List[str] = []
    for candidate in all_candidates:
        reasons.extend(candidate.rejection_reasons)
    return OptimizedCodeValidation(
        source="none",
        status="rejected",
        original_time=analysis.estimated_time,
        original_space=analysis.estimated_space,
        original_score=score.score,
        rejection_reasons=list(dict.fromkeys(reasons)) or ["No verified candidate was accepted."],
    )


def _candidate_is_better_or_safe(
    validation: OptimizedCodeValidation,
    original_score: ScoreBreakdown,
    candidate_score: ScoreBreakdown,
    original_time_rank: int,
    candidate_time_rank: int,
    original_space_rank: int,
    candidate_space_rank: int,
    benchmark_comparison: Optional[CandidateBenchmarkComparison],
) -> tuple[bool, str]:
    time_improved = candidate_time_rank < original_time_rank
    space_improved = candidate_space_rank < original_space_rank
    time_not_worse = candidate_time_rank <= original_time_rank
    space_not_worse = candidate_space_rank <= original_space_rank
    score_not_worse = candidate_score.score >= original_score.score
    score_improved = candidate_score.score >= original_score.score + 3

    if time_improved and space_not_worse:
        return True, "Accepted because estimated time complexity improves and estimated space does not worsen."
    if space_improved and time_not_worse:
        return True, "Accepted because estimated space complexity improves and estimated time does not worsen."
    if not time_not_worse:
        return False, "Rejected because estimated time complexity worsened."
    if not space_not_worse:
        return False, "Rejected because estimated space complexity worsened."
    if not score_not_worse:
        return False, (
            f"Rejected because optimization score decreased "
            f"({original_score.score} -> {candidate_score.score})."
        )

    if benchmark_comparison:
        if not benchmark_comparison.original_success or not benchmark_comparison.candidate_success:
            return False, benchmark_comparison.reason
        runtime_ok = benchmark_comparison.candidate_avg_ms <= benchmark_comparison.original_avg_ms * 1.05
        memory_ok = benchmark_comparison.candidate_peak_memory_kb <= benchmark_comparison.original_peak_memory_kb * 1.10
        benchmark_comparison.accepted = runtime_ok and memory_ok
        if not runtime_ok:
            return False, (
                "Rejected because candidate benchmark runtime is worse. "
                f"Original avg: {benchmark_comparison.original_avg_ms:.4f} ms; "
                f"candidate avg: {benchmark_comparison.candidate_avg_ms:.4f} ms."
            )
        if not memory_ok:
            return False, (
                "Rejected because candidate benchmark memory is worse. "
                f"Original peak: {benchmark_comparison.original_peak_memory_kb:.2f} KB; "
                f"candidate peak: {benchmark_comparison.candidate_peak_memory_kb:.2f} KB."
            )

    if score_improved:
        return True, "Accepted because score improves without worse estimated time or space."
    if score_not_worse and benchmark_comparison and benchmark_comparison.accepted:
        return True, "Accepted because candidate is not worse by score, complexity, runtime, or memory."
    return False, "Rejected because candidate does not improve score, time complexity, space complexity, runtime, or memory."


def validate_optimized_candidate(
    original_analysis: StaticAnalysisResult,
    original_score: ScoreBreakdown,
    candidate: Optional[OptimizedCodeCandidate],
    entrypoint: str = "",
    benchmark_input: str = "",
) -> tuple[Optional[str], OptimizedCodeValidation]:
    level = candidate.level if candidate and candidate.level in LEVEL_LABELS else "medium_refactor"
    verified = _verify_candidate_for_level(
        original_code=original_analysis.raw_code,
        original_analysis=original_analysis,
        original_score=original_score,
        candidate=candidate,
        entrypoint=entrypoint,
        benchmark_input=benchmark_input,
        level=level,
    )
    validation = _validation_from_best_candidate(
        original_analysis,
        original_score,
        verified if verified.status in {"accepted", "same_complexity"} else None,
        [verified],
    )
    validation.source = getattr(candidate, "source", validation.source) if candidate else validation.source
    validation.retry_count = getattr(candidate, "retry_count", 0)
    validation.candidate_time = verified.actual_time
    validation.candidate_space = verified.actual_space
    validation.candidate_score = verified.score
    validation.score_delta = verified.score - original_score.score if verified.score else 0
    validation.rejection_reasons = list(verified.rejection_reasons)
    validation.accepted_reason = verified.acceptance_reason
    return (verified.code if verified.status in {"accepted", "same_complexity"} else None), validation


def _tier_placeholder(tier: str) -> TieredOptimizationCandidate:
    return TieredOptimizationCandidate(
        tier=tier,
        title=f"{tier}: no verified candidate found",
        code="",
        expected_time="Not verified",
        expected_space="Not verified",
        explanation=f"No verified candidate was found for {tier.lower()} under the current benchmark input.",
        accepted=False,
        rejection_reason=f"No verified candidate was found for {tier.lower()} under the current benchmark input.",
    )


def _evaluate_tiered_candidate(
    tier: str,
    title: str,
    candidate: OptimizedCodeCandidate,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    entrypoint: str,
    benchmark_input: str,
    explanation: str,
    expected_time: str,
    expected_space: str,
) -> TieredOptimizationCandidate:
    optimized_code, validation = validate_optimized_candidate(
        analysis,
        score,
        candidate,
        entrypoint=entrypoint,
        benchmark_input=benchmark_input,
    )
    accepted = validation.status == "accepted"
    rejection_reason = "; ".join(validation.rejection_reasons)
    return TieredOptimizationCandidate(
        tier=tier,
        title=title,
        code=optimized_code or candidate.code,
        expected_time=validation.candidate_time or expected_time,
        expected_space=validation.candidate_space or expected_space,
        explanation=explanation,
        benchmark_comparison=validation.benchmark_comparison,
        accepted=accepted,
        rejection_reason="" if accepted else rejection_reason,
    )


def _complete_tier_set(candidates: List[TieredOptimizationCandidate]) -> List[TieredOptimizationCandidate]:
    order = ["Quick Wins", "Medium Refactors", "Advanced Improvements"]
    by_tier = {candidate.tier: candidate for candidate in candidates}
    return [by_tier.get(tier, _tier_placeholder(tier)) for tier in order]


def _build_tiered_candidates(
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    entrypoint: str,
    benchmark_input: str,
) -> List[TieredOptimizationCandidate]:
    if _detect_word_break_candidate(analysis.raw_code, entrypoint):
        tier_specs = [
            (
                "Quick Wins",
                "Quick Wins: DP cleanup and empty dictionary guard",
                "quick",
                "O(n^2)",
                "O(n)",
                (
                    "Keeps the standard DP structure, adds the empty wordDict guard, "
                    "and uses clearer begin/end names. Accepted only when benchmarked as no worse."
                ),
            ),
            (
                "Medium Refactors",
                "Medium Refactor: unique word-length pruning",
                "medium",
                "O(n * k)",
                "O(n + k)",
                (
                    "Checks only distinct dictionary word lengths. This can reduce substring checks for some "
                    "dictionaries, but it is rejected when preprocessing overhead does not pay off."
                ),
            ),
            (
                "Advanced Improvements",
                "Advanced Improvement: trie-index DP",
                "advanced",
                "O(n * L)",
                "O(n + total dictionary characters)",
                (
                    "Uses a trie to walk matching prefixes from reachable indices. It uses more structure, "
                    "so it is accepted only if benchmark data proves a runtime or memory win."
                ),
            ),
        ]
        return [
            _evaluate_tiered_candidate(
                tier=tier,
                title=title,
                candidate=OptimizedCodeCandidate(
                    source="local",
                    code=_word_break_code(entrypoint, variant),
                    explanation=explanation,
                    confidence=0.82,
                ),
                analysis=analysis,
                score=score,
                entrypoint=entrypoint,
                benchmark_input=benchmark_input,
                explanation=explanation,
                expected_time=expected_time,
                expected_space=expected_space,
            )
            for tier, title, variant, expected_time, expected_space, explanation in tier_specs
        ]

    local_candidate = build_local_candidate(analysis.raw_code, analysis, entrypoint)
    candidates: List[TieredOptimizationCandidate] = []
    if local_candidate:
        candidates.append(
            _evaluate_tiered_candidate(
                tier="Medium Refactors",
                title="Medium Refactor: best local candidate",
                candidate=local_candidate,
                analysis=analysis,
                score=score,
                entrypoint=entrypoint,
                benchmark_input=benchmark_input,
                explanation=local_candidate.explanation or "Validated local optimization candidate.",
                expected_time="Estimated from candidate",
                expected_space="Estimated from candidate",
            )
        )
    return _complete_tier_set(candidates)


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
    benchmark_input: str = "",
    generate_candidates: Optional[bool] = None,
    candidate_provider: Optional[CandidateProvider] = None,
    api_key: str = "",
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
        binary_search_detected = bool(
            analysis.metrics.get("binary_search_markers", 0)
            or analysis.metrics.get("recursive_binary_search_markers", 0)
        )
        advanced.append(
            OptimizationStep(
                title="No advanced rewrite verified",
                priority="low",
                why="No advanced rewrite was verified for the current benchmark input.",
                change="Focus on correctness, edge cases, output-size limits, and representative benchmarks.",
                runtime_effect="No algorithmic change needed.",
                memory_effect="No extra memory needed.",
                interview_note=(
                    "Mention that binary search is already the right algorithm for sorted input."
                    if binary_search_detected
                    else "Avoid claiming a stronger algorithm unless the problem constraints and benchmarks support it."
                ),
                expected_complexity_change="No verified advanced rewrite",
            )
        )

    summary = (
        f"Optimization quality is {score.score}/100 with {score.improvement_potential.lower()} "
        f"improvement potential. The top static estimate is {analysis.estimated_time} time and "
        f"{analysis.estimated_space} space."
    )

    before_after = (
        "Before: the current version may repeat scans, build temporary containers, or use nested traversal. "
        "After: prioritize one-pass aggregation, hash lookups, precomputation, and avoiding repeated copies."
    )
    if analysis.metrics.get("word_break_ii_output_sensitive"):
        before_after = (
            "This problem is output-sensitive. No algorithm can avoid producing all returned sentences. "
            "The best optimization is pruning, memoization, and avoiding unnecessary work."
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
    if analysis.metrics.get("word_break_ii_output_sensitive"):
        tradeoffs.insert(
            0,
            (
                "Word Break II is output-sensitive: k returned sentences can be exponential in n, so pruning "
                "can reduce wasted search but cannot remove the cost of producing the output."
            ),
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

    validation = OptimizedCodeValidation(
        source="none",
        status="not_generated",
        original_time=analysis.estimated_time,
        original_space=analysis.estimated_space,
        original_score=score.score,
        rejection_reasons=list(prior_rejection_reasons or []),
    )
    plan = OptimizationPlan(
        summary=summary,
        quick_wins=quick[:4],
        medium_refactors=medium[:4],
        advanced_improvements=advanced[:3],
        optimized_code=None,
        safe_rewrite_confidence=0.0,
        rewrite_tests=[],
        manual_checklist=manual_checklist,
        before_after=before_after,
        tradeoffs=tradeoffs,
        interview_feedback=_interview_feedback(analysis, score),
        validation=validation,
        candidate_source="none",
        candidate_explanation="",
        step_by_step_plan=[],
        tiered_candidates=[],
    )
    should_generate_candidates = (
        bool(candidate_provider)
        or bool(candidate)
        or bool((benchmark_input or "").strip())
        if generate_candidates is None
        else generate_candidates
    )
    if not should_generate_candidates:
        return plan

    seed_candidates = {"medium_refactor": candidate} if candidate else None
    verified_plan = generate_verified_optimization_candidates(
        original_code=analysis.raw_code,
        analysis=analysis,
        score=score,
        entrypoint=entrypoint,
        benchmark_input=benchmark_input,
        api_key=api_key,
        plan=plan,
        candidate_provider=candidate_provider,
        seed_candidates=seed_candidates,
    )
    if prior_rejection_reasons:
        verified_plan.validation.rejection_reasons = list(prior_rejection_reasons) + verified_plan.validation.rejection_reasons
    if verified_plan.validation.memory_tradeoff:
        verified_plan.tradeoffs.insert(
            0,
            f"This verified rewrite improves estimated time but increases estimated space from {verified_plan.validation.original_space} to {verified_plan.validation.candidate_space}.",
        )
    return verified_plan
