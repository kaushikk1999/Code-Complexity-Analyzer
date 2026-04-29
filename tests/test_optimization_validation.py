from analyzer import analyze_code
from optimization import (
    OptimizedCodeCandidate,
    build_optimization_plan,
    preserve_entrypoint_name,
    validate_optimized_candidate,
)
from scoring import calculate_optimization_score

TWO_SUM_BRUTE_FORCE = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""

BINARY_SEARCH_RECURSIVE = """
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif target < arr[mid]:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)
"""

BINARY_SEARCH_ITERATIVE = """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif target < arr[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1
"""


def test_local_two_sum_rewrite_is_verified_and_preserves_entrypoint():
    analysis = analyze_code(TWO_SUM_BRUTE_FORCE)
    score = calculate_optimization_score(analysis)

    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="two_sum",
        benchmark_input='{"args": [[2, 7, 11, 15], 9]}',
    )

    assert plan.validation.status == "accepted"
    assert plan.validation.source == "local"
    assert plan.validation.time_improved
    assert plan.validation.candidate_time == "O(n)"
    assert plan.validation.candidate_score > plan.validation.original_score
    assert "def two_sum(nums, target):" in plan.optimized_code


def test_recursive_binary_search_gets_verified_iterative_rewrite():
    analysis = analyze_code(BINARY_SEARCH_RECURSIVE)
    score = calculate_optimization_score(analysis)

    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="binary_search",
        benchmark_input='{"args": [[2, 4, 6, 8, 10, 12], 10]}',
    )

    assert plan.validation.status == "accepted"
    assert plan.validation.source == "local"
    assert plan.validation.space_improved
    assert plan.validation.candidate_time == "O(log n)"
    assert plan.validation.candidate_space == "O(1)"
    assert "while left <= right:" in plan.optimized_code
    assert "return binary_search(" not in plan.optimized_code


def test_optimal_iterative_binary_search_gets_reference_suggestion_and_plan_steps():
    analysis = analyze_code(BINARY_SEARCH_ITERATIVE)
    score = calculate_optimization_score(analysis)

    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="binary_search",
        benchmark_input='{"args": [[2, 4, 6, 8, 10, 12], 10]}',
    )

    assert plan.validation.status in {"rejected", "same_complexity"}
    if plan.validation.status == "rejected":
        assert any("improve" in reason for reason in plan.validation.rejection_reasons)
    verified = next(candidate for candidate in plan.verified_candidates if candidate.level == "quick_win")
    assert verified.actual_time == "O(log n)"
    assert verified.actual_space == "O(1)"
    if plan.validation.status == "rejected":
        assert plan.optimized_code is None
    assert plan.quick_wins
    assert plan.medium_refactors
    assert plan.advanced_improvements
    assert any("No medium refactor required" == step.title for step in plan.medium_refactors)
    assert any("No advanced rewrite verified" == step.title for step in plan.advanced_improvements)


def test_same_quality_candidate_is_rejected():
    analysis = analyze_code("def total(values):\n    return sum(values)\n")
    score = calculate_optimization_score(analysis)
    candidate = OptimizedCodeCandidate(
        source="gemini",
        code="def total(values):\n    return sum(values)\n",
    )

    optimized_code, validation = validate_optimized_candidate(analysis, score, candidate, entrypoint="total")

    assert optimized_code is None
    assert validation.status == "rejected"
    assert any("benchmark input" in reason or "identical" in reason for reason in validation.rejection_reasons)


def test_generated_candidate_is_renamed_to_configured_entrypoint():
    analysis = analyze_code(TWO_SUM_BRUTE_FORCE)
    score = calculate_optimization_score(analysis)
    candidate = OptimizedCodeCandidate(
        source="gemini",
        code="""
def solution(nums, target):
    seen = {}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []
""",
    )

    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="two_sum",
        candidate=candidate,
        benchmark_input='{"args": [[2, 7, 11, 15], 9]}',
    )

    assert plan.validation.status == "accepted"
    assert plan.validation.source == "gemini"
    assert "def two_sum(nums, target):" in plan.optimized_code
    assert "def solution" not in plan.optimized_code


def test_unsafe_candidate_is_rejected_before_showing_code():
    analysis = analyze_code("def total(values):\n    return sum(values)\n")
    score = calculate_optimization_score(analysis)
    candidate = OptimizedCodeCandidate(
        source="gemini",
        code="def total(values):\n    open('x.txt', 'w')\n    return sum(values)\n",
    )

    optimized_code, validation = validate_optimized_candidate(analysis, score, candidate, entrypoint="total")

    assert optimized_code is None
    assert validation.status == "rejected"
    assert any("safety" in reason for reason in validation.rejection_reasons)


def test_preserve_entrypoint_name_updates_recursive_calls():
    code = """
def solution(n):
    if n <= 1:
        return n
    return solution(n - 1)
"""

    renamed = preserve_entrypoint_name(code, "fib")

    assert "def fib(n):" in renamed
    assert "return fib(n - 1)" in renamed
