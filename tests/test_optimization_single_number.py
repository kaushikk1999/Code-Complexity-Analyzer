from analyzer import analyze_code
from optimization.planner import build_optimization_plan
from scoring import calculate_optimization_score


def test_single_number_generates_clean_reference_candidate():
    code = """
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for i in nums:
            ans ^= i
        return ans
"""
    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis)
    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="Solution.singleNumber",
        benchmark_input="nums = [4, 1, 2, 1, 2]",
    )

    assert plan.validation.status == "accepted"
    assert plan.optimized_code
    assert "class Solution" in plan.optimized_code
    assert "def singleNumber" in plan.optimized_code
    assert "result ^= num" in plan.optimized_code
    assert plan.validation.candidate_time == "O(n)"
    assert plan.validation.candidate_space == "O(1)"
