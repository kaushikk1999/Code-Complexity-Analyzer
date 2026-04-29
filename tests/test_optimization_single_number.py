from analyzer import analyze_code
from benchmarking.metrics import BenchmarkResult, BenchmarkRun, summarize_runs
from optimization.planner import build_optimization_plan
from scoring import calculate_optimization_score


def test_single_number_generates_clean_reference_candidate(monkeypatch):
    def fake_result(avg_ms: float) -> BenchmarkResult:
        runs = [BenchmarkRun(run_index=1, runtime_ms=avg_ms, current_memory_kb=0.0, peak_memory_kb=1.0)]
        return BenchmarkResult(
            success=True,
            entrypoint="Solution.singleNumber",
            input_description="fake",
            runs=runs,
            summary=summarize_runs(runs),
        )

    def fake_benchmark_candidate(*args, **kwargs):
        return fake_result(1.0), fake_result(0.9)

    monkeypatch.setattr("optimization.planner.benchmark_candidate", fake_benchmark_candidate)

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
