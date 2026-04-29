from analyzer import analyze_code
from analyzer.complexity_rules import complexity_rank
from benchmarking.metrics import BenchmarkResult, BenchmarkRun, summarize_runs
from optimization.planner import (
    OptimizedCodeCandidate,
    candidate_is_better,
    generate_verified_optimization_candidates,
)
from scoring import calculate_optimization_score

WORD_BREAK_II_CODE = '''from typing import List
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        word_set = set(wordDict)
        memo = {}
        def dfs(start):
            if start in memo:
                return memo[start]
            if start == len(s):
                return [""]
            res = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    suffixes = dfs(end)
                    for suffix in suffixes:
                        if suffix == "":
                            res.append(word)
                        else:
                            res.append(word + " " + suffix)
            memo[start] = res
            return res
        return dfs(0)
def wordBreak(s, wordDict):
    return Solution().wordBreak(s, wordDict)
'''

BENCHMARK_INPUT = '{"kwargs": {"s": "leetcode", "wordDict": ["leet", "code"]}}'


def _benchmark(avg_ms: float, peak_kb: float) -> BenchmarkResult:
    runs = [
        BenchmarkRun(run_index=1, runtime_ms=avg_ms, current_memory_kb=0.0, peak_memory_kb=peak_kb),
        BenchmarkRun(run_index=2, runtime_ms=avg_ms, current_memory_kb=0.0, peak_memory_kb=peak_kb),
    ]
    return BenchmarkResult(
        success=True,
        entrypoint="solve",
        input_description="fake",
        runs=runs,
        summary=summarize_runs(runs),
    )


def test_rejects_candidate_with_lower_score_and_same_complexity():
    accepted, reason = candidate_is_better(
        original_score=73,
        candidate_score=70,
        original_avg_ms=1.0,
        candidate_avg_ms=1.0,
        original_peak_kb=1.0,
        candidate_peak_kb=1.0,
        original_time_rank=complexity_rank("O(n^2)"),
        candidate_time_rank=complexity_rank("O(n^2)"),
        original_space_rank=complexity_rank("O(n)"),
        candidate_space_rank=complexity_rank("O(n)"),
    )

    assert not accepted
    assert "Rejected" in reason


def test_word_break_ii_generates_three_distinct_candidates():
    analysis = analyze_code(WORD_BREAK_II_CODE)
    score = calculate_optimization_score(analysis)

    plan = generate_verified_optimization_candidates(
        original_code=WORD_BREAK_II_CODE,
        analysis=analysis,
        score=score,
        entrypoint="wordBreak",
        benchmark_input=BENCHMARK_INPUT,
    )

    assert len(plan.verified_candidates) == 3
    assert len({candidate.code for candidate in plan.verified_candidates if candidate.code}) > 1
    assert all(candidate.level in {"quick_win", "medium_refactor", "advanced"} for candidate in plan.verified_candidates)


def test_best_candidate_selected_by_benchmark_not_static_score_only(monkeypatch):
    code = """
def solve(values):
    total = 0
    for value in values:
        total += value
    return total
"""
    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis)

    def fake_local_candidate(original_code, original_analysis, entrypoint, level):
        return OptimizedCodeCandidate(
            source="local",
            code=f"def solve(values):\n    # {level}\n    return sum(values)\n",
            level=level,
            title=level,
            explanation="Fake candidate",
            expected_time="O(n)",
            expected_space="O(1)",
        )

    def fake_benchmark(original_code, candidate_code, entrypoint, benchmark_input, repeat_count=5, timeout_seconds=5.0):
        runtimes = {"quick_win": 5.0, "medium_refactor": 2.0, "advanced": 7.0}
        matched = next(level for level in runtimes if level in candidate_code)
        return _benchmark(10.0, 2.0), _benchmark(runtimes[matched], 2.0)

    monkeypatch.setattr("optimization.planner.build_local_candidate_for_level", fake_local_candidate)
    monkeypatch.setattr("optimization.planner.benchmark_candidate", fake_benchmark)

    plan = generate_verified_optimization_candidates(
        original_code=code,
        analysis=analysis,
        score=score,
        entrypoint="solve",
        benchmark_input='{"args": [[1, 2, 3]]}',
    )

    assert plan.best_candidate
    assert plan.best_candidate.level == "medium_refactor"
    assert plan.optimized_code == plan.best_candidate.code


def test_no_candidate_claims_impossible_lower_output_complexity():
    analysis = analyze_code(WORD_BREAK_II_CODE)
    score = calculate_optimization_score(analysis)
    plan = generate_verified_optimization_candidates(
        original_code=WORD_BREAK_II_CODE,
        analysis=analysis,
        score=score,
        entrypoint="wordBreak",
        benchmark_input=BENCHMARK_INPUT,
    )

    assert analysis.estimated_time == "O(k*n)"
    assert analysis.estimated_space == "O(k*n)"
    assert any("output-sensitive" in caveat.lower() for caveat in analysis.caveats)
    assert all(
        candidate.actual_time in {"", "O(k*n)"}
        for candidate in plan.verified_candidates
        if candidate.code
    )


def test_no_verified_candidate_keeps_original():
    code = "def solve(values):\n    return sum(values)\n"
    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis)
    seeds = {
        level: OptimizedCodeCandidate(
            source="local",
            code=code,
            level=level,
            title=level,
            explanation="Identical candidate",
            expected_time=analysis.estimated_time,
            expected_space=analysis.estimated_space,
        )
        for level in ("quick_win", "medium_refactor", "advanced")
    }

    plan = generate_verified_optimization_candidates(
        original_code=code,
        analysis=analysis,
        score=score,
        entrypoint="solve",
        benchmark_input='{"args": [[1, 2, 3]]}',
        seed_candidates=seeds,
    )

    assert plan.best_candidate is None
    assert plan.optimized_code is None
    assert all(candidate.status == "rejected" for candidate in plan.verified_candidates)
