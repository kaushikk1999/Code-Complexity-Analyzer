from analyzer import analyze_code
from optimization import planner
from optimization.planner import CandidateBenchmarkComparison, OptimizedCodeCandidate
from scoring import calculate_optimization_score

WORD_BREAK_ORIGINAL = """
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        words = set(wordDict)
        dp = [False] * (n + 1)
        dp[0] = True
        max_len = max(len(word) for word in wordDict)
        for i in range(1, n + 1):
            for j in range(max(0, i - max_len), i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
"""


WORD_BREAK_WORSE_CANDIDATE = """
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if not s:
            return True
        word_set = set(wordDict)
        unique_lengths = sorted(list(set(len(w) for w in word_set)))
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for length in unique_lengths:
                if length > i:
                    break
                prev_idx = i - length
                if dp[prev_idx]:
                    if s[prev_idx:i] in word_set:
                        dp[i] = True
                        break
        return dp[n]
"""


def test_rejects_lower_score_same_complexity_candidate():
    analysis = analyze_code(WORD_BREAK_ORIGINAL)
    score = calculate_optimization_score(analysis)
    candidate = OptimizedCodeCandidate(
        source="gemini",
        code=WORD_BREAK_WORSE_CANDIDATE,
        explanation="More verbose same-complexity rewrite.",
        confidence=0.90,
    )

    _, validation = planner.validate_optimized_candidate(
        original_analysis=analysis,
        original_score=score,
        candidate=candidate,
        entrypoint="Solution.wordBreak",
        benchmark_input='{"kwargs": {"s": "leetcode", "wordDict": ["leet", "code"]}}',
    )

    assert validation.status == "rejected"
    assert validation.candidate_score <= validation.original_score
    assert any("score" in reason.lower() or "benchmark" in reason.lower() for reason in validation.rejection_reasons)


def test_candidate_benchmark_must_not_be_slower(monkeypatch):
    def fake_compare_candidate_benchmark(*args, **kwargs):
        return CandidateBenchmarkComparison(
            original_success=True,
            candidate_success=True,
            original_avg_ms=1.0,
            candidate_avg_ms=1.2,
            original_peak_memory_kb=1.0,
            candidate_peak_memory_kb=1.0,
            runtime_ratio=1.2,
            memory_ratio=1.0,
        )

    monkeypatch.setattr(planner, "compare_candidate_benchmark", fake_compare_candidate_benchmark)

    code = "def total(values):\n    return sum(values)\n"
    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis)
    candidate = OptimizedCodeCandidate(
        source="gemini",
        code=code,
        explanation="Same static quality but slower in benchmark.",
        confidence=0.90,
    )

    _, validation = planner.validate_optimized_candidate(
        original_analysis=analysis,
        original_score=score,
        candidate=candidate,
        entrypoint="total",
        benchmark_input='{"args": [[1, 2, 3]]}',
    )

    assert validation.status == "rejected"
    assert any("runtime is worse" in reason for reason in validation.rejection_reasons)
