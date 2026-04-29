from analyzer import analyze_code
from optimization.planner import build_optimization_plan
from scoring import calculate_optimization_score

WORD_BREAK_CODE = """
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


def test_wordbreak_reference_candidate_handles_empty_dictionary():
    analysis = analyze_code(WORD_BREAK_CODE)
    score = calculate_optimization_score(analysis)

    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint="Solution.wordBreak",
        benchmark_input='{"kwargs": {"s": "leetcode", "wordDict": ["leet", "code"]}}',
    )

    assert plan.validation.status == "accepted"
    assert plan.optimized_code
    assert "if not wordDict:" in plan.optimized_code
    assert 'return s == ""' in plan.optimized_code
