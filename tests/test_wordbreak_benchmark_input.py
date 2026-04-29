from benchmarking import run_benchmark

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


def test_wordbreak_accepts_kwargs_benchmark_input():
    result = run_benchmark(
        WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        input_text='{"kwargs": {"s": "leetcode", "wordDict": ["leet", "code"]}}',
        repeat_count=1,
    )

    assert result.success


def test_wordbreak_rejects_stale_two_sum_input_helpfully():
    result = run_benchmark(
        WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        input_text='{"args": [[2, 7, 11, 15, 21, 30, 42, 55], 57]}',
        repeat_count=1,
    )

    assert not result.success
    assert "type mismatch" in result.error.lower() or "Expected argument names" in result.error
    assert "s" in result.error
    assert "wordDict" in result.error
