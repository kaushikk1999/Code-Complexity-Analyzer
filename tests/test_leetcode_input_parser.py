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


def test_word_break_json_args_benchmark_input():
    result = run_benchmark(
        WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        input_text='{"args": ["leetcode", ["leet", "code"]]}',
        repeat_count=1,
    )

    assert result.success


def test_word_break_assignment_benchmark_input():
    result = run_benchmark(
        WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        input_text='s = "leetcode"\nwordDict = ["leet", "code"]',
        repeat_count=1,
    )

    assert result.success


def test_word_break_leetcode_style_benchmark_input():
    result = run_benchmark(
        WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        input_text='s =\n"leetcode"\nwordDict =\n["leet", "code"]',
        repeat_count=1,
    )

    assert result.success
