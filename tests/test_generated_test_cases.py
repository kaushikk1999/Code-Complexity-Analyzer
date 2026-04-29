from utils.entrypoints import discover_entrypoints
from utils.test_case_generator import generate_test_cases


WORD_BREAK_CODE = """
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        return True
"""


def test_generates_word_break_cases():
    definitions = discover_entrypoints(WORD_BREAK_CODE)

    cases = generate_test_cases(
        code=WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        definitions=definitions,
    )

    assert len(cases) == 5
    assert any("leetcode" in case.benchmark_input for case in cases)
    assert any("catsandog" in case.benchmark_input for case in cases)
