import json

from benchmarking import run_benchmark
from utils.entrypoints import discover_entrypoints
from utils.test_case_generator import (
    DEFAULT_BENCHMARK_CASE_COUNT,
    build_benchmark_batch_input,
    generate_test_cases,
    merge_generated_and_custom_benchmark_input,
)

WORD_BREAK_CODE = """
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        return True
"""

SORTED_ARRAY_TO_BST_CODE = """
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mid = len(nums) // 2
        return TreeNode(
            nums[mid],
            self.sortedArrayToBST(nums[:mid]),
            self.sortedArrayToBST(nums[mid + 1:]),
        )
"""

PASSWORD_CODE = """
def strongPasswordChecker(password: str) -> int:
    return len(password)
"""


def test_generates_word_break_cases():
    definitions = discover_entrypoints(WORD_BREAK_CODE)

    cases = generate_test_cases(
        code=WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        definitions=definitions,
    )

    assert len(cases) == DEFAULT_BENCHMARK_CASE_COUNT
    assert any("leetcode" in case.benchmark_input for case in cases)
    assert any("catsandog" in case.benchmark_input for case in cases)


def test_wordbreak_generated_cases_are_kwargs_not_two_sum_args():
    definitions = discover_entrypoints(WORD_BREAK_CODE)
    cases = generate_test_cases(
        code=WORD_BREAK_CODE,
        entrypoint="Solution.wordBreak",
        definitions=definitions,
    )

    assert len(cases) == DEFAULT_BENCHMARK_CASE_COUNT
    assert cases[0].benchmark_input.startswith('{"kwargs"')
    assert '"s": "leetcode"' in cases[0].benchmark_input
    assert '"wordDict": ["leet", "code"]' in cases[0].benchmark_input

    for case in cases:
        assert "[2, 7, 11, 15" not in case.benchmark_input


def test_sorted_array_to_bst_generated_batch_supplies_nums():
    definitions = discover_entrypoints(SORTED_ARRAY_TO_BST_CODE)
    cases = generate_test_cases(
        code=SORTED_ARRAY_TO_BST_CODE,
        entrypoint="Solution.sortedArrayToBST",
        definitions=definitions,
    )
    benchmark_input = build_benchmark_batch_input(cases)

    assert '"cases"' in benchmark_input
    assert '"nums"' in benchmark_input
    payload = json.loads(benchmark_input)
    assert any(case.get("kwargs", {}).get("nums") == [1, 2, 3] for case in payload["cases"])

    result = run_benchmark(
        SORTED_ARRAY_TO_BST_CODE,
        entrypoint="Solution.sortedArrayToBST",
        input_text=benchmark_input,
        repeat_count=1,
    )
    assert result.success, result.error
    assert result.input_description == "40 benchmark case(s)"


def test_string_annotation_generated_batch_supplies_strings():
    definitions = discover_entrypoints(PASSWORD_CODE)
    cases = generate_test_cases(
        code=PASSWORD_CODE,
        entrypoint="strongPasswordChecker",
        definitions=definitions,
    )
    benchmark_input = build_benchmark_batch_input(cases)
    payload = json.loads(benchmark_input)

    assert all(isinstance(case.get("kwargs", {}).get("password"), str) for case in payload["cases"])

    result = run_benchmark(
        PASSWORD_CODE,
        entrypoint="strongPasswordChecker",
        input_text=benchmark_input,
        repeat_count=1,
    )
    assert result.success, result.error


def test_custom_single_case_appends_to_mandatory_generated_cases():
    definitions = discover_entrypoints(SORTED_ARRAY_TO_BST_CODE)
    cases = generate_test_cases(
        code=SORTED_ARRAY_TO_BST_CODE,
        entrypoint="Solution.sortedArrayToBST",
        definitions=definitions,
    )

    benchmark_input = merge_generated_and_custom_benchmark_input(cases, '{"kwargs": {"nums": [9, 10]}}')
    payload = json.loads(benchmark_input)

    assert len(payload["cases"]) == DEFAULT_BENCHMARK_CASE_COUNT + 1
    assert payload["cases"][-1]["kwargs"]["nums"] == [9, 10]


def test_custom_batch_input_appends_to_mandatory_generated_cases():
    definitions = discover_entrypoints(SORTED_ARRAY_TO_BST_CODE)
    cases = generate_test_cases(
        code=SORTED_ARRAY_TO_BST_CODE,
        entrypoint="Solution.sortedArrayToBST",
        definitions=definitions,
    )
    custom = '{"cases": [{"kwargs": {"nums": [9]}}, {"kwargs": {"nums": [10]}}]}'

    benchmark_input = merge_generated_and_custom_benchmark_input(cases, custom)
    payload = json.loads(benchmark_input)

    assert len(payload["cases"]) == DEFAULT_BENCHMARK_CASE_COUNT + 2
    assert payload["cases"][-2]["kwargs"]["nums"] == [9]
    assert payload["cases"][-1]["kwargs"]["nums"] == [10]
