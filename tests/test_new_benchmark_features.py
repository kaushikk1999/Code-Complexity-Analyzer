from benchmarking import run_benchmark
from utils.constants import DEFAULT_INPUT

REMOVE_ELEMENT_CLASS_CODE = """
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k
"""


REMOVE_ELEMENT_FUNCTION_CODE = """
def removeElement(self, nums: List[int], val: int) -> int:
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k
"""


def test_benchmark_input_starts_blank():
    assert DEFAULT_INPUT == ""


def test_solution_remove_element_supports_multiple_cases():
    benchmark_input = """
{
  "cases": [
    {"name": "mixed", "kwargs": {"nums": [3, 2, 2, 3], "val": 3}},
    {"name": "none removed", "kwargs": {"nums": [1, 2, 3], "val": 9}},
    {"name": "all removed", "kwargs": {"nums": [7, 7, 7], "val": 7}}
  ]
}
"""
    result = run_benchmark(
        REMOVE_ELEMENT_CLASS_CODE,
        entrypoint="Solution.removeElement",
        input_text=benchmark_input,
        repeat_count=1,
    )

    assert result.success, result.error
    assert result.entrypoint == "Solution.removeElement"
    assert result.input_description == "3 benchmark case(s)"
    assert result.summary.repeat_count == 3
    assert [run.case_description for run in result.runs] == [
        "case 1: mixed",
        "case 2: none removed",
        "case 3: all removed",
    ]


def test_top_level_leetcode_self_parameter_is_not_required():
    result = run_benchmark(
        REMOVE_ELEMENT_FUNCTION_CODE,
        entrypoint="removeElement",
        input_text='{"kwargs": {"nums": [3, 2, 2, 3], "val": 3}}',
        repeat_count=1,
    )

    assert result.success, result.error
    assert result.entrypoint == "removeElement"


def test_code_using_input_can_read_raw_stdin_from_benchmark_box():
    code = """
def solve():
    n = int(input())
    nums = list(map(int, input().split()))
    return sum(nums[:n])
"""
    result = run_benchmark(
        code,
        entrypoint="solve",
        input_text="3\n1 2 3",
        repeat_count=2,
    )

    assert result.success, result.error
    assert result.input_description == "stdin: 2 line(s)"
    assert result.summary.repeat_count == 2


def test_code_using_input_can_read_multiple_stdin_cases():
    code = """
def solve():
    left = int(input())
    right = int(input())
    return left + right
"""
    result = run_benchmark(
        code,
        entrypoint="solve",
        input_text="""
{
  "cases": [
    {"name": "small", "stdin": "2\\n3\\n"},
    {"name": "larger", "stdin": ["10", "20"]}
  ]
}
""",
        repeat_count=1,
    )

    assert result.success, result.error
    assert result.input_description == "2 benchmark case(s)"
    assert [run.case_description for run in result.runs] == ["case 1: small", "case 2: larger"]

