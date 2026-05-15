"""Manual smoke checks for the new benchmark features.

Run:
    python3 manual_feature_check.py
"""

from benchmarking import run_benchmark
from utils.constants import DEFAULT_INPUT
from utils.entrypoints import discover_entrypoints
from utils.test_case_generator import build_benchmark_batch_input, generate_test_cases


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    message = f"[{status}] {name}"
    if detail:
        message += f" - {detail}"
    print(message)
    return passed


def main() -> int:
    all_passed = True

    all_passed &= check(
        "Benchmark input starts blank",
        DEFAULT_INPUT == "",
        f"DEFAULT_INPUT={DEFAULT_INPUT!r}",
    )

    remove_element_class_code = """
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k
"""
    remove_element_cases = """
{
  "cases": [
    {"name": "mixed", "kwargs": {"nums": [3, 2, 2, 3], "val": 3}},
    {"name": "none removed", "kwargs": {"nums": [1, 2, 3], "val": 9}},
    {"name": "all removed", "kwargs": {"nums": [7, 7, 7], "val": 7}}
  ]
}
"""
    result = run_benchmark(
        remove_element_class_code,
        entrypoint="Solution.removeElement",
        input_text=remove_element_cases,
        repeat_count=1,
    )
    all_passed &= check(
        "Solution.removeElement supports multiple cases",
        result.success and result.summary.repeat_count == 3,
        result.error or result.input_description,
    )

    remove_element_function_code = """
def removeElement(self, nums: List[int], val: int) -> int:
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k
"""
    result = run_benchmark(
        remove_element_function_code,
        entrypoint="removeElement",
        input_text='{"kwargs": {"nums": [3, 2, 2, 3], "val": 3}}',
        repeat_count=1,
    )
    all_passed &= check(
        "Top-level LeetCode self parameter is not required",
        result.success,
        result.error or result.input_description,
    )

    input_function_code = """
def solve():
    n = int(input())
    nums = list(map(int, input().split()))
    return sum(nums[:n])
"""
    result = run_benchmark(
        input_function_code,
        entrypoint="solve",
        input_text="3\n1 2 3",
        repeat_count=2,
    )
    all_passed &= check(
        "Function using input() accepts raw stdin text",
        result.success and result.summary.repeat_count == 2,
        result.error or result.input_description,
    )

    multi_stdin_cases = """
{
  "cases": [
    {"name": "small", "stdin": "2\\n3\\n"},
    {"name": "larger", "stdin": ["10", "20"]}
  ]
}
"""
    result = run_benchmark(
        "def solve():\n    return int(input()) + int(input())\n",
        entrypoint="solve",
        input_text=multi_stdin_cases,
        repeat_count=1,
    )
    all_passed &= check(
        "Function using input() accepts multiple stdin cases",
        result.success and result.summary.repeat_count == 2,
        result.error or result.input_description,
    )

    sorted_array_to_bst_code = """
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
    definitions = discover_entrypoints(sorted_array_to_bst_code)
    cases = generate_test_cases(
        code=sorted_array_to_bst_code,
        entrypoint="Solution.sortedArrayToBST",
        definitions=definitions,
    )
    generated_input = build_benchmark_batch_input(cases)
    result = run_benchmark(
        sorted_array_to_bst_code,
        entrypoint="Solution.sortedArrayToBST",
        input_text=generated_input,
        repeat_count=1,
    )
    all_passed &= check(
        "Generated benchmark input supplies nums for sortedArrayToBST",
        result.success and '"nums"' in generated_input,
        result.error or result.input_description,
    )

    top_level_driver_code = """
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid + 1:])
    return root

def printLevelOrder(root):
    if not root:
        print([])
        return
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    print(result)

nums = list(map(int, input("Enter sorted numbers: ").split()))
root = sortedArrayToBST(nums)
printLevelOrder(root)
"""
    result = run_benchmark(
        top_level_driver_code,
        entrypoint="sortedArrayToBST",
        input_text='{"kwargs": {"nums": [-10, -3, 0, 5, 9]}}',
        repeat_count=1,
    )
    all_passed &= check(
        "Named entrypoint skips top-level input driver code",
        result.success,
        result.error or result.input_description,
    )

    print()
    if all_passed:
        print("All manual feature checks passed.")
        return 0
    print("One or more manual feature checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
