from benchmarking import run_benchmark, run_scaling_benchmark, should_run_auto_benchmark
from benchmarking.docker_backend import DockerBenchmarkConfig
from benchmarking.metrics import ScalingBenchmarkPoint
from benchmarking.runner import build_scaled_input, estimate_empirical_complexity
from benchmarking.sandbox import validate_code_for_execution

BINARY_SEARCH_WITH_EXAMPLES = """
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif target < arr[mid]:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

arr = [2, 4, 6, 8, 10, 12]
print(binary_search(arr, 10))
print(binary_search(arr, 5))
"""


def test_top_level_benchmark_disabled_by_default():
    result = run_benchmark("x = 1 + 1", entrypoint="", repeat_count=1)
    assert not result.success
    assert "Top-level script benchmarking is disabled" in result.error


def test_static_only_mode_skips_auto_benchmark():
    assert should_run_auto_benchmark(False)
    assert not should_run_auto_benchmark(True)


def test_blocked_import():
    violations = validate_code_for_execution("import os\n")
    assert violations
    assert "denied" in violations[0]


def test_input_builtin_is_allowed_with_controlled_stdin():
    violations = validate_code_for_execution("def solve():\n    return input()\n")
    assert not violations


def test_successful_function_benchmark():
    result = run_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='{"args": [[1, 2, 3]]}',
        repeat_count=2,
    )
    assert result.success
    assert result.summary.repeat_count == 2


def test_function_using_input_reads_stdin_field_each_repeat():
    result = run_benchmark(
        "def solve():\n    return int(input()) + int(input())\n",
        entrypoint="solve",
        input_text='{"stdin": "2\\n3\\n"}',
        repeat_count=2,
    )
    assert result.success
    assert result.input_description == "case 1: stdin: 2 line(s)"
    assert result.summary.repeat_count == 2


def test_function_using_input_accepts_raw_stdin_text():
    result = run_benchmark(
        "def solve():\n    return int(input())\n",
        entrypoint="solve",
        input_text="7",
        repeat_count=1,
    )
    assert result.success
    assert result.input_description == "stdin: 1 line(s)"


def test_input_without_enough_stdin_fails_fast():
    result = run_benchmark(
        "def solve():\n    return input()\n",
        entrypoint="solve",
        input_text="",
        repeat_count=1,
    )
    assert not result.success
    assert "No benchmark stdin line is available" in result.error


def test_multiple_stdin_cases_benchmark_each_case():
    result = run_benchmark(
        "def solve():\n    return int(input()) + int(input())\n",
        entrypoint="solve",
        input_text='{"cases": [{"name": "small", "stdin": "2\\n3\\n"}, {"name": "larger", "stdin": ["10", "20"]}]}',
        repeat_count=1,
    )
    assert result.success
    assert result.input_description == "2 benchmark case(s)"
    assert [run.case_description for run in result.runs] == ["case 1: small", "case 2: larger"]


def test_top_level_input_script_benchmarks_when_enabled():
    result = run_benchmark(
        "n = int(input())\nanswer = n + 1\n",
        entrypoint="",
        input_text="41",
        repeat_count=1,
        allow_top_level=True,
    )
    assert result.success


def test_named_entrypoint_skips_top_level_input_driver_code():
    code = """
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
        code,
        entrypoint="sortedArrayToBST",
        input_text='{"kwargs": {"nums": [-10, -3, 0, 5, 9]}}',
        repeat_count=1,
    )
    assert result.success, result.error


def test_multiple_json_cases_benchmark_each_case():
    result = run_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='[{"args": [[1, 2, 3]]}, {"args": [[10, 20]]}, {"kwargs": {"values": []}}]',
        repeat_count=2,
    )
    assert result.success
    assert result.input_description == "3 benchmark case(s)"
    assert result.summary.repeat_count == 6
    assert [run.case_index for run in result.runs] == [1, 1, 2, 2, 3, 3]
    assert [run.case_run_index for run in result.runs] == [1, 2, 1, 2, 1, 2]


def test_multiple_cases_object_form_benchmarks_each_case():
    result = run_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='{"cases": [{"name": "small", "args": [[1]]}, {"name": "empty", "kwargs": {"values": []}}]}',
        repeat_count=1,
    )
    assert result.success
    assert result.input_description == "2 benchmark case(s)"
    assert [run.case_description for run in result.runs] == ["case 1: small", "case 2: empty"]


def test_empty_batch_cases_are_rejected():
    result = run_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='{"cases": []}',
        repeat_count=1,
    )
    assert not result.success
    assert "at least one case" in result.error


def test_stale_entrypoint_reports_available_candidates():
    result = run_benchmark(
        BINARY_SEARCH_WITH_EXAMPLES,
        entrypoint="two_sum",
        input_text='{"args": [[2, 4, 6, 8, 10, 12], 10]}',
        repeat_count=1,
    )
    assert not result.success
    assert "Entrypoint `two_sum` was not found" in result.error
    assert "Available callable entrypoints: binary_search" in result.error


def test_recursive_binary_search_benchmarks_with_assignment_input_and_example_target():
    result = run_benchmark(
        BINARY_SEARCH_WITH_EXAMPLES,
        entrypoint="binary_search",
        input_text="arr = [2, 4, 6, 8, 10, 12]",
        repeat_count=2,
    )
    assert result.success
    assert result.input_description == "assignment variables + example call"
    assert result.summary.repeat_count == 2


def test_assignment_input_uses_explicit_target_before_example_call():
    result = run_benchmark(
        BINARY_SEARCH_WITH_EXAMPLES,
        entrypoint="binary_search",
        input_text="arr = [2, 4, 6, 8, 10, 12]\ntarget = 5",
        repeat_count=1,
    )
    assert result.success
    assert result.input_description == "assignment variable(s): arr, target"


def test_argument_count_error_includes_expected_input_shape():
    code = BINARY_SEARCH_WITH_EXAMPLES.split("arr = ", 1)[0]
    result = run_benchmark(
        code,
        entrypoint="binary_search",
        input_text="arr = [2, 4, 6, 8, 10, 12]",
        repeat_count=1,
    )
    assert not result.success
    assert "missing required argument(s): target" in result.error
    assert 'Expected input shape for `binary_search`: {"args": ["arr", "target"], "kwargs": {}}.' in result.error


def test_qualified_class_method_entrypoint_benchmarks_successfully():
    result = run_benchmark(
        "class Solution:\n    def search(self, values, target):\n        return values.index(target)\n",
        entrypoint="Solution.search",
        input_text='{"args": [[2, 4, 6], 4]}',
        repeat_count=1,
    )
    assert result.success


def test_solution_remove_element_class_method_benchmarks_multiple_cases():
    code = (
        "class Solution:\n"
        "    def removeElement(self, nums: List[int], val: int) -> int:\n"
        "        k = 0\n"
        "        for i in range(len(nums)):\n"
        "            if nums[i] != val:\n"
        "                nums[k] = nums[i]\n"
        "                k += 1\n"
        "        return k\n"
    )
    result = run_benchmark(
        code,
        entrypoint="Solution.removeElement",
        input_text=(
            '{"cases": ['
            '{"name": "mixed", "kwargs": {"nums": [3, 2, 2, 3], "val": 3}}, '
            '{"name": "none removed", "kwargs": {"nums": [1, 2, 3], "val": 9}}'
            "]}"
        ),
        repeat_count=1,
    )
    assert result.success
    assert result.entrypoint == "Solution.removeElement"
    assert result.input_description == "2 benchmark case(s)"


def test_top_level_leetcode_method_benchmarks_without_self_input():
    code = (
        "def removeElement(self, nums: List[int], val: int) -> int:\n"
        "    k = 0\n"
        "    for i in range(len(nums)):\n"
        "        if nums[i] != val:\n"
        "            nums[k] = nums[i]\n"
        "            k += 1\n"
        "    return k\n"
    )
    result = run_benchmark(
        code,
        entrypoint="removeElement",
        input_text='{"kwargs": {"nums": [3, 2, 2, 3], "val": 3}}',
        repeat_count=1,
    )
    assert result.success
    assert result.entrypoint == "removeElement"


def test_top_level_leetcode_method_error_hint_excludes_self():
    code = "def removeElement(self, nums: List[int], val: int) -> int:\n    return len(nums)\n"
    result = run_benchmark(
        code,
        entrypoint="removeElement",
        input_text="",
        repeat_count=1,
    )
    assert not result.success
    assert "missing required argument(s): nums, val" in result.error
    assert "self" not in result.error


def test_entrypoint_can_use_top_level_import():
    result = run_benchmark(
        "import math\n\n"
        "def root(value):\n"
        "    return math.sqrt(value)\n",
        entrypoint="root",
        input_text='{"args": [4]}',
        repeat_count=1,
    )
    assert result.success


def test_entrypoint_can_use_top_level_constant():
    result = run_benchmark(
        "FACTOR = 2\n\n"
        "def scale(value):\n"
        "    return value * FACTOR\n",
        entrypoint="scale",
        input_text='{"args": [4]}',
        repeat_count=1,
    )
    assert result.success


def test_entrypoint_can_call_top_level_helper():
    result = run_benchmark(
        "def helper(value):\n"
        "    return value + 1\n\n"
        "def main(value):\n"
        "    return helper(value)\n",
        entrypoint="main",
        input_text='{"args": [4]}',
        repeat_count=1,
    )
    assert result.success


def test_build_scaled_input_resizes_first_arg():
    payload = build_scaled_input('{"args": [[1, 2, 3], 7]}', 10, "list")
    assert '"args": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 7]' in payload


def test_scaling_benchmark_runs():
    result = run_scaling_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='{"args": [[1, 2, 3]]}',
        sizes=[5, 10, 20],
        repeat_count=1,
        timeout_seconds=4,
    )
    assert result.success
    assert len(result.points) == 3
    assert result.empirical_complexity in {"O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "Unknown"}


def test_empirical_complexity_fit():
    points = [
        ScalingBenchmarkPoint(input_size=10, success=True, avg_ms=10),
        ScalingBenchmarkPoint(input_size=20, success=True, avg_ms=20),
        ScalingBenchmarkPoint(input_size=40, success=True, avg_ms=40),
    ]
    label, scores = estimate_empirical_complexity(points)
    assert label in scores
    assert scores[label] >= 0.9


def test_docker_config_defaults():
    config = DockerBenchmarkConfig()
    assert config.memory == "256m"
    assert config.pids_limit == 64
