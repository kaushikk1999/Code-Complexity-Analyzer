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


def test_successful_function_benchmark():
    result = run_benchmark(
        "def total(values):\n    return sum(values)\n",
        entrypoint="total",
        input_text='{"args": [[1, 2, 3]]}',
        repeat_count=2,
    )
    assert result.success
    assert result.summary.repeat_count == 2


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
