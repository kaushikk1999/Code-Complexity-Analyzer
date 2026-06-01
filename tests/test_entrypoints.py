from pathlib import Path

from utils.constants import DEFAULT_INPUT
from utils.entrypoints import benchmark_input_hint, choose_entrypoint, discover_entrypoints
from utils.test_case_generator import generate_test_cases

BINARY_SEARCH_CODE = """
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
"""


def test_stale_default_entrypoint_is_replaced_for_single_detected_function():
    assert choose_entrypoint(BINARY_SEARCH_CODE, "two_sum") == "binary_search"


def test_multiple_functions_choose_preferred_default_then_keep_valid_selection():
    code = """
def helper(value):
    return value + 1

def solve(value):
    return helper(value)

def later(value):
    return value
"""
    assert choose_entrypoint(code, "missing") == "solve"
    assert choose_entrypoint(code, "later") == "later"


def test_class_methods_are_exposed_as_qualified_entrypoints():
    code = """
class Solution:
    def search(self, nums, target):
        return -1
"""
    definitions = discover_entrypoints(code)
    assert [definition.callable_name for definition in definitions] == ["Solution.search"]
    assert choose_entrypoint(code, "search") == "Solution.search"


def test_top_level_leetcode_method_receiver_is_not_benchmark_input():
    code = """
def removeElement(self, nums: List[int], val: int) -> int:
    return len([item for item in nums if item != val])
"""
    definitions = discover_entrypoints(code)
    assert [definition.callable_name for definition in definitions] == ["removeElement"]
    assert definitions[0].needs_standalone_receiver
    assert definitions[0].benchmark_args == ["nums", "val"]
    assert definitions[0].required_positional_count == 2
    assert "self" not in benchmark_input_hint(definitions[0])


def test_generated_cases_skip_top_level_leetcode_receiver():
    code = """
def removeElement(self, nums: List[int], val: int) -> int:
    return len([item for item in nums if item != val])
"""
    definitions = discover_entrypoints(code)
    cases = generate_test_cases(code, "removeElement", definitions)
    assert cases
    assert '"nums"' in cases[0].benchmark_input
    assert '"val"' in cases[0].benchmark_input
    assert '"self"' not in cases[0].benchmark_input


def test_app_uses_auto_detected_entrypoint_selectbox():
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert "_choose_entrypoint_from_definitions" in app_source
    assert "st.selectbox(" in app_source
    assert "st.text_input(" in app_source
    assert app_source.count('"Entrypoint function"') >= 2


def test_benchmark_input_starts_empty_and_examples_do_not_prefill():
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert DEFAULT_INPUT == ""
    assert 'st.session_state.benchmark_input = example["input"]' not in app_source
    assert 'st.session_state.benchmark_input = DEFAULT_INPUT' not in app_source
    assert "_autofill_benchmark_input_from_generated_cases" in app_source
    assert "_analyze_current(None, autofill_benchmark_input=True)" in app_source
    assert 'key="benchmark_input"' not in app_source
    assert "_benchmark_input_widget_key()" in app_source


def test_app_caches_code_derived_context_and_generated_cases():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert "@st.cache_data(show_spinner=False)" in app_source
    assert "def _cached_analyze_code" in app_source
    assert "def _cached_discover_entrypoints" in app_source
    assert "def _cached_local_test_cases" in app_source
    assert "generated_test_case_cache_key" in app_source
    assert "class AnalysisContext" in app_source


def test_static_only_optimization_path_skips_benchmark_execution():
    app_source = Path("app.py").read_text(encoding="utf-8")
    static_only_branch = app_source[
        app_source.index("def _build_verified_optimization_plan") : app_source.index("def _save_current_record")
    ]

    assert "if st.session_state.static_only_mode:" in static_only_branch
    assert "_analyze_current(None)" in static_only_branch
    assert "else:\n        _run_benchmark()" in static_only_branch
